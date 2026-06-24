"""M26 (data-path step) — per-loan sequence tensor materialization (``04 §M26``).

De-risks the sequence-model spike's DATA path **before** any GRU is fit: materialize short,
fixed-length, left-padded per-loan sequences from the loan-keyed panel, with a mask and the
t→t+1 target, reusing M26a's leakage-guarded panel windowing (``history.py``). NO model, NO
training here — this module only proves the tensor path and its leakage guard.

For a prediction point t (a transient-origin loan-month with a non-null ``state_next``), the
input sequence is the loan's last ``T`` monthly rows with ``period_ym ≤ t`` (left-padded to
``T``); the target is ``state_next`` at t. Per-timestep features are the **raw** (unscaled —
this is a path proof, not modelling) state index + a few panel covariates; ``state_next`` is
NEVER an input feature.

Leakage guard for sequences (the M26a discipline, lifted to seq-to-one):
  * **Input ends at t, never future** — the sequence rows are exactly the trailing
    ``period_ym ≤ t`` block (the seq analogue of M26a's ``1 PRECEDING`` end-at-t frame), and
    the last real timestep is t itself.
  * **``state_next`` never enters the inputs** — features are drawn from {state, covariates}
    only; the same FLIP-``state_next`` test as M26a, on a sequence sample, leaves the input
    tensor + mask byte-identical (only the target, derived from ``state_next``, changes).
  * **Per-loan reset** — every sequence is built within one ``PARTITION BY loan`` group, so
    no rows bleed across loans.

Run (Mac, SSD mounted) — the smoke prints shapes, the leakage assertions, example decoded
sequences, and build wall-time + memory for sizing the full spike:
    .venv/bin/python -m floan.model.sequence --smoke --n-loans 1000
"""

from __future__ import annotations

import argparse
import datetime
import json
import resource
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import export  # reuse connect(): panel view + memory limit + temp spill
from floan.model import features as F

# Fixed sequence length and the per-timestep feature columns (panel names → tensor order).
# F[...,0] is the state index (decodable to a STATES name); the rest are raw covariates.
SEQ_LEN = 12
COV_COLS: list[str] = ["Loan Age", "Current Interest Rate", "Current Actual UPB",
                       "Original Loan-to-Value (LTV)"]
FEATURE_NAMES: list[str] = ["state_idx"] + COV_COLS
N_FEATURES = len(FEATURE_NAMES)
PAD = 0.0  # left-pad fill; the MASK (not the value) marks real vs pad


@dataclass
class SeqBatch:
    """A materialized sequence batch (numpy; ``to_torch`` wraps for the model later)."""
    X: np.ndarray          # [N, T, F] float32 per-timestep features
    mask: np.ndarray       # [N, T]    float32 (1=real, 0=left-pad)
    y: np.ndarray          # [N]       int64   target class (state_next at t)
    loans: list[str]       # [N]       loan id per sequence
    t_ym: np.ndarray       # [N]       int32   prediction-point period_ym
    state_str: list[list[str]]   # [N][T] decoded states ('·' for pad) — for eyeballing
    pym: np.ndarray        # [N, T]    int32 period_ym per timestep (0 for pad)

    def to_torch(self) -> dict:
        return {"X": torch.from_numpy(self.X), "mask": torch.from_numpy(self.mask),
                "y": torch.from_numpy(self.y)}


# ---------------------------------------------------------------------------
# Pull the trailing panel rows for a deterministic loan subsample
# ---------------------------------------------------------------------------
def _pull_rows(con, variant: str, k: int, n_loans: int) -> tuple[pl.DataFrame, int]:
    """All rows with ``period_ym ≤ t`` for ``n_loans`` sampled loans, where t = the loan's
    last test-range transient-origin month with a non-null ``state_next``. Returns the rows
    (sorted by loan, period) and the total candidate-loan count (for extrapolation)."""
    shard_lt = config.VARIANTS[variant]["train_shard_lt"]
    hi = config.test_bounds(k)[1]          # period_ym < 201512  (k's feature range)
    test_lo = config.test_bounds(k)[0]     # 201412             (label year k)
    origins = str(tuple(config.ORIGIN_STATES))
    covs = ", ".join(f'"{c}" AS {a}' for c, a in zip(
        COV_COLS, ["loan_age", "cur_rate", "cur_upb", "orig_ltv"]))
    sql = f"""
    WITH cand AS (
        SELECT "Loan Identifier" AS loan, period, period_ym, state, state_next, {covs}
        FROM panel WHERE shard < {shard_lt} AND period_ym < {hi}
    ),
    tpoint AS (
        SELECT loan, max(period_ym) AS t_ym FROM cand
        WHERE period_ym >= {test_lo} AND state IN {origins} AND state_next IS NOT NULL
        GROUP BY loan
    ),
    sampled AS (SELECT loan, t_ym FROM tpoint ORDER BY hash(loan) LIMIT {n_loans})
    SELECT c.loan, c.period_ym, c.state, c.state_next,
           c.loan_age, c.cur_rate, c.cur_upb, c.orig_ltv, s.t_ym
    FROM cand c JOIN sampled s USING (loan)
    WHERE c.period_ym <= s.t_ym
    ORDER BY c.loan, c.period
    """
    rows = con.execute(sql).pl()
    n_cand = con.execute(
        f"""SELECT count(*) FROM (SELECT "Loan Identifier" FROM panel
            WHERE shard < {shard_lt} AND period_ym >= {test_lo} AND period_ym < {hi}
            AND state IN {origins} AND state_next IS NOT NULL GROUP BY 1)""").fetchone()[0]
    return rows, int(n_cand)


# ---------------------------------------------------------------------------
# Rows → [N, T, F] tensor + mask + target  (left-padded, per-loan)
# ---------------------------------------------------------------------------
def _to_tensors(rows: pl.DataFrame, T: int) -> SeqBatch:
    """Assemble left-padded ``[N, T, F]`` sequences, one per loan. ``state_next`` feeds ONLY
    the target (last real timestep), never the feature block."""
    si = pl.col("state").replace_strict(list(F.STATE_INDEX), list(F.STATE_INDEX.values()),
                                        default=-1, return_dtype=pl.Int64)
    rows = rows.with_columns(si.alias("state_idx"))
    feat_cols = ["state_idx"] + ["loan_age", "cur_rate", "cur_upb", "orig_ltv"]

    # last T rows per loan, in period order (rows already sorted by loan, period).
    tails = rows.group_by("loan", maintain_order=True).tail(T)

    Xs, masks, ys, loans, tyms, sstrs, pyms = [], [], [], [], [], [], []
    for (loan,), g in tails.group_by(["loan"], maintain_order=True):
        g = g.sort("period_ym")
        m = g.height
        block = (g.select(feat_cols).cast(pl.Float32).fill_null(PAD).to_numpy())  # [m, F]
        states = g.get_column("state").to_list()
        gpym = g.get_column("period_ym").to_list()
        t_ym = int(g.get_column("t_ym")[0])

        # Leakage + ordering invariants, per sequence:
        assert gpym == sorted(gpym), f"{loan}: timesteps not period-ordered"
        assert gpym[-1] == t_ym, f"{loan}: last real timestep {gpym[-1]} != prediction point {t_ym}"
        assert all(p <= t_ym for p in gpym), f"{loan}: a timestep is in the future of t"

        x = np.full((T, N_FEATURES), PAD, np.float32)
        mask = np.zeros(T, np.float32)
        sstr = ["·"] * T
        prow = np.zeros(T, np.int32)
        x[T - m:] = block
        mask[T - m:] = 1.0
        for i in range(m):
            sstr[T - m + i] = states[i]
            prow[T - m + i] = gpym[i]
        Xs.append(x); masks.append(mask); sstrs.append(sstr); pyms.append(prow)
        loans.append(loan); tyms.append(t_ym)
        ys.append(F.STATE_INDEX[g.get_column("state_next")[-1]])   # target = state_next at t

    return SeqBatch(
        X=np.stack(Xs), mask=np.stack(masks), y=np.asarray(ys, np.int64),
        loans=loans, t_ym=np.asarray(tyms, np.int32), state_str=sstrs,
        pym=np.stack(pyms))


def build_sequences(variant: str, k: int, n_loans: int, T: int = SEQ_LEN
                    ) -> tuple[SeqBatch, pl.DataFrame, int]:
    """Build the sequence batch; also return the raw pulled rows (for the flip test) and the
    candidate-loan universe size."""
    con = export.connect()
    rows, n_cand = _pull_rows(con, variant, k, n_loans)
    con.close()
    return _to_tensors(rows, T), rows, n_cand


# ---------------------------------------------------------------------------
# Smoke — shapes, leakage assertions, example sequences, time + memory
# ---------------------------------------------------------------------------
def smoke(variant: str = "dev", k: int = config.TUNING_YEAR, n_loans: int = 1000,
          T: int = SEQ_LEN) -> None:
    config.require_drive()
    print(f"=== M26 sequence data-path smoke  variant={variant} k={k}  "
          f"n_loans={n_loans}  T={T}  F={N_FEATURES} ===")

    tracemalloc.start()
    t0 = time.perf_counter()
    batch, rows, n_cand = build_sequences(variant, k, n_loans, T)
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # bytes→MB (darwin)

    # (1) shapes ----------------------------------------------------------------
    tt = batch.to_torch()
    print(f"\n[1] tensors: X{tuple(tt['X'].shape)} {tt['X'].dtype}  "
          f"mask{tuple(tt['mask'].shape)}  y{tuple(tt['y'].shape)} {tt['y'].dtype}")
    print(f"    feature order F: {FEATURE_NAMES}")
    real = batch.mask.sum(1)
    print(f"    real-timestep lengths: min {int(real.min())}  median {int(np.median(real))}  "
          f"max {int(real.max())}  (left-padded to {T})")
    print(f"    target class dist: " + ", ".join(
        f"{F.STATES[c]}={int((batch.y == c).sum())}" for c in range(F.N_CLASSES)
        if (batch.y == c).sum()))

    # (2) leakage guard ---------------------------------------------------------
    print("\n[2] leakage guard")
    # (2a) input ends at t, never future — checked per-sequence in _to_tensors; restate here.
    last_real_ym = np.array([batch.pym[i][batch.mask[i] == 1].max() for i in range(len(batch.loans))])
    ends_at_t = bool((last_real_ym == batch.t_ym).all())
    no_future = bool((np.array([(batch.pym[i][batch.mask[i] == 1] <= batch.t_ym[i]).all()
                                for i in range(len(batch.loans))])).all())
    print(f"  [{'PASS' if ends_at_t and no_future else 'FAIL'}] every input window ends at t "
          f"and contains no row past t (1-PRECEDING-equivalent: last real timestep == t)")

    # (2b) FLIP state_next: rebuild inputs from the SAME rows with state_next overwritten.
    rows_flip = rows.with_columns(pl.lit("prepaid").alias("state_next"))
    flip = _to_tensors(rows_flip, T)
    x_same = bool(np.array_equal(batch.X, flip.X) and np.array_equal(batch.mask, flip.mask))
    y_changed = bool(not np.array_equal(batch.y, flip.y))
    print(f"  [{'PASS' if x_same else 'FAIL'}] flip-state_next: input tensor + mask "
          f"byte-identical (state_next never enters inputs)")
    print(f"  [{'PASS' if y_changed else 'FAIL'}] flip-state_next: target DID change "
          f"(positive control — target is derived from state_next)")

    # (2c) per-loan reset: each sequence's real rows share one loan (no bleed).
    reset_ok = len(set(batch.loans)) == len(batch.loans)
    print(f"  [{'PASS' if reset_ok else 'FAIL'}] per-loan reset: {len(batch.loans)} sequences, "
          f"{len(set(batch.loans))} distinct loans (one sequence per loan, no cross-loan rows)")

    assert ends_at_t and no_future and x_same and y_changed and reset_ok, "LEAKAGE GUARD FAILED"

    # (3) example decoded sequences — prefer varied trajectories + one padded, so the
    # ordering and per-loan reset are eyeballable on non-trivial paths (not all-current).
    print("\n[3] example sequences (decoded states over time; '·' = left-pad)")
    n = len(batch.loans)
    real_len = batch.mask.sum(1).astype(int)
    ndist = [len({batch.state_str[i][j] for j in range(T) if batch.mask[i][j]}) for i in range(n)]
    picks = sorted(range(n), key=lambda i: -ndist[i])[:2]          # most state variation
    padded = [i for i in range(n) if real_len[i] < T]              # a left-padded one
    if padded and padded[0] not in picks:
        picks.append(padded[0])
    for i in picks[:3]:
        cells = " ".join(f"{batch.state_str[i][j][:7]:>7}" for j in range(T))
        ymrow = " ".join((f"{batch.pym[i][j]:>7}" if batch.mask[i][j] else f"{'·':>7}")
                         for j in range(T))
        print(f"  loan {batch.loans[i]}  t={batch.t_ym[i]}  → target state_next="
              f"{F.STATES[batch.y[i]]}  ({int(real_len[i])} real steps)")
        print(f"    period_ym : {ymrow}")
        print(f"    state     : {cells}")
        print(f"    mask      : {' '.join(f'{int(v):>7}' for v in batch.mask[i])}")

    # (4) build time + memory + sizing -----------------------------------------
    xbytes = batch.X.nbytes + batch.mask.nbytes + batch.y.nbytes
    print(f"\n[4] build wall-time + memory ({n_loans}-loan subsample)")
    print(f"    wall: {wall:.1f}s   (DuckDB panel scan dominates; ~constant in n_loans)")
    print(f"    python peak alloc: {peak / 1e6:.1f} MB   process maxRSS: {maxrss:.0f} MB")
    print(f"    tensor footprint: {xbytes / 1e6:.2f} MB for {len(batch.loans)} seqs "
          f"({xbytes / max(1, len(batch.loans)):.0f} B/seq)")
    print(f"    candidate loans (valid t in [{config.test_bounds(k)[0]},"
          f"{config.test_bounds(k)[1]})): {n_cand:,}")
    full_mb = xbytes / max(1, len(batch.loans)) * n_cand / 1e6
    print(f"    → full one-per-loan spike ≈ {n_cand:,} seqs ≈ {full_mb:.0f} MB tensor "
          f"(RAM-trivial; multi-point-per-loan scales N linearly)")
    print("\nSMOKE PASS — sequence data path proven (no model, no training).")


# ===========================================================================
# M26 step-2a — MODEL-CONSUMABLE sequences (FF feature set per timestep)
# ===========================================================================
# The step-1 path above is the one-per-loan, raw-feature, left-pad DEMO. The model path
# below is the real thing: prediction points come from the pools (train_pool/eval_pool, the
# SAME split as M26a), each point's trailing-T history is pulled from the panel, and every
# timestep carries the FULL FF feature set (scaled continuous incl. macro + embedded
# categoricals incl. state + binaries, via export.attach_macro + features.Scaler/Vocab fit
# on TRAIN rows only) — so a GRU vs the FF net differ ONLY in architecture. Sequences are
# RIGHT-padded (real oldest→newest at [0..len-1]) + lengths, for pack_padded_sequence.

# Panel columns pulled for each timestep, in the pool's pre-macro schema (export.attach_macro
# then joins macro to reproduce the FF feature row exactly).
_PANEL_PULL_COLS: list[str] = (
    ["Loan Identifier", "shard", "period", "period_ym", "orig_ym",
     "state", "state_next", "censored"] + export.PANEL_FEATURE_COLS)

_MACRO: tuple | None = None


def _macro_tables():
    global _MACRO
    if _MACRO is None:
        _MACRO = export._load_macro()
    return _MACRO


@dataclass
class SeqArrays:
    """Right-padded model batch: ``cont``/``cat``/``bin`` ``[N, T, ·]`` (real steps at
    ``[0..len-1]``, oldest→newest; the prediction point is the last real step), ``lengths``
    ``[N]`` for pack, ``mask`` ``[N, T]``, ``y``/``w`` ``[N]``."""
    cont: np.ndarray
    cat: np.ndarray
    bin: np.ndarray
    lengths: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    w: np.ndarray


def prediction_points(variant: str, k: int, split: str, *, origin_cap: int | None = None,
                      sample_n: int | None = None, seed: int = 0) -> pl.DataFrame:
    """Prediction points for ``split`` from the pool (train_pool for train, eval_pool for
    val/test), masked by ``config.{train,val,test}_bounds(k)`` — the SAME split as M26a.
    ``origin_cap`` = up to that many points PER origin state (training stratification, so the
    ~87% current mass doesn't wash out re-delinquency); ``sample_n`` = a random subsample
    (val orientation, natural class mix). Carries the pool ``weight`` (1/p_keep on train, 1
    on eval), the target index, and ``t_mi`` (month index of t)."""
    pool = "train_pool" if split == "train" else "eval_pool"
    lo, hi = {"train": config.train_bounds, "val": config.val_bounds,
              "test": config.test_bounds}[split](k)
    df = (pl.scan_parquet(str(config.TRAINING_DIR / variant / pool / "part-*.parquet"))
            .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
            .filter(pl.col("state").is_in(list(config.ORIGIN_STATES)))
            .filter(pl.col("state_next").is_not_null())
            .select("Loan Identifier", "period_ym", "state", "state_next", "weight")
            .collect())
    rng = np.random.default_rng(seed)
    if origin_cap is not None:
        parts = []
        for s in config.ORIGIN_STATES:
            sub = df.filter(pl.col("state") == s)
            if sub.height > origin_cap:
                sub = sub[np.sort(rng.choice(sub.height, origin_cap, replace=False))]
            parts.append(sub)
        df = pl.concat(parts)
    elif sample_n is not None and df.height > sample_n:
        df = df[np.sort(rng.choice(df.height, sample_n, replace=False))]
    return (df.with_columns(
                (pl.col("period_ym") // 100 * 12 + pl.col("period_ym") % 100 - 1).alias("t_mi"),
                pl.col("state_next").replace_strict(
                    list(F.STATE_INDEX), list(F.STATE_INDEX.values()),
                    default=-1, return_dtype=pl.Int64).alias("target_idx"))
              .rename({"Loan Identifier": "loan", "period_ym": "t_ym"})
              .with_row_index("point_id"))


def _window_sql(select_cols: list[str], T: int, shard_lt: int) -> str:
    """Per prediction point, the trailing ≤T panel rows with month-index in
    ``[t_mi-(T-1), t_mi]`` — the seq analogue of M26a's end-at-t window. ``pos`` =
    recency rank (1=newest=t), ``cnt`` = real length. Selection uses only loan + period;
    ``state_next`` is carried but never drives the window."""
    sel = ", ".join(f'p."{c}"' for c in select_cols)
    return f"""
    WITH win AS (
        SELECT pt.point_id, {sel},
               row_number() OVER (PARTITION BY pt.point_id ORDER BY p.period DESC) AS pos,
               count(*)     OVER (PARTITION BY pt.point_id) AS cnt
        FROM points pt JOIN panel p
          ON p."Loan Identifier" = pt.loan AND p.shard < {shard_lt}
         AND (year(p.period) * 12 + month(p.period) - 1)
             BETWEEN pt.t_mi - {T - 1} AND pt.t_mi
    )
    SELECT * FROM win WHERE pos <= {T}
    """


def _scatter(pid, ti, ei, cnt, N, T, cont, cat, binb):
    """Right-pad scatter: place encoded row ``ei`` at sequence position ``ti = cnt-pos`` of
    point ``pid``. Pure (no SSD) — unit-tested in tests/test_sequence.py."""
    Xc = np.zeros((N, T, cont.shape[1]), np.float32)
    Xk = np.zeros((N, T, cat.shape[1]), np.int64)
    Xb = np.zeros((N, T, binb.shape[1]), np.float32)
    mask = np.zeros((N, T), np.float32)
    lengths = np.zeros(N, np.int64)
    Xc[pid, ti] = cont[ei]
    Xk[pid, ti] = cat[ei]
    Xb[pid, ti] = binb[ei]
    mask[pid, ti] = 1.0
    lengths[pid] = cnt
    return Xc, Xk, Xb, mask, lengths


def build_split(variant: str, k: int, split: str, points: pl.DataFrame, T: int = SEQ_LEN,
                scaler: F.Scaler | None = None, vocab: F.Vocab | None = None
                ) -> tuple[SeqArrays, F.Scaler, F.Vocab]:
    """Materialize right-padded ``[N, T, F]`` sequences for ``points``. Pulls each point's
    trailing window from the panel, reattaches macro (export path) so the per-timestep
    features are the FF feature row, encodes via Scaler/Vocab (fit on TRAIN rows only when
    not supplied), and scatters into the tensor. ``state_next`` is never an input feature."""
    shard_lt = config.VARIANTS[variant]["train_shard_lt"]
    con = export.connect()
    con.register("points", points.select("point_id", "loan", "t_ym", "t_mi").to_arrow())
    win = con.execute(_window_sql(_PANEL_PULL_COLS, T, shard_lt)).pl()
    con.close()

    # Encode the UNIQUE panel rows once (FF feature row via macro reattach), then gather.
    nat, st, mkt = _macro_tables()
    pu = (win.unique(subset=["Loan Identifier", "period_ym"]).select(_PANEL_PULL_COLS)
            .with_columns(pl.col("period_ym").alias("label_ym"), pl.lit(1.0).alias("weight")))
    enc_frame = F.prepare_raw(export.attach_macro(pu, nat, st, mkt))
    if scaler is None:
        scaler = F.Scaler.fit(enc_frame, cols=F.CONTINUOUS)
        vocab = F.Vocab.fit(enc_frame, cols=F.CATEGORICAL)
    cont, cat, binb = scaler.transform(enc_frame), vocab.transform(enc_frame), F.binary_matrix(enc_frame)

    enc_idx = enc_frame.select("Loan Identifier", "period_ym").with_row_index("enc_idx")
    gather = (win.select("point_id", "Loan Identifier", "period_ym", "pos", "cnt")
                 .with_columns((pl.col("cnt") - pl.col("pos")).alias("ti"))
                 .join(enc_idx, on=["Loan Identifier", "period_ym"], how="left"))
    Xc, Xk, Xb, mask, lengths = _scatter(
        gather["point_id"].to_numpy(), gather["ti"].to_numpy(), gather["enc_idx"].to_numpy(),
        gather["cnt"].to_numpy(), points.height, T, cont, cat, binb)
    return (SeqArrays(Xc, Xk, Xb, lengths, mask,
                      points["target_idx"].to_numpy().astype(np.int64),
                      points["weight"].to_numpy().astype(np.float32)), scaler, vocab)


def iter_batches(arr: SeqArrays, batch_size: int, shuffle: bool, seed: int = 0, epoch: int = 0):
    """Minibatch dicts (cont/cat/bin/lengths/y/w) — the torch_common predict contract."""
    n = arr.y.shape[0]
    order = (np.random.default_rng([seed, epoch]).permutation(n) if shuffle
             else np.arange(n))
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        yield {"cont": arr.cont[idx], "cat": arr.cat[idx], "bin": arr.bin[idx],
               "lengths": arr.lengths[idx], "y": arr.y[idx], "w": arr.w[idx]}


# ===========================================================================
# M27a — build → cache → reload  (materialize the k=2015 seq training set on the
# Mac, SHARD it to disk for upload to the GPU pod, reload round-trips).
# ===========================================================================
# A full SeqArrays split is large (full val/test ≈ 6 M sequences ≈ 16 GB dense — bigger
# than this Mac's RAM), so the cache is SHARDED: each chunk of prediction points is built
# (memory-bounded), encoded with the TRAIN-fitted scaler/vocab, and written as one
# compressed npz; the reload concatenates the shards. On-disk dtypes are LOSSLESS compact
# downcasts of the SeqArrays fields (cont f32, cat i32, bin/mask u8, lengths i16, y i8,
# w f32) restored to the native dtypes on load — `assert_arrays_equal` round-trips byte for
# byte. Each shard also carries `origin` (origin-state index, EV.OI order) + `t_ym` so the
# pod can score per-transition AUC without the SSD panel.

CACHE_FIELDS: tuple[str, ...] = ("cont", "cat", "bin", "lengths", "mask", "y", "w")
# Native dtype each field is restored to (matches SeqArrays construction in build_split).
_NATIVE: dict[str, type] = {"cont": np.float32, "cat": np.int64, "bin": np.float32,
                            "lengths": np.int64, "mask": np.float32, "y": np.int64,
                            "w": np.float32}
# Lossless on-disk dtype (the value ranges are guarded in `_compact`).
_COMPACT: dict[str, type] = {"cont": np.float32, "cat": np.int32, "bin": np.uint8,
                             "lengths": np.int16, "mask": np.uint8, "y": np.int8,
                             "w": np.float32}


def _compact(arr: SeqArrays, origin: np.ndarray, t_ym: np.ndarray) -> dict:
    """SeqArrays (+ origin / t_ym aux) → dict of LOSSLESS compact arrays for one npz shard.
    Asserts each downcast is exact (cat indices, binaries, lengths, classes all in range), so
    a precondition break fails loudly instead of silently corrupting the cache."""
    assert 0 <= int(arr.cat.min(initial=0)) and int(arr.cat.max(initial=0)) < 2 ** 31, "cat out of int32"
    assert np.isin(arr.bin, (0.0, 1.0)).all() and np.isin(arr.mask, (0.0, 1.0)).all(), "bin/mask not 0/1"
    assert int(arr.lengths.max(initial=0)) <= 32767, "lengths out of int16"
    assert -1 <= int(arr.y.min(initial=0)) and int(arr.y.max(initial=0)) <= 127, "y out of int8"
    out = {f: getattr(arr, f).astype(_COMPACT[f]) for f in CACHE_FIELDS}
    out["origin"] = np.asarray(origin, np.int8)
    out["t_ym"] = np.asarray(t_ym, np.int32)
    return out


def save_shard(npz_path: Path, arr: SeqArrays, origin: np.ndarray, t_ym: np.ndarray) -> int:
    """Write one split shard (compressed npz); return its size in bytes."""
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **_compact(arr, origin, t_ym))
    return npz_path.stat().st_size


def load_split(split_dir: Path) -> dict:
    """Concatenate a split's shards (in order) → a dict with the 7 SeqArrays fields in their
    NATIVE dtypes plus ``origin`` / ``t_ym``. ``to_seqarrays`` wraps the model batch."""
    parts = sorted(Path(split_dir).glob("part-*.npz"))
    if not parts:
        raise FileNotFoundError(f"no part-*.npz shards in {split_dir}")
    acc: dict[str, list] = {k: [] for k in (*CACHE_FIELDS, "origin", "t_ym")}
    for p in parts:
        with np.load(p) as z:
            for k in acc:
                acc[k].append(z[k])
    return {k: np.concatenate(v).astype(_NATIVE.get(k, acc[k][0].dtype))
            for k, v in acc.items()}


def to_seqarrays(d: dict) -> SeqArrays:
    """Reconstruct the model batch from a ``load_split`` dict (drops the origin/t_ym aux)."""
    return SeqArrays(**{f: d[f] for f in CACHE_FIELDS})


def assert_arrays_equal(a: SeqArrays, b: SeqArrays) -> None:
    """Byte-for-byte equality of every SeqArrays field (the round-trip contract)."""
    for f in CACHE_FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        assert x.dtype == y.dtype and np.array_equal(x, y), f"round-trip mismatch on {f}"


# ---------------------------------------------------------------------------
# Stratified train points (memory-bounded) + scaler/vocab fit
# ---------------------------------------------------------------------------
def _origin_idx(points: pl.DataFrame) -> np.ndarray:
    """Origin-state → index in ``config.ORIGIN_STATES`` (== evaluate.OI), as int8-able int."""
    return points.select(pl.col("state").replace_strict(
        list(config.ORIGIN_STATES), list(range(len(config.ORIGIN_STATES))),
        default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1)


def _fit_rows_sql(T: int, shard_lt: int) -> str:
    """DISTINCT trailing-window panel rows for the registered ``points`` — the exact rows
    ``build_split`` encodes (no ``pos``/``cnt``, deduped in SQL so the fit isn't re-weighted by
    window overlap). DuckDB spills the join/distinct; only the result lands in RAM."""
    cols = ", ".join(f'p."{c}"' for c in _PANEL_PULL_COLS)
    return f"""
    SELECT DISTINCT {cols}
    FROM points pt JOIN panel p
      ON p."Loan Identifier" = pt.loan AND p.shard < {shard_lt}
     AND (year(p.period) * 12 + month(p.period) - 1)
         BETWEEN pt.t_mi - {T - 1} AND pt.t_mi
    """


def fit_pipeline(variant: str, k: int, points: pl.DataFrame, T: int = SEQ_LEN
                 ) -> tuple[F.Scaler, F.Vocab]:
    """Fit Scaler + Vocab on ``points``' trailing-window panel rows — the SAME rows
    ``build_split`` would fit on (macro reattached, ``prepare_raw`` derived columns), but
    WITHOUT materializing any dense [N,T,F] tensor. Pass a representative subsample of the
    train points (continuous stats are stable; rare categorical levels absent here map to the
    leakage-safe UNK by design, exactly as the FF/spike protocol fits on its train sample)."""
    shard_lt = config.VARIANTS[variant]["train_shard_lt"]
    nat, st, mkt = _macro_tables()
    con = export.connect()
    con.register("points", points.select("point_id", "loan", "t_mi").to_arrow())
    pu = con.execute(_fit_rows_sql(T, shard_lt)).pl()
    con.close()
    enc = F.prepare_raw(export.attach_macro(
        pu.with_columns(pl.col("period_ym").alias("label_ym"), pl.lit(1.0).alias("weight")),
        nat, st, mkt))
    return F.Scaler.fit(enc, cols=F.CONTINUOUS), F.Vocab.fit(enc, cols=F.CATEGORICAL)


# ---------------------------------------------------------------------------
# Chunked, sharded materialization of one split
# ---------------------------------------------------------------------------
def build_split_to_shards(variant: str, k: int, split: str, points: pl.DataFrame,
                          out_dir: Path, scaler: F.Scaler, vocab: F.Vocab,
                          T: int = SEQ_LEN, chunk: int = 300_000) -> tuple[int, int]:
    """Build ``points`` in chunks and write one compressed npz shard per chunk (peak RAM ≈ one
    chunk's dense tensors, never the whole split). Reuses ``build_split`` with the supplied
    (scaler, vocab) — no refit. Returns (total bytes on disk, number of shards)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.npz"):              # idempotent: clear stale shards
        old.unlink()
    n, total_bytes, si = points.height, 0, 0
    for s in range(0, n, chunk):
        pts = points[s:s + chunk].drop("point_id").with_row_index("point_id")
        arr, _, _ = build_split(variant, k, split, pts, T, scaler, vocab)
        b = save_shard(out_dir / f"part-{si:04d}.npz", arr,
                       _origin_idx(pts), pts["t_ym"].to_numpy())
        total_bytes += b
        print(f"  [{split:>5}] shard {si:>2}  rows {s:>9,}..{min(s + chunk, n):<9,} "
              f"-> {arr.y.shape[0]:>8,} seq  {b / 1e6:>6.0f} MB", flush=True)
        si += 1
    return total_bytes, si


# ---------------------------------------------------------------------------
# Verify (small sample): faithful round-trip + leakage flip-test
# ---------------------------------------------------------------------------
def verify_sample(variant: str, k: int, points: pl.DataFrame, scaler: F.Scaler,
                  vocab: F.Vocab, T: int = SEQ_LEN, n: int = 256) -> None:
    """On a small slice of ``points``: (1) build → save shard → load → assert SeqArrays
    identical; (2) leakage flip-test — overwrite state_next (and its target), rebuild, assert
    the input tensors (cont/cat/bin/mask/lengths) are byte-identical and only y changed."""
    pts = points.head(n).drop("point_id").with_row_index("point_id")
    arr, _, _ = build_split(variant, k, "train", pts, T, scaler, vocab)

    tmp = Path(config.OUTPUTS) / "seq_cache" / "_verify_tmp"
    if tmp.exists():
        for f in tmp.glob("part-*.npz"):
            f.unlink()
    save_shard(tmp / "part-0000.npz", arr, _origin_idx(pts), pts["t_ym"].to_numpy())
    reloaded = to_seqarrays(load_split(tmp))
    assert_arrays_equal(arr, reloaded)
    print(f"  [verify] round-trip OK — {arr.y.shape[0]} seqs save→load byte-identical "
          f"(7/7 fields)", flush=True)

    flip_pts = pts.with_columns(
        pl.lit("prepaid").alias("state_next"),
        pl.lit(F.STATE_INDEX["prepaid"]).cast(pl.Int64).alias("target_idx"))
    flip, _, _ = build_split(variant, k, "train", flip_pts, T, scaler, vocab)
    inputs_same = all(np.array_equal(getattr(arr, f), getattr(flip, f))
                      for f in ("cont", "cat", "bin", "mask", "lengths"))
    y_changed = not np.array_equal(arr.y, flip.y)
    static_ok = "state_next" not in (F.CONTINUOUS + F.CATEGORICAL + F.BINARY)
    assert inputs_same and y_changed and static_ok, "LEAKAGE FLIP-TEST FAILED"
    print(f"  [verify] flip-test PASS — inputs byte-identical, target changed, "
          f"state_next ∉ feature blocks", flush=True)


# ---------------------------------------------------------------------------
# Orchestrator + report
# ---------------------------------------------------------------------------
FIT_POINTS = 500_000   # representative train subsample the scaler/vocab are fit on


def build_cache(variant: str = "full", k: int = config.TUNING_YEAR,
                train_n: int = 1_500_000, T: int = SEQ_LEN, chunk: int = 300_000,
                seed: int = 0, out_root: str | None = None, verify: bool = True) -> dict:
    """Build + cache the k sequence training set (a UNIFORM ``train_n`` sample of the train_pool
    + the export's HT weights — the M26-spike recipe, so val-NLL stays comparable to M26's; NO
    origin re-balancing, which would distort the marginal) plus the FULL frozen val/test slices,
    sharded to disk for upload to the GPU pod. Scaler/vocab fit on TRAIN only; no training."""
    config.require_drive()
    out = Path(out_root) if out_root else config.OUTPUTS / "seq_cache" / variant / f"k{k}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"=== M27a build→cache  variant={variant} k={k} T={T}  "
          f"train_n={train_n:,} (uniform+HT)  chunk={chunk:,}  out={out} ===", flush=True)

    # (1) TRAIN points — UNIFORM train_pool sample (+ HT weights), NOT origin-balanced --------
    tr_pts = prediction_points(variant, k, "train", sample_n=train_n, seed=seed)
    tr_origin = {s: int((tr_pts["state"] == s).sum()) for s in config.ORIGIN_STATES}
    cur_frac = tr_origin["current"] / max(1, tr_pts.height)
    print(f"[train] {tr_pts.height:,} UNIFORM points (+ HT weights)  by origin: "
          + ", ".join(f"{s}={c:,} ({100 * c / tr_pts.height:.0f}%)"
                      for s, c in tr_origin.items()), flush=True)
    # Sanity: the sample must keep the natural train_pool mix (current-dominant ~69%), NOT a
    # balanced 25/25/25/25 — a balanced mix means origin-rebalancing leaked back in (the M26
    # step-2b distortion that makes val-NLL incomparable to M26's 0.0906).
    assert cur_frac > 0.5, (f"train origin mix looks balanced ({tr_origin}); expected the "
                            f"natural current-dominant train_pool mix — uniform sampling broke")

    # (2) fit scaler/vocab on a representative TRAIN subsample (train rows only) ---------------
    t_fit = time.perf_counter()
    fit_pts = (tr_pts.sample(n=min(FIT_POINTS, tr_pts.height), seed=seed)
               if tr_pts.height > FIT_POINTS else tr_pts)
    scaler, vocab = fit_pipeline(variant, k, fit_pts, T)
    F.save_pipeline(out, scaler, vocab)
    print(f"[train] fit scaler({len(scaler.cols)} cont) vocab({len(vocab.cols)} cat, "
          f"sizes {vocab.vocab_sizes}) on {fit_pts.height:,} pts  "
          f"[{time.perf_counter() - t_fit:.0f}s] -> scaler.json/vocab.json", flush=True)
    if verify:                                          # fail fast before the long build
        verify_sample(variant, k, tr_pts, scaler, vocab, T)

    # (3) materialize shards: train, then the FULL frozen val/test ---------------------------
    sizes, secs, npoints, nshards = {}, {}, {}, {}
    npoints["train"], origins = tr_pts.height, {"train": tr_origin}
    t = time.perf_counter()
    sizes["train"], nshards["train"] = build_split_to_shards(
        variant, k, "train", tr_pts, out / "train", scaler, vocab, T, chunk)
    secs["train"] = time.perf_counter() - t
    del tr_pts, fit_pts

    for split in ("val", "test"):
        pts = prediction_points(variant, k, split)          # full frozen slice (no cap/sample)
        npoints[split] = pts.height
        origins[split] = {s: int((pts["state"] == s).sum()) for s in config.ORIGIN_STATES}
        print(f"[{split}] {pts.height:,} points  by origin: "
              + ", ".join(f"{s}={c:,}" for s, c in origins[split].items()), flush=True)
        t = time.perf_counter()
        sizes[split], nshards[split] = build_split_to_shards(
            variant, k, split, pts, out / split, scaler, vocab, T, chunk)
        secs[split] = time.perf_counter() - t
        del pts

    # (4) meta + report --------------------------------------------------------
    pipe_bytes = sum((out / f).stat().st_size for f in ("scaler.json", "vocab.json"))
    wall = time.perf_counter() - t0
    meta = {
        "task": "M27a build->cache", "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), "variant": variant, "k": k, "seq_len": T,
        "per_origin_cap": per_origin_cap, "chunk": chunk, "seed": seed,
        "feature_dims": {"n_cont": len(scaler.cols), "n_cat": len(vocab.cols),
                         "n_bin": len(F.BINARY), "vocab_sizes": list(vocab.vocab_sizes)},
        "points": npoints, "points_by_origin": origins, "shards": nshards,
        "bytes": {**sizes, "pipeline": pipe_bytes,
                  "total": sum(sizes.values()) + pipe_bytes},
        "wall_sec": {**secs, "total": wall},
        "fields_on_disk": {f: np.dtype(_COMPACT[f]).name for f in CACHE_FIELDS},
        "fields_native": {f: np.dtype(_NATIVE[f]).name for f in CACHE_FIELDS},
        "aux_fields": ["origin (config.ORIGIN_STATES index)", "t_ym"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    _report_cache(meta, out)
    return meta


def _report_cache(m: dict, out: Path) -> None:
    gb = lambda b: b / 1e9
    print("\n" + "=" * 76)
    print(f"M27a SEQUENCE CACHE — {m['variant']} k={m['k']}  T={m['seq_len']}  "
          f"({m['feature_dims']['n_cont']} cont, {m['feature_dims']['n_cat']} cat, "
          f"{m['feature_dims']['n_bin']} bin)")
    print("=" * 76)
    print(f"  cache root: {out}")
    print(f"\n  {'split':>6} | {'points':>12} | {'shards':>6} | {'on-disk':>9} | {'build':>8}")
    print("  " + "-" * 56)
    for sp in ("train", "val", "test"):
        print(f"  {sp:>6} | {m['points'][sp]:>12,} | {m['shards'][sp]:>6} | "
              f"{gb(m['bytes'][sp]):>6.2f} GB | {m['wall_sec'][sp] / 60:>6.1f}m")
    print("  " + "-" * 56)
    print(f"  {'TOTAL':>6} | {sum(m['points'].values()):>12,} | "
          f"{sum(m['shards'].values()):>6} | {gb(m['bytes']['total']):>6.2f} GB | "
          f"{m['wall_sec']['total'] / 60:>6.1f}m")
    print(f"\n  point counts by origin (config.ORIGIN_STATES order):")
    for sp in ("train", "val", "test"):
        print(f"    {sp:>5}: " + "  ".join(
            f"{s}={m['points_by_origin'][sp][s]:,}" for s in config.ORIGIN_STATES))
    print(f"\n  scaler/vocab: {m['bytes']['pipeline'] / 1e3:.1f} KB  (scaler.json + vocab.json)")
    print(f"  on-disk dtypes: " + ", ".join(f"{f}:{d}" for f, d in m["fields_on_disk"].items()))
    print(f"  wrote meta.json  ·  total wall {m['wall_sec']['total'] / 60:.1f} min")
    print("=" * 76)


def main() -> None:
    ap = argparse.ArgumentParser(description="M26/M27a sequence data path: smoke + build->cache.")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    ap.add_argument("--n-loans", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--smoke", action="store_true", help="run the data-path smoke")
    ap.add_argument("--build-cache", action="store_true",
                    help="M27a: build + shard the k sequence train/val/test cache to disk")
    ap.add_argument("--train-n", type=int, default=1_500_000,
                    help="UNIFORM train_pool sample size (+ HT weights); NOT origin-balanced")
    ap.add_argument("--chunk", type=int, default=300_000, help="points per cache shard")
    ap.add_argument("--out", default=None, help="cache root (default OUTPUTS/seq_cache/<variant>/k<k>)")
    ap.add_argument("--no-verify", action="store_true", help="skip the round-trip + flip-test")
    args = ap.parse_args()
    if args.build_cache:
        build_cache(args.variant, args.k, args.train_n, args.seq_len, args.chunk,
                    out_root=args.out, verify=not args.no_verify)
    else:
        smoke(args.variant, args.k, args.n_loans, args.seq_len)


if __name__ == "__main__":
    main()
