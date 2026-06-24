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
import resource
import time
import tracemalloc
from dataclasses import dataclass

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


def main() -> None:
    ap = argparse.ArgumentParser(description="M26 sequence data-path smoke (no model/training).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    ap.add_argument("--n-loans", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--smoke", action="store_true", help="run the data-path smoke")
    args = ap.parse_args()
    smoke(args.variant, args.k, args.n_loans, args.seq_len)


if __name__ == "__main__":
    main()
