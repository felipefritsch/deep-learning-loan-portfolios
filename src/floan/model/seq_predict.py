"""M27b — h=1 pool pricing for the sequence model (GRU) on the ADR-001 Predictor seam.

The fourth :class:`pool.Predictor` (after empirical / torch / ensemble), and the one ADR-001 §87
flagged as needing **sequence context**: a :class:`SeqPredictor` whose :meth:`origin_scores`
returns the four transient-origin ONE-STEP distributions of the window-``k`` GRU, scored on each
loan's trailing-``T`` observed history ending at the anchor ``t0 = Dec(k−1)``. The origin override
is the sequence analogue of :class:`pool.TorchPredictor`'s single-row state override: hold the
trailing history fixed and set the **prediction-point (last real timestep) state** to each origin,
so the GRU's 7-vector for origin ``s`` is ``GRU(history, last_state=s)`` — exactly the row
:func:`pool.assemble_matrix` needs. For a ``current``-origin loan the ``current`` block is its
genuine prediction, so ``compose(onehot(current)·M₁) = M₁[current,:]`` equals the GRU's
``evaluate`` one-step prediction — the built-in H=1 identity check (:func:`identity_check`).

**H=1 ONLY.** The deterministic matrix composition is exact at one step; beyond that the GRU would
need its trailing window advanced into the (unobserved) future, so multi-month is invalid here
(future-work MC). Everything downstream of the seam — :func:`pool.assemble_matrix`,
``_zero_impossible``, :func:`pool.roll_forward_predictor`, the SMM aggregation in ``smm_paths`` /
``economics`` — is reused unchanged; only the injected predictor differs (the ADR-001 guarantee).

Anchor sequences are pre-built once per anchor (chunked through :func:`sequence.build_split` with the
window-``k`` cached scaler/vocab); :meth:`origin_scores` then gathers the precomputed per-origin
blocks by loan id, so the engine's per-chunk scoring is a cheap lookup.
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import data as D
from floan.model import evaluate as EV
from floan.model import features as F
from floan.model import pool as PL
from floan.model import seq_model as SM
from floan.model import seq_train as ST
from floan.model import seq_transformer as XF
from floan.model import sequence as S
from floan.model import smm_paths as SP
from floan.model import torch_common as tc

VARIANT = "full"
ORIGIN_STATES = list(config.ORIGIN_STATES)              # current, dpd_30, dpd_60, dpd_90plus
STATE_COL = F.CATEGORICAL.index("state")                # 0 — the state slot in the per-timestep cat
DPD60 = F.STATE_INDEX["dpd_60"]
PREPAID = F.STATE_INDEX["prepaid"]

# Frozen M27a training config (gpu_train_all.py) — reused verbatim for the retrain.
HIDDEN, BATCH, LR, WD = 48, 4096, 1e-3, 1e-5
MAX_EPOCHS, PATIENCE = 40, 5
CACHE_ROOT = config.OUTPUTS / "seq_cache" / "full"
WEIGHTS_ROOT = config.OUTPUTS / "m27b_gpu_runs"


# ===========================================================================
# Architecture seam (W3b) — map an ``arch`` tag to its builder / ``predict`` callback /
# weights filename, so train-or-load + scoring serve BOTH the GRU (M27b) and the transformer
# (W2) without forking either path. The transformer reuses the GRU's frozen TRAINING config
# verbatim (the BATCH/LR/WD/MAX_EPOCHS/PATIENCE above + the ReduceLROnPlateau schedule below —
# identical to scripts/m27a/gpu_train_transformer.py); ONLY the architecture differs, and its
# d_model 64 / 4 heads / 2 layers live in ``seq_transformer.build``'s frozen defaults. Weights
# names keep the arms distinct WITHIN a weights dir: GRU ``k{k}_s{seed}.pt``, transformer
# ``k{k}_xf_s{seed}.pt`` (the dir is ``train_or_load``'s ``weights_root`` — M27b's H=1 path keeps
# the default ``WEIGHTS_ROOT``=m27b_gpu_runs/; the W3b 10M MC driver routes to its own dir so the
# two samples never clobber each other).
ARCHS = ("gru", "xf")


def _build_model(arch: str, scaler, vocab):
    if arch == "gru":
        return SM.build(scaler, vocab, hidden=HIDDEN)
    if arch == "xf":
        return XF.build(scaler, vocab)                      # frozen d_model 64 / 4 heads / 2 layers
    raise ValueError(f"unknown arch {arch!r} (expected one of {ARCHS})")


def _seq_predict(arch: str):
    """The arch's ``predict(model, batch, device) -> [N,7]`` callback. SM's and XF's are
    byte-identical (both tensor-ise the batch and call ``model(...)``), so ``arch='gru'``
    reproduces the M27b scoring exactly."""
    return (XF if arch == "xf" else SM).make_seq_predict()


def _weights_name(arch: str, k: int, seed: int) -> str:
    return f"k{k}_{'xf_' if arch == 'xf' else ''}s{seed}.pt"


# ===========================================================================
# Train / load the window-k GRU (weights persisted so reruns + the identity check reproduce)
# ===========================================================================
def train_or_load(k: int, seed: int, device, *, arch: str = "gru", cache_root: Path = CACHE_ROOT,
                  weights_root: Path = WEIGHTS_ROOT, retrain: bool = False):
    """The window-``k`` seed-``seed`` model (``arch`` ``"gru"`` or ``"xf"``) + its (scaler, vocab).
    Loads a persisted ``state_dict`` if present (the M27a/W2 GPU runs saved only predictions, so the
    first call retrains with the frozen config and ``torch.save``s the best weights); the cached
    pipeline is the one the sequence cache was built with. ``arch`` selects the builder / predict
    callback / weights filename via the W3b architecture seam; the training protocol (optimiser,
    schedule, batch, early stopping) is identical across arms (so a GRU-vs-transformer price
    difference is attributable to architecture alone). ``arch="gru"`` is byte-identical to M27b."""
    weights_root = Path(weights_root)
    weights_root.mkdir(parents=True, exist_ok=True)
    wpath = weights_root / _weights_name(arch, k, seed)
    scaler, vocab = F.load_pipeline(cache_root / f"k{k}")
    model = _build_model(arch, scaler, vocab).to(device)

    if wpath.exists() and not retrain:
        model.load_state_dict(torch.load(wpath, map_location=device))
        model.eval()
        return model, scaler, vocab

    tc.set_seed(seed)
    predict = _seq_predict(arch)
    tr = S.to_seqarrays(S.load_split(cache_root / f"k{k}" / "train"))
    va = S.to_seqarrays(S.load_split(cache_root / f"k{k}" / "val"))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    best_nll, best_state, best_ep, since = float("inf"), None, -1, 0
    t0 = time.perf_counter()
    for ep in range(MAX_EPOCHS):
        model.train()
        for b in S.iter_batches(tr, BATCH, shuffle=True, seed=seed, epoch=ep):
            y = tc._as_long(b["y"], device); w = tc._as_f32(b["w"], device)
            opt.zero_grad(set_to_none=True)
            tc.weighted_ce(predict(model, b, device), y, w).backward()
            opt.step()
        val_nll = EV._nll(ST._val_probs(model, predict, va, device, BATCH), va.y)
        sched.step(val_nll)
        if val_nll < best_nll - 1e-5:
            best_nll, best_ep, since = val_nll, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
        print(f"  [k{k} {arch} s{seed}] ep {ep:>2} val_nll {val_nll:.6f} "
              f"lr {opt.param_groups[0]['lr']:.1e}{'  *best' if since == 0 else ''}", flush=True)
        if since >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    # The default weights_root (m27b_gpu_runs/) holds M27b's 1.5M H=1 weights; the W3b 10M MC weights
    # are a DIFFERENT sample and must not clobber them — the W3b MC driver routes to its own dir.
    torch.save(best_state, wpath)
    print(f"  [k{k} {arch} s{seed}] trained: best val_nll {best_nll:.6f} (ep {best_ep}, "
          f"{best_ep + 1} eff) in {time.perf_counter() - t0:.0f}s -> {wpath.name}", flush=True)
    return model, scaler, vocab


# ===========================================================================
# M24 pool population — reproduced independently (no FF checkpoints needed on the pod)
# ===========================================================================
def anchor_pop(k: int) -> pl.DataFrame:
    """The M24 pricing population at anchor ``t0 = Dec(k−1)``: ``current``-origin loans with
    non-null FICO/rate/LTV, sorted by loan id — byte-identical membership/order to
    ``smm_paths.per_loan_smm_table`` (current filter → ``_anchor_features`` → drop-null → sort),
    but built from the eval pool alone (the roll-forward over the FF models is not needed to
    *form the pool*, only to score it). Carries the cashflow weights (upb/wac/wam)."""
    return (SP._anchor_features(k)
            .drop_nulls(subset=["fico", "rate", "ltv"])
            .sort("Loan Identifier"))


def anchor_points(k: int, pop: pl.DataFrame) -> pl.DataFrame:
    """A :func:`sequence.build_split`-shaped ``points`` frame for the anchor pop: one prediction
    point per loan at ``t0`` (``state = current`` — the true anchor state; ``state_next`` /
    ``target_idx`` are dummies, never used for scoring). Row order follows ``pop``."""
    t0 = config._dec(k - 1)
    t_mi = t0 // 100 * 12 + t0 % 100 - 1
    return (pop.select(pl.col("Loan Identifier").alias("loan"))
               .with_columns(pl.lit(t0, dtype=pl.Int64).alias("t_ym"),
                             pl.lit(t_mi, dtype=pl.Int64).alias("t_mi"),
                             pl.lit("current").alias("state"),
                             pl.lit("current").alias("state_next"),
                             pl.lit(0, dtype=pl.Int64).alias("target_idx"),
                             pl.lit(1.0, dtype=pl.Float32).alias("weight"))
               .with_row_index("point_id"))


def anchor_base(k: int, pop: pl.DataFrame) -> pl.DataFrame:
    """The raw anchor frame (``prepare_raw``) for the pop loans, row-aligned to ``pop`` — the
    ``base`` :func:`pool.roll_forward_predictor` rolls. All pop loans are ``current`` at ``t0``."""
    t0 = config._dec(k - 1)
    pool_dir, bounds = D.window_spec(VARIANT, k, "test")
    raw = (D._masked_scan(pool_dir, bounds).filter(pl.col("period_ym") == t0).collect())
    raw = F.prepare_raw(raw)
    order = pop.select("Loan Identifier").with_row_index("_ord")
    return (raw.join(order, on="Loan Identifier", how="inner").sort("_ord").drop("_ord"))


# ===========================================================================
# The SeqPredictor (ADR-001 / ADR-002) — the ONE model-specific surface for the GRU
# ===========================================================================
def _origin_probs(models, predict, arr: S.SeqArrays, device, batch: int) -> np.ndarray:
    """Mean-of-softmax over ``models`` for one (state-overridden) sequence batch — the
    paper Fig-7 ensemble rule (a single-member list is just that member's probs). Float64
    softmax on host, identical numerics to ``seq_train._val_probs`` / the M27a evaluate path."""
    acc = None
    for m in models:
        p = ST._val_probs(m, predict, arr, device, batch)
        acc = p.astype(np.float64) if acc is None else acc + p
    return (acc / len(models)).astype(np.float32)


class SeqPredictor:
    """Window-``k`` GRU (or seed-ensemble) behind the :class:`pool.Predictor` seam, h=1.

    Pre-builds the four transient-origin one-step blocks for every pop loan ONCE (chunked through
    :func:`sequence.build_split` with the window-``k`` scaler/vocab; per chunk the last real
    timestep's ``state`` is overridden to each origin and the GRU scored). :meth:`origin_scores`
    then gathers the precomputed blocks for the frame's loans — so the engine's per-step call is a
    lookup, not a re-score. ``models`` is a list (1 = single GRU; N = mean-of-softmax ensemble)."""

    def __init__(self, models, scaler, vocab, device, k: int, points: pl.DataFrame,
                 *, T: int = S.SEQ_LEN, chunk: int = 300_000, batch: int = BATCH):
        self.models = list(models)
        self.scaler, self.vocab, self.device, self.k = scaler, vocab, device, k
        self._state_idx = {s: vocab.maps["state"][s] for s in ORIGIN_STATES}
        predict = SM.make_seq_predict()
        n = points.height
        blocks = [[] for _ in ORIGIN_STATES]            # per-origin list of [chunk, 7]
        loans = []
        for s in range(0, n, chunk):
            pts = points[s:s + chunk].drop("point_id").with_row_index("point_id")
            arr, _, _ = S.build_split(VARIANT, k, "test", pts, T, scaler, vocab)
            rows = np.arange(arr.lengths.shape[0])
            last = arr.lengths - 1                       # right-padded: prediction point = last real step
            for j, st in enumerate(ORIGIN_STATES):
                cat = arr.cat.copy()
                cat[rows, last, STATE_COL] = self._state_idx[st]
                blocks[j].append(_origin_probs(self.models, predict, replace(arr, cat=cat),
                                               device, batch))
            loans.append(pts.get_column("loan").to_numpy())
        self.scores = [np.concatenate(b) for b in blocks]   # 4 × [N, 7] float32
        loan_ids = np.concatenate(loans)
        self._map = pl.DataFrame({"Loan Identifier": loan_ids,
                                  "_row": np.arange(loan_ids.shape[0], dtype=np.int64)})

    def _rows_for(self, frame_h: pl.DataFrame) -> np.ndarray:
        idx = (frame_h.select("Loan Identifier")
                      .join(self._map, on="Loan Identifier", how="left")
                      .get_column("_row").to_numpy())
        if np.isnan(idx.astype(np.float64)).any():
            raise KeyError("SeqPredictor: a frame loan has no pre-built anchor sequence")
        return idx.astype(np.int64)

    def origin_scores(self, frame_h: pl.DataFrame) -> list[np.ndarray]:
        idx = self._rows_for(frame_h)
        return [self.scores[j][idx] for j in range(len(ORIGIN_STATES))]


# ===========================================================================
# H=1 identity check (built-in regression guard) — the parameterized M13 Accept #2 for the GRU
# ===========================================================================
def gru_eval_probs(models, k: int, scaler, vocab, points: pl.DataFrame, device,
                   *, arch: str = "gru", T: int = S.SEQ_LEN, batch: int = BATCH) -> np.ndarray:
    """The sequence model's ``evaluate`` one-step prediction on ``points``' anchor sequences — built
    fresh via :func:`sequence.build_split` (true anchor state) and scored through the unmodified
    ``seq_train._val_probs`` softmax path (the M27a metric path). The independent reference the
    composed (M27b) / Monte-Carlo (W3b) H=1 must reproduce. ``arch`` (``"gru"``/``"xf"``) selects the
    predict callback; the default keeps the M27b GRU call byte-identical (the callbacks are the same)."""
    pts = points.drop("point_id").with_row_index("point_id")
    arr, _, _ = S.build_split(VARIANT, k, "test", pts, T, scaler, vocab)
    return _origin_probs(models, _seq_predict(arch), arr, device, batch)


def identity_check(models, k: int, scaler, vocab, device, pop: pl.DataFrame,
                   *, n: int = 20_000, tol: float = 1e-6) -> dict:
    """Assert the SeqPredictor's one-step output, **composed at H=1 through the engine**, equals
    the GRU's ``evaluate`` one-step prediction on the same loans to ``tol``. Mirrors
    ``pool.verify_h1`` stage (c): ``roll_forward_predictor(horizon=1, zero_impossible=False)`` on
    the raw (un-zeroed) path must reproduce the per-row prediction. Raises on failure."""
    t0 = config._dec(k - 1)
    sample = pop.head(n)
    pts = anchor_points(k, sample)
    base = anchor_base(k, sample)
    # Independent reference: the evaluate one-step prediction (fresh build + the metric softmax path).
    direct = gru_eval_probs(models, k, scaler, vocab, pts, device)
    # Engine path: compose the SeqPredictor at H=1, raw (un-zeroed) — the seam regression guard.
    seq_pred = SeqPredictor(models, scaler, vocab, device, k, pts)
    composed = PL.roll_forward_predictor(seq_pred, base, t0, horizon=1, snapshots=(1,),
                                         zero_impossible=False, chunk=base.height + 1,
                                         capture_h1=True)["h1"]
    max_abs = float(np.abs(composed - direct).max()) if direct.shape == composed.shape else float("inf")
    ok = direct.shape == composed.shape and max_abs <= tol
    print(f"  [identity k{k}] composed H=1 vs GRU evaluate: n={sample.height:,} "
          f"max|Δ|={max_abs:.2e} (tol {tol:g}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise AssertionError(f"H=1 identity FAILED for k{k}: max|Δ|={max_abs:.2e} > {tol:g} "
                             f"(shapes direct={direct.shape} composed={composed.shape})")
    return {"k": k, "n": int(sample.height), "max_abs_diff": max_abs, "tol": tol, "pass": ok}


# ===========================================================================
# H=1 pricing — reuse the M24 SMM aggregation + economics, GRU SMM only at h=1
# ===========================================================================
def _h1_to_12(col0: np.ndarray) -> np.ndarray:
    """Embed an H=1 per-loan value as the first column of an ``[N, 12]`` SMM array; columns 2..12
    are NaN (H>1 is invalid for the GRU). ``economics.per_pool_econ(horizons=(1,))`` reads only
    column 1, and the SMM aggregation guards empty/NaN denominators — so the NaN tail is never
    priced (asserted in the driver)."""
    out = np.full((col0.shape[0], PL.HORIZON), np.nan, dtype=np.float64)
    out[:, 0] = col0
    return out


def price_anchor_h1(models, k: int, scaler, vocab, device, pop: pl.DataFrame,
                    *, chunk: int = PL.DEFAULT_CHUNK) -> dict:
    """Price the GRU at H=1 on the M24 pools. Builds the SeqPredictor over the pop, rolls it one
    step through :func:`pool.roll_forward_predictor` (production ``zero_impossible=True``,
    ``capture_smm``), assembles the ``smm_paths``-shaped ``tbl`` with ``model_names=["gru"]``, and
    runs the unchanged ``pool_smm_paths`` → ``paths_to_frame`` → ``economics.per_pool_econ`` at
    ``horizons=(1,)``. Returns the per-(scheme,pool) econ frame (model ``gru``) + the realized SMM
    frame + meta."""
    t0 = config._dec(k - 1)
    pts = anchor_points(k, pop)
    base = anchor_base(k, pop)
    assert base.height == pop.height, f"base/pop misalignment: {base.height} vs {pop.height}"

    seq_pred = SeqPredictor(models, scaler, vocab, device, k, pts, chunk=chunk)
    rf = PL.roll_forward_predictor(seq_pred, base, t0, horizon=1, snapshots=(1,),
                                   zero_impossible=True, capture_smm=True, chunk=chunk)
    cum = rf["smm"]["cum_prepaid"][:, 0]                 # composed prepaid mass at h=1
    alive = rf["smm"]["alive_before"][:, 0]              # transient mass at start of step 1 (=1)
    dpd60 = rf["snapshots"][1][:, DPD60]                 # first-passage 60+ at h=1 = normal p[dpd_60]

    pop_ids = pop.get_column("Loan Identifier").to_numpy()
    alive_real, prepay_real, dpd60_real, dpd90_real = SP._realized_steps(k, pop_ids)
    tbl = {"df": pop, "model_names": ["gru"],
           "smm": {"gru": {"cum": _h1_to_12(cum), "alive": _h1_to_12(alive),
                           "cum_dpd60p": _h1_to_12(dpd60)}},
           "alive_real": alive_real, "prepay_real": prepay_real,
           "dpd60_real": dpd60_real, "dpd90_real": dpd90_real,
           "upb": pop.get_column("upb").to_numpy(),
           "wac": pop.get_column("wac").to_numpy(),
           "wam": pop.get_column("wam").to_numpy(), "pop_ids": pop_ids}

    paths = SP.pool_smm_paths(tbl, k)
    smm_frame = SP.paths_to_frame(paths, k)
    from floan.model import economics as EC
    econ = EC.per_pool_econ(k, smm_frame, horizons=(1,))
    meta = {"k": k, "t0": t0, "n_pop": pop.height, "n_members": len(models),
            "E_cum_prepaid_h1": float(np.nanmean(cum)),
            "E_alive_h1": float(np.nanmean(alive)),
            "E_cum_dpd60p_h1": float(np.nanmean(dpd60))}
    return {"econ": econ, "smm_frame": smm_frame, "meta": meta}
