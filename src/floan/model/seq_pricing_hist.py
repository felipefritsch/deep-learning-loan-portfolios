"""W3b (``ADR-002``) — Monte-Carlo H>1 pricing of the FF + engineered-history arm (``ff_hist``).

The engineered-history baseline (``history.py`` / M26a) is the **engineered** rung of the ladder
between the memoryless current-state models (empirical / logit / FF-current / ensemble, priced by
exact composition in :mod:`seq_price_compare`) and the **learned**-memory sequence models
(GRU / transformer, priced by :func:`seq_pricing.simulate_paths`). It is a plain feed-forward net
— same architecture as the FF-current ``nn`` baseline — but its input row carries four *causal*
summaries of the loan's state path (``history.HIST_COLS``: ``prev_state``, ``ever_delinquent``,
``months_since_last_delinq``, ``n_prior_delinq_episodes``).

Those summaries are **path-dependent**, so ff_hist cannot be composed (composition averages the
path away) and it does **not** fit :func:`seq_pricing.simulate_paths`' "encode once, vary only the
``state`` embedding" precompute — that optimization is invalid precisely because the engineered
columns change with the simulated path (exactly what :func:`seq_pricing._assert_state_only_path_
dependence` forbids for the sequence arm). So this module is a SIBLING of :mod:`seq_pricing`:
identical sampling / covariate-advance / absorption / aggregation, the ONE new piece being the
per-step **recomputation** of the four engineered features from the simulated path so far, matching
``history.py``'s strict ``period < t`` convention.

It is purely **additive** — the cashflow engine, ``compose``/``absorb_rows``, the
``smm_paths``/``economics`` aggregation, and the matched-population guards are reused unchanged
(via :mod:`seq_price_mc`); only the per-loan state-distribution *generator* differs. The two hard
gates mirror the sequence MC driver:

  * **H=1 identity** — the MC mean at H=1 reproduces ff_hist's direct one-step ``evaluate``
    prediction (the FF net scored on the **observed** pre-t0 history via ``history.join_history``)
    to Monte-Carlo tolerance. This proves the per-step feature recomputation is byte-right at the
    anchor (where the engineered features are exactly the observed-history summaries the net trained
    on).
  * **Aggregation identity** — Σ per-loan dollar value == pool value (~1e-6), the ADR-001 §M22
    linear-in-UPB check, fed by the ff_hist simulation.

Run (GPU pod; retrains the 10M ff_hist weights if absent, then prices the matched 150k subsample)::

    python -m floan.model.seq_pricing_hist --device cuda                       # 5 key anchors
    python -m floan.model.seq_pricing_hist --device cuda --anchors 2020        # one anchor
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import evaluate as EV
from floan.model import features as F
from floan.model import history as HIST
from floan.model import net as N
from floan.model import pool as PL
from floan.model import seq_predict as SQ           # anchor_pop / anchor_base / anchor_points
from floan.model import seq_price_mc as MX          # price_simulation / h1_identity / subsample_pop / MC_OUT
from floan.model import seq_pricing as MC           # _apply_post / _apply_post_batched / _sample / price_paths
from floan.model import seq_spike as SP             # ff_encode / train_arm / score_probs / _ff_batches / NN
from floan.model import sequence as S               # prediction_points
from floan.model import smm_paths as SPP            # _realized_steps
from floan.model import torch_common as tc
from floan.model import train as T                  # _git_commit (provenance)

VARIANT = "full"
ANCHORS = list(MX.ANCHORS)                          # [2015, 2019, 2020, 2023, 2025]
HORIZONS = tuple(config.HORIZONS)                   # (1, 3, 6, 12)
MC_OUT = MX.MC_OUT                                  # artifacts/results/m27b_mc — next to the gru/xf artifacts
MC_WEIGHTS_ROOT = MX.MC_WEIGHTS_ROOT               # OUTPUTS/m27b_mc — 10M weights live beside gru/xf
TRAIN_N = 10_000_000                                # match the seq caches' train_n (meta.json): the matched sample
FF_BATCH, SCORE_BATCH, MAX_EPOCHS = 4096, 16384, 40

STATES = F.STATES
SI = F.STATE_INDEX
N_CLASSES = F.N_CLASSES
ORIGIN_ROWS = MC.ORIGIN_ROWS                        # transient (alive) class indices
PREPAID = MC.PREPAID
DELINQ_ROWS = np.array([SI[s] for s in HIST.DELINQ_STATES], dtype=np.int64)   # the "delinquency episode" block
_NEVER = np.int64(-1_000_000_000)                   # last-delinq month sentinel (no prior delinquency)


def _t0_mi(t0: int) -> int:
    """Absolute month index of period ``t0`` (yyyymm), matching ``history.py``'s ``mi``."""
    return (t0 // 100) * 12 + (t0 % 100) - 1


# ===========================================================================
# FFHistPredictor — the FF+history net behind a single-row scoring seam, plus the layout
# bookkeeping the per-step engineered-feature recomputation needs.
# ===========================================================================
class FFHistPredictor:
    """Wrap a fitted ff_hist :class:`net.MortgageMLP` + its **augmented** ``(scaler, vocab)`` (fit
    over ``history.AUG_CONTINUOUS`` / ``AUG_CATEGORICAL``) behind a single-row scoring seam.

    The augmented feature row is ``[F.CONTINUOUS ‖ HIST_CONTINUOUS]`` (cont), ``[F.CATEGORICAL ‖
    prev_state]`` (cat), ``[F.BINARY ‖ ever_delinquent]`` (bin). The base ``F.*`` block evolves
    exactly as the sequence arm's does (``state`` path-dependent; everything else deterministic in
    *h* or frozen at t0), so it is encoded ONCE per (loan, h) by :meth:`encode_future` with a
    base-restricted Scaler/Vocab; the four engineered slots are overwritten per path per step by the
    simulator. ``scaler``/``vocab`` are the AUG objects; their ``F.*`` centre/scale and vocab maps
    are identical to a base fit (``join_history`` is a left-join that does not change base values)."""

    def __init__(self, model, scaler: F.Scaler, vocab: F.Vocab, device="cpu"):
        self.model = model
        self.scaler, self.vocab = scaler, vocab
        self.device = torch.device(device) if isinstance(device, str) else device
        self._predict = N.make_nn_predict()
        # Base-restricted Scaler/Vocab — encode advance_frame rows into the F.* slots of the AUG row.
        self.base_scaler = F.Scaler(
            cols=list(F.CONTINUOUS),
            center={c: scaler.center[c] for c in F.CONTINUOUS},
            scale={c: scaler.scale[c] for c in F.CONTINUOUS},
            log_cols=[c for c in F.CONTINUOUS if c in scaler.log_cols],
            robust_cols=[c for c in F.CONTINUOUS if c in scaler.robust_cols])
        self.base_vocab = F.Vocab(cols=list(F.CATEGORICAL),
                                  maps={c: vocab.maps[c] for c in F.CATEGORICAL})
        # AUG-row layout.
        self.n_cont, self.n_cat = len(scaler.cols), len(vocab.cols)
        self.bin_cols = list(F.BINARY) + list(HIST.HIST_BINARY)
        self.n_bin = len(self.bin_cols)
        self.state_col = vocab.cols.index("state")
        self.prev_state_col = vocab.cols.index("prev_state")
        self.msld_idx = scaler.cols.index("months_since_last_delinq")
        self.nepi_idx = scaler.cols.index("n_prior_delinq_episodes")
        self.ever_idx = self.bin_cols.index("ever_delinquent")
        self.c_msld, self.s_msld = scaler.center["months_since_last_delinq"], scaler.scale["months_since_last_delinq"]
        self.c_nepi, self.s_nepi = scaler.center["n_prior_delinq_episodes"], scaler.scale["n_prior_delinq_episodes"]
        # class index 0..6 -> the `state` / `prev_state` vocab indices (unknown -> UNK).
        self.svi = np.array([vocab.maps["state"].get(s, F.UNK) for s in STATES], dtype=np.int64)
        self.psvi = np.array([vocab.maps["prev_state"].get(s, F.UNK) for s in STATES], dtype=np.int64)

    # -- encode the deterministic base future rows once per (loan, h) -----------------------------
    def encode_future(self, base: pl.DataFrame, t0: int, horizon: int):
        """``[horizon, L, ·]`` AUG-width ``(cont, cat, bin)`` with ONLY the base ``F.*`` columns
        filled (engineered slots zero — overwritten per path). Row *h-1* is ``advance_frame(base,
        t0, h)`` encoded with the base-restricted Scaler/Vocab, so the F.* block is byte-identical to
        what ``encode_frame`` on the AUG pipeline would produce for those columns."""
        L = base.height
        fcont = np.zeros((horizon, L, self.n_cont), np.float32)
        fcat = np.zeros((horizon, L, self.n_cat), np.int64)
        fbin = np.zeros((horizon, L, self.n_bin), np.float32)
        for h in range(1, horizon + 1):
            raw = F.prepare_raw(PL.advance_frame(base, t0, h))
            bc = self.base_scaler.transform(raw)                 # [L, len(F.CONTINUOUS)]
            bk = self.base_vocab.transform(raw)                  # [L, len(F.CATEGORICAL)]
            bb = F.binary_matrix(raw, F.BINARY)                  # [L, len(F.BINARY)]
            fcont[h - 1, :, :bc.shape[1]] = bc
            fcat[h - 1, :, :bk.shape[1]] = bk
            fbin[h - 1, :, :bb.shape[1]] = bb
        return fcont, fcat, fbin

    # -- per-loan rolling-state seed from the OBSERVED pre-t0 history ------------------------------
    def hist_seed(self, base: pl.DataFrame, k: int, t0_mi: int) -> dict:
        """The four engineered features at the anchor (feature-month t0) via the SAME
        ``history.join_history`` the net trained/evaluated on — turned into the rolling-state seed
        for the MC recomputation. ``join_history`` is the leakage-safe ``period < t`` summary, so
        these seed the chain's "history before t0" exactly (the H=1 identity verifies it)."""
        joined = HIST.join_history(base, VARIANT, k)             # row-aligned to base (sorts back by __ord)
        prev_state = joined.get_column("prev_state").to_list()
        ever = joined.get_column("ever_delinquent").fill_null(0).cast(pl.Int64).to_numpy().astype(bool)
        msld = joined.get_column("months_since_last_delinq").to_list()       # int or None (never delinq)
        nepi = joined.get_column("n_prior_delinq_episodes").fill_null(0).cast(pl.Int64).to_numpy()
        prev_cls = np.array([SI.get(s, -1) if s is not None else -1 for s in prev_state], dtype=np.int64)
        prev_vidx = np.array([self.vocab.maps["prev_state"].get(s, F.UNK) if s is not None else F.UNK
                              for s in prev_state], dtype=np.int64)
        msld_f = np.array([np.nan if v is None else float(v) for v in msld], dtype=np.float64)
        # months_since is null exactly when never-delinquent (ever=0); fill those before the int cast
        # (the value is discarded by the ``ever`` mask) so NaN→int64 doesn't warn / yield platform junk.
        last_mi = np.where(ever, (t0_mi - np.nan_to_num(msld_f, nan=0.0)).astype(np.int64), _NEVER)
        prev_isd = np.isin(prev_cls, DELINQ_ROWS)                # was the immediately-preceding state delinquent
        return {"ever": ever, "last_mi": last_mi, "nepi": nepi.astype(np.int64),
                "prev_isd": prev_isd, "prev_vidx": prev_vidx}

    # -- assemble one step's AUG rows for n flat rows, then score ----------------------------------
    def assemble(self, base_cont, base_cat, base_bin, from_cls, prev_vidx, ever, msld_val, nepi_val):
        """Build ``(cont, cat, bin)`` for ``n`` flat rows from the per-row base block + the four
        engineered values. ``from_cls`` is the *from*-state class (the ``state`` embedding); the
        engineered values are computed by the caller per path per step (the only path-dependent
        piece). ``msld_val`` is the raw ``months_since_last_delinq`` (used only where ``ever``;
        null elsewhere → scaler centre, i.e. 0)."""
        cont = base_cont.copy()
        cat = base_cat.copy()
        binb = base_bin.copy()
        cat[:, self.state_col] = self.svi[from_cls]
        cat[:, self.prev_state_col] = prev_vidx
        cont[:, self.msld_idx] = np.where(ever, (msld_val - self.c_msld) / self.s_msld, 0.0)
        cont[:, self.nepi_idx] = (nepi_val - self.c_nepi) / self.s_nepi
        binb[:, self.ever_idx] = ever.astype(np.float32)
        return cont, cat, binb

    @torch.no_grad()
    def score(self, cont: np.ndarray, cat: np.ndarray, binb: np.ndarray,
              *, batch: int = SCORE_BATCH) -> np.ndarray:
        """``(cont, cat, bin)`` flat ``[n, ·]`` → softmax next-state probabilities ``[n, 7]``
        (float64 softmax then host, matching ``seq_spike.score_probs`` / the M27a evaluate numerics)."""
        self.model.eval()
        n = cont.shape[0]
        out = np.empty((n, N_CLASSES), np.float64)
        for s in range(0, n, batch):
            sl = slice(s, s + batch)
            b = {"cont": cont[sl], "cat": cat[sl], "bin": binb[sl]}
            out[sl] = torch.softmax(self._predict(self.model, b, self.device).double(), dim=1).cpu().numpy()
        return out

    def direct_anchor_probs(self, base: pl.DataFrame, k: int) -> np.ndarray:
        """ff_hist's direct one-step ``evaluate`` prediction on the anchor pop: the FF net scored on
        the anchor rows with the OBSERVED-history engineered features — ``ff_encode``'s exact feature
        path (``prepare_raw`` → ``join_history`` → Scaler/Vocab/``binary_matrix``), but encoding only
        the model inputs (no target/weight, which the anchor frame need not carry). The independent
        reference the MC H=1 mean must reproduce (the H=1 identity gate). Row order is ``base``'s."""
        fr = HIST.join_history(F.prepare_raw(base), VARIANT, k)
        return self.score(self.scaler.transform(fr), self.vocab.transform(fr),
                          F.binary_matrix(fr, self.bin_cols))


# ===========================================================================
# Monte-Carlo roll-forward — reference (oracle) and vectorized (production)
# ===========================================================================
def _roll_features(pred: FFHistPredictor, states, h, t0_mi, seed, roll):
    """Engineered features for step ``h`` (feature-month ``mi_tau = t0_mi + h-1``) over ``states``
    ``[..., horizon+1]`` (axis -1 = month, index j = state at month t0_mi+j) and the running ``roll``
    summary (``ever``/``last_mi``/``nepi`` over months ``< mi_tau``). Returns ``(prev_vidx, ever,
    msld_val, nepi_val)`` — all shaped like ``states[..., 0]``. ``prev_state`` = state at month
    ``mi_tau-1`` (the loan's observed pre-t0 state at h=1, else the simulated ``states[..., h-2]``)."""
    mi_tau = t0_mi + (h - 1)
    if h == 1:
        prev_vidx = np.broadcast_to(seed["prev_vidx"], states[..., 0].shape)
    else:
        prev_vidx = pred.psvi[states[..., h - 2]]
    ever = roll["ever"]
    msld_val = (mi_tau - roll["last_mi"]).astype(np.float64)     # valid only where ever (else unused)
    return prev_vidx, ever, msld_val, roll["nepi"].astype(np.float64)


def _roll_incorporate(states, h, t0_mi, roll):
    """Fold the state at month ``mi = t0_mi + h-1`` (``states[..., h-1]``, the from-state of step
    ``h``) into the running ``roll`` summary, so it covers months ``< t0_mi+h`` for step ``h+1``.
    Episode start = delinquent now AND not delinquent at the immediately-preceding record (the
    ``history.py`` ``ep_start`` rule); ``last_mi`` advances to ``mi`` on any delinquency."""
    mi = t0_mi + (h - 1)
    s = states[..., h - 1]
    isd = np.isin(s, DELINQ_ROWS)
    roll["nepi"] = roll["nepi"] + (isd & ~roll["prev_isd"]).astype(np.int64)
    roll["last_mi"] = np.where(isd, mi, roll["last_mi"])
    roll["ever"] = roll["ever"] | isd
    roll["prev_isd"] = isd


def _seed_roll(seed: dict, shape) -> dict:
    """Broadcast the per-loan pre-t0 seed to ``shape`` (``[P]`` reference / ``[Bc, P]`` vectorized)
    as the initial running summary (months ``< t0``)."""
    return {"ever": np.broadcast_to(seed["ever"], shape).copy(),
            "last_mi": np.broadcast_to(seed["last_mi"], shape).copy(),
            "nepi": np.broadcast_to(seed["nepi"], shape).copy(),
            "prev_isd": np.broadcast_to(seed["prev_isd"], shape).copy()}


def _finish(state_dist, cum_prepaid, alive_before, horizon, n_paths, t0):
    smm = PL.per_loan_smm(cum_prepaid.astype(np.float32), alive_before.astype(np.float32))
    return {"n_loans": state_dist.shape[0], "horizon": horizon, "n_paths": int(n_paths), "t0": t0,
            "state_dist": state_dist, "cum_prepaid": cum_prepaid, "alive_before": alive_before,
            "smm": smm, "h1_dist": state_dist[:, 0, :]}


def _simulate_paths_hist_reference(pred: FFHistPredictor, base: pl.DataFrame, k: int, t0: int, *,
                                   horizon: int = PL.HORIZON, n_paths: int = 256,
                                   calibrator: PL.Calibrator | None = None, zero_impossible: bool = True,
                                   seed: int = 0, seed_all: dict | None = None) -> dict:
    """**Reference (oracle)** ff_hist Monte-Carlo roll-forward — one loan at a time (vectorized over
    paths), the known-correct ground truth :func:`simulate_paths_hist` is differentially gated
    against. Per loan, draw ``n_paths`` trajectories; at each month *h* recompute the four engineered
    features from the path so far (``history.py`` strict ``<t`` convention via :func:`_roll_features`
    / :func:`_roll_incorporate`), score the FF net, apply the SAME post-processing the composition
    path applies (``calibrator`` → M20 zeroing, via :func:`seq_pricing._apply_post`), sample, and
    absorb terminal states. Seeded per loan (``default_rng([seed, li])``) — reproducible
    independent of loan ordering, identical streams to the vectorized path."""
    P, L, t0_mi = int(n_paths), base.height, _t0_mi(t0)
    fcont, fcat, fbin = pred.encode_future(base, t0, horizon)
    origin_class = MX._origin_classes(base)
    if seed_all is None:
        seed_all = pred.hist_seed(base, k, t0_mi)
    eye = np.eye(N_CLASSES)
    state_dist = np.zeros((L, horizon, N_CLASSES), np.float64)
    cum_prepaid = np.zeros((L, horizon), np.float64)
    alive_before = np.zeros((L, horizon), np.float64)

    for li in range(L):
        rng = np.random.default_rng([seed, li])
        s_li = {key: np.asarray(val)[li] for key, val in seed_all.items()}
        roll = _seed_roll(s_li, (P,))
        states = np.empty((P, horizon + 1), np.int64)
        states[:, 0] = origin_class[li]
        alive = np.ones(P, bool)
        cur = np.full(P, origin_class[li], np.int64)
        for h in range(1, horizon + 1):
            prev_vidx, ever, msld_val, nepi_val = _roll_features(pred, states, h, t0_mi, s_li, roll)
            cont, cat, binb = pred.assemble(
                np.broadcast_to(fcont[h - 1, li], (P, pred.n_cont)),
                np.broadcast_to(fcat[h - 1, li], (P, pred.n_cat)),
                np.broadcast_to(fbin[h - 1, li], (P, pred.n_bin)),
                states[:, h - 1], prev_vidx, ever, msld_val, nepi_val)
            probs = MC._apply_post(pred.score(cont, cat, binb), cur, calibrator, zero_impossible)
            nxt = np.where(alive, MC._sample(probs, rng), cur)
            states[:, h] = nxt
            _roll_incorporate(states, h, t0_mi, roll)
            alive &= np.isin(nxt, ORIGIN_ROWS)
            cur = nxt
        state_dist[li] = eye[states[:, 1:]].mean(axis=0)
        cum_prepaid[li] = (states[:, 1:] == PREPAID).mean(axis=0)
        alive_before[li] = np.isin(states[:, 0:horizon], ORIGIN_ROWS).mean(axis=0)
    return _finish(state_dist, cum_prepaid, alive_before, horizon, P, t0)


def _simulate_hist_chunk(pred, idx, fcont, fcat, fbin, origin_class, seed_all, t0_mi, *,
                         horizon, P, seed, calibrator, zero_impossible, eye, gpu_batch):
    """Roll one chunk of loans (global indices ``idx``) × ``P`` paths forward, batched — the
    reference's per-loan outputs computed together. Per month one FF forward over ``Bc·P`` flat rows
    (no trailing-T window; ff_hist scores a single advanced row), with the four engineered features
    recomputed per path. Per-loan RNG streams drawn in month order ⇒ identical draws to the
    reference."""
    Bc = len(idx)
    s_chunk = {key: np.asarray(val)[idx] for key, val in seed_all.items()}            # [Bc] per field
    roll = _seed_roll({key: val[:, None] for key, val in s_chunk.items()}, (Bc, P))   # [Bc, P]
    oc = origin_class[idx]
    states = np.empty((Bc, P, horizon + 1), np.int64)
    states[:, :, 0] = oc[:, None]
    alive = np.ones((Bc, P), bool)
    cur = np.repeat(oc[:, None], P, axis=1)
    rngs = [np.random.default_rng([seed, int(idx[j])]) for j in range(Bc)]
    seed_chunk_prev = s_chunk["prev_vidx"]                                            # [Bc] for h=1

    for h in range(1, horizon + 1):
        mi_tau = t0_mi + (h - 1)
        if h == 1:
            prev_vidx = np.broadcast_to(seed_chunk_prev[:, None], (Bc, P))
        else:
            prev_vidx = pred.psvi[states[:, :, h - 2]]
        ever = roll["ever"]
        msld_val = (mi_tau - roll["last_mi"]).astype(np.float64)
        nepi_val = roll["nepi"].astype(np.float64)
        # base rows for this h, tiled across paths -> flat [Bc*P, ·]
        bc = np.broadcast_to(fcont[h - 1, idx][:, None, :], (Bc, P, pred.n_cont)).reshape(Bc * P, pred.n_cont)
        bk = np.broadcast_to(fcat[h - 1, idx][:, None, :], (Bc, P, pred.n_cat)).reshape(Bc * P, pred.n_cat)
        bb = np.broadcast_to(fbin[h - 1, idx][:, None, :], (Bc, P, pred.n_bin)).reshape(Bc * P, pred.n_bin)
        cont, cat, binb = pred.assemble(bc, bk, bb, states[:, :, h - 1].reshape(-1),
                                        prev_vidx.reshape(-1), ever.reshape(-1),
                                        msld_val.reshape(-1), nepi_val.reshape(-1))
        probs = pred.score(cont, cat, binb, batch=gpu_batch).reshape(Bc, P, N_CLASSES)
        probs = MC._apply_post_batched(probs, cur, calibrator, zero_impossible)
        u = np.empty((Bc, P), np.float64)
        for j in range(Bc):
            u[j] = rngs[j].random(P)
        cdf = np.cumsum(probs, axis=2)
        cdf[:, :, -1] = 1.0
        nxt = np.where(alive, (u[:, :, None] < cdf).argmax(axis=2), cur)
        states[:, :, h] = nxt
        _roll_incorporate(states, h, t0_mi, roll)
        alive &= np.isin(nxt, ORIGIN_ROWS)
        cur = nxt

    sd = eye[states[:, :, 1:]].mean(axis=1)
    cp = (states[:, :, 1:] == PREPAID).mean(axis=1)
    ab = np.isin(states[:, :, 0:horizon], ORIGIN_ROWS).mean(axis=1)
    return sd, cp, ab


def simulate_paths_hist(pred: FFHistPredictor, base: pl.DataFrame, k: int, t0: int, *,
                        horizon: int = PL.HORIZON, n_paths: int = 256,
                        calibrator: PL.Calibrator | None = None, zero_impossible: bool = True,
                        seed: int = 0, loan_batch: int | None = None, seed_all: dict | None = None) -> dict:
    """Vectorized ff_hist Monte-Carlo roll-forward (production) — the throughput version of
    :func:`_simulate_paths_hist_reference` with an identical contract and (to Monte-Carlo tolerance)
    identical output. Batches ``loan_batch`` loans × ``n_paths`` paths into one FF forward per month;
    per-loan ``rng = default_rng([seed, loan_index])`` streams (drawn in month order), the engineered-
    feature recomputation, terminal absorption, and reductions all match the reference exactly. On
    CUDA OOM the loan batch is halved and retried (never reducing ``n_paths`` or the pop). NOTE: this
    path deliberately does NOT call ``seq_pricing._assert_state_only_path_dependence`` — the
    engineered ``history.HIST_COLS`` ARE path-dependent here by design (recomputed each step), which
    is exactly what that guard forbids for the sequence arm."""
    P, L = int(n_paths), base.height
    t0_mi = _t0_mi(t0)
    fcont, fcat, fbin = pred.encode_future(base, t0, horizon)
    origin_class = MX._origin_classes(base)
    if seed_all is None:
        seed_all = pred.hist_seed(base, k, t0_mi)
    eye = np.eye(N_CLASSES)
    state_dist = np.zeros((L, horizon, N_CLASSES), np.float64)
    cum_prepaid = np.zeros((L, horizon), np.float64)
    alive_before = np.zeros((L, horizon), np.float64)

    B = loan_batch or MC._loan_batch_for(P)
    li0, next_report, t_start = 0, max(1, L // 10), time.perf_counter()
    while li0 < L:
        Bc = min(B, L - li0)
        idx = np.arange(li0, li0 + Bc)
        try:
            sd, cp, ab = _simulate_hist_chunk(
                pred, idx, fcont, fcat, fbin, origin_class, seed_all, t0_mi, horizon=horizon, P=P,
                seed=seed, calibrator=calibrator, zero_impossible=zero_impossible, eye=eye,
                gpu_batch=min(MC.SIM_GPU_BATCH, Bc * P))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if B == 1:
                raise
            B = max(1, B // 2)
            continue
        state_dist[idx], cum_prepaid[idx], alive_before[idx] = sd, cp, ab
        li0 += Bc
        if li0 >= next_report and li0 < L:
            print(f"    [simulate_paths_hist] {li0:,}/{L:,} loans (P={P}, "
                  f"{li0/(time.perf_counter()-t_start):,.0f} loans/s)", flush=True)
            next_report += max(1, L // 10)
    return _finish(state_dist, cum_prepaid, alive_before, horizon, P, t0)


def _differential_check_hist(pred: FFHistPredictor, base: pl.DataFrame, k: int, t0: int, *,
                             horizon: int = PL.HORIZON, n_paths: int = 500, seed: int = 0,
                             calibrator: PL.Calibrator | None = None, seed_all: dict | None = None) -> dict:
    """Run the reference and the vectorized ff_hist simulator on the SAME inputs/seed and report
    per-array max abs diffs + ``bit_exact`` + the O(1/√N) band — the differential gate."""
    ref = _simulate_paths_hist_reference(pred, base, k, t0, horizon=horizon, n_paths=n_paths,
                                         seed=seed, calibrator=calibrator, seed_all=seed_all)
    vec = simulate_paths_hist(pred, base, k, t0, horizon=horizon, n_paths=n_paths,
                              seed=seed, calibrator=calibrator, seed_all=seed_all)
    arr = ("state_dist", "cum_prepaid", "alive_before", "smm")
    d = {a: float(np.abs(ref[a] - vec[a]).max()) for a in arr}
    d["bit_exact"] = all(d[a] == 0.0 for a in arr)
    d["mc_tol"] = 6.0 / np.sqrt(max(1, n_paths))
    d["n_paths"] = int(n_paths)
    return d


# ===========================================================================
# Train / load the 10M ff_hist net (weights + pipeline persisted beside the gru/xf MC weights)
# ===========================================================================
def _ff_hist_dir(weights_root: Path, k: int) -> Path:
    return Path(weights_root) / f"ff_hist_k{k}"


def ff_train_sample(k: int, n: int) -> pl.DataFrame:
    """The EXACT ``n`` train prediction-points the seq caches used (``prediction_points`` seed 0)
    semi-joined to the full pool — so ff_hist trains on the SAME loan-months as the GRU/transformer
    it is decomposed against (the matched controlled comparison). Mirrors ``scripts/m27a/ff_arms.
    train_sample``."""
    pts = S.prediction_points(VARIANT, k, "train", sample_n=n, seed=0)
    pool = SP._scan_pool(VARIANT, k, "train").collect()
    keys = pts.select(pl.col("loan").alias("Loan Identifier"), pl.col("t_ym").alias("period_ym"))
    df = pool.join(keys, on=["Loan Identifier", "period_ym"], how="semi")
    assert df.height == pts.height, f"train sample {df.height} != {pts.height} (key join broke)"
    return df


def train_or_load_ff_hist(k: int, device, *, seed: int = 0, weights_root: Path = MC_WEIGHTS_ROOT,
                          train_n: int = TRAIN_N, retrain: bool = False):
    """The window-``k`` ff_hist net at the 10M sample + its augmented ``(scaler, vocab)``. Loads a
    persisted checkpoint if present (the M27a ``ff_arms`` runs saved only test predictions, so the
    first call retrains and ``torch.save``s the weights + ``save_pipeline``s the scaler/vocab); the
    training protocol (FF arch from ``config.NN_SELECTED``, Adam/ReduceLROnPlateau/early-stop) is the
    SAME ``seq_spike.train_arm`` loop ``ff_arms`` uses, just at the matched 10M sample. Weights land
    in ``weights_root/ff_hist_k{k}/`` (beside the gru/xf MC weights, distinct so nothing clobbers)."""
    d = _ff_hist_dir(weights_root, k)
    nb = len(F.BINARY) + len(HIST.HIST_BINARY)
    if (d / "best_model.pt").exists() and (d / "scaler.json").exists() and not retrain:
        scaler, vocab = F.load_pipeline(d)
        model = N.build(scaler, vocab, depth=SP.NN["depth"], dropout=SP.NN["dropout"], n_binary=nb).to(device)
        model.load_state_dict(torch.load(d / "best_model.pt", map_location=device))
        model.eval()
        return model, scaler, vocab

    tc.set_seed(seed)
    SP.DEVICE = device                                            # seq_spike closures default to this
    t0 = time.perf_counter()
    tr_df = ff_train_sample(k, train_n)
    val_df = SP._scan_pool(VARIANT, k, "val").collect()
    enc_tr, scaler, vocab = SP.ff_encode(VARIANT, k, tr_df, "ff_hist")
    enc_val, _, _ = SP.ff_encode(VARIANT, k, val_df, "ff_hist", scaler, vocab)
    predict = N.make_nn_predict()
    model = N.build(scaler, vocab, depth=SP.NN["depth"], dropout=SP.NN["dropout"], n_binary=nb).to(device)
    print(f"  [k{k} ff_hist] train {enc_tr['y'].shape[0]:,} val {enc_val['y'].shape[0]:,} "
          f"(load {time.perf_counter()-t0:.0f}s) — training 10M FF+history", flush=True)

    def train_iter(ep, sd):
        return SP._ff_batches(enc_tr, FF_BATCH, True, sd, ep)

    def val_eval(m):
        return EV._nll(SP.score_probs(m, predict, SP._ff_batches(enc_val, SCORE_BATCH, False), device),
                       enc_val["y"])

    best_vnll, hist = SP.train_arm(model, predict, train_iter, val_eval,
                                   device=device, seed=seed, max_epochs=MAX_EPOCHS)
    F.save_pipeline(d, scaler, vocab)
    torch.save(model.state_dict(), d / "best_model.pt")
    (d / "metrics.json").write_text(json.dumps(
        {"k": k, "arm": "ff_hist", "train_n": int(enc_tr["y"].shape[0]), "val_nll": best_vnll,
         "epochs": len(hist), "depth": SP.NN["depth"], "dropout": SP.NN["dropout"],
         "weight_decay": SP.NN["weight_decay"], "wall_sec": round(time.perf_counter() - t0, 1)}, indent=2))
    print(f"  [k{k} ff_hist] trained: best val_nll {best_vnll:.6f} ({len(hist)} ep) in "
          f"{time.perf_counter()-t0:.0f}s -> {d}", flush=True)
    model.eval()
    return model, scaler, vocab


# ===========================================================================
# The MC pricing driver — one anchor (mirrors seq_price_mc.price_anchor_mc)
# ===========================================================================
def price_anchor_ff_hist(model, scaler, vocab, device, k: int, pop: pl.DataFrame, *,
                         horizon: int = PL.HORIZON, n_paths: int, calibrator: PL.Calibrator | None = None,
                         seed: int = 0, servicing_spread: float = PL.SERVICING_SPREAD) -> dict:
    """Monte-Carlo H>1 pricing of ff_hist at anchor ``k`` on ``pop`` (the matched subsample). Builds
    ``base = anchor_base``, wraps the net in :class:`FFHistPredictor`, rolls
    :func:`simulate_paths_hist`, then prices via the UNCHANGED :func:`seq_price_mc.price_simulation`
    (``smm_paths`` → ``economics`` at H∈{1,3,6,12}). Hard gates (raise on failure): the H=1 identity
    (MC mean H=1 == ff_hist direct one-step) and the §M22 aggregation identity."""
    t0 = config._dec(k - 1)
    base = SQ.anchor_base(k, pop)
    if base.height != pop.height:
        raise AssertionError(f"base/pop misalignment: {base.height} vs {pop.height}")
    pred = FFHistPredictor(model, scaler, vocab, device)
    sim = simulate_paths_hist(pred, base, k, t0, horizon=horizon, n_paths=int(n_paths),
                              calibrator=calibrator, seed=seed)

    # ---- Guard 1: H=1 identity (per-step recompute is byte-right at the anchor) ----
    direct = pred.direct_anchor_probs(base, k)
    h1 = MX.h1_identity(sim, direct, MX._origin_classes(base), calibrator=calibrator)
    if not h1["pass"]:
        raise AssertionError(
            f"H=1 identity FAILED k{k} ff_hist: mean|Δ|={h1['mean_abs']:.3e} (tol {h1['tol_mean']:.3e}) "
            f"max|Δ|={h1['max_abs']:.3e} (tol {h1['tol_max']:.3e}) — the MC H=1 engineered-feature "
            f"recomputation does not match ff_hist's direct one-step (observed-history) prediction")

    # ---- Pricing (unchanged smm_paths → economics) + Guard 2: aggregation identity ----
    realized = SPP._realized_steps(k, pop.get_column("Loan Identifier").to_numpy())
    priced = MX.price_simulation(sim, base, pop, k, realized, model_name="ff_hist",
                                 horizons=HORIZONS, servicing_spread=servicing_spread)
    if not priced["aggregation"]["pass"]:
        a = priced["aggregation"]
        raise AssertionError(
            f"aggregation identity FAILED k{k} ff_hist: |pool_value − Σ per-loan|={a['abs_diff']:.3e} "
            f"> {a['tol']:g}·max(1,|value|)")

    meta = {"k": k, "t0": t0, "arch": "ff_hist", "model_name": "ff_hist", "n_pop": int(pop.height),
            "horizon": int(horizon), "n_paths": int(n_paths), "seed": seed,
            "calibrated": calibrator is not None, "servicing_spread": servicing_spread,
            "pool_price_h12": priced["pool_price"], "pool_wal_h12": priced["pool_wal"],
            "E_cum_prepaid_h12": float(np.nanmean(sim["cum_prepaid"][:, -1])),
            "E_alive_before_h1": float(np.nanmean(sim["alive_before"][:, 0]))}
    return {"econ": priced["econ"], "smm_frame": priced["smm_frame"], "meta": meta,
            "guards": {"h1_identity": h1, "aggregation": priced["aggregation"]}}


def run_anchor(k: int, device, *, seed: int = 0, n_paths: int = 1000,
               calibrator: PL.Calibrator | None = None, horizon: int = PL.HORIZON,
               retrain: bool = False, subsample_n: int = MX.SUBSAMPLE_N) -> dict:
    """Train/load ff_hist at anchor ``k``, price it on the SAME seeded stratified subsample the
    sequence arms used (:func:`seq_price_mc.subsample_pop`, identical ``[seed, k]`` draw), and persist
    ``k{k}_ff_hist_econ.parquet`` + ``_smm_paths.parquet`` + ``_h.json`` under :data:`MC_OUT`.
    Resumable: an existing ``k{k}_ff_hist_h.json`` is loaded and the anchor skipped."""
    arm_json = MC_OUT / f"k{k}_ff_hist_h.json"
    if arm_json.exists() and not retrain:
        print(f"  [k{k} ff_hist] already priced — skip (resume)", flush=True)
        return json.loads(arm_json.read_text())
    full_pop = SQ.anchor_pop(k)
    pop = MX.subsample_pop(full_pop, k, target_n=subsample_n, seed=seed)
    sub_meta = {"target_n": int(subsample_n), "actual_n": int(pop.height),
                "full_pop_n": int(full_pop.height), "seed": int(seed),
                "stratify": "char_cell_ids(FICO×rate×LTV)"}
    print(f"  [k{k}] ff_hist subsample {pop.height:,}/{full_pop.height:,} loans "
          f"(target {subsample_n:,}, seed {seed}, char-cell stratified)", flush=True)
    t_start = time.perf_counter()
    model, scaler, vocab = train_or_load_ff_hist(k, device, seed=seed, retrain=retrain)
    res = price_anchor_ff_hist(model, scaler, vocab, device, k, pop, horizon=horizon,
                               n_paths=n_paths, calibrator=calibrator, seed=seed)
    MC_OUT.mkdir(parents=True, exist_ok=True)
    res["econ"].write_parquet(MC_OUT / f"k{k}_ff_hist_econ.parquet")
    res["smm_frame"].write_parquet(MC_OUT / f"k{k}_ff_hist_smm_paths.parquet")
    payload = {"meta": res["meta"], "guards": res["guards"], "subsample": sub_meta,
               "headline": MX._econ_headline(res["econ"], "ff_hist"),
               "econ_parquet": f"k{k}_ff_hist_econ.parquet",
               "smm_parquet": f"k{k}_ff_hist_smm_paths.parquet",
               "git_commit": T._git_commit(),
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "wall_sec": round(time.perf_counter() - t_start, 1)}
    arm_json.write_text(json.dumps(payload, indent=2, default=float))
    g = res["guards"]
    print(f"  [k{k} ff_hist] priced N={n_paths} pool_price(H12)={res['meta']['pool_price_h12']:.4f} "
          f"H=1 mean|Δ|={g['h1_identity']['mean_abs']:.2e} agg|Δ|={g['aggregation']['abs_diff']:.1e} "
          f"-> k{k}_ff_hist_h.json [{payload['wall_sec']:.0f}s]", flush=True)
    return payload


def run(anchors: list[int], device, *, seed: int = 0, n_paths: int = 1000,
        calibrator: PL.Calibrator | None = None, horizon: int = PL.HORIZON,
        retrain: bool = False, subsample_n: int = MX.SUBSAMPLE_N) -> dict:
    """Price ff_hist across ``anchors`` (one anchor's failure does not sink the rest)."""
    print(f"=== W3b ff_hist MC pricing · anchors={anchors} · device={device} · N={n_paths} · "
          f"subsample_n={subsample_n:,} ===", flush=True)
    out = {}
    for k in anchors:
        try:
            out[k] = run_anchor(k, device, seed=seed, n_paths=n_paths, calibrator=calibrator,
                                horizon=horizon, retrain=retrain, subsample_n=subsample_n)
        except Exception as e:
            print(f"!!! W3b ff_hist k{k} FAILED: {type(e).__name__}: {e}", flush=True)
            out[k] = {"k": k, "error": f"{type(e).__name__}: {e}"}
    return {"anchors": anchors, "results": out}


def main() -> None:
    ap = argparse.ArgumentParser(description="W3b — Monte-Carlo H>1 ff_hist (FF+history) pricing (ADR-002).")
    ap.add_argument("--anchors", type=int, nargs="*", default=None, help="default: 5 key anchors")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--n-paths", type=int, default=1000, help="paths (match the gru/xf N=1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample-n", type=int, default=MX.SUBSAMPLE_N)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    config.require_drive()
    device = (torch.device("cuda") if args.device == "cuda" else tc.resolve_device(args.device))
    anchors = args.anchors if args.anchors else ANCHORS
    run(anchors, device, seed=args.seed, n_paths=args.n_paths, retrain=args.retrain,
        subsample_n=args.subsample_n)


if __name__ == "__main__":
    main()
