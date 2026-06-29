"""M28 / W3 (``ADR-002``) — Monte-Carlo path simulation for H>1 sequence-model pricing.

ADR-001 made the roll-forward model-agnostic and ``horizon`` a parameter, but it assumes a
**point-in-time** predictor: ``roll_forward_predictor`` advances one feature frame, scores it,
composes the 7×7 transition matrix, and repeats — exact under a first-order Markov model, "no
Monte-Carlo required". The sequence models (``seq_model.SeqGRU`` / ``seq_transformer.
SeqTransformer``) break that assumption by construction: each prediction consumes the **trailing-T
trajectory**, not a single row, so the month-(h+1) distribution depends on the *realised path*
through months 1..h — which matrix composition averages away. Pricing them at H>1 therefore needs
a different roll-forward: **simulate paths** instead of composing matrices (``ADR-002`` Decision).

This module is purely **additive** (``ADR-002`` Option A): it does NOT touch ``pool.compose`` /
``absorb_rows`` / ``assemble_matrix`` / ``cashflow_engine`` / ``evaluate.py``. It adds

  1. :class:`SeqPredictor` — wraps a fitted ``SeqGRU``/``SeqTransformer`` + ``(scaler, vocab)`` and
     scores a per-loan trailing-T window → a next-state 7-vector. It also implements the
     :class:`pool.Predictor` ``origin_scores`` contract so it drops into the **unchanged**
     ``roll_forward_predictor`` at H=1 (the regression-guard cross-check: one composed step equals
     the direct one-step sequence prediction, ``ADR-002`` §"At H=1 this is harmless").
  2. :func:`simulate_paths` — the Monte-Carlo roll-forward. Per loan, draw ``n_paths``
     trajectories; each month *h* build the trailing-T window from the path so far, score the
     next-state distribution (apply the ``calibrator`` if given, exactly as the composition path
     does *before* assembly), **sample** the next state, advance covariates with the SAME rules as
     :func:`pool.advance_frame` (age/term/seasonality evolve; UPB + the macro block frozen at t0),
     and absorb terminal states. Aggregate the paths → per-loan state distribution per *h* and the
     per-loan conditional SMM path.
  3. :func:`price_paths` — feed each per-loan SMM path through the **existing**
     :func:`pool.price_loans_vec` / :func:`pool.cashflow_engine` (a one-loan "pool", the ADR-001
     §M22 pattern) and aggregate. No new engine math.

The H=1 limit is the regression guard: with no covariate advancement and no sampling done yet, the
window is the observed trailing window scored once, so the MC mean over paths → the direct one-step
prediction (= ``roll_forward_predictor`` at ``horizon=1``) to Monte-Carlo tolerance ``O(1/√N)``.
This holds for ANY model, trained or not (it is just "the empirical law of a categorical sample
converges to the categorical"), so the guard is testable on a synthetic, untrained model on CPU
(``tests/test_seq_pricing.py``) — no SSD, no GPU.

Covariate evolution is path-dependent in exactly ONE field: ``state``. Everything ``advance_frame``
touches (age, remaining term, ``period_ym`` → ``month_of_year``) is a deterministic function of *h*
alone, and everything else (UPB, rates, the macro block, statics) is frozen at t0. So the future
rows q_1..q_horizon are encoded ONCE per loan (state-independent ``cont``/``bin`` and all ``cat``
columns except ``state``), and each sampled trajectory only overwrites the single ``state`` vocab
index per step — which keeps the per-loan MC tractable (one batched model forward per loan per *h*).
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import features as F
from floan.model import history as HIST
from floan.model import pool as PL
from floan.model.sequence import SEQ_LEN

STATES = F.STATES
SI = F.STATE_INDEX
N_CLASSES = F.N_CLASSES
ORIGIN_STATES = list(config.ORIGIN_STATES)
ORIGIN_ROWS = np.array([SI[s] for s in ORIGIN_STATES])          # transient (= alive) class indices
TERMINAL_ROWS = np.array([SI[s] for s in STATES if s not in ORIGIN_STATES])
PREPAID = SI["prepaid"]
CURRENT = SI["current"]
IMPOSSIBLE_FROM_CURRENT = list(PL.IMPOSSIBLE_FROM_CURRENT)        # (foreclosure, REO) — the M20 cells


def _assert_state_only_path_dependence(scaler: F.Scaler, vocab: F.Vocab) -> None:
    """Guard the load-bearing optimization (see :func:`simulate_paths`): the per-timestep feature
    block must contain **no field derived from the (simulated) state**, so the deterministic
    ``cont``/``bin`` and the non-``state`` ``cat`` columns can be encoded ONCE per loan-month and
    reused across paths — only the ``state`` embedding varies. Verified against the canonical feature
    lists the sequence models actually consume (``sequence.build_split`` fits ``Scaler`` on
    ``F.CONTINUOUS`` and ``Vocab`` on ``F.CATEGORICAL``, and ``F.binary_matrix`` defaults to
    ``F.BINARY``):

      * ``F.CONTINUOUS`` — balances/rates/LTV/DTI/fico/MI/counts are loan statics or balances FROZEN
        at t0 by ``advance_frame``; ``Loan Age`` / ``Remaining Months to Maturity`` are deterministic
        in *h*; the macro block (``pmms30 … hpi_chg_12m``) is frozen at t0. None depend on ``state``.
      * ``F.CATEGORICAL`` — ``state`` is THE path-dependent field; ``month_of_year`` is deterministic
        in *h*; ``vintage_year`` and the loan statics (``Channel … Amortization Type``) are fixed.
      * ``F.BINARY`` — loan statics + macro fallback flags; none depend on ``state``.

    The state-DERIVED engineered-history features (``history.HIST_COLS``: ``prev_state``,
    ``ever_delinquent``, ``months_since_last_delinq``, ``n_prior_delinq_episodes``) belong to the
    FF+hist arm, NOT the sequence models — the GRU/transformer carry "memory" via the trajectory the
    network reads, not via per-row engineered counts. If any of those were wired into a sequence
    model's feature set, the per-path precompute would be **silently wrong at H>1** (the count would
    grow with the realised path but be frozen here), so this asserts they are absent."""
    enc = set(scaler.cols) | set(vocab.cols)
    derived = enc & set(HIST.HIST_COLS)
    assert not derived, (
        f"sequence feature set contains state-derived history feature(s) {sorted(derived)}; "
        f"simulate_paths' encode-once-vary-only-state precompute is INVALID when a per-timestep "
        f"feature depends on the simulated state — re-encode per path, or price via the FF+hist arm")
    extra_cont = set(scaler.cols) - set(F.CONTINUOUS)
    extra_cat = set(vocab.cols) - set(F.CATEGORICAL)
    assert not extra_cont, f"non-canonical continuous feature(s) {sorted(extra_cont)} — re-verify path-dependence"
    assert not extra_cat, f"non-canonical categorical feature(s) {sorted(extra_cat)} — re-verify path-dependence"
    assert "state" in vocab.cols, "the `state` categorical is missing — nothing to condition the chain on"


def encode_rows(raw: pl.DataFrame, scaler: F.Scaler, vocab: F.Vocab
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A raw loan-month frame → ``(cont [n, n_cont] f32, cat [n, n_cat] i64, bin [n, n_bin] f32)``
    — the per-timestep feature row the sequence models consume, via the **same** ``prepare_raw`` +
    ``Scaler``/``Vocab``/``binary_matrix`` path as ``sequence.build_split`` (so a row encoded here is
    byte-identical to the cached training tensors). Used for both the fixed history and each evolved
    future row."""
    f = F.prepare_raw(raw)
    return scaler.transform(f), vocab.transform(f), F.binary_matrix(f)


# ===========================================================================
# SeqPredictor — the ONE model-specific surface (mirrors pool.TorchPredictor's role), but its
# scoring unit is a per-loan trailing-T WINDOW, not a single advanced frame.
# ===========================================================================
class SeqPredictor:
    """Wrap a fitted ``SeqGRU``/``SeqTransformer`` + ``(scaler, vocab)`` behind a window-scoring
    seam. :meth:`score` is the only model-touching call (already-encoded ``[n, T, ·]`` tensors →
    softmax ``[n, 7]``); :meth:`origin_scores` builds, per transient origin, the window ending at
    the supplied frame and scores it, so the predictor drops into the **unchanged**
    :func:`pool.roll_forward_predictor` at H=1 (the ADR-002 regression-guard cross-check).

    ``history`` (optional) is the loan's observed trailing rows for months **before** t0 (long
    format, one row per loan-month, with ``Loan Identifier`` + the model feature columns), encoded
    ONCE at construction and grouped per loan. With ``history=None`` every window is built from the
    simulated/anchor rows alone (a valid, shorter-window setup — what the synthetic CPU test uses).
    """

    def __init__(self, model, scaler: F.Scaler, vocab: F.Vocab, device="cpu", *,
                 history: pl.DataFrame | None = None, T: int = SEQ_LEN):
        self.model = model
        self.scaler = scaler
        self.vocab = vocab
        self.device = torch.device(device) if isinstance(device, str) else device
        self.T = T
        self.state_col = vocab.cols.index("state")            # the `state` position in the cat block
        # class index 0..6 -> the window's `state` vocab index (terminal states absent from the
        # vocab fall to UNK; they only ever appear in ABSORBED paths whose scores are discarded).
        self._svi = np.array([vocab.maps["state"].get(s, F.UNK) for s in STATES], dtype=np.int64)
        self._hist: dict = {}
        if history is not None and history.height:
            hc, hk, hb = encode_rows(history, scaler, vocab)
            loans = history.get_column("Loan Identifier").to_numpy()
            # Group rows per loan in ONE pass: a STABLE sort by loan keeps each loan's rows in their
            # incoming period order, then split on the run boundaries. (The previous per-loan boolean
            # mask `loans == loan` was O(n_loans · n_rows) — quadratic, hours at the ~0.6M-loan anchor
            # scale; the grouped result is byte-identical.)
            order = np.argsort(loans, kind="stable")
            ls = loans[order]
            bounds = np.flatnonzero(ls[1:] != ls[:-1]) + 1 if ls.shape[0] > 1 else np.empty(0, np.int64)
            for g in np.split(order, bounds):
                self._hist[loans[g[0]]] = (hc[g], hk[g], hb[g])

    def history_for(self, loan):
        """The loan's encoded fixed history ``(hc, hk, hb)`` (rows < t0) or ``(None, None, None)``."""
        return self._hist.get(loan, (None, None, None))

    @torch.no_grad()
    def score(self, cont: np.ndarray, cat: np.ndarray, binb: np.ndarray,
              lengths: np.ndarray, *, batch: int = PL.EVAL_BATCH) -> np.ndarray:
        """Right-padded ``[n, T, ·]`` window arrays → softmax next-state probabilities ``[n, 7]``
        (float64 softmax then returned as float64, mirroring ``pool._forward_probs``'s numerics).
        ``lengths`` ``[n]`` are the real window lengths (for ``pack_padded`` / the padding mask)."""
        self.model.eval()
        n = cont.shape[0]
        out = np.empty((n, N_CLASSES), dtype=np.float64)
        ln = torch.as_tensor(np.asarray(lengths), dtype=torch.long)   # CPU for pack_padded
        for s in range(0, n, batch):
            sl = slice(s, s + batch)
            ct = torch.as_tensor(cont[sl], dtype=torch.float32, device=self.device)
            ck = torch.as_tensor(cat[sl], dtype=torch.long, device=self.device)
            cb = torch.as_tensor(binb[sl], dtype=torch.float32, device=self.device)
            logits = self.model(ct, ck, cb, ln[sl])
            out[sl] = torch.softmax(logits.double(), dim=1).cpu().numpy()
        return out

    def _windows_from_last(self, loans, fc, fk, fb, state_vocab_idx: int):
        """Per loan, the window ending at the supplied last row ``(fc[i], fk[i], fb[i])`` with its
        ``state`` overridden to ``state_vocab_idx``, prepended by the loan's fixed history and kept
        to the trailing ``T`` (right-padded). Returns ``[n, T, ·]`` arrays + ``lengths``."""
        n, nc, ncat, nb, T = fc.shape[0], fc.shape[1], fk.shape[1], fb.shape[1], self.T
        cont = np.zeros((n, T, nc), np.float32)
        cat = np.zeros((n, T, ncat), np.int64)
        binb = np.zeros((n, T, nb), np.float32)
        lengths = np.zeros(n, np.int64)
        for i in range(n):
            hc, hk, hb = self.history_for(loans[i])
            last_k = fk[i].copy()
            last_k[self.state_col] = state_vocab_idx
            rc = fc[i:i + 1] if hc is None else np.concatenate([hc, fc[i:i + 1]], 0)
            rk = last_k[None] if hk is None else np.concatenate([hk, last_k[None]], 0)
            rb = fb[i:i + 1] if hb is None else np.concatenate([hb, fb[i:i + 1]], 0)
            keep = min(rc.shape[0], T)
            cont[i, :keep] = rc[-keep:]
            cat[i, :keep] = rk[-keep:]
            binb[i, :keep] = rb[-keep:]
            lengths[i] = keep
        return cont, cat, binb, lengths

    def origin_scores(self, frame_h: pl.DataFrame) -> list[np.ndarray]:
        """:class:`pool.Predictor` contract: one ``[n, 7]`` block per transient origin (``current,
        dpd_30, dpd_60, dpd_90plus`` order), scoring the window that ends at ``frame_h`` (one
        advanced row per loan) with the last step's ``state`` set to that origin.

        Valid for the **composition engine only at H=1** — there the window's last real row is the
        anchor row at t0 and one composed step equals the direct one-step sequence prediction, so
        ``roll_forward_predictor(SeqPredictor, ..., horizon=1)`` reproduces the loan-level score and
        is the ADR-002 regression guard. At H>1 the composition path is **mathematically invalid**
        for a path-dependent model (``ADR-002`` Option B) — use :func:`simulate_paths` instead."""
        fc, fk, fb = encode_rows(frame_h, self.scaler, self.vocab)
        loans = frame_h.get_column("Loan Identifier").to_numpy()
        out = []
        for s in ORIGIN_STATES:
            svi = self.vocab.maps["state"].get(s, F.UNK)
            cont, cat, binb, lengths = self._windows_from_last(loans, fc, fk, fb, svi)
            out.append(self.score(cont, cat, binb, lengths))
        return out


# ===========================================================================
# Monte-Carlo roll-forward (ADR-002 Decision item 2)
# ===========================================================================
def _apply_post(probs: np.ndarray, cur: np.ndarray, calibrator, zero_impossible: bool) -> np.ndarray:
    """Apply the SAME per-origin transforms the composition path applies, in the SAME order
    (``roll_forward_predictor`` §4): the ``calibrator`` on the raw per-origin scores **first**
    (no-op when ``None``), then the M20 zeroing of the mechanically-impossible ``current→fc/REO``
    cells with a row renormalise. ``cur`` ``[n]`` is each row's *from*-state class index, so the
    transforms are keyed by the state being predicted from — exactly the composition's per-origin
    rows."""
    probs = np.array(probs, dtype=np.float64, copy=True)
    if calibrator is not None:
        for o in np.unique(cur):
            m = cur == o
            probs[m] = calibrator(probs[m], int(o))
    if zero_impossible:
        m = cur == CURRENT
        if m.any():
            probs[np.ix_(m, IMPOSSIBLE_FROM_CURRENT)] = 0.0
            probs[m] /= probs[m].sum(axis=1, keepdims=True)
    return probs


def _sample(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF categorical draw, one sample per row of ``probs`` ``[n, 7]`` → ``[n]`` class
    indices. ``P(draw = j) = probs[:, j]`` exactly (the last CDF cell is pinned to 1.0 so a
    floating-point shortfall can't leave ``u`` past the end)."""
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(probs.shape[0])
    return (u[:, None] < cdf).argmax(axis=1)


def _simulate_paths_reference(predictor: SeqPredictor, base: pl.DataFrame, t0: int, *,
                   horizon: int = PL.HORIZON, n_paths: int = 256,
                   calibrator: PL.Calibrator | None = None, zero_impossible: bool = True,
                   seed: int = 0, T: int | None = None) -> dict:
    """**Reference (oracle) implementation** — the original per-loan Monte-Carlo roll-forward,
    preserved verbatim as the known-correct ground truth that :func:`simulate_paths` (the vectorized
    production path) is differentially gated against (``tests/test_seq_price_mc.py`` /
    ``_differential_check``). Kept for that gate and as a fallback; do not delete.

    Monte-Carlo roll-forward of ``predictor`` over the loans in ``base`` (the anchor slice at
    ``t0``: one raw row per loan, with ``state`` the loan's transient origin, plus ``Loan Age`` /
    ``Remaining Months to Maturity`` / ``orig_ym`` / ``period_ym`` and the feature columns).

    For each loan, draw ``n_paths`` trajectories. At month *h* (=1..``horizon``):
      * build the trailing-``T`` window from the path so far — the loan's fixed history (rows < t0,
        held by ``predictor``) followed by the simulated current-month rows q_1..q_h, where
        ``q_m = advance_frame(base, t0, m)`` carries the sampled state at month ``t0+m-1``;
      * score the next-state distribution with :meth:`SeqPredictor.score`, then :func:`_apply_post`
        (calibrator → M20 zeroing) exactly as the composition path conditions its per-origin row;
      * **sample** the next state; absorb it if terminal (foreclosure / REO / prepaid stay put).

    Covariate evolution matches :func:`pool.advance_frame` bit-for-bit (age/term/``month_of_year``
    evolve; UPB + macro + statics frozen at t0); ``state`` is the only path-dependent field, so the
    deterministic ``cont``/``bin`` and the non-``state`` ``cat`` columns are encoded once per loan
    and reused across paths.

    Returns ``[N_loans, horizon, 7]`` ``state_dist`` (MC mean of the one-hot state at ``t0+h``),
    ``cum_prepaid`` / ``alive_before`` ``[N_loans, horizon]`` (the MC analogues of the composition
    ``capture_smm`` arrays — cumulative prepaid mass after step *h*, transient mass at the start of
    step *h*), and the per-loan conditional ``smm`` ``[N_loans, horizon]`` via the **shared**
    :func:`pool.per_loan_smm`. ``state_dist[:, 0, :]`` is the H=1 distribution (the regression-guard
    target). Seeded per loan (``rng = default_rng([seed, loan_index])``) so a loan's draw is
    reproducible independent of loan ordering."""
    T = T or predictor.T
    P = int(n_paths)
    L = base.height
    # Precondition for the per-path precompute below: `state` is the SOLE path-dependent per-timestep
    # field (checked against F.CONTINUOUS / F.CATEGORICAL / F.BINARY and history.HIST_COLS).
    _assert_state_only_path_dependence(predictor.scaler, predictor.vocab)
    loans = base.get_column("Loan Identifier").to_numpy()
    origin_class = base.select(pl.col("state").replace_strict(
        list(SI), list(SI.values()), default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1)

    # Future deterministic rows q_1..q_horizon (one per loan per h), encoded ONCE. Valid because the
    # ONLY path-dependent per-timestep field is `state` (asserted above): `cont`/`bin` and every
    # non-`state` `cat` column are either frozen at t0 (balances, macro, statics) or a deterministic
    # function of h (age/term/month_of_year), so they are identical across paths. The `state` vocab
    # index is overwritten per sampled trajectory in the loop below.
    f0c, f0k, f0b = encode_rows(PL.advance_frame(base, t0, 1), predictor.scaler, predictor.vocab)
    nc, ncat, nb = f0c.shape[1], f0k.shape[1], f0b.shape[1]
    fcont = np.empty((horizon, L, nc), np.float32)
    fcat = np.empty((horizon, L, ncat), np.int64)
    fbin = np.empty((horizon, L, nb), np.float32)
    fcont[0], fcat[0], fbin[0] = f0c, f0k, f0b
    for h in range(2, horizon + 1):
        c, k, b = encode_rows(PL.advance_frame(base, t0, h), predictor.scaler, predictor.vocab)
        fcont[h - 1], fcat[h - 1], fbin[h - 1] = c, k, b

    state_dist = np.zeros((L, horizon, N_CLASSES), np.float64)
    cum_prepaid = np.zeros((L, horizon), np.float64)
    alive_before = np.zeros((L, horizon), np.float64)
    eye = np.eye(N_CLASSES)

    for li in range(L):
        rng = np.random.default_rng([seed, li])
        hc, hk, hb = predictor.history_for(loans[li])
        Lh = 0 if hc is None else hc.shape[0]
        states = np.empty((P, horizon + 1), np.int64)         # state at months t0, t0+1, ..., t0+H
        states[:, 0] = origin_class[li]
        alive = np.ones(P, dtype=bool)
        cur = np.full(P, origin_class[li], dtype=np.int64)

        for h in range(1, horizon + 1):
            total = Lh + h
            keep = min(total, T)
            drop = total - keep
            # Assemble the [P, total, ·] window: shared history + shared future cont/bin, with the
            # future `state` column set per path from the trajectory so far (states[:, 0:h] are the
            # from-states for q_1..q_h, i.e. the state at months t0..t0+h-1).
            cont_full = np.empty((P, total, nc), np.float32)
            cat_full = np.empty((P, total, ncat), np.int64)
            bin_full = np.empty((P, total, nb), np.float32)
            if Lh:
                cont_full[:, :Lh, :] = hc[None]
                cat_full[:, :Lh, :] = hk[None]
                bin_full[:, :Lh, :] = hb[None]
            cont_full[:, Lh:, :] = fcont[:h, li, :][None]
            bin_full[:, Lh:, :] = fbin[:h, li, :][None]
            cat_full[:, Lh:, :] = fcat[:h, li, :][None]
            cat_full[:, Lh:, predictor.state_col] = predictor._svi[states[:, 0:h]]

            cont_win = np.zeros((P, T, nc), np.float32)
            cat_win = np.zeros((P, T, ncat), np.int64)
            bin_win = np.zeros((P, T, nb), np.float32)
            cont_win[:, :keep, :] = cont_full[:, drop:, :]
            cat_win[:, :keep, :] = cat_full[:, drop:, :]
            bin_win[:, :keep, :] = bin_full[:, drop:, :]
            lengths = np.full(P, keep, np.int64)

            probs = predictor.score(cont_win, cat_win, bin_win, lengths)
            probs = _apply_post(probs, cur, calibrator, zero_impossible)
            nxt = _sample(probs, rng)
            nxt = np.where(alive, nxt, cur)                   # absorbed paths carry their terminal
            states[:, h] = nxt
            alive &= np.isin(nxt, ORIGIN_ROWS)
            cur = nxt

        state_dist[li] = eye[states[:, 1:]].mean(axis=0)                          # [H, 7]
        cum_prepaid[li] = (states[:, 1:] == PREPAID).mean(axis=0)                 # cumulative (absorbing)
        alive_before[li] = np.isin(states[:, 0:horizon], ORIGIN_ROWS).mean(axis=0)

    smm = PL.per_loan_smm(cum_prepaid.astype(np.float32), alive_before.astype(np.float32))
    return {"n_loans": L, "horizon": horizon, "n_paths": P, "t0": t0,
            "state_dist": state_dist, "cum_prepaid": cum_prepaid,
            "alive_before": alive_before, "smm": smm,
            "h1_dist": state_dist[:, 0, :]}


# ---------------------------------------------------------------------------
# Vectorized roll-forward — the production path. Batches B loans × P paths into ONE
# SeqPredictor.score call per (loan-chunk, month), instead of the reference's per-loan score call.
# It is the SAME algorithm as _simulate_paths_reference (identical window assembly, post-processing,
# per-loan RNG streams drawn in month order, and reductions); the ONLY numerical difference is the
# batched GPU forward's float-rounding (batched cuDNN/matmul is not bit-associative), which is pure
# Monte-Carlo-tolerance noise and is gated against the reference. Both seq archs are per-sequence
# (GRU pack_padded; transformer within-sequence attention), so a mixed-loan batch is mathematically
# identical to the per-loan batch. Histories are ≤ T−1 rows (anchor_history pulls T−1 trailing
# months), so the per-chunk combined buffer is tiny (≤ T−1+horizon columns).
# ---------------------------------------------------------------------------
SIM_LOAN_BATCH_CAP = 4096              # max loans per vectorized chunk (bounds the transient [B,P,T,·] arrays)
SIM_SEQ_TARGET = 1_048_576            # target sequences (B·P) per chunk; B = clip(SIM_SEQ_TARGET // P)
SIM_GPU_BATCH = 32_768                # sequences per on-GPU model forward (= EVAL_BATCH). MUST stay below the
                                      # CUDA 65535 grid-dim limit the transformer's attention kernels hit
                                      # (the GRU tolerates larger; the transformer raises "invalid configuration
                                      # argument"). The per-month CPU reductions dominate anyway, so a bigger
                                      # forward batch buys little.


def _loan_batch_for(P: int) -> int:
    return max(1, min(SIM_LOAN_BATCH_CAP, SIM_SEQ_TARGET // max(1, int(P))))


def _apply_post_batched(probs: np.ndarray, cur: np.ndarray, calibrator, zero_impossible: bool) -> np.ndarray:
    """Vectorized :func:`_apply_post` over a ``[B, P, 7]`` block (``cur`` ``[B, P]`` from-states).
    Fast path for ``calibrator is None`` (this run's case — M23 no-op for the torch path); falls
    back to the exact per-loan :func:`_apply_post` when a calibrator is supplied, preserving its
    per-origin semantics bit-for-bit."""
    if calibrator is not None:
        out = np.array(probs, dtype=np.float64, copy=True)
        for j in range(probs.shape[0]):
            out[j] = _apply_post(probs[j], cur[j], calibrator, zero_impossible)
        return out
    probs = np.array(probs, dtype=np.float64, copy=True)
    if zero_impossible:
        cmask = cur == CURRENT
        if cmask.any():
            sub = probs[cmask]
            sub[:, IMPOSSIBLE_FROM_CURRENT] = 0.0
            sub /= sub.sum(axis=1, keepdims=True)
            probs[cmask] = sub
    return probs


@torch.no_grad()
def _score_paths_gpu(predictor, wc, wb, wk, statecol, keep, state_col, gpu_batch):
    """Score one (loan-chunk, month) batch, expanding paths **on the GPU**. ``wc``/``wb``/``wk``
    ``[B,T,·]`` are per-loan (path-invariant — cont/bin and non-``state`` cat are identical across
    paths), ``statecol`` ``[B,P,T]`` is the per-path ``state`` vocab index. Only these (not the
    P-expanded windows) cross the bus; the model then sees ``[B·P,T,·]`` via ``torch.expand`` +
    contiguous reshape on-device. Numerically identical to :meth:`SeqPredictor.score` (eval, no-grad,
    float64 softmax) — avoiding the per-path numpy materialization that dominated the naive
    vectorization. Returns ``[B,P,7]`` float64."""
    dev = predictor.device
    B, T, nc = wc.shape
    P = statecol.shape[1]
    ncat, nb = wk.shape[2], wb.shape[2]
    model = predictor.model
    model.eval()
    cont = torch.as_tensor(wc, dtype=torch.float32, device=dev).unsqueeze(1).expand(B, P, T, nc).reshape(B * P, T, nc)
    binb = torch.as_tensor(wb, dtype=torch.float32, device=dev).unsqueeze(1).expand(B, P, T, nb).reshape(B * P, T, nb)
    cat = torch.as_tensor(wk, dtype=torch.long, device=dev).unsqueeze(1).expand(B, P, T, ncat).clone()
    cat[:, :, :, state_col] = torch.as_tensor(statecol, dtype=torch.long, device=dev)
    cat = cat.reshape(B * P, T, ncat)
    lengths = torch.as_tensor(np.repeat(keep, P), dtype=torch.long)        # CPU for pack_padded
    n = B * P
    out = np.empty((n, N_CLASSES), np.float64)
    for s in range(0, n, gpu_batch):
        sl = slice(s, s + gpu_batch)
        logits = model(cont[sl].contiguous(), cat[sl].contiguous(), binb[sl].contiguous(), lengths[sl])
        out[sl] = torch.softmax(logits.double(), dim=1).cpu().numpy()
    return out.reshape(B, P, N_CLASSES)


def _simulate_loan_chunk(predictor, idx, Lh_all, fcont, fcat, fbin, origin_class, loans, *,
                         horizon, T, P, seed, calibrator, zero_impossible, eye, state_col, svi,
                         gpu_batch):
    """Roll one chunk of loans (global indices ``idx``) forward, batched. Returns ``(state_dist
    [Bc,H,7], cum_prepaid [Bc,H], alive_before [Bc,H])`` — the reference's per-loan outputs for these
    loans, computed together."""
    Bc = len(idx)
    nc, ncat, nb = fcont.shape[2], fcat.shape[2], fbin.shape[2]
    Lh = Lh_all[idx]
    Lhmax = int(Lh.max())
    W = Lhmax + horizon
    # Combined per-loan sequence buffer: history at cols [Lhmax-Lh : Lhmax], future at [Lhmax : W].
    seqc = np.zeros((Bc, W, nc), np.float32)
    seqk = np.zeros((Bc, W, ncat), np.int64)
    seqb = np.zeros((Bc, W, nb), np.float32)
    seqc[:, Lhmax:, :] = np.transpose(fcont[:, idx, :], (1, 0, 2))
    seqk[:, Lhmax:, :] = np.transpose(fcat[:, idx, :], (1, 0, 2))          # state col is a placeholder
    seqb[:, Lhmax:, :] = np.transpose(fbin[:, idx, :], (1, 0, 2))
    for j in range(Bc):                                                    # history fill (once per chunk)
        lh = int(Lh[j])
        if lh:
            hc, hk, hb = predictor.history_for(loans[idx[j]])
            seqc[j, Lhmax - lh:Lhmax, :] = hc
            seqk[j, Lhmax - lh:Lhmax, :] = hk
            seqb[j, Lhmax - lh:Lhmax, :] = hb

    oc = origin_class[idx]
    # zero-init (not empty): the batched state gather may read not-yet-written future indices at
    # masked-out (t≥keep) window positions; those reads must be a valid class (0) the mask discards.
    states = np.zeros((Bc, P, horizon + 1), np.int64)
    states[:, :, 0] = oc[:, None]
    alive = np.ones((Bc, P), bool)
    cur = np.repeat(oc[:, None], P, axis=1)
    rngs = [np.random.default_rng([seed, int(idx[j])]) for j in range(Bc)]  # per-loan streams (loan order)
    ar = np.arange(Bc)[:, None]
    t_ar = np.arange(T)

    for h in range(1, horizon + 1):
        keep = np.minimum(Lh + h, T)                                      # [Bc]
        start = Lhmax + h - keep                                          # [Bc]
        col = np.clip(start[:, None] + t_ar[None, :], 0, W - 1)           # [Bc,T]
        valid = (t_ar[None, :] < keep[:, None])                          # [Bc,T]
        wc = seqc[ar, col] * valid[..., None]
        wb = seqb[ar, col] * valid[..., None]
        wk = seqk[ar, col] * valid[..., None]
        is_future = (col >= Lhmax) & valid                               # [Bc,T]
        month = np.clip(col - Lhmax, 0, horizon)                         # future-row index (0..h-1)
        gathered = np.take_along_axis(states, month[:, None, :], axis=2)  # [Bc,P,T]
        statecol = np.where(is_future[:, None, :], svi[gathered],
                            np.broadcast_to(wk[:, None, :, state_col], (Bc, P, T)))   # [Bc,P,T] state vocab idx
        # Score with on-GPU path expansion (the per-path windows never materialize in numpy).
        probs = _score_paths_gpu(predictor, wc, wb, wk, statecol, keep, state_col, gpu_batch)
        probs = _apply_post_batched(probs, cur, calibrator, zero_impossible)
        u = np.empty((Bc, P), np.float64)
        for j in range(Bc):
            u[j] = rngs[j].random(P)                                      # draw in month order → reference stream
        cdf = np.cumsum(probs, axis=2)
        cdf[:, :, -1] = 1.0
        nxt = (u[:, :, None] < cdf).argmax(axis=2)                       # [Bc,P]
        nxt = np.where(alive, nxt, cur)
        states[:, :, h] = nxt
        alive &= np.isin(nxt, ORIGIN_ROWS)
        cur = nxt

    sd = eye[states[:, :, 1:]].mean(axis=1)                              # [Bc,H,7]
    cp = (states[:, :, 1:] == PREPAID).mean(axis=1)                      # [Bc,H]
    ab = np.isin(states[:, :, 0:horizon], ORIGIN_ROWS).mean(axis=1)     # [Bc,H]
    return sd, cp, ab


def simulate_paths(predictor: SeqPredictor, base: pl.DataFrame, t0: int, *,
                   horizon: int = PL.HORIZON, n_paths: int = 256,
                   calibrator: PL.Calibrator | None = None, zero_impossible: bool = True,
                   seed: int = 0, T: int | None = None, loan_batch: int | None = None) -> dict:
    """Vectorized Monte-Carlo roll-forward (production path) — the throughput version of
    :func:`_simulate_paths_reference` with an identical contract and (to Monte-Carlo tolerance)
    identical output. It batches ``loan_batch`` loans × ``n_paths`` paths into one
    :meth:`SeqPredictor.score` call per month (the reference scores one loan at a time), collapsing
    the ~``L·H`` tiny GPU forwards that left the GPU idle into a few large, well-utilized ones. The
    window assembly, :func:`_apply_post`, per-loan ``rng = default_rng([seed, loan_index])`` streams
    (drawn in month order), terminal absorption, and reductions all match the reference exactly; only
    the batched forward's float-rounding differs (gated against the reference, ``_differential_check``
    / ``tests``). ``loan_batch`` defaults to :func:`_loan_batch_for`; on CUDA OOM it is halved and the
    chunk retried (never reducing ``n_paths`` or the pop). See :func:`_simulate_paths_reference` for
    the full algorithm docstring."""
    T = T or predictor.T
    P = int(n_paths)
    L = base.height
    _assert_state_only_path_dependence(predictor.scaler, predictor.vocab)
    loans = base.get_column("Loan Identifier").to_numpy()
    origin_class = base.select(pl.col("state").replace_strict(
        list(SI), list(SI.values()), default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1)

    f0c, f0k, f0b = encode_rows(PL.advance_frame(base, t0, 1), predictor.scaler, predictor.vocab)
    nc, ncat, nb = f0c.shape[1], f0k.shape[1], f0b.shape[1]
    fcont = np.empty((horizon, L, nc), np.float32)
    fcat = np.empty((horizon, L, ncat), np.int64)
    fbin = np.empty((horizon, L, nb), np.float32)
    fcont[0], fcat[0], fbin[0] = f0c, f0k, f0b
    for h in range(2, horizon + 1):
        c, k, b = encode_rows(PL.advance_frame(base, t0, h), predictor.scaler, predictor.vocab)
        fcont[h - 1], fcat[h - 1], fbin[h - 1] = c, k, b

    Lh_all = np.array([(hc.shape[0] if (hc := predictor.history_for(loans[i])[0]) is not None else 0)
                       for i in range(L)], dtype=np.int64)
    state_dist = np.zeros((L, horizon, N_CLASSES), np.float64)
    cum_prepaid = np.zeros((L, horizon), np.float64)
    alive_before = np.zeros((L, horizon), np.float64)
    eye = np.eye(N_CLASSES)
    state_col, svi = predictor.state_col, predictor._svi

    B = loan_batch or _loan_batch_for(P)
    li0 = 0
    next_report = max(1, L // 10)                              # progress every ~10% (observability for the
    t_start = time.perf_counter()                              # multi-hour full-pop run — silent otherwise)
    while li0 < L:
        Bc = min(B, L - li0)
        idx = np.arange(li0, li0 + Bc)
        try:
            sd, cp, ab = _simulate_loan_chunk(
                predictor, idx, Lh_all, fcont, fcat, fbin, origin_class, loans,
                horizon=horizon, T=T, P=P, seed=seed, calibrator=calibrator,
                zero_impossible=zero_impossible, eye=eye, state_col=state_col, svi=svi,
                gpu_batch=min(SIM_GPU_BATCH, Bc * P))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if B == 1:
                raise
            B = max(1, B // 2)                                            # reduce B (never P / the pop), retry
            continue
        state_dist[idx], cum_prepaid[idx], alive_before[idx] = sd, cp, ab
        li0 += Bc
        if li0 >= next_report and li0 < L:
            print(f"    [simulate_paths] {li0:,}/{L:,} loans (P={P}, {li0/(time.perf_counter()-t_start):,.0f} loans/s)",
                  flush=True)
            next_report += max(1, L // 10)

    smm = PL.per_loan_smm(cum_prepaid.astype(np.float32), alive_before.astype(np.float32))
    return {"n_loans": L, "horizon": horizon, "n_paths": P, "t0": t0,
            "state_dist": state_dist, "cum_prepaid": cum_prepaid,
            "alive_before": alive_before, "smm": smm,
            "h1_dist": state_dist[:, 0, :]}


def _differential_check(predictor: SeqPredictor, base: pl.DataFrame, t0: int, *,
                        horizon: int = PL.HORIZON, n_paths: int = 500, seed: int = 0,
                        calibrator: PL.Calibrator | None = None) -> dict:
    """Run the reference and the vectorized simulator on the SAME inputs/seed and report their
    agreement — the differential gate for :func:`simulate_paths`. Returns per-array max abs diffs
    (``state_dist``/``cum_prepaid``/``alive_before``/``smm``), the pool-level ``price``/``wal`` diffs
    (through the unchanged :func:`price_paths`), ``bit_exact`` (all per-loan arrays identical), and
    the O(1/√N) band ``mc_tol`` for the caller to assert against. Callers (tests / the pre-run gate)
    must fail loudly if it is neither bit-exact nor within tolerance."""
    ref = _simulate_paths_reference(predictor, base, t0, horizon=horizon, n_paths=n_paths,
                                    seed=seed, calibrator=calibrator)
    vec = simulate_paths(predictor, base, t0, horizon=horizon, n_paths=n_paths,
                         seed=seed, calibrator=calibrator)
    arr = ("state_dist", "cum_prepaid", "alive_before", "smm")
    d = {a: float(np.abs(ref[a] - vec[a]).max()) for a in arr}
    pr_ref, pr_vec = price_paths(ref, base), price_paths(vec, base)
    d["pool_price"] = abs(float(pr_ref["pool_price"]) - float(pr_vec["pool_price"]))
    d["pool_wal"] = abs(float(pr_ref["pool_wal"]) - float(pr_vec["pool_wal"]))
    d["bit_exact"] = all(d[a] == 0.0 for a in arr)
    d["mc_tol"] = 6.0 / np.sqrt(max(1, n_paths))            # O(1/√N) band for the distribution arrays
    d["n_paths"] = int(n_paths)
    return d


# ===========================================================================
# Pricing (ADR-002 Decision item 3) — reuse the UNCHANGED §5.2 engine
# ===========================================================================
def _pricing_inputs(base: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(wac, wam, upb)`` from the anchor frame in the engine's units (``smm_paths``/``pricing_grid``
    convention): ``wac`` = ``Current Interest Rate`` as an annual **decimal**, ``wam`` = ``Remaining
    Months to Maturity``, ``upb`` = ``Current Actual UPB``."""
    wac = (base.get_column("Current Interest Rate").cast(pl.Float64) / 100.0).to_numpy()
    wam = base.get_column("Remaining Months to Maturity").cast(pl.Float64).to_numpy()
    upb = base.get_column("Current Actual UPB").cast(pl.Float64).to_numpy()
    return wac, wam, upb


def price_paths(sim: dict, base: pl.DataFrame, *,
                servicing_spread: float = PL.SERVICING_SPREAD) -> dict:
    """Price the simulation by feeding each loan's MC conditional SMM path ``sim["smm"]`` through
    the **existing** :func:`pool.price_loans_vec` (each loan a one-loan "pool" with its own
    ``wac``/``wam``/``upb`` from ``base``), then aggregate to the pool — the ADR-001 §M22 pattern,
    no new engine math. Returns ``price_loans_vec``'s dict (per-loan ``price``/``wal``/``value`` +
    the pool aggregate). Row order is ``base``'s, which ``simulate_paths`` preserves."""
    wac, wam, upb = _pricing_inputs(base)
    return PL.price_loans_vec(wac, wam, upb, sim["smm"], servicing_spread=servicing_spread)
