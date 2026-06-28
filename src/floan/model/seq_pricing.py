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
            # rows arrive grouped/period-ordered per loan (the panel window order); index by loan.
            for loan in dict.fromkeys(loans.tolist()):
                m = loans == loan
                self._hist[loan] = (hc[m], hk[m], hb[m])

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


def simulate_paths(predictor: SeqPredictor, base: pl.DataFrame, t0: int, *,
                   horizon: int = PL.HORIZON, n_paths: int = 256,
                   calibrator: PL.Calibrator | None = None, zero_impossible: bool = True,
                   seed: int = 0, T: int | None = None) -> dict:
    """Monte-Carlo roll-forward of ``predictor`` over the loans in ``base`` (the anchor slice at
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
