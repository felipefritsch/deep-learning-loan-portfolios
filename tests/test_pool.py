"""M13 unit tests — the roll-forward composition engine (``pool.py``): the pure-NumPy Layer-1
functions, exercised with no SSD, no GPU, and no model artifacts. Covers the Accept criterion
"toy-chain unit test matches closed form":

  * homogeneous chain composed H steps == ``p0 · P^H`` (matrix-power closed form);
  * a hand-computed two-step example;
  * ``assemble_matrix`` gives stochastic rows (terminal states absorbing) and conserves
    probability mass through composition;
  * the first-passage variant (``absorb_rows`` + compose) equals an independent
    brute-force path enumeration of "ever reaches the target within H steps".

Run:  .venv/bin/python -m floan.model.test_pool     (or python -m pytest)
"""

from __future__ import annotations

import numpy as np
import polars as pl

try:  # pool.py imports torch at module load; skip cleanly when torch is absent
    import torch  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("torch not installed", allow_module_level=True)

from floan.model import evaluate as E
from floan.model import pool as P


# ===========================================================================
# compose == matrix-power closed form (the toy-chain Accept)
# ===========================================================================
def _row_stochastic(rng, s):
    M = rng.random((s, s)) + 0.05
    return M / M.sum(axis=1, keepdims=True)


def _softmax_rows(rng, n, s=None):
    """``n`` random probability rows over ``s`` (default N_CLASSES) destinations."""
    r = rng.random((n, s or P.N_CLASSES))
    return r / r.sum(axis=1, keepdims=True)


def test_compose_matches_matrix_power():
    """A time-homogeneous chain (same P each step) has the closed-form H-step distribution
    ``p0 · P^H``. The engine composes H per-step copies of P and must reproduce it."""
    rng = np.random.default_rng(0)
    s, H, n = 7, 12, 5
    Pmat = _row_stochastic(rng, s)
    p0 = _softmax_rows(rng, n, s)                       # n independent loans, valid distributions
    mats = [np.broadcast_to(Pmat, (n, s, s)) for _ in range(H)]

    got = P.compose(p0, mats)
    closed = p0 @ np.linalg.matrix_power(Pmat, H)
    assert np.allclose(got, closed, atol=1e-12, rtol=0)
    assert np.allclose(got.sum(axis=1), 1.0, atol=1e-12)   # mass conserved


def test_compose_hand_two_step():
    """Hand-computed 2-state, 2-step example. P = [[.8,.2],[.3,.7]], p0 = [1,0].
    Step1: [.8,.2]; Step2: [.8·.8+.2·.3, .8·.2+.2·.7] = [.70, .30]."""
    Pmat = np.array([[0.8, 0.2], [0.3, 0.7]])
    p0 = np.array([[1.0, 0.0]])
    got = P.compose(p0, [Pmat[None], Pmat[None]])
    assert np.allclose(got, [[0.70, 0.30]], atol=1e-12)


# ===========================================================================
# assemble_matrix — terminal rows absorbing, rows stochastic, mass conserved
# ===========================================================================
def test_assemble_matrix_structure_and_mass():
    """``assemble_matrix`` places the 4 transient-origin softmax rows and makes the 3 terminal
    states absorbing (identity rows). Every row sums to 1, so composing any number of steps
    keeps the state vector a probability distribution."""
    rng = np.random.default_rng(1)
    n = 8
    scores = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    M = P.assemble_matrix(scores)
    assert M.shape == (n, P.N_CLASSES, P.N_CLASSES)
    # every row of every loan's matrix sums to 1.
    assert np.allclose(M.sum(axis=2), 1.0, atol=1e-12)
    # terminal rows are the identity.
    for ti in P.TERMINAL_ROWS:
        ident = np.zeros(P.N_CLASSES); ident[ti] = 1.0
        assert np.allclose(M[:, ti, :], ident, atol=0)
    # transient rows equal the supplied scores.
    for oi, row in zip(P.ORIGIN_ROWS, scores):
        assert np.allclose(M[:, oi, :], row, atol=0)
    # compose 12 steps → still a distribution.
    p0 = P.onehot_origin(np.zeros(n, dtype=int))       # all start "current"
    p = P.compose(p0, [M] * 12)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12)


# ===========================================================================
# M20 — two-cell zeroing (current→foreclosure / current→REO) is likelihood-neutral
# ===========================================================================
def test_zero_impossible_is_likelihood_neutral():
    """M20 (ECONOMIC_ENGINE §3.5): zeroing the two mechanically-impossible one-step cells out of
    `current` and renormalising removes phantom high-LGD mass while leaving the pooled NLL
    unchanged to ≤1e-9 — the data never realises current→foreclosure/REO (monotone delinquency),
    so the zeroed cells carry no observed likelihood."""
    rng = np.random.default_rng(20)
    n = 20_000
    cur, fc, reo = P.SI["current"], P.SI["foreclosure"], P.SI["REO"]
    reach = [P.SI[s] for s in ("current", "dpd_30", "dpd_60", "dpd_90plus", "prepaid")]

    # Per-loan current-origin prediction: dominant mass on reachable cells + realistic tiny
    # phantom (~1e-10/cell) on the mechanically-impossible foreclosure/REO cells.
    row = np.zeros((n, P.N_CLASSES))
    r = rng.random((n, len(reach)))
    row[:, reach] = r / r.sum(axis=1, keepdims=True)
    row[:, [fc, reo]] = rng.uniform(1e-11, 1e-10, size=(n, 2))
    row /= row.sum(axis=1, keepdims=True)

    scores = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    scores[P.ORIGIN_STATES.index("current")] = row
    M = P.assemble_matrix(scores)
    Mz = P._zero_impossible(M)

    # current-origin loans: the h=1 distribution is the matrix's `current` row; realised
    # next-states are drawn only from reachable cells (never foreclosure/REO).
    y = np.array(reach)[rng.integers(0, len(reach), n)]
    nll_before, nll_after = E._nll(M[:, cur, :], y), E._nll(Mz[:, cur, :], y)

    assert abs(nll_after - nll_before) <= 1e-9                       # likelihood-neutral
    assert M[:, cur, [fc, reo]].sum() > 0.0                          # phantom mass existed
    assert np.allclose(Mz[:, cur, [fc, reo]], 0.0)                   # phantom mass removed
    assert np.allclose(Mz[:, cur, :].sum(axis=1), 1.0, atol=1e-12)   # row still stochastic
    for oi in P.ORIGIN_ROWS:                                          # only the current row changed
        if oi != cur:
            assert np.array_equal(Mz[:, oi, :], M[:, oi, :])


# ===========================================================================
# first-passage variant == independent brute-force "ever reaches target"
# ===========================================================================
def _ever_hit_bruteforce(Pmat, p0, target, H):
    """Ground truth: probability of visiting ``target`` at any step 0..H, by explicit
    enumeration of every length-H path weighted by ``p0`` and the transition probabilities."""
    s = len(p0)
    total = 0.0

    def rec(state, steps_left, prob, hit):
        nonlocal total
        hit = hit or (state == target)
        if steps_left == 0:
            if hit:
                total += prob
            return
        for nxt in range(s):
            rec(nxt, steps_left - 1, prob * Pmat[state, nxt], hit)

    for st in range(s):
        rec(st, H, p0[st], False)
    return total


def test_first_passage_matches_bruteforce():
    """Making ``target`` absorbing and composing puts all its first-passage mass on ``target``;
    that horizon mass must equal the brute-force probability of ever reaching it."""
    rng = np.random.default_rng(2)
    s, H, target = 4, 3, 1
    Pmat = _row_stochastic(rng, s)
    p0 = _softmax_rows(rng, 1, s)[0]                   # one distribution over s states

    Mabs = P.absorb_rows(np.broadcast_to(Pmat, (1, s, s)), target)
    p = P.compose(p0[None], [Mabs] * H)
    got = float(p[0, target])
    truth = _ever_hit_bruteforce(Pmat, p0, target, H)
    assert abs(got - truth) < 1e-12, (got, truth)


def test_absorb_rows_is_identity_row():
    """``absorb_rows`` replaces exactly the target row with ``e_target`` and leaves the rest."""
    rng = np.random.default_rng(3)
    s = 5
    M = _row_stochastic(rng, s)[None]
    out = P.absorb_rows(M, 2)
    ident = np.zeros(s); ident[2] = 1.0
    assert np.allclose(out[0, 2, :], ident, atol=0)
    assert np.allclose(np.delete(out[0], 2, axis=0), np.delete(M[0], 2, axis=0), atol=0)


def test_first_passage_monotone_concentrates_at_dpd60():
    """In a monotone delinquency chain, making ``dpd_60`` absorbing leaves the deeper states
    (dpd_90plus, foreclosure, REO) unreachable, so all 60+ mass concentrates at ``dpd_60`` —
    the property the harness relies on for ``P(60+ within 12m)``."""
    rng = np.random.default_rng(4)
    n = 6
    # monotone delinquency structure (02 §1 / evaluate.REACHABLE): a transient origin may
    # only transition to its own level ±1 plus cure/prepay, so reaching dpd_90plus requires
    # passing through dpd_60. Build per-origin rows with mass only on reachable destinations.
    reach = {
        "current":    ["current", "dpd_30", "prepaid"],
        "dpd_30":     ["current", "dpd_30", "dpd_60", "prepaid"],
        "dpd_60":     ["current", "dpd_30", "dpd_60", "dpd_90plus", "prepaid"],
        "dpd_90plus": ["current", "dpd_30", "dpd_60", "dpd_90plus", "foreclosure", "prepaid"],
    }
    scores = []
    for s in P.ORIGIN_STATES:
        row = np.zeros((n, P.N_CLASSES))
        cols = [P.SI[d] for d in reach[s]]
        row[:, cols] = rng.random((n, len(cols))) + 0.05
        scores.append(row / row.sum(axis=1, keepdims=True))
    M = P.assemble_matrix(scores)
    Mfp = P.absorb_rows(M, P.DPD60)
    p0 = P.onehot_origin(np.zeros(n, dtype=int))       # start current
    pfp = P.compose(p0, [Mfp] * 12)
    deeper = [P.SI["dpd_90plus"], P.SI["foreclosure"], P.SI["REO"]]
    assert np.allclose(pfp[:, deeper], 0.0, atol=1e-12)   # unreachable once dpd_60 absorbs
    assert (pfp[:, P.DPD60] > 0).all()                    # some first-passage mass


# ===========================================================================
# Layer 3 — cashflow engine vs closed forms (03_POOL_LEVEL §5.2; the M15a (B) gate)
# ===========================================================================
def _annuity_pv(pmt, y_m, n):
    """PV of an ``n``-month annuity-immediate of ``pmt`` discounted at monthly ``y_m``."""
    return pmt * (1.0 - (1.0 + y_m) ** (-n)) / y_m


def test_engine_par_identity_any_smm():
    """``servicing_spread = 0`` ⇒ discount rate = the note coupon, so the pool is worth par for
    *any* prepayment vector: each outstanding dollar earns exactly the discount rate and principal
    returns at par (telescoping ``Σ cf_h/(1+c)^h = B_0 − B_n/(1+c)^n``, ``B_n=0``). The strongest
    hermetic check — exercises amortisation, the SMM prepay step and discounting at once."""
    rng = np.random.default_rng(0)
    wac, wam, upb = 0.06, 360, 250_000.0
    for smm in (np.zeros(12),                                  # no prepay
                np.full(12, 0.02),                             # constant
                rng.uniform(0, 0.08, 12),                      # arbitrary path
                rng.uniform(0, 0.4, 12)):                      # extreme speeds
        r = P.cashflow_engine(wac, wam, upb, smm, servicing_spread=0.0)
        assert abs(r["price"] - 100.0) < 1e-8, (smm[:3], r["price"])
        assert abs(r["total_principal"] - upb) < 1e-6 * upb   # all principal returned, B_n=0


def test_engine_zero_prepay_is_annuity():
    """``smm ≡ 0`` ⇒ a level-pay loan: constant payment ``PMT = UPB·c/(1−(1+c)^−n)`` for all n
    months, so the price is the closed-form annuity PV discounted at ``y = wac − spread``."""
    wac, wam, upb, spread = 0.05, 360, 100.0, 0.0025
    r = P.cashflow_engine(wac, wam, upb, np.zeros(6), servicing_spread=spread)  # extrapolates 0
    c, y_m, n = wac / 12.0, (wac - spread) / 12.0, 360
    pmt = upb * c / (1.0 - (1.0 + c) ** (-n))
    assert np.allclose(r["cashflow"], pmt, atol=1e-8)         # constant level payment
    assert abs(r["total_principal"] - upb) < 1e-8 * upb
    price_ref = 100.0 * _annuity_pv(pmt, y_m, n) / upb
    assert abs(r["price"] - price_ref) < 1e-8, (r["price"], price_ref)
    assert r["price"] > 100.0                                 # premium bond (y < coupon)


def test_engine_constant_smm_survival_schedule():
    """``smm ≡ λ`` ⇒ the closed-form survival schedule ``B_h = UPB·φ_h·(1−λ)^h`` with ``φ_h`` the
    no-prepay scheduled-balance fraction. Reconstruct balance / cashflow / price / WAL from the
    closed form (independent of the engine's recursion) and require ~1e-8 agreement."""
    wac, wam, upb, spread, lam = 0.045, 360, 500_000.0, 0.0025, 0.015
    r = P.cashflow_engine(wac, wam, upb, np.full(12, lam), servicing_spread=spread)
    c, y_m, n = wac / 12.0, (wac - spread) / 12.0, 360
    h = np.arange(0, n + 1)
    phi = ((1.0 + c) ** n - (1.0 + c) ** h) / ((1.0 + c) ** n - 1.0)   # scheduled-balance fraction
    B = upb * phi * (1.0 - lam) ** h                                   # survival schedule, h=0..n
    total_prin_ref = B[:-1] - B[1:]                                    # months 1..n
    interest_ref = B[:-1] * c
    cf_ref = interest_ref + total_prin_ref
    months = np.arange(1, n + 1)
    price_ref = 100.0 * float((cf_ref * (1.0 + y_m) ** (-months)).sum()) / upb
    wal_ref = float((months * total_prin_ref).sum() / (12.0 * total_prin_ref.sum()))

    assert np.allclose(r["balance"], B[:-1], rtol=1e-9, atol=1e-6)     # start-of-month balances
    assert np.allclose(r["cashflow"], cf_ref, rtol=1e-9, atol=1e-6)
    assert abs(r["price"] - price_ref) < 1e-8, (r["price"], price_ref)
    assert abs(r["wal"] - wal_ref) < 1e-8, (r["wal"], wal_ref)


def test_engine_price_monotone_in_prepay_speed():
    """Economic sanity (not a closed form): for a premium pass-through (``spread>0``) a faster
    constant SMM erodes the premium, so price is strictly decreasing in CPR — the property §5.2
    relies on to read price error as prepayment-model error."""
    wac, wam, upb = 0.06, 360, 100.0
    prices = [P.cashflow_engine(wac, wam, upb, np.full(12, s))["price"]
              for s in (0.0, 0.005, 0.01, 0.02, 0.04)]
    assert all(p > 100.0 for p in prices)                    # premium throughout
    assert all(a > b for a, b in zip(prices, prices[1:]))    # strictly decreasing in speed


def test_engine_constant_extrapolation_past_horizon():
    """A 12-vector SMM drives a 360-month pool by holding the terminal SMM constant past h=12 —
    so a 12-long constant vector and a full-length constant vector price identically."""
    wac, wam, upb, lam = 0.05, 360, 100.0, 0.03
    r12 = P.cashflow_engine(wac, wam, upb, np.full(12, lam))
    rfull = P.cashflow_engine(wac, wam, upb, np.full(360, lam))
    assert abs(r12["price"] - rfull["price"]) < 1e-10
    assert abs(r12["wal"] - rfull["wal"]) < 1e-10


# ===========================================================================
# M21 — the agnostic engine (roll_forward_predictor) + Predictor/Calibrator seam
# (ECONOMIC_ENGINE §3 / ADR-001). Pure: a stub Predictor + a toy frame, no SSD/GPU/models.
# ===========================================================================
class _StubPredictor:
    """A non-torch Predictor returning fixed per-origin rows regardless of ``frame_h`` — exercises
    the engine seam (assemble → calibrate → zero → absorb → compose) with no model artifacts."""

    def __init__(self, rows):
        self.rows = rows                                   # list of 4 (n, 7) arrays

    def origin_scores(self, frame_h):
        return [r.copy() for r in self.rows]


def _toy_base(n, origin="current", t0=201912):
    """Minimal anchor slice the engine needs: `state` (origin) for one-hot init, plus the three
    fields `advance_frame` evolves. Every loan starts in `origin`."""
    return pl.DataFrame({
        "Loan Identifier": np.arange(n, dtype=np.int64),
        "state": [origin] * n,
        "Loan Age": np.full(n, 10, dtype=np.int64),
        "Remaining Months to Maturity": np.full(n, 350, dtype=np.int64),
        "period_ym": np.full(n, t0, dtype=np.int64),
    })


def test_roll_forward_predictor_snapshots_match_compose():
    """H ∈ {1,3,6,12} read off a SINGLE roll must each equal the closed-form ``compose(p0,[P]*h)``
    (a time-homogeneous stub chain), proving the snapshots are the genuine composed distributions."""
    rng = np.random.default_rng(7)
    n = 12
    rows = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    out = P.roll_forward_predictor(_StubPredictor(rows), _toy_base(n, "current"), 201912,
                                   horizon=12, snapshots=(1, 3, 6, 12),
                                   zero_impossible=False, chunk=n + 1)
    assert sorted(out["snapshots"]) == [1, 3, 6, 12]        # all four produced
    Pmat = P.assemble_matrix(rows)
    p0 = P.onehot_origin(np.zeros(n, dtype=int))
    for h in (1, 3, 6, 12):
        assert np.array_equal(out["snapshots"][h], P.compose(p0, [Pmat] * h)), h


def test_identity_calibrator_is_noop():
    """``calibrator=None`` and ``calibrator=identity_calibrator`` are bit-identical (§3.2 / the M23
    disabled-path regression check)."""
    rng = np.random.default_rng(8)
    n = 10
    rows = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    base = _toy_base(n)
    a = P.roll_forward_predictor(_StubPredictor(rows), base, 201912, horizon=6, snapshots=(1, 6),
                                 zero_impossible=False, chunk=n + 1)
    b = P.roll_forward_predictor(_StubPredictor(rows), base, 201912, horizon=6, snapshots=(1, 6),
                                 calibrator=P.identity_calibrator, zero_impossible=False, chunk=n + 1)
    for h in (1, 6):
        assert np.array_equal(a["snapshots"][h], b["snapshots"][h])


def test_calibrator_applied_to_raw_scores_before_assemble():
    """The calibrator acts on the RAW per-origin (n,7) scores BEFORE assembly (§4 item 2): an engine
    run with a calibrator equals building the matrices from the hand-calibrated rows."""
    rng = np.random.default_rng(9)
    n = 8
    rows = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    fc = P.SI["foreclosure"]
    cur_block = P.ORIGIN_STATES.index("current")

    def cal(probs, origin):                                # zero current→fc, renormalise
        if origin == cur_block:
            q = probs.copy()
            q[:, fc] = 0.0
            return q / q.sum(1, keepdims=True)
        return probs

    got = P.roll_forward_predictor(_StubPredictor(rows), _toy_base(n), 201912, horizon=3,
                                   snapshots=(3,), calibrator=cal, zero_impossible=False, chunk=n + 1)
    hand = [cal(r, o) for o, r in enumerate(rows)]
    p0 = P.onehot_origin(np.zeros(n, dtype=int))
    closed = P.compose(p0, [P.assemble_matrix(hand)] * 3)
    assert np.array_equal(got["snapshots"][3], closed)


def test_absorb_param_routes_through_first_passage():
    """``absorb=DPD60`` makes that state absorbing each step (the first-passage chain), equalling the
    manual ``absorb_rows`` + compose path."""
    rng = np.random.default_rng(10)
    n = 5
    rows = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    got = P.roll_forward_predictor(_StubPredictor(rows), _toy_base(n), 201912, horizon=4,
                                   snapshots=(4,), absorb=P.DPD60, zero_impossible=False, chunk=n + 1)
    Pmat = P.absorb_rows(P.assemble_matrix(rows), P.DPD60)
    p0 = P.onehot_origin(np.zeros(n, dtype=int))
    assert np.array_equal(got["snapshots"][4], P.compose(p0, [Pmat] * 4))


def test_empirical_predictor_stub_flows_through_engine():
    """The stub second Predictor (EmpiricalPredictor — feature-free) flows through the SAME engine
    math, proving the seam is model-agnostic (M21 Accept)."""
    rng = np.random.default_rng(11)
    n = 7
    emp = _softmax_rows(rng, 4)                            # 4 origins × 7
    got = P.roll_forward_predictor(P.EmpiricalPredictor(emp), _toy_base(n), 201912, horizon=12,
                                   snapshots=(1, 12), zero_impossible=False, chunk=n + 1)
    rows = [np.broadcast_to(emp[oi].astype(np.float32), (n, P.N_CLASSES)) for oi in range(4)]
    Pmat = P.assemble_matrix(rows)
    p0 = P.onehot_origin(np.zeros(n, dtype=int))
    for h in (1, 12):
        assert np.array_equal(got["snapshots"][h], P.compose(p0, [Pmat] * h)), h


def test_zero_impossible_flag_in_engine():
    """``zero_impossible=False`` preserves the raw ``current`` row (the M21 raw path the h=1 identity
    needs); ``True`` zeroes the two mechanically-impossible cells and renormalises (M20 / §3.5)."""
    rng = np.random.default_rng(12)
    n = 6
    fc, reo = P.SI["foreclosure"], P.SI["REO"]
    ci = P.ORIGIN_STATES.index("current")
    rows = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    rows[ci] = rows[ci].copy()
    rows[ci][:, [fc, reo]] = 1e-9                          # phantom impossible mass
    rows[ci] /= rows[ci].sum(1, keepdims=True)
    base = _toy_base(n, "current")
    raw = P.roll_forward_predictor(_StubPredictor(rows), base, 201912, horizon=1, snapshots=(1,),
                                   zero_impossible=False, chunk=n + 1)["snapshots"][1]
    zed = P.roll_forward_predictor(_StubPredictor(rows), base, 201912, horizon=1, snapshots=(1,),
                                   zero_impossible=True, chunk=n + 1)["snapshots"][1]
    assert np.array_equal(raw, rows[ci])                  # raw current row preserved exactly
    assert np.allclose(zed[:, [fc, reo]], 0.0)            # phantom removed
    assert np.allclose(zed.sum(1), 1.0)                   # still a distribution


# ===========================================================================
# M22 — loan-level pricing (ECONOMIC_ENGINE §3.4 / §4.3): price each loan as a one-loan "pool"
# through cashflow_engine, then aggregate. Hermetic: synthetic SMM paths, no SSD/GPU/models.
# ===========================================================================
def test_per_loan_smm_is_survivor_conditional():
    """`per_loan_smm` recovers the conditional monthly prepay rate from a roll-forward SMM capture:
    the unconditional increment Δ_h over the surviving (transient) mass at the start of the month,
    bounded in [0,1], with fully-terminated loans (alive=0) → SMM 0."""
    # cumulative prepaid mass through each month (monotone non-decreasing, ≤ 1)
    cum = np.array([[0.02, 0.05, 0.05, 0.05],          # prepays, then no survivors prepay further
                    [0.00, 0.10, 0.25, 0.40]])
    alive = np.array([[1.00, 0.50, 0.00, 0.00],        # loan 0 fully terminated after month 2
                      [1.00, 0.80, 0.60, 0.45]])
    smm = P.per_loan_smm(cum, alive)
    inc = np.array([[0.02, 0.03, 0.00, 0.00], [0.00, 0.10, 0.15, 0.15]])   # Δ_1=cum[:,0]
    expect = np.array([[0.02, 0.06, 0.0, 0.0],         # 0.03/0.50; then 0/0 → 0 (guarded)
                       [0.00, 0.125, 0.25, 0.15 / 0.45]])
    assert np.allclose(smm, expect, atol=1e-12)
    assert (smm >= 0).all() and (smm <= 1).all()


def _const_smm_pool(rng, n, wac, wam, lam, upb_lo=5e4, upb_hi=5e5):
    """A synthetic pool sharing (wac, wam) and a COMMON constant SMM path, differing only in UPB —
    the literal 'engine is linear in UPB' setup."""
    upb = rng.uniform(upb_lo, upb_hi, n)
    smm = np.full((n, 12), lam)
    return (np.full(n, wac), np.full(n, wam), upb, smm)


def test_loan_pool_aggregation_identity_linear_in_upb():
    """§4.3 headline: loan-level prices aggregated over a pool == the pool-level price. With a shared
    (wac, wam) and a common SMM path, the pool aggregate from `price_loans` must equal a single
    `cashflow_engine` run on (wac, wam, Σupb, smm) — and every per-loan per-100 price equals the pool
    price (scale-invariance: the engine is linear in UPB). Exact to ~1e-9, far inside the ~1e-6 gate."""
    rng = np.random.default_rng(22)
    wac, wam, upb, smm = _const_smm_pool(rng, 64, 0.06, 360, 0.015)
    res = P.price_loans(wac, wam, upb, smm)
    pool = P.cashflow_engine(0.06, 360, float(upb.sum()), smm[0])          # the pool-level engine run

    assert abs(res["pool_price"] - pool["price"]) < 1e-9, (res["pool_price"], pool["price"])
    assert abs(res["pool_value"] - pool["price"] / 100.0 * upb.sum()) < 1e-6   # $; ~1e-9 relative
    assert abs(res["pool_wal"] - pool["wal"]) < 1e-9
    assert np.allclose(res["price"], pool["price"], atol=1e-9)             # per-100 price UPB-invariant


def test_loan_pool_value_scale_invariance():
    """Linear-in-UPB, the other face: scaling every loan's UPB by a constant scales the pool dollar
    value by exactly that constant and leaves the per-100 pool price unchanged."""
    rng = np.random.default_rng(23)
    wac, wam, upb, smm = _const_smm_pool(rng, 40, 0.05, 360, 0.02)
    base = P.price_loans(wac, wam, upb, smm)
    scaled = P.price_loans(wac, wam, 10.0 * upb, smm)
    assert abs(scaled["pool_value"] - 10.0 * base["pool_value"]) < 1e-6 * base["pool_value"]
    assert abs(scaled["pool_price"] - base["pool_price"]) < 1e-9


def test_loan_pool_dollar_additivity_heterogeneous_smm():
    """§4.3 in full: 'pool cashflow = Σ loan cashflows by construction.' With a shared (wac, wam) but
    a DIFFERENT SMM path per loan, the pool dollar value must equal the discounted sum of per-loan
    dollar cashflows (one shared discount curve), and the pool-level engine on the **balance-$-weighted**
    aggregate SMM must reproduce the summed cashflow exactly — so its price·Σupb/100 == Σ loan values."""
    rng = np.random.default_rng(24)
    n, wac, wam, spread = 32, 0.055, 360, P.SERVICING_SPREAD
    upb = rng.uniform(5e4, 5e5, n)
    smm = rng.uniform(0.0, 0.05, (n, 12))                                  # heterogeneous paths
    res = P.price_loans(np.full(n, wac), np.full(n, wam), upb, smm)

    # Independent reference: run each loan's engine, sum dollar cashflows over the shared schedule,
    # discount once with the shared curve y = (wac − spread)/12.
    nmax = int(round(wam))
    cf_sum = np.zeros(nmax); bal_after_sched = np.zeros((n, nmax))
    for i in range(n):
        r = P.cashflow_engine(wac, wam, float(upb[i]), smm[i], servicing_spread=spread)
        cf_sum += r["cashflow"]
        bal_after_sched[i] = r["balance"] - r["sched_prin"]                # SMM acts on this balance
    y_m = (wac - spread) / 12.0
    disc = (1.0 + y_m) ** (-np.arange(1, nmax + 1))
    # cashflow is in $, so the pool dollar value is the discounted summed cashflow; price_loans'
    # pool_value (= Σ price_i·upb_i/100) must equal it (one shared discount curve).
    assert abs(res["pool_value"] - float((cf_sum * disc).sum())) < 1e-6 * res["pool_value"]

    # Pool-level engine on the balance-$-weighted aggregate SMM == summed cashflow, exactly.
    smm_h = np.array([smm[i, h] if h < 12 else smm[i, 11] for i in range(n) for h in range(nmax)]
                     ).reshape(n, nmax)
    num = (bal_after_sched * smm_h).sum(axis=0)
    den = bal_after_sched.sum(axis=0)
    pool_smm = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    poolr = P.cashflow_engine(wac, wam, float(upb.sum()), pool_smm, servicing_spread=spread)
    assert np.allclose(poolr["cashflow"], cf_sum, rtol=0, atol=1e-6)
    assert abs(poolr["price"] / 100.0 * upb.sum() - res["pool_value"]) < 1e-6 * res["pool_value"]


def test_price_loans_per_loan_path_closed_forms():
    """The §5 closed forms still hold when routed through the loan-level wrapper (a one-loan pool):
    zero-SMM ⇒ the annuity price, constant-SMM ⇒ the survival-schedule price, to ~1e-8 — `price_loans`
    delegates to `cashflow_engine` without re-deriving any cashflow math."""
    # zero-prepay annuity
    wac, wam, upb, spread = 0.05, 360, 100.0, 0.0025
    rz = P.price_loans(np.array([wac]), np.array([wam]), np.array([upb]),
                       np.zeros((1, 6)), servicing_spread=spread)
    c, y_m, n = wac / 12.0, (wac - spread) / 12.0, 360
    pmt = upb * c / (1.0 - (1.0 + c) ** (-n))
    price_ann = 100.0 * _annuity_pv(pmt, y_m, n) / upb
    assert abs(rz["price"][0] - price_ann) < 1e-8
    assert abs(rz["pool_price"] - price_ann) < 1e-8                       # one-loan pool == the loan

    # constant-SMM survival schedule
    wac, wam, upb, spread, lam = 0.045, 360, 500_000.0, 0.0025, 0.015
    rc = P.price_loans(np.array([wac]), np.array([wam]), np.array([upb]),
                       np.full((1, 12), lam), servicing_spread=spread)
    c, y_m, n = wac / 12.0, (wac - spread) / 12.0, 360
    h = np.arange(0, n + 1)
    phi = ((1.0 + c) ** n - (1.0 + c) ** h) / ((1.0 + c) ** n - 1.0)
    B = upb * phi * (1.0 - lam) ** h
    total_prin_ref = B[:-1] - B[1:]
    cf_ref = B[:-1] * c + total_prin_ref
    months = np.arange(1, n + 1)
    price_surv = 100.0 * float((cf_ref * (1.0 + y_m) ** (-months)).sum()) / upb
    assert abs(rc["price"][0] - price_surv) < 1e-8


def test_capture_smm_wires_to_price_loans_end_to_end():
    """End-to-end M22 wiring on a stub predictor + toy frame (no SSD/GPU): `roll_forward_predictor`
    captures cum-prepaid + alive mass INSIDE the existing loop (no re-roll), the capture matches the
    closed-form compose, `per_loan_smm` → `price_loans` prices the loans, and the absorb-guard fires."""
    rng = np.random.default_rng(25)
    n, H = 16, 12
    rows = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    base = _toy_base(n, "current")
    out = P.roll_forward_predictor(_StubPredictor(rows), base, 201912, horizon=H,
                                   snapshots=(1, H), zero_impossible=False, chunk=n + 1,
                                   capture_smm=True)
    assert set(out["smm"]) == {"cum_prepaid", "alive_before"}
    assert out["smm"]["cum_prepaid"].shape == (n, H)

    # capture matches the closed-form composed chain: cum_prepaid[:,h-1] = compose(p0,[P]*h)[:,PREPAID];
    # alive_before[:,h-1] = transient mass of compose(p0,[P]*(h-1)).
    Pmat = P.assemble_matrix(rows)
    p0 = P.onehot_origin(np.zeros(n, dtype=int))
    for h in range(1, H + 1):
        ph = P.compose(p0, [Pmat] * h)
        assert np.allclose(out["smm"]["cum_prepaid"][:, h - 1], ph[:, P.PREPAID], atol=1e-6)
        prev = P.compose(p0, [Pmat] * (h - 1))
        assert np.allclose(out["smm"]["alive_before"][:, h - 1],
                           prev[:, P.TRANSIENT_COLS].sum(axis=1), atol=1e-6)

    # the captured path drives price_loans through the conditional SMM helper.
    smm = P.per_loan_smm(out["smm"]["cum_prepaid"].astype(np.float64),
                         out["smm"]["alive_before"].astype(np.float64))
    assert (smm >= -1e-7).all() and (smm <= 1 + 1e-7).all()
    wac = np.full(n, 0.06); wam = np.full(n, 360.0); upb = rng.uniform(5e4, 5e5, n)
    res = P.price_loans(wac, wam, upb, smm)
    assert res["price"].shape == (n,) and np.isfinite(res["pool_price"])

    # absorb-guard: capture_smm records the normal-chain prepay increment, so absorb must be None.
    try:
        P.roll_forward_predictor(_StubPredictor(rows), base, 201912, horizon=2, snapshots=(2,),
                                 zero_impossible=False, chunk=n + 1, capture_smm=True, absorb=P.DPD60)
        raised = False
    except ValueError:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
