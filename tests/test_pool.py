"""M13 unit tests — the roll-forward composition engine (``pool.py``): the pure-NumPy Layer-1
functions, exercised with no SSD, no GPU, and no model artifacts. Covers the Accept criterion
"toy-chain unit test matches closed form":

  * homogeneous chain composed H steps == ``p0 · P^H`` (matrix-power closed form);
  * a hand-computed two-step example;
  * ``assemble_matrix`` gives stochastic rows (terminal states absorbing) and conserves
    probability mass through composition;
  * the first-passage variant (``absorb_rows`` + compose) equals an independent
    brute-force path enumeration of "ever reaches the target within H steps".

Run:  .venv/bin/python dev/model/test_pool.py     (or python -m pytest)
"""

from __future__ import annotations

import numpy as np

try:  # pool.py imports torch at module load; skip cleanly when torch is absent
    import torch  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("torch not installed", allow_module_level=True)

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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
