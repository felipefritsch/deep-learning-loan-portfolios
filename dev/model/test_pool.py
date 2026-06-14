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

import pool as P


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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
