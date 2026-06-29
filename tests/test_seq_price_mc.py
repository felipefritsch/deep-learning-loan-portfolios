"""W3b (``ADR-002``) — hermetic tests for the Monte-Carlo H>1 sequence-pricing **driver**.

Synthetic only: a tiny untrained ``SeqGRU`` + a few synthetic loans (with a short observed
history), on CPU, no SSD / GPU / network. The SSD-touching parts of ``seq_price_mc`` are the data
pulls (``anchor_base`` / ``anchor_history`` / ``gru_eval_probs`` / ``_realized_steps``); the
*logic* lives in the two SSD-free cores — :func:`seq_price_mc.price_simulation` (the unchanged
``smm_paths`` → ``economics`` translation + the §M22 aggregation identity) and
:func:`seq_price_mc.h1_identity` (the H=1 Monte-Carlo identity guard) — which these tests exercise
directly with synthetic ``base`` / ``history`` / ``pop`` / ``realized``. The identities under test
hold for ANY model, trained or not (they are statements about a categorical sample converging to
its categorical, and about the engine's linearity in UPB), so an untrained net is a valid fixture.

The fixture builders are reused from ``test_seq_pricing`` (the W3 unit tests) so the synthetic
feature frame stays identical to the simulator's own tests.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import polars as pl
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # robust intra-tests import
import test_seq_pricing as TS                                        # noqa: E402

from floan.model import features as F          # noqa: E402
from floan.model import pool as PL             # noqa: E402
from floan.model import seq_model as SM        # noqa: E402
from floan.model import seq_pricing as MC      # noqa: E402
from floan.model import seq_price_mc as MX     # noqa: E402
from floan.model import seq_transformer as XF  # noqa: E402

K = 2020                  # config._dec(2019) == 201912 == t0 (the COVID anchor)
T0 = 201912
HORIZON = 12


def _base(n, *, seed, rng):
    """A tiny anchor ``base`` (one row per loan at t0), one loan per transient origin, cycled."""
    loans = [f"L{i}" for i in range(n)]
    states = [TS.ORIGIN_STATES[i % 4] for i in range(n)]
    upb = rng.uniform(8e4, 4e5, n)
    base = TS._rows(n, loans=loans, period_ym=T0, orig_ym=201001, age=24.0, rem=336.0,
                    rate=5.0, upb=upb, states=states, rng=rng)
    return F.prepare_raw(base), loans


def _history(loans, rng, *, n_months=4):
    """A short observed trailing history (period_ym < t0), long format — the shape
    :func:`seq_price_mc.anchor_history` produces (so the ``SeqPredictor(history=…)`` path the
    driver uses is exercised, not just the empty-history one)."""
    rows = [{"loan": loan, "ym": 201908 + m, "age": 20.0 + m,
             "state": TS.ORIGIN_STATES[(i + m) % 4]}
            for i, loan in enumerate(loans) for m in range(n_months)]   # 201908..201911 < t0
    return TS._rows(len(rows), loans=[r["loan"] for r in rows],
                    period_ym=[r["ym"] for r in rows], orig_ym=201001,
                    age=[r["age"] for r in rows], rem=340.0, rate=5.0, upb=2e5,
                    states=[r["state"] for r in rows], rng=rng)


def _fixture(n=6, *, seed=0):
    rng = np.random.default_rng(seed)
    scaler, vocab = TS._fit_pipeline(rng)
    torch.manual_seed(seed)
    model = SM.build(scaler, vocab, hidden=8)
    base, loans = _base(n, seed=seed, rng=rng)
    hist = _history(loans, rng)
    pred = MC.SeqPredictor(model, scaler, vocab, device="cpu", history=hist, T=TS.T_WIN)
    return pred, base


def _direct_one_step(pred, base):
    """The model's RAW direct one-step prediction per loan at its true origin — the hermetic stand-in
    for ``seq_predict.gru_eval_probs`` (which needs the SSD ``build_split``). Same window the MC scores
    at h=1 (history ++ the t0 base row with state=origin), so the MC mean must converge to it."""
    scores = np.stack(pred.origin_scores(PL.advance_frame(base, T0, 1)))   # [4, L, 7]
    oi = np.array([TS.ORIGIN_STATES.index(s) for s in base.get_column("state").to_list()])
    return scores[oi, np.arange(base.height)]                             # [L, 7]


# ===========================================================================
# H=1 identity guard — the integration check that the MC window == the direct prediction
# ===========================================================================
def test_h1_identity_guard_passes_and_discriminates():
    """:func:`seq_price_mc.h1_identity` PASSES when fed the matching direct one-step prediction (MC
    mean → direct to MC tolerance), and FAILS on a wrong reference — so it genuinely gates the
    history assembly rather than passing vacuously."""
    pred, base = _fixture(seed=1)
    sim = MC.simulate_paths(pred, base, T0, horizon=1, n_paths=8000, seed=0)
    direct = _direct_one_step(pred, base)
    origin_cls = np.array([F.STATE_INDEX[s] for s in base.get_column("state").to_list()])

    ok = MX.h1_identity(sim, direct, origin_cls)
    print(f"\n[H=1 guard] mean|Δ|={ok['mean_abs']:.4f} (tol {ok['tol_mean']:.4f})  "
          f"max|Δ|={ok['max_abs']:.4f} (tol {ok['tol_max']:.4f})")
    assert ok["pass"], ok

    # Negative control: a uniform (clearly wrong) reference must trip the gate.
    bad = MX.h1_identity(sim, np.full_like(direct, 1.0 / 7.0), origin_cls)
    assert not bad["pass"], bad


# ===========================================================================
# price_simulation — unchanged smm_paths → economics + the §M22 aggregation identity
# ===========================================================================
def _pop(base, rng):
    """A synthetic pricing pop row-aligned to ``base`` (same loan order): the bucketing covariates
    (fico/rate/ltv for ``char_cell_ids``) + engine weights (upb/wac/wam) taken from ``base``."""
    n = base.height
    return pl.DataFrame({
        "Loan Identifier": base.get_column("Loan Identifier").to_numpy(),
        "fico": rng.uniform(640.0, 800.0, n), "rate": rng.uniform(3.0, 7.0, n),
        "ltv": rng.uniform(50.0, 95.0, n),
        "upb": base.get_column("Current Actual UPB").to_numpy(),
        "wac": base.get_column("Current Interest Rate").to_numpy() / 100.0,
        "wam": base.get_column("Remaining Months to Maturity").to_numpy()})


def test_price_simulation_four_horizons_and_aggregation():
    """The driver's pure economic core: a full ``[N,12]`` MC simulation → the per-(scheme,pool) econ
    frame carries ALL FOUR horizons {1,3,6,12} (model column ``"gru"``, ``realized`` excluded), and
    the §M22 aggregation identity (Σ per-loan dollar value == pool value) holds to ~1e-6."""
    rng = np.random.default_rng(5)
    scaler, vocab = TS._fit_pipeline(rng)
    torch.manual_seed(5)
    model = SM.build(scaler, vocab, hidden=8)
    base, loans = _base(6, seed=5, rng=rng)
    pred = MC.SeqPredictor(model, scaler, vocab, device="cpu", history=_history(loans, rng), T=TS.T_WIN)
    sim = MC.simulate_paths(pred, base, T0, horizon=HORIZON, n_paths=600, seed=1)

    pop = _pop(base, rng)
    n = base.height
    # Synthetic realized per-step indicators (alive every step, no realized prepay) — only needs to
    # be a valid (alive, prepay, dpd60, dpd90) tuple; the test checks structure, not error values.
    realized = (np.ones((n, HORIZON)), np.zeros((n, HORIZON)),
                np.zeros((n, HORIZON)), np.zeros((n, HORIZON)))

    out = MX.price_simulation(sim, base, pop, K, realized, model_name="gru")
    econ = out["econ"]
    assert set(econ.get_column("h").to_list()) == {1, 3, 6, 12}
    assert set(econ.get_column("model").unique().to_list()) == {"gru"}      # realized is the reference, dropped
    assert out["aggregation"]["pass"], out["aggregation"]
    # Both pool schemes priced at every horizon (char merged cell + 500 random pools).
    assert set(econ.get_column("scheme").unique().to_list()) == {"char", "random"}
    print(f"\n[price_simulation] H={sorted(set(econ.get_column('h').to_list()))}  "
          f"rows={econ.height}  agg|Δ|={out['aggregation']['abs_diff']:.2e}  "
          f"pool_price(H12)={out['pool_price']:.3f}")


def _history_variable(loans, rng):
    """Histories of varying length (``0,1,…,T_WIN-1`` cycled across loans), so the **zero / partial /
    full-T** trailing-window regimes are all exercised. ``period_ym`` strictly < t0, period-ordered."""
    rows = [(loan, T0 - (i % TS.T_WIN) + m, 20.0 + m, TS.ORIGIN_STATES[(i + m) % 4])
            for i, loan in enumerate(loans) for m in range(i % TS.T_WIN)]
    if not rows:
        return None
    return TS._rows(len(rows), loans=[r[0] for r in rows], period_ym=[r[1] for r in rows],
                    orig_ym=201001, age=[r[2] for r in rows], rem=340.0, rate=5.0, upb=2e5,
                    states=[r[3] for r in rows], rng=rng)


# ===========================================================================
# Vectorized simulate_paths vs the reference oracle — the differential equivalence gate
# ===========================================================================
@pytest.mark.parametrize("arch", ["gru", "xf"])
def test_vectorized_matches_reference_variable_history(arch):
    """:func:`seq_pricing.simulate_paths` (vectorized production path) reproduces
    :func:`seq_pricing._simulate_paths_reference` (oracle) **bit-for-bit on CPU** — identical per-loan
    RNG streams + batch-invariant CPU forward — across variable history lengths (zero / partial /
    full-T) for **both** archs. Proves the vectorization is a pure performance refactor. (On GPU the
    batched forward's float-rounding gives Monte-Carlo-tolerance agreement instead — gated separately
    by ``seq_pricing._differential_check`` on real data.)"""
    rng = np.random.default_rng(11)
    scaler, vocab = TS._fit_pipeline(rng)
    torch.manual_seed(11)
    model = SM.build(scaler, vocab, hidden=8) if arch == "gru" else XF.build(scaler, vocab)
    n = 10
    loans = [f"L{i}" for i in range(n)]
    states = [TS.ORIGIN_STATES[i % 4] for i in range(n)]
    base = F.prepare_raw(TS._rows(n, loans=loans, period_ym=T0, orig_ym=201001, age=24.0, rem=336.0,
                                  rate=5.0, upb=rng.uniform(8e4, 4e5, n), states=states, rng=rng))
    Lhs = {i % TS.T_WIN for i in range(n)}
    assert 0 in Lhs and (TS.T_WIN - 1) in Lhs and any(0 < x < TS.T_WIN - 1 for x in Lhs)  # spans regimes
    pred = MC.SeqPredictor(model, scaler, vocab, device="cpu", history=_history_variable(loans, rng), T=TS.T_WIN)

    ref = MC._simulate_paths_reference(pred, base, T0, horizon=HORIZON, n_paths=1500, seed=4)
    vec = MC.simulate_paths(pred, base, T0, horizon=HORIZON, n_paths=1500, seed=4)
    for key in ("state_dist", "cum_prepaid", "alive_before", "smm"):
        d = float(np.abs(ref[key] - vec[key]).max())
        assert d == 0.0, f"{arch} {key} CPU max|Δ|={d} (must be bit-exact)"
    # multi-chunk path: a small loan_batch forces >1 chunk; per-loan RNG keys on the GLOBAL loan index
    # so chunk boundaries must not change the result.
    vec_chunked = MC.simulate_paths(pred, base, T0, horizon=HORIZON, n_paths=1500, seed=4, loan_batch=3)
    assert np.array_equal(vec["state_dist"], vec_chunked["state_dist"]), f"{arch}: chunking changed output"
    print(f"\n[vec vs ref · {arch}] bit-exact on CPU across zero/partial/full-T histories; chunk-invariant")


def test_simulate_paths_deterministic():
    """Same seed twice → identical output (the seeded per-loan sampler is reproducible)."""
    pred, base = _fixture(seed=9)
    a = MC.simulate_paths(pred, base, T0, horizon=HORIZON, n_paths=800, seed=2)
    b = MC.simulate_paths(pred, base, T0, horizon=HORIZON, n_paths=800, seed=2)
    for key in ("state_dist", "cum_prepaid", "alive_before", "smm"):
        assert np.array_equal(a[key], b[key]), key


def test_smm_tbl_uses_mc_masses():
    """The assembled ``tbl`` carries the MC cumulative-prepaid / alive-before masses verbatim (so the
    unchanged ``pool_smm_paths`` reads the simulation, not a re-derivation)."""
    pred, base = _fixture(seed=7)
    sim = MC.simulate_paths(pred, base, T0, horizon=HORIZON, n_paths=400, seed=2)
    n = base.height
    realized = (np.ones((n, HORIZON)),) + (np.zeros((n, HORIZON)),) * 3
    tbl = MX._smm_tbl(_pop(base, np.random.default_rng(0)), sim, realized, "xf")
    assert tbl["model_names"] == ["xf"]
    assert np.array_equal(tbl["smm"]["xf"]["cum"], sim["cum_prepaid"])
    assert np.array_equal(tbl["smm"]["xf"]["alive"], sim["alive_before"])
