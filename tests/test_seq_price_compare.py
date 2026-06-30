"""W3b (``ADR-002``) — hermetic tests for the matched-comparator pricing **driver**.

Synthetic only (no SSD / GPU / network): the composable comparators reach the engine through the
checkpoint-free :class:`pool.EmpiricalPredictor` (a valid first-order ``Predictor`` — feature-free
row-stochastic matrices), so the composition+economics core, the H=1 identity guard, the
population/pool-alignment guards, the subsample determinism, and the matched-exhibit builder all
exercise on synthetic ``base`` / ``pop`` / ``realized`` / econ frames. The identities under test
(composed H=1 == direct one-step; same pop ⇒ same pools ⇒ same realized reference) hold for ANY
first-order predictor, so an arbitrary row-stochastic matrix is a valid fixture.

The synthetic feature frame reuses ``test_seq_pricing``'s builders (the W3 unit-test fixtures), so
``base`` is identical to the simulator's own tests.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # robust intra-tests import
import test_seq_pricing as TS                                        # noqa: E402

from floan.model import config                # noqa: E402
from floan.model import features as F          # noqa: E402
from floan.model import pool as PL             # noqa: E402
from floan.model import seq_price_compare as PC  # noqa: E402
from floan.model import seq_price_mc as MX      # noqa: E402

K = 2020                   # config._dec(2019) == 201912 == t0 (COVID anchor)
T0 = 201912
HORIZON = 12


# ---------------------------------------------------------------------------
# Fixtures — a tiny anchor base/pop and feature-free comparator predictors
# ---------------------------------------------------------------------------
def _base(n, *, rng):
    """A tiny anchor ``base`` (one row per loan at t0), one loan per transient origin, cycled."""
    loans = [f"L{i}" for i in range(n)]
    states = [TS.ORIGIN_STATES[i % 4] for i in range(n)]
    base = TS._rows(n, loans=loans, period_ym=T0, orig_ym=201001, age=24.0, rem=336.0,
                    rate=5.0, upb=rng.uniform(8e4, 4e5, n), states=states, rng=rng)
    return F.prepare_raw(base), loans


def _pop(base, rng):
    """A synthetic pricing pop row-aligned to ``base``: bucketing covariates (fico/rate/ltv for
    ``char_cell_ids``) + the engine weights (upb/wac/wam), drawn from ``base``."""
    n = base.height
    return pl.DataFrame({
        "Loan Identifier": base.get_column("Loan Identifier").to_numpy(),
        "fico": rng.uniform(640.0, 800.0, n), "rate": rng.uniform(3.0, 7.0, n),
        "ltv": rng.uniform(50.0, 95.0, n),
        "upb": base.get_column("Current Actual UPB").to_numpy(),
        "wac": base.get_column("Current Interest Rate").to_numpy() / 100.0,
        "wam": base.get_column("Remaining Months to Maturity").to_numpy()})


def _row_stochastic(rng, *, rows=4, cols=7):
    """A random 4×7 row-stochastic matrix — a valid feature-free empirical transition stub."""
    m = rng.uniform(0.05, 1.0, (rows, cols))
    return m / m.sum(axis=1, keepdims=True)


def _anchor_pop_like(n, *, seed):
    """A synthetic ``anchor_pop``-shaped frame (Loan Identifier + fico/rate/ltv) large enough that a
    stratified subsample actually thins it — for the subsample-determinism test."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "Loan Identifier": np.arange(n, dtype=np.int64),
        "fico": rng.uniform(600.0, 820.0, n), "rate": rng.uniform(2.5, 8.0, n),
        "ltv": rng.uniform(40.0, 100.0, n)}).sort("Loan Identifier")


# ===========================================================================
# Composition + economics core — multi-comparator econ frame at H∈{1,3,6,12}
# ===========================================================================
def test_compose_and_price_multimodel():
    """:func:`compose_comparator` (the unchanged ``roll_forward_predictor`` SMM capture) feeds
    :func:`price_compare` (the unchanged ``smm_paths`` → ``economics`` grid): the matched econ frame
    carries ALL FOUR horizons, BOTH schemes, one row-set per priced comparator, with the realized
    model excluded (it is the error reference) and finite price errors."""
    rng = np.random.default_rng(3)
    base, _ = _base(6, rng=rng)
    pop = _pop(base, rng)
    n = base.height
    realized = (np.ones((n, HORIZON)), np.zeros((n, HORIZON)),
                np.zeros((n, HORIZON)), np.zeros((n, HORIZON)))

    preds = {"empirical": PL.EmpiricalPredictor(_row_stochastic(rng)),
             "logit": PL.EmpiricalPredictor(_row_stochastic(rng))}     # two distinct first-order stubs
    smm = {name: PC.compose_comparator(p, base, T0) for name, p in preds.items()}
    for name in smm:
        assert smm[name]["cum"].shape == (n, HORIZON)
        assert smm[name]["alive"].shape == (n, HORIZON)

    econ = PC.price_compare(pop, smm, K, realized)
    assert set(econ.get_column("h").to_list()) == {1, 3, 6, 12}
    assert set(econ.get_column("model").unique().to_list()) == {"empirical", "logit"}  # realized dropped
    assert set(econ.get_column("scheme").unique().to_list()) == {"char", "random"}
    assert np.isfinite(econ.get_column("price_err").to_numpy()).all()
    print(f"\n[price_compare] models={sorted(set(econ.get_column('model').to_list()))} "
          f"H={sorted(set(econ.get_column('h').to_list()))} rows={econ.height}")


# ===========================================================================
# Guard — H=1 composition reproduces the comparator's direct one-step prediction
# ===========================================================================
def test_h1_identity_holds_for_first_order_predictor():
    """:func:`h1_identity` PASSES (max|Δ| ~ 0) for a first-order predictor: composing one step
    (``p0·M``) equals the predictor's direct per-origin row gathered at each loan's true origin —
    the regression tie-back the real comparators must satisfy on the matched pop."""
    rng = np.random.default_rng(8)
    base, _ = _base(8, rng=rng)
    pred = PL.EmpiricalPredictor(_row_stochastic(rng))
    rec = PC.h1_identity(pred, base, T0)
    print(f"\n[h1_identity] max|Δ|={rec['max_abs']:.2e} (tol {rec['tol']:g})")
    assert rec["pass"], rec
    assert rec["max_abs"] <= 1e-6                                  # exact for a feature-free stub


# ===========================================================================
# Guard — population: regenerated subsample matches the sequence arm's recorded size/seed
# ===========================================================================
def test_assert_population_matches_and_discriminates(tmp_path, monkeypatch):
    monkeypatch.setattr(PC, "MC_OUT", tmp_path)
    pop = _anchor_pop_like(400, seed=0)
    # Sequence arm recorded actual_n == pop.height at seed 0 / target 150000.
    (tmp_path / f"k{K}_gru_h.json").write_text(json.dumps(
        {"subsample": {"target_n": 150000, "actual_n": pop.height, "full_pop_n": 999999,
                       "seed": 0, "stratify": "char_cell_ids(FICO×rate×LTV)"}}))
    rec = PC.assert_population(K, pop, seed=0, target_n=150000)
    assert rec["pass"] and rec["actual_n"] == pop.height

    # A size mismatch (sequence arm recorded a different actual_n) must raise.
    (tmp_path / f"k{K}_gru_h.json").write_text(json.dumps(
        {"subsample": {"target_n": 150000, "actual_n": pop.height + 1, "full_pop_n": 999999,
                       "seed": 0, "stratify": "char_cell_ids(FICO×rate×LTV)"}}))
    with pytest.raises(AssertionError):
        PC.assert_population(K, pop, seed=0, target_n=150000)
    # A seed/target mismatch must raise too.
    with pytest.raises(AssertionError):
        PC.assert_population(K, pop, seed=1, target_n=150000)


# ===========================================================================
# Guard — pool alignment: same pop ⇒ byte-identical pools + realized reference vs the GRU econ
# ===========================================================================
def _ref_econ(rng, *, n_pools=5, perturb=None):
    """A synthetic committed ``k{k}_gru_econ``-shaped frame (one row per scheme×pool×h)."""
    rows = []
    for scheme in ("char", "random"):
        for pool in range(n_pools):
            for h in (1, 3, 6, 12):
                rows.append({"scheme": scheme, "pool": pool, "h": h, "k": K,
                             "n_loans": 100 + pool, "wac": 0.04 + pool * 1e-3,
                             "wam": 320.0 + pool, "upb": 1e6 * (pool + 1),
                             "cpr_real": 0.05 + pool * 1e-3, "wal_real": 6.0 + pool * 0.01,
                             "price_real": 99.0 + pool * 0.1,
                             "model": "gru", "price_err": rng.normal()})
    df = pl.DataFrame(rows)
    if perturb:
        df = perturb(df)
    return df


def test_assert_pool_alignment_passes_and_discriminates(tmp_path, monkeypatch):
    monkeypatch.setattr(PC, "MC_OUT", tmp_path)
    rng = np.random.default_rng(1)
    ref = _ref_econ(rng)
    ref.write_parquet(tmp_path / f"k{K}_gru_econ.parquet")

    # "mine": identical pools + realized reference (different model + price_err is irrelevant).
    mine = ref.with_columns(pl.lit("logit").alias("model"),
                            pl.Series("price_err", rng.normal(size=ref.height)))
    rec = PC.assert_pool_alignment(mine, K)
    assert rec["pass"] and rec["n_loans_exact"], rec

    # n_loans mismatch ⇒ raise (population not identical).
    bad_n = mine.with_columns((pl.col("n_loans") + 1).alias("n_loans"))
    with pytest.raises(AssertionError):
        PC.assert_pool_alignment(bad_n, K)
    # realized-valuation drift beyond tol ⇒ raise.
    bad_real = mine.with_columns((pl.col("price_real") + 1e-3).alias("price_real"))
    with pytest.raises(AssertionError):
        PC.assert_pool_alignment(bad_real, K)
    # a missing pool ⇒ raise (different pool set).
    fewer = mine.filter(pl.col("pool") != 0)
    with pytest.raises(AssertionError):
        PC.assert_pool_alignment(fewer, K)


# ===========================================================================
# Subsample — deterministic, seeded, size-capped (the population the arms share)
# ===========================================================================
def test_subsample_pop_deterministic_and_capped():
    """The matched population is reproducible: :func:`seq_price_mc.subsample_pop` returns the SAME
    loan ids across calls for a fixed ``(seed, k)``, caps to ~target, and keeps every char cell
    populated (≥1 loan) so no pool drops out."""
    pop = _anchor_pop_like(6000, seed=0)
    a = MX.subsample_pop(pop, K, target_n=1500, seed=0)
    b = MX.subsample_pop(pop, K, target_n=1500, seed=0)
    assert a.get_column("Loan Identifier").to_list() == b.get_column("Loan Identifier").to_list()
    assert a.height <= pop.height and 0 < a.height <= int(1.5 * 1500)   # ~target, never the whole pop
    assert a.get_column("Loan Identifier").to_list() == sorted(a.get_column("Loan Identifier").to_list())
    # a different seed gives a different draw (overwhelmingly likely at this size)
    c = MX.subsample_pop(pop, K, target_n=1500, seed=1)
    assert c.get_column("Loan Identifier").to_list() != a.get_column("Loan Identifier").to_list()


# ===========================================================================
# Matched exhibit — comparators beside the GRU and transformer
# ===========================================================================
def test_build_matched_table(tmp_path, monkeypatch):
    monkeypatch.setattr(PC, "MC_OUT", tmp_path)
    rng = np.random.default_rng(2)
    # Committed sequence arms (gru + xf) on the same pools.
    for arm in ("gru", "xf"):
        _ref_econ(rng).with_columns(pl.lit(arm).alias("model")).write_parquet(
            tmp_path / f"k{K}_{arm}_econ.parquet")
    # This run's comparator econ (two comparators).
    comp = pl.concat([_ref_econ(rng).with_columns(pl.lit(m).alias("model"))
                      for m in ("empirical", "logit")], how="vertical_relaxed")

    out = PC.build_matched_table(K, comp)
    assert (tmp_path / f"k{K}_matched.md").exists()
    assert (tmp_path / f"k{K}_matched.parquet").exists()
    models = set(out["models"])
    assert {"empirical", "logit", "gru", "xf"} <= models           # comparators beside both seq arms
    md = (tmp_path / f"k{K}_matched.md").read_text()
    assert "GRU (MC)" in md and "Transformer (MC)" in md         # both sequence arms as columns
    assert "Logit" in md and "Empirical" in md                   # the comparator labels
    print(f"\n[matched] models={sorted(models)}\n{md}")
