"""M24 — hermetic tests for the rolling pricing grid (no SSD / GPU / models).

Covers the new pieces: the vectorised loan-level engine (== scalar `price_loans`), the
horizon-truncation pricing semantics (H=12 == M15 regression guard), the per-H count grid and
realized within-H indicators, the loan-level reconciliation, and the resumable/atomic + tables-from-
completed-subset orchestration (via a redirected artifact dir)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from floan.model import pool as P
from floan.model import economics as EC
from floan.model import pricing_grid as G


# ===========================================================================
# Vectorised loan-level engine == scalar price_loans (the M24 scale guard)
# ===========================================================================
def test_price_loans_vec_matches_scalar_heterogeneous():
    """`price_loans_vec` reproduces the per-loan Python `price_loans` to ~1e-9 on a heterogeneous
    pool (varied wac/wam/upb AND per-loan SMM paths) — so M22's closed-form / identity guarantees
    carry over to the vectorised full-population path unchanged."""
    rng = np.random.default_rng(24)
    n = 300
    wac = rng.uniform(0.03, 0.085, n)
    wam = rng.integers(48, 360, n).astype(float)
    upb = rng.uniform(4e4, 7e5, n)
    smm = rng.uniform(0.0, 0.05, (n, 12))
    vec = P.price_loans_vec(wac, wam, upb, smm)
    sca = P.price_loans(wac, wam, upb, smm)
    assert np.allclose(vec["price"], sca["price"], atol=1e-7)
    assert np.allclose(vec["wal"], sca["wal"], atol=1e-9)
    assert np.allclose(vec["value"], sca["value"], rtol=1e-9, atol=1e-6)
    for key in ("pool_price", "pool_wal", "pool_value", "pool_upb"):
        assert abs(vec[key] - sca[key]) < 1e-6 * max(1.0, abs(sca[key])), key


def test_price_loans_vec_handles_zero_and_nan_upb():
    """Full-population robustness: null/zero-UPB loans (kept in the M14 pop at 0 weight) contribute 0
    value and are excluded from the pool price — no NaN contamination of the aggregate."""
    wac = np.array([0.06, 0.06, 0.06])
    wam = np.array([360.0, 360.0, np.nan])
    upb = np.array([1e5, 0.0, np.nan])
    smm = np.full((3, 12), 0.02)
    res = P.price_loans_vec(wac, wam, upb, smm)
    assert np.isfinite(res["pool_price"]) and np.isfinite(res["pool_value"])
    assert res["value"][1] == 0.0 and res["value"][2] == 0.0
    assert abs(res["pool_upb"] - 1e5) < 1e-6


def test_price_loans_vec_truncation_is_constant_extrapolation():
    """Horizon semantics: pricing the SMM path truncated at H == passing the full vector when the tail
    is already constant (the engine constant-extrapolates a short vector). So H<12 with a flat tail and
    the full 12-vector agree, and H=12 is the full path."""
    rng = np.random.default_rng(1)
    n = 64
    wac, wam, upb = np.full(n, 0.06), np.full(n, 360.0), rng.uniform(5e4, 5e5, n)
    smm = rng.uniform(0.0, 0.03, (n, 12))
    smm[:, 6:] = smm[:, 6:7]                      # flat from month 7 on
    trunc = P.price_loans_vec(wac, wam, upb, smm[:, :7])   # truncate-at-7 + extrapolate
    full = P.price_loans_vec(wac, wam, upb, smm)           # full vector (already flat past 7)
    assert np.allclose(trunc["price"], full["price"], atol=1e-9)


# ===========================================================================
# economics.per_pool_econ — H=12 == M15 (regression guard) and the H axis
# ===========================================================================
def _synthetic_smm_frame() -> pl.DataFrame:
    """A tiny smm_paths-style frame: 2 schemes × 2 pools × {logit, realized}, smm_h01..12."""
    rng = np.random.default_rng(7)
    rows = []
    for scheme in ("char", "random"):
        for pool in (0, 1):
            for model in ("logit", "realized"):
                smm = {f"smm_h{h:02d}": float(rng.uniform(0.0, 0.03)) for h in range(1, 13)}
                rows.append({"scheme": scheme, "pool": pool, "model": model, "n_loans": 1000,
                             "wac": 0.06, "wam": 360.0, "upb": 1e8, **smm})
    return pl.DataFrame(rows)


def test_per_pool_econ_h12_matches_m15_default():
    """`per_pool_econ(horizons=None)` (M15) and the H=12 slice of `per_pool_econ(horizons=(1,3,6,12))`
    price the same full path — the H=12==M15 regression guard."""
    df = _synthetic_smm_frame()
    m15 = EC.per_pool_econ(2020, df).sort(["scheme", "pool", "model"])
    grid = (EC.per_pool_econ(2020, df, horizons=(1, 3, 6, 12))
            .filter(pl.col("h") == 12).sort(["scheme", "pool", "model"]))
    for col in ("price", "wal", "cpr", "price_err", "wal_err", "cpr_err"):
        assert np.allclose(m15.get_column(col).to_numpy(),
                           grid.get_column(col).to_numpy(), atol=1e-10), col


def test_per_pool_econ_horizon_axis_present():
    """The H axis yields one priced row per (scheme,pool,model≠realized,h)."""
    df = _synthetic_smm_frame()
    out = EC.per_pool_econ(2020, df, horizons=(1, 3, 6, 12))
    assert set(out.get_column("h").unique().to_list()) == {1, 3, 6, 12}
    # 2 schemes × 2 pools × 1 non-realized model × 4 H = 16 rows
    assert out.height == 16


# ===========================================================================
# Realized within-H + count/loan grids on a synthetic per-loan table
# ===========================================================================
def test_within_h_is_cumulative_or():
    """Realized 'ever-by-month-H' = cumulative-OR over the first H per-step indicators."""
    steps = np.zeros((3, 12))
    steps[0, 4] = 1.0          # event at month 5
    steps[1, 0] = 1.0          # event at month 1
    assert G._within_h(steps, 1).tolist() == [0.0, 1.0, 0.0]
    assert G._within_h(steps, 6).tolist() == [1.0, 1.0, 0.0]


def _synthetic_tbl(n=1500, seed=3) -> dict:
    rng = np.random.default_rng(seed)
    df = pl.DataFrame({
        "Loan Identifier": np.arange(n),
        "fico": rng.uniform(620, 800, n), "rate": rng.uniform(3.0, 7.0, n),
        "ltv": rng.uniform(50, 97, n)})
    cum = np.sort(rng.uniform(0, 0.3, (n, 12)), axis=1)           # monotone cumulative prepaid
    cumd = np.sort(rng.uniform(0, 0.2, (n, 12)), axis=1)          # monotone cumulative 60+
    alive = np.clip(1.0 - cum - cumd, 0.0, 1.0)
    prepay_real = np.zeros((n, 12)); prepay_real[np.arange(n) % 5 == 0, 3] = 1.0
    dpd60_real = np.zeros((n, 12)); dpd60_real[np.arange(n) % 7 == 0, 5] = 1.0
    dpd90_real = np.zeros((n, 12)); dpd90_real[np.arange(n) % 11 == 0, 6] = 1.0
    return {"df": df, "model_names": ["empirical", "logit"],
            "smm": {m: {"cum": cum, "alive": alive, "cum_dpd60p": cumd} for m in ("empirical", "logit")},
            "prepay_real": prepay_real, "dpd60_real": dpd60_real, "dpd90_real": dpd90_real,
            "upb": rng.uniform(5e4, 5e5, n), "wac": rng.uniform(0.03, 0.08, n),
            "wam": rng.integers(60, 360, n).astype(float)}


def test_counts_grid_shape_and_horizons():
    tbl = _synthetic_tbl()
    cg = G._counts_grid(tbl, 2020)
    # 2 models × 2 outcomes × 4 H × 2 schemes = 32 rows
    assert cg.height == 2 * len(G.OUTCOMES) * len(G.HORIZONS) * len(G.SCHEMES)
    assert set(cg.get_column("h").unique().to_list()) == set(G.HORIZONS)
    assert set(cg.get_column("outcome").unique().to_list()) == set(G.OUTCOMES)
    assert np.isfinite(cg.get_column("rmse").to_numpy()).all()


def test_loan_grid_reconciliation_and_monotone_price():
    tbl = _synthetic_tbl()
    lg = G._loan_grid(tbl, 2020)
    assert lg.height == 2 * len(G.HORIZONS)
    assert np.isfinite(lg.get_column("pool_price").to_numpy()).all()
    # reconciliation gap (loan-level pool price vs pool-level single-pool price) is small
    assert lg.get_column("recon_gap").abs().max() < 2.0           # per-100; UPB-mean wac/wam ⇒ approx


# ===========================================================================
# Orchestration: tables-from-completed-subset + resumable skip (redirected dir)
# ===========================================================================
def _seed_completed_anchor(d, k, models=("logit", "ensemble", "nn", "empirical")):
    """Write a minimal econ_k{k}.parquet + counts_k{k}.parquet as if an anchor had completed."""
    rng = np.random.default_rng(k)
    erows, crows = [], []
    for h in G.HORIZONS:
        for scheme in G.SCHEMES:
            for pool in range(3):
                for m in models:
                    if m == "empirical":
                        continue
                    erows.append({"scheme": scheme, "pool": pool, "model": m, "n_loans": 1000,
                                  "wac": 0.06, "wam": 360.0, "upb": 1e8, "h": h,
                                  "cpr": 0.1, "wal": 5.0, "price": 100.0,
                                  "cpr_real": 0.1, "wal_real": 5.0, "price_real": 100.0,
                                  "cpr_err": 0.0, "wal_err": 0.0,
                                  "price_err": float(rng.uniform(-1, 1)),
                                  "k": k, "anchor": f"Dec{k-1}"})
            for outcome in G.OUTCOMES:
                for m in models:
                    crows.append({"anchor": f"Dec{k-1}", "k": k, "scheme": scheme,
                                  "outcome": outcome, "model": m, "h": h,
                                  "r2": 0.8, "rmse": 1.0, "n_pools": 3})
    pl.DataFrame(erows).write_parquet(d / f"econ_k{k}.parquet")
    pl.DataFrame(crows).write_parquet(d / f"counts_k{k}.parquet")


def test_tables_build_from_partial_subset(tmp_path, monkeypatch):
    """The table builders must work on whatever anchors have finished — here just 2 of 11."""
    monkeypatch.setattr(G, "_m24dir", lambda: tmp_path)
    _seed_completed_anchor(tmp_path, 2020)
    _seed_completed_anchor(tmp_path, 2023)
    t51 = G.build_T51()
    t42 = G.build_T42()
    hr = G.build_horizon_regime()
    assert t51["anchors"] == [2020, 2023]
    assert (tmp_path / "t_m24_econ_errors.md").exists()
    assert {r["k"] for r in hr["grid"]} == {2020, 2023}
    # COVID-inversion-in-dollars block isolates k2020
    assert all(r["k"] == 2020 for r in hr["covid_inversion_dollars"])
    assert t42["anchors"] == [2020, 2023]


def test_run_anchor_skips_completed(tmp_path, monkeypatch):
    """Resumability: a sealed .done marker makes run_anchor skip without touching the SSD/compute."""
    monkeypatch.setattr(G, "_m24dir", lambda: tmp_path)
    (tmp_path / "anchor_k2020.done").write_text("")
    out = G.run_anchor(2020, "cpu")                  # must NOT require_drive / roll
    assert out == {"k": 2020, "skipped": True}
