"""M14 unit tests — the pure pool-construction / count / metric logic in ``pools.py``,
exercised on a synthetic per-loan table with no SSD, GPU, or model artifacts.

Covers: characteristic bucketing is an exhaustive partition with thin-cell merge; random-pool
membership is seeded-reproducible and within-pool replacement-free; predicted counts and the
Poisson-binomial normal interval match Σp / Σp(1−p); realized counts reconcile exactly between
the characteristic partition and a direct total; R²/RMSE match closed form.

Run:  .venv/bin/python dev/model/test_pools.py     (or python -m pytest)
"""

from __future__ import annotations

import numpy as np
import polars as pl

import pools as M


def _synthetic(n=20000, seed=7) -> pl.DataFrame:
    """A per-loan table with the columns ``pools`` consumes: bucket covariates, two outcomes'
    realized flags, and per-model predicted probabilities for empirical/logit/ensemble."""
    rng = np.random.default_rng(seed)
    fico = rng.integers(620, 820, n).astype(np.float64)
    rate = rng.uniform(2.5, 6.0, n)
    ltv = rng.uniform(40, 100, n)
    cols = {"Loan Identifier": [f"L{i:07d}" for i in range(n)],
            "fico": fico, "rate": rate, "ltv": ltv}
    # realized flags ~ Bernoulli; predicted probs near the realized rate (+ model noise).
    p_pre = np.clip(0.15 + 0.001 * (760 - fico), 0.01, 0.6)
    p_60 = np.clip(0.02 + 0.0005 * (700 - fico), 0.001, 0.3)
    cols["realized_prepaid"] = (rng.random(n) < p_pre).astype(np.int64)
    cols["realized_dpd60p"] = (rng.random(n) < p_60).astype(np.int64)
    for m in ("empirical", "logit", "ensemble"):
        jitter = {"empirical": 0.03, "logit": 0.01, "ensemble": 0.005}[m]
        cols[f"{m}_prepaid_12m"] = np.clip(p_pre + rng.normal(0, jitter, n), 0, 1)
        cols[f"{m}_dpd60p_12m"] = np.clip(p_60 + rng.normal(0, jitter, n), 0, 1)
    return pl.DataFrame(cols)


# ===========================================================================
# Characteristic buckets — exhaustive partition + thin merge
# ===========================================================================
def test_char_cells_partition_and_merge():
    df = _synthetic()
    cell, info = M.char_cell_ids(df)
    assert cell.shape[0] == df.height
    # exhaustive: every loan assigned a cell; no cell smaller than MIN_CELL except the
    # catch-all (id MERGED), which absorbs all thin cells.
    uniq, counts = np.unique(cell, return_counts=True)
    for v, c in zip(uniq, counts):
        if int(v) != info["merged_id"]:
            assert c >= M.MIN_CELL, (v, c)
    assert info["n_cells"] == uniq.shape[0]
    # merged-cell size equals the reported figure (or 0 cells were thin → no MERGED id).
    if info["n_thin_merged"] > 0:
        assert info["merged_id"] in uniq
        assert counts[uniq == info["merged_id"]][0] == info["n_in_merged"]


def test_char_realized_reconciles_with_total():
    """Σ realized over the characteristic partition == the direct population total — the
    Accept #2 reconciliation, exercised on the pure aggregation path."""
    df = _synthetic()
    cell, _ = M.char_cell_ids(df)
    for outcome in M.OUTCOMES:
        cc = M.pool_counts_char(df, cell, "logit", outcome)
        direct = float(M._real(df, outcome).sum())
        assert cc["realized"].sum() == direct
        # cell loan counts also partition the population.
        assert int(cc["n"].sum()) == df.height


# ===========================================================================
# Random pools — seeded reproducibility, within-pool replacement-free
# ===========================================================================
def test_random_pool_shape_and_no_replacement():
    idx = M.random_pool_index(20000, k=2020, B=50, N=1000, seed=0)
    assert idx.shape == (50, 1000)
    assert idx.min() >= 0 and idx.max() < 20000
    for row in idx:
        assert np.unique(row).shape[0] == row.shape[0]   # no replacement within a pool


def test_random_pool_reproducible_and_anchor_distinct():
    a = M.random_pool_index(20000, k=2020, B=50, N=1000, seed=0)
    b = M.random_pool_index(20000, k=2020, B=50, N=1000, seed=0)
    assert np.array_equal(a, b)                            # same (seed, k) → identical
    c = M.random_pool_index(20000, k=2023, B=50, N=1000, seed=0)
    assert not np.array_equal(a, c)                        # different anchor → different pools


# ===========================================================================
# Counts + intervals — Σp / Σp(1−p) / realized
# ===========================================================================
def test_random_counts_match_manual_sum():
    df = _synthetic()
    idx = M.random_pool_index(df.height, k=2020, B=40, N=500, seed=1)
    c = M.pool_counts_random(df, idx, "ensemble", "prepaid")
    p = df.get_column("ensemble_prepaid_12m").to_numpy()
    r = df.get_column("realized_prepaid").to_numpy().astype(float)
    assert np.allclose(c["pred"], p[idx].sum(axis=1), atol=1e-9)
    assert np.allclose(c["sd"], np.sqrt((p[idx] * (1 - p[idx])).sum(axis=1)), atol=1e-9)
    assert np.allclose(c["realized"], r[idx].sum(axis=1), atol=1e-9)
    # interval is the symmetric Normal 90% band.
    assert np.allclose(c["lo"], c["pred"] - M.Z90 * c["sd"], atol=1e-12)
    assert np.allclose(c["hi"], c["pred"] + M.Z90 * c["sd"], atol=1e-12)


def test_char_counts_match_groupby():
    df = _synthetic()
    cell, _ = M.char_cell_ids(df)
    c = M.pool_counts_char(df, cell, "logit", "dpd60p")
    p = df.get_column("logit_dpd60p_12m").to_numpy()
    for i, cid in enumerate(c["cell"]):
        m = cell == cid
        assert abs(c["pred"][i] - p[m].sum()) < 1e-7
        assert c["n"][i] == int(m.sum())


# ===========================================================================
# R² / RMSE closed form
# ===========================================================================
def test_r2_rmse_closed_form():
    realized = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([12.0, 18.0, 33.0, 39.0])
    r2, rmse = M._r2_rmse(pred, realized)
    resid = pred - realized
    assert abs(rmse - np.sqrt(np.mean(resid ** 2))) < 1e-12
    ss_tot = np.sum((realized - realized.mean()) ** 2)
    assert abs(r2 - (1 - np.sum(resid ** 2) / ss_tot)) < 1e-12
    # perfect prediction → R²=1, RMSE=0.
    r2p, rmsep = M._r2_rmse(realized.copy(), realized)
    assert abs(r2p - 1.0) < 1e-12 and rmsep == 0.0


def test_coverage_fraction():
    # construct counts where exactly half the pools' realized fall inside the band.
    c = {"realized": np.array([1.0, 1.0, 100.0, 100.0]),
         "lo": np.array([0.0, 0.0, 0.0, 0.0]),
         "hi": np.array([2.0, 2.0, 2.0, 2.0])}
    assert abs(M._coverage(c) - 0.5) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
