"""Unit tests for eda_common.py — the shared EDA stat helpers.

Hermetic (pure numpy/polars; no SSD, no torch). Run with:
    python -m pytest tests/test_eda_common.py

These three helpers feed every Phase-1 hazard figure, so a bug here propagates
widely: the Wilson interval, the equal-population bucket edges, and the
fine-histogram -> bucketed-rate collapse.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from floan.analysis import eda_common as eda


# --- wilson_ci -------------------------------------------------------------
def test_wilson_ci_bounds_and_ordering() -> None:
    lo, hi = eda.wilson_ci([0, 50, 100], [100, 100, 100])
    assert lo.shape == hi.shape == (3,)
    assert np.all(lo <= hi)
    assert np.all((lo >= 0.0) & (hi <= 1.0))         # always a valid proportion interval
    assert lo[0] == 0.0                               # k=0 -> lower bound pinned at 0
    assert lo[1] < 0.5 < hi[1]                        # k/n=0.5 straddles the centre
    assert hi[2] > 0.95                               # k=n -> upper bound near 1


def test_wilson_ci_handles_zero_n_gracefully() -> None:
    lo, hi = eda.wilson_ci([0], [0])                  # must not raise (n==0)
    assert lo.shape == (1,) and hi.shape == (1,)


# --- equal_pop_edges -------------------------------------------------------
def test_equal_pop_edges_split_uniform_mass() -> None:
    edges = eda.equal_pop_edges([1, 2, 3, 4], [1, 1, 1, 1], n=2)
    assert edges == [2.0]                              # one interior cut at the median
    assert eda.equal_pop_edges([1, 2, 3, 4], [1, 1, 1, 1], n=4) == [1.0, 2.0, 3.0]


def test_equal_pop_edges_collapses_ties() -> None:
    # A near-degenerate covariate yields FEWER than n-1 edges (ties collapse).
    edges = eda.equal_pop_edges([1, 1, 1, 2], [1, 1, 1, 1], n=3)
    assert edges == [1.0]
    assert all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))  # strictly increasing


# --- hazard_curve ----------------------------------------------------------
def test_hazard_curve_buckets_and_rates() -> None:
    fine = pl.DataFrame({"v": [1, 2, 3, 4], "num": [1, 2, 3, 4], "den": [10, 10, 10, 10]})
    out = eda.hazard_curve(fine, value_col="v", num_col="num", den_col="den", n=2)
    assert out.height == 2
    assert out["value"].to_list() == [1.5, 3.5]        # den-weighted mean covariate per bucket
    assert out["rate"].to_list() == [0.15, 0.35]       # num/den per bucket
    lo, rate, hi = out["lo"].to_numpy(), out["rate"].to_numpy(), out["hi"].to_numpy()
    assert np.all(lo <= rate) and np.all(rate <= hi)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
