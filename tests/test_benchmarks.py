"""Unit tests for benchmarks.py — the weighted-tercile edge helper (M5).

Hermetic (pure numpy; no SSD, no torch). Run with:
    python -m pytest tests/test_benchmarks.py

``_weighted_terciles`` defines the empirical benchmark's cell boundaries, so it must
place the 1/3 and 2/3 cut points by cumulative WEIGHT (not count) and be deterministic.
"""

from __future__ import annotations

import numpy as np

from floan.model import benchmarks as B


def test_weighted_terciles_uniform_weights() -> None:
    v = np.arange(1, 10, dtype=float)            # 1..9
    w = np.ones(9)
    e1, e2 = B._weighted_terciles(v, w)
    assert (e1, e2) == (3.0, 6.0)                 # lower-weighted 1/3 and 2/3 quantiles


def test_weighted_terciles_respect_weight_mass() -> None:
    # A heavy tail at value 30 pulls both cut points onto it.
    v = np.array([10.0, 20.0, 30.0])
    w = np.array([1.0, 1.0, 10.0])
    assert B._weighted_terciles(v, w) == (30.0, 30.0)


def test_weighted_terciles_is_order_independent() -> None:
    v = np.array([3.0, 1.0, 2.0])
    w = np.array([1.0, 1.0, 1.0])
    assert B._weighted_terciles(v, w) == (1.0, 2.0)  # stable sort -> deterministic


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
