"""Unit tests for mkt_rate.py — the gap-free month grid for the rate proxy.

Hermetic (pure; no SSD, no torch). Run with:
    python -m pytest tests/test_mkt_rate.py

The forward/backward fill of thin months in ``build`` is integration-level (it needs
the panel via DuckDB); the pure, separable piece is ``_month_grid``, which must emit a
contiguous YYYYMM sequence with correct year rollover so no month is skipped.
"""

from __future__ import annotations

from floan.analysis import mkt_rate as mk


def test_month_grid_is_contiguous_within_a_year() -> None:
    assert mk._month_grid(202001, 202004) == [202001, 202002, 202003, 202004]


def test_month_grid_rolls_over_the_year() -> None:
    assert mk._month_grid(201911, 202002) == [201911, 201912, 202001, 202002]


def test_month_grid_single_month_and_inclusive_bounds() -> None:
    assert mk._month_grid(200506, 200506) == [200506]
    grid = mk._month_grid(200001, 200512)
    assert grid[0] == 200001 and grid[-1] == 200512 and len(grid) == 6 * 12


def test_month_grid_has_no_gaps() -> None:
    grid = mk._month_grid(201810, 202103)
    for a, b in zip(grid, grid[1:]):
        ay, am = divmod(a, 100)
        nxt = a + 1 if am < 12 else (ay + 1) * 100 + 1
        assert b == nxt


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
