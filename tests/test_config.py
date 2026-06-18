"""Unit tests for config.py — the rolling expanding-window backtest splits.

Hermetic (pure integer arithmetic; no SSD, no torch). Run with:
    python -m pytest tests/test_config.py

Covers the §3.2 leakage rule that every test in this project ultimately rests on:
the train/val/test ``period_ym`` ranges must TILE the axis with no gap and no
overlap for every window ``k``, so a loan-month is never both trained and scored.
"""

from __future__ import annotations

from floan.model import config as cfg


# --- window bounds: tiling, no overlap, expanding train ---------------------
def test_window_bounds_tile_without_gap_or_overlap() -> None:
    """For every k, train|val|test are half-open [lo, hi) ranges that exactly tile
    [PANEL_PERIOD_MIN, _dec(k)) — contiguous, non-empty, and non-overlapping."""
    for k in cfg.TEST_YEARS:
        tr, va, te = cfg.train_bounds(k), cfg.val_bounds(k), cfg.test_bounds(k)
        # each range is non-empty (lo strictly below hi)
        for lo, hi in (tr, va, te):
            assert lo < hi, f"empty range {(lo, hi)} at k={k}"
        # contiguous: train.hi == val.lo and val.hi == test.lo (no gap, no overlap)
        assert tr[1] == va[0], f"train/val seam broken at k={k}: {tr} {va}"
        assert va[1] == te[0], f"val/test seam broken at k={k}: {va} {te}"
        # the three together span exactly [PANEL_PERIOD_MIN, _dec(k))
        assert tr[0] == cfg.PANEL_PERIOD_MIN
        assert te[1] == cfg._dec(k)


def test_bounds_land_on_the_right_decembers() -> None:
    """Splits are stated on label months; the masks bound the FEATURE month, so the
    seams sit on Dec(k-2), Dec(k-1), Dec(k) — the §3.2 boundaries verbatim."""
    for k in cfg.TEST_YEARS:
        assert cfg.train_bounds(k) == (cfg.PANEL_PERIOD_MIN, cfg._dec(k - 2))
        assert cfg.val_bounds(k) == (cfg._dec(k - 2), cfg._dec(k - 1))
        assert cfg.test_bounds(k) == (cfg._dec(k - 1), cfg._dec(k))


def test_val_and_test_are_one_year_wide() -> None:
    """val covers label-year k-1 and test covers label-year k; each spans exactly one
    calendar year (Dec→Dec is +100 in YYYYMM units)."""
    for k in cfg.TEST_YEARS:
        va, te = cfg.val_bounds(k), cfg.test_bounds(k)
        assert va[1] - va[0] == 100, f"val not one year wide at k={k}: {va}"
        assert te[1] - te[0] == 100, f"test not one year wide at k={k}: {te}"


def test_train_window_expands() -> None:
    """The train slice is an expanding window: a fixed lower bound and an upper bound
    that increases strictly with k."""
    los = {cfg.train_bounds(k)[0] for k in cfg.TEST_YEARS}
    assert los == {cfg.PANEL_PERIOD_MIN}, "train lower bound must be constant"
    his = [cfg.train_bounds(k)[1] for k in cfg.TEST_YEARS]
    assert his == sorted(his) and len(set(his)) == len(his), "train.hi must strictly increase"


# --- label_year: the +1-month label shift, with December rollover -----------
def test_label_year_handles_december_rollover() -> None:
    """A feature month's label is one calendar month ahead, so December rolls into the
    next year while every other month keeps its own year."""
    assert cfg.label_year(202012) == 2021    # Dec 2020 feature -> Jan 2021 label
    assert cfg.label_year(202011) == 2020
    assert cfg.label_year(202001) == 2020
    assert cfg.label_year(200012) == 2001    # panel-era boundary


def test_label_year_of_test_slice_equals_k() -> None:
    """Every feature month inside window k's test range carries a label in year k —
    the property that makes 'test year = k' meaningful."""
    for k in cfg.TEST_YEARS:
        lo, hi = cfg.test_bounds(k)          # [Dec(k-1), Dec(k))
        first_feature_month = lo             # Dec(k-1)  -> label Jan(k)
        last_feature_month = hi - 1          # Nov(k)    -> label Dec(k)
        assert cfg.label_year(first_feature_month) == k
        assert cfg.label_year(last_feature_month) == k


# --- sql_period_mask: half-open operators ----------------------------------
def test_sql_period_mask_is_half_open() -> None:
    """The SQL predicate is inclusive-low / exclusive-high, matching the [lo, hi)
    bounds so the upper seam belongs to the next split only."""
    assert (
        cfg.sql_period_mask("period_ym", (200001, 201512))
        == "period_ym >= 200001 AND period_ym < 201512"
    )


def test_sql_period_mask_uses_the_given_column() -> None:
    bounds = cfg.test_bounds(2015)
    sql = cfg.sql_period_mask("p.period_ym", bounds)
    assert sql == f"p.period_ym >= {bounds[0]} AND p.period_ym < {bounds[1]}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
