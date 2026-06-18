"""Unit tests for macro_features.py — the §4 lag convention, join + derived features.

Hermetic (synthetic tables only; no SSD). Run with:
    .venv/bin/python -m floan.model.test_macro_features   (or python -m pytest)
"""

from __future__ import annotations

import polars as pl

from floan.model import macro_features as mf


# --- helpers ---------------------------------------------------------------
def _month_range(a: int, b: int) -> list[int]:
    out, y, m = [], a // 100, a % 100
    while y * 100 + m <= b:
        out.append(y * 100 + m)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _nat(months, pmms, unrate, dgs10=2.0, slope=0.5) -> pl.DataFrame:
    """macro_national where pmms30/unrate_nat are callables of month_ym (else const)."""
    f = lambda v, m: float(v(m)) if callable(v) else float(v)
    return pl.DataFrame({
        "month_ym": months,
        "pmms30": [f(pmms, m) for m in months],
        "dgs10": [f(dgs10, m) for m in months],
        "slope_10y2y": [f(slope, m) for m in months],
        "unrate_nat": [f(unrate, m) for m in months],
    }).with_columns(pl.col("month_ym").cast(pl.Int32))


def _state(months, table) -> pl.DataFrame:
    """macro_state from {state: {"unrate": v_or_fn, "hpi": v_or_fn}} over `months`."""
    f = lambda v, m: float(v(m)) if callable(v) else float(v)
    rows = []
    for st, spec in table.items():
        for m in months:
            rows.append((st, m, f(spec["unrate"], m), f(spec["hpi"], m)))
    return pl.DataFrame(rows, schema=["state", "month_ym", "unrate_state", "hpi_state"],
                        orient="row").with_columns(pl.col("month_ym").cast(pl.Int32))


def _loan(**kw) -> pl.DataFrame:
    base = dict(period_ym=202001, orig_ym=201501, property_state="CA",
                current_rate=6.0, current_upb=180000.0, orig_upb=200000.0, orig_ltv=80.0)
    base.update(kw)
    return pl.DataFrame([base]).with_columns(
        pl.col("period_ym").cast(pl.Int32), pl.col("orig_ym").cast(pl.Int32))


# --- tests -----------------------------------------------------------------
def test_sub_months():
    assert mf.sub_months(202001, 0) == 202001
    assert mf.sub_months(202001, 1) == 201912   # year rollover
    assert mf.sub_months(202001, 2) == 201911
    assert mf.sub_months(202001, 14) == 201811  # 12m + 2m hpi lag
    assert mf.sub_months(202003, 2) == 202001
    assert mf.sub_months(200001, 1) == 199912    # macro-table edge


def test_lag_correctness():
    """period_ym=202001 → unrate from 2019-12 (lag 1), hpi from 2019-11 (lag 2)."""
    months = _month_range(201801, 202012)
    # values encode the month so the selected month is provable.
    nat = _nat(months, pmms=lambda m: m, unrate=lambda m: m)
    state = _state(months, {
        "CA": {"unrate": lambda m: m, "hpi": lambda m: m},
        "US": {"unrate": lambda m: m + 0.5, "hpi": lambda m: m + 0.5},  # must NOT be used
    })
    out = mf.join_macro(_loan(period_ym=202001, orig_ym=201901), nat, state).row(0, named=True)

    assert out["pmms30"] == 202001.0            # lag 0
    assert out["unrate_nat"] == 201912.0        # lag 1
    assert out["unrate_state"] == 201912.0      # lag 1, CA (not US)
    assert out["hpi_state"] == 201911.0         # lag 2  → "hpi of 2019-11"
    assert out["hpi_orig"] == 201811.0          # orig_ym 201901 − 2
    assert out["hpi_12m"] == 201811.0           # period 202001 − 14
    assert out["unrate_fallback"] == 0 and out["hpi_fallback"] == 0


def test_ltv_mtm_and_derived_exact():
    """Known UPB/LTV/HPI path → exact ltv_mtm, incentive, hpi_chg_12m."""
    months = _month_range(201401, 202012)
    # HPI(orig=201501→201411)=100, HPI(12m=201811)=110, HPI(period=201911)=125.
    hpi = {201411: 100.0, 201811: 110.0, 201911: 125.0}
    nat = _nat(months, pmms=lambda m: 4.0 if m == 202001 else 9.0, unrate=5.0)
    state = _state(months, {"CA": {"unrate": 5.0,
                                   "hpi": lambda m: hpi.get(m, 1.0)}})
    out = mf.attach_macro(
        _loan(period_ym=202001, orig_ym=201501, current_rate=6.0,
              current_upb=180000.0, orig_upb=200000.0, orig_ltv=80.0),
        nat, state).row(0, named=True)

    assert abs(out["incentive"] - 2.0) < 1e-9                    # 6.0 − 4.0
    assert abs(out["ltv_mtm"] - 57.6) < 1e-9                     # 0.9·80·(100/125)
    assert abs(out["hpi_chg_12m"] - (125 / 110 - 1)) < 1e-12     # 125/110 − 1


def test_fallback_to_national_for_territory():
    """A territory state absent from macro_state uses the US row; flags set to 1."""
    months = _month_range(201401, 202012)
    nat = _nat(months, pmms=4.0, unrate=5.0)
    state = _state(months, {
        "CA": {"unrate": 7.0, "hpi": 200.0},
        "US": {"unrate": 9.9, "hpi": lambda m: {201411: 100.0, 201811: 110.0,
                                                201911: 125.0}.get(m, 1.0)},
    })
    out = mf.attach_macro(_loan(property_state="PR"), nat, state).row(0, named=True)

    assert out["unrate_state"] == 9.9           # US row, not CA's 7.0
    assert out["unrate_fallback"] == 1
    assert out["hpi_fallback"] == 1
    assert abs(out["ltv_mtm"] - 57.6) < 1e-9    # national HPI at both ends
    # a non-territory loan in the same batch must NOT be flagged.
    ca = mf.attach_macro(_loan(property_state="CA"), nat, state).row(0, named=True)
    assert ca["unrate_fallback"] == 0 and ca["hpi_fallback"] == 0
    assert ca["unrate_state"] == 7.0


def test_feature_spec_roles_disjoint_and_named():
    roles = mf.MACRO_FEATURE_SPEC
    allcols = [c for cols in roles.values() for c in cols]
    assert len(allcols) == len(set(allcols)), "a macro column is in two roles"
    # §4: these are the emitted feature columns; HPI levels are intermediate only.
    assert set(mf.MACRO_FEATURE_COLS) == {
        "pmms30", "dgs10", "slope_10y2y", "unrate_nat", "unrate_state",
        "incentive", "ltv_mtm", "hpi_chg_12m", "unrate_fallback", "hpi_fallback"}
    assert set(roles["intermediate"]) == {"hpi_state", "hpi_orig", "hpi_12m"}
    assert set(mf.LAG) == {"pmms30", "dgs10", "slope_10y2y", "unrate", "hpi"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
