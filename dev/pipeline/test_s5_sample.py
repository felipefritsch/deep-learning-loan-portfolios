"""Unit tests for s5_sample.py — the train-only feature scaler + leakage mask.

Hermetic (synthetic Polars/DuckDB frames; no SSD, no torch). Run with:
    .venv/bin/python dev/pipeline/test_s5_sample.py   (or python -m pytest)

Covers the §6.4 "never standardise the lake" contract:
  * ``training_mask`` is the strict ``period_ym < cutoff`` leakage filter;
  * ``fit_scaler`` computes centre/scale over the TRAIN slice only (cutoff isolation),
    with the robust-log path and the zero-scale guard;
  * ``apply_scaler`` standardises (log1p then (x-centre)/scale, null -> 0) and a
    fit -> apply round-trip re-centres a column to ~mean 0 / unit scale.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# See test_s4_panel.py: avoid the dev/model vs dev/pipeline ``config`` name clash.
_PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE))
for _m in ("config", "s5_sample"):
    _c = sys.modules.get(_m)
    if _c is not None and not str(getattr(_c, "__file__", "")).startswith(str(_PIPELINE)):
        del sys.modules[_m]

import duckdb  # noqa: E402
import polars as pl  # noqa: E402

import config  # noqa: E402  (pipeline config — this dir now leads sys.path)
import schema  # noqa: E402
import s5_sample as S5  # noqa: E402

CONT = schema.FEATURE_SPEC["continuous_standardize"]


# --- training_mask: the strict leakage filter ------------------------------
def test_training_mask_none_is_pass_through() -> None:
    assert S5.training_mask(None) == "TRUE"
    assert S5.training_mask(0) == "TRUE"  # falsy cutoff -> no mask


def test_training_mask_is_strict_less_than() -> None:
    assert S5.training_mask(201501) == "period_ym < 201501"


# --- apply_scaler: pure standardisation ------------------------------------
def test_apply_scaler_standardizes_and_fills_null() -> None:
    scaler = {"columns": {"x": {"log": False, "center": 10.0, "scale": 2.0}}}
    out = S5.apply_scaler(pl.DataFrame({"x": [12.0, 8.0, None]}), scaler)
    assert out["x"].to_list() == [1.0, -1.0, 0.0]  # (x-10)/2, null -> 0 (the mean)


def test_apply_scaler_applies_log1p_before_centering() -> None:
    scaler = {"columns": {"d": {"log": True, "center": 0.0, "scale": 1.0}}}
    out = S5.apply_scaler(pl.DataFrame({"d": [0.0, math.e - 1]}), scaler)
    got = out["d"].to_list()
    assert math.isclose(got[0], 0.0, abs_tol=1e-12)       # log1p(0) = 0
    assert math.isclose(got[1], 1.0, abs_tol=1e-12)       # log1p(e-1) = 1


def test_apply_scaler_skips_columns_absent_from_frame() -> None:
    scaler = {"columns": {"x": {"log": False, "center": 0.0, "scale": 1.0},
                          "missing": {"log": False, "center": 0.0, "scale": 1.0}}}
    out = S5.apply_scaler(pl.DataFrame({"x": [1.0, 2.0]}), scaler)
    assert "missing" not in out.columns and out["x"].to_list() == [1.0, 2.0]


# --- fit_scaler: train-slice isolation, robust-log path, zero-scale guard ---
def _register_panel(con: duckdb.DuckDBPyConnection) -> None:
    """A 4-row synthetic ``panel``: rows 0-1 are below a 2015 cutoff, rows 2-3 above.
    Focus columns vary below-cutoff so their fitted stats are exactly checkable; the
    above-cutoff rows take an extreme value so leakage would be obvious."""
    cols = {c: [1.0, 1.0, 1.0, 1.0] for c in CONT}
    cols["Original Interest Rate"] = [3.0, 5.0, 100.0, 100.0]   # below: mean 4, sd sqrt(2)
    cols["Original UPB"] = [100.0, 300.0, 9999.0, 9999.0]       # a LOG_TRANSFORM column
    cols["Number of Units"] = [1.0, 1.0, 1.0, 1.0]              # constant -> zero-scale guard
    df = pl.DataFrame(cols).with_columns(
        pl.Series("period_ym", [201001, 201001, 202001, 202001]),
        pl.Series("orig_ym", [200001, 200001, 200001, 200001]),
        pl.Series("state", ["current"] * 4),
        pl.Series("state_next", ["dpd_30"] * 4),
    )
    con.register("panel", df.to_arrow())


def test_fit_scaler_uses_train_slice_only() -> None:
    assert "Original UPB" in schema.LOG_TRANSFORM  # guard against spec drift
    con = duckdb.connect()
    _register_panel(con)
    train = S5.fit_scaler(con, cutoff_ym=201501, save=False)["columns"]
    full = S5.fit_scaler(con, cutoff_ym=None, save=False)["columns"]
    con.close()

    rate = train["Original Interest Rate"]
    assert rate["method"] == "mean/std"
    assert math.isclose(rate["center"], 4.0, abs_tol=1e-9)
    assert math.isclose(rate["scale"], math.sqrt(2.0), rel_tol=1e-9)
    # Without the cutoff the extreme above-window rows leak in and shift the centre.
    assert full["Original Interest Rate"]["center"] > 40.0


def test_fit_scaler_log_path_and_zero_scale_guard() -> None:
    con = duckdb.connect()
    _register_panel(con)
    cols = S5.fit_scaler(con, cutoff_ym=201501, save=False)["columns"]
    con.close()

    upb = cols["Original UPB"]
    assert upb["log"] is True and upb["method"] == "log1p+median/IQR"
    # median of log1p of the two below-cutoff values lies between them.
    assert math.log1p(100.0) <= upb["center"] <= math.log1p(300.0)
    # A constant column has zero spread -> guarded to scale 1.0 (no divide-by-zero).
    assert cols["Number of Units"]["scale"] == 1.0


def test_fit_then_apply_roundtrip_recenters_column() -> None:
    con = duckdb.connect()
    _register_panel(con)
    scaler = S5.fit_scaler(con, cutoff_ym=201501, save=False)
    con.close()
    below = pl.DataFrame({"Original Interest Rate": [3.0, 5.0]})
    z = S5.apply_scaler(below, scaler)["Original Interest Rate"].to_list()
    assert math.isclose(sum(z) / len(z), 0.0, abs_tol=1e-9)        # re-centred
    assert math.isclose(z[0], -1 / math.sqrt(2), rel_tol=1e-9)
    assert math.isclose(z[1], 1 / math.sqrt(2), rel_tol=1e-9)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
