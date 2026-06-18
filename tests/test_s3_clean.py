"""Unit tests for s3_clean.py — the per-batch type/standardisation transform.

Hermetic (a synthetic string-typed batch; no SSD, no torch). Run with:
    python -m pytest tests/test_s3_clean.py

The Stage-3 transform takes the faithful, string-typed perf lake (``KEEP_COLS``) and
produces the typed clean lake (``OUTPUT_COLS``). This pins the conversions that are
easy to get subtly wrong: MMYYYY date parsing, the period/orig calendar keys, Y/N→bool
(unknown→null), categorical code→label maps, the coalesced origination FICO, and the
missingness indicators.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from floan.pipeline import config, schema
from floan.pipeline import s3_clean as S3


def _raw_batch() -> pl.DataFrame:
    """Two rows of the string-typed perf lake: a fully-populated loan and a sparse one
    that also exercises the FICO field migration (old blank → new ``Classic FICO``)."""
    base = {c: ["", ""] for c in schema.KEEP_COLS}
    base.update(
        {
            "Loan Identifier": ["L1", "L2"],
            "Monthly Reporting Period": ["032020", "122021"],
            "Origination Date": ["012020", "112021"],
            "Channel": ["R", "B"],
            "Original Interest Rate": ["3.5", ""],
            "Borrower Credit Score at Origination": ["740", ""],
            "Origination Classic FICO": ["", "680"],
            "Co-Borrower Credit Score at Origination": ["700", ""],
            "First-Time Home Buyer Indicator": ["Y", "N"],
            "Interest-Only Loan Indicator": ["N", "9"],
            "Modification Flag": ["N", "Y"],
            "Debt-to-Income (DTI)": ["35", ""],
            "Original Combined LTV (CLTV)": ["80", ""],
            "Mortgage Insurance Percentage": ["25", ""],
            "Number of Units": ["1", "2"],
            "Current Loan Delinquency Status": ["0", "1"],
        }
    )
    return pl.DataFrame(base)


def test_output_schema_is_exactly_output_cols() -> None:
    out = S3._transform_batch(_raw_batch())
    assert out.columns == S3.OUTPUT_COLS


def test_dates_and_calendar_keys() -> None:
    out = S3._transform_batch(_raw_batch())
    assert out["period"].to_list() == [date(2020, 3, 1), date(2021, 12, 1)]
    assert out["period_ym"].to_list() == [202003, 202112]
    assert out["orig_ym"].to_list() == [202001, 202111]


def test_yn_indicators_map_to_bool_with_unknown_null() -> None:
    out = S3._transform_batch(_raw_batch())
    assert out["First-Time Home Buyer Indicator"].to_list() == [True, False]
    assert out["Modification Flag"].to_list() == [False, True]
    assert out["Interest-Only Loan Indicator"].to_list() == [False, None]  # "9" -> null


def test_categorical_codes_map_to_labels() -> None:
    out = S3._transform_batch(_raw_batch())
    assert out["Channel"].to_list() == ["Retail", "Broker"]


def test_float_cast_and_nulls() -> None:
    out = S3._transform_batch(_raw_batch())
    assert out["Original Interest Rate"].dtype == pl.Float32
    assert out["Original Interest Rate"].to_list() == [3.5, None]


def test_fico_orig_coalesces_old_then_new() -> None:
    out = S3._transform_batch(_raw_batch())
    # row 0 uses the old field (740); row 1 falls back to the new Classic FICO (680).
    assert out["fico_orig"].to_list() == [740, 680]
    assert out["fico_co"].to_list() == [700, None]


def test_missingness_indicators() -> None:
    out = S3._transform_batch(_raw_batch())
    assert out["fico_orig_missing"].to_list() == [False, False]
    assert out["fico_co_missing"].to_list() == [False, True]
    assert out["dti_missing"].to_list() == [False, True]
    assert out["cltv_missing"].to_list() == [False, True]
    assert out["mi_pct_missing"].to_list() == [False, True]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
