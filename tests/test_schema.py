"""Unit tests for schema.py — the positional layout and the seven-state target.

Run with: ``python -m pytest test_schema.py``  (or ``python -m pytest tests/test_schema.py``).
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from floan.pipeline import schema


def test_columns_count_is_113():
    assert len(schema.COLUMNS) == 113
    assert len(set(schema.COLUMNS)) == 113  # unique
    # Positions 110-112 confirmed against the official glossary (fields 111-113).
    assert schema.COLUMNS[110:] == [
        "Origination Classic FICO", "Issuance Classic FICO", "Current Classic FICO",
    ]
    assert schema.COLUMNS[78] == "Special Eligibility Program"  # glossary fld 79


def test_fico_orig_coalesces_old_then_new():
    df = pl.DataFrame({
        "Borrower Credit Score at Origination": ["720", "", "9999"],
        "Origination Classic FICO": ["680", "700", "710"],
    })
    out = df.select(schema.fico_orig_expr())["fico_orig"].to_list()
    assert out == [720, 700, 710]  # old wins when valid; else falls back to new


def test_dtypes_cover_all_columns():
    assert set(schema.DTYPES) == set(schema.COLUMNS)


def test_keep_cols_subset_of_columns():
    assert all(c in schema.COLUMNS for c in schema.KEEP_COLS)


def test_parse_mmyyyy():
    df = pl.DataFrame({"d": ["112013", "012020", "", "XX", "12013"]})
    out = df.select(schema.parse_mmyyyy("d").alias("p"))["p"].to_list()
    assert out[0] == dt.date(2013, 11, 1)
    assert out[1] == dt.date(2020, 1, 1)
    assert out[2] is None      # empty
    assert out[3] is None      # non-numeric
    assert out[4] is None      # wrong length


def test_scrub_credit_score():
    df = pl.DataFrame({"f": ["720", "9999", "0", "299", "851", "300", "850"]})
    out = df.select(schema.scrub_credit_score("f").alias("s"))["s"].to_list()
    assert out == [720, None, None, None, None, 300, 850]


def test_derive_state_covers_seven_states_plus_null():
    # (delinq, zero_balance) → expected state
    cases = [
        ("00", "",   "current"),
        ("01", "",   "dpd_30"),
        ("02", "",   "dpd_60"),
        ("03", "",   "dpd_90plus"),
        ("04", "",   "dpd_90plus"),   # 120+ folds into 90plus
        ("00", "01", "prepaid"),       # ZB precedence over current
        ("06", "02", "foreclosure"),   # third-party sale; ZB wins over delinquency
        ("09", "09", "REO"),
        ("XX", "",   None),            # unknown delinquency → null
        ("",   "",   None),            # blank → null
    ]
    df = pl.DataFrame(
        {
            "Current Loan Delinquency Status": [c[0] for c in cases],
            "Zero Balance Code": [c[1] for c in cases],
        }
    )
    out = df.select(schema.derive_state())["state"].to_list()
    expected = [c[2] for c in cases]
    assert out == expected, f"{out} != {expected}"

    # All seven labels are reachable by the derivation.
    reachable = set(o for o in out if o is not None)
    assert {"current", "dpd_30", "dpd_60", "dpd_90plus",
            "prepaid", "foreclosure", "REO"} <= reachable | {
        schema.ZERO_BAL_STATE[k] for k in schema.ZERO_BAL_STATE}


def test_state_labels_match_constant():
    assert set(schema.STATES) == {
        "current", "dpd_30", "dpd_60", "dpd_90plus",
        "foreclosure", "REO", "prepaid",
    }
    assert set(schema.ZERO_BAL_STATE.values()) <= set(schema.STATES)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
