"""Unit tests for s1_inventory.py — the pure inventory helpers + cross-file checks.

Hermetic (synthetic manifest rows; no SSD, no torch). Run with:
    python -m pytest tests/test_s1_inventory.py

Covers the order/range helpers and the two global checks that gate ingestion:
release-cut-off divergence and the relative-volume (clean-truncation) outlier.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from floan.pipeline import config
from floan.pipeline import s1_inventory as S1


# --- pure order / range helpers --------------------------------------------
def test_quarter_key_parses_and_sorts_garbage_last() -> None:
    assert S1._quarter_key("2014Q1") == (2014, 1)
    assert S1._quarter_key("2020Q4") == (2020, 4)
    assert S1._quarter_key("not-a-quarter") == (9999, 9)  # sorts last


def test_all_quarters_between_is_inclusive_with_year_rollover() -> None:
    assert S1._all_quarters_between("2000Q1", "2000Q4") == ["2000Q1", "2000Q2", "2000Q3", "2000Q4"]
    assert S1._all_quarters_between("2019Q3", "2020Q2") == ["2019Q3", "2019Q4", "2020Q1", "2020Q2"]
    assert S1._all_quarters_between("2005Q2", "2005Q2") == ["2005Q2"]


def test_fingerprint_is_size_and_mtime() -> None:
    with tempfile.NamedTemporaryFile() as fh:
        fh.write(b"hello")
        fh.flush()
        st = os.stat(fh.name)
        assert S1._fingerprint(st) == f"{st.st_size}:{st.st_mtime_ns}"


def test_coerce_normalises_csv_strings() -> None:
    row = {
        "bytes": "100", "rows": "5", "distinct_loans": "",
        "size_mb": "1.5", "bytes_per_row": "None", "volume_ratio": "",
        "real_data_pct": "", "quarantined": "True", "nul_corrupt": "false",
        "cutoff_divergent": "", "low_volume": "true",
    }
    out = S1._coerce([row])[0]
    assert out["bytes"] == 100 and out["rows"] == 5 and out["distinct_loans"] == 0
    assert out["size_mb"] == 1.5 and out["bytes_per_row"] == 0.0 and out["volume_ratio"] == 0.0
    assert out["real_data_pct"] is None        # blank stays unknown, not 100
    assert out["quarantined"] is True and out["nul_corrupt"] is False
    assert out["cutoff_divergent"] is False and out["low_volume"] is True


# --- cross-file checks ------------------------------------------------------
def _row(q: str, distinct: int, max_period: str, **kw) -> dict:
    r = {"acq_quarter": q, "distinct_loans": distinct, "max_period": max_period}
    r.update(kw)
    return r


def test_cutoff_divergence_flags_the_odd_release() -> None:
    rows = [
        _row("2020Q1", 100_000, "202412"),
        _row("2020Q2", 100_000, "202412"),
        _row("2020Q3", 100_000, "202312"),   # came from an older release
        _row("2020Q4", 100_000, "202412"),
        _row("2021Q1", 100_000, "202412"),
    ]
    out, modal = S1._apply_cross_file_checks(rows)
    byq = {r["acq_quarter"]: r for r in out}
    assert modal == "202412"
    assert byq["2020Q3"]["cutoff_divergent"] is True and byq["2020Q3"]["quarantined"] is True
    assert "cutoff!=202412" in byq["2020Q3"]["quarantine_reason"]
    assert byq["2020Q1"]["cutoff_divergent"] is False


def test_relative_volume_outlier_is_quarantined() -> None:
    rows = [
        _row("2020Q1", 100_000, "202412"),
        _row("2020Q2", 100_000, "202412"),
        _row("2020Q3", 20_000, "202412"),    # < 25% of the neighbour median (100k)
        _row("2020Q4", 100_000, "202412"),
        _row("2021Q1", 100_000, "202412"),
    ]
    out, _ = S1._apply_cross_file_checks(rows)
    byq = {r["acq_quarter"]: r for r in out}
    assert byq["2020Q3"]["low_volume"] is True and byq["2020Q3"]["quarantined"] is True
    assert byq["2020Q3"]["volume_ratio"] == 0.2
    assert byq["2020Q1"]["low_volume"] is False


def test_corrupt_file_is_not_double_flagged_low_volume() -> None:
    rows = [
        _row("2020Q1", 100_000, "202412"),
        _row("2020Q2", 100_000, "202412"),
        _row("2020Q3", 20_000, "202412", nul_corrupt=True),  # corruption already owns it
        _row("2020Q4", 100_000, "202412"),
        _row("2021Q1", 100_000, "202412"),
    ]
    out, _ = S1._apply_cross_file_checks(rows)
    byq = {r["acq_quarter"]: r for r in out}
    assert byq["2020Q3"]["low_volume"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
