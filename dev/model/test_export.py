"""Unit tests for export.py — the shared-pool schema, thinning weight, and label shift.

Hermetic (constants + a tiny DuckDB eval; no SSD, no torch). Run with:
    .venv/bin/python dev/model/test_export.py   (or python -m pytest)

Pins the parts of the M4 export that a silent change would corrupt: the on-disk
column order, the current→current importance weight (1/p_keep), and the one-month
label shift SQL — cross-checked against config.label_year so the SQL and the Python
window logic cannot drift apart.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# See test_config.py: bind the MODEL config even if a pipeline config was cached first.
_MODEL = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODEL))
_c = sys.modules.get("config")
if _c is not None and not str(getattr(_c, "__file__", "")).startswith(str(_MODEL)):
    del sys.modules["config"]

import duckdb  # noqa: E402
import polars as pl  # noqa: E402

import config as cfg  # noqa: E402  (model config — this dir now leads sys.path)
import export as EX  # noqa: E402


def test_output_cols_layout() -> None:
    cols = EX.OUTPUT_COLS
    assert EX.KEY_COLS == ["Loan Identifier", "shard", "period", "period_ym", "orig_ym"]
    k = len(EX.KEY_COLS)
    assert list(cols[:k]) == EX.KEY_COLS              # keys first, in order
    assert cols[k] == "label_ym" and cols[k + 1] == "weight"   # then label + weight
    assert list(cols[k + 2:k + 2 + len(EX.TARGET_COLS)]) == EX.TARGET_COLS
    assert cols[0] == "Loan Identifier" and cols[-1] == "mkt_rate"
    assert len(cols) == len(set(cols))                # no duplicate columns


def test_thinning_weight_and_threshold() -> None:
    # current→current rows are kept with prob p_keep and carry weight 1/p_keep.
    assert math.isclose(1.0 / cfg.P_KEEP_CURRENT, 20.0, rel_tol=1e-12)
    assert EX._THR == int(cfg.P_KEEP_CURRENT * 1_000_000) == 50_000


def test_label_ym_sql_is_next_month_and_matches_config() -> None:
    months = [202001, 202006, 202011, 202012, 200012, 201912]
    con = duckdb.connect()
    con.register("t", pl.DataFrame({"period_ym": months}).to_arrow())
    rows = con.execute(f"SELECT period_ym, {EX._LABEL_YM} AS label_ym FROM t").fetchall()
    con.close()
    for period_ym, label_ym in rows:
        y, mo = divmod(period_ym, 100)
        mo += 1
        if mo == 13:
            y, mo = y + 1, 1
        assert label_ym == y * 100 + mo                       # exact next calendar month
        assert label_ym // 100 == cfg.label_year(period_ym)   # SQL agrees with config


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
