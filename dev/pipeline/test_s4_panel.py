"""Unit tests for s4_panel.py — the seven-state transition target's heart.

Hermetic (synthetic Polars/DuckDB frames; no SSD, no torch). Run with:
    .venv/bin/python dev/pipeline/test_s4_panel.py   (or python -m pytest)

Covers two things the panel build rests on:
  * ``_finalize`` — ``state_next`` is the next row's state only when it is the SAME
    loan (else NULL = absorbing termination / right-censored end), and ``censored``
    is set only for a known, non-absorbing state with no observed next month;
  * the ``_STATE_SQL`` CASE (used by the DuckDB build) stays bit-for-bit equivalent
    to ``schema.derive_state`` (the Polars reference) — a parity that ``validate()``
    only checks at runtime on the real lake.
"""

from __future__ import annotations

import sys
from pathlib import Path

# dev/model and dev/pipeline each ship a ``config.py``; in a single pytest session the
# model one may already sit in sys.modules under the bare name ``config`` (e.g. cached
# by dev/model/test_config.py). Put this file's own dir first and drop any stale
# ``config`` so the pipeline imports below bind to the pipeline package.
_PIPELINE = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE))
for _m in ("config", "s4_panel"):
    _c = sys.modules.get(_m)
    if _c is not None and not str(getattr(_c, "__file__", "")).startswith(str(_PIPELINE)):
        del sys.modules[_m]

import duckdb  # noqa: E402
import polars as pl  # noqa: E402

import config  # noqa: E402  (pipeline config — this dir now leads sys.path)
import schema  # noqa: E402
import s4_panel as S4  # noqa: E402


# --- _finalize: state_next within a loan, censored only for live transients ----
def _lookahead_frame() -> pl.DataFrame:
    """A hand-built batch in the shape ``_finalize`` consumes: base columns plus the
    one-row lookahead (``_next_loan``/``_next_state``) and the ``state``/``shard``
    DERIVED inputs. Three loans exercise a roll, a cure, a censor and an absorption."""
    return pl.DataFrame(
        {
            "Loan Identifier": ["A", "A", "A", "B", "B", "C"],
            "period_ym": [200001, 200002, 200003, 200001, 200002, 200001],
            "state": ["current", "dpd_30", "current", "current", "prepaid", None],
            "shard": [0, 0, 0, 1, 1, 2],
            "_next_loan": ["A", "A", "B", "B", None, None],
            "_next_state": ["dpd_30", "current", "current", "prepaid", None, None],
        }
    )


def test_finalize_state_next_is_next_state_within_loan_else_null() -> None:
    out = S4._finalize(_lookahead_frame())
    assert out["state_next"].to_list() == [
        "dpd_30",   # A: current -> dpd_30 (same loan next row)
        "current",  # A: dpd_30 -> current (cure)
        None,       # A's last row: next row is loan B -> boundary -> NULL
        "prepaid",  # B: current -> prepaid
        None,       # B's last row: no next row -> NULL
        None,       # C: single row -> NULL
    ]


def test_finalize_censored_only_for_known_nonabsorbing_terminal() -> None:
    out = S4._finalize(_lookahead_frame())
    # Censored == state_next is NULL AND state is known AND state is not absorbing.
    assert out["censored"].to_list() == [
        False,  # has a next state
        False,  # has a next state
        True,   # current, no next -> right-censored
        False,  # has a next state
        False,  # prepaid is absorbing -> not censored, it genuinely ended
        False,  # state is NULL -> not censored
    ]


def test_finalize_drops_lookahead_and_orders_derived_last() -> None:
    out = S4._finalize(_lookahead_frame())
    assert "_next_loan" not in out.columns and "_next_state" not in out.columns
    # base columns keep their order, then the DERIVED block in its fixed order.
    assert out.columns == ["Loan Identifier", "period_ym", "state", "state_next", "censored", "shard"]


# --- _STATE_SQL (DuckDB) parity with schema.derive_state (Polars) --------------
def _state_driver_frame() -> pl.DataFrame:
    """Rows spanning every Zero-Balance terminal code, each delinquency bucket, an
    unknown ZB code, blanks and non-numeric / NULL drivers."""
    zb = ["01", "02", "03", "06", "09", "15", "16", "96", "97", "98",
          "", "", "", "", "", "99", None, None, ""]
    dq = ["0", "0", "1", "2", "3", "9", "0", "1", "2", "3",
          "0", "1", "2", "3", "5", "0", "0", None, "XX"]
    return pl.DataFrame({"Zero Balance Code": zb, "Current Loan Delinquency Status": dq})


def test_sql_state_matches_schema_derive_state() -> None:
    df = _state_driver_frame()
    polars_state = df.select(schema.derive_state()).to_series().to_list()
    con = duckdb.connect()
    con.register("t", df.to_arrow())
    sql_state = con.execute(f"SELECT {S4._STATE_SQL} AS state FROM t").pl()["state"].to_list()
    con.close()
    assert sql_state == polars_state, list(zip(sql_state, polars_state))


def test_sql_state_covers_all_seven_states() -> None:
    """Sanity: the driver frame actually exercises every non-null state (so the parity
    test above is not vacuously matching on NULLs)."""
    df = _state_driver_frame()
    seen = set(df.select(schema.derive_state()).to_series().to_list()) - {None}
    assert seen == set(S4.STATES)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
