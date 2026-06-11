"""MD3 — Macro join + derived features (``05_MACRO_DATA.md`` §4).

The reusable contract for joining the two tiny macro tables (built by
``build_macro.py``) onto a loan-month batch at export time and deriving the macro
features. ``export.py`` (M4) and ``features.py`` (M6) import from here so the lag
convention and the feature-spec additions have a single source of truth.

It operates on a **RAM-sized loan-month batch** (one shard / streaming chunk — never
the whole panel), so a Polars hash-join against the tiny macro tables is memory-safe.

What it produces, per loan-month (``join_macro`` → ``add_derived``):

  contemporaneous (lagged) : pmms30, dgs10, slope_10y2y  (lag 0)
                             unrate_nat, unrate_state     (lag 1)
                             hpi_state                    (lag 2, period HPI)
  intermediate HPI levels  : hpi_orig (HPI at orig_ym), hpi_12m (HPI 12m before t)
  derived                  : incentive   = current_rate − pmms30
                             ltv_mtm      = (current_upb/orig_upb)·orig_ltv·hpi_orig/hpi_state
                             hpi_chg_12m  = hpi_state/hpi_12m − 1
  fallback indicators      : unrate_fallback, hpi_fallback  (1 ⇒ state row absent,
                             national US-row used instead — territories PR/VI/GU, etc.)

HPI is looked up at both ``period_ym`` and ``orig_ym`` under the *same* 2-month lag
(§4); the macro tables start 1998-01 precisely so those origination / 12-month-prior
lookups resolve for the earliest (1999) vintages (see ``build_macro.py``).

The companion unit tests are in ``test_macro_features.py``.
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Lag convention — SINGLE SOURCE OF TRUTH (§4, §6). Publication realism: month-t
# unemployment prints early in t+1; FMHPI ~2 months later; rates are real-time.
# A loan-month at period_ym=t joins the macro value of month sub_months(t, LAG).
# ---------------------------------------------------------------------------
LAG = {"pmms30": 0, "dgs10": 0, "slope_10y2y": 0, "unrate": 1, "hpi": 2}

# Canonical loan-frame column names this module reads. export.py aliases the panel's
# verbose names to these before calling join_macro (period_ym / orig_ym are already
# canonical in the panel).
PANEL_ALIASES = {
    "Property State": "property_state",
    "Current Interest Rate": "current_rate",
    "Current Actual UPB": "current_upb",
    "Original UPB": "orig_upb",
    "Original Loan-to-Value (LTV)": "orig_ltv",
}

# Feature-spec additions (§4). HPI *levels* are role 'intermediate' — never features.
MACRO_FEATURE_SPEC: dict[str, list[str]] = {
    "continuous_standardize": [
        "pmms30", "dgs10", "slope_10y2y", "unrate_nat", "unrate_state",
        "incentive", "ltv_mtm", "hpi_chg_12m",
    ],
    "binary": ["unrate_fallback", "hpi_fallback"],
    "intermediate": ["hpi_state", "hpi_orig", "hpi_12m"],
}
# The macro columns the export emits as model features (intermediates excluded).
MACRO_FEATURE_COLS: list[str] = (
    MACRO_FEATURE_SPEC["continuous_standardize"] + MACRO_FEATURE_SPEC["binary"]
)


# ---------------------------------------------------------------------------
# Month arithmetic on YYYYMM integers
# ---------------------------------------------------------------------------
def sub_months(ym: int, k: int) -> int:
    """``YYYYMM`` integer minus ``k`` calendar months (handles year rollover)."""
    t = (ym // 100) * 12 + (ym % 100 - 1) - k
    return (t // 12) * 100 + (t % 12) + 1


def _sub_months_expr(col: str, k: int) -> pl.Expr:
    t = (pl.col(col) // 100) * 12 + (pl.col(col) % 100 - 1) - k
    return ((t // 12) * 100 + (t % 12) + 1).cast(pl.Int32)


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------
def _join_national(loans: pl.DataFrame, nat: pl.DataFrame) -> pl.DataFrame:
    """Add pmms30/dgs10/slope_10y2y (lag 0) and unrate_nat (lag 1)."""
    rates = nat.select(
        pl.col("month_ym").cast(pl.Int32).alias("_m_rate"),
        "pmms30", "dgs10", "slope_10y2y",
    )
    unr = nat.select(
        pl.col("month_ym").cast(pl.Int32).alias("_m_unrate"),
        pl.col("unrate_nat"),
    )
    return loans.join(rates, on="_m_rate", how="left").join(unr, on="_m_unrate", how="left")


def _join_state_value(loans: pl.DataFrame, state: pl.DataFrame, month_key: str,
                      value_col: str, out: str) -> pl.DataFrame:
    """Look up ``value_col`` for (property_state, month_key); fall back to the ``US``
    pseudo-state row when the state row is absent. Adds ``out`` and ``out_fb`` (1/0)."""
    s = state.select(
        pl.col("state").cast(pl.Utf8),
        pl.col("month_ym").cast(pl.Int32).alias(month_key),
        pl.col(value_col).alias("_sv"),
    )
    us = state.filter(pl.col("state") == "US").select(
        pl.col("month_ym").cast(pl.Int32).alias(month_key),
        pl.col(value_col).alias("_usv"),
    )
    return (
        loans
        .join(s, left_on=["property_state", month_key], right_on=["state", month_key],
              how="left")
        .join(us, on=month_key, how="left")
        .with_columns(
            pl.coalesce(["_sv", "_usv"]).alias(out),
            (pl.col("_sv").is_null() & pl.col("_usv").is_not_null()).cast(pl.Int8).alias(f"{out}_fb"),
        )
        .drop("_sv", "_usv")
    )


def join_macro(loans: pl.DataFrame, nat: pl.DataFrame, state: pl.DataFrame) -> pl.DataFrame:
    """Join both macro tables onto a loan-month batch with the §4 per-variable lags.

    ``loans`` must carry ``period_ym``, ``orig_ym``, ``property_state`` (and, for the
    derived step, ``current_rate/current_upb/orig_upb/orig_ltv``). Returns ``loans``
    with the contemporaneous macro columns, the intermediate HPI levels, and the
    ``unrate_fallback`` / ``hpi_fallback`` indicators added.
    """
    out = loans.with_columns(
        _sub_months_expr("period_ym", LAG["pmms30"]).alias("_m_rate"),
        _sub_months_expr("period_ym", LAG["unrate"]).alias("_m_unrate"),
        _sub_months_expr("period_ym", LAG["hpi"]).alias("_m_hpi"),
        _sub_months_expr("orig_ym", LAG["hpi"]).alias("_m_hpi_orig"),
        _sub_months_expr("period_ym", LAG["hpi"] + 12).alias("_m_hpi_12m"),
    )
    out = _join_national(out, nat)
    out = _join_state_value(out, state, "_m_unrate", "unrate_state", "unrate_state")
    out = _join_state_value(out, state, "_m_hpi", "hpi_state", "hpi_state")
    out = _join_state_value(out, state, "_m_hpi_orig", "hpi_state", "hpi_orig")
    out = _join_state_value(out, state, "_m_hpi_12m", "hpi_state", "hpi_12m")

    return out.with_columns(
        pl.col("unrate_state_fb").alias("unrate_fallback"),
        # any HPI lookup falling back ⇒ national HPI used (consistent across the ratio).
        (pl.col("hpi_state_fb") | pl.col("hpi_orig_fb") | pl.col("hpi_12m_fb"))
        .cast(pl.Int8).alias("hpi_fallback"),
    ).drop("_m_rate", "_m_unrate", "_m_hpi", "_m_hpi_orig", "_m_hpi_12m",
           "unrate_state_fb", "hpi_state_fb", "hpi_orig_fb", "hpi_12m_fb")


def add_derived(loans: pl.DataFrame) -> pl.DataFrame:
    """Compute incentive, ltv_mtm, hpi_chg_12m from the joined macro columns (§4).

    Note for M4: the macro *join* columns are null-free, but ``incentive`` inherits
    nulls from missing ``current_rate`` (~3% of loan-months — terminal/zero-balance
    rows), to be handled by the pipeline's missingness machinery, not here.
    """
    return loans.with_columns(
        (pl.col("current_rate") - pl.col("pmms30")).alias("incentive"),
        ((pl.col("current_upb") / pl.col("orig_upb")) * pl.col("orig_ltv")
         * pl.col("hpi_orig") / pl.col("hpi_state")).alias("ltv_mtm"),
        (pl.col("hpi_state") / pl.col("hpi_12m") - 1.0).alias("hpi_chg_12m"),
    )


def attach_macro(loans: pl.DataFrame, nat: pl.DataFrame, state: pl.DataFrame) -> pl.DataFrame:
    """join_macro + add_derived — the full per-batch macro attachment (export entry)."""
    return add_derived(join_macro(loans, nat, state))
