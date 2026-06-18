"""MD2 — Build the two macro Parquet tables (``05_MACRO_DATA.md`` §3).

Reads the newest snapshot under ``raw/macro/`` (or ``--snapshot``) and writes two
tiny, time-keyed tables that ``export.py`` joins onto the loan panel at export time
(the 800 GB lake is never touched):

  processed/macro/macro_national.parquet  — month_ym, pmms30, dgs10, slope_10y2y, unrate_nat
  processed/macro/macro_state.parquet     — state, month_ym, unrate_state, hpi_state
                                            (incl. a pseudo-state ``US`` fallback row)

Values are stored **by observation month**; publication lags are applied at join
time, not baked in. FRED CSVs are parsed defensively (``.`` / empty = missing); every
series is reduced to monthly by a calendar-month mean (a no-op for already-monthly
series, the spec's month-mean for weekly PMMS, and — for DGS10/DGS2 — applied by FRED
server-side at fetch time, see ``fetch_macro.py``). Each series is asserted monotone
and gap-free over its table span (single-month interior gaps forward-filled, longer
gaps fail).

Run:  .venv/bin/python dev/model/build_macro.py [--snapshot YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

# The pipeline package is the single source of truth for paths.
from floan.pipeline import config

# Macro tables begin 1998-01, not 2000-01. The panel's period_ym starts at 200001,
# but §4's joins are LAGGED (unrate −1m, hpi −2m) and ltv_mtm/hpi_chg_12m look up HPI
# at the loan's origination month and 12 months prior — so the earliest lookups reach
# 1998-11 (min orig_ym 199901 − 2m; period 200001 − 14m). Starting at 1998-01 (every
# FRED series and FMHPI cover it gap-free) is what makes "zero null macro columns"
# achievable downstream; 2000-01 → snapshot end (the spec's stated span) is a gap-free
# subset. See dev/model/macro_features.py for the join/lag contract.
START_YM = 199801

# 50 states + DC — their FRED LAUS SA series are "{POSTAL}UR".
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
]


# ---------------------------------------------------------------------------
# Snapshot resolution & parsing
# ---------------------------------------------------------------------------
def _newest_snapshot() -> Path:
    base = config.RAW / "macro"
    snaps = sorted(p for p in base.iterdir() if p.is_dir())
    if not snaps:
        raise SystemExit(f"no snapshot folders under {base} — run fetch_macro.py first")
    return snaps[-1]


def _fred_monthly(path: Path, name: str) -> pl.DataFrame:
    """A FRED CSV → ``[month_ym (i32), <name> (f64)]``, one row per calendar month.

    Column 0 is the observation date (name varies), column 1 the value with ``.`` /
    empty = missing. Monthly mean over non-missing values reduces weekly/daily series
    and is a no-op for already-monthly ones; months with no observation are dropped.
    """
    df = pl.read_csv(path, infer_schema=False)
    date_col, val_col = df.columns[0], df.columns[1]
    return (
        df.select(
            (pl.col(date_col).str.slice(0, 4).cast(pl.Int32) * 100
             + pl.col(date_col).str.slice(5, 2).cast(pl.Int32)).alias("month_ym"),
            pl.when(pl.col(val_col).is_in([".", ""]) | pl.col(val_col).is_null())
              .then(None)
              .otherwise(pl.col(val_col))
              .cast(pl.Float64)
              .alias(name),
        )
        .group_by("month_ym")
        .agg(pl.col(name).mean())     # mean ignores nulls
        .drop_nulls(name)             # months with no observation
        .sort("month_ym")
    )


def _fmhpi(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """FMHPI master file → (national, state) seasonally-adjusted HPI, monthly.

    Schema asserted on load (names may drift): Year, Month, GEO_Type ∈
    {US, State, CBSA}, GEO_Name, an SA index column. Returns the US row as
    ``hpi_nat`` and the 50-states+DC rows as ``[state, month_ym, hpi_state]``.
    """
    df = pl.read_csv(path, infer_schema_length=1000)
    need = {"Year", "Month", "GEO_Type", "GEO_Name", "Index_SA"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"FMHPI schema unexpected — missing columns {missing}; "
                         f"have {df.columns}")
    geo = set(df["GEO_Type"].unique().to_list())
    if not {"US", "State"} <= geo:
        raise SystemExit(f"FMHPI GEO_Type missing US/State — found {geo}")

    base = df.select(
        (pl.col("Year").cast(pl.Int32) * 100 + pl.col("Month").cast(pl.Int32)).alias("month_ym"),
        pl.col("GEO_Type"), pl.col("GEO_Name"),
        pl.col("Index_SA").cast(pl.Float64),
    )
    nat = (base.filter(pl.col("GEO_Type") == "US")
               .select("month_ym", pl.col("Index_SA").alias("hpi_nat"))
               .sort("month_ym"))
    state = (base.filter((pl.col("GEO_Type") == "State")
                         & pl.col("GEO_Name").is_in(STATES))
                 .select(pl.col("GEO_Name").alias("state"), "month_ym",
                         pl.col("Index_SA").alias("hpi_state"))
                 .sort("state", "month_ym"))
    return nat, state


# ---------------------------------------------------------------------------
# Gap-free monthly reindex
# ---------------------------------------------------------------------------
def _month_range(start: int, end: int) -> list[int]:
    out, y, m = [], start // 100, start % 100
    while y * 100 + m <= end:
        out.append(y * 100 + m)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _reindex_gapfree(df: pl.DataFrame, value_cols: list[str], start: int, end: int,
                     label: str) -> pl.DataFrame:
    """Reindex ``df`` to every month in [start, end]; forward-fill single-month
    interior gaps, fail on any longer gap (per §3)."""
    grid = pl.DataFrame({"month_ym": _month_range(start, end)}, schema={"month_ym": pl.Int32})
    out = grid.join(df.with_columns(pl.col("month_ym").cast(pl.Int32)),
                    on="month_ym", how="left").sort("month_ym")
    filled = out.with_columns(pl.col(c).forward_fill(limit=1) for c in value_cols)
    for c in value_cols:
        n = filled[c].null_count()
        if n:
            holes = filled.filter(pl.col(c).is_null())["month_ym"].to_list()
            raise SystemExit(f"{label}: '{c}' has {n} unfillable gap(s) "
                             f"(>1 month) at {holes[:12]}")
    return filled


def _last_full_month(*frames_cols: tuple[pl.DataFrame, str]) -> int:
    """Latest month where every given (frame, col) has a non-null value."""
    return min(
        df.drop_nulls(col)["month_ym"].max() for df, col in frames_cols
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_national(fred: dict[str, pl.DataFrame]) -> pl.DataFrame:
    pmms, d10, d2, unr = fred["MORTGAGE30US"], fred["DGS10"], fred["DGS2"], fred["UNRATE"]
    end = _last_full_month((pmms, "MORTGAGE30US"), (d10, "DGS10"),
                           (d2, "DGS2"), (unr, "UNRATE"))
    nat = (
        pmms.rename({"MORTGAGE30US": "pmms30"})
        .join(d10.rename({"DGS10": "dgs10"}), on="month_ym", how="full", coalesce=True)
        .join(d2.rename({"DGS2": "dgs2"}), on="month_ym", how="full", coalesce=True)
        .join(unr.rename({"UNRATE": "unrate_nat"}), on="month_ym", how="full", coalesce=True)
        .with_columns((pl.col("dgs10") - pl.col("dgs2")).alias("slope_10y2y"))
        .sort("month_ym")
    )
    nat = _reindex_gapfree(nat, ["pmms30", "dgs10", "dgs2", "unrate_nat", "slope_10y2y"],
                           START_YM, end, "macro_national")
    return nat.select(
        pl.col("month_ym").cast(pl.Int32),
        pl.col("pmms30").cast(pl.Float32),
        pl.col("dgs10").cast(pl.Float32),
        pl.col("slope_10y2y").cast(pl.Float32),
        pl.col("unrate_nat").cast(pl.Float32),
    )


def build_state(fred: dict[str, pl.DataFrame], hpi_nat: pl.DataFrame,
                hpi_state: pl.DataFrame) -> pl.DataFrame:
    # Per-state unemployment from the 51 {POSTAL}UR series.
    ur = pl.concat([
        fred[f"{s}UR"].rename({f"{s}UR": "unrate_state"}).with_columns(pl.lit(s).alias("state"))
        for s in STATES
    ])
    real = ur.join(hpi_state, on=["state", "month_ym"], how="full", coalesce=True)

    # Pseudo-state US fallback row carries the national series.
    us = (fred["UNRATE"].rename({"UNRATE": "unrate_state"})
          .join(hpi_nat.rename({"hpi_nat": "hpi_state"}), on="month_ym", how="full", coalesce=True)
          .with_columns(pl.lit("US").alias("state")))

    allrows = pl.concat([real.select("state", "month_ym", "unrate_state", "hpi_state"),
                         us.select("state", "month_ym", "unrate_state", "hpi_state")])

    end = _last_full_month((ur, "unrate_state"), (hpi_state, "hpi_state"),
                           (fred["UNRATE"], "UNRATE"), (hpi_nat, "hpi_nat"))

    # Reindex each state to a gap-free [START, end] span.
    out = []
    for s in STATES + ["US"]:
        sub = allrows.filter(pl.col("state") == s).select("month_ym", "unrate_state", "hpi_state")
        filled = _reindex_gapfree(sub, ["unrate_state", "hpi_state"], START_YM, end,
                                  f"macro_state[{s}]")
        out.append(filled.with_columns(pl.lit(s).alias("state")))

    return (pl.concat(out)
            .select(pl.col("state").cast(pl.Categorical),
                    pl.col("month_ym").cast(pl.Int32),
                    pl.col("unrate_state").cast(pl.Float32),
                    pl.col("hpi_state").cast(pl.Float32))
            .sort("state", "month_ym"))


# ---------------------------------------------------------------------------
# QA spot-checks (§ MD2 Accept)
# ---------------------------------------------------------------------------
def _qa(nat: pl.DataFrame, state: pl.DataFrame) -> None:
    def val(df, flt, col):
        r = df.filter(flt)
        return r[col].item() if r.height == 1 else None

    checks = []
    unr_0910 = val(nat, pl.col("month_ym") == 200910, "unrate_nat")
    checks.append(("UNRATE 2009-10 ≈ 10.0", unr_0910, abs(unr_0910 - 10.0) < 0.15))

    pmms_2101 = val(nat, pl.col("month_ym") == 202101, "pmms30")
    checks.append(("PMMS 2021-01 ∈ [2.6,2.8]", pmms_2101, 2.6 <= pmms_2101 <= 2.8))

    nv = state.filter(pl.col("state") == "NV")
    peak = nv.filter(pl.col("month_ym").is_between(200601, 200712))["hpi_state"].max()
    trough = nv.filter(pl.col("month_ym").is_between(201001, 201212))["hpi_state"].min()
    dd = (trough - peak) / peak
    checks.append(("NV HPI 2006→2012 drawdown ≤ -50%", f"{dd*100:.1f}%", dd <= -0.50))

    print("\nQA spot-checks:")
    ok = True
    for label, got, passed in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:38s} got {got}")
    if not ok:
        raise SystemExit("QA spot-check(s) FAILED")

    # Coverage assertions.
    n_states = state["state"].n_unique()
    assert n_states == 52, f"expected 52 state codes (50 states + DC + US), got {n_states}"
    print(f"  [PASS] macro_state covers {n_states} codes (50 states + DC + US)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the macro Parquet tables (MD2).")
    ap.add_argument("--snapshot", help="snapshot folder under raw/macro/ (default: newest)")
    args = ap.parse_args()

    config.require_drive()
    snap = (config.RAW / "macro" / args.snapshot) if args.snapshot else _newest_snapshot()
    fred_dir = snap / "fred"
    print(f"snapshot: {snap.name}")

    series = ["MORTGAGE30US", "DGS10", "DGS2", "UNRATE"] + [f"{s}UR" for s in STATES]
    fred = {s: _fred_monthly(fred_dir / f"{s}.csv", s) for s in series}
    hpi_nat, hpi_state = _fmhpi(snap / "fmhpi_master_file.csv")

    nat = build_national(fred)
    state = build_state(fred, hpi_nat, hpi_state)

    _qa(nat, state)

    out_dir = config.PROCESSED / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)
    nat_path = out_dir / "macro_national.parquet"
    state_path = out_dir / "macro_state.parquet"
    nat.write_parquet(nat_path, compression="zstd", compression_level=config.ZSTD_LEVEL)
    state.write_parquet(state_path, compression="zstd", compression_level=config.ZSTD_LEVEL)

    def span(df):
        return f"{df['month_ym'].min()}–{df['month_ym'].max()}"
    print(f"\nmacro_national: {nat.height} months ({span(nat)})")
    print(nat.head(3))
    print(nat.tail(2))
    print(f"\nmacro_state: {state.height} rows, "
          f"{state['state'].n_unique()} states, span {span(state)}")
    print(state.head(3))
    print(f"\nwrote {nat_path}\n      {state_path}")


if __name__ == "__main__":
    main()
