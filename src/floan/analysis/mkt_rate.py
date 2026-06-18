"""Panel-derived market-rate proxy → ``processed/macro/mkt_rate.parquet`` (``01_EDA.md`` §4).

``mkt_rate(t) = avg(Original Interest Rate) over loans originated in month t`` — a
proxy for the prevailing 30-yr mortgage rate, built from the loan panel itself and
kept as a Phase-2 robustness alternative to the downloaded PMMS series (F4.1
validates one against the other). Indexed by calendar month (``month_ym``) so it
joins exactly like ``macro_national`` (PMMS lag 0): the proxy incentive in Phase 2
is ``current_rate − mkt_rate(period_ym)``.

The average is over **loans**, not loan-months: ``Original Interest Rate`` and
``orig_ym`` are loan-constant, so we first reduce the 3.3 B-row panel to one row per
loan (a DuckDB ``GROUP BY`` on the loan id, spilled out-of-core), then average by
origination month. Months with thin origination counts (the panel edges — the
pre-2000 originations that only appear because they were still alive in 2000-01, and
the newest vintage's trailing months) give noisy means, so they are forward-filled
(back-filled for any leading thin months) from the nearest well-populated month and
flagged (``ff_flag``). The raw mean and loan count are retained for transparency.

Run:  python dev/analysis/mkt_rate.py
"""

from __future__ import annotations

import polars as pl

from floan.analysis import eda_common as eda

config = eda.config

# A month is "thin" (mean unreliable → fill) below this loan count. Set from the
# observed distribution: the 2000–2025 interior months carry tens of thousands of
# originations each; only the pre-panel (1999) and newest trailing months fall here.
THIN_MIN_LOANS = 1_000


def build(con) -> pl.DataFrame:
    """One row per origination month: raw mean rate, loan count, filled rate, flag."""
    raw = con.execute(
        """
        SELECT orig_ym AS month_ym,
               avg(orig_rate)::DOUBLE AS mkt_rate_raw,
               count(*)               AS n_loans
        FROM (
            SELECT "Loan Identifier"          AS loan,
                   min("Original Interest Rate") AS orig_rate,  -- loan-constant
                   min(orig_ym)                  AS orig_ym
            FROM panel
            WHERE "Original Interest Rate" IS NOT NULL
            GROUP BY "Loan Identifier"
        )
        GROUP BY orig_ym
        ORDER BY orig_ym
        """
    ).pl()

    # Reindex onto a gap-free monthly grid, then forward- then back-fill the rate
    # over thin/absent months so the proxy is defined for every origination month.
    lo, hi = int(raw["month_ym"].min()), int(raw["month_ym"].max())
    grid = pl.DataFrame({"month_ym": _month_grid(lo, hi)})
    df = grid.join(raw, on="month_ym", how="left").sort("month_ym")
    thin = pl.col("n_loans").is_null() | (pl.col("n_loans") < THIN_MIN_LOANS)
    df = df.with_columns(
        thin.alias("ff_flag"),
        pl.col("n_loans").fill_null(0).cast(pl.Int64),
        pl.when(thin).then(None).otherwise(pl.col("mkt_rate_raw")).alias("_good"),
    )
    df = df.with_columns(
        pl.col("_good").forward_fill().backward_fill().alias("mkt_rate")
    ).drop("_good")
    return df.select(
        pl.col("month_ym").cast(pl.Int32),
        pl.col("mkt_rate").cast(pl.Float32),
        pl.col("mkt_rate_raw").cast(pl.Float32),
        pl.col("n_loans").cast(pl.Int32),
        pl.col("ff_flag"),
    )


def _month_grid(lo_ym: int, hi_ym: int) -> list[int]:
    out, y, m = [], lo_ym // 100, lo_ym % 100
    while y * 100 + m <= hi_ym:
        out.append(y * 100 + m)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def main() -> None:
    config.require_drive()
    con = eda.connect()
    df = build(con)
    con.close()

    out_path = config.PROCESSED / "macro" / "mkt_rate.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    # CSV copy alongside the EDA tables for inspection / the memo.
    eda.save_table(df, "mkt_rate_proxy",
                   caption="Panel-derived market-rate proxy by origination month "
                           "(avg origination rate over loans); thin months filled.")

    n_thin = int(df["ff_flag"].sum())
    gapfree = (df["month_ym"].diff().drop_nulls().is_in([1, 89])).all()  # 1 within yr, 89 across yr
    print(f"mkt_rate proxy: {df.height} months {df['month_ym'].min()}–{df['month_ym'].max()}  "
          f"(gap-free monthly grid: {gapfree})")
    print(f"  filled (thin, n_loans < {THIN_MIN_LOANS:,}) months: {n_thin}")
    print("  thin months (filled):")
    for r in df.filter(pl.col("ff_flag")).iter_rows(named=True):
        raw = "(none)" if r["mkt_rate_raw"] is None else f"{r['mkt_rate_raw']:.3f}"
        print(f"    {r['month_ym']}  n_loans={r['n_loans']:>7,}  "
              f"raw={raw}  -> filled {r['mkt_rate']:.3f}")
    good = df.filter(~pl.col("ff_flag"))
    print(f"  well-populated months: {good.height}  "
          f"mkt_rate range {good['mkt_rate'].min():.2f}–{good['mkt_rate'].max():.2f}%  "
          f"median n_loans {int(good['n_loans'].median()):,}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
