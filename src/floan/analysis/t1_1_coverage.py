"""T1.1 — Panel coverage table (``01_EDA.md`` §1).

Per acquisition vintage: loan count, loan-month count, first/last reporting
period, and % of loans right-censored at the data cut-off. Loan counts come from
the Stage-1 manifest (loans are vintage-disjoint, so ``distinct_loans`` is exact);
loan-months / period span / censoring come from a single ``GROUP BY acq_quarter``
over the panel. We assert manifest-rows == panel-loan-months (reconciliation) and
that the last reporting period is uniformly the release cut-off.

``% censored`` = right-censored loans / loans. ``censored`` is TRUE only on a
loan's final observed month when that month is in a non-absorbing state (see
``s4_panel.py``), so ``count(*) FILTER (WHERE censored)`` counts censored loans.

Run:  python -m floan.analysis.t1_1_coverage
"""

from __future__ import annotations

import polars as pl

from floan.analysis import eda_common as eda

config = eda.config


def _ym_to_str(col: str) -> pl.Expr:
    return (
        (pl.col(col) // 100).cast(pl.Utf8)
        + "-"
        + (pl.col(col) % 100).cast(pl.Utf8).str.zfill(2)
    )


def build() -> pl.DataFrame:
    config.require_drive()

    manifest = (
        pl.read_csv(config.OUTPUTS / "inventory_manifest.csv")
        .filter(~pl.col("quarantined"))
        .select(
            pl.col("acq_quarter"),
            pl.col("distinct_loans").alias("n_loans"),
            pl.col("rows").alias("manifest_rows"),
        )
    )

    con = eda.connect()
    panel = con.execute(
        """
        SELECT acq_quarter,
               count(*)                          AS n_loan_months,
               count(*) FILTER (WHERE censored)  AS n_censored_loans,
               min(period_ym)                    AS first_period_ym,
               max(period_ym)                    AS last_period_ym
        FROM panel
        GROUP BY acq_quarter
        ORDER BY acq_quarter
        """
    ).pl()
    con.close()

    df = manifest.join(panel, on="acq_quarter", how="inner").sort("acq_quarter")

    # Reconciliation: panel loan-months must equal the manifest's raw row count.
    bad = df.filter(pl.col("manifest_rows") != pl.col("n_loan_months"))
    if bad.height:
        raise SystemExit(
            f"Row reconciliation FAILED for {bad.height} vintage(s):\n{bad}"
        )

    df = df.with_columns(
        (100.0 * pl.col("n_censored_loans") / pl.col("n_loans")).alias("pct_censored"),
        _ym_to_str("first_period_ym").alias("first_period"),
        _ym_to_str("last_period_ym").alias("last_period"),
    ).select(
        "acq_quarter", "n_loans", "n_loan_months",
        "first_period", "last_period", "n_censored_loans", "pct_censored",
    )
    return df


def main() -> None:
    df = build()

    cutoffs = df["last_period"].unique().to_list()
    csv_path, tex_path = eda.save_table(
        df, "T1.1_coverage",
        caption="Panel coverage by acquisition vintage (T1.1).",
    )

    print(f"T1.1 coverage: {df.height} vintages")
    print(f"  total loans      : {df['n_loans'].sum():,}")
    print(f"  total loan-months: {df['n_loan_months'].sum():,}")
    print(f"  release cut-off  : {cutoffs} "
          f"{'-> uniform 2025-12 ✓' if cutoffs == ['2025-12'] else '-> NOT UNIFORM ✗'}")
    print(f"  % censured range : {df['pct_censored'].min():.2f}% – {df['pct_censored'].max():.2f}%")
    print("\nhead:")
    print(df.head(4))
    print("tail:")
    print(df.tail(4))
    print(f"\nwrote {csv_path}\n      {tex_path}")


if __name__ == "__main__":
    main()
