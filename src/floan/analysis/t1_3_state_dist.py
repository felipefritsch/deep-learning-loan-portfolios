"""T1.3 — State distribution (``01_EDA.md`` §1).

Loan-month counts and shares by ``state``, overall and by calendar year. This
quantifies the class imbalance (``current`` is ~95%+ of rows) that motivates the
stratified sampling and importance weighting used in Phase 2. Null ``state``
(XX/blank, unusable label) is shown as the level ``(null)``.

Run:  python dev/analysis/t1_3_state_dist.py
"""

from __future__ import annotations

import polars as pl

import eda_common as eda

config, schema = eda.config, eda.schema

# Canonical display order: the seven states, then null last.
_ORDER = {s: i for i, s in enumerate(schema.STATES)}
_ORDER["(null)"] = len(schema.STATES)


def _order_expr() -> pl.Expr:
    return pl.col("state").replace_strict(_ORDER, default=len(_ORDER))


def build(con) -> tuple[pl.DataFrame, pl.DataFrame]:
    total = con.execute("SELECT count(*) FROM panel").fetchone()[0]

    overall = (
        con.execute("SELECT state, count(*) AS n FROM panel GROUP BY state").pl()
        .with_columns(pl.col("state").fill_null("(null)"))
        .with_columns((pl.col("n") / total).alias("share"))
        .sort(_order_expr())
        .select("state", "n", "share")
    )

    by_year = (
        con.execute(
            "SELECT period_ym // 100 AS year, state, count(*) AS n "
            "FROM panel GROUP BY 1, 2"
        ).pl()
        .with_columns(pl.col("state").fill_null("(null)"))
        .with_columns((pl.col("n") / pl.col("n").sum().over("year")).alias("share"))
        .sort(["year", _order_expr()])
        .select("year", "state", "n", "share")
    )
    return overall, by_year


def main() -> None:
    config.require_drive()
    con = eda.connect()
    overall, by_year = build(con)
    con.close()

    o_csv, o_tex = eda.save_table(
        overall, "T1.3_state_distribution_overall",
        caption="State distribution over all loan-months (T1.3).",
    )
    y_csv, y_tex = eda.save_table(
        by_year, "T1.3_state_distribution_by_year",
        caption="State distribution by calendar year (T1.3).",
    )

    cur = overall.filter(pl.col("state") == "current")["share"].item()
    cur_by_year = by_year.filter(pl.col("state") == "current")
    print("T1.3 state distribution (overall):")
    print(overall.with_columns((100 * pl.col("share")).round(4).alias("pct")))
    print(f"\n`current` share overall: {100 * cur:.2f}%  "
          f"({'>= 95% dominance ✓' if cur >= 0.95 else 'BELOW 95% — check ✗'})")
    print(f"`current` share by year: min {100 * cur_by_year['share'].min():.2f}% / "
          f"max {100 * cur_by_year['share'].max():.2f}% "
          f"across {cur_by_year.height} years")
    print(f"\nwrote {o_csv}\n      {o_tex}\n      {y_csv}\n      {y_tex}")


if __name__ == "__main__":
    main()
