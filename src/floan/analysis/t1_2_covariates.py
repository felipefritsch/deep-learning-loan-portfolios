"""T1.2 — Covariate summary (``01_EDA.md`` §1).

Two tables:
  * continuous — mean / std / p1 / median / p99 / null-rate for every
    ``feature_spec`` continuous column. This is a **re-presentation** of the
    Stage-6 QA output (``outputs/feature_stats.csv``); the spec says not to
    recompute it from scratch when ``outputs/`` already has it.
  * categorical — level counts (and shares) for every ``feature_spec``
    categorical column. Not produced by Stage 6, so computed here with one
    ``GROUP BY`` per column over the panel (nulls shown as the level ``(null)``).

Run:  python dev/analysis/t1_2_covariates.py
"""

from __future__ import annotations

import polars as pl

from floan.analysis import eda_common as eda

config, schema = eda.config, eda.schema


def continuous_table() -> pl.DataFrame:
    """Re-present outputs/feature_stats.csv in the order the spec lists."""
    stats = pl.read_csv(config.OUTPUTS / "feature_stats.csv")
    return stats.select(
        "column", "null_rate", "mean", "std",
        "min", "p01", "p25", "median", "p75", "p99", "max", "log_flagged",
    )


def categorical_table(con) -> pl.DataFrame:
    total = con.execute("SELECT count(*) FROM panel").fetchone()[0]
    frames = []
    for col in schema.FEATURE_SPEC["categorical"]:
        q = '"' + col.replace('"', '""') + '"'
        df = con.execute(
            f"SELECT CAST({q} AS VARCHAR) AS level, count(*) AS n "
            f"FROM panel GROUP BY {q}"
        ).pl()
        df = (
            df.with_columns(
                pl.lit(col).alias("column"),
                pl.col("level").fill_null("(null)"),
            )
            .sort("n", descending=True)
        )
        frames.append(df)
    out = pl.concat(frames).with_columns(
        (pl.col("n") / total).alias("share")
    )
    return out.select("column", "level", "n", "share")


def main() -> None:
    config.require_drive()

    cont = continuous_table()
    c_csv, c_tex = eda.save_table(
        cont, "T1.2_covariate_summary_continuous",
        caption="Continuous covariate summary (T1.2); re-presented from Stage-6 QA.",
    )

    con = eda.connect()
    cat = categorical_table(con)
    con.close()
    k_csv, k_tex = eda.save_table(
        cat, "T1.2_categorical_levels",
        caption="Categorical covariate level counts (T1.2).",
    )

    print(f"T1.2 continuous: {cont.height} columns "
          f"(feature_spec continuous_standardize = "
          f"{len(schema.FEATURE_SPEC['continuous_standardize'])})")
    print(cont)
    n_lvls = cat.group_by("column").len().sort("column")
    print(f"\nT1.2 categorical: {cat.height} (column, level) rows across "
          f"{cat['column'].n_unique()} columns; levels per column:")
    print(n_lvls)
    print("\ntop levels per low-cardinality column:")
    print(cat.filter(pl.col("column").is_in(
        ["Channel", "Loan Purpose", "Property Type", "Occupancy Status",
         "Amortization Type"])))
    print(f"\nwrote {c_csv}\n      {c_tex}\n      {k_csv}\n      {k_tex}")


if __name__ == "__main__":
    main()
