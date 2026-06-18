"""T2.1 — Empirical transition matrix, full panel pooled over time (``01_EDA.md`` §2).

The dependent variable, described once on the whole panel: the 4 transient origin
states (``current, dpd_30, dpd_60, dpd_90plus``) × 7 destination states, as raw
one-month transition counts and as row-normalized frequencies. This is *also* the
Phase-2 benchmark in shape, but here it is computed on the full panel for
description; Phase 2 (M5) recomputes it on the training slice only, with Laplace
smoothing, for honest benchmarking.

Honest conditional distribution: only rows with an *observed* next month enter
(``state_next IS NOT NULL``). Right-censored rows and absorbing terminal rows have
``state_next = NULL`` and are excluded, so each origin row is a proper conditional
distribution that sums to 1. The three absorbing states never appear as an origin
(they have no transitions out), hence 4 origin rows, not 7.

Run:  python dev/analysis/t2_1_transition_matrix.py
"""

from __future__ import annotations

import polars as pl

from floan.analysis import eda_common as eda

config, schema = eda.config, eda.schema

# The seven destination states in canonical order; the 4 transient origins.
DEST = list(schema.STATES)
ORIGIN = [s for s in schema.STATES if s not in ("foreclosure", "REO", "prepaid")]


def build(con) -> tuple[pl.DataFrame, pl.DataFrame]:
    long = con.execute(
        """
        SELECT state, state_next, count(*) AS n
        FROM panel
        WHERE state IN ('current','dpd_30','dpd_60','dpd_90plus')
          AND state_next IS NOT NULL
        GROUP BY 1, 2
        """
    ).pl()

    # Wide counts: one row per origin, one column per destination (0-filled).
    counts = (
        long.pivot(values="n", index="state", on="state_next", aggregate_function="sum")
        .rename({"state": "origin"})
    )
    for d in DEST:                      # ensure every destination column exists
        if d not in counts.columns:
            counts = counts.with_columns(pl.lit(0).alias(d))
    counts = (
        counts.with_columns([pl.col(d).fill_null(0).cast(pl.Int64) for d in DEST])
        .select(["origin", *DEST])
        .sort(pl.col("origin").replace_strict({s: i for i, s in enumerate(ORIGIN)}))
    )

    # Row-normalized probabilities (each origin row conditional on having moved).
    row_tot = pl.sum_horizontal([pl.col(d) for d in DEST])
    probs = counts.with_columns([(pl.col(d) / row_tot).alias(d) for d in DEST])
    return counts, probs


def main() -> None:
    config.require_drive()
    con = eda.connect()
    counts, probs = build(con)
    con.close()

    c_csv, c_tex = eda.save_table(
        counts, "T2.1_transition_counts",
        caption="Empirical one-month transition counts, full panel (T2.1).",
    )
    p_csv, p_tex = eda.save_table(
        probs, "T2.1_transition_probs",
        caption="Empirical one-month transition probabilities, full panel; "
                "rows (transient origins) sum to 1 (T2.1).",
    )

    row_sums = (
        probs.select(["origin", *DEST])
        .with_columns(pl.sum_horizontal([pl.col(d) for d in DEST]).alias("row_sum"))
        .select("origin", "row_sum")
    )
    max_dev = (row_sums["row_sum"] - 1.0).abs().max()

    pl.Config.set_tbl_rows(10)
    pl.Config.set_tbl_width_chars(200)
    print("T2.1 empirical transition matrix (4 transient origins × 7 destinations)")
    print("\nrow-normalized probabilities:")
    print(probs.with_columns([pl.col(d).round(5) for d in DEST]))
    print("\nrow sums (Accept: each = 1):")
    print(row_sums.with_columns(pl.col("row_sum").round(10)))
    ok = max_dev < 1e-9
    print(f"\nmax |row_sum - 1| = {max_dev:.2e}  "
          f"{'✓ rows sum to 1' if ok else '✗ rows DO NOT sum to 1'}")
    print(f"\nwrote {c_csv}\n      {c_tex}\n      {p_csv}\n      {p_tex}")


if __name__ == "__main__":
    main()
