"""F2.3 — COVID-era delinquency-code check (``01_EDA.md`` §2; M2 Accept).

Distribution of raw ``Current Loan Delinquency Status`` codes by month over
2019-06 → 2022-06. CARES-Act forbearance shows up as an abrupt mid-2020 jump into
the shallow delinquency buckets (1–3) that then *persists and deepens* into the 4,
5, 6+ buckets through 2020–21 — borrowers paused payments without curing — and
later resolves as a wave of "cures" back to current. Plotted as a stacked area of
the delinquent buckets (status ≥ 1) as a share of all active loan-months, so the
forbearance bulge is legible against the ~95% `current` mass.

COVID handling decision (M2 Accept — documented here and carried to memo 01):
  **No special handling / no reclassification.** Forbearance loan-months flow
  through the seven-state machine exactly as reported: a paused payment increments
  `Current Loan Delinquency Status`, so the loan rolls current→dpd_30→… per the
  normal map, and the later forbearance exit appears as ordinary dpd→current
  "cure" transitions. We do not relabel these months because (a) the public SF
  layout carries no clean forbearance/borrower-assistance flag for the bulk of the
  window, so any reclassification would be a heuristic, and (b) the regime is what
  the rolling backtest is meant to expose — the 2020–21 window is one of the key
  evaluation windows, and the model should learn it from the macro covariates
  rather than have it hand-removed. The writeup must acknowledge that 2020–21
  delinquency is forbearance-dominated (not economic default), which inflates the
  current→dpd onset rate and the dpd→current cure rate in that window (visible in
  F2.2 / F2.4); the empirical-matrix benchmark in that window absorbs the same
  artifact, so model-vs-benchmark comparison stays fair.

Run:  python dev/analysis/f2_3_covid_check.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import eda_common as eda

config = eda.config

WINDOW = (201906, 202206)
# Delinquency buckets in plot order (shallow → deep). 0 = current is the baseline
# mass and is shown as the denominator, not a band.
BUCKETS = ["1", "2", "3", "4", "5", "6+", "XX/null"]


def build(con) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = con.execute(
        f"""
        SELECT period_ym,
               TRY_CAST(NULLIF(trim("Current Loan Delinquency Status"), '') AS INTEGER) AS dq,
               count(*) AS n
        FROM panel
        WHERE period_ym BETWEEN {WINDOW[0]} AND {WINDOW[1]}
        GROUP BY 1, 2
        """
    ).pl()

    bucketed = raw.with_columns(
        pl.when(pl.col("dq").is_null()).then(pl.lit("XX/null"))
        .when(pl.col("dq") <= 0).then(pl.lit("0"))
        .when(pl.col("dq") >= 6).then(pl.lit("6+"))
        .otherwise(pl.col("dq").cast(pl.Utf8))
        .alias("bucket")
    )

    monthly_total = bucketed.group_by("period_ym").agg(pl.col("n").sum().alias("total"))
    shares = (
        bucketed.group_by("period_ym", "bucket").agg(pl.col("n").sum().alias("n"))
        .join(monthly_total, on="period_ym")
        .with_columns((pl.col("n") / pl.col("total")).alias("share"))
        .sort("period_ym")
    )

    # Wide share table (months × buckets) for the figure and CSV.
    wide = (
        shares.pivot(values="share", index="period_ym", on="bucket", aggregate_function="sum")
        .sort("period_ym")
    )
    for b in ["0", *BUCKETS]:
        if b not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(b))
    wide = wide.with_columns([pl.col(b).fill_null(0.0) for b in ["0", *BUCKETS]])
    wide = wide.select(["period_ym", "0", *BUCKETS])
    return shares, wide


def figure(wide: pl.DataFrame):
    x = [_d(p // 100, p % 100) for p in wide["period_ym"].to_list()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ys = [[v * 100 for v in wide[b].to_list()] for b in BUCKETS]
    ax.stackplot(x, *ys, labels=[f"dpd {b}" if b not in ("XX/null",) else b for b in BUCKETS],
                 colors=plt.cm.YlOrRd(np.linspace(0.25, 0.95, len(BUCKETS))))
    ax.axvspan(_d(2020, 3), _d(2020, 6), color="steelblue", alpha=0.15, lw=0)
    ax.annotate("CARES-Act\nforbearance onset", xy=(_d(2020, 5), 1.0),
                xytext=(_d(2020, 8), 3.2), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="steelblue"))
    ax.set_title("F2.3  Delinquency-status mix, 2019-06 → 2022-06 "
                 "(share of active loans, status ≥ 1)", fontsize=11, loc="left")
    ax.set_ylabel("% of active loan-months")
    ax.set_xlabel("calendar month")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _d(y: int, m: int):
    import datetime
    return datetime.date(y, m, 1)


def main() -> None:
    config.require_drive()
    con = eda.connect()
    shares, wide = build(con)
    con.close()

    w_csv, w_tex = eda.save_table(
        wide, "F2.3_delinquency_mix",
        caption="Monthly delinquency-status mix, 2019-06 → 2022-06 (F2.3).",
    )
    fig = figure(wide)
    png, pdf = eda.save_figure(fig, "F2.3_covid_delinquency")

    # Evidence: total delinquent (status ≥ 1) share, pre-COVID vs the 2020 peak.
    delinq = wide.with_columns(
        pl.sum_horizontal([pl.col(b) for b in BUCKETS if b != "XX/null"]).alias("delinq_share")
    )
    pre = delinq.filter(pl.col("period_ym") == 202002)["delinq_share"].item()
    peak_row = delinq.filter(pl.col("period_ym").is_between(202004, 202103)).sort(
        "delinq_share", descending=True).head(1)
    deep = wide.filter(pl.col("period_ym").is_between(202004, 202112)).sort(
        "6+", descending=True).head(1)
    print("F2.3 COVID delinquency-code check (2019-06 → 2022-06)")
    print(f"  delinquent share (status≥1) 2020-02 (pre)  : {pre*100:.2f}%")
    print(f"  delinquent share peak                       : "
          f"{peak_row['period_ym'].item()}  {peak_row['delinq_share'].item()*100:.2f}%  "
          f"(×{peak_row['delinq_share'].item()/pre:.1f} the pre-COVID level)")
    print(f"  deepest '6+' (≥6 mo missed) share, peak     : "
          f"{deep['period_ym'].item()}  {deep['6+'].item()*100:.2f}%  "
          f"(persistence, not cures → forbearance)")
    print(f"  handling decision: NO reclassification (see script header)")
    print(f"\nwrote {png}\n      {pdf}\n      {w_csv}\n      {w_tex}")


if __name__ == "__main__":
    main()
