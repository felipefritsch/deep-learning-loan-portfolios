"""F2.4 — Cure/roll areas (``01_EDA.md`` §2).

For each of the 4 transient origin states, the monthly destination mix
(``state_next`` shares, conditional on an observed next month) as a stacked area,
2000–2025. One panel per origin; within a panel the self-transition (stay-in-state)
is drawn as the top band so the minority cure/roll/exit bands sit on the axis where
they are legible. Each panel's bands sum to 1.

Reads as the cyclicality of the roll-rate machine: the ``current`` panel's prepaid
band tracks the refi waves (cross-check on F2.2); the ``dpd_*`` panels show
cure (roll-down to a lower bucket / back to current) vs roll-forward
(to a deeper bucket, then foreclosure/REO) shifting with the cycle — cures collapse
and roll-forward swells through 2008–11, and the 2020–21 forbearance bulge cures
out afterwards.

Run:  python dev/analysis/f2_4_roll_rates.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from floan.analysis import eda_common as eda

config, schema = eda.config, eda.schema

ORIGIN = ["current", "dpd_30", "dpd_60", "dpd_90plus"]
DEST = list(schema.STATES)
# One stable colour per destination state, shared across panels and the legend.
DEST_COLOR = {
    "current": "#4daf4a", "dpd_30": "#ffd92f", "dpd_60": "#fdae61",
    "dpd_90plus": "#e6550d", "foreclosure": "#a50f15", "REO": "#6a51a3",
    "prepaid": "#377eb8",
}


def build(con) -> pl.DataFrame:
    long = con.execute(
        """
        SELECT period_ym, state, state_next, count(*) AS n
        FROM panel
        WHERE state IN ('current','dpd_30','dpd_60','dpd_90plus')
          AND state_next IS NOT NULL
        GROUP BY 1, 2, 3
        """
    ).pl()
    return long.with_columns(
        (pl.col("n") / pl.col("n").sum().over("period_ym", "state")).alias("share")
    ).sort("period_ym")


def _origin_wide(long: pl.DataFrame, origin: str) -> pl.DataFrame:
    w = (
        long.filter(pl.col("state") == origin)
        .pivot(values="share", index="period_ym", on="state_next", aggregate_function="sum")
        .sort("period_ym")
    )
    for d in DEST:
        if d not in w.columns:
            w = w.with_columns(pl.lit(0.0).alias(d))
    return w.with_columns([pl.col(d).fill_null(0.0) for d in DEST]).select(["period_ym", *DEST])


def figure(long: pl.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, origin in zip(axes.ravel(), ORIGIN):
        w = _origin_wide(long, origin)
        x = [_d(p // 100, p % 100) for p in w["period_ym"].to_list()]
        # Stack minority destinations first (bottom), self-transition last (top).
        order = [d for d in DEST if d != origin] + [origin]
        ys = [[v * 100 for v in w[d].to_list()] for d in order]
        ax.stackplot(x, *ys, colors=[DEST_COLOR[d] for d in order], labels=order)
        ax.set_title(f"origin: {origin}", fontsize=10, loc="left")
        ax.set_ylim(0, 100)
        ax.set_ylabel("destination share (%)", fontsize=9)
        ax.grid(True, alpha=0.2)
    # Single shared legend in canonical destination order.
    handles = [plt.Rectangle((0, 0), 1, 1, color=DEST_COLOR[d]) for d in DEST]
    fig.legend(handles, DEST, ncol=7, loc="lower center", fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("F2.4  Destination mix by origin state, 2000–2025 (cure / roll areas)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def _d(y: int, m: int):
    import datetime
    return datetime.date(y, m, 1)


def main() -> None:
    config.require_drive()
    con = eda.connect()
    long = build(con)
    con.close()

    series = long.select("period_ym", "state", "state_next", "share").rename(
        {"state": "origin", "state_next": "destination"})
    s_csv, s_tex = eda.save_table(
        series, "F2.4_roll_rate_series",
        caption="Monthly destination shares by origin state underlying F2.4.",
    )
    fig = figure(long)
    png, pdf = eda.save_figure(fig, "F2.4_roll_rates")

    # Evidence: dpd_30 cure (→ down/current) vs roll-forward (→ dpd_60), GFC vs calm.
    cure = long.filter((pl.col("state") == "dpd_30") & (pl.col("state_next") == "current"))
    roll = long.filter((pl.col("state") == "dpd_30") & (pl.col("state_next") == "dpd_60"))

    def at(df, lo, hi):
        return df.filter(pl.col("period_ym").is_between(lo, hi))["share"].mean() * 100

    print("F2.4 cure/roll areas — dpd_30 origin, mean monthly share:")
    print(f"  → current (cure)     calm 2017 {at(cure,201701,201712):.1f}%   "
          f"GFC 2009 {at(cure,200901,200912):.1f}%")
    print(f"  → dpd_60 (roll-fwd)  calm 2017 {at(roll,201701,201712):.1f}%   "
          f"GFC 2009 {at(roll,200901,200912):.1f}%")
    print(f"  (cures fall / roll-forward rises in the crisis → cyclicality)")
    # Per-panel band sums = 1 sanity.
    sums = (
        long.group_by("period_ym", "state").agg(pl.col("share").sum().alias("s"))
    )
    max_dev = (sums["s"] - 1.0).abs().max()
    print(f"  per-(month,origin) destination shares max |sum-1| = {max_dev:.2e}")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
