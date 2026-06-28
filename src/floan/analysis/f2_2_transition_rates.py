"""F2.2 — Transition rates over calendar time (``01_EDA.md`` §2).

Monthly one-month transition rates, 2000–2025, for the three series that carry the
regime story:

  * ``current → prepaid``                — refinancing / turnover
  * ``current → dpd_30``                 — fresh delinquency (default onset)
  * ``dpd_90plus → foreclosure | REO``   — terminal-default conversion

Each rate is conditional on an *observed* next month: numerator / count of that
origin state with ``state_next IS NOT NULL`` in the month. The final calendar
month is all-censored (no next month observed) so it has a zero denominator and is
dropped. This is the sanity check that the seven-state target derivation is right:
the 2003 and 2020–21 refi waves and the 2008–11 default wave must be visible
(`04_TASKS.md` M2 Accept), and the 2022–23 prepay collapse should appear too.

Run:  python -m floan.analysis.f2_2_transition_rates
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from floan.analysis import eda_common as eda

config = eda.config


def build(con) -> pl.DataFrame:
    df = con.execute(
        """
        SELECT period_ym,
          count(*) FILTER (WHERE state='current'    AND state_next IS NOT NULL) AS cur_obs,
          count(*) FILTER (WHERE state='dpd_90plus' AND state_next IS NOT NULL) AS d90_obs,
          count(*) FILTER (WHERE state='current'    AND state_next='prepaid')   AS cur_prepaid,
          count(*) FILTER (WHERE state='current'    AND state_next='dpd_30')    AS cur_dpd30,
          count(*) FILTER (WHERE state='dpd_90plus' AND state_next IN ('foreclosure','REO')) AS d90_fcl
        FROM panel
        GROUP BY 1
        """
    ).pl()

    return (
        df.filter((pl.col("cur_obs") > 0) & (pl.col("d90_obs") > 0))
        .sort("period_ym")
        .with_columns(
            pl.date(pl.col("period_ym") // 100, pl.col("period_ym") % 100, 1).alias("date"),
            (pl.col("cur_prepaid") / pl.col("cur_obs")).alias("rate_cur_prepaid"),
            (pl.col("cur_dpd30") / pl.col("cur_obs")).alias("rate_cur_dpd30"),
            (pl.col("d90_fcl") / pl.col("d90_obs")).alias("rate_d90_fcl"),
        )
    )


def figure(df: pl.DataFrame):
    x = df["date"].to_list()
    panels = [
        ("rate_cur_prepaid", "current → prepaid", "tab:green"),
        ("rate_cur_dpd30", "current → dpd_30", "tab:orange"),
        ("rate_d90_fcl", "dpd_90+ → foreclosure / REO", "tab:red"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, (col, title, color) in zip(axes, panels):
        ax.plot(x, [v * 100 for v in df[col].to_list()], color=color, lw=1.0)
        ax.axvspan(_d(2007, 12), _d(2011, 12), color="grey", alpha=0.10, lw=0)  # GFC default wave
        ax.axvspan(_d(2020, 3), _d(2021, 12), color="grey", alpha=0.10, lw=0)   # COVID forbearance episode
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("% / month", fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("calendar month")
    fig.suptitle("F2.2  Monthly transition rates, 2000–2025", fontsize=12)
    fig.tight_layout()
    return fig


def _d(y: int, m: int):
    import datetime
    return datetime.date(y, m, 1)


def main() -> None:
    config.require_drive()
    con = eda.connect()
    df = build(con)
    con.close()

    series = df.select(
        "period_ym", "rate_cur_prepaid", "rate_cur_dpd30", "rate_d90_fcl"
    )
    s_csv, s_tex = eda.save_table(
        series, "F2.2_transition_rates_series",
        caption="Monthly transition rates underlying F2.2.",
    )
    fig = figure(df)
    png, pdf = eda.save_figure(fig, "F2.2_transition_rates")

    # Evidence: the peak month of each headline episode.
    def peak(col, lo, hi):
        w = df.filter(pl.col("period_ym").is_between(lo, hi))
        r = w.sort(col, descending=True).head(1)
        return r["period_ym"].item(), r[col].item() * 100

    p2003 = peak("rate_cur_prepaid", 200201, 200412)
    p2020 = peak("rate_cur_prepaid", 202001, 202112)
    collapse = (
        df.filter(pl.col("period_ym").is_between(202201, 202312))
        .sort("rate_cur_prepaid").head(1)
    )
    d2009 = peak("rate_cur_dpd30", 200801, 201012)
    fcl_gfc = peak("rate_d90_fcl", 200801, 201112)

    print(f"F2.2 monthly transition rates: {df.height} months "
          f"{df['period_ym'].min()}–{df['period_ym'].max()}")
    print(f"  current→prepaid 2003 refi-wave peak : {p2003[0]}  {p2003[1]:.2f}%/mo")
    print(f"  current→prepaid 2020 refi-wave peak : {p2020[0]}  {p2020[1]:.2f}%/mo")
    print(f"  current→prepaid 2022–23 trough      : {collapse['period_ym'].item()}  "
          f"{collapse['rate_cur_prepaid'].item()*100:.2f}%/mo  (prepay collapse)")
    print(f"  current→dpd_30 GFC peak             : {d2009[0]}  {d2009[1]:.2f}%/mo")
    print(f"  dpd_90+→fcl/REO GFC peak            : {fcl_gfc[0]}  {fcl_gfc[1]:.2f}%/mo")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
