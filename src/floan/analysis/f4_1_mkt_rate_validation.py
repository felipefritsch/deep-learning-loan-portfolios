"""F4.1 — Market-rate proxy vs PMMS validation (``01_EDA.md`` §4).

Overlays the panel-derived proxy ``mkt_rate`` (built by ``mkt_rate.py``: average
origination rate of loans originated each month) on the downloaded Freddie Mac PMMS
30-yr series (``macro_national.pmms30``), and reports the tracking error
(`04_TASKS.md` M3 Accept: "proxy-vs-PMMS tracking error reported").

The proxy is a Phase-2 *robustness* alternative; the headline incentive uses PMMS.
Origination rates run a little above the PMMS survey rate (points/credit, the agency
book), so a small positive mean gap with a high correlation and low RMSE is the
expected, reassuring result. Reads only the two tiny time-keyed tables — no panel scan.

Run:  python -m floan.analysis.f4_1_mkt_rate_validation
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from floan.analysis import eda_common as eda

config = eda.config


def build() -> pl.DataFrame:
    config.require_drive()
    mdir = config.PROCESSED / "macro"
    proxy = pl.read_parquet(mdir / "mkt_rate.parquet")
    nat = pl.read_parquet(mdir / "macro_national.parquet").select("month_ym", "pmms30")
    return (proxy.join(nat, on="month_ym", how="inner")
            .filter(pl.col("pmms30").is_not_null() & pl.col("mkt_rate").is_not_null())
            .sort("month_ym")
            .with_columns((pl.col("mkt_rate") - pl.col("pmms30")).alias("gap")))


def _d(ym: int):
    import datetime
    return datetime.date(ym // 100, ym % 100, 1)


def figure(df: pl.DataFrame):
    x = [_d(m) for m in df["month_ym"].to_list()]
    fig, (ax, axg) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x, df["pmms30"], color="tab:blue", lw=1.3, label="PMMS 30-yr (FRED MORTGAGE30US)")
    ax.plot(x, df["mkt_rate"], color="tab:orange", lw=1.1, label="mkt_rate proxy (avg origination rate)")
    ax.set_ylabel("rate (%)")
    ax.set_title("F4.1  Market-rate proxy vs PMMS 30-yr", fontsize=12, loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    axg.axhline(0, color="grey", lw=0.8)
    axg.plot(x, df["gap"], color="tab:red", lw=1.0)
    axg.set_ylabel("proxy − PMMS\n(pp)", fontsize=9)
    axg.set_xlabel("calendar month")
    axg.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    df = build()
    fig = figure(df)
    png, pdf = eda.save_figure(fig, "F4.1_mkt_rate_validation")
    s_csv, s_tex = eda.save_table(
        df.select("month_ym", "mkt_rate", "pmms30", "gap", "ff_flag"),
        "F4.1_mkt_rate_validation_series",
        caption="Monthly mkt_rate proxy vs PMMS30 and their gap (F4.1).",
    )

    gap = df["gap"].to_numpy()
    rmse = float(np.sqrt(np.mean(gap ** 2)))
    mae = float(np.mean(np.abs(gap)))
    mean_gap = float(np.mean(gap))
    corr = float(np.corrcoef(df["mkt_rate"].to_numpy(), df["pmms30"].to_numpy())[0, 1])
    print(f"F4.1 proxy vs PMMS: {df.height} overlapping months "
          f"{df['month_ym'].min()}–{df['month_ym'].max()}")
    print("  --- tracking error (proxy − PMMS) ---")
    print(f"  mean gap : {mean_gap:+.3f} pp   (proxy above PMMS as expected: {mean_gap > 0})")
    print(f"  RMSE     : {rmse:.3f} pp")
    print(f"  MAE      : {mae:.3f} pp")
    print(f"  std gap  : {gap.std():.3f} pp")
    print(f"  correlation(level): {corr:.4f}")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
