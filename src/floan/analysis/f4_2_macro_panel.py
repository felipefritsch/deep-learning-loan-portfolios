"""F4.2 — Macro panel small-multiples (``01_EDA.md`` §4).

Four panels over 2000–2025, read from the macro tables (``processed/macro/``), not the
loan panel: the 30-yr mortgage rate (PMMS), the 10-yr Treasury, national unemployment
with the cross-state inter-quartile band, and the national FMHPI (rebased 2000-01=100)
with the cross-state p10–p90 dispersion band. Doubles as the writeup's macro-context
exhibit. The shapes must match known history (MD4 Accept): the 2008–09 & 2020
unemployment spikes, the 2012 & 2021 mortgage-rate troughs, and the 2006→12 HPI bust.

National HPI is the FMHPI ``US`` row carried in ``macro_state``; state dispersion
excludes that pseudo-state. HPI is rebased per state to 2000-01=100 so cumulative
growth is comparable across states (raw index *levels* are not).

Run:  python dev/analysis/f4_2_macro_panel.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from floan.analysis import eda_common as eda

config = eda.config

LO, HI = 200001, 202512  # panel era for the exhibit


def _d(ym: int):
    import datetime
    return datetime.date(ym // 100, ym % 100, 1)


def build() -> pl.DataFrame:
    config.require_drive()
    mdir = config.PROCESSED / "macro"
    nat = pl.read_parquet(mdir / "macro_national.parquet")
    state = pl.read_parquet(mdir / "macro_state.parquet").with_columns(pl.col("state").cast(pl.Utf8))

    real = state.filter(pl.col("state") != "US")
    us = state.filter(pl.col("state") == "US")

    # Rebase each HPI series to its 2000-01 value = 100 (comparable cumulative growth).
    def rebase(df):
        base = df.filter(pl.col("month_ym") == LO).select("state", pl.col("hpi_state").alias("_b"))
        return df.join(base, on="state").with_columns(
            (pl.col("hpi_state") / pl.col("_b") * 100).alias("hpi_idx")).drop("_b")

    real = rebase(real)
    us = rebase(us).select("month_ym", pl.col("hpi_idx").alias("hpi_nat_idx"))

    disp = real.group_by("month_ym").agg(
        pl.col("unrate_state").quantile(0.25).alias("unr_p25"),
        pl.col("unrate_state").quantile(0.75).alias("unr_p75"),
        pl.col("hpi_idx").quantile(0.10).alias("hpi_p10"),
        pl.col("hpi_idx").quantile(0.90).alias("hpi_p90"),
    )

    df = (nat.join(disp, on="month_ym").join(us, on="month_ym")
          .filter(pl.col("month_ym").is_between(LO, HI)).sort("month_ym"))
    return df


def figure(df: pl.DataFrame):
    x = [_d(m) for m in df["month_ym"].to_list()]
    recessions = [(_d(200712), _d(200906)), (_d(202002), _d(202004))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax in axes.flat:
        for a, b in recessions:
            ax.axvspan(a, b, color="grey", alpha=0.12, lw=0)
        ax.grid(True, alpha=0.25)

    ax = axes[0, 0]
    ax.plot(x, df["pmms30"], color="tab:blue", lw=1.1)
    ax.set_title("30-yr mortgage rate — PMMS (%)", fontsize=10, loc="left")

    ax = axes[0, 1]
    ax.plot(x, df["dgs10"], color="tab:purple", lw=1.1)
    ax.set_title("10-yr Treasury — DGS10 (%)", fontsize=10, loc="left")

    ax = axes[1, 0]
    ax.fill_between(x, df["unr_p25"], df["unr_p75"], color="tab:orange", alpha=0.25,
                    lw=0, label="state IQR")
    ax.plot(x, df["unrate_nat"], color="tab:red", lw=1.1, label="national")
    ax.set_title("Unemployment — national + state IQR (%)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    ax.fill_between(x, df["hpi_p10"], df["hpi_p90"], color="tab:green", alpha=0.22,
                    lw=0, label="state p10–p90")
    ax.plot(x, df["hpi_nat_idx"], color="tab:green", lw=1.3, label="national")
    ax.set_title("FMHPI, 2000-01 = 100 — national + state dispersion", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("F4.2  Macro context, 2000–2025", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    df = build()
    fig = figure(df)
    png, pdf = eda.save_figure(fig, "F4.2_macro_panel")
    series = df.select("month_ym", "pmms30", "dgs10", "slope_10y2y", "unrate_nat",
                       "unr_p25", "unr_p75", "hpi_nat_idx", "hpi_p10", "hpi_p90")
    s_csv, s_tex = eda.save_table(series, "F4.2_macro_panel_series",
                                  caption="Macro panel series underlying F4.2.")

    def at(ym, col):
        return df.filter(pl.col("month_ym") == ym)[col].item()

    def trough(col, lo, hi):
        w = df.filter(pl.col("month_ym").is_between(lo, hi)).sort(col).head(1)
        return w["month_ym"].item(), w[col].item()

    def peak(col, lo, hi):
        w = df.filter(pl.col("month_ym").is_between(lo, hi)).sort(col, descending=True).head(1)
        return w["month_ym"].item(), w[col].item()

    print(f"F4.2 macro panel: {df.height} months {df['month_ym'].min()}–{df['month_ym'].max()}")
    print(f"  unemployment GFC peak   : {peak('unrate_nat', 200901, 201012)}")
    print(f"  unemployment COVID peak : {peak('unrate_nat', 202003, 202012)}")
    print(f"  PMMS 2012 trough        : {trough('pmms30', 201201, 201212)}")
    print(f"  PMMS 2021 trough        : {trough('pmms30', 202101, 202112)}")
    print(f"  national HPI idx 2006-12: {at(200612, 'hpi_nat_idx'):.1f}  "
          f"2012-01: {at(201201, 'hpi_nat_idx'):.1f}  "
          f"(bust = {at(201201,'hpi_nat_idx')/at(200612,'hpi_nat_idx')-1:+.1%})")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
