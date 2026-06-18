"""F4.3 — State-dispersion check (``01_EDA.md`` §4).

Cross-state dispersion in a crisis year (2009) vs a calm year (2019), for two metrics:
annual-mean unemployment and HPI drawdown (level relative to each state's running peak,
at December). In 2009 both fan out enormously across states — sand states (NV/AZ/FL)
deep in HPI drawdown and high unemployment, others barely touched; by 2019 both collapse
to a tight band. That spread is the motivation for *state-level* macro joins rather than
national-only (MD4 Accept). National values (the FMHPI/UNRATE ``US`` series) are overlaid
as reference markers.

Run:  python -m floan.analysis.f4_3_state_dispersion
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from floan.analysis import eda_common as eda

config = eda.config

YEARS = [2009, 2019]


def build() -> tuple[pl.DataFrame, dict]:
    config.require_drive()
    mdir = config.PROCESSED / "macro"
    state = pl.read_parquet(mdir / "macro_state.parquet").with_columns(pl.col("state").cast(pl.Utf8))

    # HPI drawdown vs each state's running max up to that month (cum-max over sorted months).
    state = state.sort("state", "month_ym").with_columns(
        (pl.col("hpi_state") / pl.col("hpi_state").cum_max().over("state") - 1.0).alias("hpi_dd"))

    rows = []
    for y in YEARS:
        # annual-mean unemployment per state; HPI drawdown at December.
        unr = (state.filter((pl.col("month_ym") // 100) == y)
               .group_by("state").agg(pl.col("unrate_state").mean().alias("unrate")))
        dd = (state.filter(pl.col("month_ym") == y * 100 + 12)
              .select("state", pl.col("hpi_dd")))
        rows.append(unr.join(dd, on="state").with_columns(pl.lit(y).alias("year")))
    df = pl.concat(rows)

    real = df.filter(pl.col("state") != "US")
    nat = {y: {
        "unrate": df.filter((pl.col("state") == "US") & (pl.col("year") == y))["unrate"].item(),
        "hpi_dd": df.filter((pl.col("state") == "US") & (pl.col("year") == y))["hpi_dd"].item(),
    } for y in YEARS}
    return real, nat


def figure(real: pl.DataFrame, nat: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A — unemployment dispersion.
    data = [real.filter(pl.col("year") == y)["unrate"].to_list() for y in YEARS]
    axes[0].boxplot(data, tick_labels=[str(y) for y in YEARS], widths=0.5)
    for i, y in enumerate(YEARS, start=1):
        axes[0].scatter([i], [nat[y]["unrate"]], color="tab:red", zorder=3, label="national" if i == 1 else None)
    axes[0].set_title("State unemployment (annual mean, %)", fontsize=10, loc="left")
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25, axis="y")

    # Panel B — HPI drawdown dispersion (December level vs running peak).
    data = [[v * 100 for v in real.filter(pl.col("year") == y)["hpi_dd"].to_list()] for y in YEARS]
    axes[1].boxplot(data, tick_labels=[str(y) for y in YEARS], widths=0.5)
    for i, y in enumerate(YEARS, start=1):
        axes[1].scatter([i], [nat[y]["hpi_dd"] * 100], color="tab:red", zorder=3, label="national" if i == 1 else None)
    axes[1].axhline(0, color="k", lw=0.6, alpha=0.5)
    axes[1].set_title("State HPI drawdown vs peak, December (%)", fontsize=10, loc="left")
    axes[1].set_ylabel("%")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25, axis="y")

    fig.suptitle("F4.3  Cross-state dispersion: crisis (2009) vs calm (2019)", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    real, nat = build()
    fig = figure(real, nat)
    png, pdf = eda.save_figure(fig, "F4.3_state_dispersion")

    tbl = (real.group_by("year").agg(
        pl.col("unrate").min().alias("unr_min"), pl.col("unrate").median().alias("unr_med"),
        pl.col("unrate").max().alias("unr_max"),
        (pl.col("hpi_dd") * 100).min().alias("dd_min_pct"),
        (pl.col("hpi_dd") * 100).median().alias("dd_med_pct"),
    ).sort("year"))
    c_csv, c_tex = eda.save_table(tbl, "F4.3_state_dispersion_stats",
                                  caption="Cross-state unemployment & HPI-drawdown ranges, 2009 vs 2019.")

    print("F4.3 state dispersion (2009 vs 2019):")
    for y in YEARS:
        r = real.filter(pl.col("year") == y)
        u_lo = r.sort("unrate").head(1); u_hi = r.sort("unrate", descending=True).head(1)
        d_lo = r.sort("hpi_dd").head(1)
        print(f"  {y}: unemployment {u_lo['unrate'].item():.1f}% ({u_lo['state'].item()}) – "
              f"{u_hi['unrate'].item():.1f}% ({u_hi['state'].item()}), national {nat[y]['unrate']:.1f}%")
        print(f"        HPI drawdown worst {d_lo['hpi_dd'].item()*100:.1f}% ({d_lo['state'].item()}), "
              f"median {r['hpi_dd'].median()*100:.1f}%, national {nat[y]['hpi_dd']*100:.1f}%")
    print(f"\nwrote {png}\n      {pdf}\n      {c_csv}\n      {c_tex}")


if __name__ == "__main__":
    main()
