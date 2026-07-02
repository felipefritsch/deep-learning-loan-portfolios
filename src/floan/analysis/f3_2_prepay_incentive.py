"""F3.2 — Prepayment vs rate incentive: the S-curve (``01_EDA.md`` §3).

The paper's flagship nonlinearity. ``incentive = current_rate − pmms30(t)`` (the
PMMS-based incentive, lag 0 per ``macro_features.LAG``): how far the loan's note rate
sits above the prevailing market rate. Empirical ``current → prepaid`` rate is flat
and low when incentive is negative (nothing to gain by refinancing), turns up sharply
through zero, and saturates / burns out at large positive incentive — an S shape that
no linear term can fit.

One DuckDB pass joins ``macro_national`` (PMMS) on ``period_ym`` and emits
prepaid/at-risk counts per 5 bp incentive bin over ``current`` loan-months;
``eda.hazard_curve`` collapses to 25 equal-population buckets with Wilson 95% CIs.

Run:  python -m floan.analysis.f3_2_prepay_incentive
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from floan.analysis import eda_common as eda

config = eda.config
MACRO_NAT = config.PROCESSED / "macro" / "macro_national.parquet"
N_BUCKETS = 25
BIN = 0.05  # incentive rounding (percentage points) for the fine histogram


def build(con) -> pl.DataFrame:
    fine = con.execute(
        f"""
        SELECT round((p."Current Interest Rate" - m.pmms30) / {BIN}) * {BIN} AS incentive,
               count(*) FILTER (WHERE p.state_next IS NOT NULL) AS den,
               count(*) FILTER (WHERE p.state_next = 'prepaid') AS num
        FROM panel p
        JOIN read_parquet('{MACRO_NAT}') m ON m.month_ym = p.period_ym
        WHERE p.state = 'current' AND p."Current Interest Rate" IS NOT NULL
        GROUP BY 1
        """
    ).pl()
    return eda.hazard_curve(fine, "incentive", "num", "den", N_BUCKETS)


# Legibility: this figure renders single at ~0.8\textwidth (down-scaled ~0.65x on
# the page), so fonts are enlarged on a slightly compact canvas — a milder bump than
# the two-up f3_1/f3_3/f3_5/f3_6 panels, which are scaled down further.
FIGSIZE = (7.6, 4.6)
LABEL_FS, TICK_FS, LEG_FS = 15, 12.5, 12


def figure(curve: pl.DataFrame):
    x = curve["value"].to_list()
    rate = [v * 100 for v in curve["rate"].to_list()]
    lo = [v * 100 for v in curve["lo"].to_list()]
    hi = [v * 100 for v in curve["hi"].to_list()]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.20, lw=0, label="95% CI")
    ax.plot(x, rate, color="tab:blue", lw=1.9, marker="o", ms=4.5, label="current → prepaid")
    ax.axvline(0.0, color="grey", ls="--", lw=1.0)
    ax.set_xlabel("rate incentive = current rate − PMMS30  (percentage points)", fontsize=LABEL_FS)
    ax.set_ylabel("prepayment rate (% / month)", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    # No in-figure title — the LaTeX caption supplies it.
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=LEG_FS, loc="upper left")
    fig.tight_layout()
    return fig


def main() -> None:
    config.require_drive()
    con = eda.connect()
    curve = build(con)
    con.close()

    s_csv, s_tex = eda.save_table(
        curve, "F3.2_prepay_incentive_curve",
        caption="Empirical current→prepaid rate by PMMS-incentive bucket (F3.2).",
    )
    fig = figure(curve)
    png, pdf = eda.save_figure(fig, "F3.2_prepay_incentive")

    # Accept evidence: monotone S — out-of-the-money << at-the-money << in-the-money.
    def near(target):
        return curve.with_columns((pl.col("value") - target).abs().alias("d")).sort("d").head(1)
    neg, zero, pos = near(-1.0), near(0.0), near(1.5)
    print(f"F3.2 prepay vs incentive: {curve.height} buckets, "
          f"incentive {curve['value'].min():.2f}–{curve['value'].max():.2f} pp")
    print(f"  out-of-money (≈-1.0pp): {neg['rate'].item()*100:.3f}%/mo")
    print(f"  at-the-money (≈ 0.0pp): {zero['rate'].item()*100:.3f}%/mo")
    print(f"  in-the-money (≈+1.5pp): {pos['rate'].item()*100:.3f}%/mo")
    s_shape = neg["rate"].item() < zero["rate"].item() < pos["rate"].item()
    print(f"  S-shape (rises through zero incentive): {s_shape}")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
