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

Run:  python dev/analysis/f3_2_prepay_incentive.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

import eda_common as eda

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


def figure(curve: pl.DataFrame):
    x = curve["value"].to_list()
    rate = [v * 100 for v in curve["rate"].to_list()]
    lo = [v * 100 for v in curve["lo"].to_list()]
    hi = [v * 100 for v in curve["hi"].to_list()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.20, lw=0, label="95% CI")
    ax.plot(x, rate, color="tab:blue", lw=1.4, marker="o", ms=3, label="current → prepaid")
    ax.axvline(0.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("rate incentive = current rate − PMMS30  (percentage points)")
    ax.set_ylabel("prepayment rate (% / month)")
    ax.set_title("F3.2  Prepayment vs rate incentive — the refinancing S-curve",
                 fontsize=12, loc="left")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
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
