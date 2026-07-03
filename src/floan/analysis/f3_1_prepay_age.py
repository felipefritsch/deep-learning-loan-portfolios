"""F3.1 — Prepayment vs loan age: the seasoning hump (``01_EDA.md`` §3).

Empirical ``current → prepaid`` one-month rate as a function of loan age. The
classic seasoning hump: near-zero at origination, rising to a peak around 2–4 years
as borrowers become refinance-eligible and mobile, then declining (burnout). Strongly
non-monotone — indefensible as a linear term, which is the motivation this figure
carries (`04_TASKS.md` M3 Accept: "F3.1 shows the seasoning hump").

One DuckDB pass emits prepaid/at-risk counts per integer loan age over ``current``
loan-months with an observed next month; ``eda.hazard_curve`` then collapses ages
into 40 equal-population buckets and attaches Wilson 95% CIs.

Run:  python -m floan.analysis.f3_1_prepay_age
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from floan.analysis import eda_common as eda

config = eda.config

N_BUCKETS = 40


def build(con) -> pl.DataFrame:
    fine = con.execute(
        """
        SELECT "Loan Age" AS age,
               count(*) FILTER (WHERE state_next IS NOT NULL) AS den,
               count(*) FILTER (WHERE state_next = 'prepaid') AS num
        FROM panel
        WHERE state = 'current' AND "Loan Age" IS NOT NULL AND "Loan Age" >= 0
        GROUP BY "Loan Age"
        """
    ).pl()
    return eda.hazard_curve(fine, "age", "num", "den", N_BUCKETS)


# Legibility: this panel renders two-up at ~0.48\textwidth, so it is down-scaled to
# ~0.47x on the page. Fonts are sized large on a compact canvas to stay readable
# after that scaling (see f3_3/f3_5/f3_6 for the matching panel style).
PANEL_FIGSIZE = (6.4, 4.4)
LABEL_FS, TICK_FS, LEG_FS = 19, 16, 15


def figure(curve: pl.DataFrame):
    x = curve["value"].to_list()
    rate = [v * 100 for v in curve["rate"].to_list()]
    lo = [v * 100 for v in curve["lo"].to_list()]
    hi = [v * 100 for v in curve["hi"].to_list()]
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.fill_between(x, lo, hi, color="tab:green", alpha=0.20, lw=0, label="95% CI")
    ax.plot(x, rate, color="tab:green", lw=2.0, marker="o", ms=5, label="current → prepaid")
    ax.set_xlabel("loan age (months)", fontsize=LABEL_FS)
    # y=0.47 slides the label down the axis: centred (y=0.5) its top clips in
    # the tight export, because the axes centre sits above the canvas centre.
    ax.set_ylabel("prepayment rate (% / month)", fontsize=LABEL_FS, y=0.47)
    ax.tick_params(labelsize=TICK_FS)
    # No in-figure title — the LaTeX caption supplies it.
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=LEG_FS, loc="upper right")
    fig.tight_layout()
    return fig


def main() -> None:
    config.require_drive()
    con = eda.connect()
    curve = build(con)
    con.close()

    s_csv, s_tex = eda.save_table(
        curve, "F3.1_prepay_age_curve",
        caption="Empirical current→prepaid rate by loan-age bucket (F3.1).",
    )
    fig = figure(curve)
    png, pdf = eda.save_figure(fig, "F3.1_prepay_age")

    # Accept evidence: the seasoning hump is an early-life local maximum well above
    # origination, followed by a burnout decline. (A separate late-life uptick from
    # deep-in-the-money survivors exists too — reported honestly, not the hump.)
    c = curve.sort("value")
    young = c.head(1)                               # near origination
    early = c.filter(pl.col("value") <= 60)
    hump = early.sort("rate", descending=True).head(1)
    hump_age = hump["value"].item()
    trough = (c.filter(pl.col("value").is_between(hump_age, 90))
              .sort("rate").head(1))               # post-hump burnout
    old = c.tail(1)                                # most-seasoned bucket
    is_hump = (hump["rate"].item() > young["rate"].item()
               and hump["rate"].item() > trough["rate"].item())
    print(f"F3.1 prepay vs loan age: {curve.height} buckets, "
          f"age {c['value'].min():.0f}–{c['value'].max():.0f} mo")
    print(f"  origination  (age≈{young['value'].item():.0f}):  {young['rate'].item()*100:.3f}%/mo")
    print(f"  SEASONING HUMP (age≈{hump_age:.0f}): {hump['rate'].item()*100:.3f}%/mo")
    print(f"  burnout trough (age≈{trough['value'].item():.0f}): {trough['rate'].item()*100:.3f}%/mo")
    print(f"  late-life     (age≈{old['value'].item():.0f}): {old['rate'].item()*100:.3f}%/mo "
          f"(deep-ITM survivors; pooled over calendar time)")
    print(f"  seasoning hump present (early peak > origination and > burnout trough): {is_hump}")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
