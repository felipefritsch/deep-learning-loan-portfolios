"""F3.3 — Delinquency vs FICO: convex decay (``01_EDA.md`` §3).

Empirical ``current → dpd_30`` one-month rate by origination FICO. Fresh-delinquency
risk falls steeply as credit score rises and the decline flattens — convex, not
linear. A model-free analogue of the paper's nonlinearity exhibits and a companion to
the prepay curves: the gain from depth shows up in default onset too.

One DuckDB pass emits dpd_30/at-risk counts per integer FICO over ``current``
loan-months with an observed next month; ``eda.hazard_curve`` collapses to 20
equal-population buckets with Wilson 95% CIs.

Run:  python -m floan.analysis.f3_3_delinq_fico
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from floan.analysis import eda_common as eda

config = eda.config
N_BUCKETS = 20


def build(con) -> pl.DataFrame:
    fine = con.execute(
        """
        SELECT fico_orig AS fico,
               count(*) FILTER (WHERE state_next IS NOT NULL) AS den,
               count(*) FILTER (WHERE state_next = 'dpd_30') AS num
        FROM panel
        WHERE state = 'current' AND fico_orig IS NOT NULL
        GROUP BY fico_orig
        """
    ).pl()
    return eda.hazard_curve(fine, "fico", "num", "den", N_BUCKETS)


# Legibility: this panel renders two-up at ~0.48\textwidth (down-scaled ~0.47x on
# the page), so fonts are sized large on a compact canvas to stay readable — the
# matching panel style is shared with f3_1/f3_5/f3_6.
PANEL_FIGSIZE = (6.4, 4.4)
LABEL_FS, TICK_FS, LEG_FS = 19, 16, 15


def figure(curve: pl.DataFrame):
    x = curve["value"].to_list()
    rate = [v * 100 for v in curve["rate"].to_list()]
    lo = [v * 100 for v in curve["lo"].to_list()]
    hi = [v * 100 for v in curve["hi"].to_list()]
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.fill_between(x, lo, hi, color="tab:orange", alpha=0.20, lw=0, label="95% CI")
    ax.plot(x, rate, color="tab:orange", lw=2.0, marker="o", ms=5, label="current → dpd_30")
    ax.set_xlabel("origination FICO", fontsize=LABEL_FS)
    ax.set_ylabel("fresh-delinquency rate (% / month)", fontsize=LABEL_FS)
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
        curve, "F3.3_delinq_fico_curve",
        caption="Empirical current→dpd_30 rate by origination-FICO bucket (F3.3).",
    )
    fig = figure(curve)
    png, pdf = eda.save_figure(fig, "F3.3_delinq_fico")

    # Accept evidence: monotone decreasing + convex (slope flattens at high FICO).
    c = curve.sort("value")
    lowest, highest = c.head(1), c.tail(1)
    # Convexity proxy: average |slope| in the low-FICO half >> in the high-FICO half.
    vals = c["value"].to_numpy(); rates = c["rate"].to_numpy()
    import numpy as np
    slope = np.abs(np.diff(rates) / np.diff(vals))
    half = len(slope) // 2
    lo_slope, hi_slope = slope[:half].mean(), slope[half:].mean()
    print(f"F3.3 delinquency vs FICO: {curve.height} buckets, "
          f"FICO {c['value'].min():.0f}–{c['value'].max():.0f}")
    print(f"  low FICO  (≈{lowest['value'].item():.0f}): {lowest['rate'].item()*100:.4f}%/mo")
    print(f"  high FICO (≈{highest['value'].item():.0f}): {highest['rate'].item()*100:.4f}%/mo")
    print(f"  mean |slope| low-FICO half {lo_slope*100:.5f}  vs high-FICO half {hi_slope*100:.5f} "
          f"(%/mo per FICO pt) -> convex: {lo_slope > hi_slope}")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
