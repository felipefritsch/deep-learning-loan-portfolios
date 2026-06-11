"""F3.4 — FICO × LTV interaction heatmap (``01_EDA.md`` §3).

The single most persuasive motivation figure. Empirical ``current → dpd_30plus``
(onset of any delinquency) rate on a 2-D FICO × LTV grid. If the FICO gradient
*steepens at high LTV* — low-FICO/high-LTV loans default far more than a linear-
additive FICO+LTV model would predict — then an additive model is misspecified and
the interaction is real (`04_TASKS.md` M3 Accept: "F3.4 a visible FICO×LTV
interaction").

One DuckDB pass emits dpd_30plus/at-risk counts per (integer FICO, rounded LTV) cell
over ``current`` loan-months; the grid is then built with equal-population octile
edges per axis (``eda.equal_pop_edges``) so every cell carries comparable mass.

Run:  python dev/analysis/f3_4_fico_ltv_heatmap.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import eda_common as eda

config = eda.config
NX = 8  # FICO octiles
NY = 8  # LTV octiles
LTV = '"Original Loan-to-Value (LTV)"'


def build(con) -> pl.DataFrame:
    return con.execute(
        f"""
        SELECT fico_orig AS fico, round({LTV}) AS ltv,
               count(*) FILTER (WHERE state_next IS NOT NULL) AS den,
               count(*) FILTER (WHERE state_next IN ('dpd_30','dpd_60','dpd_90plus')) AS num
        FROM panel
        WHERE state = 'current' AND fico_orig IS NOT NULL
          AND {LTV} IS NOT NULL AND {LTV} BETWEEN 1 AND 120
        GROUP BY 1, 2
        """
    ).pl()


def grid(fine: pl.DataFrame):
    """Build the NX×NY rate grid with equal-population edges on each axis."""
    fico_marg = fine.group_by("fico").agg(pl.col("den").sum()).sort("fico")
    ltv_marg = fine.group_by("ltv").agg(pl.col("den").sum()).sort("ltv")
    fx = eda.equal_pop_edges(fico_marg["fico"], fico_marg["den"], NX)
    fy = eda.equal_pop_edges(ltv_marg["ltv"], ltv_marg["den"], NY)
    bx = np.searchsorted(fx, fine["fico"].to_numpy(), side="right")
    by = np.searchsorted(fy, fine["ltv"].to_numpy(), side="right")
    g = (fine.with_columns(pl.Series("bx", bx), pl.Series("by", by))
         .group_by("bx", "by")
         .agg(pl.col("num").sum(), pl.col("den").sum()))
    nx, ny = len(fx) + 1, len(fy) + 1
    rate = np.full((ny, nx), np.nan)
    for r in g.iter_rows(named=True):
        rate[r["by"], r["bx"]] = r["num"] / r["den"] if r["den"] else np.nan
    return rate, fx, fy


def _labels(edges, lo_txt, hi_txt):
    """Tick labels for octiles given interior edges."""
    pts = [lo_txt] + [f"{e:.0f}" for e in edges] + [hi_txt]
    return [f"{pts[i]}–{pts[i+1]}" for i in range(len(pts) - 1)]


def figure(rate, fx, fy):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(rate * 100, origin="lower", aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(rate.shape[1]))
    ax.set_xticklabels(_labels(fx, "≤", "≥"), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(rate.shape[0]))
    ax.set_yticklabels(_labels(fy, "≤", "≥"), fontsize=8)
    ax.set_xlabel("origination FICO (octiles)")
    ax.set_ylabel("origination LTV (octiles)")
    ax.set_title("F3.4  current→dpd_30+ rate (% / mo) by FICO × LTV",
                 fontsize=12, loc="left")
    for j in range(rate.shape[0]):
        for i in range(rate.shape[1]):
            if not np.isnan(rate[j, i]):
                ax.text(i, j, f"{rate[j, i]*100:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black")
    fig.colorbar(im, ax=ax, label="dpd_30+ rate (% / month)")
    fig.tight_layout()
    return fig


def main() -> None:
    config.require_drive()
    con = eda.connect()
    fine = build(con)
    con.close()
    rate, fx, fy = grid(fine)

    # Tidy table of the grid for the memo / reproducibility.
    rows = []
    for j in range(rate.shape[0]):
        for i in range(rate.shape[1]):
            rows.append({"ltv_octile": j, "fico_octile": i, "rate": rate[j, i]})
    s_csv, s_tex = eda.save_table(
        pl.DataFrame(rows), "F3.4_fico_ltv_grid",
        caption="current→dpd_30+ rate on the FICO×LTV octile grid (F3.4).",
    )
    fig = figure(rate, fx, fy)
    png, pdf = eda.save_figure(fig, "F3.4_fico_ltv_heatmap")

    # Accept evidence: a linear-ADDITIVE model forces a constant FICO effect across
    # LTV. The empirical *additive* FICO gradient (worst-FICO minus best-FICO dpd_30+
    # rate) instead grows with LTV — and the LTV effect is far larger for low-FICO
    # loans — so FICO and LTV interact and additivity is misspecified.
    g_low_ltv = (rate[0, 0] - rate[0, -1]) * 100        # FICO gradient at lowest LTV
    g_high_ltv = (rate[-1, 0] - rate[-1, -1]) * 100     # FICO gradient at highest LTV
    ltv_lowfico = (rate[-1, 0] - rate[0, 0]) * 100      # LTV effect, worst-FICO column
    ltv_hifico = (rate[-1, -1] - rate[0, -1]) * 100     # LTV effect, best-FICO column
    print(f"F3.4 FICO×LTV heatmap: {rate.shape[1]} FICO × {rate.shape[0]} LTV octiles")
    print(f"  hottest cell (low FICO, high LTV): {rate[-1,0]*100:.3f}%/mo   "
          f"coolest (high FICO, low LTV): {rate[0,-1]*100:.3f}%/mo")
    print(f"  additive FICO gradient (worst−best FICO): {g_low_ltv:.3f}pp at low LTV "
          f"-> {g_high_ltv:.3f}pp at high LTV  (+{g_high_ltv/g_low_ltv-1:.0%})")
    print(f"  LTV effect (high−low LTV): {ltv_lowfico:.3f}pp for low-FICO vs "
          f"{ltv_hifico:.3f}pp for high-FICO  (×{ltv_lowfico/ltv_hifico:.1f})")
    print(f"  interaction (FICO gradient steepens with LTV): {g_high_ltv > g_low_ltv}")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
