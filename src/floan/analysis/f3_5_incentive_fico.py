"""F3.5 — Incentive × FICO interaction: refi response by credit (``01_EDA.md`` §3).

The prepayment-incentive S-curve (F3.2), split by origination-FICO tercile. High-FICO
borrowers refinance far more responsively to a given incentive — they qualify and act;
lower-FICO borrowers are credit-constrained and burnt-out, so their S-curve is flatter
and lower even when deep in the money. The incentive effect therefore *depends on*
credit quality: an interaction a linear-additive model cannot represent.

One DuckDB pass joins ``macro_national`` (PMMS) and emits prepaid/at-risk counts per
(10 bp incentive bin, integer FICO) cell over ``current`` loan-months. FICO terciles
come from equal-population edges on the FICO margin; within each tercile the incentive
axis is collapsed to equal-population buckets with Wilson 95% CIs.

Run:  python -m floan.analysis.f3_5_incentive_fico
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from floan.analysis import eda_common as eda

config = eda.config
MACRO_NAT = config.PROCESSED / "macro" / "macro_national.parquet"
N_INC = 20      # incentive buckets per FICO tercile
BIN = 0.10      # incentive rounding (pp)
TERCILES = ["low FICO", "mid FICO", "high FICO"]


def build(con) -> pl.DataFrame:
    return con.execute(
        f"""
        SELECT round((p."Current Interest Rate" - m.pmms30) / {BIN}) * {BIN} AS incentive,
               p.fico_orig AS fico,
               count(*) FILTER (WHERE p.state_next IS NOT NULL) AS den,
               count(*) FILTER (WHERE p.state_next = 'prepaid') AS num
        FROM panel p
        JOIN read_parquet('{MACRO_NAT}') m ON m.month_ym = p.period_ym
        WHERE p.state = 'current' AND p."Current Interest Rate" IS NOT NULL
          AND p.fico_orig IS NOT NULL
        GROUP BY 1, 2
        """
    ).pl()


def curves(fine: pl.DataFrame):
    """One incentive hazard curve per FICO tercile (equal-population FICO edges)."""
    fico_marg = fine.group_by("fico").agg(pl.col("den").sum()).sort("fico")
    edges = eda.equal_pop_edges(fico_marg["fico"], fico_marg["den"], 3)  # 2 interior edges
    ti = np.searchsorted(edges, fine["fico"].to_numpy(), side="right")
    fine = fine.with_columns(pl.Series("tercile", ti))
    out = {}
    for t in range(len(edges) + 1):
        sub = (fine.filter(pl.col("tercile") == t)
               .group_by("incentive").agg(pl.col("num").sum(), pl.col("den").sum()))
        out[t] = eda.hazard_curve(sub, "incentive", "num", "den", N_INC)
    return out, edges


def figure(out, edges):
    colors = ["tab:red", "tab:purple", "tab:blue"]
    fico_lab = [f"≤{edges[0]:.0f}", f"{edges[0]:.0f}–{edges[1]:.0f}", f"≥{edges[1]:.0f}"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for t, curve in out.items():
        x = curve["value"].to_list()
        rate = [v * 100 for v in curve["rate"].to_list()]
        ax.plot(x, rate, color=colors[t], lw=1.5, marker="o", ms=3,
                label=f"{TERCILES[t]} ({fico_lab[t]})")
    ax.axvline(0.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("rate incentive = current rate − PMMS30  (percentage points)")
    ax.set_ylabel("prepayment rate (% / month)")
    # No in-figure title — the LaTeX caption supplies it.
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper left", title="origination FICO")
    fig.tight_layout()
    return fig


def main() -> None:
    config.require_drive()
    con = eda.connect()
    fine = build(con)
    con.close()
    out, edges = curves(fine)

    tidy = pl.concat([
        c.with_columns(pl.lit(TERCILES[t]).alias("fico_tercile")) for t, c in out.items()
    ])
    s_csv, s_tex = eda.save_table(
        tidy, "F3.5_incentive_fico_curves",
        caption="current→prepaid rate by incentive bucket within FICO tercile (F3.5).",
    )
    fig = figure(out, edges)
    png, pdf = eda.save_figure(fig, "F3.5_incentive_fico")

    # Accept evidence: at a fixed in-the-money incentive the high-FICO refi response
    # exceeds the low-FICO response — the incentive effect depends on FICO.
    def at_pos(curve, target=1.5):
        return (curve.with_columns((pl.col("value") - target).abs().alias("d"))
                .sort("d").head(1)["rate"].item())
    lo_resp, hi_resp = at_pos(out[0]), at_pos(out[len(edges)])
    print(f"F3.5 incentive×FICO: FICO tercile edges {edges}")
    for t, c in out.items():
        print(f"  {TERCILES[t]:9s}: prepay@(+1.5pp incentive) = {at_pos(c)*100:.3f}%/mo")
    print(f"  interaction (high-FICO refi response > low-FICO at same incentive): "
          f"{hi_resp > lo_resp}  (×{hi_resp/lo_resp:.1f})")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
