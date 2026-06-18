"""F5.2 — Vintage × shard mix (``01_EDA.md`` §5; May-8 notes).

Confirms the shards are *not* vintage-clustered — the failure mode the May-8 notes
flagged. Because ``shard = hash(loan_id) % N_SHARDS`` keys on the loan id (not the
vintage), every shard should carry the same origination-year mix as the whole panel.
This figure makes that visible: a heatmap of each shard's vintage composition (column-
normalized, so every shard's column sums to 1); a clustered hash would show vertical
stripes, a clean one shows horizontal bands matching the panel marginals.

One DuckDB pass: ``count(*)`` per (origination year, shard).

Run:  python dev/analysis/f5_2_vintage_shard.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import eda_common as eda

config = eda.config


def build(con) -> pl.DataFrame:
    return con.execute(
        """
        SELECT orig_ym // 100 AS orig_year, shard, count(*) AS n
        FROM panel
        WHERE orig_ym IS NOT NULL
        GROUP BY 1, 2
        """
    ).pl()


def matrix(df: pl.DataFrame):
    """Column-normalized vintage×shard share matrix (rows=year, cols=shard)."""
    years = sorted(df["orig_year"].unique().to_list())
    shards = sorted(df["shard"].unique().to_list())
    yi = {y: i for i, y in enumerate(years)}
    si = {s: i for i, s in enumerate(shards)}
    counts = np.zeros((len(years), len(shards)))
    for r in df.iter_rows(named=True):
        counts[yi[r["orig_year"]], si[r["shard"]]] = r["n"]
    share = counts / counts.sum(axis=0, keepdims=True)  # per-shard composition
    panel_marg = counts.sum(axis=1) / counts.sum()       # whole-panel vintage mix
    return share, panel_marg, years, shards


def figure(share, years):
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(share * 100, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=7)
    ax.set_xlabel("shard (0–255)")
    ax.set_ylabel("origination year")
    ax.set_title("F5.2  Vintage × shard mix — each shard's origination-year composition (%)",
                 fontsize=12, loc="left")
    fig.colorbar(im, ax=ax, label="share of shard's loan-months (%)")
    fig.tight_layout()
    return fig


def main() -> None:
    config.require_drive()
    con = eda.connect()
    df = build(con)
    con.close()
    share, panel_marg, years, shards = matrix(df)

    # Tidy long table for reproducibility.
    rows = [{"orig_year": y, "shard": s, "share": float(share[i, j])}
            for i, y in enumerate(years) for j, s in enumerate(shards)]
    s_csv, s_tex = eda.save_table(
        pl.DataFrame(rows), "F5.2_vintage_shard",
        caption="Per-shard origination-year composition shares (F5.2).",
    )
    fig = figure(share, years)
    png, pdf = eda.save_figure(fig, "F5.2_vintage_shard")

    # Accept evidence: each vintage's per-shard share barely deviates from the panel
    # marginal — i.e., shards are not vintage-clustered. A tiny cohort has high
    # *relative* noise on a near-zero share, so the pass criterion is judged on the
    # substantive vintages (panel share ≥ 0.5%); the smallest cohorts are reported
    # transparently alongside.
    cvs = [(y, float(share[i].std() / share[i].mean()) if share[i].mean() > 0 else 0.0,
            float(panel_marg[i])) for i, y in enumerate(years)]
    big = [(y, cv) for (y, cv, m) in cvs if m >= 0.005]
    worst_big = max(big, key=lambda t: t[1])
    worst_any = max(cvs, key=lambda t: t[1])
    median_cv = float(np.median([cv for _, cv, _ in cvs]))
    print(f"F5.2 vintage×shard: {len(years)} vintages × {len(shards)} shards")
    print(f"  across-shard CV of per-shard composition: median {median_cv*100:.2f}%")
    print(f"  worst substantive vintage (share≥0.5%): {worst_big[0]}  CV {worst_big[1]*100:.2f}%")
    print(f"  worst of all (incl. tiny cohorts): {worst_any[0]}  CV {worst_any[1]*100:.2f}% "
          f"(panel share {worst_any[2]*100:.2f}%)")
    print(f"  shards NOT vintage-clustered (substantive-vintage max CV < 5%): "
          f"{worst_big[1] < 0.05}")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
