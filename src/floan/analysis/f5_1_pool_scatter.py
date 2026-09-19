"""F5.1 — predicted vs realised pool prepayment counts, random pools (M14 §4).

Reproducible from the committed repo alone: reads the per-pool count table
``artifacts/results/m14_pools/pools_random_k{k}.parquet`` (500 random pools of
1,000 loans at the Dec2024 anchor, k=2025; produced by ``floan.model.pools``)
and emits ``artifacts/generated/F4.1_scatter_random_prepaid.pdf`` — the
predicted-vs-realised twelve-month prepayment-count scatter for the three
headline models (memoryless empirical -> logit -> ensemble).

Each panel has its own square axes. This is deliberate: the empirical matrix
over-predicts to a near-constant ~152 per pool, far above the realised 49--98
band, so a shared y-axis (sized to the well-calibrated models, ~0--100) clipped
the empirical cloud off the top entirely, leaving an empty panel. There is no
in-figure suptitle; the caption carries the outcome, anchor and pool count.

Run: ``python -m floan.analysis.f5_1_pool_scatter`` (no SSD/GPU needed).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "artifacts" / "results" / "m14_pools"
FIGS = ROOT / "artifacts" / "generated"

K = 2025                       # Dec2024 anchor
OUTCOME = "prepaid"
# the three headline models, in ladder order, with display labels
MODELS = [("empirical", "Empirical"), ("logit", "Logit"),
          ("ensemble", r"Ensemble $\times 8$")]


def _r2_rmse(pred: np.ndarray, realized: np.ndarray) -> tuple[float, float]:
    """R² (1 − SS_res/SS_tot about the realized mean) and RMSE, matching pools.py."""
    resid = pred - realized
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sstot = float(np.sum((realized - realized.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / sstot) if sstot > 0 else float("nan")
    return r2, rmse


def make_figure() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "mathtext.fontset": "cm"})
    df = pl.read_parquet(SRC / f"pools_random_k{K}.parquet")
    realized = df[f"{OUTCOME}_realized"].to_numpy()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9.6, 4.3))  # independent axes
    for ax, (m, label) in zip(axes, MODELS):
        pred = df[f"{m}_{OUTCOME}_pred"].to_numpy()
        ax.scatter(realized, pred, s=10, alpha=0.35, edgecolors="none")
        # x-axis tracks the realised range (identical across panels); y-axis
        # extends to fit each model's predictions. The empirical matrix predicts
        # far above the realised band, so only its y-axis runs higher — its
        # x-axis stays on the shared 0--100 scale rather than stretching to match.
        xmax = float(realized.max()) * 1.05 or 1.0
        ymax = max(float(realized.max()), float(pred.max())) * 1.05 or 1.0
        ax.plot([0, max(xmax, ymax)], [0, max(xmax, ymax)], ls="--", c="grey", lw=1)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        r2, rmse = _r2_rmse(pred, realized)
        ax.set_title(f"{label}\n$R^2$={r2:.3f}  RMSE={rmse:.1f}")
        ax.set_xlabel("Realized count")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted count")
    fig.tight_layout()
    out = FIGS / "F4.1_scatter_random_prepaid.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    make_figure()
