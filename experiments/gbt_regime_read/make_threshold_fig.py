"""Generate F_impossible_threshold.pdf — GBT impossible mean-mass per window,
raw (old mask) vs. after permitting the four legal skip-bucket cells, all eleven
windows, against the unchanged 1e-3 QA gate.

Replaces the preliminary 5-window version with the M20a-confirmed 11-window read:
five windows breach on raw mass (k2019/20/21/24/25), all eleven clear after the
mask correction, threshold never moves.

Sources (real artifacts; no hand-typed numbers):
  - raw (old-mask) mass   : models/gbt/full/k20XX/metrics.json -> impossible_transitions.model_mean_mass
  - corrected mass        : models/gbt/full/m20_impossible_recheck.json (M20a) -> windows[k].model_mean_mass

Run:  PYTHONPATH=src .venv/bin/python experiments/gbt_regime_read/make_threshold_fig.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from floan.model import config

GATE = 1e-3
BROWN = "#a05a2c"   # raw flagged mass (old mask)
TEAL = "#1a5f6e"    # after permitting the 4 legal cells (corrected mask)
OUT = Path("writeup/latex/figs/F_impossible_threshold")

config.require_drive()

windows = list(config.TEST_YEARS)
recheck = json.loads((config.MODELS / "gbt" / "full" / "m20_impossible_recheck.json").read_text())["windows"]

def raw_mass(k: int) -> float:
    m = json.loads((config.MODELS / "gbt" / "full" / f"k{k}" / "metrics.json").read_text())
    return m["impossible_transitions"]["model_mean_mass"]

raw = [raw_mass(k) for k in windows]
corr = [recheck[str(k)]["model_mean_mass"] for k in windows]

fig, ax = plt.subplots(figsize=(11.5, 4.8))
x = np.arange(len(windows))
w = 0.4
ax.bar(x - w / 2, raw, w, color=BROWN, edgecolor="white", linewidth=0.5, label="raw flagged mass")
ax.bar(x + w / 2, corr, w, color=TEAL, edgecolor="white", linewidth=0.5,
       label="after permitting the 4 legal cells")
ax.axhline(GATE, color="#b22222", ls="--", lw=1.6)
ax.text(len(windows) - 0.5, GATE * 1.15, r"QA gate $10^{-3}$", color="#b22222",
        fontsize=9, fontweight="bold", ha="right", va="bottom")

for xi, r in zip(x, raw):
    if r > GATE:
        ax.annotate("breach", (xi - w / 2, r), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=BROWN)

ax.set_yscale("log")
ax.set_ylim(1e-6, 3e-3)
ax.set_xticks(list(x))
ax.set_xticklabels([f"k={k}" for k in windows], fontsize=9)
ax.set_ylabel("mean impossible mass / row", fontsize=11)
ax.set_title("The threshold never moves: correcting the mask clears every window",
             fontsize=12, fontweight="bold", color=TEAL, pad=10)
ax.legend(loc="lower left", fontsize=9, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()

fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}.pdf / .png")
nb = sum(1 for r in raw if r > GATE)
print(f"raw breaches (>{GATE:g}): {nb}  ->", [k for k, r in zip(windows, raw) if r > GATE])
print("all corrected < gate:", all(c < GATE for c in corr), " max corrected:", f"{max(corr):.2e}")