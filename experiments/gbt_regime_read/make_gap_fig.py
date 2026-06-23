"""Generate F_gbt_regime_gap.pdf — the single network's per-window NLL lead over
the gradient-boosted tree, all eleven rolling windows.

Replaces the preliminary five-window bar chart (which had no committed generator)
with the full eleven-window read, and drops the retired "monotone in regime calm,
then flips" framing: the gap is thin in every window, does not order by regime
calm, and turns negative only on k2016 and the COVID window k2020.

Sources (real artifacts; no hand-typed numbers):
  - per-window single-NN NLL  : outputs/tables/loan_level/table_b.csv  (committed)
  - per-window GBT test NLL    : models/gbt/full/k20XX/metrics.json -> gbt.test_nll

Gap plotted = round(GBT,5) - round(NN,5), the same "display" rounding used in
Table~\ref{tab:gbtregime}, so every bar ties to that table's GBT-NN column to the
last digit (k2016 = -0.00028, not the full-precision -0.00027).

Run:  PYTHONPATH=src .venv/bin/python experiments/gbt_regime_read/make_gap_fig.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from floan.model import config

WINDOWS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
REGIME = {
    2015: "tuning", 2016: "calm", 2017: "calm", 2018: "calm", 2019: "calm",
    2020: "COVID", 2021: "refi wave", 2022: "rate-rise", 2023: "rate-spike",
    2024: "rate-spike", 2025: "latest",
}
TEAL = "#1a5f6e"   # net ahead (gap > 0)
BROWN = "#a05a2c"  # tree ahead (gap < 0)
OUT = Path("writeup/latex/figs/F_gbt_regime_gap")

config.require_drive()

# per-window single-network NLL from the committed Table-B export
nn = {}
with (config.OUTPUTS / "tables" / "loan_level" / "table_b.csv").open() as f:
    for r in csv.DictReader(f):
        if r["test_year"].isdigit():
            nn[int(r["test_year"])] = float(r["nn"])

# per-window GBT test NLL from each window's fit-time metrics
def gbt_nll(k: int) -> float:
    p = config.MODELS / "gbt" / "full" / f"k{k}" / "metrics.json"
    return json.loads(p.read_text())["gbt"]["test_nll"]

# display-convention gap (matches the table), in units of 1e-3
gaps = [round(round(gbt_nll(k), 5) - round(nn[k], 5), 5) * 1e3 for k in WINDOWS]

fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = range(len(WINDOWS))
bars = ax.bar(x, gaps, width=0.66,
              color=[TEAL if g >= 0 else BROWN for g in gaps],
              edgecolor="white", linewidth=0.6)
ax.axhline(0, color="black", lw=1.0)

for xi, g in zip(x, gaps):
    ax.annotate(f"{g:+.2f}", (xi, g),
                xytext=(0, 4 if g >= 0 else -4), textcoords="offset points",
                ha="center", va="bottom" if g >= 0 else "top",
                fontsize=9, fontweight="bold",
                color=TEAL if g >= 0 else BROWN)

ax.set_xticks(list(x))
ax.set_xticklabels([f"{k}\n({REGIME[k]})" for k in WINDOWS], fontsize=9)
ax.set_ylabel(r"GBT $-$ single-NN  NLL  ($\times10^{-3}$)", fontsize=11)
ax.margins(y=0.18)
ax.text(0.012, 0.97, "above 0: single net ahead", transform=ax.transAxes,
        va="top", ha="left", fontsize=9, color=TEAL)
ax.text(0.012, 0.03, "below 0: GBT ahead", transform=ax.transAxes,
        va="bottom", ha="left", fontsize=9, color=BROWN)
ax.set_title("The net's lead over the tree is thin in every window and does not "
             "order by regime calm", fontsize=12, fontweight="bold", color=TEAL,
             pad=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()

fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote", f"{OUT}.pdf / .png")
for k, g in zip(WINDOWS, gaps):
    print(f"  {k} ({REGIME[k]:<10}) {g:+.2f}e-3")
