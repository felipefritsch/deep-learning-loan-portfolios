"""F5.4 / T5.3 — the value of memory in dollars (W3b matched pricing comparison).

Reproducible from the committed repo alone: reads the matched per-anchor pricing
tables in ``src/floan/model/m27b_mc/k{k}_matched.parquet`` (mean |price error| per
100 face by model x horizon, on the identical 150k subsample) and emits

  * ``writeup/latex/figs/F5.4_value_of_memory.pdf`` — the error-by-horizon
    small-multiple across the five regime anchors, drawing the memory ladder
    (memoryless ensemble -> engineered ff_hist -> learned GRU/transformer); and
  * ``writeup/latex/figs/table_w3b_matched.tex`` — the COVID (Dec2019) anchor
    table, the full seven-model ladder x four horizons.

Run: ``python -m floan.analysis.f5_4_value_of_memory`` (no SSD/GPU needed).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "floan" / "model" / "m27b_mc"
FIGS = ROOT / "writeup" / "latex" / "figs"

ANCHORS = [2015, 2019, 2020, 2023, 2025]
ANCHOR_LABEL = {2015: "Dec 2014", 2019: "Dec 2018", 2020: "Dec 2019\n(COVID)",
                2023: "Dec 2022\n(rate shock)", 2025: "Dec 2024"}
COVID = 2020
HORIZONS = [1, 3, 6, 12]

# display names + the three-rung ladder grouping (memoryless / engineered / learned)
MODEL_NAME = {"empirical": "Empirical", "logit": "Logit", "nn": "FF current-state",
              "ensemble": r"Ensemble $\times 8$", "ff_hist": "FF + history",
              "gru": "GRU", "xf": "Transformer"}
RUNGS = [("memoryless", ["empirical", "logit", "nn", "ensemble"]),
         ("engineered", ["ff_hist"]),
         ("learned", ["gru", "xf"])]

# the four lines drawn in the figure (the rest of the memoryless rung is in the table)
LINES = [  # model, label, colour, linestyle, marker
    ("ensemble", "Ensemble (memoryless)", "#7A7A7A", "-", "o"),
    ("ff_hist",  "FF + history (engineered)", "#E07B1A", "-", "s"),
    ("gru",      "GRU (learned)", "#1F4E79", "-", "^"),
    ("xf",       "Transformer (learned)", "#1F4E79", "--", "v"),
]


def _load(k: int) -> pl.DataFrame:
    return pl.read_parquet(SRC / f"k{k}_matched.parquet")


# 2x3 panel layout: the two pre-COVID calm anchors on the top row, then COVID
# leading the bottom row of stress/recovery anchors. The empty top-right cell
# holds the legend. Wider panels (3 columns, not 5) keep the curves legible.
GRID = [[2015, 2019, None],
        [2020, 2023, 2025]]


def make_figure() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.titlesize": 9, "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.2), sharey=True)
    legend_ax = None
    for r, row in enumerate(GRID):
        for c, k in enumerate(row):
            ax = axes[r][c]
            if k is None:
                ax.axis("off")
                legend_ax = ax
                continue
            df = _load(k)
            for model, _lab, colour, ls, mk in LINES:
                sub = df.filter(pl.col("model") == model).sort("h")
                ax.plot(sub["h"].to_list(), sub["mae_price"].to_list(),
                        color=colour, linestyle=ls, marker=mk, markersize=4,
                        linewidth=1.4, markerfacecolor="white" if ls == "--" else colour)
            ax.set_title(ANCHOR_LABEL[k], fontweight="bold" if k == COVID else "normal")
            ax.set_xticks(HORIZONS)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.margins(x=0.05)
            if r == len(GRID) - 1:        # x-label on the bottom row only
                ax.set_xlabel("Horizon (months)")
            if c == 0:                    # y-label on the left column only (sharey)
                ax.set_ylabel("Mean $|$price error$|$\n(per 100 face)")
            if k == COVID:
                ax.set_facecolor("#F4F1EA")
    handles = [plt.Line2D([], [], color=c, linestyle=ls, marker=mk, markersize=4,
                          linewidth=1.4, markerfacecolor="white" if ls == "--" else c)
               for _m, _l, c, ls, mk in LINES]
    legend_ax.legend(handles, [lab for _m, lab, *_ in LINES], loc="center",
                     frameon=False, handlelength=2.4, borderaxespad=0.0)
    fig.tight_layout()
    out = FIGS / "F5.4_value_of_memory.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _cell(mae: float, bias: float) -> str:
    return f"{mae:.3f} ({bias:+.3f})"


def make_table() -> None:
    df = _load(COVID)
    wide = {(r["model"], r["h"]): (r["mae_price"], r["bias_price"])
            for r in df.iter_rows(named=True)}
    lines = [r"\begin{tabular}{l" + "r" * len(HORIZONS) + "}", r"\toprule",
             "Model & " + " & ".join(fr"$H{{=}}{h}$" for h in HORIZONS) + r" \\",
             r"\midrule"]
    for ri, (_rung, models) in enumerate(RUNGS):
        for m in models:
            row = " & ".join(_cell(*wide[(m, h)]) for h in HORIZONS)
            lines.append(f"{MODEL_NAME[m]} & {row} " + r"\\")
        if ri < len(RUNGS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = FIGS / "table_w3b_matched.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    make_figure()
    make_table()
