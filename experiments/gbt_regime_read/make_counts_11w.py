"""Edit 8 — extend the impossible-cell realised counts from the 5 regime windows to all 11.

Closes the last 5->11 gap in chapter4 §4.8: the M20a recheck confirms the gate clears on all
eleven windows on *mass*, but the realised-COUNT decomposition (F_impossible_counts.pdf + the
"41-1,063 times per window" body number) is still the 5 regime windows {2015,2019,2020,2023,2025}
-- which now cover only 3 of the 5 breaching windows (missing 2021, 2024).

Method mirrors `forensics_calib.py` task 4 exactly (so the new counts are comparable to the
committed 5-window ones), but stripped to counts only -- NO model, NO subsample, NO predictions:
per window, ONE decode of the test slice -> F.target_indices + G._origin_index -> EXACT count of
each of the six flagged (origin,dest) cells, with the test_key_hash identity check against
evaluate_summary.json. Heavy SSD scan (full test slice per window, ~minutes each); needs the drive.

Run:  PYTHONPATH=src .venv/bin/python -u experiments/gbt_regime_read/make_counts_11w.py

After it runs, it prints the new ranges and regenerates the figure. The remaining text edits in
chapter4.tex §4.8 are then mechanical (it prints the exact before/after strings):
  - F_impossible_counts caption: "range across the five windows" -> "...eleven windows"
  - body: "between 41 and 1{,}063 times per window" -> the printed legal range
  - body/caption impossible "0--10" -> the printed impossible range (if it widened)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from floan.model import config, data as D, features as F, gbt as G

VARIANT = "full"
WINDOWS = list(config.TEST_YEARS)                       # all 11
OUT = Path("experiments/gbt_regime_read")
SI = F.STATE_INDEX
ORIGIN_IDX = {s: i for i, s in enumerate(config.ORIGIN_STATES)}

# the six cells the OLD mask flagged: 4 reporting-gap-legal + 2 mechanically-impossible
CELLS = [
    ("current", "dpd_60",      "legal",      r"current $\to$ 60 DPD"),
    ("current", "dpd_90plus",  "legal",      r"current $\to$ 90+ DPD"),
    ("dpd_30",  "dpd_90plus",  "legal",      r"30 DPD $\to$ 90+ DPD"),
    ("dpd_90plus", "REO",      "legal",      r"90+ DPD $\to$ REO"),
    ("current", "foreclosure", "impossible", r"current $\to$ foreclosure"),
    ("current", "REO",         "impossible", r"current $\to$ REO"),
]
TEAL, RED = "#1a5f6e", "#b22222"

config.require_drive()

# committed per-window test_key_hash (identity: prove we scan the exact frozen test rows)
EV = json.loads((config.MODELS / "nn" / VARIANT / "evaluate_summary.json").read_text())
EV_KEYHASH = {}
for c in EV["accept"]["checks"]:
    s = c["check"]
    if s.startswith("k=") and "key_hash=" in s:
        EV_KEYHASH[int(s[2:6])] = int(s.split("key_hash=")[1].split()[0])


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


counts = {f"{o}->{d}": {} for o, d, _, _ in CELLS}      # cell -> {window: realised count}
n_by_window = {}

for k in WINDOWS:
    t0 = time.perf_counter()
    run = config.MODELS / "gbt" / VARIANT / f"k{k}"
    gm = json.loads((run / "metrics.json").read_text())
    pool_dir, (lo, hi) = D.window_spec(VARIANT, k, "test")
    raw = (pl.scan_parquet(str(pool_dir / "part-*.parquet"))
             .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
             .select(D.RAW_COLS).collect())
    n = raw.height
    y = F.target_indices(raw).astype(np.int64)
    origin = G._origin_index(raw).astype(np.int64)

    # identity: row count + test_key_hash against the committed eval rows
    assert n == gm["rows"]["test"], (k, n, gm["rows"]["test"])
    kh = (raw.select(pl.struct(["Loan Identifier", "period_ym"]).hash(seed=0))
             .to_numpy().reshape(-1).astype(np.uint64))
    tkh = int(np.bitwise_xor.reduce(kh ^ y.astype(np.uint64)))
    if k in EV_KEYHASH:
        assert tkh == EV_KEYHASH[k], (k, tkh, EV_KEYHASH[k])
        idok = "identity OK"
    else:
        idok = "identity: no committed key_hash (count still exact)"
    del raw, kh

    for o, d, _, _ in CELLS:
        counts[f"{o}->{d}"][k] = int(((origin == ORIGIN_IDX[o]) & (y == SI[d])).sum())
    n_by_window[k] = n
    log(f"k{k}: n={n:,} | {idok} | {time.perf_counter() - t0:.0f}s")

# ---- ranges (the numbers the prose/caption need) ----
legal_all = [counts[f"{o}->{d}"][k] for o, d, t, _ in CELLS if t == "legal" for k in WINDOWS]
imp_all = [counts[f"{o}->{d}"][k] for o, d, t, _ in CELLS if t == "impossible" for k in WINDOWS]
legal_lo, legal_hi = min(legal_all), max(legal_all)
imp_lo, imp_hi = min(imp_all), max(imp_all)

print("\n=== realised counts, all 11 windows ===")
hdr = "cell".ljust(26) + "  " + " ".join(f"{k}" for k in WINDOWS) + "   range"
print(hdr)
for o, d, t, _ in CELLS:
    row = counts[f"{o}->{d}"]
    rng = f"{min(row.values())}-{max(row.values())}"
    print(f"{(o+'->'+d):26}  " + " ".join(f"{row[k]:>5}" for k in WINDOWS) + f"   {rng}  [{t}]")
print(f"\nLEGAL cells overall:      {legal_lo}-{legal_hi}   (5-window was 41-1063)")
print(f"IMPOSSIBLE cells overall: {imp_lo}-{imp_hi}   (5-window was 0-10)")
print("\n--- SUGGESTED chapter4.tex §4.8 edits (copy-paste) ---")
print(f'  body:    "between 41 and 1{{,}}063 times per window"  ->  "between {legal_lo} and {legal_hi:,} times per window"'.replace(",", "{,}"))
print(f'  caption: "range across the five windows"  ->  "range across the eleven windows"')
print(f'  caption: "occur 41--1{{,}}063 times"  ->  "occur {legal_lo}--{legal_hi:,} times"'.replace(f"{legal_hi:,}", f"{legal_hi:,}".replace(",", "{,}")))
print(f'  impossible "0--10"  ->  "{imp_lo}--{imp_hi}"  (only if changed)')

# ---- regenerate F_impossible_counts.pdf, 11 windows ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 6))
yvals = list(range(len(CELLS)))[::-1]                   # first cell at top
for yi, (o, d, t, label) in zip(yvals, CELLS):
    cs = [counts[f"{o}->{d}"][k] for k in WINDOWS]
    col = TEAL if t == "legal" else RED
    pts = [max(c, 0.5) for c in cs]                     # 0 counts -> log floor
    ax.plot([min(pts), max(pts)], [yi, yi], color=col, lw=6, solid_capstyle="round", alpha=0.85, zorder=1)
    ax.scatter(pts, [yi] * len(pts), s=26, facecolor="white", edgecolor=col, lw=1.3, zorder=2)
    ax.text(max(pts) * 1.25, yi, f"{min(cs)}–{max(cs)}", va="center", ha="left",
            fontsize=10, fontweight="bold", color=col)
ax.axvline(1.0, color="grey", ls=":", lw=1)
ax.set_yticks(yvals)
ax.set_yticklabels([label for _, _, _, label in CELLS], fontsize=11)
ax.set_xscale("log")
ax.set_xlim(0.4, 5e3)
ax.set_xlabel("realised count in held-out data  (range across the eleven windows, log scale)", fontsize=11)
# No in-figure title — the LaTeX caption supplies it.
ax.text(0.985, 0.46, "legal\n(teal)", transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold", color=TEAL)
ax.text(0.985, 0.12, f"impossible\n(red): {imp_lo}–{imp_hi} errors", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, fontweight="bold", color=RED)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
figp = Path("artifacts/generated/F_impossible_counts")
figp.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{figp}.pdf", bbox_inches="tight")
fig.savefig(f"{figp}.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {figp}.pdf / .png")

# ---- persist counts (provenance; 11-window companion to impossible_cell_fix_spec.json) ----
payload = {
    "task": "Edit 8 — realised counts, all 11 windows",
    "method": "exact full-slice count per window (forensics_calib.py task-4 method, counts only)",
    "windows": WINDOWS, "n_test_by_window": n_by_window,
    "realized_counts": {f"{o}->{d}": counts[f"{o}->{d}"] for o, d, _, _ in CELLS},
    "legal_range": [legal_lo, legal_hi], "impossible_range": [imp_lo, imp_hi],
}
(OUT / "impossible_counts_11w.json").write_text(json.dumps(payload, indent=2, default=int))
print(f"wrote {OUT/'impossible_counts_11w.json'}\nDONE")
