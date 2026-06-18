"""Derive the canonical impossible-cell fix spec from forensics_calib_results.json.

Emits experiments/gbt_regime_read/impossible_cell_fix_spec.json — the machine-readable split that
M19 (and any future pass) READS instead of re-deriving the forensic forbidden-vs-legal call. Reads
the per-window EXACT realized counts straight from the forensics output so prose and artifact cannot
drift. PRELIMINARY: 5 of 11 windows.

Run:  PYTHONPATH=src .venv/bin/python experiments/gbt_regime_read/make_fix_spec.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path("experiments/gbt_regime_read")
SRC = HERE / "forensics_calib_results.json"
OUT = HERE / "impossible_cell_fix_spec.json"
WINDOWS = ["2015", "2019", "2020", "2023", "2025"]

# (b) permit — flagged-but-realized cells (reporting-gap legal); permitting them makes the QA gate
#     fire only on genuinely-impossible cells. (a) zero — mechanically-impossible loss cells.
PERMIT = [("current", "dpd_60"), ("current", "dpd_90plus"),
          ("dpd_30", "dpd_90plus"), ("dpd_90plus", "REO")]
ZERO = [("current", "foreclosure"), ("current", "REO")]

res = json.loads(SRC.read_text())


def cell_lookup(window: str, origin: str, dest: str) -> dict:
    """Find the (origin,dest) cell record in a window's forensics 'cells' list (or None)."""
    for c in res[window]["cells"]:
        if c["origin"] == origin and c["dest"] == dest:
            return c
    return None


def realized_by_window(origin: str, dest: str) -> dict:
    out = {}
    for w in WINDOWS:
        c = cell_lookup(w, origin, dest)
        out[w] = int(c["realized_count_full"]) if c else 0
    return out


def permit_contrib_by_window() -> dict:
    """Per-window predicted mean-mass contribution carried by the PERMIT set (rescaled est)."""
    out = {}
    for w in WINDOWS:
        s = 0.0
        for o, d in PERMIT:
            c = cell_lookup(w, o, d)
            if c:
                s += float(c["est_exact_contrib"])
        out[w] = s
    return out


permit_contrib = permit_contrib_by_window()
residual = {w: max(0.0, float(res[w]["exact_mean_mass"]) - permit_contrib[w]) for w in WINDOWS}

spec = {
    "spec": "impossible_cell_fix",
    "status": "PRELIMINARY — 5 of 11 regime windows; filler-window confirmation pending",
    "schema": {
        "states_order": ["current", "dpd_30", "dpd_60", "dpd_90plus", "foreclosure", "REO", "prepaid"],
        "cells": "(origin, destination) string pairs in the canonical state names",
    },
    "mask_permit_cells": [
        {"origin": o, "destination": d, "tag": "reporting-gap-legal",
         "rationale": "occurs in real test data (loan genuinely advanced past a missed monthly "
                      "snapshot); the mask over-flags it as impossible",
         "realized_count": realized_by_window(o, d)}
        for o, d in PERMIT
    ],
    "zero_cells": [
        {"origin": o, "destination": d, "tag": "mechanically-impossible / data-error",
         "rationale": "a performing loan cannot reach foreclosure/REO in one month even across a "
                      "reporting gap; the few realized instances are data errors",
         "note": "high-LGD loss state — the ~1e-5 predicted mass is severity-amplified in pricing, "
                 "so zero-and-renormalize this cell before the M19 pool roll",
         "realized_count": realized_by_window(o, d)}
        for o, d in ZERO
    ],
    "threshold_unchanged": True,
    "threshold_note": ("After the (b) permit, the residual (genuinely-impossible) mean mass is "
                       "~1e-5 per window and all 5 regime windows clear the 1e-3 mean gate WITHOUT "
                       "moving the threshold. Residual = exact_mean_mass − permitted-cell mass."),
    "residual_after_permit_per_window": {w: round(residual[w], 8) for w in WINDOWS},
    "exact_mean_mass_per_window": {w: float(res[w]["exact_mean_mass"]) for w in WINDOWS},
    "gate_pass_before_fix": {w: bool(res[w]["gate_pass"]) for w in WINDOWS},
    "provenance": {
        "source": "experiments/gbt_regime_read/forensics_calib_results.json",
        "derivation": "experiments/gbt_regime_read/make_fix_spec.py",
        "windows": [int(w) for w in WINDOWS],
        "structural_mask": "floan.model.backtest._structural_allow",
        "realized_counts": "EXACT (full-slice scan); predicted contributions rescaled from a 20% "
                           "subsample to the committed exact mean (×0.99–1.006)",
        "location_rationale": "co-located with provenance under experiments/gbt_regime_read/ rather "
                              "than a permanent config home because it is PRELIMINARY (5/11 windows); "
                              "the permanent fix is the eventual _structural_allow code edit + M19 "
                              "config promotion once all 11 windows confirm",
        "status": "PRELIMINARY — filler-window confirmation pending",
    },
}

OUT.write_text(json.dumps(spec, indent=2))
print(f"wrote {OUT}")
print("residual_after_permit_per_window:", spec["residual_after_permit_per_window"])
print("all windows clear 1e-3 after permit:",
      all(v < 1e-3 for v in residual.values()))
