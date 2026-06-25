"""M27a three-arm rolling table: ff_base / ff_hist / gru test NLL per window + deltas + pooled.
Reads artifacts in m27a_gpu_runs/ (gru = seed 0; ff arms = single seed 0)."""
from __future__ import annotations

import json
from pathlib import Path

from floan.model import config as mc

RES = mc.OUTPUTS / "m27a_gpu_runs"
WINDOWS = list(range(2015, 2026))

def jl(name):
    return json.loads((RES / name).read_text())

def main():
    rows = {}
    for k in WINDOWS:
        rows[k] = {
            "ff_base": jl(f"k{k}_ff_base.json"),
            "ff_hist": jl(f"k{k}_ff_hist.json"),
            "gru": jl(f"k{k}_s0.json"),
        }

    print("=" * 96)
    print("M27a THREE-ARM ROLLING TABLE — test NLL (single seed 0, SAME 1.5M train points, full test)")
    print("=" * 96)
    print(f"{'test_yr':>7} | {'ff_base':>9} {'ff_hist':>9} {'gru':>9} | "
          f"{'gru−base':>9} {'gru−hist':>9} | {'n_test':>11}")
    print("-" * 96)
    for k in WINDOWS:
        b = rows[k]["ff_base"]["test_nll"]; h = rows[k]["ff_hist"]["test_nll"]; g = rows[k]["gru"]["test_nll"]
        n = rows[k]["gru"]["n_test"]
        # sanity: all three arms must be scored on the same n_test
        assert rows[k]["ff_base"]["n_test"] == rows[k]["ff_hist"]["n_test"] == n, f"n_test mismatch k{k}"
        print(f"{k:>7} | {b:>9.6f} {h:>9.6f} {g:>9.6f} | {g-b:>+9.6f} {g-h:>+9.6f} | {n:>11,}")
    print("-" * 96)

    def pooled(arm_key):
        num = den = 0.0
        for k in WINDOWS:
            j = rows[k][arm_key]
            num += j["test_nll"] * j["n_test"]; den += j["n_test"]
        return num / den, den
    pb, den = pooled("ff_base"); ph, _ = pooled("ff_hist"); pg, _ = pooled("gru")
    print(f"{'POOLED':>7} | {pb:>9.6f} {ph:>9.6f} {pg:>9.6f} | {pg-pb:>+9.6f} {pg-ph:>+9.6f} | {int(den):>11,}")
    print("=" * 96)
    print(f"\nPooled all-years NLL (Σ NLL_w·n_w / Σ n_w, n={int(den):,}):")
    print(f"  ff_base = {pb:.6f}   ff_hist = {ph:.6f}   gru = {pg:.6f}")
    print(f"  ordering: ff_base > ff_hist > gru  (lower=better)  "
          f"-> {'HOLDS' if pb > ph > pg else 'VIOLATED'} (the M26 three-arm decomposition)")
    print(f"  gru − ff_base = {pg-pb:+.6f}   gru − ff_hist = {pg-ph:+.6f}")
    print(f"\nSanity k2015: ff_base={rows[2015]['ff_base']['test_nll']:.6f} (~0.105) "
          f"ff_hist={rows[2015]['ff_hist']['test_nll']:.6f} (~0.099) "
          f"gru={rows[2015]['gru']['test_nll']:.6f} (~0.0969)")

if __name__ == "__main__":
    main()
