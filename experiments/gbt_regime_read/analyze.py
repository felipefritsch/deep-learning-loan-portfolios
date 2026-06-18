"""GBT regime-only read (Step 3): 5 regime windows, GBT vs logit/NN/ensemble/empirical.

REGIME-ONLY — does NOT compute the pooled-all-11 NLL and does NOT touch the pool engine.
Scores GBT on the *same* frozen per-window test rows the other models used, proving identity
three independent ways per window:
  (1) row count == GBT/logit/NN/benchmarks stored test rows,
  (2) reproduces evaluate.py's stored test_key_hash (ties our rows to the M11 eval rows),
  (3) reproduces GBT's own stored test_nll + impossible-cell mass (ties our rows to the fit).

Run:  PYTHONPATH=src .venv/bin/python experiments/gbt_regime_read/analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from floan.model import config, data as D, features as F
from floan.model import evaluate as E, backtest as B, gbt as G

WINDOWS = [2015, 2019, 2020, 2023, 2025]
VARIANT = "full"
STATES = F.STATES
SI = F.STATE_INDEX
ORIGIN_STATES = list(config.ORIGIN_STATES)          # current, dpd_30, dpd_60, dpd_90plus
PREPAID, DPD30 = SI["prepaid"], SI["dpd_30"]
OUT = Path("experiments/gbt_regime_read")

config.require_drive()

# ---- authoritative baselines (already committed) -----------------------------
EV = json.loads((config.MODELS / "nn" / VARIANT / "evaluate_summary.json").read_text())
BENCH = json.loads((config.MODELS / "benchmarks" / VARIANT / "metrics.json").read_text())
BY_YEAR = EV["table_b"]["by_year"]                  # empirical/logit/nn/ensemble per window
# stored evaluate key_hash per window (from the M11 accept evidence string)
EV_KEYHASH = {}
for c in EV["accept"]["checks"]:
    s = c["check"]
    if s.startswith("k=") and "key_hash=" in s:
        k = int(s[2:6])
        EV_KEYHASH[k] = int(s.split("key_hash=")[1].split()[0])

ALLOW = B._structural_allow()                       # [7,7] bool; True=structurally reachable

calib = {"prepaid": [], "dpd30": [], "y": [], "label_year": []}   # pooled current-origin rows
records = {}

for k in WINDOWS:
    run = config.MODELS / "gbt" / VARIANT / f"k{k}"
    gm = json.loads((run / "metrics.json").read_text())

    # --- load the frozen test slice once (identical loader to gbt + evaluate) ---
    pool_dir, bounds = D.window_spec(VARIANT, k, "test")
    df = F.prepare_raw(D._masked_scan(pool_dir, bounds).collect())
    n = df.height
    y = F.target_indices(df).astype(np.int64)
    origin = G._origin_index(df).astype(np.int64)            # 0..6 via STATE_INDEX
    label_year = (df.get_column("label_ym").to_numpy() // 100).astype(np.int32)
    keyhash = (df.select(pl.struct(["Loan Identifier", "period_ym"]).hash(seed=0))
                 .to_numpy().reshape(-1).astype(np.uint64))
    test_key_hash = int(np.bitwise_xor.reduce(keyhash ^ y.astype(np.uint64)))

    # --- GBT predict on its own vocab/layout ---
    vocab = F.Vocab.from_dict(json.loads((run / "vocab.json").read_text()))
    X = G.build_X(df, vocab)
    booster = lgb.Booster(model_file=str(run / "model.txt"))   # saved at best_iteration
    proba = np.empty((n, F.N_CLASSES), dtype=np.float64)
    CH = 1_000_000
    for s in range(0, n, CH):
        proba[s:s + CH] = booster.predict(X[s:s + CH])
    del X

    gbt_nll = E._nll(proba, y)

    # --- identity assertions (row count + hash + NLL/impossible reproduction) ---
    bench_n = BENCH["per_window"][str(k)]["n_test"]
    assert n == gm["rows"]["test"] == bench_n, (n, gm["rows"]["test"], bench_n)
    assert test_key_hash == EV_KEYHASH[k], (k, test_key_hash, EV_KEYHASH[k])
    assert abs(gbt_nll - gm["gbt"]["test_nll"]) < 1e-9, (gbt_nll, gm["gbt"]["test_nll"])
    assert set(np.unique(origin)).issubset({0, 1, 2, 3}), np.unique(origin)

    # --- impossible-cell forensics ---
    forbidden = ~ALLOW[origin]                              # [N,7]
    imp_row = (proba * forbidden).sum(axis=1)
    realized_imp = forbidden[np.arange(n), y]
    mean_mass = float(imp_row.mean())
    max_row = float(imp_row.max())
    realized_mean = float(realized_imp.mean())
    # reproduce GBT's stored impossible block (4th identity tie)
    assert abs(mean_mass - gm["impossible_transitions"]["model_mean_mass"]) < 1e-9
    assert abs(max_row - gm["impossible_transitions"]["model_max_row_mass"]) < 1e-6

    by_origin = {}
    cells = []
    for o in range(4):
        mo = origin == o
        no = int(mo.sum())
        if no == 0:
            continue
        contrib = float(imp_row[mo].sum() / n)             # share of overall mean mass
        by_origin[STATES[o]] = {
            "n": no, "frac_rows": no / n,
            "mean_mass_in_origin": float(imp_row[mo].mean()),
            "contrib_to_overall_mean": contrib,
            "share_of_impossible": contrib / mean_mass if mean_mass else 0.0,
        }
        for d in range(F.N_CLASSES):
            if ALLOW[o, d]:
                continue                                   # only forbidden (o,d) cells
            pred_contrib = float(proba[mo, d].sum() / n)   # contribution to overall mean
            realized_count = int(((origin == o) & (y == d)).sum())
            cells.append({
                "origin": STATES[o], "dest": STATES[d],
                "pred_mean_in_origin": float(proba[mo, d].mean()),
                "pred_contrib_overall": pred_contrib,
                "share_of_impossible": pred_contrib / mean_mass if mean_mass else 0.0,
                "realized_count": realized_count,
                "realized_rate_in_origin": realized_count / no,
            })
    cells.sort(key=lambda c: -c["pred_contrib_overall"])

    # --- calibration accumulation (current-origin rows only) ---
    cur = origin == SI["current"]
    calib["prepaid"].append(proba[cur, PREPAID].astype(np.float32))
    calib["dpd30"].append(proba[cur, DPD30].astype(np.float32))
    calib["y"].append(y[cur].astype(np.int8))
    calib["label_year"].append(label_year[cur])

    records[k] = {
        "n": n, "test_key_hash": test_key_hash, "best_iteration": gm["selection"]["best_iteration"],
        "gbt_nll": gbt_nll,
        "empirical": BY_YEAR[str(k)]["empirical"], "logit": BY_YEAR[str(k)]["logit"],
        "nn": BY_YEAR[str(k)]["nn"], "ensemble": BY_YEAR[str(k)].get("ensemble"),
        "floor": BENCH["per_window"][str(k)]["nll_empirical"],
        "bucketed": BENCH["per_window"][str(k)]["nll_bucketed"],
        "imp_mean_mass": mean_mass, "imp_max_row": max_row, "imp_realized_mean": realized_mean,
        "imp_gate_pass": mean_mass < 1e-3,
        "by_origin": by_origin, "cells": cells,
    }
    del proba, df
    print(f"[k{k}] scored {n:,} rows | identity OK (rows+hash+nll+impossible) | "
          f"GBT NLL={gbt_nll:.6f} | imp_mean={mean_mass:.3e} "
          f"({'PASS' if mean_mass < 1e-3 else 'FAIL'})")

# =============================================================================
# 1. PER-WINDOW COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 100)
print("1. PER-WINDOW COMPARISON — out-of-sample test NLL (compare WITHIN a window only)")
print("=" * 100)
hdr = (f"{'win':>5} | {'floor(emp)':>10} {'bucketed':>9} | {'logit':>8} {'1-NN':>8} "
       f"{'ens×8':>8} {'GBT':>8} | {'GBT-logit':>10} {'GBT-1NN':>9} {'GBT-ens':>9}")
print(hdr); print("-" * len(hdr))
for k in WINDOWS:
    r = records[k]
    g_lo = r["gbt_nll"] - r["logit"]
    g_nn = r["gbt_nll"] - r["nn"]
    g_en = r["gbt_nll"] - r["ensemble"] if r["ensemble"] is not None else None
    print(f"{k:>5} | {r['floor']:>10.5f} {r['bucketed']:>9.5f} | "
          f"{r['logit']:>8.5f} {r['nn']:>8.5f} {r['ensemble']:>8.5f} {r['gbt_nll']:>8.5f} | "
          f"{g_lo:>+10.5f} {g_nn:>+9.5f} {g_en:>+9.5f}")
print("\nNLL is NOT comparable across windows (each test year is a different difficulty).")

# =============================================================================
# 2 & 3.  §7 lean per window
# =============================================================================
print("\n" + "=" * 100)
print("2/3. §7 LEAN per window  (gap sign: negative = GBT better)")
print("=" * 100)
TIE = 5e-4    # |GBT-1NN| within this => effective tie (flexibility wins, architecture secondary)
for k in WINDOWS:
    r = records[k]
    g_nn = r["gbt_nll"] - r["nn"]
    g_en = r["gbt_nll"] - r["ensemble"]
    if abs(g_nn) <= TIE:
        lean = "TIE (flexibility wins, architecture secondary)"
    elif g_nn < 0:
        lean = "GBT-AHEAD (trees beat the single net)"
    else:
        lean = "NET-AHEAD (smooth target rewards the smooth learner)"
    print(f"  k{k}: GBT-1NN={g_nn:+.5f}  GBT-ens={g_en:+.5f}  ->  {lean}")

# =============================================================================
# 4. IMPOSSIBLE-CELL FORENSICS
# =============================================================================
print("\n" + "=" * 100)
print("4. IMPOSSIBLE-CELL FORENSICS  (mass by origin + by destination cell)")
print("=" * 100)
for k in WINDOWS:
    r = records[k]
    print(f"\n--- k{k} ---  mean_mass={r['imp_mean_mass']:.3e} "
          f"({'PASS' if r['imp_gate_pass'] else 'FAIL'} 1e-3 gate)  "
          f"max_row={r['imp_max_row']:.4f}  realized_mean={r['imp_realized_mean']:.3e}  "
          f"best_iter={r['best_iteration']}")
    print("  by ORIGIN:")
    for s, b in r["by_origin"].items():
        print(f"    {s:11s} n={b['n']:>9,} ({b['frac_rows']:6.2%} of rows)  "
              f"mean_mass_in_origin={b['mean_mass_in_origin']:.3e}  "
              f"share_of_impossible={b['share_of_impossible']:6.1%}")
    print("  top forbidden DEST cells (by contribution to overall mean mass):")
    for c in r["cells"][:6]:
        verdict = ("RARE-BUT-LEGAL (occurs in test data)" if c["realized_count"] > 0
                   else "no realized instances")
        print(f"    {c['origin']:>10s}->{c['dest']:<11s}  "
              f"pred_in_origin={c['pred_mean_in_origin']:.3e}  "
              f"share={c['share_of_impossible']:6.1%}  "
              f"realized={c['realized_count']:>6,} ({c['realized_rate_in_origin']:.2e})  {verdict}")

# =============================================================================
# 5. CALIBRATION (raw GBT softmax; current->prepaid and current->dpd_30)
# =============================================================================
print("\n" + "=" * 100)
print("5. CALIBRATION — raw GBT softmax, current-origin pooled over 5 regime windows")
print("=" * 100)
prepaid = np.concatenate(calib["prepaid"])
dpd30 = np.concatenate(calib["dpd30"])
yc = np.concatenate(calib["y"]).astype(np.int64)
ly = np.concatenate(calib["label_year"])
covid = np.isin(ly, [2020, 2021])


def reliability(pred, label, nbins=12):
    qs = np.unique(np.quantile(pred, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.searchsorted(qs[1:-1], pred), 0, len(qs) - 2)
    mp, mo, cnt = [], [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() >= 20:
            mp.append(float(pred[m].mean())); mo.append(float(label[m].mean()))
            cnt.append(int(m.sum()))
    return np.array(mp), np.array(mo), np.array(cnt)


def summarize(name, pred, lab, mask=None):
    if mask is not None:
        pred, lab = pred[mask], lab[mask]
    mp, mo, cnt = reliability(pred, lab)
    # signed mean over-prediction across populated bins; overall pred vs obs
    bias = float((mp - mo).mean())
    side = "ON-diagonal" if abs(bias) < 0.05 * max(lab.mean(), 1e-9) else (
        "ABOVE (over-predicts)" if bias > 0 else "BELOW (under-predicts)")
    print(f"  {name:28s} n={pred.shape[0]:>10,}  pred_mean={pred.mean():.4e}  "
          f"obs_mean={lab.mean():.4e}  mean_bin_bias={bias:+.4e}  -> {side}")
    return mp, mo


curves = {}
for dest_name, pred in (("current->prepaid", prepaid), ("current->dpd_30", dpd30)):
    di = PREPAID if "prepaid" in dest_name else DPD30
    lab = (yc == di).astype(np.float64)
    print(f"\n[{dest_name}]")
    curves[(dest_name, "Pooled")] = summarize("Pooled (all 5 windows)", pred, lab)
    curves[(dest_name, "2020-21")] = summarize("2020-21 (COVID)", pred, lab, covid)
    curves[(dest_name, "Other")] = summarize("Other years", pred, lab, ~covid)

# --- figure ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex="row", sharey="row")
    panels = ["Pooled", "2020-21", "Other"]
    for ri, dest_name in enumerate(("current->prepaid", "current->dpd_30")):
        for ci, pan in enumerate(panels):
            ax = axes[ri, ci]
            mp, mo = curves[(dest_name, pan)]
            lim = float(max(mp.max(), mo.max()) * 1.1) if len(mp) else 0.1
            ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1)
            ax.plot(mp, mo, marker="o", ms=4, c="C3", label="GBT (raw)")
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_title(f"{dest_name} — {pan}")
            ax.set_xlabel("Mean predicted prob"); ax.grid(alpha=0.3)
            if ci == 0:
                ax.set_ylabel("Observed frequency"); ax.legend()
    fig.suptitle("GBT raw-softmax calibration — 5 regime windows (current origin)")
    p = OUT / "F_gbt_regime_calibration"
    fig.savefig(f"{p}.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{p}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}.png / .pdf")
except Exception as e:
    print(f"\n[figure skipped: {e}]")

# --- dump machine-readable record ---
(OUT / "regime_read_results.json").write_text(json.dumps(records, indent=2, default=float))
print(f"wrote {OUT/'regime_read_results.json'}")
