"""FOCUSED tail of the regime read: impossible-cell forensics (task 4) + calibration (task 5).

The overall NLLs, impossible-mass means, and identity hashes are already committed/derivable, so
this does NOT redo a full-row predict. Per window it:
  * ONE decode of the window's test slice (raw cols) → EXACT realized-forbidden counts per
    (origin,dest) cell + EXACT identity (row count + test_key_hash vs evaluate_summary), no model;
  * a deterministic ~20% row subsample → prepare_raw → GBT predict → predicted impossible-mass
    SHARES per cell (rescaled to the committed exact mean) + raw-softmax calibration arrays.

Subsample is unbiased for per-cell mean shares and for the reliability curves; rare-cell legality
is decided on the EXACT full-data realized counts. Run (all cores, unbuffered):
    PYTHONPATH=src .venv/bin/python -u experiments/gbt_regime_read/forensics_calib.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from floan.model import config, data as D, features as F
from floan.model import backtest as B, gbt as G

WINDOWS = [2015, 2019, 2020, 2023, 2025]
VARIANT = "full"
SUB_MOD = 5                       # keep hash%5==0 → ~20% of rows (row-level, deterministic)
STATES = F.STATES
SI = F.STATE_INDEX
PREPAID, DPD30 = SI["prepaid"], SI["dpd_30"]
OUT = Path("experiments/gbt_regime_read")
ALLOW = B._structural_allow()

config.require_drive()

EV = json.loads((config.MODELS / "nn" / VARIANT / "evaluate_summary.json").read_text())
EV_KEYHASH = {}
for c in EV["accept"]["checks"]:
    s = c["check"]
    if s.startswith("k=") and "key_hash=" in s:
        EV_KEYHASH[int(s[2:6])] = int(s.split("key_hash=")[1].split()[0])

records = {}
calib = {"prepaid": [], "dpd30": [], "y": [], "label_year": []}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


for k in WINDOWS:
    t0 = time.perf_counter()
    run = config.MODELS / "gbt" / VARIANT / f"k{k}"
    gm = json.loads((run / "metrics.json").read_text())
    exact_mean = gm["impossible_transitions"]["model_mean_mass"]
    exact_max = gm["impossible_transitions"]["model_max_row_mass"]
    pool_dir, bounds = D.window_spec(VARIANT, k, "test")
    lo, hi = bounds
    parts = str(pool_dir / "part-*.parquet")

    # ---- ONE decode: full raw slice for EXACT realized counts + identity ----
    raw = (pl.scan_parquet(parts)
             .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
             .select(D.RAW_COLS).collect())
    n = raw.height
    y_full = F.target_indices(raw).astype(np.int64)
    origin_full = G._origin_index(raw).astype(np.int64)
    keyhash = (raw.select(pl.struct(["Loan Identifier", "period_ym"]).hash(seed=0))
                  .to_numpy().reshape(-1).astype(np.uint64))
    test_key_hash = int(np.bitwise_xor.reduce(keyhash ^ y_full.astype(np.uint64)))
    assert n == gm["rows"]["test"], (n, gm["rows"]["test"])
    assert test_key_hash == EV_KEYHASH[k], (k, test_key_hash, EV_KEYHASH[k])
    assert set(np.unique(origin_full)).issubset({0, 1, 2, 3})

    forb_full = ~ALLOW[origin_full]
    realized_imp_mean = float(forb_full[np.arange(n), y_full].mean())
    realized_cells = {}          # (o,d) -> exact realized count of that transition
    realized_by_origin = {}
    for o in range(4):
        mo = origin_full == o
        no = int(mo.sum())
        realized_by_origin[o] = no
        for d in range(F.N_CLASSES):
            if ALLOW[o, d]:
                continue
            realized_cells[(o, d)] = int(((origin_full == o) & (y_full == d)).sum())

    # ---- ~20% subsample → predict → predicted impossible-mass shares + calibration ----
    sub = raw.filter(pl.struct(["Loan Identifier", "period_ym"]).hash(seed=1) % SUB_MOD == 0)
    del raw
    df = F.prepare_raw(sub)
    nsub = df.height
    vocab = F.Vocab.from_dict(json.loads((run / "vocab.json").read_text()))
    X = G.build_X(df, vocab)
    booster = lgb.Booster(model_file=str(run / "model.txt"))
    proba = np.empty((nsub, F.N_CLASSES), dtype=np.float64)
    CH = 1_000_000
    for s in range(0, nsub, CH):
        proba[s:s + CH] = booster.predict(X[s:s + CH])
    del X
    y_sub = F.target_indices(df).astype(np.int64)
    origin_sub = G._origin_index(df).astype(np.int64)
    label_year = (df.get_column("label_ym").to_numpy() // 100).astype(np.int32)
    del df

    forb_sub = ~ALLOW[origin_sub]
    imp_row = (proba * forb_sub).sum(axis=1)
    sub_mean = float(imp_row.mean())
    rescale = exact_mean / sub_mean if sub_mean else 1.0   # shares → committed exact mean

    by_origin = {}
    cells = []
    for o in range(4):
        mo = origin_sub == o
        nso = int(mo.sum())
        if nso == 0:
            continue
        by_origin[STATES[o]] = {
            "n_sub": nso,
            "pred_mean_mass_in_origin": float(imp_row[mo].mean()),
            "contrib_share_of_total": float(imp_row[mo].sum() / (sub_mean * nsub)) if sub_mean else 0.0,
        }
        for d in range(F.N_CLASSES):
            if ALLOW[o, d]:
                continue
            pred_contrib_sub = float(proba[mo, d].sum() / nsub)           # mean mass from this cell
            est_exact_contrib = pred_contrib_sub * rescale               # rescaled to committed mean
            share = pred_contrib_sub / sub_mean if sub_mean else 0.0
            rc = realized_cells[(o, d)]
            cells.append({
                "origin": STATES[o], "dest": STATES[d],
                "pred_mean_in_origin": float(proba[mo, d].mean()),
                "share_of_impossible": share,
                "est_exact_contrib": est_exact_contrib,
                "realized_count_full": rc,
                "realized_rate_in_origin_full": rc / realized_by_origin[o] if realized_by_origin[o] else 0.0,
                "verdict": "RARE-BUT-LEGAL (occurs in full test data)" if rc > 0 else "no realized instances",
            })
    cells.sort(key=lambda c: -c["share_of_impossible"])

    cur = origin_sub == SI["current"]
    calib["prepaid"].append(proba[cur, PREPAID].astype(np.float32))
    calib["dpd30"].append(proba[cur, DPD30].astype(np.float32))
    calib["y"].append(y_sub[cur].astype(np.int8))
    calib["label_year"].append(label_year[cur])

    records[k] = {
        "n": n, "n_sub": nsub, "test_key_hash": test_key_hash,
        "best_iteration": gm["selection"]["best_iteration"],
        "exact_mean_mass": exact_mean, "exact_max_row": exact_max,
        "sub_mean_mass": sub_mean, "realized_imp_mean": realized_imp_mean,
        "gate_pass": exact_mean < 1e-3, "by_origin": by_origin, "cells": cells,
    }
    del proba
    log(f"k{k}: n={n:,} sub={nsub:,} | identity OK | exact_mean={exact_mean:.3e} "
        f"sub_mean={sub_mean:.3e} (×{rescale:.3f}) | {time.perf_counter()-t0:.0f}s")

# ======================= TASK 4 REPORT =======================
print("\n" + "=" * 104)
print("TASK 4 — IMPOSSIBLE-CELL FORENSICS (exact realized counts; predicted shares from ~20% subsample)")
print("=" * 104)
for k in WINDOWS:
    r = records[k]
    print(f"\n--- k{k} --- exact mean_mass={r['exact_mean_mass']:.3e} "
          f"({'PASS' if r['gate_pass'] else 'FAIL'} 1e-3)  max_row={r['exact_max_row']:.4f}  "
          f"realized_forbidden_mean={r['realized_imp_mean']:.3e}  best_iter={r['best_iteration']}")
    print("  by ORIGIN (predicted mass share):")
    for s, b in r["by_origin"].items():
        print(f"    {s:11s} share_of_total={b['contrib_share_of_total']:6.1%}  "
              f"pred_mean_mass_in_origin={b['pred_mean_mass_in_origin']:.3e}")
    print("  forbidden DEST cells (share of impossible mass; realized count = EXACT full data):")
    for c in r["cells"][:6]:
        print(f"    {c['origin']:>10s}->{c['dest']:<11s} share={c['share_of_impossible']:6.1%}  "
              f"est_contrib={c['est_exact_contrib']:.3e}  realized_full={c['realized_count_full']:>7,} "
              f"({c['realized_rate_in_origin_full']:.2e})  {c['verdict']}")

# ======================= TASK 5 CALIBRATION =======================
print("\n" + "=" * 104)
print("TASK 5 — CALIBRATION (raw GBT softmax, current-origin pooled over 5 regime windows; ~20% subsample)")
print("=" * 104)
prepaid = np.concatenate(calib["prepaid"])
dpd30 = np.concatenate(calib["dpd30"])
yc = np.concatenate(calib["y"]).astype(np.int64)
ly = np.concatenate(calib["label_year"])
covid = np.isin(ly, [2020, 2021])


def reliability(pred, label, nbins=12):
    qs = np.unique(np.quantile(pred, np.linspace(0, 1, nbins + 1)))
    if len(qs) < 3:
        return np.array([pred.mean()]), np.array([label.mean()])
    idx = np.clip(np.searchsorted(qs[1:-1], pred), 0, len(qs) - 2)
    mp, mo = [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() >= 50:
            mp.append(float(pred[m].mean())); mo.append(float(label[m].mean()))
    return np.array(mp), np.array(mo)


curves = {}
for dest_name, pred, di in (("current->prepaid", prepaid, PREPAID), ("current->dpd_30", dpd30, DPD30)):
    lab = (yc == di).astype(np.float64)
    print(f"\n[{dest_name}]")
    for pan, mask in (("Pooled", np.ones_like(covid)), ("2020-21", covid), ("Other", ~covid)):
        p, l = pred[mask], lab[mask]
        mp, mo = reliability(p, l)
        bias = float((mp - mo).mean())
        side = ("ON-diagonal" if abs(bias) < 0.05 * max(l.mean(), 1e-9)
                else ("ABOVE/over-predicts" if bias > 0 else "BELOW/under-predicts"))
        curves[(dest_name, pan)] = (mp, mo)
        print(f"  {pan:8s} n={p.shape[0]:>10,}  pred_mean={p.mean():.4e}  obs_mean={l.mean():.4e}  "
              f"mean_bin_bias={bias:+.4e} -> {side}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ri, dest_name in enumerate(("current->prepaid", "current->dpd_30")):
        for ci, pan in enumerate(("Pooled", "2020-21", "Other")):
            ax = axes[ri, ci]
            mp, mo = curves[(dest_name, pan)]
            lim = float(max(mp.max(), mo.max()) * 1.1) if len(mp) else 0.1
            ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1)
            ax.plot(mp, mo, marker="o", ms=4, c="C3", label="GBT (raw)")
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_title(f"{dest_name} — {pan}")
            ax.set_xlabel("Mean predicted"); ax.grid(alpha=0.3)
            if ci == 0:
                ax.set_ylabel("Observed freq"); ax.legend()
    # No in-figure title — the LaTeX caption supplies it.
    p = OUT / "F_gbt_regime_calibration"
    fig.savefig(f"{p}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{p}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}.png/.pdf")
except Exception as e:
    print(f"[figure skipped: {e}]")

(OUT / "forensics_calib_results.json").write_text(json.dumps(records, indent=2, default=float))
print(f"wrote {OUT/'forensics_calib_results.json'}\nDONE")
