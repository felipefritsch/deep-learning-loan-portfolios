# Shared results index — GBT analyses (read before re-scoring anything)

Two GBT analyses are running in parallel against the **same frozen eval slices**. They are
**complementary**, but they share one expensive overlap: each independently reloads a banked
GBT booster and predicts on a frozen test slice. This note says where every number already
lives so neither task (nor the eventual full M18/M19) re-scores a model it doesn't have to.

## 0. Numbers that are ALREADY committed — read these, never recompute

| Quantity | File | Key |
|---|---|---|
| empirical / logit / best-1-NN / ensemble per-window test NLL (all 11 windows) | `models/nn/full/evaluate_summary.json` | `table_b.by_year[<year>]` |
| pooled NLL + pooled NLL by origin (logit/NN/ens/emp) | `models/nn/full/evaluate_summary.json` | `table_b.pooled`, `table_b.by_origin` |
| per-window test_key_hash + row counts (the M11 identity evidence) | `models/nn/full/evaluate_summary.json` | `accept.checks[]` (`key_hash=…`) |
| empirical floor + bucketed per window | `models/benchmarks/full/metrics.json` | `per_window[<year>]` |
| GBT per-window **test NLL**, **best_iteration**, **impossible-mass block** | `models/gbt/full/k<year>/metrics.json` | `gbt.test_nll`, `selection.best_iteration`, `impossible_transitions` |
| **Impossible-cell fix split for M19** (which cells to permit vs zero) — read, do NOT re-derive | `experiments/gbt_regime_read/impossible_cell_fix_spec.json` | `mask_permit_cells`, `zero_cells`, `threshold_unchanged` |
| regime task-4/task-5 forensics + calibration (by-origin/by-cell impossible mass, reliability) | `experiments/gbt_regime_read/forensics_calib_results.json` | per-window `by_origin`, `cells`, `exact_mean_mass` |

GBT overall NLLs are deterministic and frozen (msh=1000) — `metrics.json` is authoritative.
**Re-scoring a GBT booster only buys you something `metrics.json` does NOT store: per-row /
per-origin / per-cell predictions.** If you only need an overall NLL, read it.

## 1. Task A — regime-only read  (`experiments/gbt_regime_read/analyze.py`)

- **Scope:** 5 regime windows {2015, 2019, 2020, 2023, 2025}; GBT vs logit / 1-NN / ensemble /
  empirical, **within-window** only.
- **Produces (NOT in committed JSON):**
  - per-window comparison table + GBT−logit / GBT−1NN / GBT−ens gaps,
  - the COVID (k2020) inversion read,
  - §7 lean per window,
  - **impossible-cell forensics**: mass broken down BY ORIGIN and BY DESTINATION CELL, with a
    forbidden-vs-rare-but-legal verdict per cell (the part that needs the full predict),
  - **GBT raw-softmax calibration** for current→prepaid and current→dpd_30 (pooled + 2020–21).
- **Status: DONE.** Scored via `forensics_calib.py` (impossible by-cell from exact full-slice
  realized counts + 20%-subsample predicted shares, reproducing committed means to ×0.99–1.006) +
  calibration. (The earlier full-predict `analyze.py` was abandoned — see §3.)
- **Outputs:** `experiments/gbt_regime_read/forensics_calib_results.json`,
  `F_gbt_regime_calibration.{png,pdf}`, the M19 fix split
  `experiments/gbt_regime_read/impossible_cell_fix_spec.json`, and the memo
  `writeup/memos/gbt_regime_preliminary.md`.
- **Identity contract:** scores GBT on exactly the frozen rows the other models used — proven by
  (row count == committed) + (test_key_hash == evaluate_summary) + (reproduces GBT `metrics.json`
  test_nll & impossible mass to ≤1e-9).
- **Headline already locked from committed metrics** (does NOT need the running predict):

  | year | floor | logit | 1-NN | ens×8 | GBT | GBT−1NN | impossible gate |
  |---|---|---|---|---|---|---|---|
  | 2015 tuning | 0.11287 | 0.10825 | 0.10312 | 0.10266 | 0.10360 | +0.00048 | PASS (3.47e-4) |
  | 2019 calm | 0.10967 | 0.10635 | 0.10060 | 0.10048 | 0.10350 | +0.00290 | FAIL (1.86e-3, best_iter 130) |
  | 2020 COVID | 0.19358 | 0.19195 | 0.18487 | 0.18341 | 0.18412 | **−0.00075** | FAIL (1.56e-3, best_iter 1919) |
  | 2023 rate-spike | 0.07913 | 0.07494 | 0.06930 | 0.06932 | 0.06968 | +0.00038 | PASS (9.14e-4) |
  | 2025 latest | 0.08558 | 0.08216 | 0.07516 | 0.07509 | 0.07698 | +0.00182 | FAIL (1.43e-3) |

## 2. Task B — per-origin anchors probe  (`experiments/per_origin_anchors/probe.py`)

- **Scope:** **k2015 only**; per-origin (current / dpd_30 / dpd_60 / dpd_90plus) test NLL for the
  banked **logit** and **GBT**, to feed the spline probe ("is GBT's edge over logit *shape* or
  *interaction*, per origin?").
- **Status: DONE** → `experiments/per_origin_anchors/results.json` (+ `gbt_cache.json`). Full predict,
  full cores, k2015 only. Gates pass: logit 0.108246 / GBT 0.103597 reproduced to ≤5e-7.
  **Read:** GBT−logit edge is part-shape on `current` (splines close ~40%) and **pure-interaction on
  the delinquent origins** (plain−spline negative — additive splines overfit, don't close it).
  Two bugs fixed en route: probe loaded the wrong logit class (committed logit is `backtest.LogitEmbNet`,
  not one-hot `LogitNet`); and a LightGBM-OMP/torch-OMP in-process deadlock (torch pinned to 1 thread;
  GBT result cached so the logit-only re-run is ~18 s).
- **Produces (NOT in committed JSON):** per-origin logit NLL, per-origin GBT NLL, and their gap,
  on the k2015 slice → `experiments/per_origin_anchors/results.json`.
- **Reuse:** this is a genuine **full-M18 exhibit** (per-origin NLL by model). When M18 runs,
  pull k2015 per-origin GBT/logit from `results.json` instead of re-scoring.

## 3. The overlap — RESOLVED (was: duplicated k2015 GBT predict)

The original duplication was Task A's full-predict `analyze.py` (killed) re-scoring the k2015 GBT on
the same 6.18M slice that Task B (probe) needs. **Resolved:** A's full predict was abandoned and A
now uses `forensics_calib.py`, which predicts only a **20% subsample** and saves **no per-row preds**
— so it does NOT cover Task B's full-slice per-origin need (verified: `forensics_calib_results.json`
holds impossible-mass aggregates + current-origin calibration only, no per-origin NLL, no per-row
proba). Task B's full k2015 per-origin predict is therefore **genuinely irreducible**, not a
duplicate. The two are now disjoint.

**Reuse matrix:**
- A → B: nothing B needs (B needs per-origin, which A doesn't compute).
- B → A: nothing A needs for the regime memo (A doesn't report per-origin).
- **A & B → future M18:** M18 wants per-origin NLL for **every** model **across all 11 windows**.
  Neither A nor B produces that at full scope. The clean, non-redundant plan for M18 is **one**
  pass that (i) reads all overall NLLs from §0, (ii) re-scores each banked model **once** per
  window saving per-row preds, and (iii) derives per-origin + per-cell + calibration from those
  saved preds. A's `analyze.py` is the closest template; extend it to save per-row preds and add
  the per-origin split, then B becomes a subset of it and need not run again.

## 4. Bottom line for "don't waste / don't repeat"

- Keep both running if cores allow — they finish different deliverables.
- Do **not** re-score any GBT/logit for an **overall** NLL — it's in §0.
- The k2015 GBT predict is duplicated once across A and B; acceptable for throwaway diagnostics,
  but the **full M18 must not** repeat it 11× per model — fold A's identity-checked scorer into a
  single per-row-saving pass and reuse B's per-origin result for the k2015 cell.
