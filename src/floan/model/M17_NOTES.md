# M17 — GBT rolling sweep (PARTIAL run: banked k=2015 + k=2019; 9 windows deferred)

**Status (2026-06-17, branch `gbt-baseline`):** the full-scale GBT sweep was started on the
16 GB Mac and **stopped cleanly at a window boundary** after banking the two fastest regime
windows. The remaining **9 windows are deferred to a high-RAM box** (the 16 GB ceiling — see
§4). This file is the resume record.

## 1. Frozen config (M17 reconfirm)
`config.GBT_SELECTED = {num_leaves 31, learning_rate 0.05, min_sum_hessian_in_leaf 1000,
feature_fraction 1.0, lambda_l2 0}`. The full-scale reconfirm on the k=2015 tuning window
**moved `min_sum_hessian_in_leaf` 100 → 1000** (it is an absolute summed-leaf-Hessian
threshold; the ~34 M-row full slice carries ~10× the summed Hessian of the 3.15 M dev slice,
so the dev-calibrated 100 was ~10× too weak). Evidence + rationale: `config.py` and
`models/gbt/full/k2015_reconfirm/metrics.json`. `backtest.py` loops this config.

## 2. Completed windows (banked, mirrored to `ssd_mirror/`)
Test NLL on the frozen test slice; GBT vs the per-window logit and best NN.

| window | GBT test NLL | zone | QA | best_iter | vs logit | vs NN |
|---|---|---|---|---|---|---|
| k=2015 | 0.103597 | HEALTHY | all pass | 1208 | logit 0.108246 (**−0.00465**) | NN 0.103125 (+0.00047) |
| k=2019 | 0.103496 | HEALTHY | **FAIL: impossible_mass** | 130 | logit 0.106352 (**−0.00286**) | NN 0.100600 (+0.00290) |

Consistent story so far: **GBT clearly beats the linear baseline, the net still edges GBT**
by ~5e-4 (2015) / ~2.9e-3 (2019) — the `06 §7` "flexibility ≈ architecture, net slightly
ahead" lean. (k=2015 is the promoted reconfirm fit at msh=1000.)

## 3. Finding — k=2019 QA fail (impossible-cell mass) + best_iter variance
k=2019 fails **one** QA criterion: predicted mass on structurally-impossible origin→dest
cells = **1.86e-3 > 1e-3** (realized 7.5e-5); the other four pass (probs valid, NLL
cross-check exact, base-rate Δ=0.0030, zone HEALTHY). Its early stopping fired at
**best_iter=130** vs k=2015's **1208** — the calm 2018 val NLL plateaus fast, and since val
`multi_logloss` does not penalize impossible-cell mass, ~130 rounds leaves those cells
under-suppressed. **A calibration/memo finding (`06 §4`/`§7`), not a bug — record, do not
tune:** trees leak a little onto structurally-impossible transitions where the net's
construction does not. Watch whether the deferred windows repeat it.

## 4. Why stopped — 16 GB RAM ceiling (NOT a result)
**k=2020 was abandoned mid-training and is NOT a result** — sunk swap-compute, no artifact.
On the 16 GB Mac, k=2020's 48 M-row train slice + growing booster + val/test arrays
overshot physical RAM by ~12 GB → **~12 GB swap, heavy thrash**. A stack sample confirmed it
was still *building trees* (not finalizing) after ~5h47m; continuing locally was open-ended,
whereas it re-runs RAM-resident on a ≥64 GB box in ~1–2 h. Idempotency means killing it lost
**only the sunk swap-compute**, not the window. The two banked windows ran because they are
smaller / early-stopped (k=2019 45 M, best_iter 130 → ~47 min; k=2015 promoted).
**Consequence: the 9 remaining windows (2020 = 48 M, 2023 = 60 M, 2025 = 67 M, + fillers) go
to a high-RAM box.**

## 5. Resume — 9 deferred windows on a high-RAM box
Re-fires only the 9 missing windows (regime-first); idempotency skips the banked 2015/2019.
```bash
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk" && \
nohup .venv/bin/python -u -m floan.model.backtest --device cpu --gbt-only \
  --windows 2020 2023 2025 2016 2017 2018 2021 2022 2024 --gbt-threads N \
  >> "/Volumes/SSD Felipe/dissertation/logs/gbt_sweep.log" 2>&1 &
```
- **Thread/memory:** `--gbt-threads N` is the only knob — set it to the pod's core count.
  No memory flag; headroom = the box's RAM. A **≥64 GB** box keeps every window RAM-resident
  (no swap). If ever RAM-tight, lower `gbt.MAX_BIN` 255→63 (halves Dataset memory) — a code
  constant, not a flag; unnecessary on a big box.
- **Reproducible:** the frozen config lives in committed `config.GBT_SELECTED`, and
  `deterministic=True` + `force_row_wise` make the models **bit-identical regardless of
  `--gbt-threads`**, so a different core count on the pod yields the same results.
- **Volume:** mount the retained SSD at `/Volumes/SSD Felipe/dissertation` (or edit `ROOT` in
  `src/floan/pipeline/config.py`); `require_drive()` fails fast otherwise.
- **Optional supervisor:** `scripts/run_gbt_sweep.sh` (auto-resume on transient failure)
  hardcodes the original 10-window list; for the precise 9, use the bare command above.

## 6. Artifacts & provenance
Run folders on the SSD (`models/gbt/full/{k2015,k2019}` + `k2015_reconfirm`), mirrored to
`ssd_mirror/` via `scripts/backup_ssd.sh` (2026-06-17 11:51). `gbt-baseline` commits:
`4374c4e` (M16 trainer), `b35635f` (M17 wiring), `659497f` (reconfirm msh=1000 + atomic
run-folder writes), `99010e4` (sweep supervisor).
