# GBT Baseline — Session Handoff / Chat Context

> **Purpose:** drop-in context for a fresh Codex chat continuing the LightGBM
> gradient-boosted-tree (GBT) baseline. Self-contained snapshot of the prior session
> (M16 done, M17 partial). Pair with the binding spec `specs/model/06_GBT_BASELINE.md`
> and the task list `specs/model/04_TASKS.md` (M16–M19). Live resume details:
> `src/floan/model/M17_NOTES.md`.
>
> **As of:** 2026-06-18 · canonical branch `main` (the only long-lived branch; create your
> GBT working branch off it).

---

## 0. One-paragraph state

We added a **single 7-class multiclass-softmax LightGBM** as a baseline alongside the
neural transition model. **M16 (dev-scale trainer) is complete and committed.** **M17
(full-scale rolling sweep) is partially done:** the frozen config was reconfirmed at full
scale, GBT was wired into `backtest.py`'s per-window loop, and **2 of 11 windows are banked
(k=2015, k=2019)**. The remaining **9 windows are deferred to a high-RAM box** — the 16 GB
Mac swap-thrashes on the ≥48 M-row windows. Everything is committed on `main` (the GBT work
was merged via PR #2 and consolidated by the reorg) and mirrored to `ssd_mirror/`. **M18 and
M19 are not started.**

---

## 1. Project frame (inherited from AGENTS.md)

Oxford MCF dissertation modelling US mortgage default risk on the Fannie Mae Single-Family
Loan Performance dataset (~800 GB, ~100 vintage CSVs, 2000–2025). Sirignano-style **seven-state
monthly transition model**: `current → 30/60/90+ DPD → foreclosure → REO → prepaid`.

**Standing invariants (apply everywhere):**
1. Never load a full quarter into memory — stream via DuckDB/Polars/PyArrow.
2. `raw/` on the SSD is immutable — no code writes there.
3. All paths derive from the single `ROOT` in `src/floan/pipeline/config.py`; every stage
   calls `require_drive()` and fails fast if the SSD isn't mounted.
4. Stages idempotent + per-window; verify each task's **Accept** criteria before advancing.
5. Commit/push **only when the user asks**. One task per branch-commit.
6. The repo is an installable package (`pip install -e .`); code imports as `floan.*` and
   runs as a module — e.g. `python -m floan.model.backtest`. **Never run files by path**
   (it breaks imports).
7. Data lives on external SSD `SSD Felipe` at `/Volumes/SSD Felipe/dissertation/`
   (`raw/ → interim/ → processed/`, plus `models/ outputs/ logs/`), **not** in the repo.
8. Working style: state assumptions before coding, simplest solution that works, surgical
   diffs, verify against explicit success criteria. **Never tune GBT to chase the NN number.**

---

## 2. The M16–M19 task arc (binding: `specs/model/06_GBT_BASELINE.md`)

- **M16 — GBT trainer (dev scale, k=2015). ✅ DONE + committed.**
  Single 7-class softmax LightGBM; origin `state` as native categorical (NOT four per-origin
  models); same feature set as the nets minus standardisation; HT weight on train only; early
  stopping on the 2014 val slice; small tuning grid; winner frozen in `config.py`; CPU only.
- **M17 — full-scale rolling sweep. ◑ PARTIAL (2/11 banked, 9 deferred).**
  Reconfirm frozen `GBT_SELECTED` at full export scale on the tuning window; wire GBT into
  `backtest.py` per-window loop as the 5th model. "GBT ≤ logit on every window" is a
  **reported result, not an Accept gate**.
- **M18 — GBT into loan-level exhibits + calibration decision. ☐ NOT STARTED.**
- **M19 — GBT into pool roll-forward via a model-agnostic `pool.py` seam. ☐ NOT STARTED.**
  (`pool.py` is currently torch-specific — M19 needs a model-agnostic roll-forward seam.)

---

## 3. Key technical decisions (the "why")

### Model
- `objective="multiclass"`, `num_class=7`, `metric="multi_logloss"`.
- **Native `categorical_feature`** for the vocab-coded columns + origin `state`.
- `deterministic=True` + `force_row_wise=True` ⇒ **bit-identical models regardless of thread
  count**. So the pod's `--gbt-threads N` doesn't change results.
- `free_raw_data=True` to release the raw arrays after `Dataset` construction (memory).
- Early stopping on the per-window val year.

### `min_sum_hessian_in_leaf` is scale-sensitive — the central gotcha
It's an **absolute summed-leaf-Hessian threshold**, not a per-row average. The ~34 M-row full
slice carries ~10× the summed Hessian of the 3.15 M-row dev slice, so a dev-calibrated value
is ~10× too weak at full scale. The **M17 full-scale reconfirm moved it 100 → 1000**
(val argmin: msh=100 → 0.097324 vs msh=1000 → 0.096968).

### HT (Horvitz-Thompson) weights
Importance weights = `1/p_keep`, applied **on the train split only**. The eval pool is
unweighted (`w ≡ 1`). A mis-set default `min_sum_hessian_in_leaf=1e-3` was miscalibrated by
the HT weights and made every tuning cell overfit (best_iter 2–24) — fixed by calibrating msh
**first**, then sweeping the `num_leaves × lr` plane in the good regime.

### Feature contract (byte-identical to the nets, minus standardisation)
`[ raw continuous (NaN-native, Float32) ‖ UNK-routed vocab codes ‖ binary ]`. Built by
`gbt.build_X(df, vocab)`. Vocab UNK code = 0.

### Backtest design
Rolling **expanding-window** backtest, 11 windows k=2015…2025. For window k:
train ≤ Dec(k−2), val = year k−1 (early stopping), test = year k. **k=2015 is the tuning
window.** GBT is fit per-window with that window's vocab + HT-weighted train + early stopping
on that window's val.

### Sanity-band zones (correctness check, NOT a performance target)
Ordering, worst → best NLL: **empirical floor (covariate-free, bug tripwire)** > bucketed
matrix > logit > NN. GBT must **beat the floor** (correctness) and ideally approach the NN.
- Hard-stop only if GBT is **worse than the empirical floor** (k=2015 floor = **0.112869**).
- Yellow flag between bucketed (~0.109385) and floor.
- Healthy ≈ 0.103–0.109.
- **Never tune GBT to chase the NN number** — record findings, don't tune.

### Atomic run-folder writes
`fit_window_frozen` writes via `.tmp` + `os.replace`, with **`metrics.json` written LAST** as
the completion marker — so an SSD drop mid-write can't leave a half-written `metrics.json`
that idempotency mistakes for "complete".

---

## 4. Banked results (k=2015, k=2019)

Test NLL on the frozen test slice; GBT vs per-window logit and best NN.

| window | GBT test NLL | zone | QA | best_iter | vs logit | vs NN |
|---|---|---|---|---|---|---|
| k=2015 | 0.103597 | HEALTHY | all pass | 1208 | logit 0.108246 (**−0.00465**) | NN 0.103125 (+0.00047) |
| k=2019 | 0.103496 | HEALTHY | **FAIL: impossible_mass** | 130 | logit 0.106352 (**−0.00286**) | NN 0.100600 (+0.00290) |

**Story so far:** GBT clearly beats the linear baseline; the net still edges GBT by ~5e-4
(2015) / ~2.9e-3 (2019) — the `06 §7` "flexibility ≈ architecture, net slightly ahead" lean.
k=2015 is the promoted reconfirm fit at msh=1000.

**k=2019 QA finding (record, do not tune):** fails **one** of five QA criteria — predicted
mass on structurally-impossible origin→dest cells = **1.86e-3 > 1e-3** (realized 7.5e-5). The
other four pass. Early stopping fired at **best_iter=130** vs k=2015's 1208: the calm 2018 val
NLL plateaus fast, and since val `multi_logloss` doesn't penalize impossible-cell mass, ~130
rounds leaves those cells under-suppressed. A calibration/memo finding (`06 §4`/`§7`) — trees
leak a little onto structurally-impossible transitions where the net's construction does not.
Watch whether the deferred windows repeat it.

---

## 5. Why the sweep stopped — 16 GB RAM ceiling (NOT a result)

**k=2020 was abandoned mid-training and is NOT a result** — sunk swap-compute, no artifact.
On the 16 GB Mac, k=2020's 48 M-row train slice + growing booster + val/test arrays overshot
physical RAM by ~12 GB → ~12 GB swap, heavy thrash. A stack sample confirmed it was still
*building trees* (ComputeBestSplit/ConstructHistogram), not finalizing, after ~5h47m.
Idempotency means killing it lost only the sunk swap-compute, not the window. The two banked
windows ran because they're smaller / early-stopped.

**Consequence:** the 9 remaining windows (2020 = 48 M, 2023 = 60 M, 2025 = 67 M, + 6 fillers)
go to a **≥64 GB box**.

---

## 6. Resume — 9 deferred windows on a high-RAM box

Re-fires only the 9 missing windows (regime-first); idempotency skips banked 2015/2019.

```bash
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk" && \
nohup .venv/bin/python -u -m floan.model.backtest --device cpu --gbt-only \
  --windows 2020 2023 2025 2016 2017 2018 2021 2022 2024 --gbt-threads N \
  >> "/Volumes/SSD Felipe/dissertation/logs/gbt_sweep.log" 2>&1 &
```

- **Threads/memory:** `--gbt-threads N` = pod core count (the only knob). No memory flag;
  headroom = the box's RAM. ≥64 GB keeps every window RAM-resident (no swap). If RAM-tight,
  lower `gbt.MAX_BIN` 255→63 (halves Dataset memory) — a code constant, not a flag; unnecessary
  on a big box.
- **Reproducible:** frozen config in committed `config.GBT_SELECTED`; `deterministic=True` +
  `force_row_wise` ⇒ bit-identical regardless of `--gbt-threads`.
- **Volume:** mount the SSD at `/Volumes/SSD Felipe/dissertation` (or edit `ROOT` in
  `src/floan/pipeline/config.py`); `require_drive()` fails fast otherwise.
- **Supervisor:** `scripts/run_gbt_sweep.sh` auto-resumes on transient failure but hardcodes
  the original 10-window list — for the precise 9, use the bare command above. macOS no-sleep
  wrapper: prefix with `caffeinate -dims`.

After the sweep completes all 11 windows: finish M17 (full 11-window GBT-vs-logit-vs-NN table,
`python -m floan.model.backtest --verify-only --device cpu` for the `[6] M17 GBT` block), then
proceed to M18/M19.

---

## 7. File map (what each thing is)

### Modified / created this arc
- **`src/floan/model/gbt.py`** (core deliverable) — single 7-class softmax LightGBM. Key
  functions: `feature_layout(vocab)→(names, cat_idx)`; `build_X(df, vocab)`;
  `load_split`/`load_window`; `_base_params(n_threads)` (objective/num_class=7/metric/
  num_leaves=63/lr=0.1/msh=1e-3/ff=1.0/l2=0/max_bin=255/seed=0/deterministic=True/
  force_row_wise=True); `tune()` (calibrate-then-plane: msh sweep → num_leaves×lr plane →
  ff/l2 sweeps); `reconfirm()` (3-cell focused re-check); `fit_window_frozen(variant,k,cfg,
  n_threads)` (per-window unit, free_raw_data=True, idempotent on (window,config), atomic
  writes); `classify_zone` (empirical_floor primary, logit fallback); `predict_proba`;
  `compute_accept` (5 criteria + zone_basis + leakage probe); `write_run_folder` (atomic
  `.tmp`+os.replace, metrics.json LAST).
- **`src/floan/model/config.py`** — `GBT_SELECTED = {num_leaves: 31, learning_rate: 0.05,
  min_sum_hessian_in_leaf: 1000.0, feature_fraction: 1.0, lambda_l2: 0.0}` (full-scale val
  argmin; was 100.0 from M16 dev). `GBT_TUNING_BEST_ITERATION = 1208`. Rationale comment
  documents the msh 100→1000 reconfirm move.
- **`src/floan/model/backtest.py`** — `import os`; `_gbt_frozen()`; `fit_gbt(k,cfg,n_threads,
  *,fresh)` (lazy `import gbt as G` to break a circular import); `_read_gbt(k)`; gbt in
  build_summary; `[6] M17 GBT` verify block; main() loop flags `--gbt-only` / `--skip-gbt` /
  `--gbt-threads`. Note: `--verify-only` defaults `--device cuda` and crashes without CUDA —
  run with `--device cpu`.
- **`scripts/run_gbt_sweep.sh`** — bounded (30-attempt) auto-resuming supervisor;
  WINDOWS="2019 2020 2023 2025 2016 2017 2018 2021 2022 2024".
- **`src/floan/model/M17_NOTES.md`** — the live resume record (config, banked table, k2019 QA
  finding, swap-ceiling diagnosis, resume command, provenance).
- **`specs/model/04_TASKS.md`** — appended M16–M19; spec-map line extended with
  `06_GBT_BASELINE.md (16–19)`.
- **`pyproject.toml`** — added `lightgbm==4.6.0` (line 27).

### Reference (read, not modified)
- `src/floan/model/net.py` (MortgageMLP, make_nn_predict), `evaluate.py`
  (`_nll = mean −log p[y]`, `score_window`), `torch_common.py` (weighted_ce),
  `features.py` (CONTINUOUS/CATEGORICAL/BINARY, Scaler, Vocab UNK=0, encode_frame),
  `data.py` (window_spec, `_masked_scan`, fit_window), `benchmarks.py` (M5 empirical/bucketed
  via DuckDB on the unthinned panel; NLL on the eval_pool test slice), `pool.py` (torch-specific
  roll-forward — M19 needs a model-agnostic seam).

---

## 8. Environment gotchas (resolved)

- **LightGBM `libomp.dylib` not loaded** (OSError on import) → `brew install libomp`
  (Apple Silicon Homebrew).
- **Circular import** backtest→gbt→evaluate→backtest → lazy `import gbt` inside `fit_gbt`.
- **`backtest.py --verify-only` crash** "Torch not compiled with CUDA enabled" → pre-existing;
  defaults `--device cuda`; run `--device cpu`.
- **Grid-design flaw** (default msh=1e-3 miscalibrated by HT weights, all cells overfit) →
  calibrate msh first, then the num_leaves×lr plane.
- **NLL wiring cross-check:** LightGBM `multi_logloss` == `evaluate._nll` to ~1e-16.

---

## 9. Provenance

GBT commits (originally on `gbt-baseline`, now ancestors of `main`): `4374c4e` (M16 trainer), `b35635f` (M17 wiring),
`659497f` (reconfirm msh=1000 + atomic run-folder writes), `99010e4` (sweep supervisor),
`47fb9f8` (bank k=2015 + k=2019; abandon k=2020; defer 9). Run folders on the SSD
(`models/gbt/full/{k2015,k2019}` + `k2015_reconfirm`), mirrored to `ssd_mirror/` via
`scripts/backup_ssd.sh` (2026-06-17 11:51).

> Path note: this arc predates a repo reorg that moved model code into `src/floan/model/`,
> specs into `specs/model/`, and shell helpers into `scripts/`. Older commit messages may
> still reference the former `dev/` tree.

---

## 10. Immediate next actions (for the new chat)

1. On a ≥64 GB box, run the §6 resume command to fit the 9 deferred windows.
2. Finish **M17**: build the full 11-window GBT-vs-logit-vs-NN table; run
   `python -m floan.model.backtest --verify-only --device cpu` for the `[6] M17 GBT` block;
   note whether the k=2019 impossible-mass pattern recurs.
3. Then **M18** (loan-level exhibits + calibration decision) and **M19** (model-agnostic
   `pool.py` seam for GBT pool roll-forward) per `specs/model/06_GBT_BASELINE.md`.
