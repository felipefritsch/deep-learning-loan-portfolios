# M20 — Impossible-cell mask correction: notes + verification procedure

**Status:** implemented + locally green; **windowed QA evidence pending** (the gate is NOT closed
until §4's M20a + M20b are green). Branch `m20-impossible-cell-mask`.
Spec: `specs/model/ECONOMIC_ENGINE.md §3.5`; ADR: `ADR-001-economic-engine-seam.md`; task: `04_TASKS.md` M20.

## 1. What was implemented (3 files; every line traces to M20)

- **Fix (a) — QA mask**, `backtest._structural_allow` (`backtest.py:348-354`): now permits the four
  reporting-gap-legal skip-bucket cells (`current→dpd_60`, `current→dpd_90plus`, `dpd_30→dpd_90plus`,
  `dpd_90plus→REO`); `current/dpd_30/dpd_60→foreclosure/REO` stay forbidden. Auto-propagates to
  `base_rate_qa` (ensemble) and to GBT's QA **at fit time**.
- **Fix (b) — roll-forward predictor**, `pool._zero_impossible`: zeros `current→foreclosure`/`current→REO`
  and **renormalizes** the `current` row (matrix stays row-stochastic); applied per-step in `roll_forward`
  before composition/SMM/cashflow, **not** in `_score_direct`/`evaluate` (the loan-level path).
- **Test**, `tests/test_pool.py::test_zero_impossible_is_likelihood_neutral`: pooled `|ΔNLL| ≤ 1e-9`,
  phantom mass provably present-then-removed, row stays stochastic, no other origin row moves. **All 12
  `test_pool.py` tests pass locally (no SSD/GPU).**

## 2. Decisions honored

- **Renormalize** the current row after zeroing (row-stochastic invariant preserved).
- **`pool.verify_h1` left untouched** — its bit-exact check now shows a current-origin divergence of
  ~phantom-mass magnitude (the ≤1e-5 cross-check still passes). Re-establishing the parameterized H=1
  identity on the **raw, un-zeroed** path is **M21's** job (ADR-001, action item 2). Do **not** relax
  `verify_h1` in M20.

## 3. Verification-coverage gap discovered (read before running anything)

`--verify-only` as wired **cannot evidence M20's windowed criterion on the Mac**, for two independent reasons:

1. **`--device` defaults to `"cuda"`** (`backtest.py:593`); `main` forces `torch.device("cuda")` whenever the
   arg equals that default (`:608`), so bare `--verify-only` crashes on a CUDA-less Mac
   (`Torch not compiled with CUDA enabled`). Passing `--device cpu` avoids the crash but then `main` hands
   `verify(device=None)` (`:612`) → `build_summary(with_qa=False)` (`:481`, `:461`) → the live QA block is
   **skipped entirely**.
2. **The two QA paths are asymmetric:**
   - **Ensemble** impossible-mass (`base_rate_qa`, `:357`; check `[5]`) is **recomputed live** under the
     corrected `_structural_allow` — but only when handed a **CUDA** device. CPU/MPS ⇒ skipped.
   - **GBT** impossible-mass (check `[6]`) is **NOT recomputed** — `_read_gbt` (`:438-448`) reads
     `impossible_transitions.passed` out of each window's **stored fit-time `metrics.json`**, written under
     the **old** mask. So `[6]` keeps reporting the old breaches (k2019/2020/2021/2024/2025) until the GBT
     QA is regenerated under the corrected mask.

**Consequence:** `--device cpu --verify-only` on the Mac runs `[1]–[4]` PASS, `[5]` `[skip]`, `[6]` **stale
old-mask GBT flags** — it does not evidence M20. Two targeted steps close the gap.

## 4. Follow-up to close the gate

### M20a — GBT impossible-mass re-score under the corrected mask *(CPU; Mac; all 11 windows)* — **DECIDED: re-score, not re-fit**
GBT is deterministic and CPU, the boosters are persisted (`gbt.write_run_folder` → `model.txt` via
`booster.save_model(num_iteration=best_iteration)`), and M20 changed **only** the mask — so the exact
recomputation is `booster.predict` + new mask, **no re-fit**. Re-fit (b) is wrong, not just dearer: it produces
a different `proba_test` (model drift), conflating the mask fix with a new model and violating the
frozen-config invariant `verify [6]` asserts.
The machinery already exists: per window — load `model.txt` → `gbt.load_split(variant, k, "test", vocab)`
(rebuilds the test slice from the run folder's saved per-window vocab; no train/val) → `gbt.predict_proba`
→ `(proba * ~B._structural_allow()[origin]).sum(1).mean()` (this is exactly `gbt.compute_accept`'s impossible
block, recomputed live).
**Form (decided):** a small CPU re-score driver — `gbt.py` + a `--rescore-qa` entry — that loops the 11 windows
and writes a dedicated **`models/gbt/full/m20_impossible_recheck.json`** (per-window `model_mean_mass`,
`realized_mean_rate`, `passed @ <1e-4`). **Non-destructive:** the fit-time `metrics.json` is left untouched
(provenance preserved).
**Accept:** GBT `model_mean_mass < 1e-4` on **all 11** windows under the corrected mask (the five old breaches
clear); numbers recorded in §5; runs on the Mac (SSD mounted), no GPU, minutes.

### M20c — Wire `verify [6]` to recompute GBT QA live *(follow-up; its OWN branch — do NOT bundle into M20)*
The staleness is a real defect: `_read_gbt` (`backtest.py:438-448`) reads `impossible_transitions.passed` from
fit-time `metrics.json` (old mask), so **post-M20 `verify [6]` will report the 5 windows as FAIL forever** even
though M20 fixed them — the source-of-truth harness lying. The cure mirrors `[5]` (which already recomputes the
ensemble live, never reads stored): have `[6]` call `gbt.compute_accept` (reusing the booster + `load_split`)
instead of the stored block. **Gate it like `[5]`:** recompute live **when boosters + test slices are present**
(SSD mounted); otherwise fall back to the stored block **with an explicit "stale (pre-M20 mask)" warning** —
never silently. This keeps `[1]–[4]` cheap (no SSD for a quick structural/NLL check), makes `[5]`/`[6]`
symmetric, and kills the whole "verify reads stale" class — not just this instance. Reason it is **not** in
M20: it changes verify's contract (SSD-bound `[6]`), so it gets its own branch + Accept.
**Accept:** `verify [6]` reports the corrected GBT impossible-mass live when slices are present (matches
M20a's `m20_impossible_recheck.json`); falls back with a visible "stale" warning when they are not; `[1]–[4]`
unchanged in cost.

### M20b — Ensemble base-rate QA under the corrected mask *(RunPod GPU; 5 key windows)*
On a RunPod GPU pod with the SSD/volume mounted and frozen ensemble members present:
```
python -m floan.model.backtest --device cuda --verify-only
```
`[5]` recomputes `base_rate_qa` live under the corrected `_structural_allow` for the 5 ensemble-bearing key
windows (2015/2019/2020/2023/2025).
**Accept:** `[5]` structural-impossible mass `< 1e-4` and base-rate `max|Δ| < 0.01` on each key window;
threshold unchanged at 1e-3.
**Note:** guaranteed to pass by construction — the nets passed under the *stricter* old mask, and the
corrected mask is strictly more permissive (residual can only drop) — so ride the **next** GPU session rather
than renting a box solely for this.

## 5. Evidence (fill on completion — standing rule 1: paste the numbers)

**GBT, all 11 windows** (M20a). Old-mask breaches for reference: k2019 ≈1.86e-3 (hardest), k2020/2021/2024/2025
in 1.2–1.9e-3; other six passed (k2018 cleanest ≈1.1e-4). Expected post-correction: ~1e-5 everywhere
(brief Fig. 7).

**Result (2026-06-23, 9-core re-score, `m20_impossible_recheck.json`):** all 11 windows < 1e-4 — the five
old-mask breaches (k2019/2020/2021/2024/2025, all >1e-3) cleared by ~30–60×. Max corrected mass = 4.12e-05
(k2020); worst breach k2019 1.86e-3 → 3.04e-05 (61×). Each value is the realized rate's order of magnitude
(model tracks data, no hallucinated mass). Reproduced k2015 = 3.77e-06 bit-for-bit vs the collect-path check.

| window | model_mean_mass (corrected) | old-mask (fit-time) | < 1e-4 ? | n_test |
|---|---|---|---|---|
| 2015 | 3.77e-06 | 3.47e-04 | ✓ | 6,180,351 |
| 2016 | 2.01e-06 | 1.95e-04 | ✓ | 6,410,612 |
| 2017 | 1.56e-06 | 1.38e-04 | ✓ | 6,759,294 |
| 2018 | 1.17e-06 | 1.09e-04 | ✓ | 7,065,297 |
| 2019 | 3.04e-05 | 1.86e-03 | ✓ | 7,287,022 |
| 2020 | 4.12e-05 | 1.56e-03 | ✓ | 7,639,066 |
| 2021 | 1.59e-05 | 1.42e-03 | ✓ | 8,233,784 |
| 2022 | 1.62e-05 | 8.84e-04 | ✓ | 8,669,811 |
| 2023 | 1.05e-05 | 9.14e-04 | ✓ | 8,768,239 |
| 2024 | 1.12e-05 | 1.20e-03 | ✓ | 8,775,310 |
| 2025 | 8.65e-06 | 1.43e-03 | ✓ | 8,723,591 |

**Ensemble, 5 key windows** (M20b):

| window | impossible mass (corrected) | base-rate max\|Δ\| | pass |
|---|---|---|---|
| 2015 | | | |
| 2019 | | | |
| 2020 | | | |
| 2023 | | | |
| 2025 | | | |

## 6. Gate status

- [x] Fix (a) mask correction + Fix (b) zeroing/renorm + unit test — implemented, locally green.
- [x] **M20a** — GBT re-score (`--rescore-qa` → `m20_impossible_recheck.json`), all 11 < 1e-4 (Mac, 2026-06-23). ✅
- [ ] **M20b** — ensemble QA, 5 key < 1e-4 (next GPU session).
- [ ] **Merge gate:** once M20a **and** M20b are green and recorded → `scripts/backup_ssd.sh` + `git push`
  (standing rule 5), merge `m20-impossible-cell-mask`.

**Sequencing (refined):** fix (a)+(b)+unit test + **M20a** establish mask correctness on the model that
actually breached (GBT) — enough to **start M21's code in parallel** (CPU dev work). **M20b** is guaranteed by
the strictly-more-permissive mask (the nets passed under the *stricter* old mask), so it **rides the next GPU
session** rather than its own — it must be **recorded before the M20 branch merges**, but it does **not** block
M21 from starting. Don't silently drop it.

**Separate follow-up (does NOT gate M20):**
- [ ] **M20c** — wire `verify [6]` to recompute GBT QA live (its own branch; cures the staleness so the harness stops reporting a false GBT failure post-M20).

## 7. Branching — don't forget (M20 → M20c → M21 stack)

M20c and M21 both build on M20's `pool.py` changes (`_zero_impossible` + the `roll_forward` edit), so **until M20
merges they must branch off the M20 branch, NOT main**:

- `m20-impossible-cell-mask` — M20 base + M20a + (later) M20b evidence.
- `m20c-verify-gbt-live` — branch **off `m20-impossible-cell-mask`**.
- `m21-predictor-seam` — branch **off `m20-impossible-cell-mask`** (it carries fix (b) in `pool.py`).

**Merge order**, once M20b lands on the next GPU session: **M20 → main, then M20c, then M21.**

Avoid-stacking alternative: hold M21 until M20 merges to main after the GPU session — cleaner git history, but it
stalls the engine work on a guaranteed-pass check (M20b), which is the thing we chose not to do.
