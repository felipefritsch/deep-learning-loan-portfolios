# M23 — Per-horizon calibration decision + calibrator fit: notes + evidence

**Status:** implemented + **locally green** (pytest) and **decided** (CPU, ~20% subsample @ k=2020).
Branch `m23-calibration` (off `m22-loan-pricing`).
Spec: `ECONOMIC_ENGINE.md §6` (+ §3.2, §4, §5); `06_GBT_BASELINE.md §4`; task `04_TASKS.md` M23.
Scope: **calibration decision + calibrator fit only** — *not* M24's full-scale H×regime grid.

## 1. Decision (the headline)

**SKIPPED — calibration is a no-op identity for every model the engine currently prices.** The
raw reliability curves sit on the diagonal at H=1 (the clean calibration test, §5): the fitted
per-window temperature is ≈ 1 for all three models, expected calibration error is < 0.01 on every
diagnostic transition, and the held-out test NLL deltas are ≤ 0.5% (subsample noise). Recorded as
`config.CALIBRATION["applied"] = False`; the engine runs with `calibrator=None`, the
regression-checked identity path.

This is the `06 §4` "if they sit on the diagonal, do nothing and document it" branch. It is the
*expected* outcome here: the §4 calibration concern is specifically **GBT** over-confidence on the
rare credit-event classes; the nets/logit are trained with weighted cross-entropy (a proper scoring
rule) and come out calibrated. The fitted-but-unused calibrators and this decision machinery are
the deliverable; **when GBT enters the engine (M19, not yet built — `pool.py` has no `GBTPredictor`
today), its calibration decision re-runs through this identical machinery**, and GBT is the model
most likely to flip the decision to APPLIED.

## 2. Evidence — `python -m floan.model.calibration --k 2020 --device cpu --frac 0.2`

Fit per-window temperature on the **val** slice (year 2019), assess raw-vs-calibrated on a
**held-out ~20% test** subsample (year 2020 — the COVID anchor) at **H = 1**. `n_val ≈ 1.46 M`,
`n_test ≈ 1.53 M` loan-months; 76 s on the Mac CPU.

| model | T\* | val-NLL raw→cal | test-NLL raw→cal | ECE `cur→prepaid` r→c | ECE `cur→dpd30` r→c | ECE `90+→FC` r→c | decision |
|---|---|---|---|---|---|---|---|
| logit | 1.021 | 0.10552→0.10544 | 0.19192→0.19088 | 0.0077→0.0072 | 0.0040→0.0035 | 0.0083→0.0090 | **SKIPPED** (T≈1) |
| nn | 0.999 | 0.10038→0.10038 | 0.18468→0.18472 | 0.0038→0.0038 | 0.0048→0.0048 | 0.0048→0.0047 | **SKIPPED** (T≈1) |
| ensemble | 1.002 | 0.10027→0.10027 | 0.18325→0.18317 | 0.0026→0.0026 | 0.0046→0.0046 | 0.0062→0.0063 | **SKIPPED** (T≈1) |

Reading: |T−1| ≤ 2.1% for all three (within `T_TOL = 0.05`); every ECE < 0.0083 raw and barely
moves under calibration; the largest test-NLL move (logit, −0.54%) is held-out subsample noise on a
T that is statistically 1. The **on-diagonal conclusion does not hinge on the thresholds** — the raw
ECEs are already small and the temperatures are already ≈ 1. `dpd_90plus→foreclosure` is the
sparsest cell (base rate ~0.0007, n≈24k in the subsample) and the only place calibration
occasionally nudges ECE the wrong way — another reason not to apply a transform the data don't ask
for. Empirical-matrix (covariate-free) is excluded from the priced set; its temperature *worsened*
held-out NLL (val-fit overfit), which the held-out protocol correctly flagged — a sanity check that
the fit/assess split works.

## 3. What was implemented (`calibration.py`, `config.py`, `tests/test_calibration.py`)

- **`calibration.py` (new).** The model-agnostic calibrators + the decision driver. Calibrators
  conform to the M21 `pool.Calibrator` seam (`__call__(probs, origin) -> (n,7)`), so they drop into
  `roll_forward_predictor(..., calibrator=...)` and are applied to the **raw per-origin scores
  before assembly** (§4 item 2) at every step — per-horizon by construction, no `pool.py` change.
  - `TemperatureCalibrator(T)` — one scalar per window (the primary method, `06 §4`).
    `softmax(log p / T)`, recoverable from probabilities alone (`= softmax(z/T)`). **`T == 1`
    short-circuits to return the input array unchanged** (the user's required genuine bit-identity;
    `softmax(log p / 1)` is not bit-exact).
  - `IsotonicCalibrator(maps)` + `fit_isotonic` — per-origin-block, per-class isotonic + row
    renormalise, the §4 fallback (built and unit-tested; unused given the SKIP decision).
  - `fit_temperature` — 1-D bounded Brent minimisation of val multiclass NLL.
  - `ece`, `_binary_nll`, `decide`, `_decide_one` — the reliability read + the calibrate-vs-not
    rule (apply only if T materially ≠ 1 AND a material held-out NLL/ECE gain).
  - `_score_slice` — split-aware, **deterministic ~frac subsample** by `(loan, period_ym)` hash
    pushed into the lazy scan (bounded-CPU). **Reuses `evaluate`'s loaders/scorers — `evaluate.py`
    is NOT modified** (the metric path stays pristine; this is a deviation from the approved plan's
    "add a `split` param to `score_window`", chosen because the bounded-CPU subsample needs to push
    into the scan and because leaving `evaluate.py` untouched is more faithful to the spec's
    "evaluate.py: none for the metric path").
- **`config.CALIBRATION` (new).** Records the decision per `ECONOMIC_ENGINE §8`
  (`applied=False`, method/fallback, basis, `refit_at_scale="M24"`).
- **`tests/test_calibration.py` (new, 6 tests, synthetic — no SSD/GPU).**

## 4. Verification (Accept — paste the numbers)

- **Fit + decision on a representative ~20% subsample** — §2 above. Temperature is one scalar per
  window, subsample-robust; the 5% smoke (n≈0.37 M) and 20% run (n≈1.5 M) give the same T≈1,
  same SKIP.
- **Calibrator is a no-op identity when disabled (regression check)** — three guards, all green:
  - `test_temperature_T1_is_bit_identity` (unit: `TemperatureCalibrator(1.0)` returns input array);
  - `test_fitted_T1_calibrator_is_bit_identical_through_engine` (engine: fitted T=1 == no calibrator
    bit-for-bit at H=1 and H=6);
  - the existing M21 `test_identity_calibrator_is_noop`. With the decision SKIPPED, the engine's
    production path **is** the disabled (identity) path, so this is the live guarantee.
- **Decision documented** — §1 (skipped-with-evidence) + `config.CALIBRATION`.
- **No final calibrated-vs-raw price-error table** — deferred to M24 (the per-H × ≥3-anchor grid);
  we subsampled the FIT, not the reported numbers.
- **pytest green before AND after** — `tests/{test_calibration,test_pool,test_economics,test_config}`
  41 passed (35 pre-existing + 6 new); `test_pool.py` SSD-free. Baseline captured before any edit.

## 5. Gate status

- [x] Reliability on raw outputs (per-transition ECE at H=1, the clean calibration test §5).
- [x] Per-window temperature fit on the val slice (+ isotonic fallback implemented).
- [x] Applied via the §3.2 calibrator seam BEFORE assembly (§4 item 2) — no `pool.py` change.
- [x] Decision on a ~20% subsample; identity no-op regression-checked; documented (skipped).
- [ ] Per-H calibrated-vs-raw price-error table (≥3 anchors × H∈{1,3,6,12}) — **M24** (by design).

**Next:** M24's full-scale H×regime grid. If GBT enters the engine (M19) first, re-run
`calibration.decide` for `gbt` — the machinery is ready and GBT is the likeliest APPLIED case.
