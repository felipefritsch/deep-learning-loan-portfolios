# GBT Baseline — Implementation Plan (Claude Code handoff)

**Goal:** Add **one** gradient-boosted-tree learner (LightGBM, multiclass softmax) as a full member of the existing rolling backtest and pool-level valuation, reusing the existing data export, evaluation, roll-forward, and cashflow machinery. GBT is a *parallel flexible learner* alongside the net (not a new nested rung) — it isolates inductive bias (axis-aligned steps vs smooth compositions) holding capacity roughly constant.

**Non-goals (do not do):** rebuild any data; retune the nets; refactor the cashflow engine; add random forests / GAMs / kernel methods / a second GBT library. One learner, through the existing pipes.

## Locked decisions (confirm before coding)
1. **Single 7-class softmax model** with `origin state` as a categorical input feature — matches the nets/logit single-model design (§2.1). NOT four per-origin models.
2. **Feature set byte-identical to the nets' input** (same columns, same missingness-indicator columns). GBT skips the per-window standardiser (scale-invariant) but uses the same features.
3. **Training rows carry Horvitz–Thompson weights; validation and test do NOT** (they are unthinned, weight = 1). Early stopping and all evaluation run on weight-1 rows.
4. **CPU only** (local Apple-Silicon Mac via conda-forge `lightgbm>=4.6`, or a cheap RunPod CPU pod). GPU gives ~no benefit at ~30 features — do not spend GPU credit here.
5. **Tune once on the k=2015 window, freeze, roll** — mirrors the nets' protocol.
6. **Calibration is conditional**, decided by reliability diagrams in Phase 3, not assumed.

> Note on the "no full `.collect()`" hard rule: that rule protects the 3.3B-row raw panel. The **thinned per-window training export** (≤74M rows, ~2 GB binned) exists precisely to be materialised in RAM — loading it for LightGBM is in-scope and expected. Do not stream-fit from the raw panel.

---

## Phase 0 — Discovery (read, write nothing)
Locate and read; produce a short written **interface map**, no code yet:
- The per-window **training export** Parquet: exact column names; the weight column; the label column (destination state, 0–6); how origin state and high-cardinality categoricals (state / MSA / 3-digit zip) are encoded; how missingness indicators are named.
- The existing **model trainer**: how it iterates the 11 windows, reads the export per window, defines the validation year, does early stopping, and the **`metrics.json` run-record schema** (git commit, seed, window, params, OOS metrics, train/eval content hashes).
- The **evaluation module**: the function that takes predicted probs + labels on the frozen unthinned eval slice and returns NLL / one-vs-rest AUC / reliability bins — and the **exact array shape** it expects (almost certainly `(n_rows, 7)`).
- The **roll-forward engine**: how it obtains a 7-vector from the model at deterministically-evolved covariates (the call site that, for the nets, is a PyTorch forward pass).
- `CLAUDE.md`, the planning markdowns, and the hard rules.

**Verify:** a written map naming (a) the export columns + dtypes, (b) the run-record schema, (c) the eval function signature, (d) the roll-forward model-call site. Stop and surface anything that contradicts the locked decisions above.

---

## Phase 1 — GBT trainer, tuning window first
New module mirroring where the nets live (e.g. `src/models/gbt.py`). Reuse the existing window iterator and export reader — do not write a new data path.

```python
import lightgbm as lgb

# Training set: HT weights. Validation set: UNWEIGHTED (unthinned, weight=1).
train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                        categorical_feature=CATEGORICAL_COLS, free_raw_data=True)
valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)  # no weight

params = dict(
    objective="multiclass", num_class=7, metric="multi_logloss",
    num_leaves=NUM_LEAVES, learning_rate=LR,
    min_sum_hessian_in_leaf=MIN_HESS,   # scales with HT weights — tune on weighted data
    feature_fraction=FF, bagging_fraction=BF, bagging_freq=1,
    lambda_l1=L1, lambda_l2=L2,
    max_bin=255, two_round=True,         # 63 if memory-tight
    num_threads=N_CORES, seed=SEED, verbosity=-1,
)
booster = lgb.train(params, train_set, num_boost_round=2000,
                    valid_sets=[valid_set],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
```

- **Origin state** enters as a categorical feature (single-model design). LightGBM handles NaN natively; still pass the nets' missingness-indicator columns so the feature set is identical.
- **Tuning grid** on k=2015 only (small, deliberate): `num_leaves ∈ {31,63,127}`, `learning_rate ∈ {0.05,0.1}`, `min_sum_hessian_in_leaf` (sweep because HT weights inflate Hessians), `feature_fraction ∈ {0.7,1.0}`, `lambda_l2 ∈ {0,1}`. Select on the 2014 validation NLL, **freeze**.
- Write `metrics.json` in the existing schema. Pin the LightGBM version in the env block.
- Memory/speed: `two_round=True`, `save_binary` the constructed Dataset and reuse across windows, `free_raw_data=True`.

**Verify (gates — all must pass before Phase 2):**
- Predicted probs sum to 1 per row; no NaN.
- **Library-vs-project NLL cross-check:** LightGBM's `multi_logloss` on the eval slice equals the project's own NLL function on the same `(n,7)` predictions to ~1e-6 (the GBT analogue of your logit-vs-library check).
- **Base-rate QA:** GBT predicted base rates ≈ unthinned panel frequencies (the check already used for the nets).
- **Impossible-cell mass** (e.g. Current→60 DPD) is negligible (your §2.1 internal check).
- **Sanity band:** tuning-window OOS NLL sits below the empirical floor (~0.1127) and, if the tabular literature holds, at or below the logit (~0.1088). Below the floor but above the net (~0.1032) is the expected/interesting zone. Above the floor ⇒ a bug; stop.

---

## Phase 2 — Wire GBT into the shared evaluation (surgical)
Add a thin predictor with the signature the eval module already expects; **change no evaluation code.**

```python
def predict_proba(booster, X):           # -> (n, 7), already softmax
    return booster.predict(X, num_iteration=booster.best_iteration)
```

Refit the frozen config on each of the 11 windows (per-window categorical vocab + early stopping on that window's validation year), score the frozen unthinned eval slice, and emit GBT into the existing exhibits: the Table 4.2 NLL-by-test-year column, the Table 4.3 AUC rows, and reliability diagrams.

**Verify:** GBT appears as a new line/column in the existing tables and figures with **zero diff** to the evaluation module (only a predictor was added). Each eval row predicted exactly once by a window that never trained on it.

---

## Phase 3 — Calibration (conditional)
1. Reliability diagrams on **raw** GBT softmax (already produced in Phase 2).
2. **If** off-diagonal: add per-window post-hoc calibration **fit on the validation year only** (never training, never test). Start with **temperature scaling** (one parameter, preserves ranking/AUC, robust); isotonic per-class (one-vs-rest, then renormalise) as fallback.

```python
import numpy as np
from scipy.optimize import minimize_scalar

def fit_temperature(P_val, y_val):       # val is unweighted
    logits = np.log(np.clip(P_val, 1e-12, 1.0))
    def nll(T):
        z = logits / T; z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return -np.log(p[np.arange(len(y_val)), y_val] + 1e-12).mean()
    return minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded").x
```

**Verify:** calibrated reliability diagram closer to the diagonal; **test NLL not worsened** (calibrate on val, judge on test). Keep calibration only if it helps. Refit T per window to stay consistent with the rolling design.

---

## Phase 4 — Roll-forward + pool valuation
The roll-forward already composes monthly 7×7 matrices: four live-origin rows from the model, terminal rows = identity. The only integration point is the model call.

- If the roll-forward hardcodes a PyTorch call, do the **one justified minimal refactor**: introduce a model-agnostic predictor (a callable / small Protocol returning `(n,7)`), then pass either the net or the GBT predictor. Do not rewrite the engine.
- Apply the Phase-3 calibration transform **inside** the GBT roll-forward predictor — pool valuation consumes *levels*, so this is where calibration matters most.
- Run the 5 regime anchors (Dec2014/2018/2019/2022/2024) through the existing cashflow engine → CPR / WAL / price errors; emit GBT columns in Tables 5.1 / 5.2 and the Fig 5.2 bucket map.

**Verify:**
- **Composition consistency:** roll-forward at horizon h=1 reproduces a direct one-month GBT prediction.
- **Engine coherence preserved:** price-error sign agrees with WAL-error sign in ~99.4–100% of pools for GBT too.
- GBT columns appear in the pool tables with **zero diff** to the cashflow engine.

---

## Phase 5 — Reproducibility & bookkeeping
- A `metrics.json` run record per GBT fit (commit, seed, window, params, train/eval content hashes), same as every other model.
- Pin `lightgbm` (conda-forge ≥4.6) in each run record's env block.
- Add a short GBT section to the planning markdowns + a one-line methodology note placing GBT as the non-neural flexible learner (Hastie–Tibshirani–Friedman lineage).

---

## Result framing (state plainly in the write-up; pre-registered both ways)
- **GBT ≈ net:** "flexibility is what matters, architecture is secondary" — reinforces the 3-layer finding.
- **Net > GBT:** "the smooth structure of mortgage behaviour (incentive S-curve, seasoning hump, convex FICO decay) rewards smooth learners" — the sharper claim.
- **GBT > net:** legitimate, but note McElfresh metafeatures (huge N, high size/feature ratio, class imbalance) all favour GBT, so attributing a GBT win to anything other than smoothness needs the ablation; report it as such.

## Budget / runtime expectation
CPU, ~1.5–2.5 h per window for a full multiclass fit (extrapolated from LightGBM's Higgs benchmark × row scaling × 7-class tree multiplier); all 11 windows + tuning ≈ a long overnight job, ~$0 local or a few dollars of RunPod CPU. If wall-clock blows up: `max_bin=63`, stronger early stopping, smaller `num_leaves`.
