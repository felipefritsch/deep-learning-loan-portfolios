# 02 — Phase 2: Loan-Level Transition Models

> **Goal:** estimate `P(state_{t+1} = j | state_t, x_t)` and produce the headline comparison table — empirical matrix vs multinomial logit vs neural nets (depth grid + regularization + ensemble) — on a frozen out-of-sample test slice. Structure mirrors Sirignano et al. §2–4 (their Table 11 = depth/NLL grid, Figure 7 = ensemble curve, §3.4 = AUC).

---

## 1. Problem definition (binding for all models)

- **Unit:** loan-month with non-null `state` and `state_next`, origin state ∈ `{current, dpd_30, dpd_60, dpd_90plus}` (terminal states never re-appear — `00_OVERVIEW §4`). Censored rows excluded.
- **Target:** `state_next` ∈ 7 classes. One single multinomial model conditioned on origin state as a feature (paper's design) — *not* four separate models. Structurally impossible transitions (e.g. `current→dpd_60`) are left to the model/benchmark to learn as ≈0; the QA step asserts predicted mass on impossible cells is negligible.
- **Splits** (from `00_OVERVIEW §3.2`): rolling expanding windows, one per test year k = 2015…2025. Window k: train labels ≤ Dec(k−2), val labels = year k−1 (early stopping only), test labels = year k. Strict feature-month masks (`period_ym < cutoff`). **Window k = 2015 is the tuning window** — all architecture/hyperparameter selection happens there; configs are then frozen for the loop. Per window, the test rows are frozen and identical for every model.
- **Headline metric:** out-of-sample **mean negative log-likelihood** (NLL), reported (a) per test year by model and (b) **pooled over all 11 test years** (each test row predicted once by a model that never saw it), overall and broken out by origin state. Secondary: per-transition one-vs-rest **AUC** conditional on origin state (paper §3.4), and **reliability/calibration plots** per major transition (pooled and per regime).

## 2. Features (from `feature_spec`, fitted on train only)

- **Continuous (standardized; log where flagged):** current/original rate, **incentive** (`current_rate − pmms30(t)`; proxy version `01_EDA §4` as robustness), orig & current UPB (log), loan age, remaining months, original term, LTV, CLTV, DTI, `fico_orig`, co-borrower FICO, MI %, unscheduled principal.
- **Macro (joined per `05_MACRO_DATA.md`, lags applied at join):** `pmms30`, `unrate_state` (national fallback where the state series is missing, + fallback indicator), `hpi_state` level enters only via derived features: **`ltv_mtm` = (current UPB / orig UPB) × orig LTV × hpi_state(orig_ym) / hpi_state(period_ym)`** (mark-to-market LTV — the paper's key default driver) and `hpi_chg_12m` (state 12-month HPI change); `dgs10` and `slope_10y2y` optional. Raw HPI *levels* are not features (non-stationary index units).
- **Categorical (embeddings):** origin `state`, channel, purpose, property type, occupancy, property state, MSA (high-card), zip3 (high-card), number of units/borrowers, FTHB, modification flag, amortization/IO flags, seller (optional — high-card, test marginal value).
- **Binary:** missingness indicators.
- **Time:** loan age and `orig_ym`-derived vintage-year category are features; **calendar month-of-year** as a cyclical/seasonal categorical (prepay seasonality). **Do not feed raw `period_ym`/calendar-year as a continuous feature** — the test years are unseen values and the model would extrapolate arbitrarily; time-varying conditions enter via the macro block + incentive + seasonality, which generalize across calendar time.
- Embedding vocabularies built on train; unseen test levels map to a reserved `UNK` index. Scalers per `01_SCHEMA §6.4` (robust where skewed), persisted to the run folder.
- For the **logit benchmark**, categoricals are one-hot (drop-first); identical rows, identical split, identical target.

## 3. Training-set construction & export (`export.py`)

Because every loan-month changes role across the 11 windows (test in window k, train in window k+2), we do **not** export per-window datasets. One shared export, masks applied at load time:

1. **Train pool** (thinned, weighted): assemble via DuckDB from the panel + macro joins (`macro_national` on lagged `period_ym`; `macro_state` on `(property_state, lagged period_ym)` with national fallback — `05_MACRO_DATA.md §4`) + `mkt_rate`, origin-state filter, all years. Keep ALL rows with `state_next ≠ current` or `state_t ≠ current`; keep `current→current` rows with probability `p_keep` (config; start 0.05–0.10). Attach `weight = 1/p_keep` (1.0 otherwise). The loss is weight-averaged so estimated probabilities remain unbiased. Window k trains on `train_pool WHERE label_month ≤ Dec(k−2)`.
2. **Eval pool** (NEVER thinned): all loan-months with label months ≥ 2014-01 for loans in a **fixed shard block** (e.g. `shard < 0.2·N_SHARDS` ≈ 20% of loans — whole loans, loan-disjoint, fixed forever). Val/test slices for every window are cut from this pool by label-month mask. Evaluating on a fixed loan subsample keeps 11 windows × all models tractable while staying unbiased and identical across models; document the choice and the resulting CI on NLL. (If full-population evaluation proves cheap at scoring time, widen the block — the design doesn't change.)
3. Write `processed/training/<design>/{train_pool,eval_pool}/part-*.parquet`, sorted by `(shard, loan, period)`; ~64–128 MB row groups. Dev variant ≈ 5–10 M train-pool rows (small `p_keep`, optionally `sample_vintages`); full variant ≈ 50–100 M. A `manifest.json` records `p_keep`, the eval shard block, row counts per label year, and a content hash — every run folder references it.
4. Upload to the GPU box (rclone/gdrive). Loader (`data.py`): applies the window's label-month mask, then epoch = shuffle file/shard order → load one shard → shuffle in-memory → yield minibatches (the May-8 randomization scheme; never minibatch from a single vintage).

## 4. Model A — empirical transition matrix (`benchmarks.py`)

- 4×7 row-normalized counts on each window's **training slice (unthinned — it's one DuckDB `GROUP BY` per window, no need to touch the export)**, Laplace-smoothed (`α = 0.5`) so no test transition has probability 0. Trivially cheap → recompute for all 11 windows.
- Test NLL = mean −log p̂(state_next | state_t) on the window's test slice. This is the floor: it uses *no covariates*.
- Optional second benchmark, "bucketed matrix": condition on state × FICO-tercile × incentive-tercile. Nonparametric, still trivial; shows how far cheap conditioning gets before any fitting.

## 5. Model B — multinomial logistic regression

- Equivalent to a 0-hidden-layer network (paper §2.2); implement as such in PyTorch so the *entire* pipeline (data, loss, weights, eval) is shared and differences are attributable to architecture alone. Cross-check coefficients/NLL on a 1 M-row subsample against `sklearn` / `statsmodels` once (numerical sanity).
- L2-regularized; strength on val NLL.
- Optional "augmented logit" (paper's hand-crafted nonlinearity concession): + squared loan age, + spline/binned incentive. Cheap, and it sharpens the story: how much of the NN's edge is recoverable by manual feature engineering.

## 6. Model C — deep neural networks (`net.py`, `train.py`)

- **Architecture:** MLP on [standardized continuous ‖ embedded categoricals]. Depth grid `{1, 3, 5, 7}` hidden layers; start at the paper's optimum (5 layers: 200 then 140×4), ReLU. Width is a tunable but stay near the paper.
- **Regularization (the May-8 list):** dropout grid `{0, 0.2, 0.5}` on hidden layers; L2 `{0, 1e-5, 1e-4}`; early stopping on val NLL (patience ~3–5 evals). Replicate the paper's finding that dropout shifts the optimal depth.
- **Optimization:** Adam, lr ~1e-3 with decay-on-plateau, batch 4–16k (tabular MLPs like big batches on GPU), mixed precision. Importance weights from §3.2 enter the cross-entropy.
- **Ensemble (`ensemble.py`):** 8 independent five-layer nets — random init + bootstrapped/resampled shard subsets — **average predicted probabilities**. Produce the out-of-sample-NLL-vs-ensemble-size curve (paper Fig 7).
- **Hyperparameter protocol:** the full grid runs **once, on the tuning window (k = 2015)** at dev scale; select on that window's val NLL; spot-check the winner on one later window (e.g. k = 2019) to confirm the config isn't regime-specific. The chosen config is then **frozen** and `backtest.py` loops it over all 11 windows at full scale (per-window early stopping on that window's val; per-window scalers/vocab/incentive). Never re-tune on any test slice. Log every run (config + git commit + manifest hash + window + metrics) to `models/<run_id>/`.
- **Compute budget (settled):** every window gets empirical + logit + best single NN; the **ensemble of 8 runs only on 5 key windows** spanning regimes — k = 2015 (tuning), 2019 (calm), 2020 (COVID), 2023 (rate spike), 2025 (latest). ≈ 22 + 40 = ~60 full-scale fits. Consequence for reporting: the all-years pooled NLL and Table B compare empirical/logit/NN across all 11 years; the ensemble column is populated for the 5 key windows and the ensemble-vs-single-NN delta is reported there.

## 7. Evaluation (`evaluate.py` — one script, every model through it)

- **Table A — tuning-window depth grid** (paper Table 11 analogue, window k = 2015): in-sample and out-of-sample NLL for: empirical matrix, bucketed matrix, logit, augmented logit, NN×{1,3,5,7}, NN-5 + dropout, ensemble.
- **Table B — the headline rolling results:** test-year (2015…2025) × model (empirical, logit, best NN, ensemble) NLL matrix, plus the pooled all-years NLL; figure version = NLL by test year with one line per model. Also pooled NLL by origin state (the gains concentrate in `current` rows: prepayment).
- **AUC table:** for each origin state and each destination, one-vs-rest AUC on pooled test predictions (paper §3.4); report logit vs best NN vs ensemble.
- **Calibration:** reliability diagrams for `current→prepaid` and `current→dpd_30` (pooled, and separately for 2020–21 vs other years); predicted vs realized monthly aggregate transition rates over 2015–2025 (time-series overlay — also a Phase-3 sanity input).
- **QA assertions:** ensemble mean predicted base rates within tolerance of unthinned training base rates (importance-weighting check, `00_OVERVIEW §6.4`); predicted mass on impossible transitions < 0.1%.

## 8. Robustness & overfitting checks

(The rolling backtest is the *main design*, §1 — so robustness here means: is the model comparison itself stable?)

1. **Depth/dropout ablation table** (§6 grid, tuning window) — the direct overfitting exhibit.
2. **Ensemble-size curve** (1→8, tuning window).
3. **Seed variance:** 3 seeds for the best NN on the tuning window; report NLL mean ± sd (is the NN-vs-logit gap >> seed noise?).
4. **Ranking stability across windows:** from Table B, the model ordering per test year — does the NN edge hold in every regime (notably 2020–21 COVID and the 2022–23 prepay collapse), or only on average? This is the regime-shift exhibit replacing a fixed COVID split.
5. **Tuning-window sensitivity:** re-run the (pruned) grid on one later window (k = 2019); confirm the selected config is unchanged or note the difference.
6. **Feature-importance sanity (optional):** permutation importance on pooled test for the top NN; confirms incentive/FICO/LTV/age dominate (ties back to EDA).

## 9. Interim write-ups

- **Memo 2a (after §4–5):** benchmark results on the tuning window — empirical matrix + logit NLL/AUC, plus the augmented-logit result. ~2 pages, the "linear baseline" section of the dissertation.
- **Memo 2b (after §6–8):** Table A (depth grid), Table B (rolling results 2015–2025 + pooled), ensemble curve, calibration figures, robustness. ~4 pages — the core results chapter in embryo.

**Acceptance for Phase 2:** per-window frozen test slices evaluated identically across all models (row-count + hash asserted); Tables A & B + AUC table + calibration + all robustness items exist as regenerable scripts/figures; every model×window has a self-describing run folder; memos committed.
