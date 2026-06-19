# 04 — Sequenced Tasks (execute one at a time, verify Accept before advancing)

> Same contract as `specs/pipeline/03_CLAUDE_CODE_TASKS.md`: each task is small, has explicit acceptance criteria, and is committed before the next begins. Specs: `01_EDA.md` (M-tasks 1–3), `02_LOAN_LEVEL.md` (4–12), `03_POOL_LEVEL.md` (13–15), `06_GBT_BASELINE.md` (16–19), `ECONOMIC_ENGINE.md` + `ADR-001-economic-engine-seam.md` (20–27, the post-supervision priority — see `writeup/memos/post_supervision_roadmap.md`).

---

### M1 — EDA scaffolding + coverage & state tables
Create `src/floan/analysis/` with a shared helper (DuckDB connection to the panel views, figure/table save conventions). Produce T1.1–T1.3.
**Accept:** tables regenerate with one command each, in bounded memory; T1.3 shows the expected `current` dominance; outputs land in `outputs/tables/eda/`.

### M2 — Transition-structure figures
T2.1 pooled empirical matrix; F2.2 transition-rate time series; F2.3 COVID delinquency-code check; F2.4 roll-rate areas.
**Accept:** 4×7 matrix rows sum to 1; F2.2 visibly shows the 2003/2020 refi waves and 2008–11 default wave (sanity that the target derivation is right); COVID handling decision documented in the script header and memo.

### M2b — Macro data download + tables
Execute `05_MACRO_DATA.md` (standalone spec): fetch the series snapshot, build `processed/macro/{macro_national,macro_state}.parquet`, run its QA checks, produce F4.2/F4.3.
**Accept:** the acceptance block of `05_MACRO_DATA.md §6` passes in full.

### M3 — Nonlinearity figures + incentive variable + memo 1
`mkt_rate.parquet` (proxy) built; F4.1 proxy-vs-PMMS validation; hazard curves F3.1–F3.3 (F3.2 on the PMMS-based incentive); interaction heatmaps F3.4–F3.5; vintage F3.6; shard checks F5.1–F5.2; write `writeup/memos/01_eda.md`.
**Accept:** proxy-vs-PMMS tracking error reported; F3.1 shows the seasoning hump and F3.4 a visible FICO×LTV interaction; memo committed. **Phase-1 gate.**

### M4 — `src/floan/model/` scaffolding + export
`config.py` (the 11 rolling windows k = 2015…2025 with train/val/test label-month masks, `p_keep`, eval shard block, paths), `export.py` per `02 §3`: one shared **train pool** (thinned, weighted, all years) + **eval pool** (unthinned, fixed shard block, label years ≥ 2014). Dev-scale variant first (~5–10 M train-pool rows).
**Accept:** `manifest.json` row counts per label year reconcile with direct DuckDB counts; eval pool is loan-disjoint-by-shard and contains zero thinned rows; window masks from `config.py` slice both pools correctly (spot-check k=2015 and k=2025); macro columns joined with correct per-variable lags (spot-check one loan-month against the source tables) and `unrate_fallback` indicator present; export re-runs idempotently.

### M5 — Empirical matrix benchmark (all windows)
`benchmarks.py`: per-window 4×7 matrix (window train mask via DuckDB on the unthinned panel, Laplace α=0.5), NLL on each window's frozen test slice; bucketed-matrix variant.
**Accept:** all 11 test NLLs finite (smoothing works); pooled matrix ≈ M2's in shape; NLLs reproducible bit-for-bit across two runs.

### M6 — Feature pipeline + loader
`features.py` (scaler fit/apply, embedding vocab + UNK, one-hot path for logit), `data.py` (shard-streaming loader per `02 §3.4`).
**Accept:** unit tests — scaler train-only (val stats differ), UNK mapping on a synthetic unseen level, `ltv_mtm`/`hpi_chg_12m` formulas on a synthetic loan, loader yields every row exactly once per epoch with shuffled order, weights present and correct (`1/p_keep`).

### M7 — Logit benchmark (0-layer net, tuning window k=2015)
PyTorch multinomial logit through the shared loss/eval path; L2 on val; sklearn cross-check on 1 M rows; optional augmented logit.
**Accept:** PyTorch and sklearn NLL agree to ~1e-3 on the cross-check; beats the empirical matrix out-of-sample (it uses covariates — if not, stop and debug); run folder written.

### M8 — Single NN end-to-end (tuning window, dev scale)
`net.py` + `train.py`: 5-layer net, dropout 0.2, early stopping on the 2014 val slice, CPU smoke test then cloud-GPU run on the dev export.
**Accept:** trains without OOM; val NLL improves on logit; checkpoint + metrics.json + scaler + manifest hash + window id in the run folder; resumable from checkpoint.

### M9 — Depth/regularization grid (tuning window) + Table A
Grid `{1,3,5,7} × dropout {0,0.2,0.5} × L2 {0,1e-5,1e-4}` on the tuning window, dev export (prune sensibly — full factorial not required; anchor on the paper's optimum). Ensemble of 8 + ensemble-size curve on the tuning window. Freeze the winning config.
**Accept:** Table A (paper-Table-11 analogue, incl. ensemble) complete for k=2015; selected config documented with rationale and frozen in `config.py`; replicates qualitatively the paper's dropout-depth interaction or the deviation is noted.

### M10 — Full-scale export + rolling loop (`backtest.py`)
Full export (~50–100 M train-pool rows, revisit after M8 throughput); **first, a full-scale depth check** (M9 selected depth 3 on the 3M-row dev slice and flagged it as scale-sensitive: re-fit depth 3 vs 5 on the full-scale tuning window, take the val-NLL winner, update the frozen config with rationale); then `backtest.py` loops the frozen config over all 11 windows: per-window logit + best single NN everywhere; **ensemble of 8 on the 5 key windows only (2015/2019/2020/2023/2025, per `02 §6`)**; per-window scalers/vocab/incentive re-derived from that window's train slice.
**Accept:** 11 × {logit, NN} + 5 × ensemble run folders exist; per-window early stopping used that window's val only; spot-check one window for scaler/vocab locality (no leakage); ensemble ≥ best single net on its key windows; base-rate QA passes (`02 §7` assertions).

### M11 — Evaluation suite + memo 2a/2b
`evaluate.py`: Table B (test-year × model NLL + pooled), NLL by origin state, AUC table on pooled predictions, calibration figures (pooled + 2020–21 split), predicted-vs-realized rate overlay 2015–2025. Write memos 2a (benchmarks) and 2b (headline).
**Accept:** every model evaluated on identical per-window frozen test rows (assert row-count + hash); each test row appears exactly once in the pooled set; figures regenerate; memos committed.

### M11b — Thesis draft sync (Mac, SSD mounted)
The dissertation draft (`writeup/latex/`) was assembled mid-Phase-2 with framed placeholders and chat-transcribed numbers. Close both gaps: (1) **verify every number quoted in `writeup/latex/chapter4.tex` §4.1** (grid findings, selected-config NLLs, ensemble numbers, depth-check statement) against the authoritative run-folder `metrics.json` / `table_a.*` artifacts and correct any discrepancy; (2) **swap placeholders for real artifacts** — copy the EDA figures (`outputs/figures/eda/F2.2, F3.1–F3.6`), the loan-level artifacts (Table A, Table B, AUC table, calibration figures, ensemble-size curve) and the covariate summary table (T1.2) into `writeup/latex/` (a `figs/` subfolder), replacing each `\figplaceholder`/`\tabplaceholder` in chapters 3–4 with the real `\includegraphics`/`\input`.
**Accept:** chapter4 §4.1 numbers match run artifacts (state the diff if any was found); no placeholder remains in chapters 3–4 except the Phase-3 ones (`fig:poolscatter`, `tab:poolaccuracy`, `tab:econerrors`, `fig:pricerrorbuckets`); `main.tex` recompiles cleanly with zero undefined references.

### M12 — Robustness: ablations + stability + seeds
3-seed variance for the best NN (tuning window); ranking-stability readout from Table B (per-regime ordering); tuning-window sensitivity check (pruned grid on k=2019); **width sweep** at the selected config (half / paper / double the Sirignano layer widths, tuning window, dev scale — completeness check for the appendix: widths were inherited from the paper, not tuned); optional permutation importance.
**Accept:** seed sd reported and small vs the NN-logit gap; ranking-stability paragraph drafted; k=2019 check documented; width-sweep table written to `outputs/tables/loan_level/` (appendix artifact) with a one-line conclusion. **Phase-2 gate.**

### M13 — Roll-forward harness
`pool.py` matrix-composition engine per `03 §3`, with first-passage variant for 60+ dpd.
**Accept:** toy-chain unit test matches closed form; h=1 probabilities equal `evaluate.py` outputs exactly; runs over all t0-alive loans in bounded memory.

### M14 — Pools + predictions + realized outcomes
Characteristic buckets + random pools at the window anchors (all 11 if scoring allows; minimum 2019-12, 2022-12, 2024-12 — the ensemble-bearing key windows), each scored with its own window's frozen models; predicted counts/intervals; realized counts from the panel.
**Accept:** pool memberships reproducible (seeded); realized counts reconcile with panel aggregates; predicted-vs-realized scatter + R²/RMSE table for ensemble vs logit vs empirical at every anchor run.

### M15 — Economic translation (counts → CPR/WAL/price) + pool memo
Implement `03_POOL_LEVEL §5`: UPB-weighted pool SMM/CPR paths from the roll-forward; the level-pay pass-through cashflow engine (unit-tested against the zero-prepay annuity and constant-SMM closed forms); CPR/WAL/price errors per pool × model × anchor → **T5.1** + **F5.2**. Write `writeup/memos/03_pool.md` including the regime exhibit framing and the closing contributions paragraph. The portfolio decile exercise is deferred (future-work note only).
**Accept:** engine matches both closed forms to ~1e-8; T5.1/F5.2 exist for ensemble vs logit at ≥3 regime-spanning anchors with the headline |price-error| reduction stated; memo committed with F4.1/T4.2(+by-anchor)/F4.3/T5.1/F5.2, the frozen-macro caveat paragraph, and the contributions paragraph; **the remaining Phase-3 placeholders in `writeup/latex/chapter4.tex` §4.3 are swapped for the real artifacts** (M11b pattern: copy into `writeup/latex/figs/`, replace `\figplaceholder`/`\tabplaceholder`, recompile cleanly). **Phase-3 gate.**

### M16 — GBT trainer on the tuning window (k=2015, dev scale)
`src/floan/model/gbt.py`: a **single** 7-class multiclass-softmax LightGBM (`objective="multiclass"`, `num_class=7`, `metric="multi_logloss"`) with origin `state` as a native `categorical_feature` — **not** four per-origin models (`06 §2.1`). Same `features.py` columns as the nets **minus standardisation**: raw continuous (NaN-native, no scaler/log), the UNK-routed vocab codes for the categoricals, and the binary missingness indicators passed so the feature set matches (`06 §2.2/§3`). HT `weight = 1/p_keep` on the **train** Dataset only; val/test unthinned and unweighted (`06 §2.3` — the eval pool already carries `w≡1`). Early stopping on the k=2015 val slice (year 2014) `multi_logloss`. A small pruned grid (`num_leaves ∈ {31,63,127}`, `learning_rate ∈ {0.05,0.1}`, `min_sum_hessian_in_leaf` swept, `feature_fraction ∈ {0.7,1.0}`, `lambda_l2 ∈ {0,1}`, `06 §3`), selected on the 2014 val NLL, winner frozen in `config.py` (`GBT_SELECTED`, with rationale). CPU only — a documented deviation from the GPU default (`06 §2.4`); self-describing run folder (`00_OVERVIEW §6.6`). Glance at gain feature importance to confirm no single high-card level (zip3/MSA) split dominates (native-categorical overfit) — raise `min_data_in_leaf`/`cat_smooth` if so, diagnostic only, not gated. Add `lightgbm` to `pyproject.toml`.
**Accept (the band is a CORRECTNESS check, not a performance target — never tune GBT to chase the NN number):** predicted probabilities sum to 1 and contain no NaN; LightGBM's reported `multi_logloss` on the eval slice equals `evaluate._nll` on the same `(n,7)` predictions to ~1e-6 (the GBT analogue of M7's sklearn cross-check, `06 §3`); base-rate QA passes (predicted mean class rates ≈ realized, `02 §7`) and structural-impossible-cell mass < 0.1% (`backtest._structural_allow`). The k=2015 tuning-window OOS test NLL reads in three zones: **hard stop** if it is *worse than* the empirical-matrix floor (~0.1127) — that is a wiring bug; **yellow flag** if between the bucketed matrix (~0.1092) and the floor — review the wiring (categoricals entering natively, HT weights train-only, early stopping not too aggressive) and, only if the wiring is clean, record it as a finding (standing rule 4) without tuning toward it; **healthy** in ~0.103–0.109, where GBT lands *is* the result (near the NN's ~0.1032 ⇒ flexibility wins; near the bucketed matrix ⇒ the smoothness story, `06 §7`).

### M17 — GBT in the rolling loop (`backtest.py`, all 11 windows)
Add the frozen GBT config to `backtest.py`'s per-window loop: per-window vocab + HT-weighted train fit + early stopping on that window's val, scored on the frozen test slice — **no change to the window/masking logic** (`06 §5`), each fit a self-describing run folder (`gbt/<variant>/k{k}/`). First re-confirm the frozen grid on the **full-export** tuning window before looping (the M9→M10/M10b dev-then-full pattern, `06 §3`).
**Accept (wiring correctness):** 11 GBT run folders exist; per-window early stopping used that window's val only; spot-check one window for vocab locality (no leakage); base-rate QA passes per window (`02 §7`). **Reported result, not gated:** the per-window GBT-vs-logit (and GBT-vs-NN) NLL comparison — a window where GBT trails logit gets a wiring check, then is recorded as a possible regime finding (standing rule 4), never tuned away.

### M18 — GBT into the loan-level exhibits + calibration decision
Fold GBT in as a fifth model column on the identical frozen test rows: Table A (tuning-window grid), Table B (rolling + pooled), NLL-by-origin-state, and the one-vs-rest AUC table — read especially in the sparse deep-delinquency cells (`90+→Foreclosure`) where the net showed its edge (`06 §6`). The NLL/AUC/calibration **math** in `evaluate.py` is model-agnostic and untouched; `score_window` gains a GBT branch and `MODEL_ORDER`/`MODEL_LABELS` a `gbt` entry (the spec's "no change to evaluate.py" holds for the metric path, not the model enumeration — surface this, don't code around it). **Calibration is conditional** (`06 §4`): run the existing reliability diagrams on **raw** GBT outputs; if on-diagonal, document and do nothing; if not, fit per-window temperature scaling on the **val** slice (per-class isotonic + renormalise as fallback) and apply before every levels-consuming step. Results fold into memo 2b + chapter 4 via the M11b sync pattern.
**Accept:** GBT column present in Table A/B (+pooled), NLL-by-origin, and AUC on identical frozen test rows (row-count + hash asserted, `02 §7`); each pooled test row still counted once; calibration decision documented (applied-with-evidence or skipped-with-evidence); memo 2b / chapter-4 tables regenerate.

### M19 — GBT into the pool roll-forward (`pool.py` model-agnostic seam) + memo
Add **one model-agnostic predictor seam** to `pool.py` — a callable returning the per-loan 7-vector at the evolved covariates — so the composition (`03 §3`) and cashflow (`03 §5`) engines score the net **or** the GBT booster with the engine logic otherwise unchanged (the current `_origin_scores`/`_chunk_matrices` path is torch-specific: `model(cont,cat,binb)`→float64 softmax; GBT scores `booster.predict(X)` with the `state` column overridden per origin in the code matrix). Apply the M18 calibrator inside the roll-forward predictor if `06 §4` triggered. Produce GBT columns in T4.2 (count R²/RMSE), T5.1 (economic error) and F5.2 (price-error buckets) at the same regime anchors (`06 §6`); fold into memo 03_pool with the CPU/GPU deviation (`06 §2.4`) and the pre-registered result framing (`06 §7`).
**Accept:** h=1 GBT composition equals the GBT loan-level prediction on the same anchor rows (the M13 Accept #2 analogue); GBT columns in T4.2/T5.1/F5.2 at ≥3 regime-spanning anchors via the unchanged engine; memo 03_pool + chapter tables updated; the §7 framing and §2.4 deviation stated. **GBT-baseline gate.**

---

## Post-supervision workstream (M20–M27)

> Spec: `ECONOMIC_ENGINE.md`; architecture: `ADR-001-economic-engine-seam.md`; rationale + sequencing:
> `writeup/memos/post_supervision_roadmap.md`. **Critical path:** M20 → M21 → {M22, M23} → M24, with M25
> foldable any time after M21. **Workstream B (M26→M27) runs in parallel and must not block M20–M24.**
> **Workstream C (standby, no new compute):** the banked loan-level GBT sweep and the COVID inversion are
> preserved as a finished sub-result; the spline-logit-across-all-windows probe, EBM/GA2M, and further GBT
> elaboration are paused — revive only if A and B leave timeline.

### M20 — Impossible-cell mask correction, applied uniformly *(QA + pricing prerequisite)*
Per `ECONOMIC_ENGINE §3.5` (supervision brief §7). **Permit** the four reporting-gap-legal skip-bucket cells
(`current→dpd_60`, `current→dpd_90plus`, `dpd_30→dpd_90plus`, `dpd_90plus→REO`) in `backtest._structural_allow`
(`02 §7`); **separately zero** the two mechanically-impossible cells (`current→foreclosure`, `current→REO`) in the
roll-forward predictor before any levels-consuming step. Apply to **all** models including the nets; re-score the
nets under the corrected mask (the brief's "QA consistency" ask).
**Accept:** post-correction residual impossible mass < 1e-4 every window; the four legal cells no longer flagged;
nets re-scored, QA gate passes honestly with the **threshold unchanged** (`02 §7`); the two-cell zeroing is
unit-tested to change pooled NLL by ≤ 1e-9 while removing the phantom high-LGD mass. **Gate before M21–M24.**

### M21 — Predictor seam + horizon parameter *(extends M13/M19, governed by ADR-001)*
Per `ADR-001` + `ECONOMIC_ENGINE §3`. Extract the `Predictor` protocol from `pool._origin_scores`; wrap the torch
path as `TorchPredictor`; make `horizon` a parameter of `roll_forward`; thread the optional `calibrator`/`absorb`.
**No change** to `compose`/`absorb_rows`/`assemble_matrix`/`cashflow_engine`.
**Accept:** `roll_forward(..., horizon=1)` equals `evaluate.py` outputs **exactly** (the M13 Accept-#2 identity,
parameterized — the regression guard); H ∈ {1,3,6,12} runs complete in bounded memory at ≥3 anchors; a stub second
`Predictor` (e.g. empirical-matrix) flows through identical engine math (proves the seam is model-agnostic).

### M22 — Loan-level cashflow/valuation *(the supervisor's explicit ask)*
Per `ECONOMIC_ENGINE §3.4`. Price at the **loan** level via `cashflow_engine` (a one-loan "pool"), then aggregate.
**Accept:** loan→pool aggregation identity holds to ~1e-6 (`ECONOMIC_ENGINE §4.3`); loan-level WAL/price tables exist
at ≥3 anchors × H ∈ {1,3,6,12}; the closed-form cashflow tests (zero-prepay annuity, constant-SMM survival, `03 §5`)
pass on the per-loan path.

### M23 — Per-horizon calibration decision *(generalizes M18's calibration step)*
Per `ECONOMIC_ENGINE §6` / `06 §4`. Reliability diagrams on **raw** outputs **per horizon**; if off-diagonal, fit
per-window temperature scaling on the **val** slice (per-class isotonic + renormalise fallback) and apply via the
`§3.2` calibrator seam (before assembly, `§4.2`). Report **calibrated-vs-raw** price error **by H**.
**Accept:** calibrated-vs-raw price-error table by H ∈ {1,3,6,12} at ≥3 anchors; the calibrator is a no-op identity
when disabled (regression check); decision documented (applied-with-evidence or skipped-with-evidence).

### M24 — Rolling pricing backtest: the horizon × regime grid *(extends M15/`03 §5`)*
Per `ECONOMIC_ENGINE §5,§9`. Run M21–M23 across all available anchors × H; extend T4.2/T5.1/F5.2 with an H dimension.
**Accept:** T4.2/T5.1/F5.2 carry an H axis at ≥3 regime-spanning anchors; the headline ensemble/NN-vs-logit
price-error reduction by anchor × H is stated; the COVID-inversion-in-dollars is reproduced and shown to **deepen
with H**; memo 03_pool + chapter tables synced (M15 pattern).

### M25 — AUC as the cross-period display metric *(presentation; foldable after M21)*
Per `ECONOMIC_ENGINE §7`. NLL stays the estimation/selection/within-window metric. Add per-window one-vs-rest AUC
for `current→prepaid`, `dpd_90plus→foreclosure`, `current→dpd_30`, plotted **2015–2025 per model** — the AUC mirror
of Table B. `evaluate.py`'s metric path is unchanged; add a per-window AUC time-series driver that calls the existing
AUC function per window.
**Accept:** an AUC-by-window figure + table per key transition; an explicit "**AUC illustrates, NLL decides**" note
in the memo; every AUC traces to `evaluate.py` on the identical frozen test rows.

### M26 — Sequence-model feasibility spike *(Workstream B; parallel; strict go/no-go)*
Per the roadmap's Workstream B. Stand up a per-loan **sequence** data path (loan-keyed shards already co-locate a
loan's full history), length-bucketing/padding, and a loan-level shuffle; smoke-test one RNN and one
attention/transformer block on the tuning window at dev scale. **Frame it as a Markov-assumption test:** does loan
history beyond the current state predict next-state transitions (cured-from-delinquency vs continuously-current)?
Time-box to a fixed budget.
**Accept:** a dev-scale sequence model trains without OOM and is scored through the **same** `evaluate.py` NLL/AUC
path on the frozen test rows; a one-page go/no-go memo states the val-NLL delta vs the feed-forward net (the "value
of memory" rung) and an honest integration-cost estimate. **Parallel — must not block M20–M24.**

### M27 — Sequence-model estimation *(conditional on M26 = go)*
If go: roll the chosen architecture over the windows (frozen-config protocol, `02 §6`); calibrate per M23; feed its
probabilities into the engine via a `SeqPredictor` (`ECONOMIC_ENGINE §3.1`) for loan- and pool-level pricing. If the
seam needs per-loan history rather than point-in-time covariates, record **ADR-002** first.
**Accept:** sequence-model column in Table B (+pooled), AUC-by-window (M25), and the pricing tables, on identical
frozen test rows; the Markov-test conclusion stated; if a signature change was needed, ADR-002 is committed.

---

## Standing rules while executing

1. One task per session/branch; verify Accept criteria explicitly (paste the evidence) before moving on.
2. Anything touching the SSD goes through `require_drive()` and the existing config-derived paths.
3. GPU code must keep a `--smoke` CPU path (≤100k rows) so every task is testable locally before spending GPU time.
4. When a result contradicts the paper (possible — different dataset/credit box), record it in the memo rather than tuning until it agrees.
5. **After every milestone gate (M3, M12, M15) and every GPU session:** run `scripts/backup_ssd.sh` and `git push`. Checkpoints on a rented GPU box are synced back to `models/` before teardown — never the only copy.
