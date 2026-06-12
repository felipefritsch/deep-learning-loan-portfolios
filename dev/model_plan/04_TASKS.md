# 04 — Sequenced Tasks (execute one at a time, verify Accept before advancing)

> Same contract as `pipeline_plan/03_CLAUDE_CODE_TASKS.md`: each task is small, has explicit acceptance criteria, and is committed before the next begins. Specs: `01_EDA.md` (M-tasks 1–3), `02_LOAN_LEVEL.md` (4–12), `03_POOL_LEVEL.md` (13–15).

---

### M1 — EDA scaffolding + coverage & state tables
Create `dev/analysis/` with a shared helper (DuckDB connection to the panel views, figure/table save conventions). Produce T1.1–T1.3.
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

### M4 — `dev/model/` scaffolding + export
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

### M12 — Robustness: ablations + stability + seeds
3-seed variance for the best NN (tuning window); ranking-stability readout from Table B (per-regime ordering); tuning-window sensitivity check (pruned grid on k=2019); optional permutation importance.
**Accept:** seed sd reported and small vs the NN-logit gap; ranking-stability paragraph drafted; k=2019 check documented. **Phase-2 gate.**

### M13 — Roll-forward harness
`pool.py` matrix-composition engine per `03 §3`, with first-passage variant for 60+ dpd.
**Accept:** toy-chain unit test matches closed form; h=1 probabilities equal `evaluate.py` outputs exactly; runs over all t0-alive loans in bounded memory.

### M14 — Pools + predictions + realized outcomes
Characteristic buckets + random pools at the window anchors (all 11 if scoring allows; minimum 2019-12, 2022-12, 2024-12 — the ensemble-bearing key windows), each scored with its own window's frozen models; predicted counts/intervals; realized counts from the panel.
**Accept:** pool memberships reproducible (seeded); realized counts reconcile with panel aggregates; predicted-vs-realized scatter + R²/RMSE table for ensemble vs logit vs empirical at every anchor run.

### M15 — Pool memo + (optional) portfolio exercise
`writeup/memos/03_pool.md`; optional §5 portfolio decile exercise.
**Accept:** memo committed with F4.1/T4.2/F4.3 and the covariate-freezing assumption stated. **Phase-3 gate.**

---

## Standing rules while executing

1. One task per session/branch; verify Accept criteria explicitly (paste the evidence) before moving on.
2. Anything touching the SSD goes through `require_drive()` and the existing config-derived paths.
3. GPU code must keep a `--smoke` CPU path (≤100k rows) so every task is testable locally before spending GPU time.
4. When a result contradicts the paper (possible — different dataset/credit box), record it in the memo rather than tuning until it agrees.
5. **After every milestone gate (M3, M12, M15) and every GPU session:** run `dev/tools/backup_ssd.sh` and `git push`. Checkpoints on a rented GPU box are synced back to `models/` before teardown — never the only copy.
