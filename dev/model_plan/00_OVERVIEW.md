# Modelling Master Plan — Sirignano-Style Transition Models on the Fannie Mae Panel

> **Purpose of this folder.** The data pipeline (`dev/pipeline/`) is built: the SSD holds a cleaned loan-month panel with the seven-state target (`state`, `state_next`, `censored`), leakage-safe calendar columns (`period_ym`, `orig_ym`), a loan-keyed `shard` for minibatch randomization, and train-only scaler discipline. This folder specifies the **analysis and modelling** that sits on top of it, replicating Sirignano, Sadhwani & Giesecke (*Deep Learning for Mortgage Risk*, JFEC 2021) on this dataset. Read order: this file → `01_EDA.md` → `02_LOAN_LEVEL.md` → `03_POOL_LEVEL.md` → `04_TASKS.md` (sequenced tasks with acceptance criteria). The invariants in the root `CLAUDE.md` and `dev/pipeline/CLAUDE.md` continue to bind.

---

## 1. Objective in one paragraph

Estimate the **monthly conditional state-transition distribution** of US mortgages — `P(state_{t+1} | state_t, features_t)` over `{current, dpd_30, dpd_60, dpd_90plus, foreclosure, REO, prepaid}` — with deep neural networks, and quantify the value of nonlinearity by benchmarking against (a) an empirical (frequency-count) transition matrix and (b) multinomial logistic regression, using strictly out-of-sample negative log-likelihood and per-transition AUC. Then roll the fitted loan-level model forward to predict **pool-level** prepayment and delinquency counts over a 12-month horizon, comparing models on predictive accuracy at the pool level — the economically relevant object for MBS analysis.

## 2. The three phases (sequential)

| Phase | Spec | Output |
|---|---|---|
| **1. EDA & motivation** | `01_EDA.md` | Figures + summary tables establishing data scope, transition base rates, and *prima facie* nonlinearity/interaction evidence; short memo |
| **2. Loan-level models** | `02_LOAN_LEVEL.md` | Empirical matrix → multinomial logit → deep nets (depth grid, dropout/L2, ensemble); evaluation + robustness; two memos |
| **3. Pool-level models** | `03_POOL_LEVEL.md` | 12-month roll-forward of the loan-level model; pool-bucket predictions vs realized; memo |

Phase 3 depends on Phase 2's frozen models. Phases 1 and 2 can overlap partially (EDA first — it informs feature transforms).

## 3. Key design decisions (settled)

1. **Compute: cloud GPU** (Colab Pro / RunPod / Lambda). Training data is *exported* from the SSD as compact Parquet shards (`processed/training/`), uploaded once per split design. All training code must run both locally-on-CPU (smoke test on a tiny sample) and on a single cloud GPU. Checkpoints and metrics sync back to `models/` on the SSD.
2. **Split design: rolling expanding-window backtest is the primary design** (per Felipe, June 2026 — supersedes the fixed-split idea in the May-8 notes). One window per **test year** `k = 2015 … 2025` (2025 partial, to the release cutoff) — 11 windows:
   - **Train:** labels ≤ Dec(k−2) — expanding window starting 2000.
   - **Validation:** labels in year k−1 (≤ 1 yr, per the design) — used for **early stopping only**.
   - **Test:** labels in year k (1 yr ahead).
   - Splits are stated on **label months**; the implementing mask is on the feature month with the strict rule `period_ym < cutoff` (label is one month ahead — `01_SCHEMA.md §6.1`).
   - **Hyperparameters/architecture are tuned ONCE** on a designated tuning window (k = 2015; spot-check on one later window), then **frozen across all windows** — per-window re-tuning would be a compute explosion and a multiple-testing problem. Per-window val serves only early stopping; scalers/vocabularies/incentive index are re-derived per window from its train slice.
   - **Headline results:** per-test-year NLL/AUC by model, plus a **pooled all-years out-of-sample NLL** — every 2015–2025 test row is predicted exactly once, by a model that never saw it. Regime shifts (COVID 2020–21, the 2022–23 rate spike) appear as test-year effects rather than hinging on one cutoff choice.
3. **Scale: start small, scale up.** All code is developed and validated on a ~5–10 M row stratified sample (fits in RAM, minutes per epoch). Final headline models re-trained on a large export (target ~50–100 M training rows; revisit after seeing throughput). The empirical matrix and logit benchmarks are cheap and can use the *full* training slice via DuckDB streaming regardless.
4. **Macro covariates: a small, fixed set joined as thin time-keyed tables at export time** (June 2026 decision — supersedes the earlier out-of-scope call; the lakes are never reprocessed). The set: **PMMS 30-yr mortgage rate** (the true incentive variable: `incentive = current_rate − pmms30(t)`; the panel-derived `mkt_rate` proxy of `01_EDA §4` is kept as a robustness check), **national + state unemployment**, **Freddie Mac FMHPI house-price index (national + state)** — which also unlocks **mark-to-market LTV**, the paper's key default driver — and the **10-yr Treasury**. Rules: per-variable publication lags (rates 0m, unemployment 1m, HPI 2m); final-revision series, not real-time vintages (one-sentence stated limitation). Sources, download steps, build + join spec: **`05_MACRO_DATA.md`** (standalone, executable with Claude Code).

## 4. Known deviations from the paper (state these in the writeup)

- **Dataset:** Fannie Mae public dataset (30-yr FRM, conventional, LTV ≤ 97) vs the paper's CoreLogic private-label data. Cleaner credit box, no ARMs/subprime → expect lower base rates of delinquency and different nonlinearity strength.
- **State observability:** in this dataset `foreclosure`, `REO`, and `prepaid` are **terminal** (derived from Zero Balance codes in the loan's final month); there is no ongoing "in foreclosure" monthly state as in CoreLogic. Origin states are effectively `{current, dpd_30, dpd_60, dpd_90plus}`; destinations span all seven. The transition matrix is 4×7, not 7×7. All models condition on the same origin set, so comparisons remain apples-to-apples.
- **Macro covariates at state granularity, not the paper's zip/county level** (rates national, unemployment + HPI by state — `05_MACRO_DATA.md`). The rate, labor, and housing channels are all present, but local variation within a state is not — a stated limitation and a clean future-work hook (the join pattern extends to MSA/zip3 unchanged).
- **Sample period:** 2000–2025 (vs 1995–2014), so the test regime includes COVID forbearance — note that forbearance-era delinquency codes may behave unusually (check `dq_status` distributions in 2020–21 during EDA).

## 5. Repository layout this plan creates

(`dev/analysis/`, `dev/model/`, `writeup/memos/` exist with placeholder READMEs; the module files below are created by their tasks in `04_TASKS.md` — absence before then is expected.)

```
dev/
├── model_plan/            # these spec docs
├── analysis/              # Phase 1 EDA scripts (one figure/table per script, saved to outputs/figures)
└── model/                 # Phases 2–3 code
    ├── config.py          # split cutoffs, sample sizes, paths (imports pipeline config ROOT)
    ├── export.py          # build processed/training/<design>/ shards for GPU upload
    ├── features.py        # feature_spec → model matrix (embeddings map, scaler fit/apply)
    ├── data.py            # shard-streaming Dataset / DataLoader (epoch = shuffle shards → shuffle within)
    ├── benchmarks.py      # empirical transition matrix + multinomial logit
    ├── net.py             # MLP with categorical embeddings
    ├── train.py           # single-window training loop: early stopping on val NLL, checkpointing, metrics JSON
    ├── ensemble.py        # K independent fits + probability averaging
    ├── evaluate.py        # NLL / per-transition AUC / calibration on frozen test slices
    ├── backtest.py        # THE main harness: loops train.py/ensemble.py over the 11 rolling windows
    └── pool.py            # Phase 3: pool construction, 12-month roll-forward, pool metrics

writeup/memos/             # short interim write-ups (markdown), figures referenced from outputs/figures
models/                    # (SSD) one run-folder per fit: config.json, scaler.json, checkpoint, metrics.json
processed/training/        # (SSD) exported training shards per split design
```

## 6. Non-negotiables (carried over + new)

1. Pipeline invariants hold: stream via DuckDB/Polars, `raw/` immutable, `require_drive()`, idempotent stages.
2. **Temporal masks are strict** (`period_ym < cutoff`); censored rows (`state_next` null) are excluded from every likelihood.
3. **The test slice is never downsampled, reweighted, or touched during development.** All evaluation runs on the identical frozen test rows for every model — that's what makes the NLL comparison meaningful.
4. If training data is class-downsampled (thinning `current→current`), the loss must be **importance-weighted** (w = 1/keep-prob) so the model still estimates true conditional probabilities; reported probabilities must be verified against unsampled base rates (Phase 2 QA).
5. Scalers, embedding vocabularies, and the incentive index are fitted/derived **on the training slice only** (vocab/scaler) or contemporaneously (incentive), persisted under `models/`, and re-fitted per backtest window.
6. Every model fit writes a self-describing run folder (config, data hash, git commit, metrics) — reproducibility is a dissertation requirement.

## 7. Backup & recovery

The SSD is a single point of failure only for **small** artifacts; protect those and the rest is rebuildable:

| Tier | What | Protection |
|---|---|---|
| Code, specs, memos, writeup | this repo | git + GitHub (push after every milestone) |
| Trained models, run configs, scalers, metrics, QA reports, figures, tables, `mkt_rate`, export manifests | SSD `models/ outputs/ logs/ processed/macro/` + manifests | `dev/tools/backup_ssd.sh` → `ssd_mirror/` on the internal disk (gitignored). **Run after every milestone gate** — it's in the standing rules. |
| Parquet lakes (`interim/`, `processed/`) | SSD | not backed up — rebuildable from `raw/` by re-running the pipeline (~days, no thought) |
| `raw/` (~800 GB) | SSD | not backed up — re-downloadable from Fannie Mae; Stage 1 manifest (in `outputs/`, hence mirrored) provides checksums/row counts to verify a re-download |

Cloud-GPU runs add a free third copy of whatever was uploaded (training shards) and of checkpoints until the instance is torn down — sync checkpoints back to `models/` at the end of every GPU session, never leave the only copy on a rented box.

## 8. Definition of done

- Phase 1: figures + memo motivating nonlinear modelling on this data.
- Phase 2: (a) a tuning-window table replicating the paper's Table 11 structure (in/out-of-sample NLL: empirical matrix, logit, NN depths 1/3/5/7, ensemble); (b) the rolling-backtest results — per-test-year NLL/AUC by model 2015–2025 plus the pooled all-years NLL; (c) calibration plots and robustness (dropout/L2 ablation, ensemble-size curve, seed variance, ranking stability across windows). Memo per milestone.
- Phase 3: pool-level predicted-vs-realized analysis (scatter + R²/RMSE by pool bucket scheme) for NN vs logit vs empirical at the 12-month horizon, at a minimum of 3 window anchors (ideally all 11). Memo.
