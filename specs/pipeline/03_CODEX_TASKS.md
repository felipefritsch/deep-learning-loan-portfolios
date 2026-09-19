# 03 — Codex Build Order (sequenced tasks)

> Copy-paste these prompts into Codex **one at a time, in order**. Each has acceptance criteria — do not advance until they pass. All code goes in `src/floan/pipeline/`. Read `00_OVERVIEW.md`, `01_SCHEMA.md`, `02_PIPELINE_STAGES.md` first.
> Execute every task per the Karpathy guidelines in `../pipeline/AGENTS.md` (think before coding, simplicity first, surgical changes, verify against each task's **Accept** criteria).

---

### Task 0 — Scaffold & config
> "Create the `src/floan/pipeline/` package per `00_OVERVIEW.md §6`. Write `config.py` exactly as in `02_PIPELINE_STAGES.md`: a single `ROOT = /Volumes/SSD Felipe/dissertation` with named subpaths (`RAW`, `INTERIM`, `PROCESSED`, `MODELS`, `OUTPUTS`, `LOGS`) and derived dirs (`RAW_DIR=RAW/'Performance_All'`, `PERF_DIR`, `CLEAN_DIR`, `PANEL_DIR`, `SAMPLE_DIR`), `DUCKDB_PATH` on the internal disk, the memory/chunk/zstd knobs, and a `require_drive()` guard. No scattered absolute paths — everything derives from `ROOT`. Add `requirements.txt` (duckdb, polars, pyarrow, tqdm). Create the SSD subdirs (`raw/ interim/ processed/ models/ outputs/ logs/`) and the internal `reports/`. **Never write to `raw/`.** Don't process data yet."

**Accept:** package imports cleanly; `require_drive()` raises a clear error when the SSD is absent and passes when mounted; all subpaths derive from `ROOT`; `raw/` is treated as read-only; no data touched.

---

### Task 1 — Schema module
> "Translate `01_SCHEMA.md §1` into `schema.py`: `COLUMNS` (list of 113 names, positions 0–109 as specified, 110–112 named `extra_11x`), `DTYPES` dict, the §2 categorical maps, a `parse_mmyyyy()` helper, sentinel-scrub helpers, `KEEP_COLS` (§5), and `derive_state()` returning a Polars expression implementing the §4 seven-state logic. Also add the §6 training-prep constants: `SHARD_SEED` (pinned), a `feature_spec` dict classifying columns by ML role (`continuous_standardize`/`categorical`/`binary`/`identifier`/`time`/`target`/`split`, with `log`-transform flags), and the derived-column name lists (`period_ym`, `orig_ym`, `shard`, missingness indicators). Add a unit test asserting `len(COLUMNS) == 113`."

**Accept:** `len(COLUMNS)==113`; `derive_state` covers all 7 states + null; `feature_spec` covers every kept/derived column exactly once and only references known names; `SHARD_SEED` is pinned; tests pass.

---

### Task 2 — Stage 1 inventory (RUN IT)
> "Implement `s1_inventory.py` per `02_PIPELINE_STAGES.md §Stage 1` and run it over all ~100 vintage files. Produce `outputs/inventory_manifest.csv` and `outputs/inventory_summary.md`. Stream — never load a full file. Record each file's release cut-off (`max(Monthly Reporting Period)`), check the 2000Q1→latest vintage sequence is contiguous, and quarantine any file with <1000 distinct loans, <50 MB beside GB neighbours, or a field count ≠ 113."

**Accept:** manifest lists every vintage found; vintage sequence contiguous (missing ones listed); single shared release cut-off confirmed (divergences flagged); truncated/corrupt files quarantined. **Show me the summary before continuing.**

---

### Task 3 — Stage 2 convert (one quarter, then all)
> "Implement `s2_to_parquet.py` (DuckDB `COPY` engine, string-typed faithful copy, zstd, Hive partition by `acq_quarter`). First convert ONLY the agreed small healthy smoke-test vintage (`2017Q2`) and report Parquet size + round-trip row count. Then, once I confirm, convert all non-quarantined vintages via `run.py s2 --all`."

**Accept (smoke):** the smoke vintage round-trips exactly; Parquet noticeably smaller than CSV. **Accept (full):** every non-quarantined vintage present; total ≈120–200 GB; peak RAM stays under the limit on the largest (tens-of-GB) file.

---

### Task 4 — Stage 3 clean
> "Implement `s3_clean.py` with Polars `scan_parquet`→`sink_parquet` (streaming, never `collect` the full frame). Apply all §3 casts, project to `KEEP_COLS`, coalesce origination FICO via `schema.fico_orig_expr()` (handles the Dec-2025 Classic-FICO migration), add the §6.1 calendar columns (`period_ym`, `orig_ym`) and §6.3 missingness indicators, write `CLEAN_DIR` (`interim/clean/`). Run on the agreed smoke-test quarter first (`2017Q2` — `2014Q1` is quarantined), then `--all`."

**Accept:** cleaned partitions hold only keep-list + §6 derived cols; no `MMYYYY` strings; `period_ym`/`orig_ym` are valid integer year-months; missingness indicators present and consistent with scrubbed nulls; row counts match perf lake; runs within memory limit.

---

### Task 5 — Stage 4 panel
> "Implement `s4_panel.py`: derive `state`, `state_next` (per-loan `LEAD`/`shift` over `period`), and `censored` (keyed off the data cut-off **derived** as max `period` — never hardcoded). Add the §6.2 `shard = hash(Loan Identifier, seed=SHARD_SEED) % N_SHARDS` and write the panel **sorted by `(shard, Loan Identifier, period)`**. Document the absorbing-state convention in the docstring. Print the 7×7 transition matrix and a shard-uniformity check for the smoke-test quarter. Then `--all`."

**Accept:** `state` ∈ 7-set∪null; transition matrix mass concentrated on `current→current`; `censored` present and keyed off the derived cut-off; `shard` ∈ `[0, N_SHARDS)`, roughly uniform, and constant within each loan; output sorted by `(shard, loan, period)`; `(Loan Identifier, period)` unique.

---

### Task 6 — Stage 6 QA (run before any raw deletion)
> "Implement `s6_qa.py`: row reconciliation, null/missingness audit, target distribution, `(loan,period)` uniqueness, feature scale stats (→ `outputs/feature_stats.csv`), shard/calendar sanity (uniform `shard`, loan never spans shards, valid `YYYYMM`, report derived cut-off), coverage report → `outputs/qa_report.md`. Run across all processed quarters."

**Accept:** reconciliations pass; duplicates=0; shard uniform & loan-consistent; `feature_stats.csv` written; report written. Only now may `--delete-raw` be used per quarter.

---

### Task 7 — Stage 5 sampling & training helpers
> "Implement `s5_sample.py`: DuckDB view over the panel lake, `sample_by_state()`, `slice_quarter()`, `transition_matrix()`, and a stratified sampler that keeps all foreclosure/REO transitions and downsamples `current→current`. Add the §6 training helpers: `iter_shards()` (cheap per-shard reads via `WHERE shard=k`), `training_mask(cutoff_ym)` (leakage-safe `period_ym < cutoff_ym`), and `fit_scaler()/apply_scaler()` (train-only stats per `feature_spec`, persisted to `models/scaler_<cutoff>.json` — never standardize the lake). Also add the OPTIONAL `sample_vintages(spec=VINTAGE_SAMPLING)` helper per `02 §Stage 5` — loan-level keep-fractions per vintage range, defaulting to 1.0 everywhere (a no-op); enforce in its docstring that it's train-slice-only, never thins the crisis cohorts, and that base rates must be reported on the unsampled panel. Demonstrate a 5M-row balanced sample built in under a minute within the memory limit."

**Accept:** balanced sample builds fast, preserves minority transitions, returns a Polars/pandas frame ready for modelling; `iter_shards` reads one shard without scanning the whole panel; `training_mask` excludes any example whose labelled month is ≥ cutoff; a fitted scaler is reproducible and saved under `models/`; `sample_vintages` with the default spec is a verified no-op and at a reduced fraction samples whole loans only.

---

### Task 8 — Orchestrator & docs
> "Finalise `run.py` (per-stage, per-quarter, `--all`, `--delete-raw`, `--include-quarantined`, logging to `logs/run.log`). Write a short `src/floan/pipeline/README.md` with the run sequence and the resumability/disk-discipline notes."

**Accept:** `python -m floan.pipeline.run --help` lists all stages; a full pipeline run on one quarter end-to-end succeeds and is resumable.

---

## Guardrails to repeat to Codex

- **Never** `pd.read_csv` a raw file, never `.collect()`/`fetchall()` a full quarter. Streaming only.
- **Always** verify field count == 113 and reconcile against the official Fannie Mae glossary before trusting names for positions 110–112.
- **Never** delete a raw CSV before its Stage 6 reconciliation passes.
- **Quarantined files** are excluded by default; surface them, don't silently drop or silently include.
- Keep stages idempotent and per-quarter so a crash is recoverable.
- **Never standardize the lake, and never leak the future.** Standardization stats are fitted on the *training slice only*, per backtest window, at model time (`01_SCHEMA.md §6.4`). The rolling-backtest mask is strict (`period_ym < cutoff_ym`) because `state_next` is one month ahead. The right-censoring cut-off is **derived** from the data (max `period`), never hardcoded — added vintages extend it.
- **`shard` is `hash(loan_id) % N_SHARDS`** with a pinned `SHARD_SEED`: deterministic, loan-keyed (a loan never spans shards), vintage-stable. It is a split/batch key, never a model feature.
- **Call `require_drive()` at the start of every stage** — raw data and the Parquet lake live on the external `SSD Felipe` (`/Volumes/SSD Felipe/`); fail fast with a clear message if it isn't mounted. Write Parquet to the same SSD as the raw.

---

### Recap

Nine sequential tasks take Codex from scaffold → schema module (incl. `feature_spec` + shard/calendar constants) → inventory (run and reviewed before proceeding) → streaming Parquet conversion (smoke-tested on `2017Q2` first, since `2014Q1` is quarantined) → cleaning (+ `period_ym`/`orig_ym` + missingness indicators) → transition panel (+ derived-cut-off `censored` and `shard`, sorted by `(shard, loan, period)`) → QA reconciliation (+ scale stats, shard/calendar sanity) → stratified sampling & training helpers (shard iterator, leakage-safe mask, train-only scaler) → orchestrator. Each task carries explicit acceptance criteria, and standing guardrails enforce the no-full-load rule, schema verification, deletion safety, quarantine handling, idempotency, no-lake-standardization/no-leakage, and the loan-keyed shard convention.
