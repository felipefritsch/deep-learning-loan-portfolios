# CLAUDE.md — Fannie Mae Loan Performance Pipeline

You are building a memory-safe ingestion/cleaning/analysis pipeline for the Fannie Mae
Single-Family Loan Performance dataset. **Read the full spec before coding:**
`../pipeline_plan/00_OVERVIEW.md`, `01_SCHEMA.md`, `02_PIPELINE_STAGES.md`, `03_CLAUDE_CODE_TASKS.md`.
Execute the tasks in `03_CLAUDE_CODE_TASKS.md` in order, one at a time.

## Hard rules (non-negotiable)

1. **Never load a full quarter into memory.** No `pd.read_csv` on raw files; no
   `.collect()` / `fetchall()` on a full quarter. Use DuckDB `COPY`/`read_csv`,
   PyArrow chunked readers, or Polars `scan_*` + `sink_parquet`. Bounded chunks only.
2. **The CSVs have no header and are positional, pipe-delimited, 113 fields.** Assert
   `field_count == 113` per file. All names are confirmed against the official glossary in
   `01_SCHEMA.md §1` (positions 110–112 = the Classic FICO trio, populated Dec 2025+).
3. **Inventory before anything.** Stage 1 must run and be reviewed first. The data is the
   **complete 2000–2025 vintage set** (~100 files, one per acquisition vintage, no loan-level
   overlap — see `01_SCHEMA.md §0a`). Stage 1 confirms the vintage sequence is contiguous, that
   all files share one release cut-off, and quarantines any truncated/corrupt file.
4. **Storage is the external SSD `SSD Felipe`, rooted at `/Volumes/SSD Felipe/dissertation/`.**
   Standard immutable-raw layout: `raw/` (originals — **never write here**) → `interim/`
   (Stages 2–3) → `processed/` (Stages 4–5), plus `models/`, `outputs/` (manifests, QA reports,
   figures) and `logs/` (run.log). All paths derive from a single `ROOT` in `config.py` — no
   scattered absolute paths. Only the disposable DuckDB catalog (`lake.duckdb`) lives on the
   internal disk under `reports/`. Call `require_drive()` at the start of every stage and fail
   fast if the drive isn't mounted. Write Parquet to the same SSD as the raw (no cross-bus
   copies). Still use zstd + dictionary encoding + `KEEP_COLS` projection and convert one
   quarter at a time. `interim/` and `processed/` are fully rebuildable from `raw/`. `--delete-raw`
   is optional (off by default), gated on a passing Stage 6 reconciliation if ever used.
5. **Idempotent & per-quarter.** Re-running skips completed quarters; everything resumable.
6. **Be cost-aware.** Most stages are mechanical (conversion, casts, reconciliation) and don't
   need a frontier model. If the current task is routine, proactively tell me I can switch to a
   cheaper/faster model with `/model`. Reserve the most capable model for the schema
   reconciliation (Task 1), the seven-state target/window logic (Task 5), and real debugging.
   Keep context lean so we don't burn through rate limits — suggest `/clear` between stages and
   re-read the relevant spec section instead of carrying the whole conversation.

## Stack
DuckDB (SQL, out-of-core) + Polars (lazy/streaming DataFrames) + Parquet (zstd). pandas
**only** on already-filtered, RAM-sized slices. `SET memory_limit='6GB'` on every DuckDB
connection.

## Target
Seven-state monthly transition (`current, dpd_30, dpd_60, dpd_90plus, foreclosure, REO,
prepaid`) per Sirignano et al., derived from `Current Loan Delinquency Status` +
`Zero Balance Code` — see `01_SCHEMA.md §4`. Produce `state`, `state_next`, `censored`.

## Training-prep columns & rules (`01_SCHEMA.md §6`)
Also derive: `period_ym`/`orig_ym` (int `YYYYMM`), `shard = hash(loan_id) % N_SHARDS`
(pinned `SHARD_SEED`; loan-keyed, vintage-stable; panel sorted by `(shard, loan, period)`),
and missingness indicators. **Never standardize the lake** — fit scalers on the train slice
only, per backtest window, at model time. **Rolling-backtest mask is strict**
(`period_ym < cutoff_ym`, because `state_next` is one month ahead). The right-censoring
cut-off is **derived** (max observed `period`), never hardcoded — added vintages extend it.

## Layout
Code in `dev/pipeline/`. On the SSD under `/Volumes/SSD Felipe/dissertation/`:
`interim/{perf,clean}/acq_quarter=YYYYQn/` and `processed/{panel,samples}/`; manifests/QA in
`outputs/`, logs in `logs/`. Raw CSVs at `raw/Performance_All/` (read-only — never modify).
Only `lake.duckdb` lives in the repo's internal-disk `reports/`.
