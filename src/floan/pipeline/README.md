# Fannie Mae Loan-Performance Pipeline

Memory-safe ingestion → cleaning → modelling-target pipeline for the Fannie Mae
Single-Family Loan Performance dataset. Converts ~865 GB of raw pipe-delimited
CSVs into a compressed, columnar, queryable lake and derives the Sirignano
seven-state monthly-transition target — **never loading more than a bounded
chunk into memory** (hard rule; see `CLAUDE.md`).

Specs live in `../pipeline_plan/` (`00_OVERVIEW` → `01_SCHEMA` → `02_PIPELINE_STAGES`
→ `03_CLAUDE_CODE_TASKS`). This README is the operator guide.

## Current state (certified)

| | |
|---|---|
| Vintages | **104**, contiguous `2000Q1 → 2025Q4`, one release cut-off (`2025-12`) |
| Loan-months | **3,312,456,883** (53.7 M loans) |
| Lake sizes | raw ~865 GB → perf ~22 GB → clean ~19 GB → panel ~21 GB |
| QA | row reconciliation, `(loan,period)` uniqueness, uniform schema, shard balance — **all PASS** (`outputs/qa_report.md`) |

## Layout

Everything derives from a single `ROOT` in `config.py` (external SSD
`/Volumes/SSD Felipe/dissertation/`). `raw/` is **immutable** — no stage writes it.

```
raw/Performance_All/*.csv                       # original CSVs (read-only)
interim/perf/acq_quarter=YYYYQn/part.parquet    # Stage 2: faithful 113-col string copy (zstd)
interim/clean/acq_quarter=YYYYQn/part.parquet   # Stage 3: 41 typed/projected cols + calendar/missingness
processed/panel/acq_quarter=YYYYQn/part.parquet # Stage 4: + state/state_next/censored/shard (45 cols)
processed/samples/  models/  outputs/  logs/    # samples, scalers, QA/manifests, run.log
reports/lake.duckdb   reports/samples/          # internal disk: catalog + cached working sets
```

## Stages

| # | Module | What it does |
|---|---|---|
| 1 | `s1_inventory.py` | catalogue + integrity audit (field count, NUL/corruption, release-cut-off & relative-volume cross-checks); content-aware resume |
| 2 | `s2_to_parquet.py` | streaming DuckDB `COPY` CSV→zstd Parquet, faithful 113-col string copy, Hive-partitioned |
| 3 | `s3_clean.py` | PyArrow-batched Polars transform → typed/projected clean lake (+ `period_ym`/`orig_ym`, coalesced `fico_orig`, missingness flags) |
| 4 | `s4_panel.py` | DuckDB external-sort + lookahead → `state`/`state_next`/`censored`/`shard`, sorted by `(shard, loan, period)` |
| 5 | `s5_sample.py` | DuckDB views, stratified `balanced_sample`, `iter_shards`, `training_mask`, train-only `fit_scaler`/`apply_scaler` |
| 6 | `s6_qa.py` | reconciliation, null/missingness audit, target distribution, key/shard/calendar sanity, feature scale stats |

`schema.py` is the single source of truth (113-col layout, dtypes, code maps,
`derive_state`, `KEEP_COLS`, `FEATURE_SPEC`). `config.py` holds `ROOT`, the knobs
(`DUCKDB_MEMORY_LIMIT=6GB`, `ZSTD_LEVEL=9`, `N_SHARDS=256`, `SHARD_SEED`), and
`require_drive()`.

## Run sequence

```bash
pip install -r requirements.txt          # duckdb, polars, pyarrow, tqdm
python run.py inventory                   # Stage 1 — review outputs/inventory_summary.md FIRST
python run.py convert  --all              # Stage 2  (overnight: wrap in `caffeinate -ims`)
python run.py clean    --all              # Stage 3
python run.py panel    --all              # Stage 4
python run.py qa                          # Stage 6 — certify before any --delete-raw
python run.py sample   --cutoff-ym 201501 # Stage 5 — balanced sample + scaler for a backtest window
```

Single vintage / end-to-end: `python run.py pipeline --quarter 2020Q4`.
Tests: `python test_schema.py`.

## Operating notes

- **Memory.** Every stage stays under the 6 GB DuckDB cap. Stage 3 uses an
  explicit PyArrow batch loop and Stage 4 uses a DuckDB external sort + Python
  lookahead — both because Polars `sink_parquet`/DuckDB window functions
  materialised 100M+ row quarters (~86 GB) and were killed. **Plain `ORDER BY`
  spills reliably; window functions do not.**
- **Resumability.** Per-quarter and idempotent: a completed quarter (matching row
  count) is skipped. Writes go to a `.tmp` promoted by atomic rename only after a
  round-trip row-count reconciliation, so an interrupted run (drive unplug, sleep)
  never leaves a half-written partition that looks done. Re-run the same command
  to continue. Use `--force` to rebuild.
- **Vintage refresh.** Stage 1 fingerprints each file (size+mtime); a refreshed
  vintage auto-reconverts. (Edge case: a same-size, same-mtime replacement is
  missed — drop its manifest row to force a rescan.)
- **Sleep / external drive.** Wrap long runs in `caffeinate -ims` (idle+disk+system
  sleep) and keep the SSD seated; an unplug aborts the active quarter (resumable).
- **No leakage / no lake standardisation.** Scalers are fit on the train slice
  only, per backtest window, at model time (`models/scaler_<cutoff>.json`). The
  rolling-backtest mask is strict (`period_ym < cutoff_ym`, since `state_next` is
  one month ahead). Right-censoring cut-off is derived (max `period`), never
  hardcoded.

## Known data-quality findings (from Stage 6)

- 11 rows / 9 loans with `orig_ym > period_ym` (source dirt) — excluded by the
  training filter `state IS NOT NULL AND state_next IS NOT NULL AND orig_ym <= period_ym`.
- 2,555 rows (0.00008%) with null `state` (`XX`/blank) — excluded by the same filter.
- `Unscheduled Principal Current` is 100% null in SF data (a CAS/CIRT-only field);
  kept in the lake for faithfulness but excluded from `FEATURE_SPEC`.
- High-null-by-design: `fico_co` (~52%, no co-borrower), `Mortgage Insurance %`
  (~81%, no MI) — captured by `*_missing` indicators.
- `Original Combined LTV (CLTV)` is reported only in later vintages (schema drift).
