# Fannie Mae Loan Performance — Data Pipeline Master Plan

> **Purpose of this folder.** These four documents are a specification you feed to Codex so it can build a memory-safe ingestion → cleaning → analysis pipeline for the Fannie Mae Single-Family Loan Performance dataset. Read order: this file → `01_SCHEMA.md` → `02_PIPELINE_STAGES.md` → `03_CODEX_TASKS.md`. The repository-root `AGENTS.md` and the pipeline-specific `src/floan/pipeline/AGENTS.md` pin the hard rules. `CODEX_WORKFLOW.md` is the operator's guide — how to launch Codex against this plan, plus model-selection and rate-limit tips.

---

## 1. The problem in one paragraph

You have ~800 GB of Fannie Mae loan-performance data spread across ~100 quarterly CSV files — **one file per acquisition vintage**, Q1 2000 through ~2025 (largest files run to tens of GB). The data is a **loan-month panel**: one row per loan per reporting month, pipe-delimited, no header, 113 fields. No single file fits in RAM, and pandas `read_csv` on a 25 GB file will crash a laptop. The goal is a pipeline that (a) never loads more than a bounded chunk into memory, (b) converts the raw CSVs into a compressed, columnar, queryable store, and (c) lets you draw clean samples and build the Sirignano-style transition panel without ever materialising the full dataset.

> **Storage location (updated).** All data lives on an external SSD named **`SSD Felipe`** (mounted at **`/Volumes/SSD Felipe/`**) under a single `dissertation/` root, following the standard immutable-raw data-science layout: `raw/` (untouched originals) → `interim/` (cleaning outputs) → `processed/` (training-ready), plus `models/`, `outputs/`, `logs/`. Everything derives from one `ROOT` in `config.py` — no scattered absolute paths. **`raw/` is never written to**; interim and processed are fully rebuildable from it. The drive must be connected before any stage runs. With the SSD's room, the earlier "tight disk budget" constraint no longer binds (see §5).

## 2. Ground truth about the data (measured, not assumed)

| Property | Value |
|---|---|
| Format | Pipe-delimited (`|`), **no header row** |
| Fields per row | **113** (leading field is `Reference Pool ID`, usually blank) |
| Grain | One row per **loan × monthly reporting period** |
| Total size | ~800 GB raw CSV (full vintage set) |
| Files | ~100 quarterly CSVs, `YYYYQn.csv`, **one per acquisition vintage** (Q1 2000 → ~2025) |
| Date encoding | `MMYYYY` (e.g. `112013` = Nov 2013) |

**Coverage: the complete vintage set (2000–2025).** Each file is one **acquisition vintage** — the loans Fannie Mae acquired that quarter, with their full monthly performance history — so the ~100 files together form the whole sample with **no loan-level overlap** between files (see the "Data model" note in `01_SCHEMA.md §0a`). This supersedes an earlier partial download (2017Q2–2022Q2 with gaps and several truncated files); those gaps are now filled.

- **Scope reminder:** this public dataset is a *subset* of Fannie Mae's book — 30-year fixed-rate, fully amortizing, fully documented, conventional, LTV ≤ 97%. It excludes ARMs, balloons, interest-only, HARP/Refi Plus, reduced-doc, and government-insured loans (per the official tutorial). State this in the dissertation's data section.
- **Single-release check:** confirm all ~100 files came from the **same quarterly release**. Mixing releases gives inconsistent right-censoring cut-offs across vintages (and duplicate vintages if the same quarter was pulled twice). Stage 1 must record each file's `max(Monthly Reporting Period)` and flag any vintage whose cut-off diverges from the modal one.

> **Implication:** Stage 1 of the pipeline is a non-negotiable **inventory + integrity audit** — even with a full download, it confirms field counts, a uniform release cut-off, and catches any truncated/partial files before modelling.

## 3. Modelling target (from Sirignano, Sadhwani & Giesecke)

The seminal paper models the **monthly conditional transition probability matrix** over **seven mortgage states**:

`current → 30 DPD → 60 DPD → 90+ DPD → foreclosure → REO → prepaid`

It is a multinomial classification of *next month's state* given the current state plus loan-level and macro covariates. Concretely, the cleaning pipeline must derive, per loan-month, a categorical `state_t` (from `Current Loan Delinquency Status` + `Zero Balance Code`) and a `state_next` (the same loan's state in month *t+1*). That `(state_t, state_next)` pair is the supervised label. Exact mapping is specified in `01_SCHEMA.md §4`.

(Macro covariates — unemployment, HPI, rates by ZIP/MSA — are **out of scope for this pass** per project decision; the pipeline is structured so a macro join can be bolted on later without reprocessing.)

## 4. Chosen architecture: DuckDB + Polars + Parquet

The canonical store is **partitioned Parquet with zstd compression**, queried out-of-core by **DuckDB** (SQL) and **Polars** (DataFrame). pandas is used *only* on already-filtered slices that comfortably fit in RAM.

```
raw/Performance_All/*.csv   (~800 GB, ~100 vintage files, pipe-delim, no header — IMMUTABLE)
        │  Stage 2: stream in row-group chunks, never fully in RAM
        ▼
interim/perf/acq_quarter=YYYYQn/part-*.parquet   (zstd, faithful string copy)
        │  Stage 3: clean + standardise (dates, categoricals, sentinels)
        ▼
interim/clean/...
        │  Stage 4: build loan-month transition panel (state_t, state_next)
        ▼
processed/panel/...
        │  Stage 5: DuckDB views + stratified sampling for model dev
        ▼
processed/samples/  →  models/   (fit in RAM; hand off to model training)
```

**Why this stack for your constraints:**

- **Parquet + zstd** typically compresses this dataset 4–8×, so the ~800 GB raw lands in roughly **120–200 GB** for the full-column `interim/perf` lake, and far less for the projected `interim/clean` and `processed` lakes (keep-list columns only). Columnar layout means a query that needs 6 of 113 columns reads ~5% of the bytes.
- **DuckDB** runs SQL directly over Parquet files with predicate/projection pushdown and spills to disk when memory is tight. Its SQL dialect is close to the T-SQL you already use; you can `SELECT … WHERE acq_quarter = '2020Q4'` against the lake without loading it. Set `SET memory_limit='6GB'` to cap RAM.
- **Polars** `scan_*` is lazy and streaming, so transforms execute in bounded memory and parallelise across cores.
- **PyArrow** provides the chunked CSV reader used in Stage 2 so even a 25 GB file is read block-by-block.

**The single hard rule that makes this work:** *never read a full quarter into memory.* Every stage operates on bounded chunks (row groups / `batch_size`) or pushes the work down into DuckDB. This rule is restated in `AGENTS.md`.

## 5. Storage & disk strategy (external SSD)

Everything lives under `/Volumes/SSD Felipe/dissertation/`, so there is plenty of room and `--delete-raw` is now **optional housekeeping, not a necessity**. Keep these principles:

1. **`raw/` is immutable** — no stage writes to it. Everything in `interim/` and `processed/` is fully rebuildable from `raw/`, so it can be deleted and regenerated freely (and need not be backed up).
2. Convert **one quarter at a time** — this is about bounded memory and resumability, not disk space.
3. Keep `raw/` and the Parquet outputs on the **same drive** (the SSD) so conversion never streams 25 GB across the USB bus twice. Manifests/QA reports go to `outputs/`, run logs to `logs/`. The only thing on the Mac's internal disk is the disposable DuckDB catalog (`lake.duckdb`) under the repo `reports/`, kept there for query speed.
4. In the clean/panel stages, still **project only the columns the model needs** (`01_SCHEMA.md §5`) — smaller lakes mean faster out-of-core scans over USB, where I/O is the bottleneck.
5. `--delete-raw` (gated on a passing Stage 6 reconciliation) remains available if you later want to reclaim space, but is off by default.
6. **External-drive I/O note:** reads come over USB/Thunderbolt, slower than internal NVMe. Expect Stage 2 to be I/O-bound; this is fine for an overnight one-time conversion. The compressed Parquet lake is far cheaper to re-scan afterwards.

## 6. Repository layout the pipeline should create

```
<repo root>/
├── pyproject.toml            # installable package — `pip install -e .`
├── specs/pipeline/           # these spec docs (input to Codex)
├── src/floan/pipeline/       # the code Codex writes
│   ├── AGENTS.md             # the binding conventions file
│   ├── config.py             # single ROOT + named subpaths, memory/chunk knobs, require_drive()
│   ├── schema.py             # the 113-col layout, dtypes, code maps (from 01_SCHEMA.md)
│   ├── s1_inventory.py       # Stage 1: manifest + integrity audit
│   ├── s2_to_parquet.py      # Stage 2: streaming CSV → Parquet
│   ├── s3_clean.py           # Stage 3: standardise dtypes/dates/categoricals
│   ├── s4_panel.py           # Stage 4: transition panel (state_t, state_next)
│   ├── s5_sample.py          # Stage 5: DuckDB views + stratified sampling
│   ├── s6_qa.py              # Stage 6: reconciliation + null/target audits
│   └── run.py                # CLI orchestrator: `python -m floan.pipeline.run`
├── tests/                    # test_schema.py
└── reports/                  # lake.duckdb only — disposable catalog (internal disk)

/Volumes/SSD Felipe/dissertation/   # EXTERNAL DRIVE — this is ROOT in config.py
├── raw/Performance_All/*.csv        # original CSVs — IMMUTABLE, never written
├── interim/perf/   interim/clean/   # Stages 2 & 3 (zstd Parquet)
├── processed/panel/  processed/samples/  # Stages 4 & 5 (training-ready)
├── models/                          # trained models / checkpoints
├── outputs/                         # manifests, QA reports, figures
└── logs/                            # run.log
```

> The only path to confirm in `config.py` is `RAW_DIR = ROOT / "raw" / "Performance_All"` (adjust the `Performance_All` folder name if you renamed it). Everything else derives from `ROOT`.

## 7. How to use these docs with Codex

0. **Connect the `SSD Felipe` drive first** and confirm it is mounted at `/Volumes/SSD Felipe/`. Every stage should fail fast with a clear message if the drive is absent.
1. Open Codex in the repository root so it picks up the root `AGENTS.md`; the kickoff prompt below explicitly loads `src/floan/pipeline/AGENTS.md`.
2. Point it at this plan: *"Read `specs/pipeline/00_OVERVIEW.md` through `03_CODEX_TASKS.md`, then execute the tasks in order."*
3. Work **one task at a time** from `03_CODEX_TASKS.md`; each task has explicit acceptance criteria. Do not let it skip Stage 1.
4. Stages are idempotent and per-quarter, so a crash mid-run is recoverable — re-running skips quarters already converted.

---

### Recap

The dataset is a ~800 GB pipe-delimited, header-less loan-month panel with 113 columns, spanning the **complete 2000–2025 acquisition-vintage set** (~100 files, one per vintage, no loan-level overlap). Stage 1 still audits field counts, a uniform release cut-off, and any truncated files up front. It lives on the external SSD `SSD Felipe` (`/Volumes/SSD Felipe/`), which must be connected before any run; the Parquet lake lives on that same drive. The plan converts it — one quarter at a time, never fully in RAM — into a zstd-compressed partitioned Parquet lake (~120–200 GB full, less when column-projected) queried out-of-core by DuckDB and Polars, then derives the Sirignano seven-state monthly transition panel as the modelling target. `01_SCHEMA.md` pins the column layout and target mapping; `02_PIPELINE_STAGES.md` specifies each of the six stages with code patterns and memory rules; `03_CODEX_TASKS.md` is the sequenced, acceptance-tested build order; `AGENTS.md` enforces the no-full-load rule.
