# CLAUDE.md — Dissertation: Asset Loans Default Risk (project root)

Oxford MCF dissertation modelling US mortgage default risk on the Fannie Mae
Single-Family Loan Performance dataset (~800 GB, ~100 vintage CSVs, 2000–2025),
building a Sirignano-style seven-state monthly transition model
(`current → 30/60/90+ DPD → foreclosure → REO → prepaid`).

## Repo map

- `src/floan/pipeline/` — the data pipeline code (COMPLETE). **Its `CLAUDE.md` is binding for
  any work there** — read it (and the specs below) before touching pipeline code.
- `specs/pipeline/` — the pipeline specification (executed): `00_OVERVIEW.md` →
  `01_SCHEMA.md` → `02_PIPELINE_STAGES.md` → `03_CLAUDE_CODE_TASKS.md`;
  `HOWTO_RUN.md` is the operator guide.
- `specs/model/` — **the analysis & modelling specification (CURRENT WORK):**
  `00_OVERVIEW.md` → `01_EDA.md` → `02_LOAN_LEVEL.md` → `03_POOL_LEVEL.md` →
  `04_TASKS.md` (sequenced tasks M1–M27 with acceptance criteria);
  `05_MACRO_DATA.md` (standalone macro-data spec, tasks MD1–MD4);
  `06_GBT_BASELINE.md` (GBT learner, tasks M16–M19);
  `ECONOMIC_ENGINE.md` + `ADR-001-economic-engine-seam.md` — **the post-supervision priority:**
  the model-/horizon-/calibration-agnostic pricing engine (tasks M20–M27; rationale and
  sequencing in `writeup/memos/post_supervision_roadmap.md`).
  **Binding for any work in `src/floan/analysis/` and `src/floan/model/`.**
- `src/floan/analysis/` — Phase 1 EDA scripts (per `specs/model/01_EDA.md`).
- `src/floan/model/` — Phases 2–3 modelling code (per `specs/model/02–03`).
- `tests/` — unit tests (`test_schema.py`, `test_features.py`, `test_pool.py`, …); run with `pytest`.
- `scripts/` — shell helpers: `backup_ssd.sh` (mirrors the small irreplaceable SSD
  dirs to `ssd_mirror/`, gitignored; run after every milestone gate and GPU session),
  plus `run_backtest.sh` / `run_depth_check.sh`.
- `docs/` — source material: dataset glossary/tutorial PDFs, the Sirignano et al.
  paper, MCF dissertation guidelines, supervisor meeting notes.
- `writeup/` — the dissertation text (`latex/`) and interim result memos (`memos/`).
- `reports/` — disposable DuckDB catalog (`lake.duckdb`) only.
- `pyproject.toml` — the repo is an installable package (`pip install -e .`). Code is
  imported as `floan.*` and run as a module, e.g. `python -m floan.pipeline.run`,
  `python -m floan.model.backtest` (never run the files by path — that breaks imports).
- Data lives on the external SSD `SSD Felipe` at `/Volumes/SSD Felipe/dissertation/`
  (`raw/ → interim/ → processed/`, plus `models/ outputs/ logs/`), not in this repo.

## Critical invariants (apply everywhere, full detail in src/floan/pipeline/CLAUDE.md)

1. **Never load a full quarter into memory** — stream via DuckDB/Polars/PyArrow only.
2. **`raw/` on the SSD is immutable** — no code ever writes there.
3. All paths derive from the single `ROOT` in `src/floan/pipeline/config.py`; every stage
   calls `require_drive()` and fails fast if the SSD isn't mounted.
4. Stages are idempotent and per-quarter; execute `03_CLAUDE_CODE_TASKS.md` one task
   at a time and verify each task's **Accept** criteria before advancing.

## Working style

Follow the Karpathy guidelines (condensed in `src/floan/pipeline/CLAUDE.md`): state
assumptions before coding, simplest solution that works, surgical diffs only,
verify against explicit success criteria.
