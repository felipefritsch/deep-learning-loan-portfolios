# CLAUDE.md — Dissertation: Asset Loans Default Risk (project root)

Oxford MCF dissertation modelling US mortgage default risk on the Fannie Mae
Single-Family Loan Performance dataset (~800 GB, ~100 vintage CSVs, 2000–2025),
building a Sirignano-style seven-state monthly transition model
(`current → 30/60/90+ DPD → foreclosure → REO → prepaid`).

## Repo map

- `dev/pipeline/` — the data pipeline code (COMPLETE). **Its `CLAUDE.md` is binding for
  any work there** — read it (and the specs below) before touching pipeline code.
- `dev/pipeline_plan/` — the pipeline specification (executed): `00_OVERVIEW.md` →
  `01_SCHEMA.md` → `02_PIPELINE_STAGES.md` → `03_CLAUDE_CODE_TASKS.md`;
  `HOWTO_RUN.md` is the operator guide.
- `dev/model_plan/` — **the analysis & modelling specification (CURRENT WORK):**
  `00_OVERVIEW.md` → `01_EDA.md` → `02_LOAN_LEVEL.md` → `03_POOL_LEVEL.md` →
  `04_TASKS.md` (sequenced tasks M1–M15 with acceptance criteria);
  `05_MACRO_DATA.md` is a standalone macro-data spec (tasks MD1–MD4).
  **Binding for any work in `dev/analysis/` and `dev/model/`.**
- `dev/analysis/` — Phase 1 EDA scripts (per `model_plan/01_EDA.md`).
- `dev/model/` — Phases 2–3 modelling code (per `model_plan/02–03`).
- `dev/tools/` — `backup_ssd.sh`: mirrors the small irreplaceable SSD dirs to
  `ssd_mirror/` (gitignored). Run after every milestone gate and GPU session.
- `docs/` — source material: dataset glossary/tutorial PDFs, the Sirignano et al.
  paper, MCF dissertation guidelines, supervisor meeting notes.
- `writeup/` — the dissertation text (`latex/`) and interim result memos (`memos/`).
- `reports/` — disposable DuckDB catalog (`lake.duckdb`) only.
- Data lives on the external SSD `SSD Felipe` at `/Volumes/SSD Felipe/dissertation/`
  (`raw/ → interim/ → processed/`, plus `models/ outputs/ logs/`), not in this repo.

## Critical invariants (apply everywhere, full detail in dev/pipeline/CLAUDE.md)

1. **Never load a full quarter into memory** — stream via DuckDB/Polars/PyArrow only.
2. **`raw/` on the SSD is immutable** — no code ever writes there.
3. All paths derive from the single `ROOT` in `dev/pipeline/config.py`; every stage
   calls `require_drive()` and fails fast if the SSD isn't mounted.
4. Stages are idempotent and per-quarter; execute `03_CLAUDE_CODE_TASKS.md` one task
   at a time and verify each task's **Accept** criteria before advancing.

## Working style

Follow the Karpathy guidelines (condensed in `dev/pipeline/CLAUDE.md`): state
assumptions before coding, simplest solution that works, surgical diffs only,
verify against explicit success criteria.
