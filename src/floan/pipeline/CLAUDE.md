# CLAUDE.md — Fannie Mae Loan Performance Pipeline

You are building a memory-safe ingestion/cleaning/analysis pipeline for the Fannie Mae
Single-Family Loan Performance dataset. **Read the full spec before coding:**
`specs/pipeline/00_OVERVIEW.md`, `01_SCHEMA.md`, `02_PIPELINE_STAGES.md`, `03_CLAUDE_CODE_TASKS.md`.
Execute the tasks in `03_CLAUDE_CODE_TASKS.md` in order, one at a time.

## Part 1: Project-specific key instructions (crucial)

### Hard rules (non-negotiable)

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

### Stack
DuckDB (SQL, out-of-core) + Polars (lazy/streaming DataFrames) + Parquet (zstd). pandas
**only** on already-filtered, RAM-sized slices. `SET memory_limit='6GB'` on every DuckDB
connection.

### Target
Seven-state monthly transition (`current, dpd_30, dpd_60, dpd_90plus, foreclosure, REO,
prepaid`) per Sirignano et al., derived from `Current Loan Delinquency Status` +
`Zero Balance Code` — see `01_SCHEMA.md §4`. Produce `state`, `state_next`, `censored`.

### Training-prep columns & rules (`01_SCHEMA.md §6`)
Also derive: `period_ym`/`orig_ym` (int `YYYYMM`), `shard = hash(loan_id) % N_SHARDS`
(pinned `SHARD_SEED`; loan-keyed, vintage-stable; panel sorted by `(shard, loan, period)`),
and missingness indicators. **Never standardize the lake** — fit scalers on the train slice
only, per backtest window, at model time. **Rolling-backtest mask is strict**
(`period_ym < cutoff_ym`, because `state_next` is one month ahead). The right-censoring
cut-off is **derived** (max observed `period`), never hardcoded — added vintages extend it.

### Layout
Code in `src/floan/pipeline/`. On the SSD under `/Volumes/SSD Felipe/dissertation/`:
`interim/{perf,clean}/acq_quarter=YYYYQn/` and `processed/{panel,samples}/`; manifests/QA in
`outputs/`, logs in `logs/`. Raw CSVs at `raw/Performance_All/` (read-only — never modify).
Only `lake.duckdb` lives in the repo's internal-disk `reports/`.


## Part 2: Style Guidelines (non-negotiable too)

Following Andrej Karpathy's 4 key behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

**Think before coding** — state assumptions; if multiple interpretations exist, present them, don't pick silently.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

**Simplicity first** — minimum code that solves the problem; nothing speculative.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

**Surgical changes** — touch only what the task requires; match existing style; clean up only your own orphans.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

**Goal-driven execution** — each task's **Accept** criteria in `03_CLAUDE_CODE_TASKS.md` are the success criteria; verify before advancing.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
