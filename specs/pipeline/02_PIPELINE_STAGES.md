# 02 — Pipeline Stages (specification + code patterns)

> Six stages. Each is **per-quarter** and **idempotent**: re-running skips work already done. Every stage obeys the one hard rule — *never load a full quarter into memory*. Code patterns below are reference implementations, not the final code; Codex adapts them.

Environment (install once):
```bash
pip install duckdb polars pyarrow tqdm  # pandas only for small slices
```

`config.py` should centralise the paths and knobs. The raw data and Parquet lake live on the external SSD `SSD Felipe`:

Paths derive from a single `ROOT` (no hardcoded absolute paths scattered through the code), following the standard immutable-raw data-science layout:

```python
from pathlib import Path
ROOT      = Path("/Volumes/SSD Felipe/dissertation")   # external SSD mount (macOS)
RAW       = ROOT / "raw"          # immutable original CSVs — NEVER written to
INTERIM   = ROOT / "interim"      # cleaning outputs (rebuildable from raw)
PROCESSED = ROOT / "processed"    # training-ready data
MODELS    = ROOT / "models"
OUTPUTS   = ROOT / "outputs"      # manifests, QA reports, figures
LOGS      = ROOT / "logs"         # run.log

RAW_DIR   = RAW / "Performance_All"          # original quarterly CSVs
PERF_DIR  = INTERIM / "perf"                 # Stage 2: faithful string-typed Parquet
CLEAN_DIR = INTERIM / "clean"                # Stage 3: typed + standardised Parquet
PANEL_DIR = PROCESSED / "panel"              # Stage 4: transition panel
SAMPLE_DIR= PROCESSED / "samples"            # Stage 5: stratified samples
DUCKDB_PATH = Path(__file__).resolve().parents[2] / "reports" / "lake.duckdb"  # internal disk (fast)
DUCKDB_MEMORY_LIMIT = "6GB"; CHUNK_ROWS = 1_000_000; ZSTD_LEVEL = 9
N_SHARDS = 256   # minibatch shards: shard = hash(loan_id) % N_SHARDS (01_SCHEMA.md §6.2)

def require_drive():
    """Fail fast if the external SSD is not mounted."""
    if not ROOT.exists():
        raise SystemExit(f"External drive not mounted at {ROOT} — connect 'SSD Felipe' and retry.")
```

Every stage entry point calls `require_drive()` first. **`raw/` is immutable — no stage ever writes to it.** Interim and processed data are fully rebuildable from `raw/`, so they can be deleted and regenerated freely. The only file kept on the Mac's internal disk is the DuckDB catalog (`lake.duckdb`) for query speed — it's a disposable index over the Parquet lake, not source data.

---

## Stage 1 — Inventory & integrity audit  (`s1_inventory.py`)

**Why first:** even with a full ~100-file download you must verify it before trusting it — confirm every vintage is present, that all files share one release cut-off, and that none are truncated/corrupt. Catalogue before you model.

**Do, per file (streaming — never load whole file):**
1. Record byte size, mtime, filename → derived `acq_quarter`.
2. Count rows by streaming (`wc -l` equivalent or chunked line count).
3. Count **distinct `Loan Identifier`** (col 1) via a streaming DuckDB query straight against the CSV.
4. Verify field count on first N rows: assert every row splits into **113** fields; log any ragged rows.
5. Min/max `Monthly Reporting Period` to see the actual time span the file covers.
6. Flag **suspected truncation**: `distinct_loans < 1000` OR `bytes < 50 MB` while neighbouring vintages are GBs.
7. **Release-consistency check:** collect every file's `max(Monthly Reporting Period)`; flag any vintage whose cut-off diverges from the modal one (a sign files came from different releases).
8. **Completeness check:** confirm the vintage sequence is contiguous from `2000Q1` to the latest expected quarter — list any missing vintage.

```python
import duckdb, pathlib, csv
con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
for f in sorted(pathlib.Path(RAW_DIR).glob("*.csv")):
    q = f"""
      SELECT count(*) rows,
             count(DISTINCT column01) loans,
             min(column02) min_period, max(column02) max_period
      FROM read_csv('{f}', delim='|', header=false, all_varchar=true,
                    names=[{','.join(f"'column{i:02d}'" for i in range(113))}])
    """
    stats = con.execute(q).fetchone()
    # field-count check on first rows
    with open(f) as fh:
        bad = [i for i,l in zip(range(1000), fh) if l.count('|') != 112]
    # append row to manifest...
```

**Output:** `outputs/inventory_manifest.csv` and a human-readable `outputs/inventory_summary.md` with a vintage coverage table, the release cut-off per file, and an explicit **"quarantined files"** list (any truncated/corrupt ones). Quarantined files are excluded from default downstream runs unless `--include-quarantined`.

**Acceptance:** manifest lists every vintage file found; the vintage sequence 2000Q1→latest is contiguous (missing vintages listed); all files share one release cut-off (divergent ones flagged); any file failing the field-count, size, or loan-count check is quarantined. **Review the summary before continuing.**

---

## Stage 2 — Streaming CSV → Parquet  (`s2_to_parquet.py`)

**Goal:** convert each raw CSV to compressed, dictionary-encoded Parquet, reading in bounded chunks. Two equally valid engines — pick one and stick to it:

**Option A — DuckDB (simplest, SQL pushdown, spills to disk):**
```python
con.execute(f"""
  COPY (
    SELECT * FROM read_csv('{raw_csv}', delim='|', header=false,
            all_varchar=true, names={COLUMN_NAMES})
  ) TO '{PERF_DIR}/acq_quarter={q}/part.parquet'
  (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000);
""")
```
DuckDB streams the CSV internally; one `COPY` handles a 25 GB file without loading it. Casting can be done here or deferred to Stage 3 (recommended: keep Stage 2 a faithful string-typed copy, cast in Stage 3, so Stage 2 never fails on dirty values).

**Option B — PyArrow chunked reader (explicit control):**
```python
import pyarrow as pa, pyarrow.csv as pacsv, pyarrow.parquet as pq
ropts = pacsv.ReadOptions(column_names=COLUMN_NAMES, block_size=128<<20)  # 128 MB blocks
popts = pacsv.ParseOptions(delimiter='|')
copts = pacsv.ConvertOptions(column_types={c: pa.string() for c in COLUMN_NAMES})
reader = pacsv.open_csv(raw_csv, read_options=ropts, parse_options=popts, convert_options=copts)
writer = None
for batch in reader:                      # bounded: one block at a time
    t = pa.Table.from_batches([batch])
    if writer is None:
        writer = pq.ParquetWriter(out_path, t.schema, compression='zstd', compression_level=9)
    writer.write_table(t)
writer.close()
```

**Partitioning:** write under `interim/perf/acq_quarter=YYYYQn/` (i.e. `PERF_DIR`). This is Hive-style partitioning DuckDB/Polars read natively, so you can later filter `WHERE acq_quarter='2020Q4'` and touch only that folder.

**Disk discipline:** the SSD has room, so `--delete-raw` is **optional** (off by default). If you ever want to reclaim space, it is gated on a passing Stage 6 reconciliation — never delete pre-reconciliation. Write the Parquet output to the **same SSD** as the raw so a 25 GB file is not copied across the USB bus.

**Acceptance:** every non-quarantined vintage has a `perf/acq_quarter=…/part.parquet`; total Parquet footprint ≈ 120–200 GB (full 113-col lake); round-trip row count matches the manifest.

---

## Stage 3 — Clean & standardise  (`s3_clean.py`)

**Goal:** cast string Parquet → typed, standardised Parquet projected to the keep-list (`01_SCHEMA.md §5`). Operates lazily/streamed with Polars `scan_parquet`.

```python
import polars as pl
lf = pl.scan_parquet(f"{PERF_DIR}/acq_quarter={q}/*.parquet")
lf = (lf
  .select(KEEP_COLS)                                  # projection = less memory
  .with_columns([
      parse_mmyyyy(pl.col("Monthly Reporting Period")).alias("period"),
      parse_mmyyyy(pl.col("Origination Date")).alias("orig_date"),
      pl.col("Original Interest Rate").cast(pl.Float32, strict=False),
      pl.col("Current Actual UPB").cast(pl.Float32, strict=False),
      # credit-score sentinel scrub
      pl.when(pl.col("Borrower Credit Score at Origination").cast(pl.Int32, strict=False)
                .is_between(300, 850))
        .then(pl.col("Borrower Credit Score at Origination").cast(pl.Int16, strict=False))
        .otherwise(None).alias("fico_orig"),
      # categorical maps via replace; dictionary dtype
      pl.col("Loan Purpose").replace(LOAN_PURPOSE).cast(pl.Categorical),
      # ...
  ]))
lf.sink_parquet(f"{CLEAN_DIR}/acq_quarter={q}/part.parquet",
                compression="zstd")        # sink_parquet streams; bounded RAM
```

Key transforms (all from `01_SCHEMA.md §3`): `MMYYYY`→date, float→f32, int→smallest nullable, credit-score/numeric sentinels→null, Y/N→bool, code maps→categorical dictionary. **`sink_parquet` (not `collect`)** keeps it streaming.

**Also add the derived calendar & missingness columns (`01_SCHEMA.md §6.1, §6.3`):**
- `period_ym`, `orig_ym` — integer `YYYYMM` from `Monthly Reporting Period` / `Origination Date` (keep the parsed `date` versions too). These are the rolling-backtest mask keys.
- `fico_orig` — coalesced origination FICO across the field migration (old `Borrower Credit Score at Origination` → new `Origination Classic FICO`); use `schema.fico_orig_expr()`. Without this, origination FICO goes null for the newest activity periods (`01_SCHEMA.md` FICO-migration note).
- Missingness indicators — `fico_orig_missing` (on the **coalesced** `fico_orig`), `fico_co_missing`, `dti_missing`, `cltv_missing`, `mi_pct_missing` (boolean, set `True` where the source was sentinel/blank, computed *before* the value is scrubbed to null).

```python
.with_columns([
    (pl.col("period").dt.year() * 100 + pl.col("period").dt.month()).cast(pl.Int32).alias("period_ym"),
    (pl.col("orig_date").dt.year() * 100 + pl.col("orig_date").dt.month()).cast(pl.Int32).alias("orig_ym"),
    pl.col("Borrower Credit Score at Origination").cast(pl.Int32, strict=False)
      .is_between(300, 850).not_().alias("fico_orig_missing"),
    # … dti_missing, cltv_missing, mi_pct_missing similarly
])
```

**Acceptance:** cleaned partition has only keep-list + §6 derived columns; no `MMYYYY` strings remain; `period_ym`/`orig_ym` are sane integer year-months; missingness indicators present and consistent with the scrubbed nulls; null rates for sentinel-scrubbed fields are sane (spot-check in Stage 6); row count unchanged vs perf lake.

---

## Stage 4 — Build the transition panel  (`s4_panel.py`)

**Goal:** derive `state`, `state_next`, `censored` per loan-month (target from `01_SCHEMA.md §4`). The only step needing a per-loan ordered window — do it **within a quarter partition** so memory stays bounded (all of one loan's months live in the same acquisition-quarter file, since the file is defined by origination quarter).

```python
lf = pl.scan_parquet(f"{CLEAN_DIR}/acq_quarter={q}/*.parquet")
lf = lf.with_columns(derive_state().alias("state"))      # §4 logic as a Polars expr
lf = lf.sort(["Loan Identifier", "period"]).with_columns([
        pl.col("state").shift(-1).over("Loan Identifier").alias("state_next"),
])
lf = lf.with_columns(
        pl.col("state_next").is_null()
          .and_(~pl.col("state").is_in(["prepaid","foreclosure","REO"]))
          .alias("censored"))
# §6.2 shard: deterministic, loan-keyed, fixed-seed hash → bucket in [0, N_SHARDS)
lf = lf.with_columns(
        (pl.col("Loan Identifier").hash(seed=SHARD_SEED) % N_SHARDS)
          .cast(pl.Int16).alias("shard"))
# Sort output by (shard, loan, period): clusters rows by shard for row-group
# skipping on `WHERE shard = k`, while keeping each loan's months contiguous.
lf = lf.sort(["shard", "Loan Identifier", "period"])
lf.sink_parquet(f"{PANEL_DIR}/acq_quarter={q}/part.parquet", compression="zstd")
```

Equivalent DuckDB (if you prefer SQL): `LEAD(state) OVER (PARTITION BY loan_id ORDER BY period)` and `hash(loan_id) % N_SHARDS AS shard`.

**Notes:**
- **`state_next` window:** absorbing states (`prepaid/foreclosure/REO`) should have `state_next` = themselves or null-terminal — decide one convention and document it in the module docstring. Loans whose last month is `current`/delinquent and equals the **data cut-off** are **right-censored** (`censored=True`) — valid for training but not "survived forever." The cut-off is the max `period` actually observed (currently `2025-12`); **derive it from the data, do not hardcode** — added vintages extend it.
- **`shard` seed:** pin `SHARD_SEED` (and the Polars version's hash) in `schema.py`; changing it reshuffles all shards. The hash is on the `Loan Identifier` *string*, so it is stable across vintage refreshes (`01_SCHEMA.md §6.2`).

**Acceptance:** `state` ∈ the 7-label set ∪ null; transition counts (a 7×7 matrix) printed for one quarter look plausible (most mass on `current→current`); `censored` flag present and keyed off the derived data cut-off; `shard` ∈ `[0, N_SHARDS)`, every month of a sampled loan shares one shard, and shard counts are roughly uniform; output sorted by `(shard, loan, period)`.

---

## Stage 5 — DuckDB views & stratified sampling  (`s5_sample.py`)

**Goal:** query the panel out-of-core and draw RAM-sized samples for model development — never load the whole panel.

```python
con = duckdb.connect(str(DUCKDB_PATH))   # catalog on fast internal disk
con.execute("SET memory_limit='6GB'")
con.execute(f"""
  CREATE OR REPLACE VIEW panel AS
  SELECT * FROM read_parquet('{PANEL_DIR}/acq_quarter=*/*.parquet', hive_partitioning=true);
""")
# example: stratified sample keeping all rare transitions, downsampling current→current
sample = con.execute("""
  SELECT * FROM panel
  USING SAMPLE 2% (bernoulli)
  WHERE state IS NOT NULL
""").pl()      # -> Polars df that fits in RAM
```

Provide helper functions: `sample_by_state(frac_per_state)`, `slice_quarter(q)`, `transition_matrix()`. Because rare events (default, REO) are <<1% of rows, expose **stratified** sampling that keeps all minority-class transitions and downsamples `current→current`.

**Training-data helpers (`01_SCHEMA.md §6`):**
- `iter_shards(shards=None, where=None)` — yield one shard at a time as a RAM-sized frame via `WHERE shard = k` (row-group skipping makes this cheap). The DL loader shuffles shard order per epoch, then shuffles rows within each shard → streaming minibatch randomness.
- `training_mask(cutoff_ym)` — returns the leakage-safe filter `period_ym < cutoff_ym` (strict, because `state_next` is one month ahead — `01_SCHEMA.md §6.1`). Use for rolling backtests: `panel WHERE period_ym < cutoff_ym AND state IS NOT NULL`.
- `fit_scaler(train_frame, feature_spec)` — compute mean/std (robust median/IQR for skewed/`log`-flagged columns) over the **training slice only**, persist to `models/scaler_<cutoff_ym>.json`; `apply_scaler()` for train/val/inference. Never standardize the lake itself.

**Optional — vintage subsampling (OFF by default).** A dormant efficiency lever for when training throughput/IO bites: thin the bulky-but-low-signal modern vintages while keeping the crisis cohorts whole. It is a *compute* tool, not a class-balance tool (state stratification above handles imbalance), and must be used carefully:

```python
# Default is a no-op: every vintage at fraction 1.0. Turn on only when needed.
VINTAGE_SAMPLING = {
    ("2000Q1", "2012Q4"): 1.00,   # keep crisis-era cohorts in full (the signal)
    ("2013Q1", "2019Q4"): 1.00,
    ("2020Q1", "2025Q4"): 1.00,   # e.g. lower to 0.20 to thin the refi-boom volume
}
```

`sample_vintages(frame_or_view, spec=VINTAGE_SAMPLING)` applies each range's keep-fraction **at the loan level** (`WHERE hash(Loan Identifier) % 1000 < frac*1000` — sample whole loans, never individual months, so no loan's history is fractured). Rules baked into the helper/docstring: apply **only after** the `training_mask` temporal split (train slice only — never val/inference); **never thin the crisis cohorts**; and any base rate reported for the dissertation must be computed on the **unsampled** panel (or inverse-probability reweighted), since the switch deliberately distorts the vintage mix. Default `1.0` everywhere changes nothing.

**Acceptance:** can produce a balanced training sample (e.g. 5M rows) in <1 min without exceeding the memory limit; sample preserves all foreclosure/REO transitions; `iter_shards` reads a single shard without scanning the whole panel; `training_mask(cutoff)` excludes any example whose labelled month is ≥ cutoff; a fitted scaler is reproducible and stored under `models/`; `sample_vintages` with the default spec is a verified no-op (identical row count), and at a reduced fraction samples whole loans (no loan appears partially).

---

## Stage 6 — QA & reconciliation  (`s6_qa.py`)

**Goal:** prove each stage preserved what it should and surface data-quality issues. Run after each stage; gates the `--delete-raw` option.

Checks:
1. **Row reconciliation:** `manifest.rows == perf.rows == clean.rows` per quarter (panel may differ only by documented dedup, if any).
2. **Null audit:** null rate per keep-list column per quarter → `outputs/null_audit.csv`; flag columns that flip from ~0% to ~100% across quarters (schema drift / version change).
3. **Target distribution:** 7×7 transition matrix and state marginals per quarter; assert no impossible jumps if you enforce a state machine (optional).
4. **Key sanity:** `(Loan Identifier, period)` is unique within the panel (no duplicate loan-months).
5. **Coverage report:** restate which quarters are present/quarantined/missing.
6. **Feature scale stats (§6.4):** per `continuous_standardize` column, emit mean/std/min/max/quantiles/null-rate → `outputs/feature_stats.csv`. Informs transform choices (e.g. confirm the `log` on UPB); these are *diagnostics*, not the train-time scaler.
7. **Shard & calendar sanity (§6.1–6.2):** `shard` ∈ `[0, N_SHARDS)` and roughly uniform; a loan's months never span two shards; `period_ym`/`orig_ym` are valid `YYYYMM` with `orig_ym ≤ period_ym`; report the derived data cut-off (max `period_ym`).
8. **Missingness audit (§6.3):** null/`*_missing` rates per indicator column per quarter; flag columns flipping ~0%↔~100% across quarters (schema/version drift).

Emit a single `outputs/qa_report.md`. **Acceptance:** all reconciliations pass for non-quarantined quarters; duplicates = 0; report committed.

---

## Orchestration (`run.py`)

A thin CLI: `python -m floan.pipeline.run <stage> [--quarter 2020Q4] [--all] [--delete-raw] [--include-quarantined]`. Stages run independently and per-quarter so you can process the ~800 GB incrementally and resume after interruption. Log to `logs/run.log` with per-quarter timing and peak memory.

---

### Recap

Six idempotent, per-quarter stages: (1) inventory + integrity audit that quarantines the truncated files and maps coverage gaps before anything else; (2) streaming CSV→zstd Parquet via DuckDB `COPY` or PyArrow chunked reader, never loading a full file; (3) lazy Polars `sink_parquet` cleaning that casts types, scrubs sentinels, projects to the keep-list, and adds the `period_ym`/`orig_ym` calendar keys and missingness indicators; (4) per-loan windowed derivation of the seven-state `state`/`state_next` transition target with a `censored` flag keyed off the **derived** data cut-off, plus the `shard = hash(loan_id) % N_SHARDS` minibatch key (output sorted by `(shard, loan, period)`); (5) DuckDB views over the Parquet lake with stratified sampling so rare default/REO transitions survive RAM-sized samples, shard-iterating and leakage-safe time-masking helpers, and train-only scaler fitting; (6) QA that reconciles row counts, audits nulls/missingness, emits feature scale stats, checks key/shard sanity, and gates raw-file deletion. `run.py` drives them per-quarter so the whole dataset processes incrementally and resumably.
