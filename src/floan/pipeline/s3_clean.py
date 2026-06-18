"""Stage 3 — Clean & standardise  (s3_clean.py).

Cast the faithful string-typed perf lake into a typed, standardised, projected
``clean`` lake in bounded memory. Execution is an EXPLICIT batch loop — PyArrow
``iter_batches`` reads the perf partition one ~500K-row batch at a time (only the
KEEP_COLS source columns), the validated Polars expression transform is applied
per batch, and a PyArrow ``ParquetWriter`` appends each result. (Polars
``sink_parquet`` materialised 100M+ row quarters to ~86 GB and was killed, so the
loop is the memory-safe path that honours "never load a full quarter".)

Per ``01_SCHEMA.md`` §3/§5/§6 each quarter partition is transformed to the
keep-list plus derived training columns:

  * dates: ``MMYYYY`` strings → first-of-month ``date`` (``period``, ``orig_date``,
    ``First Payment Date``, ``Zero Balance Effective Date``); no MMYYYY strings
    survive
  * numerics: floats → f32, integers → smallest nullable int; blanks → null (not 0)
  * credit scores: sentinel-scrubbed (300–850 kept, else null); origination FICO
    is COALESCED across the Dec-2025 field migration via ``schema.fico_orig_expr``
    (``fico_orig``); co-borrower → ``fico_co``
  * categoricals: Y/N → bool; code maps (Channel/Loan Purpose/…) → readable
    dictionary-encoded categoricals; blanks → null
  * §6.1 calendar keys ``period_ym`` / ``orig_ym`` (integer ``YYYYMM``) for
    rolling-backtest masking
  * §6.3 missingness indicators ``fico_orig_missing`` / ``fico_co_missing`` /
    ``dti_missing`` / ``cltv_missing`` / ``mi_pct_missing``
  * ``Current Loan Delinquency Status`` and ``Zero Balance Code`` are kept (the
    Stage 4 target drivers)

Per-quarter, idempotent and resumable (clean row count must equal the perf row
count); writes go to a ``.tmp`` promoted by atomic rename only after that
reconciliation passes. Never writes to ``raw/``.

Run:  ``python s3_clean.py --quarter 2017Q2``  or  ``--all``.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from floan.pipeline import config
from floan.pipeline import schema

LOG_PATH = config.LOGS / "run.log"


def _quarter_key(q: str) -> tuple[int, int]:
    return (int(q[:4]), int(q[-1]))


def _log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')}  s3  {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Column expression builders (all return Polars expressions)
# ---------------------------------------------------------------------------
def _f32(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float32, strict=False).alias(name)


def _int(name: str, dtype: pl.DataType) -> pl.Expr:
    return pl.col(name).cast(dtype, strict=False).alias(name)


def _blank_null(name: str) -> pl.Expr:
    """Trimmed string; empty → null."""
    c = pl.col(name).cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.when(c.str.len_chars() == 0).then(None).otherwise(c)


def _yn(name: str) -> pl.Expr:
    """Y/N → Boolean; everything else (blank, 7, 9, …) → null."""
    c = pl.col(name).cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()
    return (
        pl.when(c == "Y").then(pl.lit(True))
        .when(c == "N").then(pl.lit(False))
        .otherwise(None)
        .alias(name)
    )


def _cat(name: str) -> pl.Expr:
    """Low-cardinality code column; blank → null; raw codes preserved.

    Kept as Utf8 (NOT a Polars ``Categorical``): the Categorical dtype needs a
    global string cache that forces the whole query in-memory and defeats
    streaming. Parquet dictionary-encodes these repetitive strings automatically,
    so storage is unchanged; Stage 4/5 cast to Categorical on small samples.
    """
    return _blank_null(name).alias(name)


def _catmap(name: str, mapping: dict) -> pl.Expr:
    """Map codes to readable labels (unmapped pass through); kept as Utf8."""
    return _blank_null(name).replace(mapping).alias(name)


# Stage-1 transforms: typed/derived columns (originals remain until the final
# select projects them away).
def _stage1_exprs() -> list[pl.Expr]:
    return [
        pl.col("Loan Identifier").cast(pl.Utf8, strict=False).str.strip_chars()
          .alias("Loan Identifier"),
        # dates
        schema.parse_mmyyyy("Monthly Reporting Period").alias("period"),
        schema.parse_mmyyyy("Origination Date").alias("orig_date"),
        schema.parse_mmyyyy("First Payment Date").alias("First Payment Date"),
        schema.parse_mmyyyy("Zero Balance Effective Date").alias("Zero Balance Effective Date"),
        # rates / balances (f32)
        _f32("Original Interest Rate"),
        _f32("Current Interest Rate"),
        _f32("Original UPB"),
        _f32("Current Actual UPB"),
        _f32("Original Loan-to-Value (LTV)"),
        _f32("Original Combined LTV (CLTV)"),
        _f32("Debt-to-Income (DTI)"),
        _f32("Mortgage Insurance Percentage"),
        _f32("Unscheduled Principal Current"),
        # integers (smallest nullable)
        _int("Original Loan Term", pl.Int16),
        _int("Loan Age", pl.Int16),
        _int("Remaining Months to Maturity", pl.Int16),
        _int("Number of Borrowers", pl.Int8),
        _int("Number of Units", pl.Int8),
        # credit scores: coalesced origination FICO + co-borrower
        schema.fico_orig_expr(),                                  # -> fico_orig
        schema.scrub_credit_score("Co-Borrower Credit Score at Origination").alias("fico_co"),
        # booleans
        _yn("First-Time Home Buyer Indicator"),
        _yn("Interest-Only Loan Indicator"),
        _yn("Modification Flag"),
        # categoricals (mapped to readable labels)
        _catmap("Channel", schema.CHANNEL),
        _catmap("Loan Purpose", schema.LOAN_PURPOSE),
        _catmap("Property Type", schema.PROPERTY),
        _catmap("Occupancy Status", schema.OCCUPANCY),
        _catmap("Amortization Type", schema.AMORT_TYPE),
        # categoricals (raw codes preserved)
        _cat("Property State"),
        _cat("Metropolitan Statistical Area (MSA)"),
        _cat("Zip Code Short"),
        _cat("Zero Balance Code"),
        # target driver kept as string codes for Stage 4 derive_state
        _blank_null("Current Loan Delinquency Status").alias("Current Loan Delinquency Status"),
    ]


# Stage-2 transforms: depend on the Stage-1 outputs above.
def _stage2_exprs() -> list[pl.Expr]:
    return [
        (pl.col("period").dt.year() * 100 + pl.col("period").dt.month())
            .cast(pl.Int32).alias("period_ym"),
        (pl.col("orig_date").dt.year() * 100 + pl.col("orig_date").dt.month())
            .cast(pl.Int32).alias("orig_ym"),
        pl.col("fico_orig").is_null().alias("fico_orig_missing"),
        pl.col("fico_co").is_null().alias("fico_co_missing"),
        pl.col("Debt-to-Income (DTI)").is_null().alias("dti_missing"),
        pl.col("Original Combined LTV (CLTV)").is_null().alias("cltv_missing"),
        pl.col("Mortgage Insurance Percentage").is_null().alias("mi_pct_missing"),
    ]


# Final projection (order = clean-lake schema). Stage 4 adds state/state_next/
# censored/shard on top of this.
OUTPUT_COLS: list[str] = [
    "Loan Identifier", "period", "period_ym", "orig_date", "orig_ym",
    "First Payment Date",
    "Channel",
    "Original Interest Rate", "Current Interest Rate",
    "Original UPB", "Current Actual UPB",
    "Original Loan Term", "Loan Age", "Remaining Months to Maturity",
    "Original Loan-to-Value (LTV)", "Original Combined LTV (CLTV)",
    "Number of Borrowers", "Debt-to-Income (DTI)",
    "fico_orig", "fico_co",
    "First-Time Home Buyer Indicator", "Loan Purpose", "Property Type",
    "Number of Units", "Occupancy Status", "Property State",
    "Metropolitan Statistical Area (MSA)", "Zip Code Short",
    "Mortgage Insurance Percentage", "Amortization Type",
    "Interest-Only Loan Indicator", "Current Loan Delinquency Status",
    "Modification Flag", "Zero Balance Code", "Zero Balance Effective Date",
    "Unscheduled Principal Current",
    "fico_orig_missing", "fico_co_missing", "dti_missing", "cltv_missing",
    "mi_pct_missing",
]


# Rows transformed per batch. Stage 3 is executed as an EXPLICIT batch loop
# (PyArrow ``iter_batches`` → per-batch Polars transform → ``ParquetWriter``)
# rather than ``LazyFrame.sink_parquet``: on 100M+ row quarters Polars' sink
# (even with engine="streaming") materialised the whole frame, spiking memory to
# ~86 GB and getting the process killed. An explicit batch loop bounds memory to
# one batch regardless of quarter size, satisfying the "never load a full
# quarter" rule. Only the ~34 source columns (KEEP_COLS) are read per batch.
BATCH_ROWS = 500_000


def _transform_batch(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the full Stage-3 transform to one (bounded) batch."""
    return (
        df.lazy()
        .with_columns(_stage1_exprs())
        .with_columns(_stage2_exprs())
        .select(OUTPUT_COLS)
        .collect()
    )


def _parquet_rows(path: Path) -> int:
    return pq.ParquetFile(str(path)).metadata.num_rows


def clean_quarter(quarter: str, force: bool = False) -> dict:
    """Clean one quarter's perf partition → clean partition. Idempotent/atomic."""
    perf_file = config.PERF_DIR / f"acq_quarter={quarter}" / "part.parquet"
    if not perf_file.exists():
        raise SystemExit(f"Perf partition not found for {quarter}: {perf_file}")
    expected = _parquet_rows(perf_file)

    out_dir = config.CLEAN_DIR / f"acq_quarter={quarter}"
    final = out_dir / "part.parquet"
    tmp = out_dir / "part.parquet.tmp"

    if final.exists() and not force:
        got = _parquet_rows(final)
        if got == expected:
            return {"quarter": quarter, "status": "skip", "rows": got,
                    "pq_mb": round(final.stat().st_size / 1048576, 1)}
        _log(f"{quarter}: existing clean {got:,} rows != perf {expected:,} — recleaning")

    out_dir.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()

    t0 = time.time()
    pf = pq.ParquetFile(str(perf_file))
    writer = None
    written = 0
    try:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=schema.KEEP_COLS):
            out = _transform_batch(pl.from_arrow(batch))   # one bounded batch
            tbl = out.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    str(tmp), tbl.schema,
                    compression="zstd", compression_level=config.ZSTD_LEVEL,
                )
            writer.write_table(tbl)
            written += out.height
    finally:
        if writer is not None:
            writer.close()
    secs = time.time() - t0

    got = _parquet_rows(tmp) if writer is not None else 0
    if got != expected:
        raise SystemExit(
            f"{quarter}: row MISMATCH — clean {got:,} vs perf {expected:,}. "
            f"Left {tmp} for inspection; NOT promoted."
        )
    os.replace(tmp, final)

    perf_mb = perf_file.stat().st_size / 1048576
    pq_mb = final.stat().st_size / 1048576
    res = {"quarter": quarter, "status": "ok", "rows": got, "secs": round(secs, 1),
           "perf_mb": round(perf_mb, 1), "pq_mb": round(pq_mb, 1)}
    _log(f"{quarter}: ok rows={got:,} perf={perf_mb:,.0f}MB clean={pq_mb:,.0f}MB "
         f"time={res['secs']}s")
    return res


def describe(quarter: str) -> None:
    """Print the clean partition's schema + key null rates (smoke-test check)."""
    f = config.CLEAN_DIR / f"acq_quarter={quarter}" / "part.parquet"
    lf = pl.scan_parquet(str(f))
    sch = lf.collect_schema()
    print(f"\n--- clean schema for {quarter} ({len(sch)} columns) ---")
    for name, dt in sch.items():
        print(f"  {str(dt):<12} {name}")
    # null rates + sanity on a sample (bounded memory)
    samp = lf.select([
        pl.len().alias("n"),
        pl.col("period").is_null().mean().alias("period_null"),
        pl.col("period_ym").min().alias("period_ym_min"),
        pl.col("period_ym").max().alias("period_ym_max"),
        pl.col("orig_ym").min().alias("orig_ym_min"),
        pl.col("fico_orig").is_null().mean().alias("fico_orig_null"),
        pl.col("fico_orig_missing").mean().alias("fico_orig_missing_rate"),
        pl.col("Debt-to-Income (DTI)").is_null().mean().alias("dti_null"),
        pl.col("Current Loan Delinquency Status").is_null().mean().alias("delinq_null"),
        pl.col("Zero Balance Code").is_not_null().mean().alias("zb_present_rate"),
    ]).collect()
    print("\n--- sanity (whole partition) ---")
    for k, v in samp.to_dicts()[0].items():
        print(f"  {k} = {v}")
    no_mmyyyy = all(str(dt) not in ("String", "Utf8")
                    or name in ("Loan Identifier", "Current Loan Delinquency Status")
                    for name, dt in sch.items())
    print(f"\n  no stray MMYYYY date-strings: {no_mmyyyy}")


def run(quarters, force: bool = False) -> list:
    config.ensure_dirs()
    if quarters is None:  # --all: every converted perf partition, chronological
        quarters = sorted(
            (p.name.split("=", 1)[1] for p in config.PERF_DIR.glob("acq_quarter=*")
             if (p / "part.parquet").exists()),
            key=_quarter_key,
        )
    _log(f"cleaning {len(quarters)} vintage(s)"
         + (f": {quarters[0]}" if len(quarters) == 1 else ""))

    results = []
    for q in quarters:
        results.append(clean_quarter(q, force=force))

    ok = [r for r in results if r["status"] == "ok"]
    sk = [r for r in results if r["status"] == "skip"]
    tot = sum(r.get("pq_mb", 0) for r in results)
    _log(f"done: {len(ok)} cleaned, {len(sk)} already-done, "
         f"clean lake footprint ~{tot / 1024:.1f} GB")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 3 — clean & standardise")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quarter", help="clean a single vintage, e.g. 2017Q2")
    g.add_argument("--all", action="store_true", help="clean all converted vintages")
    ap.add_argument("--force", action="store_true",
                    help="reclean even if a matching clean Parquet exists")
    ap.add_argument("--describe", action="store_true",
                    help="after a single-quarter run, print schema + null sanity")
    args = ap.parse_args()

    config.require_drive()
    qs = None if args.all else [args.quarter]
    run(qs, force=args.force)
    if args.describe and args.quarter:
        describe(args.quarter)
