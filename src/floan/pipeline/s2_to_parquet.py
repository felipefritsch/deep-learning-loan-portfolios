"""Stage 2 — Streaming CSV → Parquet  (s2_to_parquet.py).

Convert each raw quarterly CSV into compressed, dictionary-encoded Parquet via a
single DuckDB COPY that streams the file internally (never loads it into RAM).
This stage is a FAITHFUL, string-typed copy: every field is read as VARCHAR and
written unchanged — no casting, so it can never fail on a dirty value. Type
casting, sentinel scrubbing and projection happen in Stage 3.

  * all 113 positional columns, named from ``schema.COLUMNS`` (so Stage 3 can
    select ``KEEP_COLS`` by name)
  * zstd compression (``config.ZSTD_LEVEL``), row groups of ``PERF_ROW_GROUP_SIZE``
  * Hive-partitioned: ``interim/perf/acq_quarter=YYYYQn/part.parquet``
  * per-quarter, idempotent, resumable: a completed quarter (Parquet row count ==
    inventory row count) is skipped; the COPY writes to a ``.tmp`` file promoted
    with an atomic rename only after the round-trip row count matches, so an
    interrupted run (e.g. drive unplugged) never leaves a half-written
    ``part.parquet`` that looks done
  * quarantined vintages (per the Stage 1 manifest) are excluded unless
    ``--include-quarantined``

Never writes to ``raw/``. Output goes to the same SSD as the raw (no cross-bus
copy). Run directly:  ``python s2_to_parquet.py --quarter 2017Q2``  or  ``--all``.
"""

from __future__ import annotations

import argparse
import csv as csvmod
import os
import time
from datetime import datetime
from pathlib import Path

import duckdb

from floan.pipeline import config
from floan.pipeline import schema

MANIFEST_PATH = config.OUTPUTS / "inventory_manifest.csv"
LOG_PATH = config.LOGS / "run.log"

# Parquet row-group size for the perf lake. DuckDB's writer buffers
# ``threads × ROW_GROUP_SIZE`` rows in memory before flushing; with all 113
# columns as strings, ``config.CHUNK_ROWS`` (1,000,000) overruns the 6 GB cap.
# A smaller group keeps the COPY streaming in bounded memory on tens-of-GB files
# (the narrow typed clean/panel lakes in Stages 3-4 can use larger groups).
PERF_ROW_GROUP_SIZE = 128_000

# DuckDB list literal of the 113 column names (single quotes doubled for safety).
_COLUMN_NAMES_SQL = (
    "[" + ", ".join("'" + c.replace("'", "''") + "'" for c in schema.COLUMNS) + "]"
)


def _quarter_key(q: str) -> tuple[int, int]:
    return (int(q[:4]), int(q[-1]))


def _sql_str(p) -> str:
    """Escape a path for inlining inside a single-quoted SQL literal."""
    return str(p).replace("'", "''")


def _log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')}  s2  {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"No inventory manifest at {MANIFEST_PATH} — run Stage 1 first."
        )
    with open(MANIFEST_PATH, newline="") as fh:
        return {r["acq_quarter"]: r for r in csvmod.DictReader(fh)}


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    # A faithful copy doesn't need source row order preserved; turning this off
    # lets DuckDB stream the COPY in bounded memory on tens-of-GB files.
    con.execute("SET preserve_insertion_order=false")
    return con


def _parquet_rows(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql_str(path)}')"
    ).fetchone()[0]


def convert_quarter(
    con: duckdb.DuckDBPyConnection,
    quarter: str,
    expected_rows: "int | None",
    force: bool = False,
) -> dict:
    """Convert one quarter's CSV → Parquet. Idempotent and atomic."""
    raw = config.RAW_DIR / f"{quarter}.csv"
    if not raw.exists():
        raise SystemExit(f"Raw CSV not found: {raw}")
    q_dir = config.PERF_DIR / f"acq_quarter={quarter}"
    final = q_dir / "part.parquet"
    tmp = q_dir / "part.parquet.tmp"

    # Idempotent skip: a promoted file whose row count matches the inventory.
    if final.exists() and not force:
        got = _parquet_rows(con, final)
        if expected_rows is None or got == expected_rows:
            return {"quarter": quarter, "status": "skip", "rows": got,
                    "pq_mb": round(final.stat().st_size / 1048576, 1)}
        _log(f"{quarter}: existing Parquet {got:,} rows != manifest "
             f"{expected_rows:,} — reconverting")

    q_dir.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()  # clear a stale tmp from a prior interrupted run

    t0 = time.time()
    con.execute(f"""
        COPY (
            SELECT * FROM read_csv('{_sql_str(raw)}', delim='|', header=false,
                    all_varchar=true, names={_COLUMN_NAMES_SQL})
        ) TO '{_sql_str(tmp)}'
        (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL {config.ZSTD_LEVEL},
         ROW_GROUP_SIZE {PERF_ROW_GROUP_SIZE});
    """)
    secs = time.time() - t0

    # Round-trip reconciliation before promoting the tmp file.
    got = _parquet_rows(con, tmp)
    if expected_rows is not None and got != expected_rows:
        raise SystemExit(
            f"{quarter}: round-trip MISMATCH — Parquet {got:,} rows vs inventory "
            f"{expected_rows:,}. Left {tmp} for inspection; NOT promoted."
        )
    os.replace(tmp, final)  # atomic promote on the same filesystem

    csv_mb = raw.stat().st_size / 1048576
    pq_mb = final.stat().st_size / 1048576
    res = {"quarter": quarter, "status": "ok", "rows": got, "secs": round(secs, 1),
           "csv_mb": round(csv_mb, 1), "pq_mb": round(pq_mb, 1),
           "ratio": round(csv_mb / pq_mb, 2) if pq_mb else 0.0}
    _log(f"{quarter}: ok rows={got:,} csv={csv_mb:,.0f}MB pq={pq_mb:,.0f}MB "
         f"ratio={res['ratio']}x time={res['secs']}s")
    return res


def run(quarters, include_quarantined: bool = False, force: bool = False) -> list:
    config.ensure_dirs()
    manifest = _load_manifest()

    if quarters is None:  # --all: every non-quarantined vintage, chronological
        quarters = sorted(
            (q for q, r in manifest.items()
             if include_quarantined or r["quarantined"] != "True"),
            key=_quarter_key,
        )

    todo, skipped_q, missing = [], [], []
    for q in quarters:
        r = manifest.get(q)
        if r is None:
            missing.append(q)
            continue
        if r["quarantined"] == "True" and not include_quarantined:
            skipped_q.append(q)
            continue
        todo.append(q)

    if missing:
        _log(f"not in manifest (skipped): {', '.join(missing)}")
    if skipped_q:
        _log(f"excluding {len(skipped_q)} quarantined: {', '.join(skipped_q)}")
    _log(f"converting {len(todo)} vintage(s)"
         + (f": {todo[0]}" if len(todo) == 1 else ""))

    con = _connect()
    results = []
    try:
        for q in todo:
            raw_rows = manifest[q]["rows"]
            exp = int(raw_rows) if str(raw_rows).strip() not in ("", "0", "None") else None
            results.append(convert_quarter(con, q, exp, force=force))
    finally:
        con.close()

    ok = [r for r in results if r["status"] == "ok"]
    sk = [r for r in results if r["status"] == "skip"]
    tot_pq = sum(r.get("pq_mb", 0) for r in results)
    _log(f"done: {len(ok)} converted, {len(sk)} already-done, "
         f"perf lake footprint ~{tot_pq / 1024:.1f} GB")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 2 — streaming CSV → Parquet")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quarter", help="convert a single vintage, e.g. 2017Q2")
    g.add_argument("--all", action="store_true",
                   help="convert all non-quarantined vintages")
    ap.add_argument("--include-quarantined", action="store_true",
                    help="also convert quarantined vintages")
    ap.add_argument("--force", action="store_true",
                    help="reconvert even if a matching Parquet already exists")
    args = ap.parse_args()

    config.require_drive()
    qs = None if args.all else [args.quarter]
    run(qs, include_quarantined=args.include_quarantined, force=args.force)
