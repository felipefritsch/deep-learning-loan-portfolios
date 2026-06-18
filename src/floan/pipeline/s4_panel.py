"""Stage 4 — Build the transition panel  (s4_panel.py).

Derive the Sirignano seven-state monthly-transition target from the clean lake,
per quarter, adding four columns: ``state``, ``state_next``, ``censored`` and
``shard``. Written with DuckDB because this stage needs a per-loan ORDERED window
(``LEAD``) plus a global sort — DuckDB does both **out-of-core**, spilling to disk
under a hard ``memory_limit`` (Polars ``sink_parquet`` silently materialised
100M+ row quarters and was killed, so it is not used for the sort/window).

Per loan-month:
  * ``state`` — from ``Current Loan Delinquency Status`` + ``Zero Balance Code``
    (``01_SCHEMA.md`` §4). The SQL CASE below mirrors ``schema.derive_state``
    exactly; the smoke test cross-checks the two distributions agree.
  * ``state_next`` — the same loan's ``state`` next month via
    ``LEAD(state) OVER (PARTITION BY loan ORDER BY period)``. The window is safe
    per-quarter: a loan lives in exactly one acquisition vintage.
  * ``censored`` — TRUE iff a known, non-absorbing state has no observed next
    month (right-censored at the data cut-off). Absorbing terminations
    (prepaid/foreclosure/REO) get ``state_next = NULL`` and ``censored = FALSE``;
    null-state rows (XX/blank) get ``censored = FALSE`` (excluded from training
    anyway, since the label needs non-null state/state_next). The cut-off is the
    max ``period`` actually present — derived, never hardcoded.
  * ``shard`` — ``hash(loan_id, SHARD_SEED) % N_SHARDS`` (§6.2); output sorted by
    ``(shard, Loan Identifier, period)`` so ``WHERE shard = k`` skips row groups.

Per-quarter, idempotent, resumable; panel row count must equal the clean row
count; writes go to a ``.tmp`` promoted by atomic rename only after that
reconciliation. Never writes to ``raw/``.

Run:  ``python s4_panel.py --quarter 2017Q2 --describe``  or  ``--all``.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl
import pyarrow.parquet as pq

from floan.pipeline import config

LOG_PATH = config.LOGS / "run.log"
ABSORBING = ("prepaid", "foreclosure", "REO")
STATES = ("current", "dpd_30", "dpd_60", "dpd_90plus", "foreclosure", "REO", "prepaid")
PANEL_ROW_GROUP_SIZE = 256_000   # modest groups → fine shard skipping + bounded writer memory
STREAM_BATCH = 1_000_000         # rows pulled per Arrow batch from the sorted stream
DERIVED = ("state", "state_next", "censored", "shard")

# state derivation — mirrors schema.derive_state (01_SCHEMA.md §4). Zero Balance
# Code (terminal month) takes precedence; else delinquency status cast to int so
# 'XX'/blank → NULL naturally.
_STATE_SQL = """
  CASE
    WHEN "Zero Balance Code" IN ('01','06','16','96') THEN 'prepaid'
    WHEN "Zero Balance Code" IN ('02','03','15','97','98') THEN 'foreclosure'
    WHEN "Zero Balance Code" = '09' THEN 'REO'
    WHEN TRY_CAST("Current Loan Delinquency Status" AS INTEGER) = 0 THEN 'current'
    WHEN TRY_CAST("Current Loan Delinquency Status" AS INTEGER) = 1 THEN 'dpd_30'
    WHEN TRY_CAST("Current Loan Delinquency Status" AS INTEGER) = 2 THEN 'dpd_60'
    WHEN TRY_CAST("Current Loan Delinquency Status" AS INTEGER) >= 3 THEN 'dpd_90plus'
    ELSE NULL
  END
"""


def _quarter_key(q: str) -> tuple[int, int]:
    return (int(q[:4]), int(q[-1]))


def _sql_str(p) -> str:
    return str(p).replace("'", "''")


def _log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')}  s4  {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _parquet_rows(path: Path) -> int:
    return pq.ParquetFile(str(path)).metadata.num_rows


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")  # we ORDER BY explicitly
    # Spill sorts/windows to the fast internal disk, never the USB lake.
    tmp = config.DUCKDB_PATH.parent / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{_sql_str(tmp)}'")
    return con


def _sorted_sql(clean_file: Path) -> str:
    """Add ``state`` + ``shard`` and sort by (shard, loan, period).

    Only a plain ``ORDER BY`` — DuckDB's most reliably-spilling operator. No
    window function (those don't spill well and OOM'd at the 6 GB cap). Because
    ``shard`` is a function of the loan, sorting by (shard, loan, period) makes
    each loan's months contiguous and period-ordered, so ``state_next`` reduces
    to a one-row lookahead in the streamed output.
    """
    # hive_partitioning=false so DuckDB does NOT inject an ``acq_quarter`` column
    # from the path — that is the Hive PARTITION key and must not be stored in the
    # file (it would collide with the partition column on read).
    return f"""
      SELECT *, {_STATE_SQL} AS state,
        CAST(hash("Loan Identifier", {config.SHARD_SEED}) % {config.N_SHARDS} AS SMALLINT) AS shard
      FROM read_parquet('{_sql_str(clean_file)}', hive_partitioning=false)
      ORDER BY shard, "Loan Identifier", period
    """


def _finalize(df: pl.DataFrame) -> pl.DataFrame:
    """Compute state_next/censored from the per-row lookahead columns, order cols.

    ``state_next`` = the next row's state iff it is the same loan (else NULL =
    absorbing termination or right-censored end). ``censored`` = a known,
    non-absorbing state with no observed next month.
    """
    df = df.with_columns(
        pl.when(pl.col("_next_loan") == pl.col("Loan Identifier"))
        .then(pl.col("_next_state"))
        .otherwise(None)
        .alias("state_next")
    ).with_columns(
        (
            pl.col("state_next").is_null()
            & pl.col("state").is_not_null()
            & ~pl.col("state").is_in(list(ABSORBING))
        ).alias("censored")
    )
    base = [c for c in df.columns if c not in DERIVED and c not in ("_next_loan", "_next_state")]
    return df.select(base + list(DERIVED))


def build_quarter(con: duckdb.DuckDBPyConnection, quarter: str, force: bool = False) -> dict:
    clean_file = config.CLEAN_DIR / f"acq_quarter={quarter}" / "part.parquet"
    if not clean_file.exists():
        raise SystemExit(f"Clean partition not found for {quarter}: {clean_file}")
    expected = _parquet_rows(clean_file)

    out_dir = config.PANEL_DIR / f"acq_quarter={quarter}"
    final = out_dir / "part.parquet"
    tmp = out_dir / "part.parquet.tmp"

    if final.exists() and not force:
        got = _parquet_rows(final)
        if got == expected:
            return {"quarter": quarter, "status": "skip", "rows": got,
                    "pq_mb": round(final.stat().st_size / 1048576, 1)}
        _log(f"{quarter}: existing panel {got:,} rows != clean {expected:,} — rebuilding")

    out_dir.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()

    t0 = time.time()
    # DuckDB external-sorts (spilling to disk) and streams sorted Arrow batches;
    # we compute state_next in a bounded one-row-lookahead pass and write each
    # batch out. A row's state_next lives in the NEXT row, which may be the first
    # row of the next batch — so we hold each batch's last row over as `carry`,
    # prepend it to the next batch, and finalise it once its successor is known.
    reader = con.execute(_sorted_sql(clean_file)).fetch_record_batch(STREAM_BATCH)
    writer = None
    carry = None
    written = 0

    def _write(df: pl.DataFrame) -> int:
        nonlocal writer
        tbl = _finalize(df).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(
                str(tmp), tbl.schema,
                compression="zstd", compression_level=config.ZSTD_LEVEL,
            )
        writer.write_table(tbl, row_group_size=PANEL_ROW_GROUP_SIZE)
        return tbl.num_rows

    try:
        for batch in reader:
            df = pl.from_arrow(batch)
            if carry is not None:
                df = pl.concat([carry, df], how="vertical")
            df = df.with_columns([
                pl.col("Loan Identifier").shift(-1).alias("_next_loan"),
                pl.col("state").shift(-1).alias("_next_state"),
            ])
            carry = df.tail(1).drop(["_next_loan", "_next_state"])  # last row pending
            if df.height > 1:
                written += _write(df.head(df.height - 1))
        # final held row: no successor → state_next NULL
        if carry is not None:
            last = carry.with_columns([
                pl.lit(None, dtype=pl.Utf8).alias("_next_loan"),
                pl.lit(None, dtype=pl.Utf8).alias("_next_state"),
            ])
            written += _write(last)
    finally:
        if writer is not None:
            writer.close()
    secs = time.time() - t0

    got = _parquet_rows(tmp) if writer is not None else 0
    if got != expected:
        raise SystemExit(
            f"{quarter}: row MISMATCH — panel {got:,} vs clean {expected:,}. "
            f"Left {tmp} for inspection; NOT promoted."
        )
    os.replace(tmp, final)

    pq_mb = final.stat().st_size / 1048576
    res = {"quarter": quarter, "status": "ok", "rows": got, "secs": round(secs, 1),
           "pq_mb": round(pq_mb, 1)}
    _log(f"{quarter}: ok rows={got:,} panel={pq_mb:,.0f}MB time={res['secs']}s")
    return res


def validate(con: duckdb.DuckDBPyConnection, quarter: str) -> None:
    """Smoke-test checks for one quarter's panel (+ cross-check vs derive_state)."""
    f = _sql_str(config.PANEL_DIR / f"acq_quarter={quarter}" / "part.parquet")
    rd = f"read_parquet('{f}')"

    # 1. state domain
    states = [r[0] for r in con.execute(
        f"SELECT DISTINCT state FROM {rd} ORDER BY 1").fetchall()]
    bad = [s for s in states if s is not None and s not in STATES]
    print(f"\n--- {quarter} panel validation ---")
    print(f"distinct states: {states}")
    print(f"states outside the 7-set: {bad or 'none'}  (null allowed)")

    # 2. shard sanity
    nshard, mn, mx, avg = con.execute(
        f"SELECT count(*), min(c), max(c), avg(c) FROM "
        f"(SELECT shard, count(*) c FROM {rd} GROUP BY shard)").fetchone()
    loan_multi = con.execute(
        f'SELECT count(*) FROM (SELECT "Loan Identifier", count(DISTINCT shard) d '
        f'FROM {rd} GROUP BY 1 HAVING d > 1)').fetchone()[0]
    print(f"shards: {nshard} (expect {config.N_SHARDS}); per-shard rows "
          f"min={mn:,} max={mx:,} avg={avg:,.0f}; loans spanning >1 shard: {loan_multi}")

    # 3. (loan, period) uniqueness
    dups = con.execute(
        f'SELECT count(*) FROM (SELECT "Loan Identifier", period, count(*) c '
        f'FROM {rd} GROUP BY 1,2 HAVING c > 1)').fetchone()[0]
    print(f"duplicate (loan, period) rows: {dups}")

    # 4. censored sanity
    cens = con.execute(
        f"SELECT count(*) FILTER (WHERE censored), "
        f"count(*) FILTER (WHERE state_next IS NULL AND state IN "
        f"('prepaid','foreclosure','REO')) FROM {rd}").fetchone()
    print(f"right-censored rows: {cens[0]:,}; absorbing terminations: {cens[1]:,}")

    # 5. derived cut-off
    cutoff = con.execute(f"SELECT max(period_ym) FROM {rd}").fetchone()[0]
    print(f"derived data cut-off (max period_ym): {cutoff}")

    # 6. 7x7 transition matrix
    print("\n7x7 transition matrix (rows=state, cols=state_next; '·' = absorbing/censored null):")
    order = list(STATES)
    rows = con.execute(
        f"SELECT state, state_next, count(*) FROM {rd} "
        f"WHERE state IS NOT NULL GROUP BY 1,2").fetchall()
    mat = {(a, b): c for a, b, c in rows}
    hdr = "          " + "".join(f"{s[:9]:>11}" for s in order) + f"{'(end)':>11}"
    print(hdr)
    for a in order:
        line = f"{a:>10}"
        for b in order:
            line += f"{mat.get((a, b), 0):>11,}"
        line += f"{mat.get((a, None), 0):>11,}"
        print(line)

    # 7. CROSS-CHECK: SQL state distribution == schema.derive_state (Polars) on clean
    import polars as pl
    from floan.pipeline import schema
    cf = _sql_str(config.CLEAN_DIR / f"acq_quarter={quarter}" / "part.parquet")
    pol = (pl.scan_parquet(cf.replace("''", "'"))
           .select(schema.derive_state().alias("state"))
           .group_by("state").len().collect())
    pol_d = {r["state"]: r["len"] for r in pol.to_dicts()}
    sql = con.execute(f"SELECT state, count(*) FROM {rd} GROUP BY 1").fetchall()
    sql_d = {a: b for a, b in sql}
    match = pol_d == sql_d
    print(f"\nCROSS-CHECK SQL state vs schema.derive_state (Polars): "
          f"{'MATCH ✓' if match else 'MISMATCH ✗'}")
    if not match:
        keys = set(pol_d) | set(sql_d)
        for k in sorted(keys, key=lambda x: (x is None, x)):
            print(f"  {k}: polars={pol_d.get(k, 0):,}  sql={sql_d.get(k, 0):,}")


def run(quarters, force: bool = False) -> list:
    config.ensure_dirs()
    if quarters is None:  # --all
        quarters = sorted(
            (p.name.split("=", 1)[1] for p in config.CLEAN_DIR.glob("acq_quarter=*")
             if (p / "part.parquet").exists()),
            key=_quarter_key,
        )
    _log(f"building panel for {len(quarters)} vintage(s)"
         + (f": {quarters[0]}" if len(quarters) == 1 else ""))

    con = _connect()
    results = []
    try:
        for q in quarters:
            results.append(build_quarter(con, q, force=force))
    finally:
        con.close()

    ok = [r for r in results if r["status"] == "ok"]
    sk = [r for r in results if r["status"] == "skip"]
    tot = sum(r.get("pq_mb", 0) for r in results)
    _log(f"done: {len(ok)} built, {len(sk)} already-done, "
         f"panel lake footprint ~{tot / 1024:.1f} GB")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 4 — transition panel")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quarter", help="build a single vintage, e.g. 2017Q2")
    g.add_argument("--all", action="store_true", help="build all cleaned vintages")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a matching panel exists")
    ap.add_argument("--describe", action="store_true",
                    help="after a single-quarter run, print validation + matrix")
    args = ap.parse_args()

    config.require_drive()
    qs = None if args.all else [args.quarter]
    run(qs, force=args.force)
    if args.describe and args.quarter:
        con = _connect()
        try:
            validate(con, args.quarter)
        finally:
            con.close()
