"""Stage 1 — Inventory & integrity audit.

Coverage of this dataset is partial and at least three files are truncated, so
this audit MUST run and be reviewed before any conversion. It catalogues every
raw CSV by streaming — no file is ever loaded into memory:

  * byte size, mtime, derived ``acq_quarter`` (from filename)
  * row count, distinct ``Loan Identifier`` count, min/max reporting period
    (one DuckDB ``read_csv`` query; DuckDB streams a 25 GB file and spills to
    disk under the 6 GB memory limit)
  * field-count check on the first N rows (each row must split into 113 fields)
  * suspected-truncation flag → quarantine

Outputs ``outputs/inventory_manifest.csv`` (one row per file) and a
human-readable ``outputs/inventory_summary.md``. The run is idempotent: files
already in the manifest are skipped unless ``force=True``.

Never writes to ``raw/``.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path

import duckdb
from tqdm import tqdm

import config

# Quarantine thresholds (02_PIPELINE_STAGES.md §Stage 1).
MIN_DISTINCT_LOANS = 1000
MIN_BYTES = 50 * 1024 * 1024          # 50 MB
FIELD_SCAN_ROWS = 1000                # rows to check for the 113-field contract
EXPECTED_FIELDS = 113

# Corruption signals. A healthy row is ~270 bytes; a NUL-padded partial download
# (e.g. 2014Q2) inflates byte size with binary padding while keeping few real
# rows, so byte-size alone is not a trustworthy completeness signal.
MAX_BYTES_PER_ROW = 1000              # >this ⇒ size not explained by real rows
NUL_PROBE_BYTES = 1 << 20             # 1 MB read at each probe offset
NUL_PROBE_OFFSETS = (0.25, 0.50, 0.75)
NUL_FRACTION_THRESH = 0.01            # >1% NUL bytes in a probe ⇒ corrupt padding

MANIFEST_PATH = config.OUTPUTS / "inventory_manifest.csv"
SUMMARY_PATH = config.OUTPUTS / "inventory_summary.md"

MANIFEST_FIELDS = [
    "filename", "acq_quarter", "bytes", "size_mb", "mtime", "fingerprint",
    "rows", "distinct_loans", "min_period", "max_period",
    "bytes_per_row", "field_count_ok", "ragged_rows", "parse_errors",
    "nul_corrupt", "quarantined", "quarantine_reason",
]


def _fingerprint(st: os.stat_result) -> str:
    """Content fingerprint for change detection (vintage refreshes).

    Fannie Mae periodically re-releases a quarter as a newer, restated vintage
    under the same ``YYYYQn.csv`` name. Size + nanosecond mtime is a cheap,
    reliable signal that the file's content changed and the quarter must be
    re-inventoried (and later reconverted). No full read required.
    """
    return f"{st.st_size}:{st.st_mtime_ns}"

_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")
_COL_NAMES = "[" + ",".join(f"'column{i:02d}'" for i in range(EXPECTED_FIELDS)) + "]"


def _acq_quarter(path: Path) -> str:
    """Derive ``YYYYQn`` from the filename stem (e.g. ``2014Q1.csv``)."""
    return path.stem


def _quarter_key(q: str) -> tuple[int, int]:
    m = _QUARTER_RE.match(q)
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)


def _all_quarters_between(first: str, last: str) -> list[str]:
    (y0, q0), (y1, q1) = _quarter_key(first), _quarter_key(last)
    out = []
    y, q = y0, q0
    while (y, q) <= (y1, q1):
        out.append(f"{y}Q{q}")
        q += 1
        if q > 4:
            q, y = 1, y + 1
    return out


def _field_scan(path: Path, n: int = FIELD_SCAN_ROWS) -> tuple[bool, list[int]]:
    """Stream the first ``n`` lines; return (all_ok, ragged_line_numbers).

    A correct row has EXPECTED_FIELDS-1 pipe delimiters. Reads line-by-line so
    even a 25 GB file costs only the first n lines.
    """
    ragged: list[int] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            if line.count("|") != EXPECTED_FIELDS - 1:
                ragged.append(i + 1)
    return (len(ragged) == 0, ragged)


def _probe_nul(path: Path, size: int) -> tuple[bool, float]:
    """Seek to a few offsets and check for NUL-byte padding (corrupt download).

    Cheap (reads a few 1 MB windows, not the whole file). Catches partial
    downloads padded to full size with NUL bytes, where real CSV lives only at
    the start so a first-N-lines scan looks clean. Returns (is_corrupt,
    max_nul_fraction_seen).
    """
    if size == 0:
        return (False, 0.0)
    worst = 0.0
    with open(path, "rb") as fh:
        for frac in NUL_PROBE_OFFSETS:
            fh.seek(int(size * frac))
            chunk = fh.read(NUL_PROBE_BYTES)
            if not chunk:
                continue
            nul_fraction = chunk.count(0) / len(chunk)
            worst = max(worst, nul_fraction)
    return (worst > NUL_FRACTION_THRESH, round(worst, 4))


def _stats(con: duckdb.DuckDBPyConnection, path: Path) -> tuple[dict, bool]:
    """Row count, distinct loans, min/max period via streaming DuckDB.

    Returns (stats_dict, parse_errors). Falls back to ``ignore_errors=true``
    if strict parsing raises, flagging that the numbers are approximate.
    """
    p = str(path).replace("'", "''")

    def _query(ignore_errors: bool) -> tuple:
        opts = (
            f"delim='|', header=false, all_varchar=true, names={_COL_NAMES}"
            + (", ignore_errors=true" if ignore_errors else "")
        )
        sql = f"""
          SELECT count(*) AS rows,
                 count(DISTINCT column01) AS distinct_loans,
                 min(try_strptime(column02, '%m%Y')) AS min_period,
                 max(try_strptime(column02, '%m%Y')) AS max_period
          FROM read_csv('{p}', {opts})
        """
        return con.execute(sql).fetchone()

    parse_errors = False
    try:
        rows, loans, lo, hi = _query(ignore_errors=False)
    except duckdb.Error:
        parse_errors = True
        rows, loans, lo, hi = _query(ignore_errors=True)

    def _fmt(ts) -> str:
        return ts.strftime("%Y-%m") if ts is not None else ""

    return (
        {
            "rows": rows,
            "distinct_loans": loans,
            "min_period": _fmt(lo),
            "max_period": _fmt(hi),
        },
        parse_errors,
    )


def _inventory_one(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    cached: dict | None = None,
) -> dict:
    """Catalogue one file. If ``cached`` (a prior manifest row with valid stats)
    is given, reuse the expensive DuckDB stats and only recompute the cheap
    integrity probes — makes corrective re-runs fast.
    """
    st = path.stat()
    field_ok, ragged = _field_scan(path)

    if cached is not None:
        stats = {
            "rows": int(cached["rows"]),
            "distinct_loans": int(cached["distinct_loans"]),
            "min_period": cached["min_period"],
            "max_period": cached["max_period"],
        }
        parse_errors = str(cached.get("parse_errors", "")).lower() == "true"
    else:
        stats, parse_errors = _stats(con, path)

    bytes_per_row = round(st.st_size / stats["rows"], 1) if stats["rows"] else 0.0
    nul_corrupt, _nul_frac = _probe_nul(path, st.st_size)

    reasons = []
    if st.st_size < MIN_BYTES:
        reasons.append(f"size<{MIN_BYTES // (1024*1024)}MB")
    if stats["distinct_loans"] < MIN_DISTINCT_LOANS:
        reasons.append(f"distinct_loans<{MIN_DISTINCT_LOANS}")
    if nul_corrupt:
        reasons.append("corrupt_nul_padding")
    if bytes_per_row > MAX_BYTES_PER_ROW:
        reasons.append(f"bytes_per_row>{MAX_BYTES_PER_ROW}")
    if not field_ok:
        reasons.append("ragged_rows")
    quarantined = bool(reasons)

    return {
        "filename": path.name,
        "acq_quarter": _acq_quarter(path),
        "bytes": st.st_size,
        "size_mb": round(st.st_size / (1024 * 1024), 2),
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "fingerprint": _fingerprint(st),
        "rows": stats["rows"],
        "distinct_loans": stats["distinct_loans"],
        "min_period": stats["min_period"],
        "max_period": stats["max_period"],
        "bytes_per_row": bytes_per_row,
        "field_count_ok": field_ok,
        "ragged_rows": ";".join(map(str, ragged)),
        "parse_errors": parse_errors,
        "nul_corrupt": nul_corrupt,
        "quarantined": quarantined,
        "quarantine_reason": ";".join(reasons),
    }


def _load_existing() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, newline="") as fh:
        return {r["filename"]: r for r in csv.DictReader(fh)}


def _coerce(rows: list[dict]) -> list[dict]:
    """Normalise types loaded back from CSV (strings) for sorting/summary."""
    out = []
    for r in rows:
        r = dict(r)
        for k in ("bytes", "rows", "distinct_loans"):
            r[k] = int(r[k]) if str(r[k]).strip() not in ("", "None") else 0
        for k in ("size_mb", "bytes_per_row"):
            r[k] = float(r[k]) if str(r.get(k, "")).strip() not in ("", "None") else 0.0
        for k in ("quarantined", "nul_corrupt"):
            r[k] = str(r.get(k, "")).lower() == "true"
        out.append(r)
    return out


def _write_manifest(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: _quarter_key(r["acq_quarter"]))
    with open(MANIFEST_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


def _write_summary(rows: list[dict]) -> None:
    rows = sorted(_coerce(rows), key=lambda r: _quarter_key(r["acq_quarter"]))
    present = [r["acq_quarter"] for r in rows]
    quarantined = [r for r in rows if r["quarantined"]]
    quarantined_qs = {r["acq_quarter"] for r in quarantined}

    full_range = _all_quarters_between(present[0], present[-1]) if present else []
    missing = [q for q in full_range if q not in set(present)]

    L = []
    L.append("# Stage 1 — Inventory & Integrity Audit\n")
    L.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}._\n")
    L.append(f"- Files catalogued: **{len(rows)}**")
    L.append(f"- Quarantined: **{len(quarantined)}**")
    L.append(f"- Coverage span: **{present[0]} → {present[-1]}** "
             f"({len(present)} present, {len(missing)} missing in range)\n")

    L.append("## Coverage table\n")
    L.append("| Quarter | Size (MB) | Rows | Distinct loans | Bytes/row | Period span | Fields OK | Status |")
    L.append("|---|--:|--:|--:|--:|---|:--:|---|")
    for r in rows:
        span = f"{r['min_period']} → {r['max_period']}".strip(" →")
        status = "🚫 QUARANTINE" if r["quarantined"] else "ok"
        flags = "✓" if str(r["field_count_ok"]).lower() == "true" else "✗ ragged"
        L.append(
            f"| {r['acq_quarter']} | {r['size_mb']:,.2f} | {r['rows']:,} | "
            f"{r['distinct_loans']:,} | {r['bytes_per_row']:,.0f} | {span} | "
            f"{flags} | {status} |"
        )
    L.append("")

    L.append("## Quarantined files (excluded from default downstream runs)\n")
    if quarantined:
        for r in quarantined:
            L.append(f"- **{r['acq_quarter']}** ({r['filename']}, "
                     f"{r['size_mb']:,.2f} MB, {r['distinct_loans']:,} loans) — "
                     f"reason: `{r['quarantine_reason']}`")
    else:
        L.append("_None._")
    L.append("")

    L.append("## Coverage gaps\n")
    if missing:
        L.append(f"Missing quarters between {present[0]} and {present[-1]} "
                 f"({len(missing)}):\n")
        L.append("`" + "`, `".join(missing) + "`")
    else:
        L.append("_No gaps in the observed range._")
    L.append("")
    L.append("> Quarantined quarters are excluded from default downstream runs "
             "unless `--include-quarantined` is passed.\n")

    SUMMARY_PATH.write_text("\n".join(L))


def run_inventory(force: bool = False, reprobe: bool = False) -> list[dict]:
    """Catalogue every raw CSV.

    - ``force``   : ignore the existing manifest and rescan everything.
    - ``reprobe`` : reuse cached DuckDB stats for files already in the manifest
                    but recompute the cheap integrity probes (size, bytes/row,
                    NUL padding, field count) and quarantine decision. Fast way
                    to apply improved corruption rules without re-streaming.
    - default     : content-aware resume — fully scan files that are new or
                    whose fingerprint changed (a refreshed vintage), reuse the
                    cached row for unchanged files, and report removed files.
    """
    config.ensure_dirs()
    files = sorted(config.RAW_DIR.glob("*.csv"), key=lambda p: _quarter_key(p.stem))
    if not files:
        raise SystemExit(f"No CSVs found under {config.RAW_DIR}")

    existing = {} if force else _load_existing()

    # Detect removed files: in the manifest but no longer on disk. Drop them
    # from the rebuilt manifest so it always reflects current truth.
    on_disk = {f.name for f in files}
    removed = [name for name in existing if name not in on_disk]
    results: dict[str, dict] = {n: r for n, r in existing.items() if n in on_disk}

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")

    if reprobe:
        todo = [f for f in files if existing.get(f.name, {}).get("rows", "0") not in ("", "0")]
        new = changed = []
        label = "reprobe"
    else:
        # Classify each on-disk file as new / changed (vintage refresh) / unchanged.
        new = [f for f in files if f.name not in existing]
        changed = [
            f for f in files
            if f.name in existing
            and existing[f.name].get("fingerprint", "") != _fingerprint(f.stat())
        ]
        todo = sorted(set(new) | set(changed), key=lambda p: _quarter_key(p.stem))
        label = "inventory"

    print(f"Stage 1 {label}: {len(files)} files on disk — "
          f"{len(new)} new, {len(changed)} changed (vintage refresh), "
          f"{len(removed)} removed, {len(todo)} to (re)scan.")
    for f in changed:
        print(f"  ~ refreshed vintage: {f.name} (fingerprint changed → reconvert needed)")
    for name in removed:
        print(f"  - removed from disk:  {name} (dropped from manifest)")

    for f in (tqdm(todo, desc=label, unit="file") if todo else []):
        cached = existing.get(f.name) if reprobe else None
        try:
            row = _inventory_one(con, f, cached=cached)
        except Exception as e:  # noqa: BLE001 — record and continue the audit
            print(f"  ! {f.name}: {type(e).__name__}: {e}")
            st = f.stat()
            row = {
                "filename": f.name, "acq_quarter": _acq_quarter(f),
                "bytes": st.st_size, "size_mb": round(st.st_size / 1048576, 2),
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "fingerprint": _fingerprint(st),
                "rows": 0, "distinct_loans": 0, "min_period": "", "max_period": "",
                "bytes_per_row": 0.0, "field_count_ok": False, "ragged_rows": "",
                "parse_errors": True, "nul_corrupt": False,
                "quarantined": True, "quarantine_reason": f"scan_error:{type(e).__name__}",
            }
        results[f.name] = row
        _write_manifest(list(results.values()))  # checkpoint after each file

    _write_summary(list(results.values()))
    con.close()

    nq = sum(1 for r in _coerce(list(results.values())) if r["quarantined"])
    print(f"Done. Manifest: {MANIFEST_PATH}")
    print(f"      Summary:  {SUMMARY_PATH}")
    print(f"      {len(results)} files, {nq} quarantined.")
    return list(results.values())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage 1 — inventory & integrity audit")
    ap.add_argument("--force", action="store_true",
                    help="rescan all files from scratch (ignore existing manifest)")
    ap.add_argument("--reprobe", action="store_true",
                    help="reuse cached DuckDB stats; recompute integrity probes only")
    args = ap.parse_args()
    run_inventory(force=args.force, reprobe=args.reprobe)
