"""Stage 1 — Inventory & integrity audit.

The dataset is the complete 2000–2025 acquisition-vintage set (~104 files, one
per vintage, no loan-level overlap). Even with a full download this audit MUST
run and be reviewed before any conversion — to confirm every vintage is present,
that all files share one release cut-off, and that none are truncated/corrupt.
It catalogues every raw CSV by streaming — no file is ever loaded into memory:

  * byte size, mtime, derived ``acq_quarter`` (from filename)
  * row count, distinct ``Loan Identifier`` count, min/max reporting period
    (one DuckDB ``read_csv`` query; DuckDB streams a tens-of-GB file and spills
    to disk under the 6 GB memory limit)
  * field-count check on the first N rows (each row must split into 113 fields)
  * NUL-padding probe (corrupt partial download) → quarantine

After every file is catalogued, two CROSS-FILE checks run (``_apply_cross_file_checks``):

  * release-cut-off consistency: all vintages should share the same modal
    ``max(Monthly Reporting Period)``; divergence ⇒ files came from different
    releases (inconsistent right-censoring) → flag + quarantine
  * relative-volume outlier: a vintage whose distinct-loan count is < 25% of its
    neighbours' median is almost certainly a clean truncation that the absolute
    thresholds miss (e.g. a quarter with ~80k loans between neighbours of ~500k)
    → flag + quarantine

Outputs ``outputs/inventory_manifest.csv`` (one row per file) and a
human-readable ``outputs/inventory_summary.md``. The run is idempotent: files
already in the manifest are skipped unless ``force=True``.

Never writes to ``raw/``.
"""

from __future__ import annotations

import csv
import os
import re
import statistics
from collections import Counter
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
# inflates byte size with binary padding while keeping few real rows, so
# byte-size alone is not a trustworthy completeness signal.
MAX_BYTES_PER_ROW = 1000              # >this ⇒ size not explained by real rows
NUL_PROBE_BYTES = 1 << 20             # 1 MB read at each probe offset
NUL_PROBE_OFFSETS = (0.25, 0.50, 0.75)
NUL_FRACTION_THRESH = 0.01            # >1% NUL bytes in a probe ⇒ corrupt padding

# Cross-file checks (applied after every file is catalogued; see
# _apply_cross_file_checks). These catch problems a single-file view cannot.
DATASET_START = "2000Q1"              # expected first vintage — anchor completeness here
RELATIVE_VOLUME_FRAC = 0.25           # quarantine if distinct_loans < this × neighbour median
VOLUME_NEIGHBOR_HALFWIN = 2           # neighbours = ±2 vintages (excluding self & corrupt)

MANIFEST_PATH = config.OUTPUTS / "inventory_manifest.csv"
SUMMARY_PATH = config.OUTPUTS / "inventory_summary.md"

MANIFEST_FIELDS = [
    "filename", "acq_quarter", "bytes", "size_mb", "mtime", "fingerprint",
    "rows", "distinct_loans", "min_period", "max_period",
    "bytes_per_row", "field_count_ok", "ragged_rows", "parse_errors",
    "nul_corrupt", "real_data_pct", "cutoff_divergent", "volume_ratio",
    "low_volume", "quarantined", "quarantine_reason",
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


def _real_data_pct(path: Path, size: int, nul_corrupt: bool) -> float:
    """Percentage of the file that is real data before a NUL tail begins.

    Only meaningful when ``nul_corrupt`` (a truncated write padded to full size
    with NUL bytes); returns ``100.0`` for a clean file. Binary-searches the
    boundary between the real-data prefix and the NUL tail — ~log2(size/4KB)
    tiny reads, no full scan — so the QA report can show, e.g., "real data ends
    ~55% of file" for eyeballing how much of a vintage actually downloaded.
    """
    if not nul_corrupt or size <= 4096:
        return 100.0
    with open(path, "rb") as fh:
        def _is_nul(off: int) -> bool:
            fh.seek(off)
            b = fh.read(4096)
            return bool(b) and b.count(0) / len(b) > 0.95

        # No sustained NUL tail at EOF ⇒ not the truncate-and-pad pattern.
        if not _is_nul(size - 4096):
            return 100.0
        lo, hi = 0, size
        while hi - lo > 4096:
            mid = (lo + hi) // 2
            if _is_nul(mid):
                hi = mid
            else:
                lo = mid
    return round(100.0 * hi / size, 1)


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
    real_data_pct = _real_data_pct(path, st.st_size, nul_corrupt)

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
        "real_data_pct": real_data_pct,
        "cutoff_divergent": False,   # set by _apply_cross_file_checks
        "volume_ratio": "",          # set by _apply_cross_file_checks
        "low_volume": False,         # set by _apply_cross_file_checks
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
        for k in ("size_mb", "bytes_per_row", "volume_ratio"):
            r[k] = float(r[k]) if str(r.get(k, "")).strip() not in ("", "None") else 0.0
        # real_data_pct: blank (unknown, e.g. reused cached row) stays None, not 100.
        rdp = str(r.get("real_data_pct", "")).strip()
        r["real_data_pct"] = float(rdp) if rdp not in ("", "None") else None
        for k in ("quarantined", "nul_corrupt", "cutoff_divergent", "low_volume"):
            r[k] = str(r.get(k, "")).lower() == "true"
        out.append(r)
    return out


def _apply_cross_file_checks(rows: list[dict]) -> tuple[list[dict], str]:
    """Cross-file integrity checks that a single-file view cannot make.

    Run once after every file is catalogued. Mutates and returns the (coerced)
    rows plus the modal release cut-off.

    1. Release-cut-off consistency — all vintages of one release share the same
       latest ``max_period``. Any divergent file is flagged ``cutoff_divergent``
       and quarantined (it likely came from a different release, giving an
       inconsistent right-censoring date).
    2. Relative-volume outlier — compares each vintage's distinct-loan count to
       the median of its ±VOLUME_NEIGHBOR_HALFWIN neighbours (excluding self and
       any corruption-flagged file, whose counts are unreliable). Below
       RELATIVE_VOLUME_FRAC of that median ⇒ ``low_volume`` + quarantine. This
       catches a clean truncation (no NUL padding, size/loan thresholds passed)
       that the per-file checks let through.
    """
    rows = sorted(rows, key=lambda r: _quarter_key(r["acq_quarter"]))

    cutoffs = [r["max_period"] for r in rows if r["max_period"]]
    modal_cutoff = Counter(cutoffs).most_common(1)[0][0] if cutoffs else ""

    def _is_corrupt(r: dict) -> bool:
        return bool(r.get("nul_corrupt")) or "corrupt" in str(r.get("quarantine_reason", ""))

    for i, r in enumerate(rows):
        # (1) release cut-off divergence
        divergent = bool(modal_cutoff and r["max_period"] and r["max_period"] != modal_cutoff)

        # (2) relative-volume outlier vs healthy neighbours
        lo, hi = max(0, i - VOLUME_NEIGHBOR_HALFWIN), min(len(rows), i + VOLUME_NEIGHBOR_HALFWIN + 1)
        neigh = [rows[j]["distinct_loans"] for j in range(lo, hi)
                 if j != i and not _is_corrupt(rows[j]) and rows[j]["distinct_loans"] > 0]
        med = statistics.median(neigh) if neigh else 0
        ratio = round(r["distinct_loans"] / med, 3) if med else 0.0
        low_vol = bool(med and not _is_corrupt(r)
                       and r["distinct_loans"] < RELATIVE_VOLUME_FRAC * med)

        r["cutoff_divergent"] = divergent
        r["volume_ratio"] = ratio
        r["low_volume"] = low_vol

        extra = []
        if divergent:
            extra.append(f"cutoff!={modal_cutoff}")
        if low_vol:
            extra.append(f"low_volume<{int(RELATIVE_VOLUME_FRAC * 100)}%_of_neighbours")
        if extra:
            prev = str(r.get("quarantine_reason", "")).strip()
            r["quarantine_reason"] = ";".join(([prev] if prev else []) + extra)
            r["quarantined"] = True

    return rows, modal_cutoff


def _write_manifest(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: _quarter_key(r["acq_quarter"]))
    with open(MANIFEST_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


def _write_summary(rows: list[dict], modal_cutoff: str = "") -> None:
    rows = sorted(_coerce(rows), key=lambda r: _quarter_key(r["acq_quarter"]))
    present = [r["acq_quarter"] for r in rows]
    quarantined = [r for r in rows if r["quarantined"]]
    divergent = [r for r in rows if r.get("cutoff_divergent")]
    low_vol = [r for r in rows if r.get("low_volume")]

    # Completeness anchored to the expected dataset start (DATASET_START), so a
    # missing *leading* vintage is caught, not just interior gaps.
    if present:
        start = min(present[0], DATASET_START, key=_quarter_key)
        full_range = _all_quarters_between(start, present[-1])
    else:
        full_range = []
    missing = [q for q in full_range if q not in set(present)]

    L = []
    L.append("# Stage 1 — Inventory & Integrity Audit\n")
    L.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}._\n")
    L.append(f"- Files catalogued: **{len(rows)}**")
    L.append(f"- Quarantined: **{len(quarantined)}** "
             f"(corrupt/truncated {len([r for r in quarantined if not r.get('cutoff_divergent')])}, "
             f"release-divergent {len(divergent)})")
    L.append(f"- Release cut-off (modal latest period): **{modal_cutoff or 'n/a'}** — "
             + ("✓ shared by all vintages" if not divergent
                else f"⚠️ {len(divergent)} divergent (possible mixed release)"))
    L.append(f"- Coverage span: **{present[0] if present else 'n/a'} → "
             f"{present[-1] if present else 'n/a'}** "
             f"({len(present)} present, {len(missing)} missing vs {DATASET_START}→latest)\n")

    L.append("## Coverage table\n")
    L.append("| Quarter | Size (MB) | Rows | Distinct loans | Vol vs nbrs | Bytes/row | Period span | Fields OK | Status |")
    L.append("|---|--:|--:|--:|--:|--:|---|:--:|---|")
    for r in rows:
        span = f"{r['min_period']} → {r['max_period']}".strip(" →")
        status = "🚫 QUARANTINE" if r["quarantined"] else "ok"
        flags = "✓" if str(r["field_count_ok"]).lower() == "true" else "✗ ragged"
        vr = r.get("volume_ratio", 0.0)
        vol = f"{vr:.2f}×" if vr else "—"
        L.append(
            f"| {r['acq_quarter']} | {r['size_mb']:,.2f} | {r['rows']:,} | "
            f"{r['distinct_loans']:,} | {vol} | {r['bytes_per_row']:,.0f} | {span} | "
            f"{flags} | {status} |"
        )
    L.append("")

    L.append("## Release cut-off consistency\n")
    L.append(f"Modal latest reporting period across all vintages: **{modal_cutoff or 'n/a'}**.\n")
    if divergent:
        L.append("⚠️ Vintages whose cut-off differs (likely a different release — re-pull from one release):\n")
        for r in divergent:
            L.append(f"- **{r['acq_quarter']}** — cut-off `{r['max_period']}` ≠ `{modal_cutoff}`")
    else:
        L.append("✓ All vintages share one release cut-off; no mixed-release inconsistency.")
    L.append("")

    if low_vol:
        L.append("## Relative-volume outliers (suspected clean truncation)\n")
        L.append(f"Vintages with distinct loans < {int(RELATIVE_VOLUME_FRAC*100)}% of neighbour median "
                 "— likely incomplete downloads the absolute thresholds missed:\n")
        for r in low_vol:
            L.append(f"- **{r['acq_quarter']}** — {r['distinct_loans']:,} loans "
                     f"({r.get('volume_ratio', 0):.2f}× neighbour median) — re-download & re-inventory")
        L.append("")

    L.append("## Quarantined files (excluded from default downstream runs)\n")
    if quarantined:
        for r in quarantined:
            line = (f"- **{r['acq_quarter']}** ({r['filename']}, "
                    f"{r['size_mb']:,.2f} MB, {r['distinct_loans']:,} loans) — "
                    f"reason: `{r['quarantine_reason']}`")
            rdp = r.get("real_data_pct")
            if r.get("nul_corrupt") and rdp is not None and rdp < 100:
                real_gb = r["bytes"] * rdp / 100 / 1e9
                line += (f" — real data ends ~{rdp:.1f}% in "
                         f"(~{real_gb:.1f} GB before NUL tail)")
            L.append(line)
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
                "parse_errors": True, "nul_corrupt": False, "real_data_pct": "",
                "cutoff_divergent": False, "volume_ratio": "", "low_volume": False,
                "quarantined": True, "quarantine_reason": f"scan_error:{type(e).__name__}",
            }
        results[f.name] = row
        _write_manifest(list(results.values()))  # checkpoint after each file

    # Cross-file pass (release-cut-off consistency + relative-volume outliers),
    # then rewrite the manifest+summary so both reflect the final quarantine set.
    final_rows, modal_cutoff = _apply_cross_file_checks(_coerce(list(results.values())))
    _write_manifest(final_rows)
    _write_summary(final_rows, modal_cutoff)
    con.close()

    nq = sum(1 for r in final_rows if r["quarantined"])
    print(f"Done. Manifest: {MANIFEST_PATH}")
    print(f"      Summary:  {SUMMARY_PATH}")
    print(f"      {len(final_rows)} files, {nq} quarantined (cut-off modal {modal_cutoff}).")
    return final_rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage 1 — inventory & integrity audit")
    ap.add_argument("--force", action="store_true",
                    help="rescan all files from scratch (ignore existing manifest)")
    ap.add_argument("--reprobe", action="store_true",
                    help="reuse cached DuckDB stats; recompute integrity probes only")
    args = ap.parse_args()
    run_inventory(force=args.force, reprobe=args.reprobe)
