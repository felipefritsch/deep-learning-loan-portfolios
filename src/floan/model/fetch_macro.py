"""MD1 — Fetch the macro series snapshot (``05_MACRO_DATA.md`` §5).

Downloads the FRED CSV endpoint (keyless) for the rates / unemployment series and
verifies the manually-placed Freddie Mac FMHPI master file, writing everything into
one dated snapshot folder on the SSD::

    raw/macro/<snapshot>/fred/<SERIES>.csv     # 55 FRED series, one CSV each
    raw/macro/<snapshot>/fmhpi_master_file.csv # placed manually (verified here)
    raw/macro/<snapshot>/snapshot_manifest.json

The 55 FRED series are the 4 national series (30-yr PMMS, 10-yr & 2-yr Treasury,
national unemployment) plus the 50 states + DC seasonally-adjusted unemployment
series (``{POSTAL}UR``). Idempotent: a series whose CSV already exists is left
untouched, so re-running into the same dated folder performs no network fetches.

This is the one deliberate, documented exception to raw/ immutability
(``05_MACRO_DATA.md`` §2): we may write *only* into a new dated snapshot folder;
``raw/Performance_All/`` is never touched.

Run:  .venv/bin/python dev/model/fetch_macro.py [--snapshot YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

# The pipeline package is the single source of truth for paths (RAW, require_drive).
from floan.pipeline import config

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FMHPI_NAME = "fmhpi_master_file.csv"
SLEEP_S = 1.0          # polite gap between FRED requests
MIN_MONTHLY_ROWS = 300  # ≥ 26 yrs (Accept criterion)

# The daily Treasury series (DGS10, DGS2) are too large for FRED's graph-CSV
# renderer to stream within the 60 s network gateway window — every full-history
# request 504s. We fetch them already month-mean-aggregated server-side
# (fq=Monthly, fam=avg = average over the calendar month), which returns in <20 s
# and is exactly the month-mean MD2 would compute. Consequence: slope_10y2y becomes
# (monthly DGS10 − monthly DGS2) rather than the month-mean of the daily spread —
# numerically identical for these two co-published daily series (same quotation
# days), a documented, negligible deviation. All other series are fetched raw and
# aggregated locally in MD2.
DAILY_MONTHLY_AGG = {"DGS10", "DGS2"}

# 50 states + DC postal codes → FRED LAUS SA series are "{POSTAL}UR".
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
]
NATIONAL = ["MORTGAGE30US", "DGS10", "DGS2", "UNRATE"]
FRED_SERIES = NATIONAL + [f"{p}UR" for p in STATES]
assert len(FRED_SERIES) == 55, f"expected 55 FRED series, got {len(FRED_SERIES)}"


def _fred_url(series: str) -> str:
    url = FRED_CSV.format(series=series)
    if series in DAILY_MONTHLY_AGG:
        url += "&fq=Monthly&fam=avg"  # server-side month-mean (see DAILY_MONTHLY_AGG)
    return url


def _download(series: str, dest: Path, tries: int = 3) -> None:
    """Fetch one FRED series CSV to ``dest`` (small file, held in memory once)."""
    url = _fred_url(series)
    req = urllib.request.Request(url, headers={"User-Agent": "dissertation-macro-fetch/1.0"})
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
            if not body.strip():
                raise ValueError("empty body")
            dest.write_bytes(body)
            return
        # OSError covers urllib.error.URLError and socket.timeout (both subclasses).
        except (OSError, ValueError) as exc:  # network / timeout / empty body
            last = exc
            if attempt < tries:
                time.sleep(2 * attempt)
    raise SystemExit(f"FRED download failed for {series} after {tries} tries: {last}")


def _monthly_rows(csv_path: Path) -> int:
    """Distinct calendar months with a non-missing value (the post-aggregation count).

    FRED CSVs are date-keyed: column 0 is the observation date (name varies:
    ``observation_date`` / ``DATE``), column 1 the value with ``.`` = missing.
    Counting distinct ``YYYYMM`` of non-missing values gives the number of monthly
    rows the series yields after MD2's month-mean / as-is aggregation, regardless of
    native frequency (weekly PMMS, daily Treasuries, monthly UR all reduce here).
    """
    df = pl.read_csv(csv_path, infer_schema=False)  # all-utf8: parse defensively
    date_col, val_col = df.columns[0], df.columns[1]
    return (
        df.filter(pl.col(val_col).is_not_null())
        .filter(~pl.col(val_col).is_in([".", ""]))  # FRED missing / empty trailing month
        .select(pl.col(date_col).str.slice(0, 7).alias("ym"))  # "YYYY-MM"
        .n_unique()
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch the macro series snapshot (MD1).")
    ap.add_argument("--snapshot", default=date.today().isoformat(),
                    help="snapshot folder name under raw/macro/ (default: today)")
    args = ap.parse_args()

    config.require_drive()
    snap_dir = config.RAW / "macro" / args.snapshot
    fred_dir = snap_dir / "fred"
    fred_dir.mkdir(parents=True, exist_ok=True)

    # --- FRED loop (idempotent: skip files already present) -----------------
    fetched, skipped = 0, 0
    for series in FRED_SERIES:
        dest = fred_dir / f"{series}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        _download(series, dest)
        fetched += 1
        print(f"  fetched {series}", flush=True)
        time.sleep(SLEEP_S)
    print(f"FRED: {fetched} fetched, {skipped} already present "
          f"({len(FRED_SERIES)} total)")

    # --- FMHPI master file must already be in the snapshot (manual step) -----
    fmhpi = snap_dir / FMHPI_NAME
    if not fmhpi.exists():
        raise SystemExit(
            f"FMHPI master file not found at {fmhpi}\n"
            "Download it manually from "
            "https://www.freddiemac.com/research/indices/house-price-index "
            f"(the master file CSV) into {snap_dir}/ before re-running."
        )

    # --- Verify each FRED CSV yields enough monthly rows --------------------
    manifest_files = []
    bad = []
    for series in FRED_SERIES:
        path = fred_dir / f"{series}.csv"
        n = _monthly_rows(path)
        if n < MIN_MONTHLY_ROWS:
            bad.append((series, n))
        manifest_files.append({
            "name": f"fred/{series}.csv",
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "monthly_rows": n,
        })
    if bad:
        raise SystemExit(
            f"{len(bad)} FRED series under {MIN_MONTHLY_ROWS} monthly rows "
            f"(truncated download?): {bad}"
        )

    manifest_files.append({
        "name": FMHPI_NAME,
        "bytes": fmhpi.stat().st_size,
        "sha256": _sha256(fmhpi),
        "monthly_rows": None,  # built/asserted in MD2
    })

    manifest = {
        "snapshot": args.snapshot,
        "fetch_date": date.today().isoformat(),
        "fred_endpoint": FRED_CSV,
        "n_fred_series": len(FRED_SERIES),
        "monthly_aggregated_server_side": sorted(DAILY_MONTHLY_AGG),
        "notes": (
            "DGS10/DGS2 fetched with fq=Monthly&fam=avg (daily series 504 on the "
            "graph-CSV endpoint within the 60s gateway window); all others fetched "
            "raw and aggregated in build_macro.py."
        ),
        "files": sorted(manifest_files, key=lambda r: r["name"]),
    }
    manifest_path = snap_dir / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{len(FRED_SERIES)} FRED CSVs verified "
          f"(monthly rows {min(f['monthly_rows'] for f in manifest_files[:-1])}–"
          f"{max(f['monthly_rows'] for f in manifest_files[:-1])})")
    print(f"FMHPI present: {fmhpi.name} "
          f"({fmhpi.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
