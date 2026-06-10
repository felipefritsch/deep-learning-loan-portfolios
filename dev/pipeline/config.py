"""Central configuration for the Fannie Mae loan-performance pipeline.

Single source of truth for every path and resource knob. All storage paths
derive from ROOT on the external SSD ``SSD Felipe`` and follow the
immutable-raw layout: raw/ (originals, NEVER written) -> interim/ (Stages 2-3)
-> processed/ (Stages 4-5), plus models/, outputs/, logs/. The only file on
the internal disk is the disposable DuckDB catalog under the repo's reports/.

Every stage entry point must call ``require_drive()`` before touching data.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Storage root — external SSD. Everything below derives from this.
# ---------------------------------------------------------------------------
ROOT = Path("/Volumes/SSD Felipe/dissertation")

RAW       = ROOT / "raw"        # immutable original CSVs — NEVER written to
INTERIM   = ROOT / "interim"    # cleaning outputs (rebuildable from raw)
PROCESSED = ROOT / "processed"  # training-ready data
MODELS    = ROOT / "models"     # trained models / checkpoints
OUTPUTS   = ROOT / "outputs"    # manifests, QA reports, figures
LOGS      = ROOT / "logs"       # run.log

RAW_DIR    = RAW / "Performance_All"   # original quarterly CSVs (read-only)
PERF_DIR   = INTERIM / "perf"          # Stage 2: faithful string-typed Parquet
CLEAN_DIR  = INTERIM / "clean"         # Stage 3: typed + standardised Parquet
PANEL_DIR  = PROCESSED / "panel"       # Stage 4: transition panel
SAMPLE_DIR = PROCESSED / "samples"     # Stage 5: stratified samples

# Disposable DuckDB catalog — kept on the fast internal disk (repo reports/).
# parents[2] = repository root (config.py lives in <repo>/dev/pipeline/).
DUCKDB_PATH = Path(__file__).resolve().parents[2] / "reports" / "lake.duckdb"

# ---------------------------------------------------------------------------
# Resource knobs
# ---------------------------------------------------------------------------
DUCKDB_MEMORY_LIMIT = "6GB"   # SET memory_limit on every DuckDB connection
CHUNK_ROWS = 1_000_000        # row-group size / streaming batch size
ZSTD_LEVEL = 9                # Parquet zstd compression level


def require_drive() -> None:
    """Fail fast if the external SSD is not mounted (or not accessible).

    macOS note: a mounted drive can still be unreadable if the terminal app
    lacks the 'Removable Volumes' privacy permission — the error message
    covers both cases.
    """
    try:
        accessible = ROOT.is_dir() and any(True for _ in ROOT.iterdir())
    except PermissionError:
        accessible = False
    if not accessible:
        raise SystemExit(
            f"External drive not accessible at {ROOT} — connect 'SSD Felipe' "
            "and retry. If it is connected, grant this terminal access to "
            "removable volumes (System Settings > Privacy & Security > "
            "Files and Folders)."
        )


def ensure_dirs() -> None:
    """Create the writable directory tree (idempotent).

    Deliberately excludes RAW / RAW_DIR: raw/ is immutable and must already
    exist with the original CSVs — this function never creates or writes it.
    """
    require_drive()
    for d in (INTERIM, PROCESSED, MODELS, OUTPUTS, LOGS,
              PERF_DIR, CLEAN_DIR, PANEL_DIR, SAMPLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
