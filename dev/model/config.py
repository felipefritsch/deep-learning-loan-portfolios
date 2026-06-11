"""Phase 2-3 model configuration — the rolling backtest windows + export knobs.

Single source of truth for the modelling code (``dev/model/``), layered on top of
the pipeline's ``config.py`` (which owns ``ROOT`` and the lake paths). Everything
storage-related still derives from the pipeline ``ROOT`` on the external SSD.

What lives here
---------------
* **The 11 rolling expanding-window splits** (``00_OVERVIEW §3.2``): one window per
  test year ``k = 2015 … 2025``. Splits are stated on *label* months; the
  implementing masks are the strict ``period_ym`` ranges below (the label is one
  month ahead of the feature month — ``01_SCHEMA §6.1`` leakage rule).
* **Export knobs** (``02_LOAN_LEVEL §3``): ``P_KEEP_CURRENT`` (current→current
  thinning), the fixed never-thinned ``EVAL_SHARD_LT`` block + its label-month
  floor, and the dev-scale loan cap.
* Re-exported pipeline paths/helpers so model scripts import once from here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# The pipeline package owns ROOT and the lake layout. This module is itself named
# ``config``, so a plain ``import config`` would resolve to *this* file — load the
# pipeline config from its path under a distinct name to avoid the collision. Also
# put the pipeline dir on sys.path so siblings can ``import schema`` (the pipeline
# feature spec), which does not import config and so is collision-free.
_PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

_spec = importlib.util.spec_from_file_location("pipeline_config", _PIPELINE / "config.py")
pipeline_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline_config)

# Re-exported pipeline single-source-of-truth handles.
ROOT = pipeline_config.ROOT
PROCESSED = pipeline_config.PROCESSED
MODELS = pipeline_config.MODELS
OUTPUTS = pipeline_config.OUTPUTS
N_SHARDS = pipeline_config.N_SHARDS
DUCKDB_MEMORY_LIMIT = pipeline_config.DUCKDB_MEMORY_LIMIT
ZSTD_LEVEL = pipeline_config.ZSTD_LEVEL
require_drive = pipeline_config.require_drive

# Derived paths.
PANEL_GLOB = f"{pipeline_config.PANEL_DIR}/acq_quarter=*/part.parquet"
MACRO_DIR = PROCESSED / "macro"
TRAINING_DIR = PROCESSED / "training"      # processed/training/<design>/

# ---------------------------------------------------------------------------
# Rolling expanding-window backtest (00_OVERVIEW §3.2, 02_LOAN_LEVEL §1)
# ---------------------------------------------------------------------------
# One window per test year k. For window k (a label is one month ahead of its
# feature month, so masks bound the FEATURE month period_ym):
#   train : labels ≤ Dec(k−2)  ⟺  period_ym <  Dec(k−2)          (expanding from 2000)
#   val   : labels in year k−1 ⟺  period_ym ∈ [Dec(k−2), Dec(k−1))   (early stopping only)
#   test  : labels in year k   ⟺  period_ym ∈ [Dec(k−1), Dec(k))
# The three ranges tile the period_ym axis and a loan-month's role shifts across
# windows (test in k → train in k+2), which is why export.py materialises ONE
# shared train pool + ONE eval pool and masks are applied at load time.
TEST_YEARS = list(range(2015, 2026))       # k = 2015 … 2025 (11 windows; 2025 partial)
TUNING_YEAR = 2015                         # architecture/HP selection happens here only
PANEL_PERIOD_MIN = 200001                  # expanding-train lower bound (panel start month)


def _dec(year: int) -> int:
    """December of ``year`` as a ``YYYYMM`` integer (the split boundary month)."""
    return year * 100 + 12


def train_bounds(k: int) -> tuple[int, int]:
    """``[lo, hi)`` period_ym range for window ``k``'s train slice (labels ≤ Dec(k−2))."""
    return (PANEL_PERIOD_MIN, _dec(k - 2))


def val_bounds(k: int) -> tuple[int, int]:
    """``[lo, hi)`` period_ym range for window ``k``'s val slice (labels in year k−1)."""
    return (_dec(k - 2), _dec(k - 1))


def test_bounds(k: int) -> tuple[int, int]:
    """``[lo, hi)`` period_ym range for window ``k``'s test slice (labels in year k)."""
    return (_dec(k - 1), _dec(k))


def sql_period_mask(col: str, bounds: tuple[int, int]) -> str:
    """SQL predicate selecting ``col`` (a period_ym column) inside ``[lo, hi)``."""
    lo, hi = bounds
    return f"{col} >= {lo} AND {col} < {hi}"


def label_year(period_ym: int) -> int:
    """Calendar year of the one-month-ahead label for a feature month ``period_ym``."""
    return period_ym // 100 + 1 if period_ym % 100 == 12 else period_ym // 100


# period_ym → label month (YYYYMM), the +1-calendar-month shift used in the export
# and in the eval-pool floor. Dec rolls to the next January (+89 in YYYYMM units).
LABEL_YM_SQL = "(CASE WHEN {c} % 100 = 12 THEN {c} + 89 ELSE {c} + 1 END)"

# ---------------------------------------------------------------------------
# Export construction knobs (02_LOAN_LEVEL §3)
# ---------------------------------------------------------------------------
# Train pool: keep ALL rows with state_next≠current OR state≠current; keep
# current→current rows with this probability and attach weight = 1/p_keep so the
# loss stays an unbiased estimate of the true conditional probabilities
# (00_OVERVIEW §6.4). Thinning is a deterministic hash of (loan, period_ym).
P_KEEP_CURRENT = 0.05

# Eval pool: NEVER thinned; a fixed, loan-disjoint block of whole loans (selected by
# shard) carrying every loan-month whose label month is ≥ this floor. The block is
# fixed forever so every window's val/test slice is identical across all models.
EVAL_SHARD_LT = round(0.20 * N_SHARDS)     # shard < 51  (≈ 20% of loans), full scale
EVAL_LABEL_YM_MIN = 201401                 # labels ≥ 2014-01 (covers k=2015 val onward)

# The 4 transient origin states; terminal states never originate a transition.
ORIGIN_STATES = ("current", "dpd_30", "dpd_60", "dpd_90plus")

# ---------------------------------------------------------------------------
# Export variants — dev (this milestone) vs full (M10). The dev variant caps loans
# to a small shard band so the whole export streams in seconds and the train pool
# lands at the spec's ~5–10 M rows; loan-level (whole-shard) capping is MCAR, so it
# does not distort conditional probabilities (only current→current thinning needs a
# weight). `dev_shard_lt` = None ⇒ full population.
VARIANTS: dict[str, dict] = {
    "dev":  {"dev_shard_lt": 6},     # shards 0–5 ⇒ train ≈ 7 M, eval ≈ 45 M (label ≥ 2014)
    "full": {"dev_shard_lt": None},  # all 256 shards (train ~hundreds of M) — M10
}
