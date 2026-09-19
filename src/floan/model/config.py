"""Phase 2-3 model configuration — the rolling backtest windows + export knobs.

Single source of truth for the modelling code (``src/floan/model/``), layered on top of
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

# The pipeline package owns ROOT and the lake layout; import its config under a
# distinct name (no collision now that this is floan.model.config) so model
# scripts get both layers from this single module.
from floan.pipeline import config as pipeline_config

# Re-exported pipeline single-source-of-truth handles.
ROOT = pipeline_config.ROOT
REPO_ROOT = pipeline_config.REPO_ROOT
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
# fixed forever (per variant) so every window's val/test slice is identical across all
# models. 02 §3.2's design-intent block is 20% of loans (≈51 shards); the spec also
# permits a SMALLER fixed block when the full eval pool is intractable to upload/score.
# The operative per-variant size is `eval_shard_lt` in VARIANTS below — the full export
# uses a 12-shard (≈4.7%) block, still ~7.5 M unthinned loan-months per test year (an NLL
# standard error ~20× below the NN-vs-logit gap; see the M10a notes / PROMPTS.md).
EVAL_SHARD_DESIGN_INTENT = round(0.20 * N_SHARDS)   # 51 shards (≈20%) — reference only
EVAL_LABEL_YM_MIN = 201401                          # labels ≥ 2014-01 (covers k=2015 val onward)

# The 4 transient origin states; terminal states never originate a transition.
ORIGIN_STATES = ("current", "dpd_30", "dpd_60", "dpd_90plus")

# Pricing horizons (ECONOMIC_ENGINE §2/§5, task M21): H ∈ {1,3,6,12} months, produced as
# SNAPSHOTS of a single roll to max(HORIZONS) — the intermediates are free. H=1 is the direct
# one-step prediction and must equal evaluate.py exactly (the engine's regression guard).
HORIZONS = (1, 3, 6, 12)

# Calibration decision (ECONOMIC_ENGINE §6 / §8, task M23). roll_forward_predictor applies an
# optional pool.Calibrator to the RAW per-origin scores BEFORE assembly (§4 item 2); this records
# whether it is enabled and the basis. The DECISION is made on a ~20% subsample at H=1 (the clean
# calibration test, §5); if APPLIED, per-window temperatures are refit at full scale in M24
# ("subsample the fit, never the reported numbers"). Evidence: src/floan/model/M23_NOTES.md.
CALIBRATION = {
    "method": "temperature",            # one scalar per window (06 §4); primary method
    "fallback": "isotonic_per_class",   # per-class isotonic + row renormalise (06 §4 fallback)
    "applied": False,                   # filled from the M23 decision (see M23_NOTES.md)
    "basis": "M23 H=1 reliability, ~20% subsample @ k=2020 (CPU)",
    "refit_at_scale": "M24",
}

# ---------------------------------------------------------------------------
# Export variants — dev (M4–M9 tuning) vs full (M10). Each variant fixes a train-pool
# loan block `train_shard_lt` (shards [0, train_shard_lt)) and a never-thinned eval
# block `eval_shard_lt` ⊆ the train block (eval loans live in the train pool, separated
# only by the period_ym mask). Loan-level (whole-shard) capping is MCAR, so it does not
# distort conditional probabilities — only current→current thinning needs the 1/p_keep
# weight. Per-shard yield ≈ 1.16 M train rows and ≈ 7.55 M eval rows (eval unthinned,
# hence ~6.5× denser per shard).
#
# Full sizing (M10a): a 25% loan sample (64 shards) at P_KEEP_CURRENT=0.05 lands the
# train pool at ≈74 M rows — central in 02 §3's 50–100 M target, ~10.6× the dev pool,
# so the k=2015 tuning slice grows 3.15 M → ≈34 M (a decisive re-test of M9's
# scale-sensitive depth-3-vs-5 call). The eval block is held to 12 shards for a tractable
# upload/scoring footprint (≈90 M rows, ~0.75 GB); full-population eval (51 shards) would
# be ≈385 M rows / 3.2 GB and score 11×+ensemble for no CI gain.
VARIANTS: dict[str, dict] = {
    "dev":  {"train_shard_lt": 6,  "eval_shard_lt": 6},   # train ≈7 M,  eval ≈45 M (label ≥ 2014)
    "full": {"train_shard_lt": 64, "eval_shard_lt": 12},  # train ≈74 M, eval ≈90 M — M10
}

# ---------------------------------------------------------------------------
# Frozen NN config (02_LOAN_LEVEL §6) — depth/dropout/L2 selection (M9 → M10b)
# ---------------------------------------------------------------------------
# The pruned grid (grid.py: depth×dropout plane at L2=0 + an L2 sweep at the anchor,
# 14 cells) ran once on the tuning window k=2015 (DEV export, ~3.15 M train rows) and
# selected depth 3, dropout 0.2 on that window's VAL NLL — shallower than the paper's
# 5 layers (Sirignano et al. fit billions of loan-months; on the dev slice the deeper net
# overfits and only a small L2 *rescues* it back to depth-3's level). M9 flagged this as
# **scale-sensitive** and deferred a full-scale re-check to M10b.
#
# M10b full-scale re-check (k=2015 FULL export, 33.5 M train rows = 10.6× dev): re-fit
# depth 3 vs depth 5 at dropout 0.2, ±L2 1e-5, on the GPU-resident fast path, selecting on
# this window's val NLL. Result — **depth 3 still wins, decisively over depth 5** (the
# selection is NOT a dev-scale artefact):
#     d3 do0.2 wd1e-5 : val 0.095723  test 0.103125   <- val argmin (frozen)
#     d3 do0.2 wd0    : val 0.095746  test 0.103091
#     d5 do0.2 wd0    : val 0.095912  test 0.103142
#     d5 do0.2 wd1e-5 : val 0.096020  test 0.102970
# Depth gap d5−d3 = +1.66e-4 val (robust: ~7× the within-depth-3 L2 spread). The L2
# sub-choice IS within noise — wd=1e-5 beats wd=0 by only 2.3e-5 on val (and test marginally
# favours wd=0) — but the stated protocol is "take the val-NLL winner", so the frozen config
# is the strict argmin: depth 3, dropout 0.2, L2 1e-5. A small L2 is the mild-regularisation
# direction M9 already noted; the choice is immaterial to performance either way.
#
# backtest.py (M10) freezes THIS config and loops it over all 11 windows (per-window early
# stopping / scalers / vocab — never re-tuned on a test slice, §6 protocol). Optimization
# hyperparameters (lr, batch, patience, schedule) were not searched — paper-anchored M8
# values in train.py. The ensemble (paper Fig 7) is 8 nets at this SAME config, so the
# ensemble-vs-single delta is an architecture-held-constant comparison.
#
# Evidence: models/nn/dev/grid_summary.json (dev 14-cell ranking); the four full-scale cells
# models/nn/full/k2015_d{3,5}_do0.2_wd{0,1e-05}/metrics.json; logs/m10b/depth_check.log.
NN_SELECTED = {"depth": 3, "dropout": 0.2, "weight_decay": 1e-5}  # k=2015 FULL-scale val argmin (M10b)
NN_ENSEMBLE_MEMBERS = 8                                           # paper Fig 7

# ---------------------------------------------------------------------------
# Frozen GBT config (06_GBT_BASELINE §3) — LightGBM multiclass softmax (M16)
# ---------------------------------------------------------------------------
# The pruned dev-scale grid (gbt.tune: a min_sum_hessian calibration → num_leaves×lr plane
# → feature_fraction/lambda_l2 sweeps, 13 cells) ran once on the tuning window k=2015 (DEV
# export, 3.15 M train rows) and selected on that window's 2014 VAL multi_logloss.
#
# The decisive axis was min_sum_hessian_in_leaf, exactly as 06 §3 anticipated: the HT
# importance weights (Σw=31.1 M over 3.15 M rows ⇒ mean ≈ 9.9) inflate the *summed* leaf
# Hessians, so LightGBM's default 1e-3 is miscalibrated and overfits within ~3 rounds. The
# stage-1 calibration sweep is monotone in the fix —
#     min_sum_hessian_in_leaf  1e-3    1e-1    1       10       100
#     val_nll                  0.1390  0.1361  0.1296  0.10173  0.09871
# — so the plane ran at min_sum_hessian=100, where num_leaves=31, lr=0.05 won (shallow trees
# + heavy leaf regularisation generalise best under the weighted objective):
#     val_nll 0.097488  test_nll 0.104486  best_iteration 291
# feature_fraction=0.7 and lambda_l2=1 did not beat the anchor (0.098265 / 0.097580).
#
# M16 Accept (all pass, zone=HEALTHY): probs sum to 1 / no NaN; LightGBM multi_logloss ==
# evaluate._nll to 9e-16; base-rate QA max|Δ|=0.0050<0.01; impossible-cell mass 3.6e-4<1e-3;
# dev test NLL 0.104486 sits below the empirical floor (0.11266) and the bucketed matrix
# (0.10925)/logit (0.10943) — between the bucketed matrix and the best dev NN (0.10319), a
# credible flexible learner the net still edges by ~1.3e-3 here (a 06 §7 finding, reported
# not tuned). best_iteration is a reference; M17 re-runs early stopping per window.
# Evidence: models/gbt/dev/k2015/metrics.json.
#
# M17 full-scale reconfirm (k=2015 FULL export, 33.5 M train rows = 10.6× dev) — the M10b
# depth-check analogue. min_sum_hessian_in_leaf is an ABSOLUTE summed-leaf-Hessian threshold,
# and the full slice carries ~10× the summed Hessian of the dev slice, so the dev-calibrated
# msh=100 is ~10× too weak at scale. Re-checking the frozen cell vs scaled-up neighbours on
# the 2014 VAL NLL:
#     msh=100  (dev winner)   val 0.097324  test 0.103798  best_iter 951
#     msh=1000 (×10)          val 0.096968  test 0.103597  best_iter 1208   <- val argmin (FROZEN)
#     nl63 @ msh=100          val 0.097273  test —          best_iter 905
# The winner MOVES to min_sum_hessian_in_leaf=1000 (better on val AND test; the capacity
# neighbour nl63 did not move it) — exactly the scale-shift the M10b protocol catches, so the
# dev config is NOT forced through. test 0.103597 is still HEALTHY (below the full empirical
# floor 0.11287 / bucketed 0.10939 / logit 0.10825) and ~ties the full NN (0.103125, +5e-4).
# Evidence: models/gbt/full/k2015_reconfirm/metrics.json; backtest.py loops THIS config.
GBT_SELECTED = {"num_leaves": 31, "learning_rate": 0.05, "min_sum_hessian_in_leaf": 1000.0,
                "feature_fraction": 1.0, "lambda_l2": 0.0}  # k=2015 FULL-scale val argmin (M17 reconfirm)
GBT_TUNING_BEST_ITERATION = 1208                            # reference (full msh=1000); per-window early stopping in M17
