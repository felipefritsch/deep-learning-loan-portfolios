"""M16/M17 — GBT baseline trainer (``06_GBT_BASELINE §3``): one 7-class softmax LightGBM.

The non-neural flexible learner of the loan-level comparison — added beside the net, not
as a new nested rung (``06 §1``). A **single** multiclass-softmax booster with origin
``state`` as a native categorical input (``06 §2.1``), so a net-vs-GBT gap is attributable
to the *learner family* (smooth compositions vs axis-aligned partitions) and not to a
different problem decomposition. Everything else is the existing machinery:

* the **same** ``train_pool`` / ``eval_pool`` and window masks the net reads (``data.py``),
* the **same** ``features.py`` feature set **minus standardisation** (``06 §2.2``): raw
  continuous (NaN-native — LightGBM handles missing internally), the UNK-routed vocab codes
  for the categoricals, and the binary missingness indicators passed so the columns match;
* the **same** HT importance weights ``1/p_keep`` on the **train** Dataset only — val/test
  are the never-thinned eval pool, so they carry ``w≡1`` and enter unweighted (``06 §2.3``);
* the **same** out-of-sample NLL object — ``evaluate._nll`` — for the headline metric, and
  the **same** ``backtest._structural_allow`` impossible-cell mask for the §7 QA.

The feature matrix layout is fixed: ``[ raw continuous ‖ categorical codes ‖ binary ]``,
columns named so feature importance is legible and the per-origin ``state`` override (M19)
can address one column. CPU only — a documented deviation from the GPU default (``06 §2.4``).

Three drivers, all on this shared loader + QA:
* ``run`` (M16) — the dev-scale tuning grid on the tuning window (k=2015), selects on the
  2014 val NLL, freezes the winner (``config.GBT_SELECTED``).
* ``reconfirm`` (M17) — the M10b depth-check analogue: re-check the frozen config at **full**
  export scale on the tuning window (GBT capacity can scale-shift, mostly via the
  absolute-summed-Hessian threshold ``min_sum_hessian_in_leaf``). Update the frozen config if
  the winner moves.
* ``fit_window_frozen`` (M17) — one frozen-config fit per window (per-window vocab, HT-weighted
  train, early stopping on that window's val), the unit ``backtest.py`` loops over the 11
  windows. Idempotent on (window, config).

Run (CPU; needs the SSD + the export):
    python -m floan.model.gbt                                  # M16 dev k=2015 tuning
    python -m floan.model.gbt --mode frozen --variant full --k 2015   # one frozen fit
    python -m floan.model.gbt --mode reconfirm --variant full --k 2015 # full-scale re-check
    python -m floan.model.gbt --max-rows 100000               # quick wiring smoke
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from floan.model import backtest as B   # _structural_allow — the §7 monotone-delinquency mask
from floan.model import config
from floan.model import data as D
from floan.model import evaluate as E    # _nll — the headline out-of-sample NLL object (the cross-check target)
from floan.model import features as F
from floan.model import train as T       # _git_commit — shared run-folder provenance

VARIANT_DEFAULT = "dev"
TUNING_K = config.TUNING_YEAR          # 2015 — architecture/HP selection happens here only
SEED = 0
EARLY_STOP_ROUNDS = 50                 # boosting-round patience (≈ the net's eval patience, §3)
MAX_ROUNDS = 3000                      # cap; early stopping selects best_iteration well below
MAX_BIN = 255                          # §3 (drop to 63 if RAM tight; ample here)

# §7 QA tolerances — identical thresholds to backtest.base_rate_qa.
BASE_RATE_TOL = 0.01
IMPOSSIBLE_TOL = 1e-3
SUM_TO_ONE_TOL = 1e-5
XCHECK_TOL = 1e-6                       # LightGBM multi_logloss vs evaluate._nll (06 §3)

# The 5 sweepable hyperparameters that make up a frozen GBT config (config.GBT_SELECTED).
CFG_KEYS = ("num_leaves", "learning_rate", "min_sum_hessian_in_leaf",
            "feature_fraction", "lambda_l2")


# ===========================================================================
# Feature matrix — the net's columns MINUS standardisation (06 §2.2)
# ===========================================================================
def feature_layout(vocab: F.Vocab) -> tuple[list[str], list[int]]:
    """``(feature_name, categorical_index)`` for the ``[cont ‖ cat ‖ bin]`` matrix. Names
    are ``F.CONTINUOUS`` then the vocab's categorical columns (``state`` first, §2.1) then
    ``F.BINARY``; the categorical indices are the middle block (what LightGBM treats
    natively)."""
    names = list(F.CONTINUOUS) + list(vocab.cols) + list(F.BINARY)
    n_cont = len(F.CONTINUOUS)
    cat_idx = list(range(n_cont, n_cont + len(vocab.cols)))
    return names, cat_idx


def build_X(df: pl.DataFrame, vocab: F.Vocab) -> np.ndarray:
    """``[N, d]`` float32 design matrix ``[ raw continuous ‖ vocab codes ‖ binary ]``.

    Continuous columns are the **raw** values (no scaler, no log) with nulls left as NaN so
    LightGBM routes them via its native missing-value handling — trees are scale- and
    monotone-invariant, so this is the byte-identical feature *content* the net sees, only
    without the standardisation the net needs (``06 §2.2``). Categoricals are the
    ``Vocab.transform`` integer codes (UNK=0 for unseen/null, identical routing to the net);
    they are stored as floats here but declared categorical to LightGBM via ``cat_idx`` so it
    casts them back to int and partitions natively. Binary indicators pass straight through."""
    cont = df.select([pl.col(c).cast(pl.Float32) for c in F.CONTINUOUS]).to_numpy()
    cat = vocab.transform(df).astype(np.float32)        # [N, n_cat] in vocab.cols order
    binb = F.binary_matrix(df)                          # [N, n_bin] float32 0/1
    return np.ascontiguousarray(np.concatenate([cont, cat, binb], axis=1), dtype=np.float32)


def _origin_index(df: pl.DataFrame) -> np.ndarray:
    """Per-row origin **class** index (0..6) from the ``state`` column — for the §7 QA."""
    return (df.select(pl.col("state").replace_strict(
        list(F.STATE_INDEX), list(F.STATE_INDEX.values()),
        default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1))


def load_split(variant: str, k: int, split: str, vocab: F.Vocab,
               max_rows: int | None = None) -> dict:
    """Encode one window/split into arrays (+ its period_ym range, for the leakage probe).
    ``vocab`` is the window's fitted vocab (fit on the train slice, applied to every split —
    no leakage). Train carries the HT ``weight``; val/test are the unthinned eval pool
    (``w≡1``)."""
    pool_dir, bounds = D.window_spec(variant, k, split)
    df = F.prepare_raw(D._masked_scan(pool_dir, bounds).collect())
    if max_rows is not None:
        df = df.head(max_rows)
    ym = df.get_column("period_ym")
    return {
        "X": build_X(df, vocab),
        "y": F.target_indices(df).astype(np.int32),
        "w": df.get_column(F.WEIGHT_COL).cast(pl.Float32).to_numpy(),
        "origin": _origin_index(df),
        "n": df.height,
        "ym_min": int(ym.min()) if df.height else None,
        "ym_max": int(ym.max()) if df.height else None,
    }


def load_window(variant: str, k: int, max_rows: int | None = None) -> dict:
    """Load the train slice, **fit the per-window vocab on it only** (no leakage), and encode
    all three splits. Returns the design arrays + vocab + feature layout + train metadata. The
    train design matrix ``X_tr`` is returned raw so the caller decides whether to free it after
    LightGBM binning (the M17 frozen fit does, to keep peak RAM low at full scale)."""
    tr_dir, tr_bounds = D.window_spec(variant, k, "train")
    df_tr = F.prepare_raw(D._masked_scan(tr_dir, tr_bounds).collect())
    if max_rows is not None:
        df_tr = df_tr.head(max_rows)
    vocab = F.Vocab.fit(df_tr, min_count=1)            # same min_count as data.fit_window
    names, cat_idx = feature_layout(vocab)
    X_tr = build_X(df_tr, vocab)
    y_tr = F.target_indices(df_tr).astype(np.int32)
    w_tr = df_tr.get_column(F.WEIGHT_COL).cast(pl.Float32).to_numpy()
    ym = df_tr.get_column("period_ym")
    meta = {"n_train": df_tr.height, "train_sum_w": float(w_tr.sum()),
            "train_ym_min": int(ym.min()), "train_ym_max": int(ym.max()),
            "cont_nan_cells": int(np.isnan(X_tr[:, :len(F.CONTINUOUS)]).sum())}
    del df_tr
    val = load_split(variant, k, "val", vocab, max_rows)
    test = load_split(variant, k, "test", vocab, max_rows)
    return {"vocab": vocab, "names": names, "cat_idx": cat_idx,
            "X_tr": X_tr, "y_tr": y_tr, "w_tr": w_tr, "meta": meta,
            "val": val, "test": test}


# ===========================================================================
# Tuning grid (06 §3) — num_leaves × lr plane, calibrate min_sum_hessian first
# ===========================================================================
def _base_params(n_threads: int) -> dict:
    """Fixed objective + the anchor of the sweepable hyperparameters (§3). Softmax outputs
    sum to one (required by the NLL headline + the pool roll-forward — ``multiclassova`` is
    wrong here). ``deterministic`` + ``force_row_wise`` make the CPU fit reproducible."""
    return dict(
        objective="multiclass", num_class=F.N_CLASSES, metric="multi_logloss",
        num_leaves=63, learning_rate=0.1, min_sum_hessian_in_leaf=1e-3,
        feature_fraction=1.0, lambda_l2=0.0,
        max_bin=MAX_BIN, seed=SEED, deterministic=True, force_row_wise=True,
        num_threads=n_threads, verbosity=-1,
    )


def _cell_label(ov: dict) -> str:
    return "_".join(f"{k}{ov[k]:g}" for k in sorted(ov))


def _train_cell(dtrain: lgb.Dataset, dval: lgb.Dataset, params: dict) -> lgb.Booster:
    """One LightGBM fit with early stopping on the val multi_logloss."""
    return lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                     valid_sets=[dval], valid_names=["val"],
                     callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)])


def tune(dtrain: lgb.Dataset, dval: lgb.Dataset, n_threads: int) -> tuple[lgb.Booster, dict, list]:
    """Pruned dev-scale grid (``06 §3``), calibrate-then-plane.

    The HT importance weights inflate the *summed* leaf Hessians by ~mean(weight), so
    LightGBM's default ``min_sum_hessian_in_leaf=1e-3`` is badly miscalibrated to this
    weighted objective (``06 §3`` flags exactly this) — at the default the trees split on
    near-zero effective support and overfit within a handful of rounds. So **stage 1
    calibrates ``min_sum_hessian_in_leaf`` first** (at a reference ``num_leaves``/``lr``),
    **stage 2** runs the ``num_leaves × learning_rate`` plane in that good regime, and
    **stage 3** sweeps ``feature_fraction`` / ``lambda_l2`` at the plane winner. Selected on
    the 2014 val multi_logloss (never a test slice). Returns ``(winner_booster,
    selected_overrides, all_results)``."""
    base = _base_params(n_threads)
    results: list[dict] = []
    best = {"val_nll": float("inf"), "booster": None, "overrides": None}

    def run_cell(overrides: dict) -> float:
        booster = _train_cell(dtrain, dval, {**base, **overrides})
        vnll = float(booster.best_score["val"]["multi_logloss"])
        results.append({"label": _cell_label(overrides), "overrides": overrides,
                        "val_nll": vnll, "best_iteration": int(booster.best_iteration)})
        print(f"  {_cell_label(overrides):<46} val_nll={vnll:.6f}  best_iter={booster.best_iteration}")
        if vnll < best["val_nll"]:
            best.update(val_nll=vnll, booster=booster, overrides=dict(overrides))
        return vnll

    # Stage 1 — calibrate min_sum_hessian_in_leaf (the §3-flagged axis) at a reference cell.
    print("[grid] stage 1 — min_sum_hessian_in_leaf calibration (06 §3; HT weights inflate it)")
    ref = {"num_leaves": 63, "learning_rate": 0.1}
    for msh in (1e-3, 1e-1, 1.0, 10.0, 100.0):
        run_cell({**ref, "min_sum_hessian_in_leaf": msh})
    msh_star = best["overrides"]["min_sum_hessian_in_leaf"]

    # Stage 2 — num_leaves × learning_rate plane in the calibrated-Hessian regime.
    print(f"[grid] stage 2 — num_leaves × learning_rate plane at min_sum_hessian={msh_star:g}")
    for nl in (31, 63, 127):
        for lr in (0.05, 0.1):
            run_cell({"num_leaves": nl, "learning_rate": lr, "min_sum_hessian_in_leaf": msh_star})
    anchor = dict(best["overrides"])     # winning {num_leaves, learning_rate, min_sum_hessian}

    # Stage 3 — feature_fraction / lambda_l2 sweeps anchored at the plane winner.
    print(f"[grid] stage 3 — regularisation sweeps at {_cell_label(anchor)}")
    run_cell({**anchor, "feature_fraction": 0.7})
    run_cell({**anchor, "lambda_l2": 1.0})

    return best["booster"], best["overrides"], results


def _full_cfg(overrides: dict, n_threads: int) -> dict:
    """The 5-key frozen config from a grid winner's overrides (base defaults fill the rest)."""
    merged = {**_base_params(n_threads), **overrides}
    return {kk: merged[kk] for kk in CFG_KEYS}


# ===========================================================================
# Reference band (the dev k=2015 sanity zones) — read live from the run folders
# ===========================================================================
def reference_band(variant: str, k: int) -> dict:
    """The §7 sanity band for the tuning window, read from the committed empirical/logit/NN
    run folders so it cannot go stale: ``empirical_floor`` and ``bucketed`` (benchmarks),
    ``logit`` and (if present) ``nn_best`` test NLLs. The GBT test NLL is read against these
    three zones — hard-stop above the floor, yellow between bucketed and floor, healthy
    at/under bucketed (M16 Accept)."""
    band: dict = {}
    bench = config.MODELS / "benchmarks" / variant / "metrics.json"
    if bench.exists():
        pw = json.loads(bench.read_text()).get("per_window", {}).get(str(k), {})
        band["empirical_floor"] = pw.get("nll_empirical")
        band["bucketed"] = pw.get("nll_bucketed")
    logit = config.MODELS / "logit" / variant / f"k{k}" / "metrics.json"
    if logit.exists():
        band["logit"] = json.loads(logit.read_text()).get("logit", {}).get("test_nll")
    for tag in (f"k{k}_d3_do0.2_wd0", f"k{k}_d3_do0.2_wd1e-05", f"k{k}_d5_do0.2_wd0"):
        nn = config.MODELS / "nn" / variant / tag / "metrics.json"
        if nn.exists():
            band["nn_best"] = json.loads(nn.read_text()).get("eval", {}).get("test_nll")
            break
    return band


def classify_zone(test_nll: float, band: dict) -> str:
    floor, bucketed, logit = band.get("empirical_floor"), band.get("bucketed"), band.get("logit")
    if floor is not None:
        if test_nll >= floor:
            return "HARD_STOP"      # worse than the covariate-free floor ⇒ a wiring bug
        if bucketed is not None and test_nll > bucketed:
            return "YELLOW"         # between bucketed matrix and floor ⇒ review wiring
        return "HEALTHY"            # at/under the bucketed matrix
    # Fallback when the covariate-free benchmark isn't computed at this scale (e.g. full has
    # only logit/NN): gate against the logit — a covariate model worse than the linear baseline
    # is suspect (wiring), at/under it is sane.
    if logit is not None:
        return "HARD_STOP" if test_nll >= logit else "HEALTHY"
    return "HEALTHY"


# ===========================================================================
# Predict adapter (06 §5) — the (n, 7) array every downstream consumer expects
# ===========================================================================
def predict_proba(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """``[N, 7]`` class probabilities at the booster's best iteration. LightGBM multiclass
    returns softmax-normalised rows (they sum to one), which is exactly the array
    ``evaluate.py`` / ``pool.py`` consume from every other model."""
    return booster.predict(X, num_iteration=booster.best_iteration)


# ===========================================================================
# Accept / QA — identical checks for the M16 winner and every M17 per-window fit
# ===========================================================================
def compute_accept(booster: lgb.Booster, win: dict, variant: str, k: int, *, label: str) -> dict:
    """Run the §7 QA + sanity band on a fitted booster and PRINT the evidence. Shared by the
    M16 tuning winner and every M17 per-window fit, so a window's run folder always carries
    the same self-describing evidence. Returns the metric sub-dicts assembled into the run
    folder's ``metrics.json``."""
    names, cat_idx, vocab = win["names"], win["cat_idx"], win["vocab"]
    val, test, meta = win["val"], win["test"], win["meta"]

    proba_val = predict_proba(booster, val["X"])
    proba_test = predict_proba(booster, test["X"])

    # (1) probabilities sum to one / no NaN
    sum_dev = float(np.abs(proba_test.sum(axis=1) - 1.0).max())
    any_nan = bool(np.isnan(proba_test).any())
    a1 = (sum_dev < SUM_TO_ONE_TOL) and not any_nan

    # (2) LightGBM multi_logloss == evaluate._nll on the same (n,7) preds (~1e-6, 06 §3)
    lgb_val = float(booster.best_score["val"]["multi_logloss"])
    eval_val = E._nll(proba_val, val["y"].astype(np.int64))
    xdiff = abs(lgb_val - eval_val)
    a2 = xdiff <= XCHECK_TOL

    # (3) base-rate QA — predicted mean class rates ≈ realized (importance-weighting check)
    realized = np.bincount(test["y"], minlength=F.N_CLASSES) / test["n"]
    pred_mean = proba_test.mean(axis=0)
    br_max = float(np.abs(realized - pred_mean).max())
    a3 = br_max < BASE_RATE_TOL

    # (4) structural-impossible-cell mass < 0.1% (backtest._structural_allow)
    allow = B._structural_allow()
    forbidden = ~allow[test["origin"]]
    imp_mass = (proba_test * forbidden).sum(axis=1)
    realized_imp = forbidden[np.arange(test["n"]), test["y"]]
    imp_mean = float(imp_mass.mean())
    a4 = imp_mean < IMPOSSIBLE_TOL

    # (5) out-of-sample test NLL vs the sanity band (correctness zones, not a target)
    test_nll = E._nll(proba_test, test["y"].astype(np.int64))
    band = reference_band(variant, k)
    zone = classify_zone(test_nll, band)
    # Which baseline classify_zone gated against — the real empirical floor (the bug
    # tripwire) or the logit fallback when benchmarks/<variant> isn't computed at this scale.
    zone_basis = ("empirical_floor" if band.get("empirical_floor") is not None
                  else "logit_fallback" if band.get("logit") is not None else "none")
    a5 = zone != "HARD_STOP"

    # feature-importance glance — flag high-card (zip3/MSA) native-categorical dominance
    gain = booster.feature_importance(importance_type="gain").astype(float)
    g_order = np.argsort(gain)[::-1]
    top_gain = [{"feature": names[i], "gain": float(gain[i])} for i in g_order[:15]]
    cat_cards = sorted(zip(vocab.cols, vocab.vocab_sizes), key=lambda t: -t[1])[:2]
    highcard_cols = [c for c, _ in cat_cards]
    gtot = float(gain.sum()) or 1.0
    highcard_share = float(sum(gain[names.index(c)] for c in highcard_cols) / gtot)
    top1_highcard = names[g_order[0]] in highcard_cols

    # leakage probe (no vocab leakage, single-window): the top-cardinality categorical routes
    # a positive fraction of val/test rows to UNK (=0) ⇒ those levels were unseen at fit time,
    # i.e. the vocab was fit on the train slice ONLY; and the three masks are period-disjoint.
    hc_col = max(zip(vocab.cols, vocab.vocab_sizes), key=lambda t: t[1])[0]
    hc_pos = cat_idx[vocab.cols.index(hc_col)]
    val_unk = float((val["X"][:, hc_pos] == 0).mean())
    test_unk = float((test["X"][:, hc_pos] == 0).mean())
    disjoint = bool(meta["train_ym_max"] < val["ym_min"] <= val["ym_max"] < test["ym_min"])
    leakage = {"highcard_col": hc_col,
               "train_ym": [meta["train_ym_min"], meta["train_ym_max"]],
               "val_ym": [val["ym_min"], val["ym_max"]],
               "test_ym": [test["ym_min"], test["ym_max"]],
               "val_unk_rate": val_unk, "test_unk_rate": test_unk,
               "masks_disjoint": disjoint}

    overall = a1 and a2 and a3 and a4 and a5

    print(f"\n================= {label}: ACCEPT/QA EVIDENCE =================")
    print(f"[{'PASS' if a1 else 'FAIL'}] (1) probs sum to 1 (max|Σ−1|={sum_dev:.2e} < {SUM_TO_ONE_TOL:g}) "
          f"& no NaN (any_nan={any_nan})")
    print(f"[{'PASS' if a2 else 'FAIL'}] (2) LightGBM multi_logloss {lgb_val:.8f} == evaluate._nll "
          f"{eval_val:.8f}  |Δ|={xdiff:.2e} (≤{XCHECK_TOL:g})")
    print(f"[{'PASS' if a3 else 'FAIL'}] (3) base-rate QA  max|realized−pred|={br_max:.4f} < {BASE_RATE_TOL}")
    print(f"[{'PASS' if a4 else 'FAIL'}] (4) impossible-cell mass mean={imp_mean:.2e} < {IMPOSSIBLE_TOL:g}  "
          f"(realized {float(realized_imp.mean()):.2e}; max row {float(imp_mass.max()):.2e})")
    fl, bk, lg, nn = band.get("empirical_floor"), band.get("bucketed"), band.get("logit"), band.get("nn_best")
    print(f"[{'PASS' if a5 else 'FAIL'}] (5) test NLL={test_nll:.6f}  zone={zone} (basis={zone_basis})")
    if fl is not None and bk is not None:
        print(f"        band: empirical_floor={fl:.5f}  bucketed={bk:.5f}"
              + (f"  logit={lg:.5f}" if lg is not None else "")
              + (f"  nn_best={nn:.5f}" if nn is not None else ""))
    print(f"[leakage] vocab fit on train slice only: top-card '{hc_col}' UNK rate "
          f"val={val_unk:.3%} test={test_unk:.3%} (>0 ⇒ unseen levels routed to UNK); "
          f"masks disjoint train{leakage['train_ym']}<val{leakage['val_ym']}<test{leakage['test_ym']}={disjoint}")
    print(f"[glance] feature-importance: high-card {highcard_cols} gain share={highcard_share:.1%} "
          f"(top-1 feature is high-card: {top1_highcard}); top: "
          f"{', '.join(t['feature'] for t in top_gain[:6])}")
    print(f">>> {label}: {'ALL CRITERIA PASS' if overall else 'FAILURE — debug before advancing'}  (zone={zone})")
    print("=" * 62)

    return {
        "eval_val": eval_val, "test_nll": test_nll, "lgb_val": lgb_val,
        "crosscheck": {"lgb_val_multi_logloss": lgb_val, "evaluate_nll": eval_val,
                       "abs_diff": xdiff, "tol": XCHECK_TOL, "passed": a2},
        "probs_sanity": {"sum_to_one_max_abs_dev": sum_dev, "any_nan": any_nan, "passed": a1},
        "base_rate_qa": {"states": F.STATES, "realized": realized.tolist(),
                         "predicted_mean": pred_mean.tolist(), "max_abs_diff": br_max,
                         "tol": BASE_RATE_TOL, "passed": a3},
        "impossible": {"model_mean_mass": imp_mean, "model_max_row_mass": float(imp_mass.max()),
                       "realized_mean_rate": float(realized_imp.mean()),
                       "threshold": IMPOSSIBLE_TOL, "passed": a4},
        "band": {**band, "test_nll": test_nll, "zone": zone, "zone_basis": zone_basis,
                 "passed": a5},
        "feature_importance": {"top_gain": top_gain, "highcard_cols": highcard_cols,
                               "highcard_gain_share": highcard_share,
                               "top1_is_highcard": top1_highcard},
        "leakage": leakage,
        "accept": {"all_pass": overall,
                   "criteria": {"probs_valid": a1, "nll_crosscheck": a2, "base_rate_qa": a3,
                                "impossible_mass": a4, "band_not_hardstop": a5}},
    }


# ===========================================================================
# Provenance + run-folder writer
# ===========================================================================
def _environment() -> dict:
    import os
    return {"python": platform.python_version(), "lightgbm": lgb.__version__,
            "numpy": np.__version__, "polars": pl.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count()}


def _base_metrics(variant: str, k: int, win: dict, acc: dict, *, smoke: bool, wall: float) -> dict:
    """The common metrics.json body (provenance + features + rows + the QA from
    :func:`compute_accept`). Each driver adds its own ``selection`` / ``hyperparams`` block."""
    meta, vocab, val, test = win["meta"], win["vocab"], win["val"], win["test"]
    mpath = config.TRAINING_DIR / variant / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "lightgbm_multiclass_softmax", "variant": variant, "window_k": k,
        "tuning_window": (k == config.TUNING_YEAR), "smoke": smoke,
        "device": "cpu", "seed": SEED, "git_commit": T._git_commit(), "environment": _environment(),
        "source": {"train_pool_content_hash": manifest.get("train_pool", {}).get("content_hash"),
                   "eval_pool_content_hash": manifest.get("eval_pool", {}).get("content_hash"),
                   "panel_glob": config.PANEL_GLOB},
        "features": {"n_continuous": len(F.CONTINUOUS), "n_categorical": len(vocab.cols),
                     "n_binary": len(F.BINARY), "n_features": len(win["names"]),
                     "vocab_sizes": dict(zip(vocab.cols, vocab.vocab_sizes)),
                     "continuous_nan_cells_train": meta["cont_nan_cells"]},
        "rows": {"train": meta["n_train"], "train_sum_w": meta["train_sum_w"],
                 "val": val["n"], "test": test["n"]},
        "gbt": {"val_nll": acc["eval_val"], "test_nll": acc["test_nll"],
                "lgb_val_multi_logloss": acc["lgb_val"]},
        "crosscheck": acc["crosscheck"], "probs_sanity": acc["probs_sanity"],
        "base_rate_qa": acc["base_rate_qa"], "impossible_transitions": acc["impossible"],
        "band": acc["band"], "feature_importance": acc["feature_importance"],
        "leakage": acc["leakage"], "accept": acc["accept"], "wall_sec": wall,
    }


def write_run_folder(run_dir: Path, booster: lgb.Booster, vocab: F.Vocab,
                     names: list[str], cat_idx: list[int], metrics: dict) -> None:
    """Self-describing run folder (``00_OVERVIEW §6.6``): the booster, the window vocab, the
    feature layout (so the predict adapter / M19 seam can rebuild ``X``), and metrics.json.

    **Atomic** writes: every artifact goes to a ``.tmp`` sibling and is ``os.replace``d into
    place (an atomic rename on the SSD filesystem), with ``metrics.json`` renamed **last**.
    Because ``metrics.json`` is the (window, config) idempotency *completion marker*, this
    ordering guarantees it can only appear after the model + vocab + layout are fully on disk —
    an SSD disconnect mid-write leaves at most a stray ``.tmp`` (ignored), never a truncated
    ``metrics.json`` that a later run would mistake for a complete fit and skip."""
    run_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_text(path: Path, text: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)                       # atomic rename on the same filesystem

    model_tmp = run_dir / "model.txt.tmp"
    booster.save_model(str(model_tmp), num_iteration=booster.best_iteration)
    os.replace(model_tmp, run_dir / "model.txt")
    _atomic_text(run_dir / "vocab.json", json.dumps(vocab.to_dict(), indent=2))
    _atomic_text(run_dir / "feature_layout.json", json.dumps(
        {"feature_name": names, "categorical_index": cat_idx,
         "n_continuous": len(F.CONTINUOUS), "n_categorical": len(vocab.cols),
         "n_binary": len(F.BINARY)}, indent=2))
    _atomic_text(run_dir / "metrics.json", json.dumps(metrics, indent=2))   # LAST = completion marker


def run_dir_for(variant: str, k: int, smoke: bool = False) -> Path:
    return config.MODELS / "gbt" / variant / (f"k{k}" + ("_smoke" if smoke else ""))


# ===========================================================================
# M16 driver — dev-scale tuning grid on the tuning window
# ===========================================================================
def run(variant: str, k: int, max_rows: int | None, n_threads: int) -> dict:
    t0 = time.perf_counter()
    smoke = max_rows is not None
    print(f"=== M16 GBT tune  variant={variant}  k={k}  device=cpu  threads={n_threads}"
          f"{'  max_rows=' + format(max_rows, ',') if smoke else ''} ===")

    win = load_window(variant, k, max_rows)
    names, cat_idx = win["names"], win["cat_idx"]
    print(f"design: {len(F.CONTINUOUS)} cont + {len(win['vocab'].cols)} cat (native) + "
          f"{len(F.BINARY)} bin = {len(names)} cols   train={win['meta']['n_train']:,} rows  "
          f"Σw={win['meta']['train_sum_w']:,.0f}  (cont NaN cells={win['meta']['cont_nan_cells']:,})")
    print(f"rows: train={win['meta']['n_train']:,}  val={win['val']['n']:,}  test={win['test']['n']:,}")

    # Datasets reused across grid cells ⇒ free_raw_data=False; feature_pre_filter=False so the
    # shared train Dataset survives cells that vary the leaf-regularisation params.
    ds_params = {"max_bin": MAX_BIN, "feature_pre_filter": False}
    dtrain = lgb.Dataset(win["X_tr"], label=win["y_tr"], weight=win["w_tr"], feature_name=names,
                         categorical_feature=cat_idx, params=ds_params, free_raw_data=False)
    dval = lgb.Dataset(win["val"]["X"], label=win["val"]["y"], reference=dtrain, feature_name=names,
                       categorical_feature=cat_idx, params=ds_params, free_raw_data=False)  # NO weight
    dtrain.construct(); dval.construct()

    winner, selected, grid = tune(dtrain, dval, n_threads)
    selected_full = _full_cfg(selected, n_threads)
    print(f"[grid] selected: {_cell_label(selected)}  "
          f"val_nll={winner.best_score['val']['multi_logloss']:.6f}  best_iter={winner.best_iteration}")

    acc = compute_accept(winner, win, variant, k, label=f"M16 tune k={k}")
    metrics = _base_metrics(variant, k, win, acc, smoke=smoke, wall=time.perf_counter() - t0)
    metrics["hyperparams"] = {"max_bin": MAX_BIN, "early_stop_rounds": EARLY_STOP_ROUNDS,
                              "max_rounds": MAX_ROUNDS, "grid": grid}
    metrics["selection"] = {"selected": selected_full, "best_iteration": int(winner.best_iteration),
                            "val_multi_logloss": acc["lgb_val"], "source": "dev tuning grid (M16)"}
    run_dir = run_dir_for(variant, k, smoke)
    write_run_folder(run_dir, winner, win["vocab"], names, cat_idx, metrics)
    print(f"wrote {run_dir/'metrics.json'}  [{metrics['wall_sec']/60:.1f} min]")
    return metrics


# ===========================================================================
# M17 driver A — one frozen-config fit per window (the backtest loop unit)
# ===========================================================================
def fit_window_frozen(variant: str, k: int, cfg: dict, n_threads: int, *,
                      max_rows: int | None = None, fresh: bool = False) -> Path:
    """Fit ONE booster at the frozen ``cfg`` on window ``k`` — per-window vocab (train slice
    only), HT-weighted train, early stopping on that window's val — and write the run folder.
    The unit ``backtest.py`` loops over the 11 windows (``06 §5``). Idempotent on (window,
    config): a finished run at this exact config is skipped; a different config (e.g. after a
    ``reconfirm`` updated ``config.GBT_SELECTED``) triggers a refit."""
    cfg = {kk: cfg[kk] for kk in CFG_KEYS}
    run_dir = run_dir_for(variant, k, smoke=max_rows is not None)
    mpath = run_dir / "metrics.json"
    if mpath.exists() and not fresh:
        try:
            prev = json.loads(mpath.read_text()).get("selection", {}).get("selected")
        except (json.JSONDecodeError, OSError):
            prev = None     # corrupt/partial metrics.json ⇒ treat as incomplete ⇒ refit
        if prev == cfg:
            print(f"[gbt k={k}] already complete at this config — skipping ({run_dir.name})")
            return run_dir
        print(f"[gbt k={k}] stored config {prev} != frozen {cfg} — refitting")

    t0 = time.perf_counter()
    print(f"=== M17 GBT frozen  variant={variant}  k={k}  cfg={cfg}  threads={n_threads} ===")
    win = load_window(variant, k, max_rows)
    names, cat_idx = win["names"], win["cat_idx"]
    print(f"rows: train={win['meta']['n_train']:,}  val={win['val']['n']:,}  test={win['test']['n']:,}")

    # Single fit ⇒ free the (large) raw train matrix as soon as LightGBM has binned it, to
    # keep peak RAM low at full scale (the binned Dataset is ~2 GB; the raw float32 matrix is
    # several × larger on the biggest expanding windows).
    dtrain = lgb.Dataset(win["X_tr"], label=win["y_tr"], weight=win["w_tr"], feature_name=names,
                         categorical_feature=cat_idx, params={"max_bin": MAX_BIN}, free_raw_data=True)
    dtrain.construct()
    win["X_tr"] = None
    dval = lgb.Dataset(win["val"]["X"], label=win["val"]["y"], reference=dtrain, feature_name=names,
                       categorical_feature=cat_idx, params={"max_bin": MAX_BIN}, free_raw_data=False)
    booster = _train_cell(dtrain, dval, {**_base_params(n_threads), **cfg})
    print(f"  fit: best_iteration={booster.best_iteration}  "
          f"val_multi_logloss={booster.best_score['val']['multi_logloss']:.6f}")

    acc = compute_accept(booster, win, variant, k, label=f"M17 frozen k={k}")
    metrics = _base_metrics(variant, k, win, acc, smoke=max_rows is not None,
                            wall=time.perf_counter() - t0)
    metrics["hyperparams"] = {"max_bin": MAX_BIN, "early_stop_rounds": EARLY_STOP_ROUNDS,
                              "max_rounds": MAX_ROUNDS}
    metrics["selection"] = {"selected": cfg, "best_iteration": int(booster.best_iteration),
                            "val_multi_logloss": acc["lgb_val"], "source": "frozen (config.GBT_SELECTED)"}
    write_run_folder(run_dir, booster, win["vocab"], names, cat_idx, metrics)
    print(f"wrote {run_dir/'metrics.json'}  [{metrics['wall_sec']/60:.1f} min]")
    return run_dir


# ===========================================================================
# M17 driver B — full-scale re-confirm of the frozen config (M10b analogue)
# ===========================================================================
def reconfirm(variant: str, k: int, n_threads: int) -> dict:
    """Re-check ``config.GBT_SELECTED`` at full export scale on the tuning window (the M10b
    depth-check analogue). The scale-sensitive axis is ``min_sum_hessian_in_leaf``: it is an
    **absolute** summed-Hessian threshold, so ~10× the rows makes the dev-calibrated value
    relatively looser — re-check the frozen cell against scaled-up neighbours and a
    ``num_leaves`` neighbour, select on val, and report whether the winner moved. Writes a
    ``k{k}_reconfirm`` run folder; if the winner changes, update ``config.GBT_SELECTED``."""
    t0 = time.perf_counter()
    sel = {kk: config.GBT_SELECTED[kk] for kk in CFG_KEYS}
    print(f"=== M17 GBT reconfirm  variant={variant}  k={k}  frozen={sel}  threads={n_threads} ===")
    win = load_window(variant, k, None)
    names, cat_idx = win["names"], win["cat_idx"]
    print(f"rows: train={win['meta']['n_train']:,}  val={win['val']['n']:,}  test={win['test']['n']:,}")

    ds_params = {"max_bin": MAX_BIN, "feature_pre_filter": False}
    dtrain = lgb.Dataset(win["X_tr"], label=win["y_tr"], weight=win["w_tr"], feature_name=names,
                         categorical_feature=cat_idx, params=ds_params, free_raw_data=False)
    dval = lgb.Dataset(win["val"]["X"], label=win["val"]["y"], reference=dtrain, feature_name=names,
                       categorical_feature=cat_idx, params=ds_params, free_raw_data=False)
    dtrain.construct(); dval.construct()
    base = _base_params(n_threads)

    msh0 = sel["min_sum_hessian_in_leaf"]
    # Focused M10b-style re-check, one axis at a time from the frozen anchor on the two
    # scale-sensitive axes: min_sum_hessian_in_leaf UP ~10× (the ~34M-row slice carries ~10×
    # the summed Hessian that calibrated msh on the 3.15M dev slice, so effective leaf-reg is
    # ~10× weaker at scale — 06 §3) and a num_leaves capacity neighbour. Kept to 3 cells
    # (~82 min/cell full-scale) so the overnight 11-window sweep can launch tonight; widen if
    # a neighbour wins and the trend warrants resolving the optimum.
    cells = [("frozen", sel),
             ("msh×10", {**sel, "min_sum_hessian_in_leaf": msh0 * 10.0}),
             ("nl63", {**sel, "num_leaves": 63})]

    results, best = [], {"val_nll": float("inf"), "cfg": None, "booster": None, "label": None}
    for lbl, cfg in cells:
        cfg = {kk: cfg[kk] for kk in CFG_KEYS}
        booster = _train_cell(dtrain, dval, {**base, **cfg})
        vnll = float(booster.best_score["val"]["multi_logloss"])
        results.append({"label": lbl, "cfg": cfg, "val_nll": vnll,
                        "best_iteration": int(booster.best_iteration)})
        print(f"  {lbl:<10} {_cell_label(cfg):<54} val_nll={vnll:.6f}  best_iter={booster.best_iteration}")
        if vnll < best["val_nll"]:
            best.update(val_nll=vnll, cfg=cfg, booster=booster, label=lbl)

    changed = best["cfg"] != sel
    print(f"[reconfirm] winner = {best['label']} ({_cell_label(best['cfg'])})  val_nll={best['val_nll']:.6f}")
    print(f"[reconfirm] frozen config {'CHANGES' if changed else 'HOLDS'} at full scale"
          + (f"  → update config.GBT_SELECTED to {best['cfg']}" if changed else ""))

    acc = compute_accept(best["booster"], win, variant, k, label=f"M17 reconfirm k={k}")
    metrics = _base_metrics(variant, k, win, acc, smoke=False, wall=time.perf_counter() - t0)
    metrics["hyperparams"] = {"max_bin": MAX_BIN, "early_stop_rounds": EARLY_STOP_ROUNDS,
                              "max_rounds": MAX_ROUNDS, "grid": results}
    metrics["reconfirm"] = {"dev_frozen": sel, "fullscale_winner": best["cfg"],
                            "winner_label": best["label"], "winner_val_nll": best["val_nll"],
                            "changed": changed,
                            "action": ("update config.GBT_SELECTED" if changed else "keep config.GBT_SELECTED")}
    metrics["selection"] = {"selected": best["cfg"], "best_iteration": int(best["booster"].best_iteration),
                            "val_multi_logloss": acc["lgb_val"], "source": "full-scale reconfirm (M17)"}
    run_dir = config.MODELS / "gbt" / variant / f"k{k}_reconfirm"
    write_run_folder(run_dir, best["booster"], win["vocab"], names, cat_idx, metrics)
    print(f"wrote {run_dir/'metrics.json'}  [{metrics['wall_sec']/60:.1f} min]")
    return metrics


def main() -> None:
    import os
    ap = argparse.ArgumentParser(description="GBT baseline trainer (06 §3): tune / frozen / reconfirm.")
    ap.add_argument("--mode", default="tune", choices=["tune", "frozen", "reconfirm"],
                    help="tune=M16 grid; frozen=one fit at config.GBT_SELECTED; reconfirm=full-scale re-check")
    ap.add_argument("--variant", default=VARIANT_DEFAULT, choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=TUNING_K, help="test year / window (tuning = 2015)")
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows per split (wiring smoke)")
    ap.add_argument("--threads", type=int, default=0, help="LightGBM num_threads (0 = all cores)")
    ap.add_argument("--fresh", action="store_true", help="refit even if a run folder exists (frozen mode)")
    args = ap.parse_args()
    config.require_drive()
    n_threads = args.threads or (os.cpu_count() or 1)
    if args.mode == "tune":
        run(args.variant, args.k, args.max_rows, n_threads)
    elif args.mode == "frozen":
        fit_window_frozen(args.variant, args.k, dict(config.GBT_SELECTED), n_threads,
                          max_rows=args.max_rows, fresh=args.fresh)
    else:
        reconfirm(args.variant, args.k, n_threads)


if __name__ == "__main__":
    main()
