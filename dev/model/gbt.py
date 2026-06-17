"""M16 — GBT baseline trainer (``06_GBT_BASELINE §3``): one 7-class softmax LightGBM.

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
can address one column. CPU only — a documented deviation from the GPU default (``06 §2.4``):
at ~30 features the histogram work a GPU parallelises is small, so GBT is CPU-bound.

M16 runs the dev-scale tuning grid on the tuning window (k=2015), selects on the 2014 val
NLL, freezes the winner (recorded in ``config.GBT_SELECTED``), and writes a self-describing
run folder (``00_OVERVIEW §6.6``). M17 loops the frozen config over the 11 windows.

Run (CPU; needs the SSD + the dev export):
    .venv/bin/python dev/model/gbt.py                       # full dev k=2015 tuning + Accept
    .venv/bin/python dev/model/gbt.py --max-rows 100000     # quick wiring smoke
    .venv/bin/python dev/model/gbt.py --variant dev --k 2015
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import backtest as B          # _structural_allow — the §7 monotone-delinquency mask
import config
import data as D
import evaluate as E          # _nll — the headline out-of-sample NLL object (the cross-check target)
import features as F
import train as T             # _git_commit — shared run-folder provenance

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
    """Load + encode one window/split into arrays. ``vocab`` is the window's fitted vocab
    (fit on the train slice, applied to every split — no leakage). Train carries the HT
    ``weight``; val/test are the unthinned eval pool (``w≡1``)."""
    pool_dir, bounds = D.window_spec(variant, k, split)
    df = F.prepare_raw(D._masked_scan(pool_dir, bounds).collect())
    if max_rows is not None:
        df = df.head(max_rows)
    return {
        "X": build_X(df, vocab),
        "y": F.target_indices(df).astype(np.int32),
        "w": df.get_column(F.WEIGHT_COL).cast(pl.Float32).to_numpy(),
        "origin": _origin_index(df),
        "n": df.height,
    }


# ===========================================================================
# Tuning grid (06 §3) — num_leaves × lr plane, then anchored sweeps
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
        params = {**base, **overrides}
        booster = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)])
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
    floor, bucketed = band.get("empirical_floor"), band.get("bucketed")
    if floor is not None and test_nll >= floor:
        return "HARD_STOP"      # worse than the covariate-free floor ⇒ a wiring bug
    if bucketed is not None and test_nll > bucketed:
        return "YELLOW"         # between bucketed matrix and floor ⇒ review wiring
    return "HEALTHY"            # at/under the bucketed matrix


# ===========================================================================
# Predict adapter (06 §5) — the (n, 7) array every downstream consumer expects
# ===========================================================================
def predict_proba(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """``[N, 7]`` class probabilities at the booster's best iteration. LightGBM multiclass
    returns softmax-normalised rows (they sum to one), which is exactly the array
    ``evaluate.py`` / ``pool.py`` consume from every other model."""
    return booster.predict(X, num_iteration=booster.best_iteration)


# ===========================================================================
# Provenance
# ===========================================================================
def _environment() -> dict:
    import os
    return {"python": platform.python_version(), "lightgbm": lgb.__version__,
            "numpy": np.__version__, "polars": pl.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count()}


# ===========================================================================
# Driver — M16 tuning + Accept gate on the dev tuning window
# ===========================================================================
def run(variant: str, k: int, max_rows: int | None, n_threads: int) -> dict:
    t0 = time.perf_counter()
    smoke = max_rows is not None
    print(f"=== M16 GBT  variant={variant}  k={k}  device=cpu  threads={n_threads}"
          f"{'  max_rows=' + format(max_rows, ',') if smoke else ''} ===")

    # --- Load train, fit the per-window vocab on the train slice only (no leakage) ---
    tr_dir, tr_bounds = D.window_spec(variant, k, "train")
    df_tr = F.prepare_raw(D._masked_scan(tr_dir, tr_bounds).collect())
    if max_rows is not None:
        df_tr = df_tr.head(max_rows)
    vocab = F.Vocab.fit(df_tr, min_count=1)            # same min_count as data.fit_window
    names, cat_idx = feature_layout(vocab)
    X_tr = build_X(df_tr, vocab)
    y_tr = F.target_indices(df_tr).astype(np.int32)
    w_tr = df_tr.get_column(F.WEIGHT_COL).cast(pl.Float32).to_numpy()
    n_train, train_sum_w = df_tr.height, float(w_tr.sum())
    n_cont_nan = int(np.isnan(X_tr[:, :len(F.CONTINUOUS)]).sum())
    del df_tr
    print(f"design: {len(F.CONTINUOUS)} cont + {len(vocab.cols)} cat (native) + {len(F.BINARY)} bin "
          f"= {X_tr.shape[1]} cols   train={n_train:,} rows  Σw={train_sum_w:,.0f}  "
          f"(cont NaN cells={n_cont_nan:,})")

    val = load_split(variant, k, "val", vocab, max_rows)
    test = load_split(variant, k, "test", vocab, max_rows)
    print(f"rows: train={n_train:,}  val={val['n']:,}  test={test['n']:,}")

    # --- Datasets: HT weight on TRAIN only; val unweighted (06 §2.3) ---
    # feature_pre_filter=False so the single shared train Dataset is reusable across grid
    # cells that vary the leaf-regularisation params.
    ds_params = {"max_bin": MAX_BIN, "feature_pre_filter": False}
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=names,
                         categorical_feature=cat_idx, params=ds_params, free_raw_data=False)
    dval = lgb.Dataset(val["X"], label=val["y"], reference=dtrain, feature_name=names,
                       categorical_feature=cat_idx, params=ds_params,
                       free_raw_data=False)  # NO weight (val is the unthinned eval pool)
    dtrain.construct(); dval.construct()

    # --- Tune on the 2014 val NLL, freeze the winner ---
    winner, selected, grid = tune(dtrain, dval, n_threads)
    print(f"[grid] selected: {_cell_label(selected)}  "
          f"val_nll={winner.best_score['val']['multi_logloss']:.6f}  best_iter={winner.best_iteration}")
    selected_full = {k_: {**_base_params(n_threads), **selected}[k_] for k_ in
                     ("num_leaves", "learning_rate", "min_sum_hessian_in_leaf",
                      "feature_fraction", "lambda_l2")}

    # ============================ Accept gate ============================
    proba_val = predict_proba(winner, val["X"])
    proba_test = predict_proba(winner, test["X"])

    # (1) probabilities sum to one / no NaN
    sum_dev = float(np.abs(proba_test.sum(axis=1) - 1.0).max())
    any_nan = bool(np.isnan(proba_test).any())
    a1 = (sum_dev < SUM_TO_ONE_TOL) and not any_nan

    # (2) LightGBM multi_logloss == evaluate._nll on the same (n,7) preds (~1e-6, 06 §3)
    lgb_val = float(winner.best_score["val"]["multi_logloss"])
    eval_val_nll = E._nll(proba_val, val["y"].astype(np.int64))
    xdiff = abs(lgb_val - eval_val_nll)
    a2 = xdiff <= XCHECK_TOL

    # (3) base-rate QA — predicted mean class rates ≈ realized (importance-weighting check)
    realized = np.bincount(test["y"], minlength=F.N_CLASSES) / test["n"]
    pred_mean = proba_test.mean(axis=0)
    br_max = float(np.abs(realized - pred_mean).max())
    a3 = br_max < BASE_RATE_TOL

    # (4) structural-impossible-cell mass < 0.1% (backtest._structural_allow)
    allow = B._structural_allow()
    forbidden = ~allow[test["origin"]]                       # [N, 7]
    imp_mass = (proba_test * forbidden).sum(axis=1)
    realized_imp = forbidden[np.arange(test["n"]), test["y"]]
    imp_mean = float(imp_mass.mean())
    a4 = imp_mean < IMPOSSIBLE_TOL

    # (5) out-of-sample test NLL vs the sanity band (correctness zones, not a target)
    test_nll = E._nll(proba_test, test["y"].astype(np.int64))
    band = reference_band(variant, k)
    zone = classify_zone(test_nll, band)
    a5 = zone != "HARD_STOP"

    # feature-importance glance — flag high-card (zip3/MSA) native-categorical dominance
    gain = winner.feature_importance(importance_type="gain").astype(float)
    g_order = np.argsort(gain)[::-1]
    top_gain = [{"feature": names[i], "gain": float(gain[i])} for i in g_order[:15]]
    # the two highest-cardinality categoricals are the geo high-card columns (MSA/zip3)
    cat_cards = sorted(zip(vocab.cols, vocab.vocab_sizes), key=lambda t: -t[1])[:2]
    highcard_cols = [c for c, _ in cat_cards]
    gtot = float(gain.sum()) or 1.0
    highcard_share = float(sum(gain[names.index(c)] for c in highcard_cols) / gtot)
    top1_highcard = names[g_order[0]] in highcard_cols

    overall = a1 and a2 and a3 and a4 and a5

    # ---- print the evidence ----
    print("\n================= M16 ACCEPT EVIDENCE =================")
    print(f"[{'PASS' if a1 else 'FAIL'}] (1) probs sum to 1 (max|Σ−1|={sum_dev:.2e} < {SUM_TO_ONE_TOL:g}) "
          f"& no NaN (any_nan={any_nan})")
    print(f"[{'PASS' if a2 else 'FAIL'}] (2) LightGBM multi_logloss {lgb_val:.8f} == evaluate._nll "
          f"{eval_val_nll:.8f}  |Δ|={xdiff:.2e} (≤{XCHECK_TOL:g})")
    print(f"[{'PASS' if a3 else 'FAIL'}] (3) base-rate QA  max|realized−pred|={br_max:.4f} < {BASE_RATE_TOL}")
    print(f"[{'PASS' if a4 else 'FAIL'}] (4) impossible-cell mass mean={imp_mean:.2e} < {IMPOSSIBLE_TOL:g}  "
          f"(realized {float(realized_imp.mean()):.2e}; max row {float(imp_mass.max()):.2e})")
    fl, bk, lg = band.get("empirical_floor"), band.get("bucketed"), band.get("logit")
    nn = band.get("nn_best")
    print(f"[{'PASS' if a5 else 'FAIL'}] (5) test NLL={test_nll:.6f}  zone={zone}")
    print(f"        band: empirical_floor={fl:.5f}  bucketed={bk:.5f}  logit={lg:.5f}"
          + (f"  nn_best={nn:.5f}" if nn is not None else ""))
    print(f"        (hard-stop ≥{fl:.4f}; yellow ({bk:.4f},{fl:.4f}]; healthy ≤{bk:.4f})")
    print(f"[glance] feature-importance: high-card {highcard_cols} gain share={highcard_share:.1%} "
          f"(top-1 feature is high-card: {top1_highcard})")
    print(f"         top gain: {', '.join(t['feature'] for t in top_gain[:6])}")
    print(f"\n{'>>> M16 ACCEPT: ALL CRITERIA PASS' if overall else '>>> M16 ACCEPT: FAILURE — debug before advancing'}"
          f"  (zone={zone})")
    print("=" * 54)

    # ---- self-describing run folder (00_OVERVIEW §6.6) ----
    run_dir = config.MODELS / "gbt" / variant / (f"k{k}" + ("_smoke" if smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    winner.save_model(str(run_dir / "model.txt"), num_iteration=winner.best_iteration)
    (run_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), indent=2))
    (run_dir / "feature_layout.json").write_text(json.dumps(
        {"feature_name": names, "categorical_index": cat_idx,
         "n_continuous": len(F.CONTINUOUS), "n_categorical": len(vocab.cols),
         "n_binary": len(F.BINARY)}, indent=2))

    manifest_path = config.TRAINING_DIR / variant / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    metrics = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "lightgbm_multiclass_softmax", "variant": variant, "window_k": k,
        "tuning_window": (k == config.TUNING_YEAR), "smoke": smoke,
        "device": "cpu", "seed": SEED, "git_commit": T._git_commit(),
        "environment": _environment(),
        "source": {"train_pool_content_hash": manifest.get("train_pool", {}).get("content_hash"),
                   "eval_pool_content_hash": manifest.get("eval_pool", {}).get("content_hash"),
                   "panel_glob": config.PANEL_GLOB},
        "features": {"n_continuous": len(F.CONTINUOUS), "n_categorical": len(vocab.cols),
                     "n_binary": len(F.BINARY), "n_features": X_tr.shape[1],
                     "vocab_sizes": dict(zip(vocab.cols, vocab.vocab_sizes)),
                     "continuous_nan_cells_train": n_cont_nan},
        "rows": {"train": n_train, "train_sum_w": train_sum_w, "val": val["n"], "test": test["n"]},
        "hyperparams": {"max_bin": MAX_BIN, "early_stop_rounds": EARLY_STOP_ROUNDS,
                        "max_rounds": MAX_ROUNDS, "grid": grid},
        "selection": {"selected": selected_full, "best_iteration": int(winner.best_iteration),
                      "val_multi_logloss": lgb_val},
        "gbt": {"val_nll": eval_val_nll, "test_nll": test_nll, "lgb_val_multi_logloss": lgb_val},
        "crosscheck": {"lgb_val_multi_logloss": lgb_val, "evaluate_nll": eval_val_nll,
                       "abs_diff": xdiff, "tol": XCHECK_TOL, "passed": a2},
        "probs_sanity": {"sum_to_one_max_abs_dev": sum_dev, "any_nan": any_nan, "passed": a1},
        "base_rate_qa": {"states": F.STATES, "realized": realized.tolist(),
                         "predicted_mean": pred_mean.tolist(), "max_abs_diff": br_max,
                         "tol": BASE_RATE_TOL, "passed": a3},
        "impossible_transitions": {"model_mean_mass": imp_mean,
                                   "model_max_row_mass": float(imp_mass.max()),
                                   "realized_mean_rate": float(realized_imp.mean()),
                                   "threshold": IMPOSSIBLE_TOL, "passed": a4},
        "band": {**band, "test_nll": test_nll, "zone": zone, "passed": a5},
        "feature_importance": {"top_gain": top_gain, "highcard_cols": highcard_cols,
                               "highcard_gain_share": highcard_share,
                               "top1_is_highcard": top1_highcard},
        "accept": {"all_pass": overall,
                   "criteria": {"probs_valid": a1, "nll_crosscheck": a2, "base_rate_qa": a3,
                                "impossible_mass": a4, "band_not_hardstop": a5}},
        "wall_sec": time.perf_counter() - t0,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {run_dir/'metrics.json'}  [{metrics['wall_sec']/60:.1f} min]")
    return metrics


def main() -> None:
    import os
    ap = argparse.ArgumentParser(description="M16 GBT baseline trainer (06 §3) — dev tuning window.")
    ap.add_argument("--variant", default=VARIANT_DEFAULT, choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=TUNING_K, help="test year / window (tuning = 2015)")
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows per split (wiring smoke)")
    ap.add_argument("--threads", type=int, default=0, help="LightGBM num_threads (0 = all cores)")
    args = ap.parse_args()
    config.require_drive()
    n_threads = args.threads or (os.cpu_count() or 1)
    run(args.variant, args.k, args.max_rows, n_threads)


if __name__ == "__main__":
    main()
