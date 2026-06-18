#!/usr/bin/env python
"""THROWAWAY go/no-go probe (k2015). NOT a model member; do not integrate.

Question: how much of the full-scale logit -> GBT NLL gap on the k=2015 tuning window is
*additive* univariate nonlinearity (per-feature shape the linear logit can't bend to) versus
genuine interactions (which only a tree/net captures)?

External full-scale anchors (do not recompute):
    empirical 0.112869 | bucketed 0.109385 | logit 0.108246 | GBT 0.103597 | NN 0.103125
    logit -> GBT gap = 0.004649.

DELIVERABLES (clean -> noisy):
  * PRIMARY   : raw plain-spline test-NLL gap, overall AND per origin state.
  * SHAPES    : spline-logit PDP vs empirical equal-population-bin hazard, on a 2015
                current-origin background (the test-slice analogue of the Ch.3 hazards).
  * SECONDARY : fraction of the logit->GBT gap closed -- ORIENTATION ONLY, doubly approximate
                (subsample vs full; geo mismatch -- GBT denom geo-inclusive, probe geo-dropped).

Method:
  * Reuse the pipeline logit's EXACT persisted transforms (scaler.json + vocab.json from
    models/logit/full/k2015) so the cont / one-hot / binary blocks are bit-identical to what
    the pipeline logit fed its softmax. The ONLY new manipulation is the spline basis.
  * PLAIN  = [ standardized cont (23) || one-hot cats || binary (10) ].
  * SPLINE = [ per-feature cubic B-spline (quantile knots, additive, NO interactions) of the
    same 23 standardized cont || one-hot || binary ].  PLAIN is nested in SPLINE.
  * Single multinomial softmax (sklearn lbfgs); origin `state` one-hot (matches GBT/net).
    SAME solver + SAME C for plain and spline -> the gap reflects basis expansion alone.
  * Train = proportional stratified subsample (deterministic hash) to 1.5M rows, streamed.
    Test = the FULL frozen k2015 eval slice (6,180,351 rows), UNWEIGHTED, scored per part.
    HT weights on TRAIN only (normalised to mean 1).
  * Deviation (flagged): drop the two high-card geos MSA(407)+Zip3(989) -- a 1.5M x 1506 lbfgs
    fit (float64 upcast) ~= 24 GB > 12 GB budget -- exactly logit.py:CROSS_CHECK_DROP. Keep
    Property State(55). Representativeness is checked against the FULL train slice (geo-blind),
    not via the geo-confounded |plain - 0.108246| offset.

Run:  PYTHONPATH=src .venv/bin/python experiments/spline_logit_probe/run_probe.py
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))  # package not pip-installed in this venv

from floan.model import config            # noqa: E402  (read-only: bounds/paths)
from floan.model import data as D          # noqa: E402  (read-only: RAW_COLS, unused but kept)
from floan.model import features as F      # noqa: E402  (read-only: transforms/target)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import SplineTransformer  # noqa: E402
import matplotlib                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

# --------------------------------------------------------------------------- config
VARIANT = "full"
K = 2015
ANCHORS = {"empirical": 0.112869, "bucketed": 0.109385,
           "logit": 0.108246, "gbt": 0.103597, "nn": 0.103125}
ANCHOR_GAP = ANCHORS["logit"] - ANCHORS["gbt"]        # 0.004649
EXPECTED_TEST_ROWS = 6_180_351                        # models/logit/full/k2015 test_n_rows
N_TOTAL_TRAIN = 33_505_908                            # k2015 train slice row count (verified)
N_TRAIN = 1_500_000                                   # subsample target (<=2M; 1.5M for float64 upcast)
N_KNOTS = 8
SPLINE_DEGREE = 3
KNOTS_ALT = (5, 12)                                   # robustness, only if fraction in trigger band
FRACTION_TRIGGER = (0.25, 0.60)
DROP_CATS = ("Metropolitan Statistical Area (MSA)", "Zip Code Short")  # == logit CROSS_CHECK_DROP
SEED = 0
L2_C = 0.1                                            # == logit.py:CROSS_CHECK_C — conditions the
#                                                       over-param softmax gauge for this geo-dropped
#                                                       sklearn design (still near-MLE; shared by both)
MAX_ITER = 500
BASE_RATE_TOL = 0.01                                  # subsample vs FULL train per-cell base rate
BG_TARGET = 250_000                                   # current-origin 2015 background for shapes
N_EMP_BINS = 20
SHAPE_FEATURES = ["incentive", "Loan Age", "ltv_mtm", "fico_orig"]
SHAPE_LABELS = {"incentive": "incentive (rate gap, %)", "Loan Age": "loan age (months)",
                "ltv_mtm": "current LTV (mark-to-market)", "fico_orig": "FICO at origination"}
HASH_DEN = 1_000_000

ORIGIN_STATES = list(config.ORIGIN_STATES)            # current, dpd_30, dpd_60, dpd_90plus
ORIG_IDX = {o: i for i, o in enumerate(ORIGIN_STATES)}
PREPAID, DPD30 = F.STATE_INDEX["prepaid"], F.STATE_INDEX["dpd_30"]

# categoricals kept after dropping the two high-card geos (order preserved from vocab.cols).
KEEP_RAW_CATS = [c for c in F.RAW_CATEGORICAL if c not in DROP_CATS]
LOAD_COLS = sorted(set(
    F.CONTINUOUS + KEEP_RAW_CATS + F.BINARY
    + [F.TARGET_COL, F.WEIGHT_COL, "period_ym", "orig_ym", "Loan Identifier"]
))
TRAIN_GLOB = str(config.TRAINING_DIR / VARIANT / "train_pool" / "part-*.parquet")
EVAL_PARTS = sorted((config.TRAINING_DIR / VARIANT / "eval_pool").glob("part-*.parquet"))


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}  rss={_rss_gb():4.1f}GB] {msg}", flush=True)


# --------------------------------------------------------------------------- pipeline reuse
def load_pipeline():
    scaler, vocab = F.load_pipeline(config.MODELS / "logit" / VARIANT / f"k{K}")
    keep = [c for c in vocab.cols if c not in set(DROP_CATS)]
    sub_vocab = F.Vocab(cols=keep, maps={c: vocab.maps[c] for c in keep})
    return scaler, vocab, sub_vocab


def design_blocks(df, scaler, sub_vocab):
    """Bit-identical to the pipeline logit's blocks, minus the two dropped geo cats."""
    cont = scaler.transform(df)                       # [N,23] standardized (incl log1p/robust)
    oh = F.one_hot_block(sub_vocab.transform(df), sub_vocab)
    binb = F.binary_matrix(df)                        # [N,10] 0/1
    y = F.target_indices(df)                          # [N] in 0..6
    return cont, oh, binb, y


# --------------------------------------------------------------------------- splines
def fit_spline_columns(C, n_knots=N_KNOTS):
    """One SplineTransformer per continuous column (additive, NO cross terms). Quantile knots,
    degree 3; gracefully degrade low-cardinality columns (e.g. Number of Units) so the basis
    never degenerates."""
    transformers = []
    for j in range(C.shape[1]):
        col = C[:, j:j + 1]
        nk = int(min(n_knots, max(2, int(np.unique(col).size))))
        deg = int(min(SPLINE_DEGREE, nk - 1))
        try:
            st = SplineTransformer(n_knots=nk, degree=deg, knots="quantile",
                                   include_bias=False).fit(col)
        except ValueError:
            st = SplineTransformer(n_knots=nk, degree=deg, knots="uniform",
                                   include_bias=False).fit(col)
        transformers.append(st)
    return transformers


def apply_spline(transformers, C):
    return np.concatenate(
        [t.transform(C[:, j:j + 1]) for j, t in enumerate(transformers)], axis=1
    ).astype(np.float32)


# --------------------------------------------------------------------------- base rates
def counts_matrix(origin, y):
    """[4 origin x 7 next] count matrix."""
    M = np.zeros((len(ORIGIN_STATES), F.N_CLASSES), np.float64)
    for o, oi in ORIG_IDX.items():
        yy = y[origin == o]
        if yy.size:
            M[oi] = np.bincount(yy, minlength=F.N_CLASSES)
    return M


def full_train_base_rate_counts(scaler):
    """Per-origin next-state counts over the FULL 33.5M k2015 train slice (streaming groupby,
    reads only state/state_next) -- the geo-blind reference for the subsample check."""
    lo, hi = config.train_bounds(K)
    g = (pl.scan_parquet(TRAIN_GLOB)
           .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
           .group_by(["state", F.TARGET_COL]).len()
           .collect(engine="streaming"))
    M = np.zeros((len(ORIGIN_STATES), F.N_CLASSES), np.float64)
    for row in g.iter_rows(named=True):
        oi = ORIG_IDX.get(row["state"])
        ni = F.STATE_INDEX.get(row[F.TARGET_COL], -1)
        if oi is not None and ni >= 0:
            M[oi, ni] = row["len"]
    return M


def rates(M):
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, np.where(s == 0, 1, s))


def max_rate_diff(M_a, M_b):
    Ra, Rb = rates(M_a), rates(M_b)
    per_origin = {ORIGIN_STATES[i]: float(np.abs(Ra[i] - Rb[i]).max())
                  for i in range(len(ORIGIN_STATES))}
    return float(np.abs(Ra - Rb).max()), per_origin


# --------------------------------------------------------------------------- train load
def load_train_subsample(scaler, sub_vocab):
    """Proportional stratified subsample (deterministic hash) -> <=N_TRAIN rows, streamed so the
    33.5M-row slice never lands in RAM. Equal keep-fraction across origins keeps the HT weights
    valid (MCAR) and lets rare origins survive."""
    lo, hi = config.train_bounds(K)
    frac = min(1.0, N_TRAIN / N_TOTAL_TRAIN)
    thr = int(frac * HASH_DEN)
    lf = (pl.scan_parquet(TRAIN_GLOB)
            .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
            .select(LOAD_COLS)
            .filter((pl.struct(["Loan Identifier", "period_ym"]).hash(seed=SEED)
                     % HASH_DEN) < thr))
    df = F.prepare_raw(lf.collect(engine="streaming"))
    log(f"train subsample: {df.height:,} rows (target ~{N_TRAIN:,}, frac={frac:.4f})")
    origin = df.get_column("state").to_numpy()
    cont, oh, binb, y = design_blocks(df, scaler, sub_vocab)
    w = df.get_column(F.WEIGHT_COL).cast(pl.Float64).to_numpy()
    del df
    gc.collect()
    return cont, oh, binb, y, w, origin


# --------------------------------------------------------------------------- fit / eval
def fit_logit(X, y, w):
    clf = LogisticRegression(C=L2_C, solver="lbfgs", max_iter=MAX_ITER, tol=1e-5,
                             fit_intercept=True, n_jobs=-1)
    clf.fit(X, y, sample_weight=w)
    return clf


def full_proba(clf, X):
    """predict_proba mapped into the canonical 7-class column order (0 for absent classes)."""
    P = clf.predict_proba(X)
    if P.shape[1] == F.N_CLASSES and np.array_equal(clf.classes_, np.arange(F.N_CLASSES)):
        return P
    out = np.zeros((P.shape[0], F.N_CLASSES), dtype=P.dtype)
    out[:, clf.classes_] = P
    return out


def iter_test_parts(scaler, sub_vocab):
    """Yield (cont, oh, binb, y, origin) for each non-empty k2015 eval (test) part."""
    lo, hi = config.test_bounds(K)
    for part in EVAL_PARTS:
        df = F.prepare_raw(pl.scan_parquet(str(part))
                           .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
                           .select(LOAD_COLS).collect())
        if df.height == 0:
            continue
        origin = df.get_column("state").to_numpy()
        cont, oh, binb, y = design_blocks(df, scaler, sub_vocab)
        del df
        yield cont, oh, binb, y, origin


def _nll_sum(clf, X, y):
    P = full_proba(clf, X)
    err = float(np.abs(P.sum(axis=1) - 1.0).max())
    assert np.isfinite(P).all(), "non-finite probs"
    nll = -np.log(np.clip(P[np.arange(y.shape[0]), y], 1e-300, 1.0))
    return nll, err


def eval_dual(clf_plain, clf_spline, scaler, sub_vocab, transformers):
    """Stream the FULL frozen test slice (unweighted). Overall + per-origin sum(-log p) and the
    test per-origin next-state counts (for the drift FYI)."""
    acc = {"plain": [0.0, 0], "spline": [0.0, 0]}
    po = {o: {"plain": 0.0, "spline": 0.0, "n": 0} for o in ORIGIN_STATES}
    M_test = np.zeros((len(ORIGIN_STATES), F.N_CLASSES), np.float64)
    max_err = 0.0
    n_total = 0
    for i, (cont, oh, binb, y, origin) in enumerate(iter_test_parts(scaler, sub_vocab)):
        Xpl = np.concatenate([cont, oh, binb], axis=1).astype(np.float32)
        Xsp = np.concatenate([apply_spline(transformers, cont), oh, binb], axis=1).astype(np.float32)
        for name, clf, X in (("plain", clf_plain, Xpl), ("spline", clf_spline, Xsp)):
            nll, err = _nll_sum(clf, X, y)
            max_err = max(max_err, err)
            acc[name][0] += float(nll.sum()); acc[name][1] += nll.shape[0]
            for o in ORIGIN_STATES:
                po[o][name] += float(nll[origin == o].sum())
        for o in ORIGIN_STATES:
            po[o]["n"] += int((origin == o).sum())
        M_test += counts_matrix(origin, y)
        n_total += y.shape[0]
        del Xpl, Xsp, cont, oh, binb, y, origin
        gc.collect()
        log(f"  eval part {i + 1}/{len(EVAL_PARTS)}: cum test rows {n_total:,}")
    out = {
        "plain_nll": acc["plain"][0] / acc["plain"][1],
        "spline_nll": acc["spline"][0] / acc["spline"][1],
        "test_rows": n_total, "max_prob_sum_err": max_err,
        "per_origin": {o: {"plain_nll": d["plain"] / d["n"], "spline_nll": d["spline"] / d["n"],
                           "gap": (d["plain"] - d["spline"]) / d["n"], "n": d["n"]}
                       for o, d in po.items()},
        "M_test": M_test,
    }
    return out


def eval_spline_overall(clf, scaler, sub_vocab, transformers):
    """Overall test NLL for one spline model (knot-robustness re-runs)."""
    s, n = 0.0, 0
    for cont, oh, binb, y, _ in iter_test_parts(scaler, sub_vocab):
        X = np.concatenate([apply_spline(transformers, cont), oh, binb], axis=1).astype(np.float32)
        nll, _ = _nll_sum(clf, X, y)
        s += float(nll.sum()); n += nll.shape[0]
        del X, cont, oh, binb, y
        gc.collect()
    return s / n


# --------------------------------------------------------------------------- shapes (PDP + EDA)
def load_background_current(scaler, sub_vocab):
    """Hash-sampled current-origin rows from the 2015 eval (test) slice -- UNTHINNED, UNWEIGHTED
    -> the true 2015 population for empirical hazards. (Train rows are thinned, so unusable.)"""
    lo, hi = config.test_bounds(K)
    frac = min(1.0, BG_TARGET / EXPECTED_TEST_ROWS) * 1.05  # ~current share is ~98%, headroom
    thr = int(min(1.0, frac) * HASH_DEN)
    lf = (pl.scan_parquet([str(p) for p in EVAL_PARTS])
            .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi)
                    & (pl.col("state") == "current"))
            .select(LOAD_COLS)
            .filter((pl.struct(["Loan Identifier", "period_ym"]).hash(seed=SEED)
                     % HASH_DEN) < thr))
    df = F.prepare_raw(lf.collect(engine="streaming"))
    cont, oh, binb, y = design_blocks(df, scaler, sub_vocab)
    raw = {f: df.get_column(f).cast(pl.Float64).to_numpy() for f in SHAPE_FEATURES}
    log(f"shape background: {df.height:,} current-origin 2015 rows")
    del df
    return cont, oh, binb, y, raw


def shape_plots(clf_spline, transformers, scaler, cont_bg, oh_bg, bin_bg, y_bg, raw, out_png):
    """4 panels: spline-logit PDP (line) vs empirical equal-pop-bin hazard (markers), both on the
    SAME 2015 current-origin background, for P(prepaid) and P(dpd_30)."""
    spl_blocks = [t.transform(cont_bg[:, j:j + 1]) for j, t in enumerate(transformers)]
    widths = [b.shape[1] for b in spl_blocks]
    offs = np.concatenate([[0], np.cumsum(widths)])
    Xsp_bg = np.concatenate(spl_blocks + [oh_bg, bin_bg], axis=1).astype(np.float32)
    del spl_blocks

    def z_to_raw(col, z):
        v = z * scaler.scale[col] + scaler.center[col]
        return np.expm1(v) if col in scaler.log_cols else v

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    summary = {}
    G = 80
    for ax, feat in zip(axes.ravel(), SHAPE_FEATURES):
        j = scaler.cols.index(feat)
        # --- model PDP (marginalise over the real background) ---
        zlo, zhi = np.percentile(cont_bg[:, j], [2, 98])
        zgrid = np.linspace(zlo, zhi, G)
        Xw = Xsp_bg.copy()
        pdp_pp, pdp_dd = np.empty(G), np.empty(G)
        for g, z in enumerate(zgrid):
            Xw[:, offs[j]:offs[j + 1]] = transformers[j].transform(np.array([[z]]))
            P = full_proba(clf_spline, Xw)
            pdp_pp[g] = P[:, PREPAID].mean(); pdp_dd[g] = P[:, DPD30].mean()
        del Xw
        xgrid = z_to_raw(feat, zgrid)
        # --- empirical equal-population-bin hazard on the same background ---
        rv = raw[feat]
        fin = np.isfinite(rv)
        edges = np.unique(np.quantile(rv[fin], np.linspace(0, 1, N_EMP_BINS + 1)))
        bidx = np.clip(np.digitize(rv, edges[1:-1]), 0, len(edges) - 2)
        emp_x, emp_pp, emp_dd = [], [], []
        for b in range(len(edges) - 1):
            m = fin & (bidx == b)
            if m.sum() >= 200:
                emp_x.append(float(rv[m].mean()))
                emp_pp.append(float((y_bg[m] == PREPAID).mean()))
                emp_dd.append(float((y_bg[m] == DPD30).mean()))
        ax.plot(xgrid, pdp_pp, color="C0", lw=2, label="spline PDP  P(prepaid)")
        ax.scatter(emp_x, emp_pp, color="C0", s=20, marker="o", zorder=3,
                   label="empirical  P(prepaid)")
        ax.plot(xgrid, pdp_dd, color="C3", lw=2, label="spline PDP  P(dpd_30)")
        ax.scatter(emp_x, emp_dd, color="C3", s=20, marker="s", zorder=3,
                   label="empirical  P(dpd_30)")
        ax.set_title(feat); ax.set_xlabel(SHAPE_LABELS[feat])
        ax.set_ylabel("prob (origin=current)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        summary[feat] = {"pdp_prepaid_range": [float(pdp_pp.min()), float(pdp_pp.max())],
                         "pdp_dpd30_range": [float(pdp_dd.min()), float(pdp_dd.max())],
                         "emp_bins": len(emp_x)}
    fig.suptitle("Additive spline-logit: PDP vs empirical hazard  (2015 test-slice background, "
                 "origin=current)\nTest-slice analogue of the Ch.3 full-panel hazards -- not the "
                 "same curves (2015 range only).", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return summary


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    config.require_drive()
    log(f"probe start  variant={VARIANT} k={K}  drop_cats={list(DROP_CATS)}")
    scaler, vocab, sub_vocab = load_pipeline()
    log(f"reused transforms: {len(scaler.cols)} cont, {len(sub_vocab.cols)} cats "
        f"(was {len(vocab.cols)}; dropped {list(DROP_CATS)}), {len(F.BINARY)} binary")

    # ---- full-train base-rate reference (geo-blind subsample check) -------------------
    M_full = full_train_base_rate_counts(scaler)
    log(f"full-train base rates accumulated (per-origin n = {M_full.sum(axis=1).astype(int)})")

    # ---- train subsample --------------------------------------------------------------
    cont_tr, oh_tr, bin_tr, y_tr, w_tr, origin_tr = load_train_subsample(scaler, sub_vocab)
    n_train = y_tr.shape[0]
    assert n_train <= 2_000_000, f"subsample {n_train} exceeds 2M cap"
    sub_origin_counts = {o: int((origin_tr == o).sum()) for o in ORIGIN_STATES}
    classes_present = sorted(set(np.unique(y_tr).tolist()))
    assert classes_present == list(range(F.N_CLASSES)), f"missing classes: {classes_present}"
    M_sub = counts_matrix(origin_tr, y_tr)
    br_max, br_per_origin = max_rate_diff(M_sub, M_full)
    br_pass = br_max < BASE_RATE_TOL
    log(f"representativeness (subsample vs FULL train): max|Δ base rate|={br_max:.5f} "
        f"(tol {BASE_RATE_TOL}) -> {'PASS' if br_pass else 'FAIL'}")
    assert br_pass, f"subsample base rates diverge from full train ({br_max:.5f}) -- stratifier bug?"
    w_tr = (w_tr / w_tr.mean()).astype(np.float64)  # mean-1 (preserves HT relative weights)

    # ---- designs + fits (SAME C/solver for both) --------------------------------------
    transformers = fit_spline_columns(cont_tr, N_KNOTS)
    X_plain_tr = np.concatenate([cont_tr, oh_tr, bin_tr], axis=1).astype(np.float32)
    w_plain = X_plain_tr.shape[1]
    log("fitting PLAIN logit (lbfgs, multinomial)...")
    clf_plain = fit_logit(X_plain_tr, y_tr, w_tr)
    plain_iter = int(np.max(clf_plain.n_iter_)); del X_plain_tr; gc.collect()
    X_spline_tr = np.concatenate([apply_spline(transformers, cont_tr), oh_tr, bin_tr],
                                 axis=1).astype(np.float32)
    w_spline = X_spline_tr.shape[1]
    assert w_spline > w_plain and X_spline_tr.shape[0] == n_train
    assert np.isfinite(X_spline_tr).all()
    log(f"designs: plain={w_plain}, spline={w_spline} cols")
    log("fitting SPLINE logit (lbfgs, multinomial)...")
    clf_spline = fit_logit(X_spline_tr, y_tr, w_tr)
    spline_iter = int(np.max(clf_spline.n_iter_)); del X_spline_tr; gc.collect()
    log(f"converged: plain n_iter={plain_iter}, spline n_iter={spline_iter} (max {MAX_ITER})")

    # ---- evaluate on the FULL frozen test (unweighted, chunked) -----------------------
    log("evaluating both on the full frozen k2015 eval slice (unweighted, per-part)...")
    ev = eval_dual(clf_plain, clf_spline, scaler, sub_vocab, transformers)
    assert ev["test_rows"] == EXPECTED_TEST_ROWS, f"test rows {ev['test_rows']} != {EXPECTED_TEST_ROWS}"
    assert ev["max_prob_sum_err"] < 1e-6, f"probs not summing to 1: {ev['max_prob_sum_err']}"

    plain_nll, spline_nll = ev["plain_nll"], ev["spline_nll"]
    gap = plain_nll - spline_nll
    fraction = gap / ANCHOR_GAP
    plain_vs_anchor = abs(plain_nll - ANCHORS["logit"])
    drift_max, drift_per_origin = max_rate_diff(M_full, ev["M_test"])  # FYI only
    log(f"PRIMARY  raw gap (plain-spline) = {gap:.6f}  | plain {plain_nll:.6f}  spline {spline_nll:.6f}")
    for o in ORIGIN_STATES:
        log(f"   per-origin gap  {o:<11} = {ev['per_origin'][o]['gap']:+.6f}  (n={ev['per_origin'][o]['n']:,})")
    log(f"SECONDARY (orientation only) fraction of logit->GBT gap ~= {fraction*100:.1f}%")
    log(f"diagnostics: |plain-0.108246|={plain_vs_anchor:.6f} (geo+subsample+reg, not a gate)  "
        f"| train->test drift max|Δ|={drift_max:.4f} (FYI)")

    # ---- conditional knot robustness --------------------------------------------------
    if FRACTION_TRIGGER[0] <= fraction <= FRACTION_TRIGGER[1]:
        knot_rob = {"triggered": True, "note": f"fraction {fraction:.3f} in {FRACTION_TRIGGER}",
                    "knot_8_gap": gap, "by_knots": {}}
        for nk in KNOTS_ALT:
            log(f"knot robustness: refitting spline n_knots={nk}...")
            tr_nk = fit_spline_columns(cont_tr, nk)
            X_nk = np.concatenate([apply_spline(tr_nk, cont_tr), oh_tr, bin_tr],
                                  axis=1).astype(np.float32)
            clf_nk = fit_logit(X_nk, y_tr, w_tr); del X_nk; gc.collect()
            sp_nk = eval_spline_overall(clf_nk, scaler, sub_vocab, tr_nk)
            knot_rob["by_knots"][str(nk)] = {"spline_nll": sp_nk, "gap": plain_nll - sp_nk}
            log(f"   n_knots={nk}: spline_nll={sp_nk:.6f}  gap={plain_nll - sp_nk:.6f}")
    else:
        knot_rob = {"triggered": False,
                    "note": f"skipped (fraction {fraction:.3f} outside {FRACTION_TRIGGER})"}

    del cont_tr, oh_tr, bin_tr, y_tr, w_tr, origin_tr; gc.collect()

    # ---- shapes (PDP + empirical overlay, 2015 background) ----------------------------
    log("building shape overlays (PDP vs empirical, 2015 current-origin background)...")
    cont_bg, oh_bg, bin_bg, y_bg, raw_bg = load_background_current(scaler, sub_vocab)
    out_png = HERE / "shapes.png"
    shapes = shape_plots(clf_spline, transformers, scaler, cont_bg, oh_bg, bin_bg, y_bg,
                         raw_bg, out_png)
    log(f"wrote {out_png} ({out_png.stat().st_size:,} bytes)")

    # ---- results.json -----------------------------------------------------------------
    results = {
        "probe": "additive spline-expanded multinomial logit vs plain logit (k2015 go/no-go)",
        "throwaway": True, "variant": VARIANT, "window_k": K, "seed": SEED,
        "anchors_full_scale": ANCHORS,
        "headline": {
            "raw_gap_overall": gap,
            "plain_nll": plain_nll, "spline_nll": spline_nll,
            "raw_gap_per_origin": {o: ev["per_origin"][o]["gap"] for o in ORIGIN_STATES},
            "expectation": "spline expected to close LESS on `current` (incentive×FICO interaction "
                           "lives there) and MORE on the delinquent / pure-shape origins",
        },
        "per_origin_nll": ev["per_origin"],
        "secondary_orientation_only": {
            "closed_fraction_of_anchor_gap": fraction, "anchor_gap": ANCHOR_GAP,
            "caveat": "DOUBLY approximate: (a) 1.5M subsample vs 33.5M full train; (b) geo "
                      "mismatch -- the GBT denominator 0.103597 is geo-inclusive while the probe "
                      "is geo-dropped (MSA+Zip3). Use the raw gaps + shape overlays as the clean "
                      "deliverables, not this number.",
        },
        "representativeness": {
            "reference": "full un-subsampled k2015 train slice (same era -> isolates sampling "
                         "from 2013->2015 drift)",
            "max_abs_baserate_diff_subsample_vs_fulltrain": br_max, "tol": BASE_RATE_TOL,
            "pass": br_pass, "per_origin_max_diff": br_per_origin,
        },
        "drift_fyi": {
            "note": "train(<=2013) vs test(2015) per-origin next-state base-rate drift -- NOT a "
                    "gate; reported so the |plain-0.108246| offset is read in context",
            "max_abs_diff": drift_max, "per_origin_max_diff": drift_per_origin,
        },
        "regularization": {
            "probe_C": L2_C, "solver": "lbfgs", "max_iter": MAX_ITER,
            "plain_n_iter": plain_iter, "spline_n_iter": spline_iter,
            "converged": plain_iter < MAX_ITER and spline_iter < MAX_ITER,
            "note": "deployed logit used torch Adam weight_decay=1e-5 (val-selected), NOT an "
                    "sklearn C. Probe uses C=0.1 == logit.py:CROSS_CHECK_C (the repo's own "
                    "conditioning value for fitting this exact geo-dropped multinomial with "
                    "sklearn lbfgs) -> mean-1 weights give effective L2 ~3e-6 (still near-MLE, the "
                    "same light-reg regime) while pinning the over-param softmax gauge so lbfgs "
                    "converges. Plain and spline SHARE C, so the gap reflects the basis expansion "
                    "alone.",
        },
        "subsample": {"target": N_TRAIN, "actual": n_train,
                      "strategy": "proportional stratified deterministic hash",
                      "per_origin": sub_origin_counts},
        "spline": {"n_knots": N_KNOTS, "degree": SPLINE_DEGREE, "knots": "quantile",
                   "include_bias": False, "additive": True, "interactions": False},
        "dropped_categoricals": list(DROP_CATS),
        "design_width": {"plain": w_plain, "spline": w_spline, "pipeline_logit_onehot_equiv": 1506},
        "test": {"rows": ev["test_rows"], "expected_rows": EXPECTED_TEST_ROWS,
                 "unweighted": True, "max_prob_sum_err": ev["max_prob_sum_err"]},
        "plain_vs_pipeline_logit_abs": plain_vs_anchor,
        "knot_robustness": knot_rob,
        "shape_background": {"slice": "2015 eval (test) current-origin", "n": int(y_bg.shape[0]),
                             "label": "test-slice analogue of the Ch.3 hazards, not the same curves"},
        "shape_summary": shapes,
        "wall_sec": round(time.time() - t0, 1), "peak_rss_gb": round(_rss_gb(), 2),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    log(f"wrote {HERE / 'results.json'}  | wall {results['wall_sec']}s  peak {results['peak_rss_gb']}GB")
    return results


if __name__ == "__main__":
    main()
