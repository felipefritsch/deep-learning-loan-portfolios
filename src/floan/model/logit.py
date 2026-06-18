"""M7 — Multinomial-logit benchmark (Model B, ``02_LOAN_LEVEL §5``).

The linear baseline of the headline comparison: a PyTorch **0-hidden-layer network**
(``nn.Linear`` over [standardized continuous ‖ one-hot drop-first categoricals ‖ binary])
fitted on the tuning window k=2015, scored by out-of-sample mean NLL on that window's
frozen test slice. Implementing the logit *as a torch net* means it shares the exact
data, importance-weighted loss and NLL evaluator (``torch_common.py``) with the M8+ deep
nets, so every later NLL difference is attributable to architecture alone (``§5``).

What this script does
---------------------
1. Fit the scaler + vocab on the k=2015 **train slice only** (``data.fit_window``), then
   preload the three split slices (train ≤ Dec-2013, val = label-year 2014, test =
   label-year 2015) into RAM as index-encoded arrays — dev scale fits comfortably
   (``data.py`` note); the full-scale loop (M10) swaps back to the streaming loader.
2. Train the logit with importance-weighted cross-entropy, Adam + minibatches, **L2
   selected on the val slice** (a small weight-decay grid, early stopping on val NLL).
3. Score the val-selected model on the frozen test slice and compare to the M5
   empirical-matrix floor (the no-covariate benchmark): the logit must beat it
   out-of-sample (``Accept`` — if not, stop and debug).
4. **sklearn cross-check** (numerical sanity, ``§5``): on a large (200k-row) train
   subsample, fit the *same* weighted multinomial-logit objective with both this code's
   torch model (full-batch, float64 LBFGS) and ``sklearn.LogisticRegression``, sharing one
   ridge (sklearn ``C`` ⇔ torch ``1/(2C)``); their NLLs must agree to ~1e-3. The check uses
   the design minus the two high-cardinality geos (MSA, zip3) and a moderate ridge so the
   lbfgs fit stays well-conditioned (the validated machinery — softmax, weighted CE, ridge
   — is identical); see the ``CROSS_CHECK_ROWS`` note on why not the full 1M.
5. Optional **augmented logit** (``§5``): + squared loan age + binned incentive — how much
   of any NN edge a hand-crafted nonlinearity already recovers.
6. Write a self-describing run folder (metrics + scaler/vocab + checkpoint + git/manifest
   provenance + window id).

Run:
    .venv/bin/python dev/model/logit.py --smoke          # CPU pipeline check (≤100k rows)
    .venv/bin/python dev/model/logit.py                  # full dev run, window k=2015
    .venv/bin/python dev/model/logit.py --verify-only    # re-run Accept checks on output
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F_
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from floan.model import config
from floan.model import data as D  # M6 loader: window_spec / _masked_scan / fit_window
from floan.model import features as F  # M6 feature pipeline: Scaler / Vocab / one-hot / encode_frame
from floan.model import torch_common as tc

_REPO = Path(__file__).resolve().parents[2]

# --- Hyperparameters (tuning window only; frozen for the M10 loop later) -----------
WD_GRID = [0.0, 1e-6, 1e-5, 1e-4]   # L2 (weight-decay) grid, selected on val NLL (§5)
LR = 5e-3
BATCH_SIZE = 8192
MAX_EPOCHS = 25
PATIENCE = 4
SEED = 0
EVAL_BATCH = 16384

# sklearn cross-check (§5): a large train subsample, one shared ridge, the two high-card
# geos excluded. Why 200k and not the full 1M the spec floats: a multinomial-logit lbfgs
# fit's cost scales ~n² here (per-iteration cost ∝ n, and the iteration count grows with
# the conditioning, which itself scales with Σweights ∝ n), so independently grinding two
# solvers to a shared optimum on 1M rows runs for many minutes. 200k makes the per-iter
# cost small enough that both converge in ~1-2 min while remaining a perfectly valid
# numerical-sanity sample (the check's validity is n-independent — it asks whether two
# implementations of the *same* objective agree). The ridge is deliberately non-trivial
# (C=0.1 ⇒ a genuine ½·(1/C)‖W‖² penalty) so the over-parameterised multinomial's softmax
# gauge direction is well-conditioned rather than near-flat. This regularised fit is a
# sanity probe, not the deployed model (that is the val-selected streaming logit above).
CROSS_CHECK_ROWS = 200_000
CROSS_CHECK_C = 0.1
CROSS_CHECK_DROP = ("Metropolitan Statistical Area (MSA)", "Zip Code Short")
LBFGS_MAX_ITER = 200          # torch reference solver cap (well-conditioned ⇒ ample)
SK_MAX_ITER = 2000            # sklearn lbfgs cap

# Augmented logit (§5): incentive binned into deciles (one-hot, drop-first).
N_INC_BINS = 10


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model — the 0-hidden-layer net (= multinomial logistic regression)
# ---------------------------------------------------------------------------
class LogitNet(nn.Module):
    """Single affine map to the 7 class logits — paper §2.2's depth-0 network."""

    def __init__(self, in_features: int, n_classes: int = F.N_CLASSES):
        super().__init__()
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Design assembly — [cont ‖ one-hot(cat) ‖ bin] from an index-encoded batch
# ---------------------------------------------------------------------------
def _build_x(batch: dict, vocab: F.Vocab, device) -> torch.Tensor:
    oh = F.one_hot_block(np.asarray(batch["cat"]), vocab)          # drop-first dummies (§2)
    x = np.concatenate([np.asarray(batch["cont"]), oh, np.asarray(batch["bin"])], axis=1)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def make_logit_predict(vocab: F.Vocab):
    """The ``predict(model, batch, device) -> logits`` callback torch_common expects."""
    def predict(model, batch, device):
        return model(_build_x(batch, vocab, device))
    return predict


def design_width(scaler: F.Scaler, vocab: F.Vocab) -> dict:
    onehot = sum(vocab.vocab_size(c) - 1 for c in vocab.cols)
    cont, binary = len(scaler.cols), len(F.BINARY)
    return {"n_continuous": cont, "onehot_width": onehot, "n_binary": binary,
            "in_features": cont + onehot + binary}


# ---------------------------------------------------------------------------
# Split loading — preload index-encoded arrays once (dev scale fits in RAM)
# ---------------------------------------------------------------------------
def load_split(variant: str, k: int, split: str, scaler: F.Scaler, vocab: F.Vocab,
               cap: int | None = None) -> dict:
    """Index-encode window k's ``split`` slice into NumPy arrays (the M6 masking +
    featurization, materialised once instead of re-streamed each epoch). Also keeps the
    raw ``Loan Age`` / ``incentive`` columns for the augmented logit."""
    pool_dir, bounds = D.window_spec(variant, k, split)
    lf = D._masked_scan(pool_dir, bounds)
    if cap is not None:
        lf = lf.limit(cap)
    df = F.prepare_raw(lf.collect())
    enc = F.encode_frame(df, scaler, vocab, encode="index")
    enc["_age"] = df.get_column("Loan Age").cast(pl.Float64).to_numpy()
    enc["_inc"] = df.get_column("incentive").cast(pl.Float64).to_numpy()
    return enc


def iter_enc(enc: dict, batch_size: int):
    """Yield sequential minibatch dicts from a preloaded split (deterministic eval order)."""
    n = enc["y"].shape[0]
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        yield {kk: enc[kk][sl] for kk in ("cont", "cat", "bin", "y", "w")}


# ---------------------------------------------------------------------------
# Train — weighted CE, Adam, early stopping on val NLL
# ---------------------------------------------------------------------------
def train_logit(enc_train: dict, enc_val: dict, predict, in_features: int,
                weight_decay: float, *, device, seed: int = SEED,
                batch_size: int = BATCH_SIZE, max_epochs: int = MAX_EPOCHS,
                patience: int = PATIENCE, lr: float = LR) -> tuple:
    """One L2 setting: train to the best val NLL (early stopping), return that model."""
    tc.set_seed(seed)
    model = LogitNet(in_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = enc_train["y"].shape[0]
    rng = np.random.default_rng(seed)
    best = {"val_nll": float("inf"), "state": None, "epoch": -1}
    history, bad = [], 0
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            b = {kk: enc_train[kk][idx] for kk in ("cont", "cat", "bin")}
            y = tc._as_long(enc_train["y"][idx], device)
            w = tc._as_f32(enc_train["w"][idx], device)
            opt.zero_grad()
            tc.weighted_ce(predict(model, b, device), y, w).backward()
            opt.step()
        val = tc.evaluate_nll(model, iter_enc(enc_val, EVAL_BATCH), predict, device)
        history.append({"epoch": epoch, "val_nll": val["weighted_nll"]})
        if val["weighted_nll"] < best["val_nll"] - 1e-7:
            best = {"val_nll": val["weighted_nll"],
                    "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best["state"])
    return model, best, history


def fit_l2_grid(enc_train, enc_val, predict, in_features, wd_grid, *, device,
                **kw) -> tuple:
    """Train one logit per L2 value; select the model with the lowest val NLL (§5)."""
    runs, best = {}, {"wd": None, "val_nll": float("inf"), "model": None}
    for wd in wd_grid:
        model, info, hist = train_logit(enc_train, enc_val, predict, in_features, wd,
                                        device=device, **kw)
        runs[wd] = {"val_nll": info["val_nll"], "best_epoch": info["epoch"],
                    "n_epochs": len(hist)}
        print(f"    wd={wd:<7g} val_nll={info['val_nll']:.6f}  "
              f"(best@{info['epoch']}, {len(hist)} epochs)")
        if info["val_nll"] < best["val_nll"]:
            best = {"wd": wd, "val_nll": info["val_nll"], "model": model}
    return best, runs


# ---------------------------------------------------------------------------
# Augmented logit (§5) — widen cont (age²) + bin (incentive deciles)
# ---------------------------------------------------------------------------
def fit_augment(enc_train: dict) -> dict:
    """Train-only standardization for age² + interior decile edges for incentive."""
    a2 = enc_train["_age"] ** 2
    mu = float(np.nanmean(a2))
    sd = float(np.nanstd(a2))
    inc = enc_train["_inc"]
    edges = np.quantile(inc[np.isfinite(inc)], np.linspace(0, 1, N_INC_BINS + 1))[1:-1]
    return {"age2_mu": mu, "age2_sd": sd if sd > 1e-12 else 1.0,
            "inc_edges": edges.tolist()}


def augment_enc(enc: dict, aug: dict) -> dict:
    """Return a copy of ``enc`` with age² appended to ``cont`` and the incentive-decile
    one-hot (drop-first) appended to ``bin`` (0/1 columns ride the binary passthrough)."""
    a2 = enc["_age"] ** 2
    a2 = np.where(np.isfinite(a2), a2, aug["age2_mu"])             # null age → centre (z=0)
    a2z = ((a2 - aug["age2_mu"]) / aug["age2_sd"]).astype(np.float32)[:, None]
    inc, edges = enc["_inc"], np.asarray(aug["inc_edges"])
    valid = np.isfinite(inc)
    bin_idx = np.digitize(np.where(valid, inc, -np.inf), edges)    # 0..N_INC_BINS-1
    oneh = np.zeros((inc.shape[0], N_INC_BINS), np.float32)
    oneh[np.arange(inc.shape[0])[valid], bin_idx[valid]] = 1.0
    oneh = oneh[:, 1:]                                             # drop-first; null → all-0
    new = dict(enc)
    new["cont"] = np.concatenate([enc["cont"], a2z], axis=1)
    new["bin"] = np.concatenate([enc["bin"], oneh], axis=1)
    return new


# ---------------------------------------------------------------------------
# sklearn cross-check (§5) — same objective, two solvers, NLL must agree ~1e-3
# ---------------------------------------------------------------------------
def _weighted_unweighted_nll(P: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple:
    ce = -np.log(np.clip(P[np.arange(y.shape[0]), y], 1e-300, 1.0))
    return float((w * ce).sum() / w.sum()), float(ce.mean())


def sklearn_cross_check(enc_train: dict, vocab: F.Vocab, *, n_rows: int,
                        C: float = CROSS_CHECK_C, seed: int = SEED) -> dict:
    """Fit the identical weighted multinomial-logit + ridge objective with sklearn
    (lbfgs, float64) and this code's torch model (full-batch float64 LBFGS), on a common
    dense design, and compare NLL. sklearn minimises ``C·Σ wᵢCEᵢ + ½‖W‖²``; torch
    minimises ``Σ wᵢCEᵢ + 1/(2C)·‖W‖²`` — the same function up to the factor ``C``, so the
    minimiser (hence NLL) is shared. Intercept is unpenalised on both sides."""
    drop = set(CROSS_CHECK_DROP)
    sub_cols = [c for c in vocab.cols if c not in drop]
    sub_pos = [vocab.cols.index(c) for c in sub_cols]
    sub_vocab = F.Vocab(cols=sub_cols, maps={c: vocab.maps[c] for c in sub_cols})

    n = min(n_rows, enc_train["y"].shape[0])
    cont = enc_train["cont"][:n].astype(np.float64)
    binb = enc_train["bin"][:n].astype(np.float64)
    oh = F.one_hot_block(enc_train["cat"][:n][:, sub_pos], sub_vocab).astype(np.float64)
    X = np.concatenate([cont, oh, binb], axis=1)
    y = enc_train["y"][:n].astype(np.int64)
    w = enc_train["w"][:n].astype(np.float64)

    # sklearn (multinomial via lbfgs for >2 classes; intercept unpenalised). The numpy
    # overflow/invalid warnings are benign lbfgs trial-step probes (the objective itself
    # is log-sum-exp stable); silence them but keep ConvergenceWarning meaningful.
    converged = True
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=SK_MAX_ITER,
                             tol=1e-9, fit_intercept=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            clf.fit(X, y, sample_weight=w)
        except ConvergenceWarning:
            converged = False
            warnings.simplefilter("ignore")
            clf.fit(X, y, sample_weight=w)
        P_sk = clf.predict_proba(X)
    nll_sk_w, nll_sk_u = _weighted_unweighted_nll(P_sk, y, w)

    # torch reference — same objective, full-batch float64 LBFGS --------------------
    tc.set_seed(seed)
    dev = torch.device("cpu")
    Xt = torch.tensor(X, dtype=torch.float64, device=dev)
    yt = torch.tensor(y, dtype=torch.long, device=dev)
    wt = torch.tensor(w, dtype=torch.float64, device=dev)
    model = LogitNet(X.shape[1]).double().to(dev)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_MAX_ITER,
                            history_size=10, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)
    reg = 1.0 / (2.0 * C)

    def closure():
        opt.zero_grad()
        ce = F_.cross_entropy(model(Xt), yt, reduction="none")
        loss = (wt * ce).sum() + reg * (model.linear.weight ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        P_t = torch.softmax(model(Xt), dim=1).cpu().numpy()
    nll_t_w, nll_t_u = _weighted_unweighted_nll(P_t, y, w)

    # coefficient agreement (secondary; both use the full K-class param + same ridge) --
    W_sk = np.concatenate([clf.coef_, clf.intercept_[:, None]], axis=1)
    W_t = np.concatenate([model.linear.weight.detach().numpy(),
                          model.linear.bias.detach().numpy()[:, None]], axis=1)
    return {
        "n_rows": int(n), "C": C, "ridge_torch_coef": reg,
        "features_excluded": list(CROSS_CHECK_DROP), "design_width": int(X.shape[1]),
        "sklearn_converged": bool(converged), "sklearn_n_iter": int(np.max(clf.n_iter_)),
        "sklearn": {"nll_weighted": nll_sk_w, "nll_unweighted": nll_sk_u},
        "torch": {"nll_weighted": nll_t_w, "nll_unweighted": nll_t_u},
        "abs_diff_weighted": abs(nll_sk_w - nll_t_w),
        "abs_diff_unweighted": abs(nll_sk_u - nll_t_u),
        "coef_max_abs_diff": float(np.abs(W_sk - W_t).max()),
    }


# ---------------------------------------------------------------------------
# Empirical-matrix floor (M5 output) — the no-covariate benchmark to beat
# ---------------------------------------------------------------------------
def empirical_floor(variant: str, k: int) -> dict | None:
    p = config.MODELS / "benchmarks" / variant / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    pw = m["per_window"].get(str(k))
    if pw is None:
        return None
    return {"empirical": pw["nll_empirical"], "bucketed": pw["nll_bucketed"],
            "pooled_empirical": m["pooled"]["nll_empirical"]}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(variant: str, k: int, smoke: bool, device_name: str) -> Path:
    device = tc.resolve_device(device_name)
    cap = 100_000 if smoke else None
    wd_grid = [0.0, 1e-5] if smoke else WD_GRID
    max_epochs = 4 if smoke else MAX_EPOCHS
    xcheck_rows = 50_000 if smoke else CROSS_CHECK_ROWS
    tag = f"k{k}" + ("_smoke" if smoke else "")
    print(f"=== M7 logit benchmark  variant={variant}  window k={k}  "
          f"device={device}  smoke={smoke} ===")

    # 1. fit scaler/vocab on the train slice only; preload the three split slices ------
    scaler, vocab = D.fit_window(variant, k)
    dims = design_width(scaler, vocab)
    print(f"design: {dims['n_continuous']} cont + {dims['onehot_width']} one-hot "
          f"+ {dims['n_binary']} bin = {dims['in_features']} features "
          f"({len(vocab.cols)} categoricals)")
    enc_train = load_split(variant, k, "train", scaler, vocab, cap=cap)
    enc_val = load_split(variant, k, "val", scaler, vocab, cap=cap)
    enc_test = load_split(variant, k, "test", scaler, vocab, cap=cap)
    print(f"rows: train={enc_train['y'].shape[0]:,} (Σw={enc_train['w'].sum():,.0f})  "
          f"val={enc_val['y'].shape[0]:,}  test={enc_test['y'].shape[0]:,}")
    predict = make_logit_predict(vocab)

    # 2. L2-on-val grid -------------------------------------------------------------
    print("L2 selection (val NLL):")
    best, wd_runs = fit_l2_grid(enc_train, enc_val, predict, dims["in_features"],
                                wd_grid, device=device, seed=SEED, max_epochs=max_epochs)
    model = best["model"]
    print(f"  selected wd={best['wd']:g}  val_nll={best['val_nll']:.6f}")

    # 3. test NLL + floor comparison -------------------------------------------------
    test = tc.evaluate_nll(model, iter_enc(enc_test, EVAL_BATCH), predict, device)
    train_fit = tc.evaluate_nll(model, iter_enc(enc_train, EVAL_BATCH), predict, device)
    floor = empirical_floor(variant, k)
    beats = None
    if floor is not None:
        margin = floor["empirical"] - test["unweighted_nll"]
        beats = {"test_nll": test["unweighted_nll"], "empirical_floor": floor["empirical"],
                 "bucketed_floor": floor["bucketed"], "margin": margin,
                 "passed": test["unweighted_nll"] < floor["empirical"]}
        print(f"test NLL={test['unweighted_nll']:.6f}  vs empirical floor "
              f"{floor['empirical']:.6f}  → margin {margin:+.6f} "
              f"({'BEATS' if beats['passed'] else 'DOES NOT BEAT'})")

    # 4. augmented logit (optional) --------------------------------------------------
    aug_block = None
    aug = fit_augment(enc_train)
    at, av, ate = (augment_enc(enc_train, aug), augment_enc(enc_val, aug),
                   augment_enc(enc_test, aug))
    aug_in = dims["in_features"] + 1 + (N_INC_BINS - 1)
    print(f"augmented logit (+age², +{N_INC_BINS}-bin incentive): "
          f"{aug_in} features, wd={best['wd']:g}")
    amodel, ainfo, _ = train_logit(at, av, predict, aug_in, best["wd"], device=device,
                                   seed=SEED, max_epochs=max_epochs)
    atest = tc.evaluate_nll(amodel, iter_enc(ate, EVAL_BATCH), predict, device)
    aug_block = {"in_features": aug_in, "weight_decay": best["wd"], "n_inc_bins": N_INC_BINS,
                 "val_nll": ainfo["val_nll"], "test_nll": atest["unweighted_nll"],
                 "augment": aug}
    print(f"  augmented val_nll={ainfo['val_nll']:.6f}  test_nll={atest['unweighted_nll']:.6f}")

    # 5. sklearn cross-check ---------------------------------------------------------
    print(f"sklearn cross-check ({xcheck_rows:,} rows, C={CROSS_CHECK_C}, "
          f"excl {list(CROSS_CHECK_DROP)})…")
    xcheck = sklearn_cross_check(enc_train, vocab, n_rows=xcheck_rows)
    xcheck["passed"] = xcheck["abs_diff_weighted"] < 1e-3
    print(f"  sklearn NLL(w)={xcheck['sklearn']['nll_weighted']:.6f}  "
          f"torch NLL(w)={xcheck['torch']['nll_weighted']:.6f}  "
          f"|Δ|={xcheck['abs_diff_weighted']:.2e}  "
          f"(coef max|Δ|={xcheck['coef_max_abs_diff']:.2e}, "
          f"{'AGREE' if xcheck['passed'] else 'DISAGREE'})")

    # 6. write run folder ------------------------------------------------------------
    run = config.MODELS / "logit" / variant / tag
    run.mkdir(parents=True, exist_ok=True)
    F.save_pipeline(run, scaler, vocab)
    torch.save(model.state_dict(), run / "model.pt")
    torch.save(amodel.state_dict(), run / "model_augmented.pt")
    manifest = json.loads((config.TRAINING_DIR / variant / "manifest.json").read_text())
    metrics = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "multinomial_logit", "variant": variant, "window_k": k,
        "tuning_window": (k == config.TUNING_YEAR), "smoke": smoke,
        "device": str(device), "seed": SEED, "git_commit": _git_commit(),
        "source": {
            "train_pool_content_hash": manifest["train_pool"]["content_hash"],
            "eval_pool_content_hash": manifest["eval_pool"]["content_hash"],
            "panel_glob": config.PANEL_GLOB,
        },
        "features": {**dims, "n_categoricals": len(vocab.cols),
                     "vocab_sizes": dict(zip(vocab.cols, vocab.vocab_sizes))},
        "rows": {"train": int(enc_train["y"].shape[0]),
                 "train_sum_w": float(enc_train["w"].sum()),
                 "val": int(enc_val["y"].shape[0]), "test": int(enc_test["y"].shape[0])},
        "hyperparams": {"lr": LR, "batch_size": BATCH_SIZE, "max_epochs": max_epochs,
                        "patience": PATIENCE, "wd_grid": wd_grid},
        "l2_selection": {"per_wd": {str(w): r for w, r in wd_runs.items()},
                         "selected_wd": best["wd"]},
        "logit": {"val_nll": best["val_nll"], "train_nll": train_fit["unweighted_nll"],
                  "test_nll": test["unweighted_nll"],
                  "test_nll_weighted": test["weighted_nll"],
                  "test_n_rows": test["n_rows"], "test_sum_w": test["sum_w"]},
        "augmented_logit": aug_block,
        "empirical_floor": floor,
        "beats_empirical": beats,
        "cross_check": xcheck,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {run / 'metrics.json'}")
    return run


# ---------------------------------------------------------------------------
# Verify — every M7 Accept criterion, with evidence
# ---------------------------------------------------------------------------
def verify(variant: str, k: int) -> None:
    run = config.MODELS / "logit" / variant / f"k{k}"
    metrics = json.loads((run / "metrics.json").read_text())
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    # --- Accept 1: PyTorch and sklearn NLL agree to ~1e-3 -------------------------
    print("\n[1] sklearn cross-check: torch vs sklearn NLL agree to ~1e-3")
    xc = metrics["cross_check"]
    check(f"sklearn converged on {xc['n_rows']:,}-row design "
          f"({xc['design_width']} feats, excl {xc['features_excluded']})",
          xc["sklearn_converged"], f"lbfgs n_iter={xc['sklearn_n_iter']}")
    check(f"|Δ weighted-NLL| < 1e-3  (sklearn {xc['sklearn']['nll_weighted']:.6f} "
          f"vs torch {xc['torch']['nll_weighted']:.6f})",
          xc["abs_diff_weighted"] < 1e-3, f"|Δ|={xc['abs_diff_weighted']:.3e}")
    check(f"|Δ unweighted-NLL| < 1e-3  (sklearn {xc['sklearn']['nll_unweighted']:.6f} "
          f"vs torch {xc['torch']['nll_unweighted']:.6f})",
          xc["abs_diff_unweighted"] < 1e-3, f"|Δ|={xc['abs_diff_unweighted']:.3e}")
    # Coefficients are reported as a diagnostic, not gated: the over-parameterised
    # multinomial has a per-feature gauge freedom (adding a constant to all 7 class
    # weights leaves softmax unchanged) that the shared mild ridge makes very flat, so
    # the two solvers can stop at slightly different points along it while predicting
    # identically. NLL — invariant to the gauge — is the criterion.
    print(f"       (diagnostic) coefficient max|Δ| = {xc['coef_max_abs_diff']:.3e} "
          f"(gauge-flat under the shared ridge; NLL is what is asserted)")

    # --- Accept 2: beats the empirical matrix out-of-sample -----------------------
    print("\n[2] logit beats the M5 empirical-matrix floor out-of-sample (uses covariates)")
    be = metrics["beats_empirical"]
    check("M5 empirical-matrix metrics present for comparison", be is not None)
    if be is not None:
        check(f"test NLL {be['test_nll']:.6f} < empirical floor {be['empirical_floor']:.6f}",
              be["passed"], f"margin {be['margin']:+.6f}")
        print(f"       (bucketed-matrix floor was {be['bucketed_floor']:.6f}; "
              f"augmented-logit test NLL "
              f"{metrics['augmented_logit']['test_nll']:.6f})")

    # --- Sanity: eval weights are 1 ⇒ weighted==unweighted on the frozen test ------
    print("\n[sanity] frozen test slice: unthinned (weighted NLL == plain NLL)")
    lg = metrics["logit"]
    check("test weighted-NLL == unweighted-NLL (eval pool weight≡1)",
          abs(lg["test_nll"] - lg["test_nll_weighted"]) < 1e-9,
          f"|Δ|={abs(lg['test_nll'] - lg['test_nll_weighted']):.2e}")
    check(f"test rows == eval-pool label-year {k} count",
          lg["test_n_rows"] == metrics["rows"]["test"], f"n={lg['test_n_rows']:,}")

    # --- Accept 3: run folder written -------------------------------------------
    print("\n[3] self-describing run folder written")
    for fname in ("metrics.json", "scaler.json", "vocab.json", "model.pt"):
        check(f"{fname} present", (run / fname).exists())
    check("provenance recorded (git commit + train/eval manifest hashes + window id)",
          bool(metrics["git_commit"]) and bool(metrics["source"]["train_pool_content_hash"])
          and metrics["window_k"] == k)

    print(f"\n{'ALL ACCEPT CHECKS PASS' if ok else 'SOME CHECKS FAILED'} "
          f"(variant={variant}, k={k})")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/verify the multinomial-logit benchmark (M7).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    ap.add_argument("--smoke", action="store_true",
                    help="CPU pipeline check on ≤100k rows (standing rule 3)")
    ap.add_argument("--verify-only", action="store_true",
                    help="run the Accept checks against existing output (no rebuild)")
    args = ap.parse_args()
    if not args.verify_only:
        build(args.variant, args.k, args.smoke, args.device)
    if not args.smoke:
        verify(args.variant, args.k)


if __name__ == "__main__":
    main()
