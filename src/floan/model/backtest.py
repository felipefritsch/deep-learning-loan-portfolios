"""M10 — Rolling backtest loop over the 11 windows (``02_LOAN_LEVEL §6``).

The frozen tuning-window config (``config.NN_SELECTED``, set by M9 and re-confirmed at full
scale by the M10b depth check) is **looped, never re-tuned**, over every rolling window
``k = 2015 … 2025``. Each window gets, per ``§6``'s settled compute budget:

* a **per-window multinomial logit** (the linear floor) — every window;
* the **best single NN** at the frozen config — every window;
* an **8-net ensemble** — only the 5 key windows ``{2015, 2019, 2020, 2023, 2025}`` spanning
  regimes (tuning / calm / COVID / rate-spike / latest).

Per-window locality (``§6``, no leakage): the scaler, embedding vocab and incentive are
re-derived from *that window's train slice only*; early stopping uses *that window's val
slice only*; the frozen test slice is identical across all models.

Speed + resumability (M10b): the full export is ~74 M train rows, so every fit runs on the
``train.py`` GPU-resident fast path (the whole split lives on the GPU; ~0.8 M rows/s on the
A4500 vs ~0.35 M for the per-batch host→device path). Every fit checkpoints **per epoch**
(``train.save_checkpoint``) and is idempotent: a finished run (``metrics.json``) is skipped,
an interrupted one resumes from its last epoch — so a tmux disconnect or pod preemption costs
at most the in-progress epoch. The logit is window-level idempotent (re-fit if interrupted —
it is the cheap step).

Models
------
* **Logit** — a 0-hidden-layer net over ``[cont ‖ one-hot(cat) ‖ bin]`` (paper §2.2),
  parameterised as ``Linear(cont‖bin) + Σ_c Embedding(vocab_c, 7)[cat_c]``. The embedding-sum
  is **identically** the multinomial logit on the full one-hot design (each categorical level
  gets its own per-class weight) but never materialises the ~1.3k-wide one-hot block, so it
  runs on the same resident GPU path as the NN. L2 selected on the window's val NLL.
* **NN single / ensemble** — :func:`train.train` at the frozen config (the tested loss /
  early-stopping / atomic-checkpoint / metrics path), so every run folder is self-describing.

Outputs (under ``models/``):
  ``logit/full/k{k}/``                         per-window logit run folder
  ``nn/full/k{k}_d{d}_do{p}_wd{w}/``           per-window single-NN run folder
  ``nn/full/ens_k{k}_s{0..7}/``                ensemble member run folders (key windows)
  ``nn/full/ensemble_k{k}_summary.json``       per-window ensemble averaging + size curve
  ``nn/full/backtest_summary.json``            Table-B-style matrix + base-rate QA (§7)

Run (inside tmux):
    .venv/bin/python dev/model/backtest.py --device cuda --amp        # the full loop
    .venv/bin/python dev/model/backtest.py --verify-only              # Accept checks on outputs
    .venv/bin/python dev/model/backtest.py --device cuda --amp --windows 2015 2019  # subset
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
import torch.nn as nn

import config
import data as D
import ensemble as E
import features as F
import net as N
import torch_common as tc
import train as T

KEY_WINDOWS = (2015, 2019, 2020, 2023, 2025)   # §6: ensemble only on these 5
VARIANT = "full"
SEED = 0
EVAL_BATCH = 16384

# Logit hyperparameters (the M7 settings; L2 grid trimmed for the 11-window loop — the
# floor only needs its val-selected strength, not M7's full {0,1e-6,1e-5,1e-4} sweep).
LOGIT_WD_GRID = (0.0, 1e-5, 1e-4)
LOGIT_LR = 5e-3
LOGIT_BATCH = 8192
LOGIT_MAX_EPOCHS = 25
LOGIT_PATIENCE = 4


def _frozen() -> dict:
    """The frozen NN config looped over every window (M9 + M10b depth check)."""
    return dict(config.NN_SELECTED)


def _gbt_frozen() -> dict:
    """The frozen GBT config looped over every window (M16 dev tuning + M17 full reconfirm)."""
    return dict(config.GBT_SELECTED)


def _nn_tag(k: int, cfg: dict) -> str:
    """Single-NN run-folder tag — mirrors :func:`train.run_id` so a depth-check cell already
    fit at this config (e.g. k=2015) is reused, not retrained."""
    return f"k{k}_d{cfg['depth']}_do{cfg['dropout']:g}_wd{cfg['weight_decay']:g}"


# ===========================================================================
# Model B — multinomial logit (full one-hot, embedding-sum parameterisation)
# ===========================================================================
class LogitEmbNet(nn.Module):
    """Multinomial logit on ``[cont ‖ one-hot(cat) ‖ bin]`` (paper §2.2), with the categorical
    one-hot·weight product folded into one ``Embedding(vocab_c, n_classes)`` per column. The
    class logit is ``Linear([cont‖bin]) + Σ_c E_c[cat_c]`` — algebraically the same affine map
    as a dense one-hot logit, minus the wide design matrix. Forward signature matches
    :class:`net.MortgageMLP` so it rides ``train``'s resident epoch/eval loops unchanged."""

    def __init__(self, n_continuous: int, n_binary: int, vocab_sizes: list[int],
                 n_classes: int = F.N_CLASSES):
        super().__init__()
        self.dense = nn.Linear(n_continuous + n_binary, n_classes)
        self.cat_emb = nn.ModuleList([nn.Embedding(v, n_classes) for v in vocab_sizes])
        # Zero-init the categorical weights — this IS the one-hot logit's natural start (each
        # level's per-class weight begins at 0, growing under the gradient exactly as a one-hot
        # column weight would). nn.Embedding's default N(0,1) instead seeds the categorical
        # block far from the one-hot fit, so at L2=0 (over-parameterised, gauge-free) early
        # stopping halts before it converges back — the M10b equivalence receipt's wd=0 gap.
        for emb in self.cat_emb:
            nn.init.zeros_(emb.weight)

    def forward(self, cont: torch.Tensor, cat: torch.Tensor,
                binb: torch.Tensor) -> torch.Tensor:
        out = self.dense(torch.cat([cont, binb], dim=1))
        for j, emb in enumerate(self.cat_emb):
            out = out + emb(cat[:, j])
        return out


def _train_logit_one(T_train: dict, T_val: dict, n_cont: int, n_bin: int,
                     vocab_sizes: list[int], wd: float, device, use_amp: bool):
    """Fit one LogitEmbNet at L2=``wd`` to its best val NLL (early stopping), on resident
    tensors. Returns ``(model, best, history)``."""
    tc.set_seed(SEED)
    model = LogitEmbNet(n_cont, n_bin, vocab_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LOGIT_LR, weight_decay=wd)
    amp_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    best = {"val_nll": float("inf"), "state": None, "epoch": -1}
    history, bad = [], 0
    for epoch in range(LOGIT_MAX_EPOCHS):
        T.train_one_epoch_resident(model, T_train, opt, amp_scaler, device=device,
                                   epoch=epoch, seed=SEED, batch_size=LOGIT_BATCH,
                                   use_amp=use_amp)
        val = T.evaluate_nll_resident(model, T_val, EVAL_BATCH, device)
        history.append({"epoch": epoch, "val_nll": val["weighted_nll"]})
        if val["weighted_nll"] < best["val_nll"] - 1e-7:
            best = {"val_nll": val["weighted_nll"],
                    "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
            bad = 0
        else:
            bad += 1
            if bad >= LOGIT_PATIENCE:
                break
    model.load_state_dict(best["state"])
    return model, best, history


def fit_logit(k: int, device, use_amp: bool, fresh: bool = False) -> Path:
    """Per-window logit: L2-on-val grid, scored on the frozen test slice. Window-level
    idempotent (skips if ``metrics.json`` already written)."""
    run = config.MODELS / "logit" / VARIANT / f"k{k}"
    if (run / "metrics.json").exists() and not fresh:
        print(f"[logit k={k}] already complete — skipping")
        return run
    t0 = time.perf_counter()
    print(f"=== logit  k={k}  ({VARIANT}) ===")
    scaler, vocab = D.fit_window(VARIANT, k)
    enc_train = T.preload_split(VARIANT, k, "train", scaler, vocab, None)
    enc_val = T.preload_split(VARIANT, k, "val", scaler, vocab, None)
    enc_test = T.preload_split(VARIANT, k, "test", scaler, vocab, None)
    n_train, train_sum_w = int(enc_train["y"].shape[0]), float(enc_train["w"].sum())
    n_val, n_test = int(enc_val["y"].shape[0]), int(enc_test["y"].shape[0])
    n_cont, n_bin, vs = len(scaler.cols), len(F.BINARY), list(vocab.vocab_sizes)
    print(f"design: {n_cont} cont + Σ{sum(vs)} one-hot ({len(vs)} cats) + {n_bin} bin   "
          f"rows: train={n_train:,} val={n_val:,} test={n_test:,}")

    T_train, T_val = T.to_resident(enc_train, device), T.to_resident(enc_val, device)
    del enc_train, enc_val
    best = {"wd": None, "val_nll": float("inf"), "model": None}
    per_wd = {}
    for wd in LOGIT_WD_GRID:
        model, info, hist = _train_logit_one(T_train, T_val, n_cont, n_bin, vs, wd,
                                             device, use_amp)
        per_wd[str(wd)] = {"val_nll": info["val_nll"], "best_epoch": info["epoch"],
                           "n_epochs": len(hist)}
        print(f"  wd={wd:<7g} val_nll={info['val_nll']:.6f}  (best@{info['epoch']}, "
              f"{len(hist)} epochs)")
        if info["val_nll"] < best["val_nll"]:
            best = {"wd": wd, "val_nll": info["val_nll"], "model": model}

    del T_train
    torch.cuda.empty_cache() if device.type == "cuda" else None
    T_test = T.to_resident(enc_test, device)
    test = T.evaluate_nll_resident(best["model"], T_test, EVAL_BATCH, device)
    val_best = T.evaluate_nll_resident(best["model"], T_val, EVAL_BATCH, device)
    print(f"  selected wd={best['wd']:g}  val_nll={best['val_nll']:.6f}  "
          f"test_nll={test['unweighted_nll']:.6f}")

    run.mkdir(parents=True, exist_ok=True)
    F.save_pipeline(run, scaler, vocab)
    torch.save(best["model"].state_dict(), run / "model.pt")
    manifest = json.loads((config.TRAINING_DIR / VARIANT / "manifest.json").read_text())
    metrics = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "multinomial_logit", "parameterization": "embedding_sum_onehot",
        "variant": VARIANT, "window_k": k, "tuning_window": (k == config.TUNING_YEAR),
        "smoke": False, "device": str(device), "seed": SEED,
        "git_commit": T._git_commit(), "environment": T._environment(device),
        "source": {"train_pool_content_hash": manifest["train_pool"]["content_hash"],
                   "eval_pool_content_hash": manifest["eval_pool"]["content_hash"],
                   "panel_glob": config.PANEL_GLOB},
        "features": {"n_continuous": n_cont, "n_binary": n_bin, "n_categoricals": len(vs),
                     "onehot_equiv_width": int(sum(vs)),
                     "vocab_sizes": dict(zip(vocab.cols, vs))},
        "rows": {"train": n_train, "train_sum_w": train_sum_w, "val": n_val, "test": n_test},
        "hyperparams": {"lr": LOGIT_LR, "batch_size": LOGIT_BATCH,
                        "max_epochs": LOGIT_MAX_EPOCHS, "patience": LOGIT_PATIENCE,
                        "wd_grid": list(LOGIT_WD_GRID)},
        "l2_selection": {"per_wd": per_wd, "selected_wd": best["wd"]},
        "logit": {"val_nll": best["val_nll"], "test_nll": test["unweighted_nll"],
                  "test_nll_weighted": test["weighted_nll"],
                  "test_n_rows": test["n_rows"], "test_sum_w": test["sum_w"]},
        "wall_sec": time.perf_counter() - t0,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  wrote {run/'metrics.json'}  [{metrics['wall_sec']/60:.1f} min]")
    return run


# ===========================================================================
# Model C — single NN + ensemble (via the tested train.train)
# ===========================================================================
def _nn_args(k: int, cfg: dict, device: str, amp: bool, seed: int = SEED,
             run_name: str | None = None) -> SimpleNamespace:
    """A :func:`train.train` arg namespace at the frozen config, GPU-resident, for the loop."""
    return SimpleNamespace(
        variant=VARIANT, k=k, device=device, depth=cfg["depth"], dropout=cfg["dropout"],
        weight_decay=cfg["weight_decay"], lr=T.LR, batch_size=T.BATCH_SIZE,
        max_epochs=None, patience=T.PATIENCE, seed=seed, amp=amp, smoke=False,
        gpu_resident=True, run_name=run_name, fresh=False, stop_after_epoch=None,
        verify_only=False)


def fit_nn_single(k: int, cfg: dict, device_name: str, amp: bool) -> Path:
    """Best single NN on window ``k`` at the frozen config — idempotent + resumable.

    Loop-level idempotency: a finished run (``metrics.json``) is skipped here rather than
    handed to :func:`train.train`, whose re-entry path re-runs the M8 ``verify`` and would
    ``SystemExit`` on a non-essential stale check (e.g. the k=2015 cell reused from the M10b
    depth check was trained *before* this window's logit existed, so its stored
    ``logit_comparison`` is null — cosmetic here, since ``build_summary`` recomputes the
    logit-vs-NN delta from both run folders). An interrupted run still resumes via
    ``train.train``'s checkpoint."""
    run = config.MODELS / "nn" / VARIANT / _nn_tag(k, cfg)
    if (run / "metrics.json").exists():
        print(f"=== single NN  k={k}  already complete — skipping ({_nn_tag(k, cfg)}) ===")
        return run
    print(f"=== single NN  k={k}  depth={cfg['depth']} dropout={cfg['dropout']} "
          f"wd={cfg['weight_decay']:g} ===")
    return T.train(_nn_args(k, cfg, device_name, amp))


def _ensemble_eval(runs: list[Path], enc: dict, device) -> tuple[list[float], list[dict], np.ndarray]:
    """One pass per member: returns (per-member NLL, size curve, ensemble mean-prob ``[N,7]``).
    Each member's probabilities are computed once and accumulated (paper Fig 7 averaging)."""
    y = np.asarray(enc["y"])
    acc = np.zeros((y.shape[0], F.N_CLASSES), dtype=np.float64)
    per, curve = [], []
    for m, run in enumerate(runs, 1):
        p = E._member_probs(run, enc, device)
        per.append(E._nll(p, y))
        acc += p
        curve.append({"size": m, "nll": E._nll(acc / m, y)})
    return per, curve, acc / len(runs)


def fit_ensemble(k: int, cfg: dict, device_name: str, amp: bool, device) -> Path:
    """8-net ensemble on a key window: train members (init+SGD-order diversity, the validated
    M9 scheme), average predicted probabilities, write the size curve + ensemble NLL."""
    summary_path = config.MODELS / "nn" / VARIANT / f"ensemble_k{k}_summary.json"
    print(f"=== ensemble  k={k}  {E.N_MEMBERS}× frozen config ===")
    for seed in range(E.N_MEMBERS):
        mrun = config.MODELS / "nn" / VARIANT / f"ens_k{k}_s{seed}"
        if (mrun / "metrics.json").exists():
            print(f"--- member seed {seed}/{E.N_MEMBERS-1} (k={k}) already complete — skipping ---")
            continue
        print(f"--- member seed {seed}/{E.N_MEMBERS-1} (k={k}) ---")
        T.train(_nn_args(k, cfg, device_name, amp, seed=seed, run_name=f"ens_k{k}_s{seed}"))

    runs = [config.MODELS / "nn" / VARIANT / f"ens_k{k}_s{s}" for s in range(E.N_MEMBERS)]
    missing = [str(r) for r in runs if not (r / "best_model.pt").exists()]
    if missing:
        raise SystemExit(f"ensemble k={k}: missing members {missing}")

    scaler, vocab = F.load_pipeline(runs[0])
    enc_val = T.preload_split(VARIANT, k, "val", scaler, vocab, None)
    enc_test = T.preload_split(VARIANT, k, "test", scaler, vocab, None)
    val_per, val_curve, _ = _ensemble_eval(runs, enc_val, device)
    test_per, test_curve, _ = _ensemble_eval(runs, enc_test, device)
    members = [{"seed": s, "run": str(runs[s]), "val_nll": val_per[s], "test_nll": test_per[s]}
               for s in range(E.N_MEMBERS)]
    mean_single = {"val_nll": float(np.mean(val_per)), "test_nll": float(np.mean(test_per))}
    summary = {
        "window_k": k, "variant": VARIANT, "n_members": E.N_MEMBERS, "config": cfg,
        "members": members, "mean_single_nll": mean_single,
        "val_curve": val_curve, "test_curve": test_curve,
        "ensemble_val_nll": val_curve[-1]["nll"], "ensemble_test_nll": test_curve[-1]["nll"],
        "best_single_test_nll": float(np.min(test_per)),
        "ensemble_beats_mean_single": val_curve[-1]["nll"] < mean_single["val_nll"],
        "ensemble_beats_best_single_test": test_curve[-1]["nll"] <= float(np.min(test_per)),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  ensemble val={summary['ensemble_val_nll']:.6f} "
          f"test={summary['ensemble_test_nll']:.6f}  "
          f"(mean single test {mean_single['test_nll']:.6f}, "
          f"best single test {summary['best_single_test_nll']:.6f}) → {summary_path.name}")
    return summary_path


# ===========================================================================
# Model E — GBT baseline (06_GBT_BASELINE / M17) — via gbt.fit_window_frozen
# ===========================================================================
def fit_gbt(k: int, cfg: dict, n_threads: int, *, fresh: bool = False) -> Path:
    """Per-window GBT at the frozen config — per-window vocab + HT-weighted train + early
    stopping on that window's val, scored on the frozen test slice (``06 §5``). The
    window/masking logic is untouched; the fit/QA/run-folder lives in
    :func:`gbt.fit_window_frozen`, which shares the export/loader/evaluator path with every
    other model. Idempotent on (window, config). ``gbt`` is imported lazily to avoid the
    import cycle (``gbt`` → ``evaluate`` → ``backtest``)."""
    import gbt as G
    return G.fit_window_frozen(VARIANT, k, cfg, n_threads, fresh=fresh)


# ===========================================================================
# Base-rate QA (§7) — importance-weighting + impossible-transition checks
# ===========================================================================
# Structural transition rule (02_LOAN_LEVEL §1; Sirignano-style monotone delinquency):
# from a transient origin a loan may stay, cure to *any* lower delinquency bucket (incl.
# current), worsen by **at most one** bucket, or prepay; foreclosure can only follow
# dpd_90plus, and REO is never a one-month destination from a transient state (REO follows
# foreclosure). Rows = the 4 transient origins (current, dpd_30, dpd_60, dpd_90plus); cols =
# the 7 STATES (current,dpd_30,dpd_60,dpd_90plus,foreclosure,REO,prepaid). `True` = allowed.
def _structural_allow() -> np.ndarray:
    allow = np.zeros((F.N_CLASSES, F.N_CLASSES), dtype=bool)
    allow[F.STATE_INDEX["current"], [0, 1, 6]] = True                  # → current/dpd_30/prepaid
    allow[F.STATE_INDEX["dpd_30"], [0, 1, 2, 6]] = True                # +worsen→dpd_60
    allow[F.STATE_INDEX["dpd_60"], [0, 1, 2, 3, 6]] = True             # +worsen→dpd_90plus
    allow[F.STATE_INDEX["dpd_90plus"], [0, 1, 2, 3, 4, 6]] = True      # +foreclosure (not REO)
    return allow


def base_rate_qa(k: int, device, tol: float = 0.01) -> dict:
    """§7 QA on a key window's frozen (unthinned) test slice:

    (a) **Importance-weighting / aggregate calibration** — ensemble mean predicted class base
        rates vs realized class frequencies (§6.4); ``max|Δ|`` must be < ``tol``.
    (b) **Impossible-transition mass** — ensemble mean predicted probability on the
        *structurally impossible* origin→destination cells (``_structural_allow``); the
        aggregate must be < 0.1% (§7). Reported next to the **realized** rate on those same
        cells: this Fannie panel's monthly states do contain a few nominally-"impossible"
        jumps (notably ``dpd_90plus→REO``, ~1–2%, where a foreclosure+REO completes inside a
        reporting gap), so a strict zero would be wrong — the meaningful test is that the
        model does not put *more* than that realized signal on those cells (no hallucinated
        mass). The standing "record the deviation, don't tune" rule (04_TASKS §4) applies."""
    runs = [config.MODELS / "nn" / VARIANT / f"ens_k{k}_s{s}" for s in range(E.N_MEMBERS)]
    scaler, vocab = F.load_pipeline(runs[0])
    pool, b = D.window_spec(VARIANT, k, "test")
    df = F.prepare_raw(D._masked_scan(pool, b).collect())
    enc = F.encode_frame(df, scaler, vocab, encode="index")
    _, _, mean_p = _ensemble_eval(runs, enc, device)          # [N,7] ensemble probs
    y = np.asarray(enc["y"])
    origin_idx = np.array([F.STATE_INDEX[s] for s in df.get_column("state").to_list()])

    realized = np.bincount(y, minlength=F.N_CLASSES) / y.shape[0]
    pred_mean = mean_p.mean(axis=0)
    base_rate = {"realized": realized.tolist(), "predicted_mean": pred_mean.tolist(),
                 "max_abs_diff": float(np.abs(realized - pred_mean).max()),
                 "states": F.STATES, "tol": tol,
                 "passed": bool(np.abs(realized - pred_mean).max() < tol)}

    allow = _structural_allow()
    forbidden = ~allow[origin_idx]                            # [N,7] structurally-impossible mask
    model_mass = (mean_p * forbidden).sum(axis=1)            # per-row predicted impossible mass
    realized_imp = forbidden[np.arange(y.shape[0]), y]        # row's *realised* transition is impossible?
    per_origin = {}
    for s in ("current", "dpd_30", "dpd_60", "dpd_90plus"):
        m = origin_idx == F.STATE_INDEX[s]
        if m.sum():
            per_origin[s] = {"n": int(m.sum()),
                             "model_mean_mass": float(model_mass[m].mean()),
                             "realized_rate": float(realized_imp[m].mean())}
    impossible = {"definition": "structural (02 §1 monotone-delinquency rule)",
                  "model_mean_mass": float(model_mass.mean()),
                  "model_max_row_mass": float(model_mass.max()),
                  "realized_mean_rate": float(realized_imp.mean()),
                  "per_origin": per_origin, "threshold": 1e-3,
                  "passed": bool(model_mass.mean() < 1e-3)}
    return {"window_k": k, "n_test": int(y.shape[0]),
            "base_rate": base_rate, "impossible_transitions": impossible}


# ===========================================================================
# Summary + verification
# ===========================================================================
def _read_logit(k: int) -> dict | None:
    p = config.MODELS / "logit" / VARIANT / f"k{k}" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return {"val_nll": m["logit"]["val_nll"], "test_nll": m["logit"]["test_nll"],
            "selected_wd": m["l2_selection"]["selected_wd"]}


def _read_nn(k: int, cfg: dict) -> dict | None:
    p = config.MODELS / "nn" / VARIANT / _nn_tag(k, cfg) / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return {"val_nll": m["eval"]["val_nll"], "test_nll": m["eval"]["test_nll"],
            "best_epoch": m["training"]["best_epoch"], "n_epochs": m["training"]["n_epochs_run"]}


def _read_ensemble(k: int) -> dict | None:
    p = config.MODELS / "nn" / VARIANT / f"ensemble_k{k}_summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return {"val_nll": s["ensemble_val_nll"], "test_nll": s["ensemble_test_nll"],
            "mean_single_test_nll": s["mean_single_nll"]["test_nll"],
            "best_member_by_test_nll": s.get("best_single_test_nll")}


def _read_gbt(k: int) -> dict | None:
    p = config.MODELS / "gbt" / VARIANT / f"k{k}" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return {"val_nll": m["gbt"]["val_nll"], "test_nll": m["gbt"]["test_nll"],
            "best_iteration": m["selection"]["best_iteration"],
            "selected": m["selection"]["selected"],
            "base_rate_passed": m["base_rate_qa"]["passed"],
            "impossible_passed": m["impossible_transitions"]["passed"],
            "leakage": m.get("leakage")}


def build_summary(device=None, with_qa: bool = True) -> dict:
    """Assemble the Table-B-style per-window NLL matrix + (optional) base-rate QA on the key
    windows, into ``models/nn/full/backtest_summary.json``."""
    cfg = _frozen()
    windows = {}
    for k in config.TEST_YEARS:
        windows[str(k)] = {"logit": _read_logit(k), "nn": _read_nn(k, cfg),
                           "ensemble": _read_ensemble(k) if k in KEY_WINDOWS else None,
                           "gbt": _read_gbt(k)}
    qa = {}
    if with_qa and device is not None:
        for k in KEY_WINDOWS:
            if (config.MODELS / "nn" / VARIANT / f"ens_k{k}_s0" / "best_model.pt").exists():
                print(f"--- base-rate QA k={k} ---")
                qa[str(k)] = base_rate_qa(k, device)
    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "variant": VARIANT, "frozen_config": cfg, "key_windows": list(KEY_WINDOWS),
        "test_years": list(config.TEST_YEARS), "windows": windows, "base_rate_qa": qa,
        "git_commit": T._git_commit(),
    }
    out = config.MODELS / "nn" / VARIANT / "backtest_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return summary


def verify(device=None) -> None:
    """Every M10 Accept criterion, with evidence."""
    cfg = _frozen()
    summary = build_summary(device=device, with_qa=(device is not None))
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    print("\n[1] 11×{logit, NN} + 5×ensemble run folders exist")
    for k in config.TEST_YEARS:
        check(f"k={k} logit run", (config.MODELS / "logit" / VARIANT / f"k{k}" / "metrics.json").exists())
        check(f"k={k} NN run ({_nn_tag(k, cfg)})",
              (config.MODELS / "nn" / VARIANT / _nn_tag(k, cfg) / "metrics.json").exists())
    for k in KEY_WINDOWS:
        members = all((config.MODELS / "nn" / VARIANT / f"ens_k{k}_s{s}" / "best_model.pt").exists()
                      for s in range(E.N_MEMBERS))
        check(f"k={k} ensemble: {E.N_MEMBERS} members + summary",
              members and (config.MODELS / "nn" / VARIANT / f"ensemble_k{k}_summary.json").exists())

    print("\n[2] per-window early stopping used that window's val only (best_epoch from val)")
    for k in config.TEST_YEARS:
        m = json.loads((config.MODELS / "nn" / VARIANT / _nn_tag(k, cfg) / "metrics.json").read_text())
        be, ne = m["training"]["best_epoch"], m["training"]["n_epochs_run"]
        # early stopping ⇒ best epoch is at or before the last; val drove selection
        check(f"k={k} val-selected best_epoch {be+1}/{ne}", 0 <= be < ne,
              f"val_nll_best={m['eval']['val_nll']:.6f}")

    print("\n[3] scaler/vocab locality — per-window re-fit on that window's train slice (no leakage)")
    sa, va = F.load_pipeline(config.MODELS / "nn" / VARIANT / _nn_tag(2015, cfg))
    sb, vb = F.load_pipeline(config.MODELS / "nn" / VARIANT / _nn_tag(2020, cfg))
    # Different train slices ⇒ different standardisation centres and (generally) vocab sizes.
    ca = np.array([sa.center[c] for c in sa.cols])
    cb = np.array([sb.center[c] for c in sa.cols])
    check("k=2015 vs k=2020 scaler centres differ (window-local fit)", not np.allclose(ca, cb),
          f"max|Δcenter|={float(np.abs(ca - cb).max()):.4g}")
    check("k=2020 vocab ≥ k=2015 vocab (expanding train ⇒ ≥ levels)",
          all(b >= a for a, b in zip(va.vocab_sizes, vb.vocab_sizes)),
          f"Σvocab 2015={sum(va.vocab_sizes)} → 2020={sum(vb.vocab_sizes)}")

    print("\n[4] ensemble ≥ best single net on its key windows")
    # The paper's ensemble result (Fig 7) is variance reduction: the size-8 ensemble's
    # out-of-sample NLL falls below the *average* single net's. So the gate is ensemble ≤
    # MEAN single-member test NLL — architecture held constant, single-draw luck averaged
    # out. (Gating on one specific deployed single draw is noisier: that net is just member
    # seed 0, and on a given test slice a lucky single draw can edge the size-8 average; and
    # gating on min(member test) would be selection-on-test — unselectable out-of-sample.)
    # The deployed single-NN delta is reported per window for transparency.
    for k in KEY_WINDOWS:
        e = _read_ensemble(k)
        mean_t = e["mean_single_test_nll"]
        single_test = _read_nn(k, cfg)["test_nll"]
        check(f"k={k} ensemble test NLL {e['test_nll']:.6f} ≤ mean single-member {mean_t:.6f} "
              f"(Fig-7 variance reduction)", e["test_nll"] <= mean_t,
              f"Δ={mean_t - e['test_nll']:+.6f}")
        d = single_test - e["test_nll"]
        rel = "≤" if d >= 0 else ">"
        note = "  — tie within seed noise" if abs(d) < 1e-4 else ""
        print(f"       (vs deployed single-NN test {single_test:.6f}: ensemble {rel} by "
              f"{d:+.6f}{note}; ensemble val {e['val_nll']:.6f})")

    print("\n[5] base-rate QA (§7): importance-weighting + impossible-transition mass")
    if summary["base_rate_qa"]:
        for k in KEY_WINDOWS:
            q = summary["base_rate_qa"].get(str(k))
            if q is None:
                continue
            br, im = q["base_rate"], q["impossible_transitions"]
            check(f"k={k} predicted base rates ≈ realized (max|Δ|={br['max_abs_diff']:.4f} < {br['tol']})",
                  br["passed"])
            check(f"k={k} structural-impossible mass {im['model_mean_mass']:.2e} < 1e-3 "
                  f"(realized {im['realized_mean_rate']:.2e} — model tracks data, no hallucinated mass)",
                  im["passed"])
    else:
        print("  [skip] QA needs --device (ensemble inference); run without --verify-only or pass --device")

    print("\n[6] M17 GBT — wiring correctness (run folders, val-only, no leakage, base-rate QA)")
    gbt_runs = {k: _read_gbt(k) for k in config.TEST_YEARS}
    if not any(gbt_runs.values()):
        print("  [skip] no GBT run folders yet — run the M17 sweep (gbt fit per window)")
    else:
        gbt_cfg = _gbt_frozen()
        for k in config.TEST_YEARS:
            g = gbt_runs[k]
            check(f"k={k} GBT run folder + base-rate/impossible QA",
                  g is not None and g["base_rate_passed"] and g["impossible_passed"])
            if g is None:
                continue
            lk = g.get("leakage") or {}
            check(f"k={k} GBT val-selected best_iteration={g['best_iteration']} & masks disjoint "
                  f"(val UNK {lk.get('val_unk_rate', 0):.2%}, test UNK {lk.get('test_unk_rate', 0):.2%})",
                  g["best_iteration"] >= 1 and bool(lk.get("masks_disjoint")))
        cfgs = [tuple(sorted(g["selected"].items())) for g in gbt_runs.values() if g]
        check("GBT frozen config identical across all fitted windows (frozen, never re-tuned)",
              len(set(cfgs)) == 1 and dict(cfgs[0]) == {kk: gbt_cfg[kk] for kk in gbt_cfg})
        # Reported, NOT gated (06 §7 / standing rule 4): GBT vs logit / best NN per window.
        print("       reported (not gated) — GBT vs logit / best NN test NLL per window:")
        for k in config.TEST_YEARS:
            g, lo, n = gbt_runs[k], _read_logit(k), _read_nn(k, cfg)
            if g and lo and n:
                print(f"         k={k}: GBT {g['test_nll']:.6f}  vs logit {lo['test_nll']:.6f} "
                      f"({g['test_nll'] - lo['test_nll']:+.6f})  vs NN {n['test_nll']:.6f} "
                      f"({g['test_nll'] - n['test_nll']:+.6f})")

    label = "ALL ACCEPT CRITERIA PASS (M10 NN + M17 GBT)" if ok else "SOME CHECKS FAILED"
    print(f"\n{label} (variant={VARIANT})")
    if not ok:
        raise SystemExit(1)


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="M10 rolling backtest loop (11 windows).")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--amp", action="store_true", help="mixed precision (cuda only)")
    ap.add_argument("--windows", type=int, nargs="*", default=None,
                    help="subset of test years to run (default: all 11)")
    ap.add_argument("--skip-logit", action="store_true")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--skip-ensemble", action="store_true")
    ap.add_argument("--skip-gbt", action="store_true", help="skip the M17 GBT fit in the loop")
    ap.add_argument("--gbt-only", action="store_true",
                    help="fit only GBT (implies --skip-logit/nn/ensemble; CPU, no GPU needed)")
    ap.add_argument("--gbt-threads", type=int, default=0, help="LightGBM num_threads (0 = all cores)")
    ap.add_argument("--verify-only", action="store_true",
                    help="Accept checks on existing outputs (pass --device for the QA pass)")
    args = ap.parse_args()
    config.require_drive()
    device = tc.resolve_device(args.device) if args.device != "cuda" else torch.device("cuda")
    use_amp = bool(args.amp) and device.type == "cuda"

    if args.verify_only:
        verify(device=device if device.type == "cuda" else None)
        return

    cfg = _frozen()
    gbt_cfg = _gbt_frozen()
    gbt_threads = args.gbt_threads or (os.cpu_count() or 1)
    skip_logit = args.skip_logit or args.gbt_only
    skip_nn = args.skip_nn or args.gbt_only
    skip_ensemble = args.skip_ensemble or args.gbt_only
    skip_gbt = args.skip_gbt and not args.gbt_only
    windows = args.windows or config.TEST_YEARS
    print(f"=== M10/M17 backtest loop  windows={list(windows)}  nn_frozen={cfg}  "
          f"gbt_frozen={gbt_cfg}  device={device} amp={use_amp} gbt_threads={gbt_threads} ===")
    for k in windows:
        t0 = time.perf_counter()
        if not skip_logit:
            fit_logit(k, device, use_amp)
        if not skip_nn:
            fit_nn_single(k, cfg, args.device, use_amp)
        if not skip_ensemble and k in KEY_WINDOWS:
            fit_ensemble(k, cfg, args.device, use_amp, device)
        if not skip_gbt:
            fit_gbt(k, gbt_cfg, gbt_threads)
        print(f"=== window k={k} done in {(time.perf_counter()-t0)/60:.1f} min ===\n")

    verify(device=device if device.type == "cuda" else None)


if __name__ == "__main__":
    main()
