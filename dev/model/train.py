"""M8 — Single-window NN training loop (``02_LOAN_LEVEL §6``).

Trains one :class:`net.MortgageMLP` on one rolling window (default the tuning window
k=2015, dev export) through the shared importance-weighted loss + NLL evaluator
(``torch_common.py``): Adam, decay-LR-on-plateau, **early stopping on val NLL**, and a
self-describing run folder under ``models/nn/<variant>/<run_id>/`` carrying the scaler,
vocab, a resumable checkpoint, the best weights, and ``metrics.json`` (architecture +
hyperparameters + per-epoch history + the train/eval manifest hashes + window id, per
``§6``'s "log every run" rule).

Resumability (``Accept``): every epoch writes ``checkpoint.pt`` **atomically** (tmp +
rename) with model/optimizer/scheduler/AMP state, the best-so-far weights, the patience
counter, and the RNG states. A relaunch with the same args reloads it and continues from
the next epoch. Because the data order is a pure function of ``(seed, epoch)`` and the
torch RNG is restored, a killed-and-resumed run reproduces an uninterrupted run's
trajectory bit-for-bit (the M8a CPU proof) — so a crash on the rented GPU box never loses
more than the in-progress epoch.

Data: dev scale fits in RAM, so each split is preloaded once (index-encoded) and reshuffled
in-memory per epoch — a **full** shuffle, the strongest form of ``§3.4``'s "never minibatch
from a single vintage". The full-scale loop (M10) swaps this preload for the streaming
``data.WindowLoader`` (same batch-dict contract, same training loop).

Run:
    .venv/bin/python dev/model/train.py --smoke                 # CPU pipeline check (≤100k rows)
    .venv/bin/python dev/model/train.py --smoke --verify-only   # Accept checks on the smoke run
    .venv/bin/python dev/model/train.py --device cuda --amp     # GPU dev-export fit (M8 GPU session)
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import torch

import config           # dev/model/config.py — also puts dev/pipeline on sys.path
import data as D        # M6 loader: window_spec / _masked_scan / fit_window
import features as F    # M6 feature pipeline: Scaler / Vocab / encode_frame / save_pipeline
import net as N         # M8 MLP + predict callback
import torch_common as tc

_REPO = Path(__file__).resolve().parents[2]

# --- Hyperparameters (tuning window only; the M9 grid sweeps depth/dropout/L2) ----------
DEPTH = 5                # paper optimum: 5 hidden layers (200 then 140×4) — net.depth_to_hidden
DROPOUT = 0.2            # §6 dropout on each hidden layer
WEIGHT_DECAY = 0.0       # L2; M8 single net uses 0, M9 grids {0, 1e-5, 1e-4}
LR = 1e-3               # Adam lr (~1e-3, §6), decayed on val-NLL plateau
BATCH_SIZE = 8192        # §6: tabular MLPs like big batches (4–16k)
MAX_EPOCHS = 40
PATIENCE = 5             # early-stopping patience on val NLL (§6: 3–5 evals)
EVAL_BATCH = 16384
SEED = 0
# Decay-on-plateau (§6): halve the LR after 2 non-improving val evals.
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 2

SMOKE_CAP = 100_000      # standing rule 3: the CPU smoke path touches ≤100k rows
SMOKE_EPOCHS = 8         # enough to show the val NLL fall and to kill+resume mid-run


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def run_id(args) -> str:
    """Run-folder name: the explicit ``--run-name`` override, else a config-encoding tag
    (k / depth / dropout / weight-decay) so the M9 grid's runs never collide."""
    if args.run_name:
        return args.run_name
    tag = f"k{args.k}_d{args.depth}_do{args.dropout:g}_wd{args.weight_decay:g}"
    return tag + ("_smoke" if args.smoke else "")


# ---------------------------------------------------------------------------
# Split preloading — index-encode each split once into RAM (dev scale, §3 note)
# ---------------------------------------------------------------------------
def preload_split(variant: str, k: int, split: str, scaler: F.Scaler, vocab: F.Vocab,
                  cap: int | None) -> dict:
    """Window ``k``'s ``split`` slice (M6 masking) → index-encoded NumPy arrays. ``cap``
    bounds the rows for the smoke path; Polars pushes the ``limit`` into the scan so only
    ``cap`` rows are read."""
    pool_dir, bounds = D.window_spec(variant, k, split)
    lf = D._masked_scan(pool_dir, bounds)
    if cap is not None:
        lf = lf.limit(cap)
    df = F.prepare_raw(lf.collect())
    return F.encode_frame(df, scaler, vocab, encode="index")


def epoch_batches(enc: dict, batch_size: int, epoch: int, seed: int):
    """Shuffled minibatches for a training epoch. The permutation is a pure function of
    ``(seed, epoch)`` (``np.random.default_rng([seed, epoch])`` — the data.py convention),
    so epoch ``e`` yields the identical sequence on a fresh run and on a resumed one."""
    n = enc["y"].shape[0]
    order = np.random.default_rng([seed, epoch]).permutation(n)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        yield {kk: enc[kk][idx] for kk in ("cont", "cat", "bin", "y", "w")}


def eval_batches(enc: dict, batch_size: int):
    """Sequential (deterministic) minibatches for evaluation."""
    n = enc["y"].shape[0]
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        yield {kk: enc[kk][sl] for kk in ("cont", "cat", "bin", "y", "w")}


# ---------------------------------------------------------------------------
# Checkpointing — atomic write of the full resumable state, every epoch
# ---------------------------------------------------------------------------
def _rng_state(device: torch.device) -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if device.type == "cuda":
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _restore_rng(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, *, epoch: int, model, opt, sched, amp_scaler,
                    best: dict, bad: int, history: list, arch: dict, cfg: dict,
                    device: torch.device) -> None:
    """Write the full resumable state atomically (tmp + ``os.replace``), so a hard kill
    mid-write never corrupts the prior epoch's good checkpoint."""
    ckpt = {
        "epoch": epoch, "arch": arch, "config": cfg,
        "model_state": model.state_dict(), "optim_state": opt.state_dict(),
        "sched_state": sched.state_dict(), "amp_state": amp_scaler.state_dict(),
        "best": best, "bad": bad, "history": history,
        "rng": _rng_state(device),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Train one epoch — importance-weighted CE, AMP-aware (no-op off cuda)
# ---------------------------------------------------------------------------
def train_one_epoch(model, enc_train, predict, opt, amp_scaler, *, device, epoch,
                    seed, batch_size, use_amp) -> float:
    """One pass over the (reshuffled) train slice. Returns the running weight-averaged
    train NLL (Σ w·CE / Σ w over the epoch's batches) — a cheap loss-decrease signal."""
    model.train()
    swce = sw = 0.0
    for b in epoch_batches(enc_train, batch_size, epoch, seed):
        y = tc._as_long(b["y"], device)
        w = tc._as_f32(b["w"], device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = tc.weighted_ce(predict(model, b, device), y, w)
        amp_scaler.scale(loss).backward()
        amp_scaler.step(opt)
        amp_scaler.update()
        swce += float(loss.detach().double()) * float(w.double().sum())
        sw += float(w.double().sum())
    return swce / sw if sw else float("nan")


# ---------------------------------------------------------------------------
# Build / resume the training run
# ---------------------------------------------------------------------------
def train(args) -> Path:
    device = tc.resolve_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    cap = SMOKE_CAP if args.smoke else None
    max_epochs = args.max_epochs if args.max_epochs is not None else (
        SMOKE_EPOCHS if args.smoke else MAX_EPOCHS)
    run = config.MODELS / "nn" / args.variant / run_id(args)
    run.mkdir(parents=True, exist_ok=True)
    ckpt_path = run / "checkpoint.pt"

    print(f"=== M8 NN training  variant={args.variant}  window k={args.k}  "
          f"depth={args.depth} dropout={args.dropout} wd={args.weight_decay}  "
          f"device={device}  amp={use_amp}  smoke={args.smoke} ===")

    # Idempotency: a finished run (metrics.json present) is left alone unless --fresh.
    if (run / "metrics.json").exists() and not args.fresh:
        print(f"run already complete at {run} — re-verifying (use --fresh to retrain)")
        verify(args)
        return run

    cfg = {"variant": args.variant, "k": args.k, "depth": args.depth,
           "dropout": args.dropout, "weight_decay": args.weight_decay, "lr": args.lr,
           "batch_size": args.batch_size, "max_epochs": max_epochs,
           "patience": args.patience, "seed": args.seed, "smoke": args.smoke,
           "use_amp": use_amp, "cap": cap}

    resuming = ckpt_path.exists() and not args.fresh
    if resuming:
        # Resume: scaler/vocab from the run folder (identical featurization), model rebuilt
        # from the checkpoint's arch, all optimizer/scheduler/RNG state restored.
        scaler, vocab = F.load_pipeline(run)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = N.from_arch(ckpt["arch"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        opt.load_state_dict(ckpt["optim_state"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE)
        sched.load_state_dict(ckpt["sched_state"])
        amp_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        amp_scaler.load_state_dict(ckpt["amp_state"])
        best, bad, history = ckpt["best"], ckpt["bad"], ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        arch = ckpt["arch"]
        _restore_rng(ckpt["rng"], device)   # RNG restored LAST, after the init above
        print(f"resuming from checkpoint: last completed epoch {ckpt['epoch']}, "
              f"best val_nll={best['val_nll']:.6f}@{best['epoch']} → start epoch {start_epoch}")
    else:
        # Fresh: fit scaler/vocab on the train slice only, build the net, seed everything.
        tc.set_seed(args.seed)
        scaler, vocab = D.fit_window(args.variant, args.k)
        F.save_pipeline(run, scaler, vocab)
        model = N.build(scaler, vocab, depth=args.depth, dropout=args.dropout).to(device)
        arch = model.arch()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE)
        amp_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        best = {"val_nll": float("inf"), "epoch": -1, "state": None}
        bad, history, start_epoch = 0, [], 0
        print(f"net: {arch['hidden_dims']} hidden, {model.n_params():,} params, "
              f"in_dim={model.in_dim} "
              f"({arch['n_continuous']} cont + {sum(arch['emb_dims'])} emb + "
              f"{arch['n_binary']} bin)")

    predict = N.make_nn_predict()
    enc_train = preload_split(args.variant, args.k, "train", scaler, vocab, cap)
    enc_val = preload_split(args.variant, args.k, "val", scaler, vocab, cap)
    print(f"rows: train={enc_train['y'].shape[0]:,} (Σw={enc_train['w'].sum():,.0f})  "
          f"val={enc_val['y'].shape[0]:,}")

    # --- training loop: early stopping on val NLL, atomic checkpoint each epoch ----------
    for epoch in range(start_epoch, max_epochs):
        train_nll = train_one_epoch(model, enc_train, predict, opt, amp_scaler,
                                    device=device, epoch=epoch, seed=args.seed,
                                    batch_size=args.batch_size, use_amp=use_amp)
        val = tc.evaluate_nll(model, eval_batches(enc_val, EVAL_BATCH), predict, device)
        sched.step(val["weighted_nll"])
        lr_now = opt.param_groups[0]["lr"]
        improved = val["weighted_nll"] < best["val_nll"] - 1e-7
        if improved:
            best = {"val_nll": val["weighted_nll"], "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
        history.append({"epoch": epoch, "train_nll": train_nll,
                        "val_nll": val["weighted_nll"], "lr": lr_now, "bad": bad})
        save_checkpoint(ckpt_path, epoch=epoch, model=model, opt=opt, sched=sched,
                        amp_scaler=amp_scaler, best=best, bad=bad, history=history,
                        arch=arch, cfg=cfg, device=device)
        print(f"  epoch {epoch:>2}: train_nll={train_nll:.6f}  val_nll={val['weighted_nll']:.6f}"
              f"  lr={lr_now:.2e}  (best@{best['epoch']}, bad={bad})"
              f"{'  *' if improved else ''}")

        if bad >= args.patience:
            print(f"  early stopping (no val improvement for {args.patience} evals)")
            break
        # Debug hook: clean-exit after a given epoch to simulate a mid-run interruption
        # (leaves a checkpoint, no metrics.json — so a relaunch resumes). Used by the M8a
        # resume proof; a real kill -9 mid-epoch exercises the same path.
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            print(f"  --stop-after-epoch {args.stop_after_epoch}: simulating interruption "
                  f"(checkpoint at epoch {epoch}; relaunch to resume)")
            return run

    # --- finalize: best model, frozen-test NLL, metrics.json ----------------------------
    model.load_state_dict(best["state"])
    enc_test = preload_split(args.variant, args.k, "test", scaler, vocab, cap)
    test = tc.evaluate_nll(model, eval_batches(enc_test, EVAL_BATCH), predict, device)
    val_best = tc.evaluate_nll(model, eval_batches(enc_val, EVAL_BATCH), predict, device)

    torch.save(best["state"], run / "best_model.pt")
    manifest = json.loads((config.TRAINING_DIR / args.variant / "manifest.json").read_text())
    logit_cmp = _logit_comparison(args.variant, args.k, val_best["weighted_nll"],
                                  test["unweighted_nll"])
    val0 = history[0]["val_nll"] if history else float("nan")
    metrics = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "nn_mlp", "variant": args.variant, "window_k": args.k,
        "tuning_window": (args.k == config.TUNING_YEAR), "smoke": args.smoke,
        "device": str(device), "use_amp": use_amp, "seed": args.seed,
        "git_commit": _git_commit(),
        "source": {
            "train_pool_content_hash": manifest["train_pool"]["content_hash"],
            "eval_pool_content_hash": manifest["eval_pool"]["content_hash"],
            "panel_glob": config.PANEL_GLOB,
        },
        "architecture": {**arch, "n_params": model.n_params(),
                         "vocab_cols": vocab.cols},
        "hyperparams": {"depth": args.depth, "dropout": args.dropout,
                        "weight_decay": args.weight_decay, "lr": args.lr,
                        "batch_size": args.batch_size, "max_epochs": max_epochs,
                        "patience": args.patience, "sched_factor": SCHED_FACTOR,
                        "sched_patience": SCHED_PATIENCE},
        "rows": {"train": int(enc_train["y"].shape[0]),
                 "train_sum_w": float(enc_train["w"].sum()),
                 "val": int(enc_val["y"].shape[0]), "test": int(enc_test["y"].shape[0])},
        "training": {"n_epochs_run": len(history), "best_epoch": best["epoch"],
                     "val_nll_first": val0, "val_nll_best": best["val_nll"],
                     "loss_decreased": bool(best["val_nll"] < val0),
                     "history": history},
        "eval": {"val_nll": val_best["weighted_nll"],
                 "test_nll": test["unweighted_nll"],
                 "test_nll_weighted": test["weighted_nll"],
                 "test_n_rows": test["n_rows"], "test_sum_w": test["sum_w"]},
        "logit_comparison": logit_cmp,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nbest epoch {best['epoch']}  val_nll={best['val_nll']:.6f}  "
          f"test_nll={test['unweighted_nll']:.6f}")
    if logit_cmp is not None:
        print(f"logit (M7) val_nll={logit_cmp['logit_val_nll']:.6f} "
              f"test_nll={logit_cmp['logit_test_nll']:.6f}  → "
              f"NN test margin {logit_cmp['test_margin']:+.6f} "
              f"({'beats' if logit_cmp['beats_logit_test'] else 'does not beat'}"
              f"{' — informational on smoke' if args.smoke else ''})")
    print(f"wrote {run / 'metrics.json'}")
    return run


def _logit_comparison(variant: str, k: int, nn_val_nll: float,
                      nn_test_nll: float) -> dict | None:
    """Read the M7 logit run's val/test NLL for the same window and report the NN margin.
    The Accept criterion 'val NLL improves on logit' is judged on the **GPU dev fit** (this
    is informational on the capped smoke run, where neither model is converged)."""
    p = config.MODELS / "logit" / variant / f"k{k}" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    lg = m["logit"]
    return {"logit_val_nll": lg["val_nll"], "logit_test_nll": lg["test_nll"],
            "nn_val_nll": nn_val_nll, "nn_test_nll": nn_test_nll,
            "val_margin": lg["val_nll"] - nn_val_nll,
            "test_margin": lg["test_nll"] - nn_test_nll,
            "beats_logit_val": nn_val_nll < lg["val_nll"],
            "beats_logit_test": nn_test_nll < lg["test_nll"]}


# ---------------------------------------------------------------------------
# Verify — the M8a CPU-checkable Accept criteria, with evidence
# ---------------------------------------------------------------------------
def verify(args) -> None:
    run = config.MODELS / "nn" / args.variant / run_id(args)
    metrics = json.loads((run / "metrics.json").read_text())
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    print(f"\n[1] run folder has checkpoint + metrics + scaler + vocab + best weights")
    for fname in ("checkpoint.pt", "metrics.json", "scaler.json", "vocab.json",
                  "best_model.pt"):
        check(f"{fname} present", (run / fname).exists())

    print(f"\n[2] provenance: manifest hash + window id recorded")
    src = metrics["source"]
    check("train-pool manifest content_hash recorded",
          bool(src["train_pool_content_hash"]),
          src["train_pool_content_hash"][:12] + "…")
    check("eval-pool manifest content_hash recorded", bool(src["eval_pool_content_hash"]))
    check(f"window id == k ({args.k})", metrics["window_k"] == args.k)
    check("git commit recorded", bool(metrics["git_commit"]))

    print(f"\n[3] training ran end-to-end and the loss decreased")
    tr = metrics["training"]
    check(f"≥2 epochs run ({tr['n_epochs_run']})", tr["n_epochs_run"] >= 2)
    check(f"best val NLL {tr['val_nll_best']:.6f} < first-epoch val NLL "
          f"{tr['val_nll_first']:.6f}", tr["loss_decreased"],
          f"Δ={tr['val_nll_first'] - tr['val_nll_best']:+.6f}")

    print(f"\n[4] checkpoint is a complete resumable state")
    ckpt = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    for key in ("epoch", "model_state", "optim_state", "sched_state", "best", "bad",
                "history", "rng", "arch"):
        check(f"checkpoint['{key}'] present", key in ckpt)
    check("checkpoint RNG state captured (torch + numpy)",
          "torch" in ckpt.get("rng", {}) and "numpy" in ckpt.get("rng", {}))

    print(f"\n[note] 'val NLL improves on logit' + 'no OOM at scale' are the GPU dev-fit "
          f"Accept criteria — deferred (smoke is capped/unconverged).")
    print(f"\n{'ALL CPU-CHECKABLE ACCEPT CRITERIA PASS' if ok else 'SOME CHECKS FAILED'} "
          f"(variant={args.variant}, run={run_id(args)})")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train/verify a single-window NN (M8).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override (default 40, or 8 in --smoke)")
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (active only on cuda; no-op on cpu/mps)")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU pipeline check on ≤100k rows (standing rule 3)")
    ap.add_argument("--run-name", default=None,
                    help="override the run-folder name (default: k/depth/dropout/wd tag)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and retrain from scratch")
    ap.add_argument("--stop-after-epoch", type=int, default=None,
                    help="debug: clean-exit after this epoch to simulate an interruption")
    ap.add_argument("--verify-only", action="store_true",
                    help="run the Accept checks against an existing run folder")
    args = ap.parse_args()
    config.require_drive()
    if args.verify_only:
        verify(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
