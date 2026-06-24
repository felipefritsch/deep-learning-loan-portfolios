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
    .venv/bin/python -m floan.model.train --smoke                 # CPU pipeline check (≤100k rows)
    .venv/bin/python -m floan.model.train --smoke --verify-only   # Accept checks on the smoke run
    .venv/bin/python -m floan.model.train --device cuda --amp     # GPU dev-export fit (M8 GPU session)
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import data as D  # M6 loader: window_spec / _masked_scan / fit_window
from floan.model import features as F  # M6 feature pipeline: Scaler / Vocab / encode_frame / save_pipeline
from floan.model import net as N  # M8 MLP + predict callback
from floan.model import torch_common as tc

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


def _environment(device: torch.device) -> dict:
    """Hardware/software provenance for the run record (the box is a rented GPU: torch
    comes from the template image via a --system-site-packages venv, so the exact build is
    not pinned in this repo and must be captured per-run). Records the torch/CUDA/cuDNN
    build and, on cuda, the GPU model + compute capability."""
    env = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        env["gpu_name"] = torch.cuda.get_device_name(idx)
        env["gpu_capability"] = list(torch.cuda.get_device_capability(idx))
        env["gpu_count"] = torch.cuda.device_count()
    return env


def run_id(args) -> str:
    """Run-folder name: the explicit ``--run-name`` override, else a config-encoding tag
    (k / depth / dropout / weight-decay) so the M9 grid's runs never collide."""
    if args.run_name:
        return args.run_name
    tag = f"k{args.k}_d{args.depth}_do{args.dropout:g}_wd{args.weight_decay:g}"
    return tag + ("_aug" if getattr(args, "augment", False) else "") + ("_smoke" if args.smoke else "")


# ---------------------------------------------------------------------------
# Split preloading — index-encode each split once into RAM (dev scale, §3 note)
# ---------------------------------------------------------------------------
def preload_split(variant: str, k: int, split: str, scaler: F.Scaler, vocab: F.Vocab,
                  cap: int | None, augment: bool = False) -> dict:
    """Window ``k``'s ``split`` slice (M6 masking) → index-encoded NumPy arrays. ``cap``
    bounds the rows for the smoke path; Polars pushes the ``limit`` into the scan so only
    ``cap`` rows are read. ``augment`` (M26a): left-join the per-(loan, period_ym) history
    summaries and encode the extended binary block — the only difference between the two
    arms is the feature set."""
    pool_dir, bounds = D.window_spec(variant, k, split)
    lf = D._masked_scan(pool_dir, bounds)
    if cap is not None:
        lf = lf.limit(cap)
    df = F.prepare_raw(lf.collect())
    if augment:
        from floan.model import history as H
        df = H.join_history(df, variant, k)
        return F.encode_frame(df, scaler, vocab, encode="index",
                              binary_cols=F.BINARY + H.HIST_BINARY)
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
# GPU-resident fast path (M10 full scale) — keep the whole split on the GPU
# ---------------------------------------------------------------------------
# The dev-scale numpy path above re-indexes NumPy arrays and copies host→device every
# minibatch; on the full export that pins the GPU at ~10% util (CPU/transfer-bound, the
# M10b throughput probe: ~0.35 M rows/s on the A4500). The whole index-encoded split fits
# in 20 GB (k=2015 ≈ 8 GB, the k=2025 ≈74 M-row slice ≈14 GB with int32 ``cat``), so for
# ``--gpu-resident`` we upload it once and index ON the GPU. The minibatch **order** is the
# identical ``np.random.default_rng([seed, epoch]).permutation`` the numpy path uses (moved
# to a device LongTensor), so a resumed run still reproduces an uninterrupted one. Loss/eval
# are the same weight-averaged CE / float64-accumulated NLL as ``torch_common`` — only the
# tensors' residence changes, so NN-vs-logit differences stay attributable to architecture.
def to_resident(enc: dict, device: torch.device) -> dict:
    """Upload an index-encoded split (``preload_split`` output) to persistent GPU tensors.
    ``cat`` is held as int32 (halves its footprint; cast to long per-batch for the embedding
    lookup, which nn.Embedding requires)."""
    return {
        "cont": torch.as_tensor(np.asarray(enc["cont"]), dtype=torch.float32, device=device),
        "cat": torch.as_tensor(np.asarray(enc["cat"]), dtype=torch.int32, device=device),
        "bin": torch.as_tensor(np.asarray(enc["bin"]), dtype=torch.float32, device=device),
        "y": torch.as_tensor(np.asarray(enc["y"]), dtype=torch.long, device=device),
        "w": torch.as_tensor(np.asarray(enc["w"]), dtype=torch.float32, device=device),
    }


def _resident_logits(model, T: dict, idx) -> torch.Tensor:
    """Forward ``model`` on rows ``idx`` of a resident split (``idx`` a device LongTensor or
    a slice). ``cat`` int32 → long only for the gathered minibatch."""
    if isinstance(idx, slice):
        cont, cat, binb = T["cont"][idx], T["cat"][idx], T["bin"][idx]
    else:
        cont = T["cont"].index_select(0, idx)
        cat = T["cat"].index_select(0, idx)
        binb = T["bin"].index_select(0, idx)
    return model(cont, cat.long(), binb)


def train_one_epoch_resident(model, T: dict, opt, amp_scaler, *, device, epoch, seed,
                             batch_size, use_amp) -> float:
    """One reshuffled pass over a resident train split. Same ``(seed, epoch)`` permutation
    and weight-averaged-CE loss as :func:`train_one_epoch`; accumulators stay on-GPU and sync
    once at the end (no per-batch host round-trip)."""
    model.train()
    n = T["y"].shape[0]
    order = torch.as_tensor(np.random.default_rng([seed, epoch]).permutation(n), device=device)
    swce = torch.zeros((), dtype=torch.float64, device=device)
    sw = torch.zeros((), dtype=torch.float64, device=device)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        y = T["y"].index_select(0, idx)
        w = T["w"].index_select(0, idx)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = tc.weighted_ce(_resident_logits(model, T, idx), y, w)
        amp_scaler.scale(loss).backward()
        amp_scaler.step(opt)
        amp_scaler.update()
        wsum = w.double().sum()
        swce += loss.detach().double() * wsum
        sw += wsum
    return float((swce / sw).item()) if float(sw) else float("nan")


@torch.no_grad()
def evaluate_nll_resident(model, T: dict, batch_size: int, device) -> dict:
    """Streaming NLL over a resident split — the :func:`torch_common.evaluate_nll` contract
    (weighted + unweighted + n_rows + sum_w), float64 accumulation, GPU-side."""
    model.eval()
    n = int(T["y"].shape[0])
    swce = torch.zeros((), dtype=torch.float64, device=device)
    sw = torch.zeros((), dtype=torch.float64, device=device)
    sce = torch.zeros((), dtype=torch.float64, device=device)
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        y, w = T["y"][sl], T["w"][sl]
        ce = torch.nn.functional.cross_entropy(
            _resident_logits(model, T, sl), y, reduction="none").double()
        swce += (w.double() * ce).sum()
        sw += w.double().sum()
        sce += ce.sum()
    return {"weighted_nll": float((swce / sw).item()),
            "unweighted_nll": float((sce / n).item()),
            "n_rows": n, "sum_w": float(sw.item())}


# ---------------------------------------------------------------------------
# Checkpointing — atomic write of the full resumable state, every epoch
# ---------------------------------------------------------------------------
def _rng_state(device: torch.device) -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if device.type == "cuda":
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _restore_rng(state: dict, device: torch.device) -> None:
    # The checkpoint is loaded with map_location=device (so model/optimizer tensors land on
    # the GPU), which also moves these CPU RNG-state tensors onto cuda. ``set_rng_state`` and
    # ``set_rng_state_all`` require CPU ByteTensors, so coerce back before restoring — without
    # this the resume path raises "RNG state must be a torch.ByteTensor" on cuda (it is a
    # no-op on cpu, where map_location never moved them: the M8a CPU proof).
    def _cpu_byte(t):
        return t.cpu().to(torch.uint8) if torch.is_tensor(t) else t
    torch.set_rng_state(_cpu_byte(state["torch"]))
    np.random.set_state(state["numpy"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([_cpu_byte(t) for t in state["cuda"]])


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

    augment = bool(getattr(args, "augment", False))
    cfg = {"variant": args.variant, "k": args.k, "depth": args.depth,
           "dropout": args.dropout, "weight_decay": args.weight_decay, "lr": args.lr,
           "batch_size": args.batch_size, "max_epochs": max_epochs,
           "patience": args.patience, "seed": args.seed, "smoke": args.smoke,
           "use_amp": use_amp, "cap": cap,
           "width_mult": getattr(args, "width_mult", 1.0), "augment": augment}

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
        # M26a augment: fit over the extended blocks (history joined) and size the binary
        # block accordingly; everything else (loop, loss, eval) is identical across arms.
        tc.set_seed(args.seed)
        if augment:
            from floan.model import history as H
            scaler, vocab = H.fit_window_aug(args.variant, args.k)
            n_binary = len(F.BINARY) + len(H.HIST_BINARY)
        else:
            scaler, vocab = D.fit_window(args.variant, args.k)
            n_binary = None
        F.save_pipeline(run, scaler, vocab)
        model = N.build(scaler, vocab, depth=args.depth, dropout=args.dropout,
                        width_mult=getattr(args, "width_mult", 1.0),
                        n_binary=n_binary).to(device)
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
    resident = bool(getattr(args, "gpu_resident", False)) and device.type == "cuda"
    enc_train = preload_split(args.variant, args.k, "train", scaler, vocab, cap, augment)
    enc_val = preload_split(args.variant, args.k, "val", scaler, vocab, cap, augment)
    print(f"rows: train={enc_train['y'].shape[0]:,} (Σw={enc_train['w'].sum():,.0f})  "
          f"val={enc_val['y'].shape[0]:,}")
    n_train = int(enc_train["y"].shape[0])
    if resident:
        # Upload both splits to the GPU once; drop the host arrays (keep only the keys/shape
        # the finalize + metrics need) so RAM isn't doubled on the big windows.
        T_train, T_val = to_resident(enc_train, device), to_resident(enc_val, device)
        train_sum_w = float(enc_train["w"].sum())
        n_val = int(enc_val["y"].shape[0])
        del enc_train, enc_val
        torch.cuda.synchronize()
        print(f"gpu-resident: train+val uploaded, "
              f"{torch.cuda.memory_allocated(device) / 1e9:.1f} GB allocated")

    # --- training loop: early stopping on val NLL, atomic checkpoint each epoch ----------
    for epoch in range(start_epoch, max_epochs):
        # Per-epoch wall-clock (M8b throughput, sizes the M10 full export). train_sec times
        # the GPU pass over the train slice (cuda-synced so async kernels are accounted);
        # epoch_sec adds the val eval + atomic checkpoint write.
        t0 = time.perf_counter()
        if resident:
            train_nll = train_one_epoch_resident(model, T_train, opt, amp_scaler,
                                                 device=device, epoch=epoch, seed=args.seed,
                                                 batch_size=args.batch_size, use_amp=use_amp)
        else:
            train_nll = train_one_epoch(model, enc_train, predict, opt, amp_scaler,
                                        device=device, epoch=epoch, seed=args.seed,
                                        batch_size=args.batch_size, use_amp=use_amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - t0
        val = (evaluate_nll_resident(model, T_val, EVAL_BATCH, device) if resident
               else tc.evaluate_nll(model, eval_batches(enc_val, EVAL_BATCH), predict, device))
        sched.step(val["weighted_nll"])
        lr_now = opt.param_groups[0]["lr"]
        improved = val["weighted_nll"] < best["val_nll"] - 1e-7
        if improved:
            best = {"val_nll": val["weighted_nll"], "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
        epoch_sec = time.perf_counter() - t0
        rows_per_sec = n_train / train_sec if train_sec > 0 else float("nan")
        history.append({"epoch": epoch, "train_nll": train_nll,
                        "val_nll": val["weighted_nll"], "lr": lr_now, "bad": bad,
                        "train_sec": train_sec, "epoch_sec": epoch_sec,
                        "rows_per_sec": rows_per_sec})
        save_checkpoint(ckpt_path, epoch=epoch, model=model, opt=opt, sched=sched,
                        amp_scaler=amp_scaler, best=best, bad=bad, history=history,
                        arch=arch, cfg=cfg, device=device)
        print(f"  epoch {epoch:>2}: train_nll={train_nll:.6f}  val_nll={val['weighted_nll']:.6f}"
              f"  lr={lr_now:.2e}  (best@{best['epoch']}, bad={bad})"
              f"  [{train_sec:.1f}s train, {epoch_sec:.1f}s epoch, {rows_per_sec:,.0f} rows/s]"
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
    enc_test = preload_split(args.variant, args.k, "test", scaler, vocab, cap, augment)
    if resident:
        # Free the resident train slice before uploading test (keeps the big windows under
        # 20 GB); val is re-scored from T_val, test from a freshly uploaded resident slice.
        del T_train
        torch.cuda.empty_cache()
        T_test = to_resident(enc_test, device)
        n_test = int(enc_test["y"].shape[0])
        del enc_test
        test = evaluate_nll_resident(model, T_test, EVAL_BATCH, device)
        val_best = evaluate_nll_resident(model, T_val, EVAL_BATCH, device)
    else:
        n_train = int(enc_train["y"].shape[0])
        train_sum_w = float(enc_train["w"].sum())
        n_val, n_test = int(enc_val["y"].shape[0]), int(enc_test["y"].shape[0])
        test = tc.evaluate_nll(model, eval_batches(enc_test, EVAL_BATCH), predict, device)
        val_best = tc.evaluate_nll(model, eval_batches(enc_val, EVAL_BATCH), predict, device)

    torch.save(best["state"], run / "best_model.pt")
    manifest = json.loads((config.TRAINING_DIR / args.variant / "manifest.json").read_text())
    logit_cmp = _logit_comparison(args.variant, args.k, val_best["weighted_nll"],
                                  test["unweighted_nll"])
    val0 = history[0]["val_nll"] if history else float("nan")
    throughput = _throughput_summary(history, n_train=n_train)
    metrics = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "nn_mlp", "variant": args.variant, "window_k": args.k,
        "tuning_window": (args.k == config.TUNING_YEAR), "smoke": args.smoke,
        "augment": augment,
        "device": str(device), "use_amp": use_amp, "seed": args.seed,
        "git_commit": _git_commit(),
        "environment": _environment(device),
        "source": {
            "train_pool_content_hash": manifest["train_pool"]["content_hash"],
            "eval_pool_content_hash": manifest["eval_pool"]["content_hash"],
            "panel_glob": config.PANEL_GLOB,
        },
        "architecture": {**arch, "n_params": model.n_params(),
                         "vocab_cols": vocab.cols},
        "hyperparams": {"depth": args.depth, "dropout": args.dropout,
                        "weight_decay": args.weight_decay, "lr": args.lr,
                        "width_mult": getattr(args, "width_mult", 1.0),
                        "batch_size": args.batch_size, "max_epochs": max_epochs,
                        "patience": args.patience, "sched_factor": SCHED_FACTOR,
                        "sched_patience": SCHED_PATIENCE},
        "rows": {"train": n_train, "train_sum_w": train_sum_w,
                 "val": n_val, "test": n_test},
        "training": {"n_epochs_run": len(history), "best_epoch": best["epoch"],
                     "val_nll_first": val0, "val_nll_best": best["val_nll"],
                     "loss_decreased": bool(best["val_nll"] < val0),
                     "history": history},
        "eval": {"val_nll": val_best["weighted_nll"],
                 "test_nll": test["unweighted_nll"],
                 "test_nll_weighted": test["weighted_nll"],
                 "test_n_rows": test["n_rows"], "test_sum_w": test["sum_w"]},
        "throughput": throughput,
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
    if throughput.get("measured_epochs"):
        print(f"throughput: {throughput['median_rows_per_sec']:,.0f} train rows/s  "
              f"({throughput['median_train_sec']:.1f}s train / {throughput['min_per_epoch']:.2f} "
              f"min per epoch, median over {throughput['measured_epochs']} steady-state epochs"
              f"{'; epoch0 warmup excluded' if throughput.get('excluded_warmup_epoch0') else ''})")
    print(f"wrote {run / 'metrics.json'}")
    return run


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def _throughput_summary(history: list, n_train: int) -> dict:
    """Per-epoch wall-clock throughput for sizing the M10 full export (`02 §3`: dev
    ≈5–10 M train rows → full ≈50–100 M). Reported on the **steady-state** epochs (epoch 0
    excluded — it pays CUDA context + cuDNN autotune init), median to shrug off scheduler
    jitter. ``rows_per_sec`` is over the train slice (what scales with export size);
    ``min_per_epoch`` is the full epoch incl. val eval + checkpoint."""
    steady = [h for h in history if h["epoch"] > 0 and "train_sec" in h] or \
             [h for h in history if "train_sec" in h]
    if not steady:
        return {"n_train_rows": n_train, "measured_epochs": 0}
    train_secs = [h["train_sec"] for h in steady]
    epoch_secs = [h["epoch_sec"] for h in steady]
    med_train = _median(train_secs)
    return {
        "n_train_rows": n_train,
        "measured_epochs": len(steady),
        "excluded_warmup_epoch0": any(h["epoch"] == 0 for h in history),
        "median_train_sec": med_train,
        "median_epoch_sec": _median(epoch_secs),
        "median_rows_per_sec": n_train / med_train if med_train > 0 else float("nan"),
        "min_per_epoch": _median(epoch_secs) / 60.0,
        "epoch0_train_sec": next((h["train_sec"] for h in history if h["epoch"] == 0), None),
    }


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

    if metrics.get("smoke"):
        print(f"\n[note] 'val NLL improves on logit' + 'no OOM at scale' are the GPU dev-fit "
              f"Accept criteria — deferred (smoke is capped/unconverged).")
    else:
        print(f"\n[5] GPU dev-fit Accept criteria (val NLL beats logit; throughput recorded)")
        cmp = metrics.get("logit_comparison")
        check("logit_comparison present (M7 run found)", cmp is not None)
        if cmp is not None:
            check(f"val NLL {cmp['nn_val_nll']:.6f} < logit val NLL {cmp['logit_val_nll']:.6f}",
                  cmp["beats_logit_val"], f"Δ={cmp['val_margin']:+.6f}")
            # Informational (not gated): test-slice margin — the headline §1 metric.
            print(f"  [info] test NLL {cmp['nn_test_nll']:.6f} vs logit {cmp['logit_test_nll']:.6f}"
                  f"  (Δ={cmp['test_margin']:+.6f}, {'beats' if cmp['beats_logit_test'] else 'does not beat'})")
        thr = metrics.get("throughput", {})
        check("per-epoch throughput recorded (rows/sec + min/epoch)",
              bool(thr.get("measured_epochs")),
              f"{thr.get('median_rows_per_sec', float('nan')):,.0f} rows/s, "
              f"{thr.get('min_per_epoch', float('nan')):.2f} min/epoch" if thr.get("measured_epochs") else "")
    print(f"\n{'ALL ACCEPT CRITERIA PASS' if ok else 'SOME CHECKS FAILED'} "
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
    ap.add_argument("--width-mult", type=float, default=1.0,
                    help="scale the paper hidden widths (M12 width sweep; 1.0=200/140)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override (default 40, or 8 in --smoke)")
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (active only on cuda; no-op on cpu/mps)")
    ap.add_argument("--gpu-resident", action="store_true",
                    help="M10 full-scale: hold the whole split on the GPU and index on-device "
                         "(removes the per-batch host→device bottleneck; cuda only)")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU pipeline check on ≤100k rows (standing rule 3)")
    ap.add_argument("--augment", action="store_true",
                    help="M26a: add the history-summary features (prev_state, "
                         "months-since-delinq, ever-delinquent, episode count) to the input")
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
