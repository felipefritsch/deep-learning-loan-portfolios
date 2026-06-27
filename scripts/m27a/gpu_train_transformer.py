"""W2 — Transformer fourth arm: train+score the sequence transformer on the 10M caches.

A faithful mirror of ``gpu_train_all.py`` (the GRU trainer) — SAME caches, importance-weighted
loss, ``evaluate`` metric path, optimiser, early-stopping protocol, and seed plan — with the ONLY
change being the architecture: ``floan.model.seq_transformer`` (self-attention) instead of
``seq_model`` (GRU). So a transformer-vs-GRU out-of-sample NLL difference is attributable to
architecture alone, exactly as the GRU-vs-FF arms are.

Artifacts: ``RESULTS/k{k}_xf_s{seed}.npz`` (test probs/y/origin) + ``.json`` (metrics) — the ``_xf_``
tag keeps them distinct from the GRU's ``k{k}_s{seed}`` and the FF arms' ``k{k}_ff_*``. Resumable per
(k, seed): a finished run is skipped.

  python scripts/m27a/gpu_train_transformer.py --windows 2015   # k2015 go/no-go (seed 0) vs the GRU
  python scripts/m27a/gpu_train_transformer.py                  # full roll: 11 windows + key seeds 1,2
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time

import numpy as np
import torch

from floan.model import config as mc
from floan.model import features as F
from floan.model import evaluate as EV
from floan.model import sequence as S
from floan.model import seq_transformer as XF
from floan.model import seq_train as ST
from floan.model import torch_common as tc

# ---- frozen config (training protocol mirrors gpu_train_all.py; the architecture lives in
#      seq_transformer.py). No HIDDEN — the transformer width is d_model in seq_transformer. ----
BATCH, LR, WD = 4096, 1e-3, 1e-5
MAX_EPOCHS, PATIENCE = 40, 5
WINDOWS = list(range(2015, 2026))
KEY = [2015, 2019, 2020, 2023, 2025]
EXTRA_SEEDS = [1, 2]
CACHE = mc.OUTPUTS / "seq_cache" / "full"
RESULTS = mc.OUTPUTS / "m27a_gpu_runs"
RESULTS.mkdir(parents=True, exist_ok=True)


def per_transition_auc(probs, y, origin):
    y = y.astype(np.int64); rows = []
    for s, oi in EV.OI.items():
        m = origin == oi
        if m.sum() == 0:
            continue
        ys = y[m]
        for dest in EV.REACHABLE[s]:
            di = EV.SI[dest]; pos = (ys == di)
            rows.append({"origin": s, "destination": dest, "n": int(m.sum()),
                         "n_pos": int(pos.sum()), "auc": EV._auc_one_vs_rest(probs[m][:, di], pos)})
    return rows


def run_one(k: int, seed: int):
    npz = RESULTS / f"k{k}_xf_s{seed}.npz"
    js = RESULTS / f"k{k}_xf_s{seed}.json"
    if npz.exists() and js.exists():
        print(f"[k{k} xf s{seed}] SKIP (artifacts exist)", flush=True)
        return
    dev = tc.resolve_device("cuda"); tc.set_seed(seed)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()

    scaler, vocab = F.load_pipeline(CACHE / f"k{k}")
    tr = S.to_seqarrays(S.load_split(CACHE / f"k{k}" / "train"))
    dva = S.load_split(CACHE / f"k{k}" / "val"); va = S.to_seqarrays(dva)
    load_s = time.perf_counter() - t0

    model = XF.build(scaler, vocab).to(dev)
    predict = XF.make_seq_predict()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_nll, best_state, best_ep, since = float("inf"), None, -1, 0
    t_tr = time.perf_counter()
    for ep in range(MAX_EPOCHS):
        model.train()
        for b in S.iter_batches(tr, BATCH, shuffle=True, seed=seed, epoch=ep):
            y = tc._as_long(b["y"], dev); w = tc._as_f32(b["w"], dev)
            opt.zero_grad(set_to_none=True)
            tc.weighted_ce(predict(model, b, dev), y, w).backward()
            opt.step()
        val_nll = EV._nll(ST._val_probs(model, predict, va, dev, BATCH), va.y)
        sched.step(val_nll)
        if val_nll < best_nll - 1e-5:
            best_nll, best_ep, since = val_nll, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
        print(f"[k{k} xf s{seed}] ep {ep:>2} val_nll {val_nll:.6f} lr {opt.param_groups[0]['lr']:.1e}"
              f"{'  *best' if since == 0 else ''}", flush=True)
        if since >= PATIENCE:
            break
    train_s = time.perf_counter() - t_tr
    model.load_state_dict(best_state)
    del tr, va, dva; gc.collect()

    # score TEST
    dte = S.load_split(CACHE / f"k{k}" / "test")
    te = S.to_seqarrays(dte); te_org = dte["origin"]
    t_sc = time.perf_counter()
    te_probs = ST._val_probs(model, predict, te, dev, BATCH)
    test_nll = EV._nll(te_probs, te.y)
    test_auc = per_transition_auc(te_probs, te.y, te_org)
    score_s = time.perf_counter() - t_sc
    peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9

    np.savez(npz, probs=te_probs.astype(np.float32), y=te.y.astype(np.int64),
             origin=te_org.astype(np.int64))
    js.write_text(json.dumps({
        "k": k, "seed": seed, "model": "transformer", "n_test": int(te.y.shape[0]),
        "test_nll": test_nll, "val_nll": best_nll, "best_epoch": best_ep,
        "eff_epochs": best_ep + 1, "test_auc": test_auc,
        "load_s": load_s, "train_s": train_s, "score_s": score_s, "peak_gpu_gb": peak_gb,
    }, indent=2))
    print(f"[k{k} xf s{seed}] DONE test_nll {test_nll:.6f} val_nll {best_nll:.6f} "
          f"(best ep {best_ep}, {best_ep+1} eff)  load {load_s:.0f}s train {train_s:.0f}s "
          f"score {score_s:.0f}s  peakGPU {peak_gb:.2f}GB", flush=True)
    del te, dte, te_probs; gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="W2 transformer arm — train+score (mirror of gpu_train_all).")
    ap.add_argument("--windows", type=int, nargs="+", default=None,
                    help="restrict to these windows, seed 0 only (e.g. --windows 2015 for the go/no-go); "
                         "default = all 11 windows seed 0 + key-window seeds 1,2")
    args = ap.parse_args()
    if args.windows:
        runs = [(k, 0) for k in args.windows]
    else:
        runs = [(k, 0) for k in WINDOWS] + [(k, s) for k in KEY for s in EXTRA_SEEDS]
    print(f"=== W2 transformer run: {len(runs)} (window,seed) runs ===", flush=True)
    for i, (k, seed) in enumerate(runs):
        print(f"\n--- [{i+1}/{len(runs)}] k{k} xf seed{seed} ---", flush=True)
        run_one(k, seed)
    print("\n=== ALL TRANSFORMER RUNS COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
