"""M26 (step 2a) — seq-to-one training loop + TINY smoke (``04 §M26``).

Trains the :class:`seq_model.SeqGRU` on k=2015 sequences through the SAME importance-weighted
loss + NLL/AUC metric path as every other model: ``torch_common.weighted_ce`` for the loss,
and ``evaluate._nll`` / ``evaluate._auc_one_vs_rest`` for the reported val numbers (the metric
is NOT reimplemented). The GRU is a *discriminating* model for the value-of-memory probe, not
a tuned competitor.

This module is the **smoke only** (1 seed, few epochs, small stratified train subsample, val
on a random subsample for orientation). The full 3-seed / full-val spike (step 2b) is the same
code at larger caps — NOT run here.

Run (Mac, SSD mounted):
    .venv/bin/python -m floan.model.seq_train --smoke
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from floan.model import config
from floan.model import evaluate as EV      # _nll / _auc_one_vs_rest — the SAME metric path
from floan.model import sequence as S
from floan.model import seq_model as SM
from floan.model import torch_common as tc

# M26a banked FF arms on the FULL k=2015 val slice (orientation reference; see M26a_NOTES.md).
M26A_FF_BASELINE_VAL = 0.096190
M26A_FF_HISTORY_VAL = 0.089308
# The three M25 display transitions (illustrative).
TRANS = {"current->prepaid": ("current", "prepaid"),
         "dpd_90plus->foreclosure": ("dpd_90plus", "foreclosure"),
         "current->dpd_30": ("current", "dpd_30")}


@torch.no_grad()
def _val_probs(model, predict, arr: S.SeqArrays, device, batch: int = 4096) -> np.ndarray:
    """Softmax probs ``[N,7]`` over a split (float64 softmax on host — the evaluate numerics)."""
    model.eval()
    out = []
    for b in S.iter_batches(arr, batch, shuffle=False):
        logits = predict(model, b, device).float().cpu()
        out.append(torch.softmax(logits.double(), dim=1).float().numpy())
    return np.concatenate(out)


def _transition_aucs(probs: np.ndarray, y: np.ndarray, origin: np.ndarray) -> dict:
    res = {}
    for name, (o, d) in TRANS.items():
        m = origin == EV.OI[o]
        res[name] = (EV._auc_one_vs_rest(probs[m][:, EV.SI[d]], (y[m] == EV.SI[d]))
                     if m.any() else None)
    return res


def smoke(variant: str = "dev", k: int = config.TUNING_YEAR, *, hidden: int = 48,
          epochs: int = 5, seed: int = 0, T: int = S.SEQ_LEN, train_n: int = 40_000,
          train_per_origin: int | None = None, val_n: int = 50_000, batch_size: int = 512,
          lr: float = 1e-3, device: str = "cpu") -> dict:
    dev = tc.resolve_device(device)
    tc.set_seed(seed)
    print(f"=== M26 step-2a seq training SMOKE  variant={variant} k={k}  GRU hidden={hidden}  "
          f"epochs={epochs} seed={seed} device={dev} ===")

    # Train sampling: UNIFORM over train_pool + the HT weights (1/p_keep) is the M26a-faithful,
    # unbiased, calibration-preserving choice — the train_pool is already current-thinned
    # (~25% current-origin, not 87%), so equal-per-origin stratification OVER-balances it and
    # shifts the model's marginals, inflating the natural-mix val-NLL. (Stratified path kept
    # via --train-per-origin for inspection, but it is NOT comparable to M26a's numbers.)
    t0 = time.perf_counter()
    if train_per_origin is not None:
        tr_pts = S.prediction_points(variant, k, "train", origin_cap=train_per_origin, seed=seed)
        tdesc = f"STRATIFIED ≤{train_per_origin}/origin (NOT calibration-comparable)"
    else:
        tr_pts = S.prediction_points(variant, k, "train", sample_n=train_n, seed=seed)
        tdesc = "uniform over train_pool (+ HT weights; M26a-faithful)"
    va_pts = S.prediction_points(variant, k, "val", sample_n=val_n, seed=seed)
    print(f"points: train={tr_pts.height:,} [{tdesc}]  origins "
          + ", ".join(f"{s}={tr_pts.filter(tr_pts['state'] == s).height}"
                      for s in config.ORIGIN_STATES) + f"  val={va_pts.height:,} (random)")
    tr, scaler, vocab = S.build_split(variant, k, "train", tr_pts, T)
    va, _, _ = S.build_split(variant, k, "val", va_pts, T, scaler, vocab)
    print(f"tensors: train cont{tr.cont.shape} cat{tr.cat.shape} bin{tr.bin.shape}  "
          f"val cont{va.cont.shape}   built in {time.perf_counter() - t0:.0f}s")

    model = SM.build(scaler, vocab, hidden=hidden).to(dev)
    predict = SM.make_seq_predict()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"GRU: input_dim={model.input_dim}, hidden={hidden}, {model.n_params():,} params")

    va_origin = (va_pts.select(va_pts["state"].replace_strict(
        list(EV.OI), list(EV.OI.values()), default=-1, return_dtype=int))
        .to_numpy().reshape(-1))

    print(f"\n  {'epoch':>5}  {'train_loss':>10}  {'val_nll(EV._nll)':>16}")
    history = []
    for ep in range(epochs):
        model.train()
        swce = sw = 0.0
        for b in S.iter_batches(tr, batch_size, shuffle=True, seed=seed, epoch=ep):
            y = tc._as_long(b["y"], dev)
            w = tc._as_f32(b["w"], dev)
            opt.zero_grad(set_to_none=True)
            loss = tc.weighted_ce(predict(model, b, dev), y, w)
            loss.backward()
            opt.step()
            wsum = float(w.double().sum())
            swce += float(loss.detach().double()) * wsum
            sw += wsum
        train_loss = swce / sw if sw else float("nan")
        val_nll = EV._nll(_val_probs(model, predict, va, dev), va.y)
        history.append({"epoch": ep, "train_loss": train_loss, "val_nll": val_nll})
        print(f"  {ep:>5}  {train_loss:>10.6f}  {val_nll:>16.6f}")

    # Final orientation read — GRU val-NLL via evaluate._nll + the 3 transition AUCs.
    probs = _val_probs(model, predict, va, dev)
    val_nll = EV._nll(probs, va.y)
    aucs = _transition_aucs(probs, va.y, va_origin)
    decreased = history[-1]["train_loss"] < history[0]["train_loss"]

    print("\n" + "=" * 70)
    print("M26 step-2a SMOKE — GRU value-of-memory (ORIENTATION ONLY; NOT a verdict)")
    print("=" * 70)
    print(f"  (a) train loss decreased: {history[0]['train_loss']:.6f} → "
          f"{history[-1]['train_loss']:.6f}  [{'YES' if decreased else 'NO'}]")
    print(f"  (b) GRU val-NLL via evaluate._nll: {val_nll:.6f}  "
          f"(val subsample n={va.y.shape[0]:,})")
    print("  (d) orientation vs M26a banked FF arms (FULL val):")
    print(f"        FF current-state-only : {M26A_FF_BASELINE_VAL:.6f}")
    print(f"        FF + history (M26a)   : {M26A_FF_HISTORY_VAL:.6f}")
    print(f"        GRU (this smoke)      : {val_nll:.6f}   "
          f"← 1 seed / {epochs} ep / train {tr.y.shape[0]:,} subsample / val subsample")
    print("      NOT comparable as a verdict — subsampled, under-trained, 1 seed.")
    print("  AUC (val, illustrative): " + "  ".join(
        f"{n}={'—' if v is None else f'{v:.3f}'}" for n, v in aucs.items()))
    print("=" * 70)
    return {"history": history, "val_nll": val_nll, "aucs": aucs,
            "n_train": int(tr.y.shape[0]), "n_val": int(va.y.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description="M26 step-2a seq training smoke (no full run).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-n", type=int, default=40_000,
                    help="uniform train_pool subsample (M26a-faithful; default)")
    ap.add_argument("--train-per-origin", type=int, default=None,
                    help="stratified alternative (NOT calibration-comparable to M26a)")
    ap.add_argument("--val-n", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--smoke", action="store_true", help="run the tiny training smoke")
    args = ap.parse_args()
    config.require_drive()
    smoke(args.variant, args.k, hidden=args.hidden, epochs=args.epochs, seed=args.seed,
          train_n=args.train_n, train_per_origin=args.train_per_origin, val_n=args.val_n,
          batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
