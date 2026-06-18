"""M10b — equivalence receipt for the backtest's reimplemented logit.

The rolling loop's per-window logit (:class:`backtest.LogitEmbNet`) is a *new* implementation
of the multinomial logit: it folds the categorical one-hot·weight product into one
``Embedding(vocab_c, n_classes)`` per column instead of materialising the ~1.4k-wide drop-first
one-hot design the M7 benchmark (:mod:`logit`) used and cross-checked against sklearn. The two
are algebraically the same affine map, but only the M7 path carries the validated receipt — so
before trusting the new logit across all 11 windows we reproduce M7's numbers once.

The clean comparison point is **L2 = 0**: there the unregularised multinomial logit over the
*full* one-hot (this code) and over the *drop-first* one-hot (M7) span the same column space and
softmax is invariant to the dropped reference, so the fitted predictions — hence NLL — are
identical up to optimisation tolerance. With the same optimiser settings (lr 5e-3, batch 8192,
≤25 epochs, patience 4, seed 0) the val NLL must agree to ~1e-3. We also re-run M7's full L2 grid
on the dev k=2015 slice and confirm the selected-config test NLL matches.

Runs on CPU at dev scale in a couple of minutes (no GPU needed; leaves the box's GPU for the
depth check). Writes ``models/logit/logit_equiv_receipt.json``.

    .venv/bin/python -m floan.model.verify_logit_equiv
"""

from __future__ import annotations

import json

import numpy as np
import torch

from floan.model import backtest as B
from floan.model import config
from floan.model import data as D
from floan.model import features as F
from floan.model import train as T

VARIANT = "dev"
K = 2015
WD_GRID = (0.0, 1e-6, 1e-5, 1e-4)   # M7's exact grid (logit.WD_GRID)
TOL = 1e-3


def main() -> None:
    config.require_drive()
    device = torch.device("cpu")
    m7 = json.loads((config.MODELS / "logit" / VARIANT / f"k{K}" / "metrics.json").read_text())
    m7_val = {w: r["val_nll"] for w, r in m7["l2_selection"]["per_wd"].items()}
    print(f"=== logit equivalence receipt  ({VARIANT} k={K}, CPU) ===")
    print(f"M7 (drop-first one-hot, sklearn-validated): selected wd={m7['l2_selection']['selected_wd']:g}"
          f"  val={m7['logit']['val_nll']:.6f}  test={m7['logit']['test_nll']:.6f}")

    scaler, vocab = D.fit_window(VARIANT, K)
    enc_train = T.preload_split(VARIANT, K, "train", scaler, vocab, None)
    enc_val = T.preload_split(VARIANT, K, "val", scaler, vocab, None)
    enc_test = T.preload_split(VARIANT, K, "test", scaler, vocab, None)
    T_train, T_val, T_test = (T.to_resident(enc_train, device), T.to_resident(enc_val, device),
                              T.to_resident(enc_test, device))
    n_cont, n_bin, vs = len(scaler.cols), len(F.BINARY), list(vocab.vocab_sizes)

    rows, best = [], {"wd": None, "val_nll": float("inf"), "model": None}
    for wd in WD_GRID:
        model, info, hist = B._train_logit_one(T_train, T_val, n_cont, n_bin, vs, wd,
                                               device, use_amp=False)
        new_val = info["val_nll"]
        ref_val = m7_val.get(str(wd))
        d = abs(new_val - ref_val) if ref_val is not None else None
        rows.append({"wd": wd, "new_val_nll": new_val, "m7_val_nll": ref_val,
                     "abs_diff": d, "best_epoch": info["epoch"], "n_epochs": len(hist)})
        print(f"  wd={wd:<7g}  new val={new_val:.6f}  M7 val={ref_val:.6f}  "
              f"|Δ|={d:.2e}  {'OK' if d < TOL else 'DIFF'}")
        if new_val < best["val_nll"]:
            best = {"wd": wd, "val_nll": new_val, "model": model}

    test = T.evaluate_nll_resident(best["model"], T_test, B.EVAL_BATCH, device)
    test_diff = abs(test["unweighted_nll"] - m7["logit"]["test_nll"])
    wd0 = next(r for r in rows if r["wd"] == 0.0)

    passed = bool(wd0["abs_diff"] < TOL and all(r["abs_diff"] < TOL for r in rows)
                  and test_diff < TOL)
    receipt = {
        "variant": VARIANT, "window_k": K, "tol": TOL, "passed": passed,
        "comparison": "backtest.LogitEmbNet (full one-hot, embedding-sum) vs M7 logit.py "
                      "(drop-first one-hot, sklearn-validated)",
        "optimizer": {"lr": B.LOGIT_LR, "batch_size": B.LOGIT_BATCH,
                      "max_epochs": B.LOGIT_MAX_EPOCHS, "patience": B.LOGIT_PATIENCE,
                      "seed": B.SEED},
        "per_wd": rows,
        "wd0_abs_diff": wd0["abs_diff"],
        "selected": {"new_wd": best["wd"], "new_val_nll": best["val_nll"],
                     "new_test_nll": test["unweighted_nll"],
                     "m7_selected_wd": m7["l2_selection"]["selected_wd"],
                     "m7_test_nll": m7["logit"]["test_nll"], "test_abs_diff": test_diff},
        "git_commit": T._git_commit(),
    }
    out = config.MODELS / "logit" / "logit_equiv_receipt.json"
    out.write_text(json.dumps(receipt, indent=2))
    print(f"\nL2=0 val |Δ| = {wd0['abs_diff']:.2e}  (provable-equivalence point)")
    print(f"selected test |Δ| = {test_diff:.2e}  (new {test['unweighted_nll']:.6f} vs "
          f"M7 {m7['logit']['test_nll']:.6f})")
    print(f"{'EQUIVALENCE CONFIRMED' if passed else 'EQUIVALENCE FAILED'} (tol {TOL}) → {out}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
