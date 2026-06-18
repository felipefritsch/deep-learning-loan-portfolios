"""M9 — Deep-net ensemble + ensemble-size curve on the tuning window (``02_LOAN_LEVEL §6``).

Trains **8 independent nets** at the grid-selected config (``§6`` writes "8 five-layer
nets"; that was the *paper's* tuning outcome — we ensemble the config our own val-NLL
selection froze, so the ensemble-vs-single delta holds architecture constant) — independent
random init + per-seed shuffle order are the diversity source — then **averages their
predicted probabilities** and reports the
out-of-sample-NLL-vs-ensemble-size curve (paper Fig 7). Each member is a full
:func:`train.train` run (same tested loss / early stopping / metrics), so members are
self-describing run folders ``models/nn/<variant>/ens_k<k>_s<seed>/`` and the step is
member-level idempotent.

Probability averaging (not logit averaging) is the ensemble rule: for member set ``S``,
``p̄ = mean_{m∈S} softmax(logits_m)`` and ``NLL = mean_row −log p̄[y]``. The curve adds
members in seed order and recomputes ``p̄`` at each size with an O(N×7) running sum, so the
8-member sweep needs one probability accumulator, not eight prediction matrices.

Bagging note (``§6`` lists "bootstrapped/resampled shard subsets"): at dev scale the pool is
small and the whole train slice lives in RAM, so the ensemble's variance reduction comes
from random init + SGD-order diversity alone. Full shard-subset bagging is the M10
full-scale ensemble's job (where data subsetting actually bites); the averaging machinery
here is identical, so M10 reuses it unchanged.

Run:
    .venv/bin/python -m floan.model.ensemble --device cuda --amp           # train 8 + curve
    .venv/bin/python -m floan.model.ensemble --device cuda --amp --curve-only  # curve from done members
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch

from floan.model import config
from floan.model import data as D
from floan.model import features as F
from floan.model import net as N
from floan.model import torch_common as tc
from floan.model import train as T

N_MEMBERS = 8                  # paper Fig 7: ensemble of 8
EVAL_BATCH = 16384


def _ens_config(variant: str, k: int) -> dict:
    """The grid-selected 5-layer ensemble config (depth/dropout/L2) from grid_summary.json."""
    p = config.MODELS / "nn" / variant / "grid_summary.json"
    if not p.exists():
        raise SystemExit(f"no grid_summary.json at {p} — run grid.py first")
    return json.loads(p.read_text())["selected"]["ensemble"]


def _member_args(cfg: dict, seed: int, base: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        variant=base.variant, k=base.k, device=base.device,
        depth=cfg["depth"], dropout=cfg["dropout"], weight_decay=cfg["weight_decay"],
        lr=T.LR, batch_size=T.BATCH_SIZE, max_epochs=base.max_epochs, patience=T.PATIENCE,
        seed=seed, amp=base.amp, smoke=base.smoke,
        run_name=f"ens_k{base.k}_s{seed}", fresh=False,
        stop_after_epoch=None, verify_only=False)


def _member_run(variant: str, k: int, seed: int):
    return config.MODELS / "nn" / variant / f"ens_k{k}_s{seed}"


@torch.no_grad()
def _member_probs(run, enc: dict, device) -> np.ndarray:
    """Softmax class probabilities ``[N, 7]`` for one trained member on a preloaded split."""
    arch = json.loads((run / "metrics.json").read_text())["architecture"]
    model = N.from_arch({k: arch[k] for k in
                         ("n_continuous", "n_binary", "vocab_sizes", "emb_dims",
                          "hidden_dims", "n_classes", "dropout")}).to(device)
    model.load_state_dict(torch.load(run / "best_model.pt", map_location=device))
    model.eval()
    predict = N.make_nn_predict()
    out = np.empty((enc["y"].shape[0], F.N_CLASSES), dtype=np.float64)
    for s in range(0, enc["y"].shape[0], EVAL_BATCH):
        b = {kk: enc[kk][s:s + EVAL_BATCH] for kk in ("cont", "cat", "bin")}
        logits = predict(model, b, device)
        out[s:s + EVAL_BATCH] = torch.softmax(logits.double(), dim=1).cpu().numpy()
    return out


def _nll(probs: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Mean −log p[y]; weighted (Σw·/Σw) when weights given (the §6.4 in-sample estimator)."""
    pick = probs[np.arange(probs.shape[0]), y]
    ll = -np.log(np.clip(pick, 1e-300, None))
    return float((w * ll).sum() / w.sum()) if w is not None else float(ll.mean())


def curve(runs: list, enc: dict, device, weighted: bool) -> list[dict]:
    """Ensemble-size curve over members added in order: NLL of mean-prob at sizes 1..len."""
    y = np.asarray(enc["y"])
    w = np.asarray(enc["w"], dtype=np.float64) if weighted else None
    acc = np.zeros((y.shape[0], F.N_CLASSES), dtype=np.float64)
    pts = []
    for m, run in enumerate(runs, 1):
        acc += _member_probs(run, enc, device)
        pts.append({"size": m, "nll": _nll(acc / m, y, w)})
    return pts


def run(base: argparse.Namespace) -> dict:
    cfg = _ens_config(base.variant, base.k)
    print(f"=== M9 ensemble: {N_MEMBERS}× depth={cfg['depth']} dropout={cfg['dropout']} "
          f"L2={cfg['weight_decay']:g} on k={base.k} ({base.variant}) ===")

    if not base.curve_only:
        for seed in range(N_MEMBERS):
            print(f"\n--- member seed {seed}/{N_MEMBERS - 1} ---")
            T.train(_member_args(cfg, seed, base))

    runs = [_member_run(base.variant, base.k, s) for s in range(N_MEMBERS)]
    missing = [str(r) for r in runs if not (r / "best_model.pt").exists()]
    if missing:
        raise SystemExit(f"missing trained members: {missing}")

    # One scaler/vocab (deterministic on the train slice — seed-independent), encode once.
    scaler, vocab = F.load_pipeline(runs[0])
    cap = T.SMOKE_CAP if base.smoke else None
    enc_val = T.preload_split(base.variant, base.k, "val", scaler, vocab, cap)
    enc_test = T.preload_split(base.variant, base.k, "test", scaler, vocab, cap)
    device = tc.resolve_device(base.device)

    members = [{"seed": s, "run": str(runs[s]),
                "val_nll": _nll(_member_probs(runs[s], enc_val, device), np.asarray(enc_val["y"])),
                "test_nll": _nll(_member_probs(runs[s], enc_test, device), np.asarray(enc_test["y"]))}
               for s in range(N_MEMBERS)]
    val_curve = curve(runs, enc_val, device, weighted=False)    # eval pool: w≡1
    test_curve = curve(runs, enc_test, device, weighted=False)

    mean_single = {"val_nll": float(np.mean([m["val_nll"] for m in members])),
                   "test_nll": float(np.mean([m["test_nll"] for m in members]))}
    summary = {
        "window_k": base.k, "variant": base.variant, "n_members": N_MEMBERS,
        "config": cfg, "members": members, "mean_single_nll": mean_single,
        "val_curve": val_curve, "test_curve": test_curve,
        "ensemble_val_nll": val_curve[-1]["nll"], "ensemble_test_nll": test_curve[-1]["nll"],
    }
    out = config.MODELS / "nn" / base.variant / "ensemble_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== ensemble-size curve (test NLL, members added in seed order) ===")
    for v, t in zip(val_curve, test_curve):
        print(f"  size {t['size']}: val={v['nll']:.6f}  test={t['nll']:.6f}")
    print(f"\nmean single net: val={mean_single['val_nll']:.6f} test={mean_single['test_nll']:.6f}")
    print(f"8-net ensemble : val={summary['ensemble_val_nll']:.6f} "
          f"test={summary['ensemble_test_nll']:.6f}  "
          f"(Δtest vs mean single {mean_single['test_nll'] - summary['ensemble_test_nll']:+.6f})")
    print(f"wrote {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="M9 deep-net ensemble + size curve (tuning window).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--curve-only", action="store_true",
                    help="compute the curve from already-trained members (no training)")
    args = ap.parse_args()
    config.require_drive()
    run(args)


if __name__ == "__main__":
    main()
