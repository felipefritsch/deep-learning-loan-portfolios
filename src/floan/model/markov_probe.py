"""M26a — the Markov-assumption probe driver (``04 §M26a``).

Trains TWO arms at dev k=2015 through the **identical** ``train.py`` loop — a baseline
(current-state-only, ``--augment`` off) and an augmented net (``--augment`` on, the four
``history.py`` summaries added) — across the M12 3-seed protocol, then re-scores both on
the SAME val/test slices through the SAME ``evaluate.py`` metric path (``_nll`` /
``_auc_one_vs_rest``). The only difference between the arms is the feature set, so the
val-NLL gap is attributable to path-dependence alone.

Decision (improvement = baseline−augmented val-NLL, against the baseline seed sd):
  * **≥ 2σ**  → signal: path-dependence exists at dev scale → escalate to M26;
  * **1σ–2σ** → suggestive only → record as future-work (M26 NOT green-lit);
  * **< 1σ**  → current-state conditioning sufficient at dev scale (M26/M27 future-work).

Run (Mac, SSD mounted):
    .venv/bin/python -m floan.model.markov_probe --smoke --seeds 0 1   # wiring check
    .venv/bin/python -m floan.model.markov_probe --device auto         # the real 3-seed probe
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import data as D
from floan.model import evaluate as EV     # _nll / _auc_one_vs_rest — the SAME metric path
from floan.model import features as F
from floan.model import history as H
from floan.model import robustness as R    # _mean / _sd — the M12 seed-stats helpers
from floan.model import torch_common as tc
from floan.model import train as T

NN = config.NN_SELECTED                     # frozen depth/dropout/L2 (k=2015 full-scale argmin)
EVAL_BATCH = 16384
# The three M25 display transitions (illustrative; NLL decides).
TRANS = {"current->prepaid": ("current", "prepaid"),
         "dpd_90plus->foreclosure": ("dpd_90plus", "foreclosure"),
         "current->dpd_30": ("current", "dpd_30")}


def _run_name(augment: bool, seed: int, smoke: bool) -> str:
    return f"mp_{'aug' if augment else 'base'}_s{seed}" + ("_smoke" if smoke else "")


def _args(variant: str, k: int, seed: int, augment: bool, device_name: str,
          smoke: bool) -> SimpleNamespace:
    """A ``train.main``-shaped namespace — same optimizer/schedule/patience as every other
    net; only ``augment`` (and the run name / seed) vary."""
    return SimpleNamespace(
        variant=variant, k=k, device=device_name,
        depth=NN["depth"], dropout=NN["dropout"], weight_decay=NN["weight_decay"],
        width_mult=1.0, lr=T.LR, batch_size=T.BATCH_SIZE,
        max_epochs=(3 if smoke else None), patience=T.PATIENCE, seed=seed,
        amp=False, gpu_resident=False, smoke=smoke,
        run_name=_run_name(augment, seed, smoke), fresh=False, stop_after_epoch=None,
        augment=augment)


# ---------------------------------------------------------------------------
# Scoring — build the slice encoding (augment-aware) and route through evaluate.py
# ---------------------------------------------------------------------------
def _encode_split(variant: str, k: int, split: str, scaler: F.Scaler, vocab: F.Vocab,
                  augment: bool, cap: int | None) -> tuple[dict, np.ndarray]:
    pool_dir, bounds = D.window_spec(variant, k, split)
    lf = D._masked_scan(pool_dir, bounds)
    if cap is not None:
        lf = lf.limit(cap)
    df = F.prepare_raw(lf.collect())
    if augment:
        df = H.join_history(df, variant, k)
        enc = F.encode_frame(df, scaler, vocab, encode="index",
                             binary_cols=F.BINARY + H.HIST_BINARY)
    else:
        enc = F.encode_frame(df, scaler, vocab, encode="index")
    origin = (df.select(pl.col("state").replace_strict(
        list(EV.OI), list(EV.OI.values()), default=-1, return_dtype=pl.Int64))
        .to_numpy().reshape(-1))
    return enc, origin


@torch.no_grad()
def _probs(model, enc: dict, device) -> np.ndarray:
    """Softmax class probs ``[N,7]`` — float64 softmax on host (the evaluate.py numerics;
    also mps-safe, which has no float64)."""
    model.eval()
    n = int(enc["y"].shape[0])
    out = np.empty((n, F.N_CLASSES), dtype=np.float32)
    for s in range(0, n, EVAL_BATCH):
        sl = slice(s, s + EVAL_BATCH)
        cont = torch.as_tensor(np.asarray(enc["cont"][sl]), dtype=torch.float32, device=device)
        cat = torch.as_tensor(np.asarray(enc["cat"][sl]), dtype=torch.long, device=device)
        binb = torch.as_tensor(np.asarray(enc["bin"][sl]), dtype=torch.float32, device=device)
        logits = model(cont, cat, binb).float().cpu()
        out[sl] = torch.softmax(logits.double(), dim=1).float().numpy()
    return out


def _transition_aucs(probs: np.ndarray, y: np.ndarray, origin: np.ndarray) -> dict:
    res = {}
    for name, (o, d) in TRANS.items():
        m = origin == EV.OI[o]
        res[name] = (EV._auc_one_vs_rest(probs[m][:, EV.SI[d]], (y[m] == EV.SI[d]))
                     if m.any() else None)
    return res


def _score_run(run: Path, variant: str, k: int, augment: bool, device,
               cap: int | None) -> dict:
    scaler, vocab = F.load_pipeline(run)
    model = EV._load_nn(run, device)
    out = {}
    for split in ("val", "test"):
        enc, origin = _encode_split(variant, k, split, scaler, vocab, augment, cap)
        probs = _probs(model, enc, device)
        y = np.asarray(enc["y"]).astype(np.int64)
        out[split] = {"nll": EV._nll(probs, y), "n": int(y.shape[0]),
                      "auc": _transition_aucs(probs, y, origin)}
    return out


# ---------------------------------------------------------------------------
# Aggregation + decision
# ---------------------------------------------------------------------------
def _auc_mean(rows: list[dict], arm_key, name: str) -> float | None:
    vals = [arm_key(r)["val"]["auc"][name] for r in rows]
    vals = [v for v in vals if v is not None]
    return R._mean(vals) if vals else None


def run(variant: str, k: int, seeds: list[int], device_name: str, smoke: bool) -> dict:
    config.require_drive()
    device = tc.resolve_device(device_name)
    t0 = time.perf_counter()
    print(f"=== M26a Markov-assumption probe  variant={variant} k={k}  seeds={seeds}  "
          f"device={device}  smoke={smoke} ===")
    H.build(variant, k)                                   # cache the history (idempotent)
    cap = T.SMOKE_CAP if smoke else None

    res: dict[tuple[bool, int], dict] = {}
    for seed in seeds:
        for augment in (False, True):
            arm = "augmented" if augment else "baseline "
            print(f"\n--- train {arm} seed={seed} ---")
            T.train(_args(variant, k, seed, augment, device_name, smoke))
            run_dir = config.MODELS / "nn" / variant / _run_name(augment, seed, smoke)
            res[(augment, seed)] = _score_run(run_dir, variant, k, augment, device, cap)
            print(f"    {arm} s{seed}: val_nll={res[(augment, seed)]['val']['nll']:.6f}  "
                  f"test_nll={res[(augment, seed)]['test']['nll']:.6f}")

    base_val = [res[(False, s)]["val"]["nll"] for s in seeds]
    aug_val = [res[(True, s)]["val"]["nll"] for s in seeds]
    base_test = [res[(False, s)]["test"]["nll"] for s in seeds]
    aug_test = [res[(True, s)]["test"]["nll"] for s in seeds]
    seed_sd = R._sd(base_val)                              # M12-style band: baseline reseeding sd
    base_mean, aug_mean = R._mean(base_val), R._mean(aug_val)
    delta = aug_mean - base_mean                           # < 0 ⇒ augmented improves
    improvement = -delta
    paired = [res[(True, s)]["val"]["nll"] - res[(False, s)]["val"]["nll"] for s in seeds]

    if improvement >= 2 * seed_sd and improvement > 0:
        decision, verdict = "signal", ("path-dependence exists at dev scale (≥2σ) — "
                                       "ESCALATE to M26")
    elif improvement >= seed_sd:
        decision, verdict = "suggestive", ("suggestive only (1σ–2σ) — record as future-work; "
                                           "do NOT green-light M26")
    else:
        decision, verdict = "sufficient", ("current-state conditioning sufficient at dev "
                                           "scale (<1σ) — M26/M27 future-work")

    spot = H.spot_check(variant, k)

    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": "M26a", "variant": variant, "k": k, "seeds": seeds, "smoke": smoke,
        "device": str(device), "git_commit": T._git_commit(),
        "nn_config": NN, "delinq_states": list(H.DELINQ_STATES),
        "history_features": H.HIST_COLS,
        "val_nll": {"baseline": base_val, "augmented": aug_val,
                    "baseline_mean": base_mean, "augmented_mean": aug_mean,
                    "baseline_sd": seed_sd, "augmented_sd": R._sd(aug_val)},
        "test_nll": {"baseline": base_test, "augmented": aug_test,
                     "baseline_mean": R._mean(base_test), "augmented_mean": R._mean(aug_test)},
        "delta_val_nll": delta, "improvement_val_nll": improvement,
        "seed_sd": seed_sd, "one_sigma": seed_sd, "two_sigma": 2 * seed_sd,
        "improvement_over_sd": (improvement / seed_sd) if seed_sd else float("nan"),
        "paired_delta_val": {"values": paired, "mean": R._mean(paired), "sd": R._sd(paired)},
        "decision": decision, "verdict": verdict,
        "auc_val_mean": {
            name: {"baseline": _auc_mean([res[(False, s)] for s in seeds],
                                         lambda r: r, name),
                   "augmented": _auc_mean([res[(True, s)] for s in seeds],
                                          lambda r: r, name)}
            for name in TRANS},
        "leakage_spot_check": spot,
        "wall_sec": time.perf_counter() - t0,
    }
    out = config.MODELS / "nn" / variant / (
        "markov_probe_summary" + ("_smoke" if smoke else "") + ".json")
    out.write_text(json.dumps(summary, indent=2, default=str))
    _report(summary)
    print(f"\nwrote {out}  [{summary['wall_sec'] / 60:.1f} min]")
    return summary


def _report(s: dict) -> None:
    def f(x):
        return "    —   " if x is None else f"{x:.6f}"
    print("\n" + "=" * 64)
    print(f"M26a RESULT — Markov-assumption probe (k={s['k']}, {s['variant']}, "
          f"{len(s['seeds'])} seeds{', SMOKE' if s['smoke'] else ''})")
    print("=" * 64)
    print(f"  {'seed':>4}  {'baseline':>10}  {'augmented':>10}  {'Δ(aug−base)':>12}")
    for i, sd in enumerate(s["seeds"]):
        b, a = s["val_nll"]["baseline"][i], s["val_nll"]["augmented"][i]
        print(f"  {sd:>4}  {b:>10.6f}  {a:>10.6f}  {a - b:>+12.6f}")
    v = s["val_nll"]
    print(f"  {'mean':>4}  {v['baseline_mean']:>10.6f}  {v['augmented_mean']:>10.6f}  "
          f"{s['delta_val_nll']:>+12.6f}")
    print(f"\n  baseline seed sd (band) = {s['seed_sd']:.6e}   "
          f"(1σ={s['one_sigma']:.2e}, 2σ={s['two_sigma']:.2e})")
    print(f"  improvement (base−aug val-NLL) = {s['improvement_val_nll']:+.6e}  "
          f"= {s['improvement_over_sd']:+.2f}σ")
    print(f"  paired Δ per seed: mean {s['paired_delta_val']['mean']:+.6e}  "
          f"sd {s['paired_delta_val']['sd']:.2e}")
    print(f"\n  test-NLL mean: baseline {s['test_nll']['baseline_mean']:.6f}  "
          f"augmented {s['test_nll']['augmented_mean']:.6f}")
    print("\n  AUC (val, mean over seeds — illustrative; NLL decides):")
    print(f"    {'transition':>26}  {'baseline':>10}  {'augmented':>10}")
    for name in TRANS:
        a = s["auc_val_mean"][name]
        print(f"    {name:>26}  {f(a['baseline'])}  {f(a['augmented'])}")
    print(f"\n  DECISION [{s['decision'].upper()}]: {s['verdict']}")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="M26a Markov-assumption probe (history-feature "
                                             "augmented net vs current-state-only net).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # CPU per the M26a brief ("runs on the Mac, CPU"); train.py's loss/eval loop uses
    # float64 accumulation, which MPS does not support (CUDA is the only GPU path).
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--smoke", action="store_true",
                    help="wiring check: ≤100k rows, 3 epochs (standing rule 3)")
    args = ap.parse_args()
    run(args.variant, args.k, args.seeds, args.device, args.smoke)


if __name__ == "__main__":
    main()
