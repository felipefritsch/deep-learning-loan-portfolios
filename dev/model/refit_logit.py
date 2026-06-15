"""M15a (logit recompute) — memory-safe **streaming** re-fit of the frozen full-scale logit.

The logit/full checkpoints (k=2015/2019/2020/2023/2025) were never synced off the GPU pod
(only NN + ensemble were), so the §5 ensemble-vs-logit headline has no logit to score. This
script recomputes them on the Mac. The multinomial logit is **convex** (and the embedding-sum
parameterisation + zero-init + L2 reproduce the one-hot fit), so the recomputed model reproduces
M14's logit up to optimiser noise — verified below.

Why not ``backtest.fit_logit``: it holds the whole train split resident (the GPU fast path), and
the full-scale slices (36–63 M rows) overflow this 16 GB Mac. Here we instead

  * **reuse the saved scaler/vocab** from the NN run folder — byte-identical to logit's (they were
    fit on the same train slice; ``evaluate.score_window`` asserts the match), so no featurisation
    divergence and no memory-heavy ``D.fit_window``;
  * **stream the train slice** shard-by-shard via ``data.WindowLoader`` (bounded memory), with the
    *identical* model / loss / hyperparameters as ``backtest.fit_logit``;
  * preload only the small val slice (year k−1) for early stopping.

Self-gating: fit **k=2015 first** and validate its 12-month pool counts against the committed
``pools_random_k2015.parquet`` ``logit_*_pred`` columns to ~1e-3 (the M14 anchor predictions). Only
if that passes do we fit the other four windows and **merge** logit's SMM path into
``smm_paths_k*.parquet`` (``smm_paths`` merge mode). If k=2015 misses the gate, STOP and report —
the operator falls back to a GPU-pod re-fit (the standard ``backtest.fit_logit``).

Run (detached background job — no session held open):
    nohup caffeinate -i .venv/bin/python -u dev/model/refit_logit.py --device cpu \
        > "/Volumes/SSD Felipe/dissertation/logs/m15/refit_logit.log" 2>&1 &
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
import time

import numpy as np
import polars as pl
import torch

import backtest as B
import config
import data as D
import features as F
import pool as PL
import pools as PP
import smm_paths as S
import torch_common as tc
import train as T

VARIANT = "full"
KEY_WINDOWS = [2015, 2019, 2020, 2023, 2025]      # k=2015 FIRST (the validation gate)
EVAL_BATCH = 32768
GATE_REL_RMSE = 1e-3                               # k=2015 prepaid pool-count rel-RMSE gate (~1e-3)


# ===========================================================================
# Streaming fit of one window's logit (identical model/loss/HPs to backtest.fit_logit)
# ===========================================================================
def _nn_run(k: int):
    return config.MODELS / "nn" / VARIANT / B._nn_tag(k, dict(config.NN_SELECTED))


def _train_epoch_streaming(model, loader, opt, device, epoch: int) -> float:
    """One reshuffled streaming pass over the train slice (shard order + in-shard shuffle keyed
    on ``(seed, epoch)``). Same weight-averaged-CE loss as ``train.train_one_epoch``; bounded
    memory (one shard resident at a time)."""
    model.train()
    swce = sw = 0.0
    for b in loader.iter_batches(epoch):
        cont = tc._as_f32(b["cont"], device)
        binb = tc._as_f32(b["bin"], device)
        cat = tc._as_long(b["cat"], device)
        y = tc._as_long(b["y"], device)
        w = tc._as_f32(b["w"], device)
        opt.zero_grad(set_to_none=True)
        loss = tc.weighted_ce(model(cont, cat, binb), y, w)
        loss.backward()
        opt.step()
        wsum = float(w.double().sum())
        swce += float(loss.detach().double()) * wsum
        sw += wsum
    return swce / sw if sw else float("nan")


def _fit_one_wd(loader, Tval, n_cont, n_bin, vocab_sizes, wd, device) -> dict:
    """Fit one ``LogitEmbNet`` at L2=``wd`` to its best streamed-train val NLL (early stopping)."""
    tc.set_seed(B.SEED)
    model = B.LogitEmbNet(n_cont, n_bin, vocab_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=B.LOGIT_LR, weight_decay=wd)
    best = {"val_nll": float("inf"), "state": None, "epoch": -1}
    bad = 0
    for epoch in range(B.LOGIT_MAX_EPOCHS):
        tr = _train_epoch_streaming(model, loader, opt, device, epoch)
        val = T.evaluate_nll_resident(model, Tval, EVAL_BATCH, device)["weighted_nll"]
        print(f"      wd={wd:<7g} epoch {epoch:2d}: train {tr:.6f}  val {val:.6f}", flush=True)
        if val < best["val_nll"] - 1e-7:
            best = {"val_nll": val, "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
            bad = 0
        else:
            bad += 1
            if bad >= B.LOGIT_PATIENCE:
                break
    model.load_state_dict(best["state"])
    return {"wd": wd, "val_nll": best["val_nll"], "epoch": best["epoch"], "model": model}


def fit_logit_streaming(k: int, device, *, fresh: bool = False):
    """Streaming re-fit of window ``k``'s logit over the L2 grid; idempotent (skips if a
    metrics.json already exists). Writes ``logit/full/k{k}/`` (model.pt + scaler/vocab + metrics)."""
    run = config.MODELS / "logit" / VARIANT / f"k{k}"
    if (run / "metrics.json").exists() and not fresh:
        print(f"[logit k={k}] already complete — skipping", flush=True)
        return run
    t0 = time.perf_counter()
    print(f"=== streaming logit re-fit  k={k}  ({VARIANT}) ===", flush=True)
    scaler, vocab = F.load_pipeline(_nn_run(k))        # byte-identical to M14 logit's pipeline
    n_cont, n_bin, vs = len(scaler.cols), len(F.BINARY), list(vocab.vocab_sizes)

    pool_dir, bounds = D.window_spec(VARIANT, k, "train")
    loader = D.WindowLoader(pool_dir, bounds, scaler, vocab, batch_size=B.LOGIT_BATCH,
                            encode="index", seed=B.SEED, shuffle=True)
    val_enc = T.preload_split(VARIANT, k, "val", scaler, vocab, None)
    Tval = T.to_resident(val_enc, device)
    n_val = int(val_enc["y"].shape[0])
    del val_enc
    print(f"  design: {n_cont} cont + Σ{sum(vs)} one-hot ({len(vs)} cats) + {n_bin} bin   "
          f"val rows={n_val:,}", flush=True)

    per_wd, best = {}, {"val_nll": float("inf")}
    for wd in B.LOGIT_WD_GRID:
        r = _fit_one_wd(loader, Tval, n_cont, n_bin, vs, wd, device)
        per_wd[str(wd)] = {"val_nll": r["val_nll"], "best_epoch": r["epoch"]}
        print(f"  wd={wd:<7g} → val_nll={r['val_nll']:.6f} (best@{r['epoch']})", flush=True)
        if r["val_nll"] < best["val_nll"]:
            best = r

    run.mkdir(parents=True, exist_ok=True)
    F.save_pipeline(run, scaler, vocab)
    torch.save(best["model"].state_dict(), run / "model.pt")
    metrics = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": "multinomial_logit", "parameterization": "embedding_sum_onehot",
        "variant": VARIANT, "window_k": k, "recomputed": True,
        "recompute_note": "logit/full never synced off the pod; streamed re-fit on Mac "
                          "(saved scaler/vocab; identical LogitEmbNet/loss/hyperparams)",
        "device": str(device), "seed": B.SEED, "git_commit": T._git_commit(),
        "features": {"n_continuous": n_cont, "n_binary": n_bin, "n_categoricals": len(vs),
                     "onehot_equiv_width": int(sum(vs))},
        "hyperparams": {"lr": B.LOGIT_LR, "batch_size": B.LOGIT_BATCH,
                        "max_epochs": B.LOGIT_MAX_EPOCHS, "patience": B.LOGIT_PATIENCE,
                        "wd_grid": list(B.LOGIT_WD_GRID)},
        "l2_selection": {"per_wd": per_wd, "selected_wd": best["wd"]},
        "logit": {"val_nll": best["val_nll"]},
        "wall_sec": time.perf_counter() - t0,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  selected wd={best['wd']:g}  val_nll={best['val_nll']:.6f}  "
          f"wrote {run/'model.pt'}  [{metrics['wall_sec']/60:.1f} min]", flush=True)
    return run


# ===========================================================================
# k=2015 validation gate — recomputed logit's 12m pool counts vs the committed M14 parquet
# ===========================================================================
def validate_k2015(device) -> dict:
    """Roll the recomputed k=2015 logit forward 12m, aggregate to the M14 random pools, and
    compare predicted prepaid / dpd60p counts to ``pools_random_k2015.parquet``'s
    ``logit_*_pred`` columns. PASS iff the prepaid per-pool relative RMSE ≤ ``GATE_REL_RMSE``."""
    print("\n=== validate k=2015 recomputed logit vs committed M14 pool counts ===", flush=True)
    tbl = S.per_loan_smm_table(2015, device, ["empirical", "logit"])
    df = tbl["df"]
    idx = PP.random_pool_index(df.height, 2015)
    ref = pl.read_parquet(config.OUTPUTS / "tables" / "pool_level" / "pools_random_k2015.parquet")

    out = {}
    for outcome in ("prepaid", "dpd60p"):
        pred = PP.pool_counts_random(df, idx, "logit", outcome)["pred"]
        refc = ref.get_column(f"logit_{outcome}_pred").to_numpy()
        resid = pred - refc
        rel_rmse = float(np.sqrt(np.mean(resid ** 2)) / np.mean(np.abs(refc)))
        out[outcome] = {"rel_rmse": rel_rmse, "max_abs": float(np.abs(resid).max()),
                        "mean_ref": float(np.abs(refc).mean()),
                        "max_rel": float((np.abs(resid) / np.maximum(np.abs(refc), 1e-9)).max())}
        print(f"  {outcome:>7}: rel-RMSE={rel_rmse:.2e}  max|Δ|={out[outcome]['max_abs']:.4f}  "
              f"(mean ref count {out[outcome]['mean_ref']:.2f})", flush=True)
    out["pass"] = out["prepaid"]["rel_rmse"] <= GATE_REL_RMSE
    print(f"  GATE ({'PASS' if out['pass'] else 'FAIL'}): prepaid rel-RMSE "
          f"{out['prepaid']['rel_rmse']:.2e} {'≤' if out['pass'] else '>'} {GATE_REL_RMSE:g}",
          flush=True)
    return out


# ===========================================================================
# Driver
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="M15a — streaming logit re-fit + k=2015 gate + merge.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--fresh", action="store_true", help="re-fit even if metrics.json exists")
    ap.add_argument("--no-merge", action="store_true", help="fit+validate only; skip SMM merge")
    args = ap.parse_args()
    config.require_drive()
    device = (torch.device("cuda") if args.device == "cuda" else tc.resolve_device(args.device))
    t0 = time.perf_counter()
    print(f"=== M15a logit recompute · windows={KEY_WINDOWS} · device={device} ===", flush=True)

    fit_logit_streaming(2015, device, fresh=args.fresh)
    gate = validate_k2015(device)
    (config.MODELS / "logit" / VARIANT / "refit_gate_k2015.json").write_text(
        json.dumps({"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "gate_rel_rmse": GATE_REL_RMSE, **gate}, indent=2, default=float))
    if not gate["pass"]:
        print("\nSTOP — k=2015 recomputed logit missed the ~1e-3 gate. NOT fitting the other "
              "windows and NOT merging. Fall back to a GPU-pod re-fit (backtest.fit_logit). "
              f"Detail in logit/full/refit_gate_k2015.json  [{(time.perf_counter()-t0)/60:.1f} min]",
              flush=True)
        sys.exit(2)

    for k in [w for w in KEY_WINDOWS if w != 2015]:
        fit_logit_streaming(k, device, fresh=args.fresh)

    if not args.no_merge:
        print("\n=== merging recomputed logit SMM path into smm_paths_k*.parquet ===", flush=True)
        S.run(KEY_WINDOWS, device, ["empirical", "logit"], merge=True)

    print(f"\nALL DONE — recomputed logit + merged (gate passed)  "
          f"[{(time.perf_counter()-t0)/60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
