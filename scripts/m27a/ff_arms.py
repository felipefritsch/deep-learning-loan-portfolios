"""M27a matched baselines — ff_base / ff_hist rolling on all 11 windows, SAME train points
as the GRU caches (prediction_points seed 0), full-val early-stop, full-test scoring on cuda.
Reuses seq_spike (ff_encode/train_arm/score_probs/_ff_batches), net, history, evaluate. Resumable.

Train sample size is the SEQ_TRAIN_N knob (default 1_500_000, byte-identical to the banked build),
read from the same env var build_window.py uses so the FF arms stay matched to the GRU caches at the
W1 scale-up S (prediction_points caps at the per-window pool, so full-pool runs match automatically)."""
from __future__ import annotations

import json
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from floan.model import config as mc
mc.DUCKDB_MEMORY_LIMIT = "48GB"

from floan.model import evaluate as EV
from floan.model import features as F
from floan.model import history as H
from floan.model import net as N
from floan.model import sequence as S
from floan.model import seq_spike as SP
from floan.model import torch_common as tc

VARIANT = "full"
WINDOWS = list(range(2015, 2026))
ARMS = ("ff_base", "ff_hist")
# Match the GRU caches' per-window sample S (build_window.py reads the same env var). Default
# 1.5M keeps the banked build byte-identical; set SEQ_TRAIN_N to scale all three arms together.
TRAIN_N = int(os.environ.get("SEQ_TRAIN_N", "1500000"))
FF_BATCH, SCORE_BATCH = 4096, 16384
MAX_EPOCHS = 40
RESULTS = mc.OUTPUTS / "m27a_gpu_runs"
DEV = torch.device("cuda")
SP.DEVICE = DEV   # seq_spike closures/score default to this


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


def train_sample(k) -> pl.DataFrame:
    """The EXACT 1.5M points the GRU cache used: prediction_points(seed 0) semi-joined to the
    full pool (so we get all feature columns for those same loan-months). No order assumptions."""
    pts = S.prediction_points(VARIANT, k, "train", sample_n=TRAIN_N, seed=0)
    pool = SP._scan_pool(VARIANT, k, "train").collect()
    keys = pts.select(pl.col("loan").alias("Loan Identifier"), pl.col("t_ym").alias("period_ym"))
    df = pool.join(keys, on=["Loan Identifier", "period_ym"], how="semi")
    # Match the exact rows prediction_points returned (which is min(TRAIN_N, pool) — the cap bites
    # at full pool), NOT the requested TRAIN_N, so the semi-join contract holds at any S.
    assert df.height == pts.height, f"train sample {df.height} != {pts.height} (key join broke)"
    return df


def run_arm(k, arm):
    npz = RESULTS / f"k{k}_{arm}.npz"; js = RESULTS / f"k{k}_{arm}.json"
    if npz.exists() and js.exists():
        print(f"[k{k} {arm}] SKIP (exists)", flush=True)
        return
    tc.set_seed(0); torch.cuda.reset_peak_memory_stats(DEV)
    t0 = time.perf_counter()
    tr_df = train_sample(k)
    val_df = SP._scan_pool(VARIANT, k, "val").collect()
    test_df = SP._scan_pool(VARIANT, k, "test").collect()
    val_y = SP._to_points(val_df)["target_idx"].to_numpy()
    test_y = SP._to_points(test_df)["target_idx"].to_numpy()
    test_org = SP._origin(test_df)

    enc_tr, scaler, vocab = SP.ff_encode(VARIANT, k, tr_df, arm)
    enc_val, _, _ = SP.ff_encode(VARIANT, k, val_df, arm, scaler, vocab)
    enc_test, _, _ = SP.ff_encode(VARIANT, k, test_df, arm, scaler, vocab)
    nb = len(F.BINARY) + len(H.HIST_BINARY) if arm == "ff_hist" else None
    predict = N.make_nn_predict()
    load_s = time.perf_counter() - t0
    print(f"[k{k} {arm}] data: train {enc_tr['y'].shape[0]:,} val {val_y.shape[0]:,} "
          f"test {test_y.shape[0]:,}  ({load_s:.0f}s)", flush=True)

    model = N.build(scaler, vocab, depth=SP.NN["depth"], dropout=SP.NN["dropout"], n_binary=nb).to(DEV)

    def train_iter(ep, seed):
        return SP._ff_batches(enc_tr, FF_BATCH, True, seed, ep)

    def val_eval(m):
        return EV._nll(SP.score_probs(m, predict, SP._ff_batches(enc_val, SCORE_BATCH, False), DEV),
                       enc_val["y"])

    t_tr = time.perf_counter()
    best_vnll, hist = SP.train_arm(model, predict, train_iter, val_eval,
                                   device=DEV, seed=0, max_epochs=MAX_EPOCHS)
    train_s = time.perf_counter() - t_tr

    t_sc = time.perf_counter()
    te_probs = SP.score_probs(model, predict, SP._ff_batches(enc_test, SCORE_BATCH, False), DEV)
    test_nll = EV._nll(te_probs, test_y)
    test_auc = per_transition_auc(te_probs, test_y, test_org)
    score_s = time.perf_counter() - t_sc
    peak_gb = torch.cuda.max_memory_allocated(DEV) / 1e9

    np.savez(npz, probs=te_probs.astype(np.float32), y=test_y.astype(np.int64),
             origin=test_org.astype(np.int64))
    js.write_text(json.dumps({
        "k": k, "arm": arm, "n_test": int(test_y.shape[0]),
        "test_nll": test_nll, "val_nll": best_vnll, "epochs": len(hist),
        "test_auc": test_auc, "load_s": load_s, "train_s": train_s, "score_s": score_s,
        "peak_gpu_gb": peak_gb,
    }, indent=2))
    print(f"[k{k} {arm}] DONE test_nll {test_nll:.6f} val_nll {best_vnll:.6f} "
          f"({len(hist)} ep)  load {load_s:.0f}s train {train_s:.0f}s score {score_s:.0f}s", flush=True)
    del enc_tr, enc_val, enc_test, te_probs, tr_df, val_df, test_df; gc.collect()
    torch.cuda.empty_cache()


def main():
    runs = [(k, a) for k in WINDOWS for a in ARMS]
    print(f"=== M27a FF arms: {len(runs)} (window,arm) runs on {DEV} ===", flush=True)
    for i, (k, arm) in enumerate(runs):
        print(f"\n--- [{i+1}/{len(runs)}] k{k} {arm} ---", flush=True)
        run_arm(k, arm)
    print("\n=== ALL FF RUNS COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
