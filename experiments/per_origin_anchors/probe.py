#!/usr/bin/env python
"""THROWAWAY read-only diagnostic. NO fitting, NO geo-drop, NO integration.

Measures per-origin (current, dpd_30, dpd_60, dpd_90plus) out-of-sample test NLL on the
frozen k2015 eval slice (6,180,351 rows) for the two FULL-SCALE, GEO-INCLUSIVE anchors:
  (a) the deployed k2015 logit  (models/logit/full/k2015/ : torch LogitNet + scaler + vocab)
  (b) the banked   k2015 GBT    (models/gbt/full/k2015/   : LightGBM Booster + vocab + layout)

Neither metrics.json carries per-origin NLL and neither run saved per-row predictions, so we
RELOAD each banked model and predict on the frozen eval pool (no refit, test unweighted),
gating on reproducing the known overalls (logit 0.108246, GBT 0.103597) to ~1e-4.

This turns the spline probe's inferred claim -- "the `current` residual to GBT is shape, the
delinquent residual is interaction" -- into a measured per-origin logit-minus-GBT gap, printed
next to the probe's per-origin plain-minus-spline.

CAVEAT (printed): these logit/GBT anchors are full-scale & geo-inclusive; the spline probe's
plain/spline are geo-dropped on a 1.5M subsample. Geography adds ~nothing here (the probe's
plain reproduced the deployed logit to 1.1e-4), so the side-by-side is approximate-but-fair.

Run:  PYTHONPATH=src .venv/bin/python experiments/per_origin_anchors/probe.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

# Uncapped — full cores (full predict, full cores). MUST precede numpy/polars/lgb/torch import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_v] = str(os.cpu_count() or 1)

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))  # package not pip-installed in this venv

from floan.model import config            # noqa: E402  (read-only)
from floan.model import data as D          # noqa: E402  (read-only: window_spec, RAW_COLS)
from floan.model import features as F      # noqa: E402  (read-only: transforms/target)
from floan.model import gbt as G           # noqa: E402  (read-only: load_split/build_X)
from floan.model import backtest as B      # noqa: E402  (read-only: LogitEmbNet — the committed logit)
import lightgbm as lgb                      # noqa: E402
import torch                                # noqa: E402

N_THREADS = os.cpu_count() or 1             # full cores for LightGBM predict (uncapped)
# torch SINGLE-THREADED on purpose: LightGBM's in-process OpenMP predict leaves libomp in a state
# that deadlocks torch's first parallel op (torch.cat → kmp_flag_64::wait). The logit forward is
# light, so 1 thread is plenty and sidesteps the conflict entirely.
torch.set_num_threads(1)

VARIANT = "full"
K = 2015
EXPECTED_TEST_ROWS = 6_180_351
ANCHOR = {"logit": 0.108246, "gbt": 0.103597}
ORIGIN_STATES = list(config.ORIGIN_STATES)          # current, dpd_30, dpd_60, dpd_90plus
ORIG_IDX = {o: i for i, o in enumerate(ORIGIN_STATES)}
TOL_OVERALL = 1e-4                                   # reproduce known overall NLL to this
LOGIT_BATCH = 131_072                                # sub-batch for the 1506-wide one-hot
# The spline probe's measured per-origin plain-minus-spline (experiments/spline_logit_probe).
PROBE_PLAIN_MINUS_SPLINE = {"current": 0.001653, "dpd_30": -0.000759,
                            "dpd_60": -0.003119, "dpd_90plus": -0.015607}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _per_origin_nll(proba: np.ndarray, y: np.ndarray, origin_idx: np.ndarray) -> dict:
    """Overall + per-origin mean -log p(realized). origin_idx is the class index (0..3 for the
    four origin states; never >3 on the eval slice)."""
    assert np.isfinite(proba).all(), "non-finite probabilities"
    sum_err = float(np.abs(proba.sum(axis=1) - 1.0).max())
    nll = -np.log(np.clip(proba[np.arange(y.shape[0]), y], 1e-300, 1.0))
    per = {}
    for o, oi in ORIG_IDX.items():
        m = origin_idx == oi
        per[o] = {"n": int(m.sum()), "nll": float(nll[m].mean()) if m.any() else None}
    return {"overall_nll": float(nll.mean()), "n": int(y.shape[0]),
            "max_prob_sum_err": sum_err, "per_origin": per}


# ---------------------------------------------------------------------------- GBT
def eval_gbt() -> dict:
    cache = HERE / "gbt_cache.json"
    if cache.exists():                                   # cheap re-run insurance (frozen model)
        res = json.loads(cache.read_text())
        log(f"GBT: loaded cached per-origin result (overall NLL={res['overall_nll']:.6f}, "
            f"n={res['n']:,}) — skipping the ~20-min re-predict")
        return res
    gdir = config.MODELS / "gbt" / VARIANT / f"k{K}"
    log(f"GBT: loading booster + vocab from {gdir.name}/ ...")
    booster = lgb.Booster(model_file=str(gdir / "model.txt"))
    gvocab = F.Vocab.from_dict(json.loads((gdir / "vocab.json").read_text()))
    # Stream the frozen eval slice per part (small resident set avoids compressed-memory thrash
    # and gives progress); gbt.build_X reproduces the exact training design [raw cont || vocab
    # codes || binary] using the booster's own (train-fit) vocab, and _origin_index the per-row
    # origin class. booster.predict over all saved trees == best_iteration.
    log(f"GBT: {booster.num_trees()} trees; streaming per part (num_threads={N_THREADS})...")
    pool_dir, (lo, hi) = D.window_spec(VARIANT, K, "test")
    parts = sorted(pool_dir.glob("part-*.parquet"))
    nll_sum, n_total, sum_err = 0.0, 0, 0.0
    po = {o: [0.0, 0] for o in ORIGIN_STATES}
    for pi, part in enumerate(parts):
        df = F.prepare_raw(pl.scan_parquet(str(part))
                           .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
                           .select(D.RAW_COLS).collect())
        if df.height == 0:
            continue
        X = G.build_X(df, gvocab)                        # [N,44] raw cont || vocab codes || binary
        y = F.target_indices(df).astype(np.int64)
        origin_idx = G._origin_index(df)                 # class index 0..6 (origins are 0..3)
        del df
        proba = booster.predict(X, num_threads=N_THREADS)
        sum_err = max(sum_err, float(np.abs(proba.sum(axis=1) - 1.0).max()))
        assert np.isfinite(proba).all(), f"GBT non-finite proba (part {pi})"
        nll = -np.log(np.clip(proba[np.arange(y.shape[0]), y], 1e-300, 1.0))
        nll_sum += float(nll.sum())
        for o, oi in ORIG_IDX.items():
            m = origin_idx == oi
            po[o][0] += float(nll[m].sum()); po[o][1] += int(m.sum())
        n_total += y.shape[0]
        del X, y, origin_idx, proba, nll
        gc.collect()
        log(f"  GBT eval part {pi + 1}/{len(parts)}: cum rows {n_total:,}")
    assert n_total == EXPECTED_TEST_ROWS, f"GBT test n={n_total} != {EXPECTED_TEST_ROWS}"
    res = {"overall_nll": nll_sum / n_total, "n": n_total, "max_prob_sum_err": sum_err,
           "per_origin": {o: {"n": po[o][1], "nll": po[o][0] / po[o][1] if po[o][1] else None}
                          for o in ORIGIN_STATES}}
    assert res["max_prob_sum_err"] < 1e-6, f"GBT probs sum err {res['max_prob_sum_err']}"
    assert abs(res["overall_nll"] - ANCHOR["gbt"]) < TOL_OVERALL, \
        f"GBT overall {res['overall_nll']:.6f} != anchor {ANCHOR['gbt']} (|Δ|>{TOL_OVERALL})"
    log(f"GBT: overall NLL={res['overall_nll']:.6f} (anchor {ANCHOR['gbt']}, "
        f"|Δ|={abs(res['overall_nll']-ANCHOR['gbt']):.2e}); probs sum err {res['max_prob_sum_err']:.1e}")
    (HERE / "gbt_cache.json").write_text(json.dumps(res, indent=2))   # so a re-run skips the predict
    return res


# ---------------------------------------------------------------------------- logit
def eval_logit() -> dict:
    ldir = config.MODELS / "logit" / VARIANT / f"k{K}"
    log(f"logit: loading LogitEmbNet + scaler/vocab from {ldir.name}/ ...")
    scaler, vocab = F.load_pipeline(ldir)               # FULL geo-inclusive vocab (11 cats)
    # The committed full-scale logit is the EMBEDDING logit (backtest.LogitEmbNet: dense[cont‖bin]
    # + one Embedding(vocab_c, 7) per categorical, summed), NOT a one-hot LogitNet. Load + score it
    # exactly as evaluate.py does so we reproduce the 0.108246 anchor.
    model = B.LogitEmbNet(len(scaler.cols), len(F.BINARY), list(vocab.vocab_sizes))
    model.load_state_dict(torch.load(ldir / "model.pt", map_location="cpu"))
    model.eval()
    log(f"logit: LogitEmbNet({len(scaler.cols)} cont + {len(F.BINARY)} bin + "
        f"{len(vocab.cols)} cat-emb); streaming per part...")

    pool_dir, (lo, hi) = D.window_spec(VARIANT, K, "test")   # eval_pool, test bounds
    parts = sorted(pool_dir.glob("part-*.parquet"))
    nll_sum, n_total, sum_err = 0.0, 0, 0.0
    po = {o: [0.0, 0] for o in ORIGIN_STATES}            # [sum_nll, n]
    for pi, part in enumerate(parts):
        df = F.prepare_raw(pl.scan_parquet(str(part))
                           .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
                           .select(D.RAW_COLS).collect())
        if df.height == 0:
            continue
        cont = scaler.transform(df).astype(np.float32)   # [N,23] standardized
        cat = vocab.transform(df)                        # [N,11] integer vocab indices (UNK-routed)
        binb = F.binary_matrix(df).astype(np.float32)    # [N,10]
        y = F.target_indices(df)
        origin = df.get_column("state").to_numpy()
        del df
        n = y.shape[0]
        for s in range(0, n, LOGIT_BATCH):               # batch for memory; emb logit is light
            sl = slice(s, min(s + LOGIT_BATCH, n))
            with torch.no_grad():
                logits = model(torch.from_numpy(cont[sl]),
                               torch.from_numpy(cat[sl]).long(),
                               torch.from_numpy(binb[sl]))
                proba = torch.softmax(logits.double(), dim=1).numpy()   # eval-path float64 softmax
            sum_err = max(sum_err, float(np.abs(proba.sum(axis=1) - 1.0).max()))
            assert np.isfinite(proba).all(), f"logit non-finite proba (part {pi})"
            yb = y[sl]
            nll = -np.log(np.clip(proba[np.arange(yb.shape[0]), yb], 1e-300, 1.0))
            nll_sum += float(nll.sum())
            ob = origin[sl]
            for o in ORIGIN_STATES:
                m = ob == o
                po[o][0] += float(nll[m].sum()); po[o][1] += int(m.sum())
            del logits, proba, nll
        n_total += n
        del cont, cat, binb, y, origin
        gc.collect()
        log(f"  logit eval part {pi + 1}/{len(parts)}: cum rows {n_total:,}")

    assert n_total == EXPECTED_TEST_ROWS, f"logit test n={n_total} != {EXPECTED_TEST_ROWS}"
    res = {"overall_nll": nll_sum / n_total, "n": n_total, "max_prob_sum_err": sum_err,
           "per_origin": {o: {"n": po[o][1], "nll": po[o][0] / po[o][1] if po[o][1] else None}
                          for o in ORIGIN_STATES}}
    assert res["max_prob_sum_err"] < 1e-6, f"logit probs sum err {res['max_prob_sum_err']}"
    assert abs(res["overall_nll"] - ANCHOR["logit"]) < TOL_OVERALL, \
        f"logit overall {res['overall_nll']:.6f} != anchor {ANCHOR['logit']} (|Δ|>{TOL_OVERALL})"
    log(f"logit: overall NLL={res['overall_nll']:.6f} (anchor {ANCHOR['logit']}, "
        f"|Δ|={abs(res['overall_nll']-ANCHOR['logit']):.2e}); probs sum err {res['max_prob_sum_err']:.1e}")
    return res


# ---------------------------------------------------------------------------- report
def main() -> None:
    t0 = time.time()
    config.require_drive()
    log(f"per-origin anchors  variant={VARIANT}  k={K}  (read-only; reload + predict, no refit)")
    gbt = eval_gbt()
    logit = eval_logit()

    # n must agree across the two evaluations (same frozen slice).
    for o in ORIGIN_STATES:
        assert gbt["per_origin"][o]["n"] == logit["per_origin"][o]["n"], \
            f"per-origin n mismatch at {o}"

    rows = []
    for o in ORIGIN_STATES:
        n = logit["per_origin"][o]["n"]
        lg = logit["per_origin"][o]["nll"]
        gb = gbt["per_origin"][o]["nll"]
        rows.append((o, n, lg, gb, lg - gb, PROBE_PLAIN_MINUS_SPLINE[o]))

    print("\n" + "=" * 96)
    print("PER-ORIGIN OUT-OF-SAMPLE TEST NLL — frozen k2015 eval slice (6,180,351 rows, unweighted)")
    print("Full-scale, geo-inclusive banked anchors: logit (deployed) and GBT (banked).")
    print("=" * 96)
    hdr = f"{'origin':<12}{'n':>11}{'logit NLL':>13}{'GBT NLL':>13}{'logit−GBT':>13}{'probe plain−spline':>20}"
    print(hdr)
    print("-" * 96)
    for o, n, lg, gb, gap, probe in rows:
        print(f"{o:<12}{n:>11,}{lg:>13.6f}{gb:>13.6f}{gap:>+13.6f}{probe:>+20.6f}")
    print("-" * 96)
    print(f"{'OVERALL':<12}{logit['n']:>11,}{logit['overall_nll']:>13.6f}"
          f"{gbt['overall_nll']:>13.6f}{logit['overall_nll']-gbt['overall_nll']:>+13.6f}"
          f"{0.001442:>+20.6f}")
    print("=" * 96)
    print("Read: logit−GBT is the per-origin gap the GBT actually closes. The probe's")
    print("plain−spline is how much additive splines close. Where logit−GBT is large but")
    print("plain−spline is ~0/negative, the GBT's edge there is INTERACTION (splines don't reach it).")
    print()
    print("CAVEAT: logit/GBT anchors are full-scale & geo-inclusive; the probe's plain/spline are")
    print("geo-dropped on a 1.5M subsample. Geography adds ~nothing (probe plain reproduced the")
    print("deployed logit to 1.1e-4), so the side-by-side is approximate-but-fair, not exact.")
    print("=" * 96)

    out = {
        "diagnostic": "per-origin test NLL of the banked full-scale geo-inclusive logit & GBT (k2015)",
        "throwaway": True, "read_only": True, "variant": VARIANT, "window_k": K,
        "test_rows": logit["n"], "expected_rows": EXPECTED_TEST_ROWS,
        "method": "reloaded banked models + predicted on frozen eval pool (no refit, no geo-drop, "
                  "test unweighted); per-row predictions were not saved",
        "overall": {"logit": logit["overall_nll"], "gbt": gbt["overall_nll"],
                    "logit_minus_gbt": logit["overall_nll"] - gbt["overall_nll"],
                    "anchors": ANCHOR,
                    "reproduction_abs": {"logit": abs(logit["overall_nll"] - ANCHOR["logit"]),
                                         "gbt": abs(gbt["overall_nll"] - ANCHOR["gbt"])}},
        "per_origin": {o: {"n": n, "logit_nll": lg, "gbt_nll": gb,
                           "logit_minus_gbt": gap, "probe_plain_minus_spline": probe}
                       for o, n, lg, gb, gap, probe in rows},
        "prob_sum_err": {"logit": logit["max_prob_sum_err"], "gbt": gbt["max_prob_sum_err"]},
        "caveat": "logit/GBT anchors are full-scale & geo-inclusive; probe plain/spline are "
                  "geo-dropped on a 1.5M subsample. Geography adds ~nothing (probe plain "
                  "reproduced the deployed logit to 1.1e-4) -> approximate-but-fair.",
        "wall_sec": round(time.time() - t0, 1),
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {HERE / 'results.json'}  | wall {out['wall_sec']}s")


if __name__ == "__main__":
    main()
