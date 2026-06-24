"""M26 step-2b — three-arm spike + go/no-go (``04 §M26``).

Apples-to-apples value-of-memory test at k=2015. ONE uniform train_pool sample (+ HT
weights) is shared by all three arms; all train through the SAME loop (Adam, ReduceLROnPlateau,
early-stop on a fixed val subsample) and are scored on the SAME FULL frozen val/test slices via
``evaluate._nll`` / ``evaluate._auc_one_vs_rest`` — so the only difference is architecture:

  * ``ff_base`` — FF net, current-state-only (M26a baseline);
  * ``ff_hist`` — FF net + engineered history summaries (M26a augmented);
  * ``gru``     — 1-layer GRU over the trailing-T sequence (LEARNED memory).

The two findings (both in pooled-seed-sd units):
  * GRU − ff_base : does a sequence architecture beat plain current-state conditioning?
  * GRU − ff_hist : does LEARNED memory beat ENGINEERED one-step memory?

``--calibrate`` times one seed (capped epochs) and projects the full 3-seed wall-time (so the
Mac-vs-pod call is made on measured numbers, never by shrinking the sample). ``--run`` does the
real thing. CPU per the project (train.py's float64 loss path has no MPS).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import time

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import evaluate as EV
from floan.model import features as F
from floan.model import history as H
from floan.model import net as N
from floan.model import seq_model as SM
from floan.model import sequence as S
from floan.model import torch_common as tc

ARMS = ("ff_base", "ff_hist", "gru")
ARM_LABELS = {"ff_base": "FF baseline", "ff_hist": "FF + history", "gru": "GRU (seq)"}
TRANS = {"current->prepaid": ("current", "prepaid"),
         "dpd_90plus->foreclosure": ("dpd_90plus", "foreclosure"),
         "current->dpd_30": ("current", "dpd_30")}

NN = config.NN_SELECTED              # depth/dropout/weight_decay (k=2015 full-scale argmin)
HIDDEN, T = 48, S.SEQ_LEN
FF_BATCH, GRU_BATCH = 4096, 512
LR, MAX_EPOCHS, PATIENCE = 1e-3, 40, 5
SAMPLE_SEED = 777                    # the fixed training-sample draw (seeds vary init/order only)


# ---------------------------------------------------------------------------
# Shared data: points + features
# ---------------------------------------------------------------------------
def _scan_pool(variant, k, split) -> pl.LazyFrame:
    pool = "train_pool" if split == "train" else "eval_pool"
    lo, hi = {"train": config.train_bounds, "val": config.val_bounds,
              "test": config.test_bounds}[split](k)
    return (pl.scan_parquet(str(config.TRAINING_DIR / variant / pool / "part-*.parquet"))
              .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
              .filter(pl.col("state").is_in(list(config.ORIGIN_STATES)))
              .filter(pl.col("state_next").is_not_null()))


def sample_train(variant, k, n) -> pl.DataFrame:
    df = _scan_pool(variant, k, "train").collect()
    if n is not None and df.height > n:
        df = df[np.sort(np.random.default_rng(SAMPLE_SEED).choice(df.height, n, replace=False))]
    return df


def _subsample(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    if df.height <= n:
        return df
    return df[np.sort(np.random.default_rng(seed).choice(df.height, n, replace=False))]


def _origin(df: pl.DataFrame) -> np.ndarray:
    return (df.select(pl.col("state").replace_strict(
        list(EV.OI), list(EV.OI.values()), default=-1, return_dtype=pl.Int64))
        .to_numpy().reshape(-1))


def _to_points(df: pl.DataFrame) -> pl.DataFrame:
    return (df.select("Loan Identifier", "period_ym", "state", "state_next", "weight")
              .with_columns(
                  (pl.col("period_ym") // 100 * 12 + pl.col("period_ym") % 100 - 1).alias("t_mi"),
                  pl.col("state_next").replace_strict(
                      list(F.STATE_INDEX), list(F.STATE_INDEX.values()),
                      default=-1, return_dtype=pl.Int64).alias("target_idx"))
              .rename({"Loan Identifier": "loan", "period_ym": "t_ym"})
              .with_row_index("point_id"))


def ff_encode(variant, k, df, arm, scaler=None, vocab=None):
    """M26a feature pipeline on pool rows: ff_base = current-state-only; ff_hist = + history."""
    fr = F.prepare_raw(df)
    if arm == "ff_hist":
        fr = H.join_history(fr, variant, k)
        cont_cols, cat_cols, bcols = H.AUG_CONTINUOUS, H.AUG_CATEGORICAL, F.BINARY + H.HIST_BINARY
    else:
        cont_cols, cat_cols, bcols = F.CONTINUOUS, F.CATEGORICAL, F.BINARY
    if scaler is None:
        scaler = F.Scaler.fit(fr, cols=cont_cols)
        vocab = F.Vocab.fit(fr, cols=cat_cols)
    enc = F.encode_frame(fr, scaler, vocab, encode="index", binary_cols=bcols)
    return enc, scaler, vocab


# ---------------------------------------------------------------------------
# Batch iterators + probability scoring (arch-agnostic via predict)
# ---------------------------------------------------------------------------
def _ff_batches(enc, bs, shuffle, seed=0, epoch=0):
    n = enc["y"].shape[0]
    order = (np.random.default_rng([seed, epoch]).permutation(n) if shuffle else np.arange(n))
    for s in range(0, n, bs):
        idx = order[s:s + bs]
        yield {kk: enc[kk][idx] for kk in ("cont", "cat", "bin", "y", "w")}


@torch.no_grad()
def score_probs(model, predict, batches, device) -> np.ndarray:
    model.eval()
    out = [torch.softmax(predict(model, b, device).double(), dim=1).float().cpu().numpy()
           for b in batches]
    return np.concatenate(out)


def gru_probs_full(model, predict, points, variant, k, scaler, vocab, device,
                   chunk=200_000) -> np.ndarray:
    """Score the GRU on a full slice WITHOUT materializing all sequences: build_split per
    chunk, forward, concatenate (memory-bounded — full val is ~3M sequences)."""
    out = []
    for s in range(0, points.height, chunk):
        pts = points[s:s + chunk].drop("point_id").with_row_index("point_id")
        arr, _, _ = S.build_split(variant, k, "val", pts, T, scaler, vocab)
        out.append(score_probs(model, predict, S.iter_batches(arr, 4096, shuffle=False), device))
    return np.concatenate(out)


def _transition_aucs(probs, y, origin) -> dict:
    res = {}
    for name, (o, d) in TRANS.items():
        m = origin == EV.OI[o]
        res[name] = (EV._auc_one_vs_rest(probs[m][:, EV.SI[d]], (y[m] == EV.SI[d]))
                     if m.any() else None)
    return res


# ---------------------------------------------------------------------------
# Unified training loop — identical across arms
# ---------------------------------------------------------------------------
def train_arm(model, predict, train_iter, val_eval, *, device, seed, max_epochs):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=NN["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(max_epochs):
        model.train()
        t0 = time.perf_counter()
        for b in train_iter(ep, seed):
            y, w = tc._as_long(b["y"], device), tc._as_f32(b["w"], device)
            opt.zero_grad(set_to_none=True)
            loss = tc.weighted_ce(predict(model, b, device), y, w)
            loss.backward()
            opt.step()
        tr_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        vnll = val_eval(model)
        sched.step(vnll)
        if vnll < best - 1e-7:
            best, best_state, bad = vnll, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        hist.append({"epoch": ep, "val_nll": vnll, "train_sec": tr_sec,
                     "val_sec": time.perf_counter() - t1})
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return best, hist


# ---------------------------------------------------------------------------
# Per-arm data + closures
# ---------------------------------------------------------------------------
def build_arm(arm, variant, k, train_df, vsub_df, val_df, test_df, val_points, test_points):
    """Returns (build_model, predict, train_iter, val_eval, score_full) for one arm; data
    (seed-independent) is built once here."""
    if arm in ("ff_base", "ff_hist"):
        tr, scaler, vocab = ff_encode(variant, k, train_df, arm)
        vsub, _, _ = ff_encode(variant, k, vsub_df, arm, scaler, vocab)
        vfull, _, _ = ff_encode(variant, k, val_df, arm, scaler, vocab)
        tfull, _, _ = ff_encode(variant, k, test_df, arm, scaler, vocab)
        predict = N.make_nn_predict()
        nb = len(F.BINARY) + len(H.HIST_BINARY) if arm == "ff_hist" else None

        def build_model():
            return N.build(scaler, vocab, depth=NN["depth"], dropout=NN["dropout"], n_binary=nb)

        def train_iter(ep, seed):
            return _ff_batches(tr, FF_BATCH, True, seed, ep)

        def val_eval(m):
            return EV._nll(score_probs(m, predict, _ff_batches(vsub, 16384, False), DEVICE), vsub["y"])

        def score_full(m):
            return (score_probs(m, predict, _ff_batches(vfull, 16384, False), DEVICE),
                    score_probs(m, predict, _ff_batches(tfull, 16384, False), DEVICE))
    else:
        tr_pts, vsub_pts = _to_points(train_df), _to_points(vsub_df)
        tr_arr, scaler, vocab = S.build_split(variant, k, "train", tr_pts, T)
        vsub_arr, _, _ = S.build_split(variant, k, "val", vsub_pts, T, scaler, vocab)
        predict = SM.make_seq_predict()

        def build_model():
            return SM.build(scaler, vocab, hidden=HIDDEN)

        def train_iter(ep, seed):
            return S.iter_batches(tr_arr, GRU_BATCH, True, seed, ep)

        def val_eval(m):
            return EV._nll(score_probs(m, predict, S.iter_batches(vsub_arr, 4096, False), DEVICE),
                           vsub_arr.y)

        def score_full(m):
            return (gru_probs_full(m, predict, val_points, variant, k, scaler, vocab, DEVICE),
                    gru_probs_full(m, predict, test_points, variant, k, scaler, vocab, DEVICE))

    return build_model, predict, train_iter, val_eval, score_full


def _sd(xs):
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


DEVICE = torch.device("cpu")


def run(variant="dev", k=config.TUNING_YEAR, seeds=(0, 1, 2), train_n=200_000,
        val_sub_n=150_000, max_epochs=MAX_EPOCHS, calibrate=False) -> dict:
    global DEVICE
    config.require_drive()
    t_start = time.perf_counter()
    print(f"=== M26 step-2b 3-arm spike  variant={variant} k={k}  seeds={list(seeds)}  "
          f"train_n={train_n}  device={DEVICE}{'  [CALIBRATE]' if calibrate else ''} ===")

    train_df = sample_train(variant, k, train_n)
    val_df = _scan_pool(variant, k, "val").collect()
    test_df = _scan_pool(variant, k, "test").collect()
    vsub_df = _subsample(val_df, val_sub_n, seed=SAMPLE_SEED)
    val_points, test_points = _to_points(val_df), _to_points(test_df)
    val_y, test_y = val_points["target_idx"].to_numpy(), test_points["target_idx"].to_numpy()
    val_origin = _origin(val_df)
    print(f"sample: train={train_df.height:,} (uniform+HT weights)  "
          f"val(full)={val_df.height:,}  test(full)={test_df.height:,}  "
          f"val_sub(early-stop)={vsub_df.height:,}")

    results = {a: [] for a in ARMS}
    timing = {}
    for arm in ARMS:
        t_arm = time.perf_counter()
        bm, predict, train_iter, val_eval, score_full = build_arm(
            arm, variant, k, train_df, vsub_df, val_df, test_df, val_points, test_points)
        build_sec = time.perf_counter() - t_arm
        for seed in seeds:
            tc.set_seed(seed)
            model = bm().to(DEVICE)
            best, hist = train_arm(model, predict, train_iter, val_eval,
                                   device=DEVICE, seed=seed, max_epochs=max_epochs)
            t_sc = time.perf_counter()
            vprobs, tprobs = score_full(model)
            sc_sec = time.perf_counter() - t_sc
            rec = {"seed": seed, "val_nll": EV._nll(vprobs, val_y),
                   "test_nll": EV._nll(tprobs, test_y),
                   "auc": _transition_aucs(vprobs, val_y, val_origin),
                   "epochs": len(hist), "val_sub_nll": best,
                   "epoch_train_sec": _mean(h["train_sec"] for h in hist),
                   "epoch_val_sec": _mean(h["val_sec"] for h in hist), "score_full_sec": sc_sec}
            results[arm].append(rec)
            print(f"  [{arm:>7}] seed {seed}: val_nll={rec['val_nll']:.6f}  "
                  f"test_nll={rec['test_nll']:.6f}  ({rec['epochs']} ep, "
                  f"{rec['epoch_train_sec']:.0f}s/ep train, score_full {sc_sec:.0f}s)")
        timing[arm] = {"build_sec": build_sec,
                       "epoch_train_sec": _mean(r["epoch_train_sec"] for r in results[arm]),
                       "score_full_sec": _mean(r["score_full_sec"] for r in results[arm]),
                       "epochs": _mean(r["epochs"] for r in results[arm])}

    summary = _summarize(results, timing, variant, k, list(seeds), train_df.height,
                         val_df.height, test_df.height, time.perf_counter() - t_start, calibrate,
                         max_epochs)
    out = config.MODELS / "nn" / variant / (
        "seq_spike_summary" + ("_calib" if calibrate else "") + ".json")
    out.write_text(json.dumps(summary, indent=2, default=str))
    _report(summary)
    if calibrate:
        _project(timing, list(seeds))
    print(f"\nwrote {out}  [{summary['wall_min']:.1f} min]")
    return summary


def _summarize(results, timing, variant, k, seeds, n_tr, n_val, n_test, wall, calib, max_ep):
    agg = {}
    for arm in ARMS:
        rs = results[arm]
        agg[arm] = {
            "val_nll_mean": _mean(r["val_nll"] for r in rs), "val_nll_sd": _sd(r["val_nll"] for r in rs),
            "test_nll_mean": _mean(r["test_nll"] for r in rs), "test_nll_sd": _sd(r["test_nll"] for r in rs),
            "auc": {t: _mean(r["auc"][t] for r in rs if r["auc"][t] is not None) or None
                    for t in TRANS}, "per_seed": rs}
    deltas = {}
    for a, b in (("gru", "ff_base"), ("gru", "ff_hist")):
        d = agg[a]["val_nll_mean"] - agg[b]["val_nll_mean"]
        pooled = ((agg[a]["val_nll_sd"] ** 2 + agg[b]["val_nll_sd"] ** 2) / 2) ** 0.5
        deltas[f"{a}_minus_{b}"] = {"delta_val_nll": d, "pooled_seed_sd": pooled,
                                    "sigma": (d / pooled) if pooled else float("nan")}
    return {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task": "M26-2b", "variant": variant, "k": k, "seeds": seeds,
            "calibrate": calib, "max_epochs": max_ep,
            "n_train": n_tr, "n_val_full": n_val, "n_test_full": n_test,
            "m26a_banked_full": {"ff_base": 0.096190, "ff_hist": 0.089308},
            "arms": agg, "deltas": deltas, "timing": timing, "wall_min": wall / 60.0}


def _report(s):
    print("\n" + "=" * 78)
    print(f"M26 step-2b — three-arm spike (k={s['k']}, {s['variant']}, {len(s['seeds'])} seeds, "
          f"train {s['n_train']:,}, val {s['n_val_full']:,})"
          + ("   [CALIBRATION — not converged]" if s["calibrate"] else ""))
    print("=" * 78)
    print(f"  {'arm':>13} | {'val-NLL (mean±sd)':>22} | {'test-NLL (mean±sd)':>22}")
    for arm in ARMS:
        a = s["arms"][arm]
        print(f"  {ARM_LABELS[arm]:>13} | {a['val_nll_mean']:.6f} ± {a['val_nll_sd']:.2e}"
              f"   | {a['test_nll_mean']:.6f} ± {a['test_nll_sd']:.2e}")
    print(f"  {'M26a full FF':>13} | base {s['m26a_banked_full']['ff_base']:.6f} / "
          f"hist {s['m26a_banked_full']['ff_hist']:.6f}  (sanity: subsample arms ≈ these, slightly worse)")
    print("\n  AUC by transition (val, mean over seeds):")
    print(f"    {'transition':>26} | " + " | ".join(f"{ARM_LABELS[a]:>13}" for a in ARMS))
    for t in TRANS:
        cells = " | ".join(("      —      " if s["arms"][a]["auc"][t] is None
                            else f"{s['arms'][a]['auc'][t]:>13.4f}") for a in ARMS)
        print(f"    {t:>26} | {cells}")
    print("\n  Findings (val-NLL Δ in pooled-seed-sd σ):")
    for key, lbl in (("gru_minus_ff_base", "GRU − FF baseline  (seq arch vs current-state)"),
                     ("gru_minus_ff_hist", "GRU − FF + history (LEARNED vs ENGINEERED memory)")):
        d = s["deltas"][key]
        print(f"    {lbl:<48}: Δ={d['delta_val_nll']:+.6f}  "
              f"= {d['sigma']:+.2f}σ  (pooled sd {d['pooled_seed_sd']:.2e})")
    print("=" * 78)


def _project(timing, seeds):
    """Project the full-converged 3-seed wall-time from calibration per-epoch + scoring times."""
    print("\n--- wall-time projection (assumes ~25 converged epochs/arm) ---")
    EST_EPOCHS = 25
    total = 0.0
    for arm in ARMS:
        t = timing[arm]
        per_seed = EST_EPOCHS * (t["epoch_train_sec"]) + t["score_full_sec"]
        arm_total = t["build_sec"] + len(seeds) * per_seed
        total += arm_total
        print(f"  {ARM_LABELS[arm]:>13}: build {t['build_sec']:.0f}s + "
              f"{len(seeds)}×(~{EST_EPOCHS}ep×{t['epoch_train_sec']:.0f}s + "
              f"score {t['score_full_sec']:.0f}s) ≈ {arm_total/60:.0f} min")
    print(f"  PROJECTED FULL 3-SEED RUN ≈ {total/60:.0f} min ({total/3600:.1f} h) on this Mac (CPU)")


def main():
    ap = argparse.ArgumentParser(description="M26 step-2b three-arm spike.")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-n", type=int, default=200_000)
    ap.add_argument("--val-sub-n", type=int, default=150_000)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--calibrate", action="store_true",
                    help="1 seed, capped epochs, measure + project (no full run)")
    args = ap.parse_args()
    if args.calibrate:
        run(args.variant, args.k, seeds=args.seeds[:1], train_n=args.train_n,
            val_sub_n=args.val_sub_n, max_epochs=args.max_epochs, calibrate=True)
    else:
        run(args.variant, args.k, seeds=args.seeds, train_n=args.train_n,
            val_sub_n=args.val_sub_n, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
