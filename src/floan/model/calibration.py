"""M23 — Per-horizon calibration decision + calibrator fit (ECONOMIC_ENGINE §6 / 06 §4).

This module holds the **calibrate-vs-not decision** and the **fitted calibrators** that conform to
the ``pool.Calibrator`` seam (``ECONOMIC_ENGINE §3.2``). The engine
(:func:`pool.roll_forward_predictor`) applies the calibrator to the **raw per-origin scores BEFORE
assembly** (§4 item 2) at every roll-forward step, so a per-window calibrator is per-horizon by
construction; nothing in ``pool.py`` changes.

**Why H=1 anchors the decision (ECONOMIC_ENGINE §5).** ``H = 1`` is "the clean calibration test":
the composed one-step distribution equals the direct model prediction, so the realised outcome is
the existing per-row test label ``y`` — no multi-horizon panel scan. It is "the most direct read of
whether the level is right". The per-horizon *compounding* question (does the gap widen with H) is
the **reported result that rides M24's full grid** — here we make the DECISION on a ~20% subsample
and leave the per-H calibrated-vs-raw price-error table to M24 ("subsample the fit, never the
reported numbers"). ``evaluate.py`` is left untouched — this module reuses its loaders/scorers and
does its own split-aware, subsampled scoring (bounded-CPU; the metric path stays pristine).

Calibrators (both conform to ``pool.Calibrator`` — ``__call__(probs, origin) -> (n, 7)``):
  * :class:`TemperatureCalibrator` — ONE scalar ``T`` per window (``06 §4``), the primary method;
    ``T == 1`` is a genuine bit-identity (short-circuit).
  * :class:`IsotonicCalibrator` — per-origin-block, per-class isotonic + row renormalise; the
    §4 fallback when temperature leaves the reliability curve off-diagonal.

Run (the decision evidence — CPU, self-contained on the Mac):
    .venv/bin/python -m floan.model.calibration --k 2020 --device cpu --frac 0.2
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from floan.model import backtest as B
from floan.model import config
from floan.model import data as D
from floan.model import ensemble as Ens
from floan.model import evaluate as E
from floan.model import features as F
from floan.model import torch_common as tc
from floan.model import train as T

VARIANT = "full"
ORIGIN_STATES = list(config.ORIGIN_STATES)             # the 4 transient origins, block order 0..3
OI = {s: i for i, s in enumerate(ORIGIN_STATES)}       # origin state -> block index
SI = F.STATE_INDEX                                     # state -> class index (0..6)
N_CLASSES = F.N_CLASSES
_EPS = 1e-12

# The diagnostic transitions where the nonlinearity lives (ECONOMIC_ENGINE §7 / 06 §6):
# (origin block index, destination class index, label). These drive the reliability read.
DIAG_TRANSITIONS = [
    (OI["current"],    SI["prepaid"],     "current->prepaid"),
    (OI["current"],    SI["dpd_30"],      "current->dpd_30"),
    (OI["dpd_90plus"], SI["foreclosure"], "dpd_90plus->foreclosure"),
]

# Decision thresholds (documented in M23_NOTES). Temperature is "applied" only if it is materially
# away from 1 AND earns a material improvement on the HELD-OUT test subsample; otherwise the raw
# level is on-diagonal and calibration is a no-op identity.
T_TOL = 0.05            # |T - 1| <= 5% -> indistinguishable from identity
NLL_REL_TOL = 1e-3      # < 0.1% relative multiclass-NLL gain -> immaterial
ECE_REL_TOL = 0.10      # < 10% relative mean-ECE reduction -> immaterial


# ===========================================================================
# Temperature scaling — one scalar per window (the primary method)
# ===========================================================================
def _temper(probs: np.ndarray, T: float) -> np.ndarray:
    """``softmax(log(p) / T)`` — temperature scaling expressed on probabilities. It equals
    ``softmax(z / T)`` on the original logits ``z`` (since ``log p = z - logsumexp(z)`` and the
    constant cancels in the softmax), so the single temperature is recoverable from probabilities
    alone — no logits needed. The clip guards exact zeros (Laplace-smoothed empirical cells,
    terminal-state rows)."""
    z = np.log(np.clip(probs, _EPS, None)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureCalibrator:
    """One scalar ``T`` per window (``ECONOMIC_ENGINE §6`` / ``06 §4``), applied to every origin
    block — conforms to ``pool.Calibrator`` (``__call__(probs, origin) -> (n, 7)``).

    ``T == 1`` is a genuine **bit-identity** via the short-circuit: ``softmax(log p / 1)`` is not
    bit-exact (the log/exp round-trip perturbs the last ULPs), so the disabled path must return the
    input array unchanged — the M23 no-op regression check."""

    def __init__(self, T: float):
        self.T = float(T)

    def __call__(self, probs: np.ndarray, origin: int) -> np.ndarray:
        if self.T == 1.0:                                  # genuine bit-identity (the no-op path)
            return probs
        return _temper(probs, self.T)


def fit_temperature(probs: np.ndarray, y: np.ndarray, *, bounds=(0.05, 20.0)) -> dict:
    """Fit the single temperature minimising multiclass NLL on the (val) slice — a 1-D bounded
    Brent search (the NLL is well-behaved and unimodal in ``log T``). Returns
    ``{"T", "nll_raw", "nll_cal"}`` (the raw/calibrated val NLL bracketing the fit)."""
    y = np.asarray(y).astype(np.int64)
    idx = np.arange(probs.shape[0])
    logp = np.log(np.clip(probs, _EPS, None))

    def nll(t: float) -> float:
        z = logp / t
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e[idx, y] / e.sum(axis=1)
        return float(-np.log(np.clip(p, _EPS, None)).mean())

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    T_star = float(res.x)
    return {"T": T_star, "nll_raw": nll(1.0), "nll_cal": nll(T_star)}


# ===========================================================================
# Per-class isotonic — the §4 fallback (used only if temperature is insufficient)
# ===========================================================================
class IsotonicCalibrator:
    """Per-origin-block, per-class isotonic regression + row renormalise (``06 §4`` fallback when
    temperature leaves the reliability curve off-diagonal). Conforms to ``pool.Calibrator``.
    ``maps[origin][c]`` is a fitted ``IsotonicRegression`` (or ``None`` to pass that column
    through). Renormalising preserves a valid distribution; rows with no fitted column are
    returned unchanged."""

    def __init__(self, maps: dict[int, list]):
        self.maps = maps

    def __call__(self, probs: np.ndarray, origin: int) -> np.ndarray:
        ms = self.maps.get(origin)
        if ms is None:
            return probs
        q = probs.astype(np.float64, copy=True)
        for c, m in enumerate(ms):
            if m is not None:
                q[:, c] = m.predict(probs[:, c])
        s = q.sum(axis=1, keepdims=True)
        return np.where(s > _EPS, q / np.clip(s, _EPS, None), probs)


def fit_isotonic(probs: np.ndarray, y: np.ndarray, origin_idx: np.ndarray,
                 *, min_n: int = 200) -> dict[int, list]:
    """Fit per-origin-block per-class isotonic maps on the (val) slice. A class is fit only where
    it has at least one positive and is non-degenerate within the origin block; others pass
    through. ``origin_idx`` is the per-row origin block (0..3)."""
    y = np.asarray(y).astype(np.int64)
    maps: dict[int, list] = {}
    for o in range(len(ORIGIN_STATES)):
        m = origin_idx == o
        if m.sum() < min_n:
            continue
        po, yo = probs[m], y[m]
        cols: list = [None] * N_CLASSES
        for c in range(N_CLASSES):
            pos = (yo == c).astype(np.float64)
            if 0 < pos.sum() < m.sum():                    # non-degenerate column
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                ir.fit(po[:, c], pos)
                cols[c] = ir
        maps[o] = cols
    return maps


# ===========================================================================
# Reliability / ECE — the scalar summary of evaluate._reliability's curve
# ===========================================================================
def ece(pred: np.ndarray, label: np.ndarray, nbins: int = 12, min_bin: int = 20):
    """Quantile-binned expected calibration error of a 1-D probability ``pred`` against the binary
    ``label``: ``Σ_bin (n_bin/N) |E[pred] − E[label]|`` over quantile bins with ``>= min_bin`` rows
    — the scalar reduction of ``evaluate._reliability``'s diagram. ``None`` if too few rows."""
    if pred.shape[0] < nbins * 5:
        return None
    qs = np.unique(np.quantile(pred, np.linspace(0, 1, nbins + 1)))
    if qs.shape[0] < 3:
        return None
    idx = np.clip(np.searchsorted(qs[1:-1], pred), 0, len(qs) - 2)
    num, den = 0.0, 0
    for b in range(len(qs) - 1):
        mb = idx == b
        nb = int(mb.sum())
        if nb >= min_bin:
            num += nb * abs(float(pred[mb].mean()) - float(label[mb].mean()))
            den += nb
    return float(num / den) if den else None


def _binary_nll(p: np.ndarray, pos: np.ndarray) -> float:
    """One-vs-rest binary NLL of column probability ``p`` against indicator ``pos``."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-(pos * np.log(p) + (1 - pos) * np.log(1 - p)).mean())


# ===========================================================================
# Scoring a (subsampled) split — reuse evaluate's loaders; evaluate.py untouched
# ===========================================================================
def _score_slice(k: int, device, *, split: str, frac: float, seed: int, models) -> dict:
    """Raw per-row probabilities for ``models`` on window-``k``'s ``split`` slice, subsampled to
    ~``frac`` of rows by a deterministic ``(loan, period_ym)`` hash (loan-month-representative and
    reproducible, so val and test draws are stable). The hash filter pushes into the lazy scan, so
    only ~``frac`` of the slice is ever materialised — the bounded-CPU path. Reuses
    ``evaluate``'s loaders/scorers; ``evaluate.py`` is not modified.

    Returns ``{"probs": {model: (n, 7)}, "y", "origin", "n"}``."""
    nn_run = config.MODELS / "nn" / VARIANT / B._nn_tag(k, dict(config.NN_SELECTED))
    logit_run = config.MODELS / "logit" / VARIANT / f"k{k}"
    scaler, vocab = F.load_pipeline(nn_run)

    pool_dir, bounds = D.window_spec(VARIANT, k, split)
    lf = D._masked_scan(pool_dir, bounds)
    if frac < 1.0:
        m = max(1, int(round(1.0 / frac)))                 # keep ~1/m of loan-months
        lf = lf.filter((pl.struct(["Loan Identifier", "period_ym"]).hash(seed=seed) % m) == 0)
    df = F.prepare_raw(lf.collect())
    enc = F.encode_frame(df, scaler, vocab, encode="index")
    y = np.asarray(enc["y"]).astype(np.int64)
    origin = (df.select(pl.col("state").replace_strict(
        list(OI), list(OI.values()), default=-1, return_dtype=pl.Int64))
        .to_numpy().reshape(-1)).astype(np.int64)
    n = int(y.shape[0])

    probs: dict[str, np.ndarray] = {}
    Tres = T.to_resident(enc, device)
    del enc
    if "empirical" in models:
        probs["empirical"] = E.empirical_matrix(k)[origin].astype(np.float32)
    if "logit" in models:
        probs["logit"] = E._resident_probs(
            E._load_logit(logit_run, scaler, vocab, device), Tres, device)
    if "nn" in models:
        probs["nn"] = E._resident_probs(E._load_nn(nn_run, device), Tres, device)
    if "ensemble" in models and k in E.KEY_WINDOWS:
        acc = np.zeros((n, N_CLASSES), dtype=np.float64)
        for s in range(Ens.N_MEMBERS):
            mrun = config.MODELS / "nn" / VARIANT / f"ens_k{k}_s{s}"
            acc += E._resident_probs(E._load_nn(mrun, device), Tres, device)
        probs["ensemble"] = (acc / Ens.N_MEMBERS).astype(np.float32)
    del Tres
    return {"probs": probs, "y": y, "origin": origin, "n": n}


# ===========================================================================
# The decision — fit on VAL, assess raw-vs-calibrated on a held-out TEST subsample (H=1)
# ===========================================================================
def _decide_one(T_star: float, nll_raw: float, nll_cal: float,
                mean_ece_raw, mean_ece_cal) -> dict:
    """Calibrate-vs-not for one model: apply only if ``T`` is materially ≠ 1 AND the held-out
    subsample shows a material NLL or mean-ECE improvement (else the raw level is on-diagonal)."""
    rel_nll = (nll_raw - nll_cal) / nll_raw if nll_raw else 0.0
    rel_ece = ((mean_ece_raw - mean_ece_cal) / mean_ece_raw
               if (mean_ece_raw and mean_ece_cal is not None) else 0.0)
    t_material = abs(T_star - 1.0) > T_TOL
    applied = bool(t_material and (rel_nll > NLL_REL_TOL or rel_ece > ECE_REL_TOL))
    reason = ("T≈1 (on-diagonal)" if not t_material
              else f"materially helps (rel_nll={rel_nll:+.2%}, rel_ece={rel_ece:+.2%})" if applied
              else f"T≠1 but immaterial (rel_nll={rel_nll:+.2%}, rel_ece={rel_ece:+.2%})")
    return {"applied": applied, "rel_nll_gain": rel_nll, "rel_ece_gain": rel_ece, "reason": reason}


def decide(k: int, device, *, frac: float = 0.2, seed: int = 0,
           models=("logit", "nn", "ensemble")) -> dict:
    """Fit per-window temperature on the **VAL** slice, then assess raw-vs-calibrated reliability
    (per-transition ECE + binary NLL) and multiclass NLL on a HELD-OUT **TEST** subsample at H=1
    (the clean calibration test, §5), per model. Returns the per-model decision + evidence."""
    fit = _score_slice(k, device, split="val", frac=frac, seed=seed, models=models)
    asn = _score_slice(k, device, split="test", frac=frac, seed=seed, models=models)

    out: dict[str, dict] = {}
    for m in models:
        if m not in fit["probs"]:
            continue
        ft = fit_temperature(fit["probs"][m], fit["y"])
        T_star = ft["T"]
        p_raw = asn["probs"][m]
        p_cal = p_raw if T_star == 1.0 else _temper(p_raw, T_star)
        nll_raw = E._nll(p_raw, asn["y"])
        nll_cal = E._nll(p_cal, asn["y"])

        trans = []
        for (o, d, lbl) in DIAG_TRANSITIONS:
            mo = asn["origin"] == o
            if int(mo.sum()) < 100:
                trans.append({"transition": lbl, "n": int(mo.sum()), "note": "too few rows"})
                continue
            pos = (asn["y"][mo] == d).astype(np.float64)
            trans.append({
                "transition": lbl, "n": int(mo.sum()), "base_rate": float(pos.mean()),
                "ece_raw": ece(p_raw[mo][:, d], pos), "ece_cal": ece(p_cal[mo][:, d], pos),
                "bnll_raw": _binary_nll(p_raw[mo][:, d], pos),
                "bnll_cal": _binary_nll(p_cal[mo][:, d], pos)})

        ece_raws = [t["ece_raw"] for t in trans if t.get("ece_raw") is not None]
        ece_cals = [t["ece_cal"] for t in trans if t.get("ece_cal") is not None]
        mean_ece_raw = float(np.mean(ece_raws)) if ece_raws else None
        mean_ece_cal = float(np.mean(ece_cals)) if ece_cals else None
        dec = _decide_one(T_star, nll_raw, nll_cal, mean_ece_raw, mean_ece_cal)

        out[m] = {"T": T_star, "nll_val_raw": ft["nll_raw"], "nll_val_cal": ft["nll_cal"],
                  "nll_test_raw": nll_raw, "nll_test_cal": nll_cal,
                  "mean_ece_raw": mean_ece_raw, "mean_ece_cal": mean_ece_cal,
                  "transitions": trans, **dec}
    return {"k": k, "horizon": 1, "frac": frac, "seed": seed,
            "n_val": fit["n"], "n_test": asn["n"], "models": out}


# ===========================================================================
# Driver
# ===========================================================================
def _print_decision(res: dict) -> None:
    print(f"\n=== M23 calibration decision  k={res['k']}  H=1 (clean test, §5)  "
          f"frac={res['frac']}  n_val={res['n_val']:,}  n_test={res['n_test']:,} ===")
    for m, r in res["models"].items():
        print(f"\n  [{m}]  T*={r['T']:.4f}  "
              f"val-NLL {r['nll_val_raw']:.6f}->{r['nll_val_cal']:.6f}  "
              f"test-NLL {r['nll_test_raw']:.6f}->{r['nll_test_cal']:.6f}")
        print(f"    {'transition':<26}{'n':>10}{'base':>10}{'ECE_raw':>11}"
              f"{'ECE_cal':>11}{'bNLL_raw':>11}{'bNLL_cal':>11}")
        for t in r["transitions"]:
            if "note" in t:
                print(f"    {t['transition']:<26}{t['n']:>10}   ({t['note']})")
                continue
            def _f(x, w=11, p=5):
                return (f"{x:.{p}f}".rjust(w)) if x is not None else "n/a".rjust(w)
            print(f"    {t['transition']:<26}{t['n']:>10}{t['base_rate']:>10.4f}"
                  f"{_f(t['ece_raw'])}{_f(t['ece_cal'])}{_f(t['bnll_raw'])}{_f(t['bnll_cal'])}")
        print(f"    DECISION: {'APPLIED' if r['applied'] else 'SKIPPED'} — {r['reason']}")
    any_applied = any(r["applied"] for r in res["models"].values())
    print(f"\n  window verdict: calibration {'APPLIED' if any_applied else 'SKIPPED'} "
          f"(config.CALIBRATION['applied'] = {any_applied}); per-window temperatures refit at "
          f"full scale in M24 if applied (subsample the fit, never the reported numbers).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="M23 — per-horizon calibration decision + calibrator fit (ECONOMIC_ENGINE §6).")
    ap.add_argument("--k", type=int, default=2020, help="window (anchor = Dec(k-1)); 2020 = COVID")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--frac", type=float, default=0.2, help="subsample fraction (~0.2 = 20%)")
    ap.add_argument("--seed", type=int, default=0, help="subsample hash seed")
    ap.add_argument("--models", nargs="+", default=["logit", "nn", "ensemble"])
    args = ap.parse_args()
    config.require_drive()
    device = tc.resolve_device(args.device)

    t0 = time.perf_counter()
    res = decide(args.k, device, frac=args.frac, seed=args.seed, models=tuple(args.models))
    res["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    res["git_commit"] = T._git_commit()
    res["device"] = str(device)
    res["wall_sec"] = time.perf_counter() - t0
    _print_decision(res)

    out_dir = config.MODELS / "nn" / VARIANT
    out = out_dir / f"calibration_m23_k{args.k}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {out}  [{res['wall_sec']:.0f}s]")


if __name__ == "__main__":
    main()
