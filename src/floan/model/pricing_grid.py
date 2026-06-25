"""M24 — the rolling pricing grid: the horizon × regime exhibit (``ECONOMIC_ENGINE §5,§9``;
``04_TASKS`` M24; ``03_POOL_LEVEL §2–5``).

Runs the M21–M23 economic engine across **all available anchors × H ∈ {1,3,6,12}** for the torch
models (logit / best-NN / 8-net ensemble), **raw** — M23 decided calibration is a no-op for the torch
path (T≈1, on-diagonal), so calibrated ≡ raw and this is a **single pass, no calibrated second
compose**. No GBT (M19 not built). Extends the M14/M15 exhibits with an H dimension:

* **T5.1** — CPR / WAL / price error by model × anchor × **H** (pool level); headline = the
  ensemble/NN-vs-logit mean |price-error| reduction by anchor × H.
* **T4.2** — pool count R² / RMSE by model × outcome (prepaid, 60+) × anchor × **H**.
* **F5.2** — signed price-error by characteristic bucket, per anchor × H (figures via
  :func:`economics.fig_price_error_buckets`, built in the table pass).
* **Loan level** — per-loan WAL / price distribution (M22 :func:`pool.price_loans_vec`, the full
  population vectorised) + the §4.3 loan→pool aggregation-identity reconciliation, by model × H.
* **COVID-inversion-in-dollars** (k2020) reproduced and shown to **deepen with H**.

**Horizon semantics** (approved): horizon H prices the monthly SMM path **truncated at month H**,
constant-extrapolating SMM(H) beyond (the cashflow engine already extrapolates a short vector; §4
item 1 / M15). H=12 reproduces M15 exactly — the regression guard. The four snapshots are read off a
**single roll to 12** (intermediates free): one ``capture_smm`` roll per anchor scores once and
composes the prepay chain + the 60+ first-passage chain in lockstep (``pool.roll_forward``), so the
per-H counts cost no extra scoring.

**Resumable + atomic per anchor** (the GBT-sweep pattern): each anchor's artifacts are written
``tmp → os.replace`` and sealed with a ``.done`` marker; a re-run **skips** completed anchors, so a
hiccup costs at most the in-progress anchor. The table builders assemble T4.2 / T5.1 / F5.2 + the
headline from **whatever anchors have finished** — they never require all 11, so Thursday's
deliverable can be assembled from the first (ensemble-bearing) anchors while the rest still run.

Run::

    python -m floan.model.pricing_grid --device cuda                  # full grid, all 11 anchors
    python -m floan.model.pricing_grid --device cuda --anchors 2020 2023 2025
    python -m floan.model.pricing_grid --smoke                        # k2020, CPU, capped (local)
    python -m floan.model.pricing_grid --only-tables                  # assemble from completed anchors
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

import torch

from floan.model import config
from floan.model import economics as EC
from floan.model import pool as PL
from floan.model import pools as PP
from floan.model import smm_paths as S
from floan.model import torch_common as tc

VARIANT = "full"
HORIZONS = tuple(config.HORIZONS)                  # (1, 3, 6, 12)
HMAX = PL.HORIZON                                  # 12
# Priority order: ensemble-bearing first (deadline-minimum k2020/k2023/k2025, then the full ensemble
# set k2019/k2015), then the 6 logit/NN-only windows. A partial run degrades gracefully.
ANCHORS_PRIORITY = [2020, 2023, 2025, 2019, 2015, 2024, 2022, 2021, 2018, 2017, 2016]
REQUESTED = ("empirical", "logit", "nn", "ensemble")   # ensemble auto-dropped on non-key windows
OUTCOMES = ("prepaid", "dpd60p")                       # predicted+realized both available per H
SCHEMES = ("char", "random")
PRICED_MODELS_EXCLUDE = ("empirical",)                 # empirical kept as a baseline column in tables
COVID_ANCHOR = 2020


# ===========================================================================
# Artifact layout + atomic IO
# ===========================================================================
def _m24dir() -> Path:
    d = config.OUTPUTS / "tables" / "pool_level" / "m24"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _done_marker(k: int) -> Path:
    return _m24dir() / f"anchor_k{k}.done"


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.write_parquet(tmp)
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ===========================================================================
# Per-anchor compute (one capture_smm roll → pricing + counts + loan level)
# ===========================================================================
def _within_h(step_indicator: np.ndarray, h: int) -> np.ndarray:
    """Realized 'ever-by-month-H' flag per loan from per-step indicators: cumulative-OR over the
    first ``h`` steps (the realized analogue of the predicted first-passage mass at H)."""
    return (step_indicator[:, :h].sum(axis=1) > 0).astype(np.float64)


def _agg_random(pred: np.ndarray, real: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pred[idx].sum(axis=1), real[idx].sum(axis=1)


def _agg_char(pred: np.ndarray, real: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(cell, kind="stable")
    cs = cell[order]
    _, start = np.unique(cs, return_index=True)
    return np.add.reduceat(pred[order], start), np.add.reduceat(real[order], start)


def _counts_grid(tbl: dict, k: int) -> pl.DataFrame:
    """Pool count R²/RMSE by scheme × outcome × model × H (T4.2 with the H axis). Predicted within-H
    = the first-passage cumulative mass at H (``cum_prepaid`` / ``cum_dpd60p`` column H−1); realized
    within-H = cumulative-OR of the realized per-step indicators. Uses the SAME char cells / random
    pools as the pricing aggregation (``pools.char_cell_ids`` / ``random_pool_index``)."""
    df = tbl["df"]
    cell = PP.char_cell_ids(df)[0]
    idx = PP.random_pool_index(df.height, k)
    real_steps = {"prepaid": tbl["prepay_real"], "dpd60p": tbl["dpd60_real"]}
    rows: list[dict] = []
    for m in tbl["model_names"]:
        pred_cum = {"prepaid": tbl["smm"][m]["cum"], "dpd60p": tbl["smm"][m]["cum_dpd60p"]}
        for outcome in OUTCOMES:
            for h in HORIZONS:
                pred = pred_cum[outcome][:, h - 1]
                real = _within_h(real_steps[outcome], h)
                for scheme, agg in (("char", _agg_char), ("random", _agg_random)):
                    pool_pred, pool_real = (agg(pred, real, cell) if scheme == "char"
                                            else agg(pred, real, idx))
                    r2, rmse = PP._r2_rmse(pool_pred, pool_real)
                    rows.append({"anchor": f"Dec{k-1}", "k": k, "scheme": scheme,
                                 "outcome": outcome, "model": m, "h": h,
                                 "r2": r2, "rmse": rmse, "n_pools": int(pool_pred.shape[0])})
    return pl.DataFrame(rows)


def _loan_grid(tbl: dict, k: int) -> pl.DataFrame:
    """Loan-level pricing distribution + §4.3 reconciliation by model × H. Prices the FULL population
    per model with :func:`pool.price_loans_vec` (vectorised; the M24 scale guard), truncating each
    loan's conditional SMM path at H. Reports per-loan price/WAL moments + quantiles, the pool
    aggregate, and the reconciliation gap vs the pool-level single-pool valuation (the §4.3 identity,
    approximate at population scale where WAC/WAM are UPB-weighted means — stated, not a wiring bug)."""
    wac, wam, upb = tbl["wac"], tbl["wam"], tbl["upb"]
    w = np.nan_to_num(upb, nan=0.0)
    upb_pool = float(w.sum())
    wac_pool = float((w * np.where(np.isfinite(wac), wac, 0.0)).sum() / w.sum()) if w.sum() else float("nan")
    wam_pool = float((w * np.where(np.isfinite(wam), wam, 0.0)).sum() / w.sum()) if w.sum() else float("nan")
    rows: list[dict] = []
    for m in tbl["model_names"]:
        smm_loans = PL.per_loan_smm(tbl["smm"][m]["cum"], tbl["smm"][m]["alive"])      # [N,12]
        # whole-population pool SMM path (UPB-weighted) for the reconciliation reference
        num = S._increments(tbl["smm"][m]["cum"]) * w[:, None]
        den = tbl["smm"][m]["alive"] * w[:, None]
        smm_pool = S._smm_from_sums(num.sum(axis=0), den.sum(axis=0))                  # [12]
        for h in HORIZONS:
            res = PL.price_loans_vec(wac, wam, upb, smm_loans[:, :h])
            price = res["price"][np.isfinite(res["price"])]
            wal = res["wal"][np.isfinite(res["wal"])]
            pool_ref = PL.cashflow_engine(wac_pool, wam_pool, upb_pool, smm_pool[:h])["price"]
            rows.append({"anchor": f"Dec{k-1}", "k": k, "model": m, "h": h,
                         "n_loans": int(res["n_loans"]), "n_priced": int(price.shape[0]),
                         "pool_price": res["pool_price"], "pool_wal": res["pool_wal"],
                         "pool_value": res["pool_value"], "pool_upb": res["pool_upb"],
                         "price_mean": float(price.mean()) if price.size else float("nan"),
                         "price_p05": float(np.percentile(price, 5)) if price.size else float("nan"),
                         "price_p50": float(np.percentile(price, 50)) if price.size else float("nan"),
                         "price_p95": float(np.percentile(price, 95)) if price.size else float("nan"),
                         "wal_mean": float(wal.mean()) if wal.size else float("nan"),
                         "recon_pool_ref_price": float(pool_ref),
                         "recon_gap": float(res["pool_price"] - pool_ref)})
    return pl.DataFrame(rows)


def run_anchor(k: int, device, *, max_loans: int | None = None, chunk: int = PL.DEFAULT_CHUNK,
               force: bool = False) -> dict:
    """Compute + atomically write one anchor's M24 artifacts (econ / counts / loan / summary), then
    seal the ``.done`` marker. Resumable: a finished anchor (marker present, ``force`` False) is
    skipped. One ``capture_smm`` roll feeds the pool SMM paths (pricing), the per-H counts, and the
    loan-level pricing — no second roll."""
    marker = _done_marker(k)
    smoke = max_loans is not None
    if marker.exists() and not force:
        print(f"=== M24 k{k}: already done ({marker.name}) — skipping ===", flush=True)
        return {"k": k, "skipped": True}

    t_start = time.time()
    print(f"=== M24 k{k} (device={device}{', SMOKE' if smoke else ''}) — rolling once to H={HMAX} ===",
          flush=True)
    tbl = S.per_loan_smm_table(k, device, REQUESTED, max_loans=max_loans, chunk=chunk)
    model_names = tbl["model_names"]
    print(f"  population n_pop={tbl['meta']['n_pop']:,} models={model_names}", flush=True)

    # --- pool SMM paths (model + realized) → pool-level pricing grid (T5.1 / F5.2 input) ---
    paths = S.pool_smm_paths(tbl, k)
    smm_frame = S.paths_to_frame(paths, k)
    econ = EC.per_pool_econ(k, smm_frame, horizons=HORIZONS)            # per (scheme,pool,model,h)
    counts = _counts_grid(tbl, k)                                      # T4.2 per-H
    loan = _loan_grid(tbl, k)                                          # loan-level + reconciliation

    sm = "_smoke" if smoke else ""
    _atomic_parquet(econ, _m24dir() / f"econ_k{k}{sm}.parquet")
    _atomic_parquet(counts, _m24dir() / f"counts_k{k}{sm}.parquet")
    _atomic_parquet(loan, _m24dir() / f"loan_k{k}{sm}.parquet")
    _atomic_parquet(smm_frame, _m24dir() / f"smm_paths_k{k}{sm}.parquet")

    wall = time.time() - t_start
    summary = {"k": k, "anchor": f"Dec{k-1}", "variant": VARIANT, "device": str(device),
               "smoke": smoke, "max_loans": max_loans, "horizons": list(HORIZONS),
               "model_names": list(model_names), "meta": tbl["meta"],
               "headline": _anchor_headline(econ), "wall_sec": round(wall, 1),
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    _atomic_text(json.dumps(summary, indent=2, default=float), _m24dir() / f"summary_k{k}{sm}.json")
    if not smoke:                                                      # smoke runs are not "complete" anchors
        _atomic_text("", marker)
    print(f"  wrote econ/counts/loan/smm + summary; wall={wall:.1f}s"
          f"{'' if smoke else ' — sealed .done'}", flush=True)
    return {"k": k, "skipped": False, "wall_sec": wall, "summary": summary}


def _anchor_headline(econ: pl.DataFrame) -> dict:
    """Ensemble/NN-vs-logit mean |price error| reduction by H for one anchor (char scheme), if those
    models are present. Used in the per-anchor summary; the cross-anchor table is built separately."""
    out: dict = {}
    sub = econ.filter(pl.col("scheme") == "char")
    have = set(sub.get_column("model").unique().to_list())
    if "logit" not in have:
        return out
    for h in HORIZONS:
        row = sub.filter(pl.col("h") == h)
        mae = {m: float(row.filter(pl.col("model") == m).get_column("price_err").abs().mean())
               for m in have}
        rec = {}
        base = mae.get("logit")
        for m in ("ensemble", "nn"):
            if m in mae and base and base > 0:
                rec[f"{m}_vs_logit_pct"] = round(100.0 * (base - mae[m]) / base, 2)
        out[str(h)] = {"mae_price": {m: round(v, 4) for m, v in mae.items()}, "reduction": rec}
    return out


# ===========================================================================
# Table assembly — from WHATEVER anchors have completed (never requires all 11)
# ===========================================================================
def _load_completed(kind: str, smoke: bool) -> pl.DataFrame | None:
    # Anchor windows are 4-digit years (k2015…k2025) — match `k20[0-9][0-9]` EXACTLY so the
    # full-run glob can never pick up a `*_smoke.parquet` (the `k*` wildcard used to: `econ_k*`
    # matched `econ_k2020_smoke`, concatenating the 50k local-smoke subsample into the real grid).
    pat = (f"{kind}_k20[0-9][0-9]_smoke.parquet" if smoke
           else f"{kind}_k20[0-9][0-9].parquet")
    parts = sorted(_m24dir().glob(pat))
    parts = [p for p in parts if not p.name.endswith(".tmp")]
    if not parts:
        return None
    return pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")


def _t51_agg(econ: pl.DataFrame) -> pl.DataFrame:
    """T5.1 aggregation (pure): mean |error| and signed bias by scheme × anchor × k × h × model.
    Extracted so the M27b GRU-merge can compute the GRU column with the identical reduction."""
    return (econ.group_by(["scheme", "anchor", "k", "h", "model"])
            .agg(pl.col("price_err").abs().mean().alias("price_mae"),
                 pl.col("price_err").mean().alias("price_bias"),
                 pl.col("cpr_err").abs().mean().alias("cpr_mae"),
                 pl.col("cpr_err").mean().alias("cpr_bias"),
                 pl.col("wal_err").abs().mean().alias("wal_mae"),
                 pl.col("wal_err").mean().alias("wal_bias"))
            .sort(["scheme", "k", "h", "model"]))


def build_T51(smoke: bool = False) -> dict:
    """T5.1 with the H axis: mean |error| and signed bias by scheme × metric × model × anchor × H,
    from every completed ``econ_k*`` artifact. Writes JSON + Markdown under the M24 dir."""
    econ = _load_completed("econ", smoke)
    if econ is None:
        print("T5.1: no completed anchors yet"); return {}
    agg = _t51_agg(econ)
    _atomic_parquet(agg, _m24dir() / f"t_m24_econ_errors{'_smoke' if smoke else ''}.parquet")
    out = {"anchors": sorted(econ.get_column("k").unique().to_list()),
           "horizons": list(HORIZONS), "rows": agg.to_dicts()}
    _atomic_text(json.dumps(out, indent=2, default=float),
                 _m24dir() / f"t_m24_econ_errors{'_smoke' if smoke else ''}.json")
    _atomic_text(_md_T51(agg), _m24dir() / f"t_m24_econ_errors{'_smoke' if smoke else ''}.md")
    return out


def build_T42(smoke: bool = False) -> dict:
    """T4.2 with the H axis: pool count R²/RMSE by scheme × outcome × model × anchor × H, from every
    completed ``counts_k*`` artifact."""
    counts = _load_completed("counts", smoke)
    if counts is None:
        print("T4.2: no completed anchors yet"); return {}
    counts = counts.sort(["scheme", "outcome", "k", "h", "model"])
    _atomic_parquet(counts, _m24dir() / f"t_m24_pool_counts{'_smoke' if smoke else ''}.parquet")
    out = {"anchors": sorted(counts.get_column("k").unique().to_list()),
           "horizons": list(HORIZONS), "rows": counts.to_dicts()}
    _atomic_text(json.dumps(out, indent=2, default=float),
                 _m24dir() / f"t_m24_pool_counts{'_smoke' if smoke else ''}.json")
    return out


def _horizon_regime_grid(econ: pl.DataFrame) -> list[dict]:
    """Per anchor × H (char scheme): mean |price error| per model, the UPB-weighted dollar price
    error Σ(price_err·upb)/100, and the ensemble/NN-vs-logit % reduction. Pure; extracted so the
    M27b GRU-merge reuses the identical per-cell computation (its extra models simply ride the same
    ``have`` set; the GRU-vs-logit reduction is added by the caller from ``mae_price``)."""
    sub = econ.filter(pl.col("scheme") == "char")
    grid: list[dict] = []
    for (k, h), g in sub.group_by(["k", "h"], maintain_order=True):
        have = set(g.get_column("model").unique().to_list())
        mae = {m: float(g.filter(pl.col("model") == m).get_column("price_err").abs().mean())
               for m in have}
        # dollar price error per model = Σ_pool price_err · upb / 100 (signed; magnitude deepens with H)
        dollars = {m: float((g.filter(pl.col("model") == m)
                             .select((pl.col("price_err") * pl.col("upb") / 100.0).sum()).item()))
                   for m in have}
        row = {"k": k, "anchor": f"Dec{k-1}", "h": h,
               "mae_price": {m: round(mae[m], 4) for m in have},
               "dollar_price_err": {m: dollars[m] for m in have}}
        base = mae.get("logit")
        if base and base > 0:
            for m in ("ensemble", "nn"):
                if m in mae:
                    row[f"{m}_vs_logit_pct"] = round(100.0 * (base - mae[m]) / base, 2)
        grid.append(row)
    grid.sort(key=lambda r: (r["k"], r["h"]))
    return grid


def build_horizon_regime(smoke: bool = False) -> dict:
    """The headline horizon × regime exhibit + the COVID-inversion-in-dollars. Per anchor × H (char
    scheme): mean |price error| per model, the ensemble/NN-vs-logit % reduction, and — in dollars —
    the UPB-weighted pool price error Σ(price_err·upb)/100, shown to deepen with H at k2020."""
    econ = _load_completed("econ", smoke)
    if econ is None:
        print("horizon×regime: no completed anchors yet"); return {}
    grid = _horizon_regime_grid(econ)
    covid = [r for r in grid if r["k"] == COVID_ANCHOR]
    out = {"anchors": sorted({r["k"] for r in grid}), "horizons": list(HORIZONS),
           "grid": grid, "covid_inversion_dollars": covid}
    _atomic_text(json.dumps(out, indent=2, default=float),
                 _m24dir() / f"t_m24_horizon_regime{'_smoke' if smoke else ''}.json")
    _atomic_text(_md_headline(grid), _m24dir() / f"t_m24_horizon_regime{'_smoke' if smoke else ''}.md")
    return out


def build_figures(smoke: bool = False) -> list[str]:
    """F5.2 price-error buckets per anchor × H (char scheme), via the M15 heatmap builder with an
    ``h`` suffix. Best-effort (skips if matplotlib/data unavailable)."""
    econ = _load_completed("econ", smoke)
    if econ is None:
        return []
    made: list[str] = []
    for (k,), gk in econ.group_by(["k"], maintain_order=True):
        for h in HORIZONS:
            gh = gk.filter((pl.col("scheme") == "char") & (pl.col("h") == h))
            if gh.height == 0:
                continue
            try:
                p = EC.fig_price_error_buckets(gh, int(k), suffix=f"_h{h:02d}"
                                               + ("_smoke" if smoke else ""))
                made.append(str(p))
            except Exception as e:                                   # figures are non-gating
                print(f"  F5.2 k{k} h{h}: skipped ({type(e).__name__}: {e})")
    return made


def build_tables(smoke: bool = False, figures: bool = False) -> dict:
    res = {"T51": build_T51(smoke), "T42": build_T42(smoke),
           "horizon_regime": build_horizon_regime(smoke)}
    if figures:
        res["figures"] = build_figures(smoke)
    print(f"=== M24 tables assembled from {len(res['T51'].get('anchors', []))} completed anchor(s) "
          f"→ {_m24dir()} ===", flush=True)
    return res


# ===========================================================================
# Markdown renderers
# ===========================================================================
def _md_T51(agg: pl.DataFrame) -> str:
    lines = ["# T5.1 — pool economic error by model × anchor × H (M24)", "",
             "Mean |price error| (per 100 face); char scheme. Lower is better.", ""]
    char = agg.filter(pl.col("scheme") == "char")
    models = [m for m in ("empirical", "logit", "nn", "ensemble")
              if m in char.get_column("model").unique().to_list()]
    lines.append("| anchor | H | " + " | ".join(models) + " |")
    lines.append("|" + "---|" * (len(models) + 2))
    for (k, h), g in char.group_by(["k", "h"], maintain_order=True):
        cells = []
        for m in models:
            v = g.filter(pl.col("model") == m).get_column("price_mae")
            cells.append(f"{v[0]:.3f}" if len(v) else "—")
        lines.append(f"| Dec{k-1} | {h} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _md_headline(grid: list[dict]) -> str:
    lines = ["# Horizon × regime — headline price-error reduction (M24)", "",
             "Mean |price error| (char, per 100); NN/ensemble vs logit % reduction; dollar price "
             "error Σ(err·UPB)/100.", "",
             "| anchor | H | logit | nn | ensemble | nn-vs-logit | ens-vs-logit | $err(ens) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in grid:
        m = r["mae_price"]
        d = r.get("dollar_price_err", {})
        lines.append(
            f"| {r['anchor']} | {r['h']} | "
            f"{m.get('logit', float('nan')):.3f} | {m.get('nn', float('nan')):.3f} | "
            f"{m.get('ensemble', float('nan')):.3f} | "
            f"{r.get('nn_vs_logit_pct', float('nan'))} | {r.get('ensemble_vs_logit_pct', float('nan'))} | "
            f"{d.get('ensemble', float('nan')):,.0f} |")
    return "\n".join(lines) + "\n"


# ===========================================================================
# Driver
# ===========================================================================
def run(anchors: list[int], device, *, max_loans: int | None = None, chunk: int = PL.DEFAULT_CHUNK,
        force: bool = False, figures: bool = False) -> dict:
    config.require_drive()
    results = []
    for k in anchors:
        try:
            results.append(run_anchor(k, device, max_loans=max_loans, chunk=chunk, force=force))
        except Exception as e:                          # one anchor's failure must not sink the rest
            print(f"!!! M24 k{k} FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({"k": k, "error": f"{type(e).__name__}: {e}"})
    tables = build_tables(smoke=max_loans is not None, figures=figures)
    return {"anchors": anchors, "results": results, "tables": tables}


def main() -> None:
    ap = argparse.ArgumentParser(description="M24 rolling pricing grid (horizon × regime).")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--anchors", type=int, nargs="*", default=None,
                    help="anchor windows (default: all 11 in priority order)")
    ap.add_argument("--chunk", type=int, default=PL.DEFAULT_CHUNK)
    ap.add_argument("--max-loans", type=int, default=None, help="cap alive loans (smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"k{COVID_ANCHOR} only, CPU, cap {PL.SMOKE_CAP:,} loans (standing rule 3)")
    ap.add_argument("--force", action="store_true", help="recompute even if .done exists")
    ap.add_argument("--figures", action="store_true", help="also render F5.2 bucket heatmaps")
    ap.add_argument("--only-tables", action="store_true",
                    help="assemble T4.2/T5.1/headline from completed anchors; no compute")
    args = ap.parse_args()

    if args.only_tables:
        config.require_drive()
        build_tables(smoke=args.smoke, figures=args.figures)
        return

    if args.smoke:
        device = torch.device("cpu")
        anchors = [COVID_ANCHOR]
        max_loans = args.max_loans if args.max_loans is not None else PL.SMOKE_CAP
    else:
        device = (torch.device("cuda") if args.device == "cuda" else tc.resolve_device(args.device))
        anchors = args.anchors if args.anchors else ANCHORS_PRIORITY
        max_loans = args.max_loans
    run(anchors, device, max_loans=max_loans, chunk=args.chunk, force=args.force, figures=args.figures)


if __name__ == "__main__":
    main()
