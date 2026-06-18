"""M15a (A) — monthly pool SMM paths (``03_POOL_LEVEL §5.1``), the economic-translation input.

The §4 pool exercise (M14) compared *counts*; §5 converts those into the units an MBS investor
prices. The first link in that chain is the pool **prepayment speed**: a monthly single-monthly-
mortality (SMM) path ``SMM_P(h), h=1..12`` per pool × model, plus the realized analogue from the
panel. The cashflow engine (``pool.py``, M15a (B)) then turns ``(WAC, WAM, UPB, SMM_P)`` into WAL
and price; M15b computes the CPR/WAL/price errors (T5.1 / F5.2).

Construction (consumes the extended M13 roll-forward, :func:`pool.roll_forward` with
``capture_smm``):

  * **Per loan, per step** the roll-forward records the composed cumulative prepaid mass and the
    transient (alive) mass. The unconditional prepay increment ``Δ_h(i) = p_i(h)[prepaid] −
    p_i(h−1)[prepaid]`` is the model's probability that loan ``i`` prepays during month ``h``.
  * **Pool SMM** (UPB-weighted, balance-of-survivors-conditional):
    ``SMM_P(h) = Σ_i w_i·Δ_h(i) / Σ_i w_i·alive_i(h−1)`` with ``w_i`` the loan's Current Actual
    UPB at ``t0`` (frozen, per the §3 covariate-freezing assumption — stated as a caveat).
  * **Realized SMM** mirrors it from the panel: ``Σ_i w_i·1{i prepays in month h} / Σ_i w_i·1{i
    alive at start of month h}`` over the same 12 transitions ``[t0, t0+12)``. Same ``w_i``, same
    denominator convention ⇒ model and realized are directly comparable.

Pools and population are **exactly M14's** (``pools.py``): the t0-alive ``current`` loans with
non-null FICO/rate/LTV, partitioned into FICO×rate×LTV characteristic cells *and* drawn into
``B=500`` random pools of ``N=1000`` — both rebuilt from the same seeded
:func:`pools.char_cell_ids` / :func:`pools.random_pool_index` (membership is reproducible, M14
Accept #1), then **cross-checked** against the committed ``pools_random_k*.parquet`` realized
counts so the pools are provably identical (no rebuild). Models: ``{empirical, logit, ensemble}``
+ ``realized``.

Output (one file per anchor): ``outputs/tables/pool_level/smm_paths_k{k}.parquet`` — long over
``(scheme, pool, model)`` with the 12 SMM columns ``smm_h01..smm_h12`` and the pool's
engine-ready ``wac`` (annual **decimal**, UPB-weighted Current Interest Rate), ``wam`` (months),
``upb`` (face $), ``n_loans``.

Run (CPU background job — the roll-forward scoring is the slow piece, ~1–2 h for all 5 anchors):
    .venv/bin/python -u dev/model/smm_paths.py --device cpu            # all 5 key anchors
    .venv/bin/python -u dev/model/smm_paths.py --k 2020 --device cpu   # one anchor
    .venv/bin/python    dev/model/smm_paths.py --k 2020 --max-loans 40000 --device cpu  # smoke
Hermetic engine tests live in ``test_pool.py``; this driver needs the SSD + frozen models.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import data as D
from floan.model import pool as PL
from floan.model import pools as PP
from floan.model import torch_common as tc
from floan.model import train as T

VARIANT = "full"
ANCHOR_WINDOWS = list(PL.E.KEY_WINDOWS)              # the 5 ensemble-bearing key windows
HORIZON = PL.HORIZON                                 # 12
# §5 headline set is {empirical, logit, ensemble}; the logit/full checkpoints were never synced
# off the pod, so the default background run scores {empirical, ensemble} and logit is merged in a
# second pass (``--add-logit``) once re-fit + validated. ``realized`` is always computed (panel).
DEFAULT_MODELS = ("empirical", "ensemble")
MODEL_LABELS = {"empirical": "Empirical", "logit": "Logit", "nn": "Best NN",
                "ensemble": "Ensemble ×8", "realized": "Realized"}


def _which(requested) -> set:
    """Torch-model subset for ``pool.roll_forward(which=...)`` — the requested models minus the
    feature-free empirical matrix (always built) and realized (panel-derived, computed here)."""
    return {m for m in requested if m not in ("empirical", "realized")}


# ===========================================================================
# Population — the exact M14 pool population, plus UPB / WAC / WAM weights
# ===========================================================================
def _anchor_features(k: int) -> pl.DataFrame:
    """t0-alive ``current`` loans keyed by loan id with the M14 bucketing covariates
    (fico/rate/ltv — identical casts to ``pools._anchor_features`` so membership reproduces)
    *plus* the cashflow weights: ``upb`` (Current Actual UPB), ``wac`` (Current Interest Rate,
    annual decimal), ``wam`` (Remaining Months to Maturity). One row per loan at the anchor."""
    t0 = config._dec(k - 1)
    pool_dir, _ = D.window_spec(VARIANT, k, "test")
    return (pl.scan_parquet(str(pool_dir / "part-*.parquet"))
            .filter((pl.col("period_ym") == t0) & (pl.col("state") == "current"))
            .select(pl.col("Loan Identifier"),
                    pl.col("fico_orig").cast(pl.Float64).alias("fico"),
                    pl.col("Original Interest Rate").cast(pl.Float64).alias("rate"),
                    pl.col("Original Loan-to-Value (LTV)").cast(pl.Float64).alias("ltv"),
                    pl.col("Current Actual UPB").cast(pl.Float64).alias("upb"),
                    (pl.col("Current Interest Rate").cast(pl.Float64) / 100.0).alias("wac"),
                    pl.col("Remaining Months to Maturity").cast(pl.Float64).alias("wam"))
            .collect())


def _realized_steps(k: int, pop_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[n_pop, 12]`` realized indicator arrays aligned to ``pop_ids``: ``alive[i, j] = 1`` if
    loan ``i`` has a panel transition at month ``t0+j`` (alive at the start of step ``j+1``);
    ``prepay[i, j] = 1`` if that transition's ``state_next == prepaid``. Built over the realized
    window ``test_bounds(k) = [t0, t0+12)`` — the same 12 transitions the roll-forward composes."""
    t0 = config._dec(k - 1)
    pool_dir, (lo, hi) = D.window_spec(VARIANT, k, "test")
    pop_map = pl.DataFrame({"Loan Identifier": pop_ids,
                            "row": np.arange(pop_ids.shape[0], dtype=np.int64)})
    rows = (pl.scan_parquet(str(pool_dir / "part-*.parquet"))
            .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
            .select("Loan Identifier", "period_ym", "state_next")
            .with_columns(((pl.col("period_ym") // 100 - t0 // 100) * 12
                           + (pl.col("period_ym") % 100 - t0 % 100)).alias("step"))
            .filter((pl.col("step") >= 0) & (pl.col("step") < HORIZON))
            .join(pop_map.lazy(), on="Loan Identifier", how="inner")
            .select("row", "step", (pl.col("state_next") == "prepaid").alias("prep"))
            .collect())
    r = rows.get_column("row").to_numpy()
    s = rows.get_column("step").to_numpy()
    pp = rows.get_column("prep").to_numpy().astype(np.float64)
    alive = np.zeros((pop_ids.shape[0], HORIZON), np.float64)
    prepay = np.zeros((pop_ids.shape[0], HORIZON), np.float64)
    alive[r, s] = 1.0                                  # (loan, month) is unique ⇒ no double count
    prepay[r, s] = pp
    return alive, prepay


def per_loan_smm_table(k: int, device, requested, *, max_loans: int | None = None,
                       chunk: int = PL.DEFAULT_CHUNK) -> dict:
    """Build the per-loan master arrays for the SMM aggregation: roll-forward SMM capture
    (cumulative-prepaid + alive mass per model in ``requested``), realized per-step indicators,
    and UPB/WAC/WAM, all on the **exact M14 population** (current origin, non-null fico/rate/ltv,
    sorted by loan id — the same rows and order ``pools.py`` bucketed/sampled)."""
    rf = PL.roll_forward(k, device, chunk=chunk, max_loans=max_loans, horizon=HORIZON,
                         capture_smm=True, which=_which(requested))
    model_names = rf["model_names"]                    # ["empirical", *torch in roll_forward order]
    preds = rf["result"].with_row_index("rf_row")      # rf_row aligns to rf["smm"] array rows
    n_alive = preds.height

    feats = _anchor_features(k)
    df = (preds.filter(pl.col("origin") == "current")
               .join(feats, on="Loan Identifier", how="inner"))
    n_current = df.height
    df = df.drop_nulls(subset=["fico", "rate", "ltv"])          # M14's exact membership filter
    df = df.sort("Loan Identifier")                             # M14's exact ordering
    n_pop = df.height

    # M14's membership drops nulls on fico/rate/ltv ONLY, so a few current loans carry a null
    # Remaining-Months (and, in principle, UPB). Those loans STAY in the pool (membership must
    # match M14); null UPB → 0 weight, null WAM → excluded from the pool WAM mean only — neither
    # touches the SMM path. Logged for the record.
    null_upb, null_wac, null_wam = df.select(pl.col(["upb", "wac", "wam"]).is_null().sum()).row(0)
    if null_upb or null_wac or null_wam:
        print(f"  [{k}] null weights kept in pop (0-weight / excluded from WAM): "
              f"upb={null_upb} wac={null_wac} wam={null_wam} of {n_pop:,}", flush=True)

    rf_row = df.get_column("rf_row").to_numpy()
    smm = {m: {"cum": rf["smm"][m]["cum_prepaid"][rf_row].astype(np.float64),
               "alive": rf["smm"][m]["alive_before"][rf_row].astype(np.float64)}
           for m in model_names}
    pop_ids = df.get_column("Loan Identifier").to_numpy()
    alive_real, prepay_real = _realized_steps(k, pop_ids)

    meta = {"k": k, "t0": config._dec(k - 1), "n_alive": n_alive, "n_current": n_current,
            "n_pop": n_pop, "n_dropped_nullbucket": n_current - n_pop, "rf_n_loans": rf["n_loans"],
            "model_names": list(model_names)}
    return {"df": df, "model_names": list(model_names), "smm": smm, "alive_real": alive_real,
            "prepay_real": prepay_real, "upb": df.get_column("upb").to_numpy(),
            "wac": df.get_column("wac").to_numpy(), "wam": df.get_column("wam").to_numpy(),
            "pop_ids": pop_ids, "meta": meta}


# ===========================================================================
# Pool SMM aggregation — UPB-weighted, per scheme
# ===========================================================================
def _increments(cum: np.ndarray) -> np.ndarray:
    """First differences of the cumulative-prepaid path: ``Δ_h`` = mass prepaying *during* step
    ``h`` (``Δ_1`` = ``cum[:,0]`` since current-origin loans carry zero prepaid mass at t0)."""
    d = np.empty_like(cum)
    d[:, 0] = cum[:, 0]
    d[:, 1:] = cum[:, 1:] - cum[:, :-1]
    return d


def _smm_from_sums(num_pool: np.ndarray, den_pool: np.ndarray) -> np.ndarray:
    """``SMM_P(h)`` = UPB-weighted prepay-$ over UPB-weighted alive-$, guarding empty denominators."""
    return np.divide(num_pool, den_pool, out=np.zeros_like(num_pool), where=den_pool > 0)


def _pool_paths_random(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Pool SMM matrix ``[B, 12]`` for the random scheme: sum the UPB-weighted numerator/alive
    arrays over each pool's ``N`` members (membership matrix ``idx`` ``[B, N]``)."""
    num_pool = num[idx].sum(axis=1)
    den_pool = den[idx].sum(axis=1)
    return _smm_from_sums(num_pool, den_pool)


def _cell_groupsum(cell: np.ndarray, *arrs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group-sum each ``[n, ...]`` array by characteristic-cell id (exhaustive partition)."""
    order = np.argsort(cell, kind="stable")
    cs = cell[order]
    uniq, start = np.unique(cs, return_index=True)
    return uniq, [np.add.reduceat(a[order], start, axis=0) for a in arrs]


def pool_smm_paths(tbl: dict, k: int) -> dict:
    """Assemble per-scheme pool SMM paths (model + realized) and the pool WAC/WAM/UPB weights.
    Returns ``{scheme: {"meta": DataFrame[pool, n_loans, wac, wam, upb], "smm": {model: [P,12]}}}``."""
    df = tbl["df"]
    w = np.nan_to_num(tbl["upb"], nan=0.0)             # UPB weight; a null-UPB loan ⇒ 0 weight
    wac, wam = tbl["wac"], tbl["wam"]
    cell = PP.char_cell_ids(df)[0]
    idx = PP.random_pool_index(df.height, k)

    # Per-loan, per-step UPB-weighted numerator (prepay $) and alive denominator ($) per source.
    num, den = {}, {}
    for m in tbl["model_names"]:
        num[m] = _increments(tbl["smm"][m]["cum"]) * w[:, None]
        den[m] = tbl["smm"][m]["alive"] * w[:, None]
    num["realized"] = tbl["prepay_real"] * w[:, None]
    den["realized"] = tbl["alive_real"] * w[:, None]
    sources = (*tbl["model_names"], "realized")

    # UPB-weighted-mean components for WAC/WAM — each excludes its own non-finite entries from
    # *its* denominator (so a null WAM drops only from the WAM mean, not from UPB/WAC/SMM).
    wacw = w * np.where(np.isfinite(wac), wac, 0.0); wac_wd = w * np.isfinite(wac)
    wamw = w * np.where(np.isfinite(wam), wam, 0.0); wam_wd = w * np.isfinite(wam)

    def _wmean(num_sum, den_sum):
        return np.divide(num_sum, den_sum, out=np.zeros_like(num_sum), where=den_sum > 0)

    out: dict = {}
    # ---- random pools ----
    upb_r = w[idx].sum(axis=1)
    meta_r = pl.DataFrame({
        "pool": np.arange(idx.shape[0], dtype=np.int64),
        "n_loans": np.full(idx.shape[0], idx.shape[1], dtype=np.int64),
        "wac": _wmean(wacw[idx].sum(axis=1), wac_wd[idx].sum(axis=1)),
        "wam": _wmean(wamw[idx].sum(axis=1), wam_wd[idx].sum(axis=1)),
        "upb": upb_r})
    out["random"] = {"meta": meta_r,
                     "smm": {m: _pool_paths_random(num[m], den[m], idx) for m in sources}}

    # ---- characteristic cells ----
    uniq, (upb_c, wacw_c, wacd_c, wamw_c, wamd_c, n_c) = _cell_groupsum(
        cell, w, wacw, wac_wd, wamw, wam_wd, np.ones_like(w))
    meta_c = pl.DataFrame({"pool": uniq.astype(np.int64), "n_loans": n_c.astype(np.int64),
                           "wac": _wmean(wacw_c, wacd_c), "wam": _wmean(wamw_c, wamd_c),
                           "upb": upb_c})
    smm_c = {}
    for m in sources:
        _, (num_c, den_c) = _cell_groupsum(cell, num[m], den[m])
        smm_c[m] = _smm_from_sums(num_c, den_c)
    out["char"] = {"meta": meta_c, "smm": smm_c}
    return out


def paths_to_frame(paths: dict, k: int) -> pl.DataFrame:
    """Flatten the per-scheme pool SMM paths into one long ``smm_paths_k{k}`` frame: one row per
    ``(scheme, pool, model)`` with ``smm_h01..smm_h12`` + the pool's ``wac/wam/upb/n_loans``."""
    smm_cols = [f"smm_h{h:02d}" for h in range(1, HORIZON + 1)]
    frames = []
    for scheme in ("char", "random"):
        meta = paths[scheme]["meta"]
        for model, mat in paths[scheme]["smm"].items():
            blk = meta.with_columns(pl.lit(scheme).alias("scheme"), pl.lit(model).alias("model"))
            blk = blk.with_columns([pl.Series(c, mat[:, j]) for j, c in enumerate(smm_cols)])
            frames.append(blk.select(["scheme", "pool", "model", "n_loans",
                                      "wac", "wam", "upb", *smm_cols]))
    return pl.concat(frames)


# ===========================================================================
# Membership cross-check vs the committed M14 parquet (full runs only)
# ===========================================================================
def crosscheck_membership(tbl: dict, k: int) -> dict:
    """Prove the rebuilt random pools are *exactly* M14's: the realized prepaid count per pool
    (Σ over members of "ever prepaid in 12m") must match the committed
    ``pools_random_k{k}.parquet`` ``prepaid_realized`` column bit-for-bit. Identical realized
    counts over 500 seeded pools ⇒ identical membership (and a faithful realized extraction)."""
    pq = config.OUTPUTS / "tables" / "pool_level" / f"pools_random_k{k}.parquet"
    if not pq.exists():
        return {"checked": False, "reason": f"{pq} absent"}
    idx = PP.random_pool_index(tbl["df"].height, k)
    ever_prepaid = (tbl["prepay_real"].max(axis=1) > 0).astype(np.float64)   # per-loan 12m flag
    mine = ever_prepaid[idx].sum(axis=1)
    ref = pl.read_parquet(pq).get_column("prepaid_realized").to_numpy()
    max_abs = float(np.abs(mine - ref).max()) if mine.shape == ref.shape else float("inf")
    return {"checked": True, "n_pools": int(idx.shape[0]), "max_abs_diff": max_abs,
            "exact": max_abs == 0.0}


# ===========================================================================
# Driver
# ===========================================================================
def _tabdir():
    d = config.OUTPUTS / "tables" / "pool_level"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cpr(smm: float) -> float:
    return 1.0 - (1.0 - smm) ** 12


def _merge_into_parquet(out_path, new_frame: pl.DataFrame, replace_models: set) -> int:
    """Replace the ``replace_models`` rows of an existing ``smm_paths`` parquet with ``new_frame``'s
    rows for those models, leaving every other model untouched (the ``--add-logit`` second pass).
    Returns the merged row count."""
    keep = (pl.read_parquet(out_path).filter(~pl.col("model").is_in(list(replace_models)))
            if out_path.exists() else new_frame.clear())
    add = new_frame.filter(pl.col("model").is_in(list(replace_models)))
    merged = pl.concat([keep, add]).sort(["scheme", "model", "pool"])
    merged.write_parquet(out_path)
    return merged.height


def run_anchor(k: int, device, requested, *, max_loans: int | None = None,
               chunk: int = PL.DEFAULT_CHUNK, merge: bool = False) -> dict:
    print(f"\n=== M15a (A) SMM paths · anchor Dec{k-1} (k={k}) · models={list(requested)}"
          f"{' (MERGE)' if merge else ''} ===", flush=True)
    tk = time.perf_counter()

    print(f"  [{k}] roll-forward over all t0-alive loans (slow CPU step; capturing SMM)...",
          flush=True)
    tbl = per_loan_smm_table(k, device, requested, max_loans=max_loans, chunk=chunk)
    m = tbl["meta"]
    print(f"  [{k}] population: {m['n_alive']:,} alive → {m['n_current']:,} current → "
          f"{m['n_pop']:,} with buckets (dropped {m['n_dropped_nullbucket']:,}) "
          f"[{time.perf_counter()-tk:.0f}s]", flush=True)

    paths = pool_smm_paths(tbl, k)
    frame = paths_to_frame(paths, k)
    sources = [*tbl["model_names"], "realized"]
    out = _tabdir() / f"smm_paths_k{k}{'_smoke' if max_loans else ''}.parquet"
    if merge and not max_loans:
        replace = _which(requested)                  # e.g. {"logit"} — replace only those rows
        n = _merge_into_parquet(out, frame, replace)
        print(f"  [{k}] merged {sorted(replace)} into {out.name} → {n} rows total", flush=True)
    else:
        frame.write_parquet(out)
        print(f"  [{k}] wrote {out.name}: {frame.height} rows "
              f"({len(paths['char']['meta'])} char cells + {len(paths['random']['meta'])} random "
              f"pools) × {len(sources)} sources {sources}", flush=True)

    # ---- sanity: mean monthly SMM and implied CPR, model vs realized (random pools) ----
    sane = {}
    for src in sources:
        mat = paths["random"]["smm"][src]            # [B, 12]
        mean_smm = float(mat.mean())
        sane[src] = {"mean_smm": mean_smm, "implied_cpr": _cpr(mean_smm),
                     "smm_min": float(mat.min()), "smm_max": float(mat.max()),
                     "in_01": bool((mat >= 0).all() and (mat <= 1).all())}
        print(f"      {MODEL_LABELS[src]:>11}: mean SMM={mean_smm:.4f}  "
              f"CPR≈{_cpr(mean_smm)*100:5.1f}%  range[{mat.min():.4f},{mat.max():.4f}]  "
              f"in[0,1]={sane[src]['in_01']}", flush=True)

    xcheck = crosscheck_membership(tbl, k) if max_loans is None else {"checked": False,
                                                                      "reason": "smoke (max_loans)"}
    if xcheck.get("checked"):
        print(f"  [{k}] membership cross-check vs pools_random_k{k}.parquet: "
              f"exact={xcheck['exact']} (max|Δ|={xcheck['max_abs_diff']:g} over "
              f"{xcheck['n_pools']} pools)", flush=True)

    return {"k": k, "meta": m, "sources": sources, "sanity": sane, "crosscheck": xcheck,
            "out_parquet": str(out), "wall_sec": time.perf_counter() - tk}


def run(windows: list[int], device, requested, *, max_loans: int | None = None,
        chunk: int = PL.DEFAULT_CHUNK, merge: bool = False) -> dict:
    t0 = time.perf_counter()
    print(f"=== M15a (A) monthly pool SMM paths · anchors={windows} · models={list(requested)}"
          f"{' (MERGE)' if merge else ''} · device={device}"
          f"{'  max_loans=' + format(max_loans, ',') if max_loans else ''} ===", flush=True)
    per_anchor = {k: run_anchor(k, device, requested, max_loans=max_loans, chunk=chunk,
                                merge=merge) for k in windows}

    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "variant": VARIANT, "anchors": windows, "horizon": HORIZON,
        "requested_models": list(requested), "merge": merge,
        "git_commit": T._git_commit(), "max_loans": max_loans, "per_anchor": per_anchor,
        "wall_sec": time.perf_counter() - t0,
    }
    tag = "_merge_" + "-".join(sorted(_which(requested))) if merge else ""
    out = config.MODELS / "nn" / VARIANT / ("smm_paths_summary" + tag
                                            + ("_smoke" if max_loans else "") + ".json")
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out}  [{summary['wall_sec']/60:.1f} min]  ALL ANCHORS DONE", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="M15a (A) — monthly pool SMM paths (§5.1).")
    ap.add_argument("--k", type=int, default=None, help="single anchor (default: all 5 key)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--chunk", type=int, default=PL.DEFAULT_CHUNK, help="loans per roll-forward chunk")
    ap.add_argument("--max-loans", type=int, default=None, help="cap alive loans (smoke; skips xcheck)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma list ⊂ {empirical,logit,nn,ensemble} (default empirical,ensemble)")
    ap.add_argument("--add-logit", action="store_true",
                    help="merge mode: score logit only and replace its rows in the existing parquet")
    args = ap.parse_args()
    config.require_drive()
    device = (torch.device("cuda") if args.device == "cuda" else tc.resolve_device(args.device))
    windows = [args.k] if args.k else ANCHOR_WINDOWS
    requested = ["empirical", "logit"] if args.add_logit else [
        m.strip() for m in args.models.split(",") if m.strip()]
    run(windows, device, requested, max_loans=args.max_loans, chunk=args.chunk,
        merge=args.add_logit)


if __name__ == "__main__":
    main()
