"""M27b driver — h=1 GRU pool pricing across anchors + the H=1 identity guard.

Per anchor: train/load the window-k GRU (seed 0), assert the H=1 identity (composed == GRU
evaluate, before pricing), then price the GRU on the SAME M24 pools (char buckets + random pools)
through the reused engine. On the 5 key windows also price the seed-{0,1,2} GRU ensemble (mirrors
the FF mean-of-members rule). Writes per-anchor econ artifacts (schema-compatible with M24's
econ_k*.parquet, so the FF side-by-side merges on the box that holds the M24 grid) + a comparison
summary. H=1 ONLY (the GRU's weakest horizon — framed honestly in the report).

    python scripts/m27b/price_h1.py --device cuda                      # all 11 anchors + key ensembles
    python scripts/m27b/price_h1.py --device cuda --anchors 2020       # one anchor
    python scripts/m27b/price_h1.py --device cuda --anchors 2020 --max-loans 20000   # wiring smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import seq_predict as SQ
from floan.model import torch_common as tc

KEY = list((2015, 2019, 2020, 2023, 2025))
ENS_SEEDS = (0, 1, 2)
# Committed M24 FF H=1 reference (char scheme, mean|price err|) — the only FF H=1 numbers that live
# on the pod (M24_NOTES.md §3; full per-anchor FF grid is on the SSD/Mac). k2020 = the COVID anchor.
M24_FF_H1 = {2020: {"logit": 0.1822, "ensemble": 0.1030}}


def _outdir() -> Path:
    d = config.OUTPUTS / "tables" / "pool_level" / "m27b"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _char_mae(econ: pl.DataFrame, scheme: str, col: str = "price_err") -> float:
    e = econ.filter(pl.col("scheme") == scheme).get_column(col).to_numpy()
    return float(np.abs(e).mean()) if e.size else float("nan")


def _bias(econ: pl.DataFrame, scheme: str, col: str) -> float:
    e = econ.filter(pl.col("scheme") == scheme).get_column(col).to_numpy()
    return float(e.mean()) if e.size else float("nan")


def price_one(label: str, models, k: int, scaler, vocab, device, pop, *, chunk: int) -> dict:
    """Identity guard (asserts before pricing) → H=1 pricing → per-anchor row for the summary."""
    SQ.identity_check(models, k, scaler, vocab, device, pop)
    out = SQ.price_anchor_h1(models, k, scaler, vocab, device, pop, chunk=chunk)
    econ, meta = out["econ"], out["meta"]
    # The NaN H>1 tail must never be priced: every priced row is h=1 with finite price/err.
    assert econ.get_column("h").unique().to_list() == [1], "non-H=1 rows leaked into the GRU econ"
    assert bool(np.isfinite(econ.select(["price", "price_err", "cpr", "wal"]).to_numpy()).all()), \
        "non-finite GRU price/err at H=1"
    row = {"k": k, "anchor": f"Dec{k - 1}", "model": label, "n_pop": meta["n_pop"],
           "n_members": meta["n_members"],
           "char_price_mae": _char_mae(econ, "char"), "char_price_bias": _bias(econ, "char", "price_err"),
           "rand_price_mae": _char_mae(econ, "random"), "rand_price_bias": _bias(econ, "random", "price_err"),
           "char_cpr_mae": _char_mae(econ, "char", "cpr_err"), "char_cpr_bias": _bias(econ, "char", "cpr_err"),
           "char_wal_mae": _char_mae(econ, "char", "wal_err"), "char_wal_bias": _bias(econ, "char", "wal_err"),
           "E_cum_prepaid_h1": meta["E_cum_prepaid_h1"], "E_cum_dpd60p_h1": meta["E_cum_dpd60p_h1"]}
    return {"econ": econ, "smm_frame": out["smm_frame"], "row": row}


def run(anchors: list[int], device, *, max_loans: int | None, chunk: int, retrain: bool) -> dict:
    d = _outdir()
    smoke = max_loans is not None
    rows: list[dict] = []
    for k in anchors:
        print(f"\n=== M27b k{k} (anchor Dec{k - 1}){' SMOKE' if smoke else ''} ===", flush=True)
        t0 = time.perf_counter()
        model, scaler, vocab = SQ.train_or_load(k, 0, device, retrain=retrain)
        pop = SQ.anchor_pop(k)
        if smoke:
            pop = pop.head(max_loans)
        print(f"  pop n={pop.height:,}", flush=True)

        single = price_one("gru", [model], k, scaler, vocab, device, pop, chunk=chunk)
        rows.append(single["row"])
        if not smoke:
            single["econ"].write_parquet(d / f"econ_gru_k{k}.parquet")
            single["smm_frame"].write_parquet(d / f"smm_paths_gru_k{k}.parquet")

        if k in KEY:                                   # seed-ensemble, mirroring the FF ensemble
            members = [model]
            for s in ENS_SEEDS:
                if s == 0:
                    continue
                m_s, _, _ = SQ.train_or_load(k, s, device, retrain=retrain)
                members.append(m_s)
            ens = price_one("gru_ens", members, k, scaler, vocab, device, pop, chunk=chunk)
            rows.append(ens["row"])
            if not smoke:
                ens["econ"].write_parquet(d / f"econ_gru_ens_k{k}.parquet")
                ens["smm_frame"].write_parquet(d / f"smm_paths_gru_ens_k{k}.parquet")
        print(f"  k{k} done in {time.perf_counter() - t0:.0f}s", flush=True)

    summary = pl.DataFrame(rows)
    if not smoke:
        summary.write_parquet(d / "m27b_summary.parquet")
        (d / "m27b_summary.json").write_text(json.dumps(rows, indent=2, default=float))
    _report(summary)
    return {"rows": rows}


def _report(summary: pl.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("M27b — GRU H=1 pool pricing (char buckets). Errors = GRU − realized through "
          "pool.cashflow_engine.")
    print("=" * 92)
    hdr = f"  {'anchor':>8} {'model':>9} {'n_pop':>8} | {'char|perr|':>10} {'(bias)':>9} | " \
          f"{'rand|perr|':>10} | {'char|cprE|pp':>12} {'char|walE|mo':>12}"
    print(hdr); print("  " + "-" * 88)
    for r in summary.iter_rows(named=True):
        # cpr_err is already pp and wal_err already months (per_pool_econ scales them) — no re-convert.
        print(f"  {r['anchor']:>8} {r['model']:>9} {r['n_pop']:>8,} | "
              f"{r['char_price_mae']:>10.4f} {r['char_price_bias']:>+9.3f} | "
              f"{r['rand_price_mae']:>10.4f} | {r['char_cpr_mae']:>12.3f} "
              f"{r['char_wal_mae']:>12.3f}")
    # k2020 H=1 vs the committed FF reference (the only on-pod FF H=1 numbers).
    print("\n  -- k2020 (COVID anchor) H=1 char mean|price err|: GRU vs committed M24 FF --")
    g = summary.filter((pl.col("k") == 2020) & (pl.col("model") == "gru"))
    ge = summary.filter((pl.col("k") == 2020) & (pl.col("model") == "gru_ens"))
    ref = M24_FF_H1.get(2020, {})
    if g.height:
        print(f"     logit (M24)   : {ref.get('logit', float('nan')):.4f}")
        print(f"     ensemble (M24): {ref.get('ensemble', float('nan')):.4f}")
        print(f"     GRU single    : {g.get_column('char_price_mae')[0]:.4f}")
        if ge.height:
            print(f"     GRU ensemble  : {ge.get_column('char_price_mae')[0]:.4f}")
    print("=" * 92)
    print("NOTE: full per-anchor FF (logit/NN/ensemble) H=1 grid lives in the M24 econ_k*.parquet on "
          "the SSD/Mac; the\n      econ_gru_k*.parquet emitted here are schema-compatible for that "
          "merge. H=1 is the GRU's weakest horizon.")


def main() -> None:
    ap = argparse.ArgumentParser(description="M27b — h=1 GRU pool pricing + identity guard.")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--anchors", type=int, nargs="*", default=list(range(2015, 2026)))
    ap.add_argument("--max-loans", type=int, default=None, help="cap pop (wiring smoke; no artifacts)")
    ap.add_argument("--chunk", type=int, default=300_000, help="loans per build/score chunk")
    ap.add_argument("--retrain", action="store_true", help="ignore persisted weights, retrain")
    ap.add_argument("--duckdb-mem", default="32GB", help="runtime DuckDB memory_limit (no config edit)")
    args = ap.parse_args()
    config.require_drive()
    config.DUCKDB_MEMORY_LIMIT = args.duckdb_mem        # bump at runtime, as in the cache build
    device = (torch.device("cuda") if args.device == "cuda" else tc.resolve_device(args.device))
    run(args.anchors, device, max_loans=args.max_loans, chunk=args.chunk, retrain=args.retrain)


if __name__ == "__main__":
    main()
