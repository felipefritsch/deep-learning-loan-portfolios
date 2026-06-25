"""M27b — Mac-merge: extend the M24 pricing exhibits with the GRU **H=1** column.

The M24 FF econ grid (logit / best-NN / 8-net ensemble × H{1,3,6,12}) lives on the SSD under
``outputs/tables/pool_level/m24/``. The GRU econ — single GRU (``econ_gru_k*``) and a 3-member GRU
ensemble (``econ_gru_ens_k*``) at **H=1 only** — comes from the pod under
``outputs/tables/pool_level/m27b/``. A standalone **pool-alignment gate** (see the M27b notes /
``align_gate``) has already confirmed the pod reproduced the M24 pool structure faithfully at H=1
(``n_loans`` exact; ``wac/wam/upb/cpr_real/wal_real/price_real`` to ~1e-14). On that clean alignment
this module merges the GRU column into the three M24 exhibits and writes them **alongside** the M24
originals (``t_m27b_*`` / ``F5.2_*_m27b_*``) — the M24 ``t_m24_*`` files are never touched.

Reuse: the merged tables are built with the **same** reduction helpers as M24
(:func:`pricing_grid._t51_agg`, :func:`pricing_grid._horizon_regime_grid`,
:func:`economics.fig_price_error_buckets`), so the GRU column is computed identically to the FF
columns. The GRU model is present **only at H=1**; its H{3,6,12} cells stay blank (future work).

Run (Mac, SSD mounted; seconds — pure post-processing of the parquet, no GPU/torch)::

    .venv/bin/python -m floan.model.m27b_merge
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from floan.model import config
from floan.model import economics as EC
from floan.model import pools as PP
from floan.model import pricing_grid as PG

# GRU artifact coverage (from the pod): single GRU at all 11 anchors; the 3-net GRU ensemble at the
# 5 key windows (mirrors the FF ensemble's coverage). Pool alignment verified at H=1 before merge.
GRU_ALL = list(range(2015, 2026))
GRU_ENS = [2015, 2019, 2020, 2023, 2025]
H1 = 1

# Display order + labels for the merged exhibits. logit/nn/gru span all 11 anchors; ensemble/gru_ens
# only the 5 key windows (blank elsewhere — same convention as the M24 ensemble column).
MODELS_ORDER = ("empirical", "logit", "nn", "ensemble", "gru", "gru_ens")
LABELS = {**PP.MODEL_LABELS, "gru": "GRU (H=1)", "gru_ens": "GRU ×3"}
# F5.2 panels (per anchor): logit + GRU everywhere; the FF/GRU ensembles where present.
F52_PANELS = ("logit", "ensemble", "gru", "gru_ens")


def _m27bdir() -> Path:
    d = config.OUTPUTS / "tables" / "pool_level" / "m27b"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ===========================================================================
# Merged econ frame: M24 (all H) + GRU (H=1) + GRU-ens (H=1, relabelled)
# ===========================================================================
def merged_econ() -> pl.DataFrame:
    """M24 econ (all anchors, all H) ⊕ the GRU econ at H=1. The ``econ_gru_ens_*`` files carry
    ``model='gru'`` on disk (same builder as the single GRU); they are relabelled to ``gru_ens`` so
    the two GRU variants are distinguishable in the merged exhibits."""
    m24 = PG._load_completed("econ", False)                  # 11 anchors × {empirical,logit,nn,ensemble} × H{1,3,6,12}
    if m24 is None:
        raise SystemExit("M27b: no M24 econ artifacts found — run M24 first.")
    parts = [m24]
    d = _m27bdir()
    for k in GRU_ALL:
        parts.append(pl.read_parquet(d / f"econ_gru_k{k}.parquet"))         # model='gru'
    for k in GRU_ENS:
        g = pl.read_parquet(d / f"econ_gru_ens_k{k}.parquet")
        parts.append(g.with_columns(pl.lit("gru_ens").alias("model")))      # relabel gru → gru_ens
    return pl.concat(parts, how="vertical_relaxed")


# ===========================================================================
# T5.1 (merged) — mean |price err| + signed bias by model × anchor × H, GRU at H=1
# ===========================================================================
def _md_T51_merged(agg: pl.DataFrame) -> str:
    char = agg.filter(pl.col("scheme") == "char")
    present = char.get_column("model").unique().to_list()
    models = [m for m in MODELS_ORDER if m in present]
    lines = ["# T5.1 (M27b merge) — pool economic error by model × anchor × H",
             "",
             "Mean |price error| (per 100 face) with signed bias in parentheses; **char** scheme. "
             "Lower |err| is better. GRU / GRU×3 are scored at **H=1 only** (H{3,6,12} blank — "
             "future work); ensemble / GRU×3 exist only at the 5 key windows.",
             "",
             "| anchor | H | " + " | ".join(LABELS[m] for m in models) + " |",
             "|" + "---|" * (len(models) + 2)]
    for (k, h), g in char.sort(["k", "h"]).group_by(["k", "h"], maintain_order=True):
        cells = []
        for m in models:
            r = g.filter(pl.col("model") == m)
            if r.height:
                cells.append(f"{r['price_mae'][0]:.3f} ({r['price_bias'][0]:+.3f})")
            else:
                cells.append("—")
        lines.append(f"| Dec{k-1} | {h} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_T51_merged(econ: pl.DataFrame) -> dict:
    agg = PG._t51_agg(econ)                                   # identical reduction to M24
    d = _m27bdir()
    PG._atomic_parquet(agg, d / "t_m27b_econ_errors.parquet")
    out = {"anchors": sorted(econ.get_column("k").unique().to_list()),
           "horizons": sorted(econ.get_column("h").unique().to_list()),
           "note": "M24 FF grid + GRU/GRU×3 at H=1 (relabelled from econ_gru_ens); "
                   "GRU H{3,6,12} blank (future work).",
           "rows": agg.to_dicts()}
    PG._atomic_text(json.dumps(out, indent=2, default=float), d / "t_m27b_econ_errors.json")
    PG._atomic_text(_md_T51_merged(agg), d / "t_m27b_econ_errors.md")
    return out


# ===========================================================================
# Horizon × regime (merged) — GRU at H=1; gru/gru_ens-vs-logit reduction added from mae_price
# ===========================================================================
def _add_gru_reductions(grid: list[dict]) -> list[dict]:
    """The shared :func:`pricing_grid._horizon_regime_grid` emits ensemble/nn-vs-logit reductions;
    add the GRU variants from the per-cell ``mae_price`` (same formula, same logit base)."""
    for r in grid:
        mae = r["mae_price"]
        base = mae.get("logit")
        if base and base > 0:
            for m in ("gru", "gru_ens"):
                if m in mae:
                    r[f"{m}_vs_logit_pct"] = round(100.0 * (base - mae[m]) / base, 2)
    return grid


def _md_horizon_merged(grid: list[dict]) -> str:
    lines = ["# Horizon × regime (M27b merge) — price-error reduction, GRU at H=1",
             "",
             "Mean |price error| (char, per 100); model-vs-logit % reduction (+ = beats logit). "
             "GRU / GRU×3 at **H=1 only**.",
             "",
             "| anchor | H | logit | nn | ensemble | gru | gru×3 | "
             "nn↓ | ens↓ | gru↓ | gru×3↓ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    def cell(d, m): return f"{d[m]:.3f}" if m in d else "—"
    def red(r, m):  return f"{r[m]:+.1f}%" if m in r else "—"
    for r in grid:
        m = r["mae_price"]
        lines.append(
            f"| {r['anchor']} | {r['h']} | "
            f"{cell(m,'logit')} | {cell(m,'nn')} | {cell(m,'ensemble')} | "
            f"{cell(m,'gru')} | {cell(m,'gru_ens')} | "
            f"{red(r,'nn_vs_logit_pct')} | {red(r,'ensemble_vs_logit_pct')} | "
            f"{red(r,'gru_vs_logit_pct')} | {red(r,'gru_ens_vs_logit_pct')} |")
    return "\n".join(lines) + "\n"


def build_horizon_regime_merged(econ: pl.DataFrame) -> dict:
    grid = _add_gru_reductions(PG._horizon_regime_grid(econ))   # identical per-cell mae/$ to M24
    d = _m27bdir()
    out = {"anchors": sorted({r["k"] for r in grid}),
           "horizons": sorted(econ.get_column("h").unique().to_list()),
           "grid": grid,
           "covid_inversion_dollars": [r for r in grid if r["k"] == PG.COVID_ANCHOR]}
    PG._atomic_text(json.dumps(out, indent=2, default=float), d / "t_m27b_horizon_regime.json")
    PG._atomic_text(_md_horizon_merged(grid), d / "t_m27b_horizon_regime.md")
    return out


# ===========================================================================
# F5.2 (merged) — GRU H=1 signed price-error by characteristic bucket, per anchor
# ===========================================================================
def build_F52_gru(econ: pl.DataFrame) -> list[str]:
    """Per-anchor F5.2 at H=1: signed UPB-weighted price error by FICO × orig-rate-quartile bucket,
    GRU alongside logit (and the FF/GRU ensembles where present). Written with an ``_m27b_h01``
    suffix so the M24 F5.2 panels are not overwritten."""
    made: list[str] = []
    h1 = econ.filter((pl.col("scheme") == "char") & (pl.col("h") == H1))
    for (k,), gk in h1.group_by(["k"], maintain_order=True):
        present = set(gk.get_column("model").unique().to_list())
        panels = tuple(m for m in F52_PANELS if m in present)
        try:
            p = EC.fig_price_error_buckets(gk, int(k), suffix="_m27b_h01",
                                           models=panels, labels=LABELS)
            made.append(str(p))
        except Exception as e:                                 # figures are non-gating
            print(f"  F5.2 (m27b) k{k}: skipped ({type(e).__name__}: {e})")
    return made


# ===========================================================================
# Step 4 — FULL H=1 pricing comparison across all 11 anchors (char + random)
# ===========================================================================
COMPARE_MODELS = ("logit", "nn", "ensemble", "gru", "gru_ens")


def h1_compare(econ: pl.DataFrame) -> dict:
    """Mean |price error| at H=1 by scheme × anchor × model (per-anchor) plus two pooled rows:
    pooled over every anchor a model has, and pooled over the 5 key windows (all 5 models present →
    apples-to-apples). Returns frames + the Markdown; the caller prints + persists."""
    h1 = econ.filter(pl.col("h") == H1)
    per = (h1.group_by(["scheme", "k", "model"])
           .agg(pl.col("price_err").abs().mean().alias("mae"),
                pl.col("price_err").mean().alias("bias"),
                pl.len().alias("n_pools"))
           .sort(["scheme", "k", "model"]))
    pooled_all = (h1.group_by(["scheme", "model"])
                  .agg(pl.col("price_err").abs().mean().alias("mae"),
                       pl.col("price_err").mean().alias("bias"),
                       pl.col("k").n_unique().alias("n_anchors"))
                  .sort(["scheme", "model"]))
    key = h1.filter(pl.col("k").is_in(GRU_ENS))
    pooled_key = (key.group_by(["scheme", "model"])
                  .agg(pl.col("price_err").abs().mean().alias("mae"),
                       pl.col("price_err").mean().alias("bias"))
                  .sort(["scheme", "model"]))
    return {"per": per, "pooled_all": pooled_all, "pooled_key": pooled_key}


def _md_compare(cmp: dict) -> str:
    per, p_all, p_key = cmp["per"], cmp["pooled_all"], cmp["pooled_key"]
    lines = ["# M27b — full H=1 pricing comparison (mean |price error| per 100 face)",
             "",
             "Logit / NN / Ensemble×8 / GRU / GRU×3 at **H=1**, by anchor and scheme. Ensemble and "
             "GRU×3 exist only at the 5 key windows (Dec2014/18/19/22/24). 'Pooled (key 5)' pools "
             "the five windows where every model is present (apples-to-apples); 'Pooled (all)' "
             "pools every anchor a model has.",
             ""]
    for scheme in ("char", "random"):
        sname = "Characteristic buckets" if scheme == "char" else "Random pools (500×1000)"
        lines += [f"## {sname}", "",
                  "| anchor | " + " | ".join(LABELS[m] for m in COMPARE_MODELS) + " |",
                  "|" + "---|" * (len(COMPARE_MODELS) + 1)]
        sub = per.filter(pl.col("scheme") == scheme)
        for k in sorted(sub.get_column("k").unique().to_list()):
            g = sub.filter(pl.col("k") == k)
            cells = []
            for m in COMPARE_MODELS:
                r = g.filter(pl.col("model") == m)
                cells.append(f"{r['mae'][0]:.3f}" if r.height else "—")
            lines.append(f"| Dec{k-1} | " + " | ".join(cells) + " |")
        # pooled rows
        for tag, pf in (("Pooled (key 5)", p_key), ("Pooled (all)", p_all)):
            g = pf.filter(pl.col("scheme") == scheme)
            cells = []
            for m in COMPARE_MODELS:
                r = g.filter(pl.col("model") == m)
                cells.append(f"**{r['mae'][0]:.3f}**" if r.height else "—")
            lines.append(f"| _{tag}_ | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_compare(econ: pl.DataFrame) -> dict:
    cmp = h1_compare(econ)
    d = _m27bdir()
    PG._atomic_parquet(cmp["per"], d / "t_m27b_h1_compare.parquet")
    md = _md_compare(cmp)
    PG._atomic_text(md, d / "t_m27b_h1_compare.md")
    cmp["md"] = md
    return cmp


# ===========================================================================
# Driver
# ===========================================================================
def run(figures: bool = True) -> dict:
    config.require_drive()
    econ = merged_econ()
    print(f"merged econ: {econ.shape}  models={sorted(econ.get_column('model').unique().to_list())}  "
          f"anchors={sorted(econ.get_column('k').unique().to_list())}", flush=True)
    res = {"T51": build_T51_merged(econ),
           "horizon_regime": build_horizon_regime_merged(econ),
           "compare": build_compare(econ)}
    if figures:
        res["figures"] = build_F52_gru(econ)
    print(f"=== M27b merged exhibits → {_m27bdir()} "
          f"({len(res.get('figures', []))} F5.2 panels) ===", flush=True)
    print("\n" + res["compare"]["md"])
    return res


def main() -> None:
    run()


if __name__ == "__main__":
    main()
