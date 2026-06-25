"""M15b — economic translation: CPR / WAL / price errors per pool × model × anchor (``03_POOL_LEVEL §5.3``).

Consumes the **M15a** artefacts only (no re-fit, no re-score):
  * ``outputs/tables/pool_level/smm_paths_k{k}.parquet`` — one row per ``(scheme, pool, model)`` with
    the pool's ``wac / wam / upb / n_loans`` (identical across models, verified in M15a) and a 12-month
    monthly ``SMM`` path; models = ``empirical / logit / ensemble / realized``.
  * ``pool.cashflow_engine`` — the hermetic level-pay pass-through (closed-form tested to ~1e-8).

For each pool × model: run the engine on the model's SMM path → **price** (per 100 face) and **WAL**
(years); annualise the path to **CPR** (``1−(1−SMM̄)¹²`` on the 12-month-mean SMM, the M15a convention).
Errors are *model-implied minus realized-SMM* valuation through the **same** engine (same WAC/WAM/UPB and
the same WAC−25bp discount curve), so all price dispersion isolates the prepayment-model component (§5.2).

Sign convention (errors = model − realized): on a premium pass-through a model that **under-predicts**
prepayment (model CPR < realized CPR) over-values the pool (price too high) and dates the cashflows too
late (WAL too long) → ``cpr_err < 0`` but ``price_err > 0`` and ``wal_err > 0``.

Artifacts:
  T5.1  ``outputs/tables/pool_level/t_m15_econ_errors.{md,csv,json}`` — mean ``|error|`` and signed bias by
        model × metric{cpr,wal,price} × anchor, **both schemes**; the headline is the char-bucket
        ensemble-vs-logit reduction in mean ``|price error|``, **reported per anchor** (not pooled — the
        COVID anchor inverts and a pooled mean would hide it).
  F5.2  ``outputs/figures/pool_level/F5.2_price_error_buckets_k{k}.{png,pdf}`` — UPB-weighted signed price
        error over FICO × original-rate-quartile cells (LTV collapsed), empirical/logit/ensemble side by
        side; the linear model's errors concentrate in the high-incentive (top rate-quartile) cells — the
        pricing mirror of ``02``'s S-curve.
  per-pool detail ``outputs/tables/pool_level/econ_errors_k{k}.parquet``; summary + Accept evidence
  ``models/nn/full/pool_m15b_summary.json``.

Run (Mac, SSD mounted; seconds — pure post-processing of the parquet, no GPU/torch):
  ``.venv/bin/python -m floan.model.economics``            # all 5 anchors → T5.1 + F5.2 + summary
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from floan.model import config
from floan.model import pool as PL
from floan.model import pools as PP

ANCHORS = list(PL.E.KEY_WINDOWS)                 # 2015, 2019, 2020, 2023, 2025 → Dec(k−1)
HORIZON = PL.HORIZON                             # 12
SMM_COLS = [f"smm_h{h:02d}" for h in range(1, HORIZON + 1)]

# Models carried in the M15a parquet (the single NN was not scored for SMM paths; realized is the
# valuation reference). HEADLINE is the §5 comparison set; the table/figure lead with it.
REALIZED = "realized"
HEADLINE = ("empirical", "logit", "ensemble")
MODEL_LABELS = PP.MODEL_LABELS

# The three §5.3 error "outcomes" (metric axis of T5.1): per-pool error column, label, fmt, unit.
METRICS = (
    ("price", "Price error (per 100 face)", "{:.3f}", "/100"),
    ("cpr",   "CPR error (pp)",             "{:.2f}", "pp"),
    ("wal",   "WAL error (months)",         "{:.2f}", "mo"),
)
SCHEMES = ("char", "random")

# FICO / LTV band labels for F5.2 (mirror pools.char_cell_ids: raw = fb*100 + rb*10 + lb).
FICO_LABELS = ("<680", "680–719", "720–759", "≥760")
LTV_LABELS = ("≤80", "80–90", ">90")
RATE_LABELS = ("Q1\n(low rate)", "Q2", "Q3", "Q4\n(high inc.)")
MERGED_CELL = -1                                 # pools.char_cell_ids catch-all (thin cells)


# ---------------------------------------------------------------------------
def _tabdir() -> Path:
    d = config.OUTPUTS / "tables" / "pool_level"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _figdir() -> Path:
    d = config.OUTPUTS / "figures" / "pool_level"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cpr(smm_path: np.ndarray) -> float:
    """Annualised CPR from a monthly SMM path: ``1−(1−SMM̄)¹²`` on the 12-month mean SMM — the §5.1
    headline unit and the exact M15a sanity convention (``smm_paths._cpr`` on the mean)."""
    return 1.0 - (1.0 - float(smm_path.mean())) ** 12


def per_pool_econ(k: int, df: pl.DataFrame | None = None, *,
                  horizons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """Run the cashflow engine on every ``(scheme, pool, model)`` SMM path at anchor k and attach the
    realized-SMM valuation + the three error columns. One row per ``(scheme, pool, model≠realized)``.

    ``df`` defaults to the M15a ``smm_paths_k{k}.parquet``; an in-memory frame may be passed instead
    (hermetic tests). The frame must carry ``scheme/pool/model/n_loans/wac/wam/upb`` + ``smm_h01..12``
    and a ``realized`` model per ``(scheme, pool)``.

    ``horizons`` (M24): when given (e.g. ``(1,3,6,12)``), the SMM path is **truncated at month H** and
    constant-extrapolated past it (the cashflow engine already extrapolates a short vector — §4 item 1 /
    M15 semantics), pricing each pool at every H and tagging the output with an ``h`` column. The
    realized reference is truncated at the **same** H, so errors stay model−realized per H. ``None``
    (default) prices the full 12-month path with **no** ``h`` column — byte-identical to M15."""
    if df is None:
        df = pl.read_parquet(_tabdir() / f"smm_paths_k{k}.parquet")
    models = [m for m in df.get_column("model").unique().to_list() if m != REALIZED]
    hs = list(horizons) if horizons is not None else None

    # Engine outputs per row, vectorised over rows by a Python loop (≈2k pools × ~330 mo = cheap).
    # H = the full path (M15) or each truncation point (M24); smm[:H] is constant-extrapolated.
    recs: list[dict] = []
    for r in df.iter_rows(named=True):
        smm_full = np.array([r[c] for c in SMM_COLS], dtype=np.float64)
        for h in (hs if hs is not None else [HORIZON]):
            smm = smm_full[:h]
            cf = PL.cashflow_engine(r["wac"], r["wam"], r["upb"], smm)
            rec = {"scheme": r["scheme"], "pool": r["pool"], "model": r["model"],
                   "n_loans": r["n_loans"], "wac": r["wac"], "wam": r["wam"], "upb": r["upb"],
                   "cpr": _cpr(smm), "wal": cf["wal"], "price": cf["price"]}
            if hs is not None:
                rec["h"] = h
            recs.append(rec)
    econ = pl.DataFrame(recs)

    # Realized valuation per (scheme, pool[, h]) → broadcast as the reference; errors = model − realized.
    join_keys = ["scheme", "pool"] + (["h"] if hs is not None else [])
    real = (econ.filter(pl.col("model") == REALIZED)
            .select([*join_keys,
                     pl.col("cpr").alias("cpr_real"), pl.col("wal").alias("wal_real"),
                     pl.col("price").alias("price_real")]))
    out = (econ.filter(pl.col("model").is_in(models))
           .join(real, on=join_keys, how="left")
           .with_columns(
               ((pl.col("cpr") - pl.col("cpr_real")) * 100.0).alias("cpr_err"),     # pp
               ((pl.col("wal") - pl.col("wal_real")) * 12.0).alias("wal_err"),       # months
               (pl.col("price") - pl.col("price_real")).alias("price_err"),          # per 100 face
           )
           .with_columns(pl.lit(k).alias("k"), pl.lit(f"Dec{k-1}").alias("anchor")))
    return out


def _decode_char(pool_ids: np.ndarray):
    """Decode pools.char_cell_ids: raw = fb·100 + rb·10 + lb (MERGED=-1 → masked NaN)."""
    pid = pool_ids.astype(np.int64)
    merged = pid == MERGED_CELL
    fb = np.where(merged, -1, pid // 100)
    rb = np.where(merged, -1, (pid // 10) % 10)
    lb = np.where(merged, -1, pid % 10)
    return fb, rb, lb, merged


# ---------------------------------------------------------------------------
# T5.1 — mean |error| and signed bias by model × metric × anchor (both schemes)
# ---------------------------------------------------------------------------
def aggregate(per_anchor: dict[int, pl.DataFrame]) -> dict:
    """Aggregate per-pool errors to T5.1 cells + the per-anchor headline price-error reduction."""
    rows: list[dict] = []          # one record per (anchor, scheme, metric, model)
    cells: dict = {}               # cells[k][scheme][metric][model] = (mean_abs, bias, n)
    for k, df in sorted(per_anchor.items()):
        cells[k] = {}
        for scheme in SCHEMES:
            cells[k][scheme] = {}
            sub = df.filter(pl.col("scheme") == scheme)
            for mkey, _lab, _fmt, _u in METRICS:
                cells[k][scheme][mkey] = {}
                for m in HEADLINE:
                    e = sub.filter(pl.col("model") == m).get_column(f"{mkey}_err").to_numpy()
                    if e.size == 0:
                        continue
                    mae, bias = float(np.abs(e).mean()), float(e.mean())
                    cells[k][scheme][mkey][m] = (mae, bias, int(e.size))
                    rows.append({"anchor": f"Dec{k-1}", "k": k, "scheme": scheme, "metric": mkey,
                                 "model": m, "mean_abs_err": mae, "bias": bias, "n_pools": int(e.size)})

    # Headline: ensemble-vs-logit reduction in mean |price error|, per anchor, per scheme.
    headline: dict = {}
    for scheme in SCHEMES:
        headline[scheme] = {}
        for k in sorted(per_anchor):
            pe = cells[k][scheme]["price"]
            if "logit" in pe and "ensemble" in pe:
                lo, en = pe["logit"][0], pe["ensemble"][0]
                red = (1.0 - en / lo) * 100.0 if lo > 0 else float("nan")
                headline[scheme][k] = {"logit_mae": lo, "ensemble_mae": en, "reduction_pct": red}
    return {"rows": rows, "cells": cells, "headline": headline}


def write_table(agg: dict) -> None:
    d = _tabdir()
    (d / "t_m15_econ_errors.json").write_text(json.dumps(
        {"rows": agg["rows"], "headline": {s: {str(k): v for k, v in h.items()}
                                           for s, h in agg["headline"].items()},
         "note": "errors = model − realized-SMM valuation through pool.cashflow_engine "
                 "(same WAC/WAM/UPB, WAC−25bp curve); CPR=1−(1−SMM̄)^12; WAL months; price /100 face."},
        indent=2))

    csv = ["anchor,k,scheme,metric,model,mean_abs_err,bias,n_pools"]
    for r in agg["rows"]:
        csv.append(f"{r['anchor']},{r['k']},{r['scheme']},{r['metric']},{r['model']},"
                   f"{r['mean_abs_err']:.6f},{r['bias']:.6f},{r['n_pools']}")
    (d / "t_m15_econ_errors.csv").write_text("\n".join(csv) + "\n")

    cells, head = agg["cells"], agg["headline"]
    ks = sorted(cells)
    md = ["### T5.1 (M15b) — pool economic error: mean |error| and signed bias by model × metric × anchor",
          "",
          "*Errors = model-implied minus realized-SMM valuation through the same level-pay engine "
          "(`pool.cashflow_engine`: identical WAC/WAM/UPB and a fixed WAC−25bp discount curve per pool, "
          "so price dispersion isolates the prepayment model — §5.2). CPR error in pp; WAL error in "
          "months; price error per 100 face. Each cell is `mean|err| (bias)` over the scheme's pools.*",
          ""]

    # Headline block first — the contribution number, per anchor, both schemes.
    md += ["#### Headline — ensemble-vs-logit reduction in mean |price error| (per anchor)", "",
           "*The §5 contribution number. Reported per anchor, never pooled: the Dec2019 COVID anchor "
           "inverts under frozen-t0 macro (both models blind to the Mar-2020 rate collapse), and a "
           "pooled mean would average that inversion away. Characteristic buckets are the headline "
           "(the interpretable cross-pool metric, as in T4.2); random pools are near-homogeneous draws "
           "so their aggregate price error is a level-bias near-tie — the value of nonlinearity lives "
           "in the cross-bucket discrimination.*", ""]
    for scheme in SCHEMES:
        sname = "Characteristic buckets" if scheme == "char" else "Random pools (500×1000)"
        md += [f"**{sname}**", "",
               "| Anchor | Logit mean&#124;price err&#124; | Ensemble mean&#124;price err&#124; | "
               "Ens-vs-Logit reduction |", "|---|---|---|---|"]
        for k in ks:
            h = head[scheme].get(k)
            if h is None:
                md.append(f"| Dec{k-1} | — | — | — |"); continue
            md.append(f"| Dec{k-1} | {h['logit_mae']:.3f} | {h['ensemble_mae']:.3f} | "
                      f"{h['reduction_pct']:+.0f}% |")
        md.append("")

    # Full T5.1 — per scheme × metric, headline models as columns, anchors as rows.
    for scheme in SCHEMES:
        sname = "Characteristic buckets" if scheme == "char" else "Random pools (500×1000)"
        md += [f"#### {sname}", ""]
        for mkey, lab, fmt, _u in METRICS:
            md += [f"**{lab}**", "",
                   "| Anchor | " + " | ".join(MODEL_LABELS[m] for m in HEADLINE) + " |",
                   "|" + "---|" * (1 + len(HEADLINE))]
            for k in ks:
                c = cells[k][scheme][mkey]
                vals = " | ".join(
                    (f"{fmt.format(c[m][0])} ({c[m][1]:+.2f})" if m in c else "—") for m in HEADLINE)
                md.append(f"| Dec{k-1} | {vals} |")
            md.append("")
    (d / "t_m15_econ_errors.md").write_text("\n".join(md) + "\n")
    print(f"wrote {d/'t_m15_econ_errors.md'} (+ .csv/.json)")


# ---------------------------------------------------------------------------
# F5.2 — signed price error by FICO × original-rate-quartile bucket (LTV collapsed, UPB-weighted)
# ---------------------------------------------------------------------------
def fig_price_error_buckets(df_k: pl.DataFrame, k: int, *, suffix: str = "",
                            models: tuple[str, ...] | None = None,
                            labels: dict[str, str] | None = None) -> Path:
    """``suffix`` (M24) tags the output filename, e.g. ``_h06`` for the per-horizon F5.2 panels —
    the M15 call (no suffix) is unchanged. ``models``/``labels`` (M27b) override the panel set and
    titles for the GRU comparison panel; both default to the §5 ``HEADLINE`` set + ``MODEL_LABELS``,
    so the M15/M24 calls are byte-identical."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = HEADLINE if models is None else tuple(models)
    lab = MODEL_LABELS if labels is None else labels

    char = df_k.filter((pl.col("scheme") == "char") & (pl.col("pool") != MERGED_CELL))
    pid = char.get_column("pool").to_numpy()
    fb, rb, _lb, _merged = _decode_char(pid)
    upb = char.get_column("upb").to_numpy()
    perr = char.get_column("price_err").to_numpy()
    model = char.get_column("model").to_numpy()

    nF, nR = len(FICO_LABELS), len(RATE_LABELS)
    grids = {}
    for m in panels:
        g = np.full((nF, nR), np.nan)
        sel = model == m
        for i in range(nF):
            for j in range(nR):
                cell = sel & (fb == i) & (rb == j)
                w = upb[cell]
                if w.sum() > 0:
                    g[i, j] = float(np.sum(w * perr[cell]) / w.sum())   # UPB-weighted signed price err
        grids[m] = g

    vmax = max(1e-6, float(np.nanmax([np.nanmax(np.abs(g)) for g in grids.values()])))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4),
                             sharey=True)
    if len(panels) == 1:
        axes = [axes]
    im = None
    for ax, m in zip(axes, panels):
        g = grids[m]
        im = ax.imshow(g, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(nR)); ax.set_xticklabels(RATE_LABELS, fontsize=8)
        ax.set_yticks(range(nF)); ax.set_yticklabels(FICO_LABELS, fontsize=8)
        ax.set_title(lab.get(m, m), fontsize=11)
        ax.set_xlabel("orig-rate quartile")
        for i in range(nF):
            for j in range(nR):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                            color="black")
    axes[0].set_ylabel("FICO band")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="signed price error (/100 face)")
    fig.suptitle(f"F5.2 — pool price error by characteristic bucket  ·  anchor Dec{k-1} (k={k})\n"
                 "UPB-weighted, LTV collapsed; +ve = model over-values (under-predicts prepay). "
                 "High-incentive = top orig-rate quartile (right).", y=1.06, fontsize=10)
    p = _figdir() / f"F5.2_price_error_buckets_k{k}{suffix}"
    fig.savefig(f"{p}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{p}.pdf", bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
def run(anchors: list[int] | None = None) -> dict:
    anchors = anchors or ANCHORS
    per_anchor: dict[int, pl.DataFrame] = {}
    for k in anchors:
        df = per_pool_econ(k)
        df.write_parquet(_tabdir() / f"econ_errors_k{k}.parquet")
        per_anchor[k] = df
        # sanity: realized-vs-realized is the zero reference (engine reproduces it exactly)
        rr = (pl.read_parquet(_tabdir() / f"smm_paths_k{k}.parquet")
              .filter(pl.col("model") == REALIZED))
        finite = bool(np.isfinite(df.select(["price", "wal", "cpr"]).to_numpy()).all())
        print(f"  [k{k}] Dec{k-1}: {df.height} pool×model rows "
              f"({df.filter(pl.col('scheme')=='char').get_column('pool').n_unique()} char cells + "
              f"{df.filter(pl.col('scheme')=='random').get_column('pool').n_unique()} random pools) · "
              f"all price/WAL/CPR finite={finite} · realized pools={rr.height}", flush=True)

    agg = aggregate(per_anchor)
    write_table(agg)
    figs = [str(fig_price_error_buckets(per_anchor[k], k)) for k in anchors]
    print(f"wrote {len(figs)} F5.2 figures → {_figdir()}")

    # ---- Accept verification: T5.1/F5.2 exist for ensemble vs logit at ≥3 regime-spanning anchors ----
    head = agg["headline"]["char"]
    spanning = [k for k in anchors if k in head]                 # all carry logit+ensemble
    summary = {
        "anchors": anchors, "n_anchors_ens_vs_logit": len(spanning),
        "headline_price_error_reduction_char": {f"Dec{k-1}": head[k] for k in spanning},
        "headline_price_error_reduction_random": {f"Dec{k-1}": v
                                                  for k, v in agg["headline"]["random"].items()},
        "t5_1": str(_tabdir() / "t_m15_econ_errors.md"),
        "f5_2_figs": figs,
        "per_pool_parquet": [str(_tabdir() / f"econ_errors_k{k}.parquet") for k in anchors],
        "accept_ge3_regime_anchors": len(spanning) >= 3,
    }
    (config.MODELS / "nn" / "full" / "pool_m15b_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== M15b headline — char-bucket ens-vs-logit reduction in mean |price error| (per anchor) ===")
    for k in spanning:
        h = head[k]
        print(f"  Dec{k-1}: logit {h['logit_mae']:.3f} → ensemble {h['ensemble_mae']:.3f} "
              f"per-100  ({h['reduction_pct']:+.0f}%)")
    print(f"\nAccept: ensemble-vs-logit T5.1/F5.2 at {len(spanning)} regime-spanning anchors "
          f"(≥3 required) → {'PASS' if len(spanning) >= 3 else 'FAIL'}")
    return summary


if __name__ == "__main__":
    PL.config.require_drive()
    run()
