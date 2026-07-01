"""M14 — Pools + predictions + realized outcomes (``03_POOL_LEVEL §2 / §4``).

Consumes the M13 roll-forward engine (:func:`pool.roll_forward`) to turn the frozen
loan-level models into **pool-level** predicted counts, then compares them to the
**realized** 12-month outcomes read straight from the eval pool. Two pool schemes
(``§2``):

  1. **Characteristic buckets** — the t0-alive ``current`` loans partitioned by
     FICO × original-rate × LTV (4×4×3 = 48 cells; cells below ``MIN_CELL`` loans are
     merged into one catch-all so the partition stays exhaustive — exact reconciliation).
  2. **Random pools** — ``B = 500`` pools of ``N = 1,000`` loans drawn at random from the
     same population (``§2.2`` device for a predicted-vs-realized scatter without bucketing
     artefacts). Membership is seeded ⇒ reproducible.

Per pool, per model, per outcome:

  * **Predicted count** = Σ_i P(event ≤ 12m | i)            (``§4`` formula);
    **90% interval** = μ ± 1.645·σ with the Poisson-binomial normal approximation
    μ = Σ p_i, σ² = Σ p_i(1−p_i) (loans conditionally independent under the model, ``§4``).
  * **Realized count** = Σ_i 1{event observed in (t0, t0+12]} from the panel (eval pool).

Outcomes (``§1``), consistent with the M13 chains:
  * **prepaid ≤ 12m**  — model: composed ``p[prepaid]``; realized: ``state_next == prepaid``
    at any of the 12 monthly transitions from t0.
  * **60+ dpd ≤ 12m**  — model: first-passage ``p_fp[dpd_60]``; realized:
    ``state_next ∈ {dpd_60, dpd_90plus, foreclosure, REO}`` at any of those transitions
    (monotone delinquency ⇒ reaching any deeper state passes through 60+).

**Population (both schemes):** the t0-alive ``current`` loans (``§2`` — buckets partition the
current loans; random pools draw "from the same population"), restricted to non-null
FICO/rate/LTV so a single population serves both schemes and the reconciliation is exact
(the dropped fraction is logged; ≈0.1%, FICO-only).

**Anchors:** the five ensemble-bearing key windows (``02 §6``) k ∈ {2015, 2019, 2020, 2023,
2025} → t0 ∈ {2014-12, 2018-12, 2019-12, 2022-12, 2024-12}. This covers the spec minimum
(2019-12, 2022-12, 2024-12) and adds 2014-12 (calm) and 2018-12 (late-cycle) for regime span;
every anchor carries an ensemble, so the Accept "ensemble vs logit vs empirical at every
anchor run" holds at all five. Non-key windows have no ensemble and so cannot serve that
headline comparison; the five key windows already span calm → COVID → rate-shock.

Realized-outcome window = ``config.test_bounds(k) = [Dec(k−1), Dec(k))`` — exactly the 12
monthly transitions composed by the roll-forward, so predicted and realized are horizon-aligned.

Run (GPU box; the full eval pool lives there):
    .venv/bin/python -m floan.model.pools --device cuda                 # all 5 anchors
    .venv/bin/python -m floan.model.pools --k 2020 --device cuda        # one anchor
    .venv/bin/python -m floan.model.pools --k 2020 --max-loans 50000    # quick smoke
Hermetic unit tests (no SSD/GPU): ``test_pools.py``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import data as D
from floan.model import pool as PL
from floan.model import torch_common as tc
from floan.model import train as T

VARIANT = "full"
ANCHOR_WINDOWS = list(PL.E.KEY_WINDOWS)          # the 5 ensemble-bearing key windows
HORIZON = PL.HORIZON                             # 12

# Pool schemes (§2).
MIN_CELL = 2000                                  # merge characteristic cells below this
RANDOM_B = 500                                   # number of random pools
RANDOM_N = 1000                                  # loans per random pool
SEED = 0                                         # pool-membership seed (reproducibility Accept)
Z90 = 1.6448536269514722                         # Normal 90% two-sided half-width

# Characteristic-bucket edges (§2.1). FICO + LTV are fixed credit-box bands (stable across
# anchors, interpretable); the original-rate axis uses per-anchor quartiles so cells stay
# populated across regimes (2015 coupons ≠ 2023 coupons).
FICO_EDGES = (680, 720, 760)                     # → 4 bands: <680, 680–719, 720–759, ≥760
LTV_EDGES = (80.0, 90.0)                         # → 3 bands: ≤80, 80–90, >90 (right-open at 80/90)
RATE_QUANTILES = (0.25, 0.50, 0.75)              # → 4 per-anchor bands

# Outcomes and the realized-side state sets.
OUTCOMES = ("prepaid", "dpd60p")                 # column suffixes match pool.roll_forward
SIXTY_PLUS = ("dpd_60", "dpd_90plus", "foreclosure", "REO")   # realized 60+ set (first passage)

# Models: the M14 Accept headline is ensemble vs logit vs empirical; the single NN rides along.
MODELS = ("empirical", "logit", "nn", "ensemble")
HEADLINE = ("empirical", "logit", "ensemble")
MODEL_LABELS = {"empirical": "Empirical", "logit": "Logit", "nn": "Best NN",
                "ensemble": "Ensemble ×8"}


# ===========================================================================
# Per-loan master table — predictions (M13) ⋈ bucket features ⋈ realized outcomes
# ===========================================================================
def _anchor_features(k: int) -> pl.DataFrame:
    """The t0-alive ``current`` loans with their bucketing covariates, keyed by loan id.
    One row per loan (anchor month has a single row per loan)."""
    t0 = config._dec(k - 1)
    pool_dir, _ = D.window_spec(VARIANT, k, "test")
    return (pl.scan_parquet(str(pool_dir / "part-*.parquet"))
            .filter((pl.col("period_ym") == t0) & (pl.col("state") == "current"))
            .select(pl.col("Loan Identifier"),
                    pl.col("fico_orig").cast(pl.Float64).alias("fico"),
                    pl.col("Original Interest Rate").cast(pl.Float64).alias("rate"),
                    pl.col("Original Loan-to-Value (LTV)").cast(pl.Float64).alias("ltv"))
            .collect())


def _realized(k: int) -> pl.DataFrame:
    """Realized 12-month outcome flags per loan over ``test_bounds(k) = [t0, t0+12)`` — the
    same 12 monthly transitions the roll-forward composes. One row per loan that has any
    transition in the window."""
    pool_dir, (lo, hi) = D.window_spec(VARIANT, k, "test")
    return (pl.scan_parquet(str(pool_dir / "part-*.parquet"))
            .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
            .group_by("Loan Identifier")
            .agg((pl.col("state_next") == "prepaid").max().alias("realized_prepaid"),
                 (pl.col("state_next").is_in(SIXTY_PLUS)).max().alias("realized_dpd60p"))
            .collect())


def per_loan_table(k: int, device, *, max_loans: int | None = None,
                   chunk: int = PL.DEFAULT_CHUNK) -> pl.DataFrame:
    """Master per-loan table for the pool population: roll-forward predictions (every model,
    both outcomes) joined to the bucketing covariates and the realized flags, restricted to
    ``current``-origin loans with non-null FICO/rate/LTV."""
    rf = PL.roll_forward(k, device, chunk=chunk, max_loans=max_loans, horizon=HORIZON)
    preds = rf["result"]                                   # Loan Identifier, origin, {m}_{o}_12m
    model_names = rf["model_names"]

    feats = _anchor_features(k)
    real = _realized(k)

    n_alive = preds.height
    df = (preds.filter(pl.col("origin") == "current")
               .join(feats, on="Loan Identifier", how="inner")
               .join(real, on="Loan Identifier", how="left"))
    # A current loan alive at t0 always has its t0 transition row in the window, so realized
    # flags are never null in practice; coalesce defensively.
    df = df.with_columns(pl.col("realized_prepaid").fill_null(False),
                         pl.col("realized_dpd60p").fill_null(False))
    n_current = df.height
    df = df.drop_nulls(subset=["fico", "rate", "ltv"])
    df = df.with_columns(pl.col(["realized_prepaid", "realized_dpd60p"]).cast(pl.Int64))

    df = df.sort("Loan Identifier")                        # stable order for seeded sampling
    meta = {"k": k, "t0": config._dec(k - 1), "n_alive": n_alive, "n_current": n_current,
            "n_pop": df.height, "n_dropped_nullbucket": n_current - df.height,
            "model_names": [m for m in model_names], "rf_n_loans": rf["n_loans"]}
    return df, meta


# ===========================================================================
# Pool construction — characteristic buckets (§2.1) and random pools (§2.2)
# ===========================================================================
def char_cell_ids(df: pl.DataFrame) -> tuple[np.ndarray, dict]:
    """Assign each loan a characteristic-bucket id (FICO × rate × LTV). Cells below
    ``MIN_CELL`` loans are merged into a single catch-all (id ``MERGED``), preserving an
    exhaustive partition (exact realized reconciliation). Returns ``(cell_id[n], info)``."""
    fico = df.get_column("fico").to_numpy()
    rate = df.get_column("rate").to_numpy()
    ltv = df.get_column("ltv").to_numpy()
    fb = np.digitize(fico, FICO_EDGES)                          # 0..3
    lb = np.digitize(ltv, [LTV_EDGES[0] + 1e-9, LTV_EDGES[1] + 1e-9])   # 0..2 (≤80 / 80–90 / >90)
    r_edges = np.quantile(rate, RATE_QUANTILES)
    rb = np.digitize(rate, r_edges)                            # 0..3
    raw = (fb * 100 + rb * 10 + lb).astype(np.int64)

    vals, counts = np.unique(raw, return_counts=True)
    thin = set(int(v) for v, c in zip(vals, counts) if c < MIN_CELL)
    MERGED = -1
    cell = np.where(np.isin(raw, list(thin)), MERGED, raw)
    n_cells = int(np.unique(cell).shape[0])
    info = {"rate_edges": [float(x) for x in r_edges], "n_raw_cells": int(vals.shape[0]),
            "n_thin_merged": len(thin), "n_cells": n_cells, "merged_id": MERGED,
            "n_in_merged": int((cell == MERGED).sum())}
    return cell, info


def random_pool_index(n: int, k: int, *, B: int = RANDOM_B, N: int = RANDOM_N,
                      seed: int = SEED) -> np.ndarray:
    """``[B, N]`` integer membership matrix: ``B`` random pools of ``N`` loans drawn without
    replacement *within* a pool from the ``n`` population rows. Independent across pools (a
    loan may recur across pools — the paper's random-pool device). Seeded on ``(seed, k)`` so
    each anchor's pools are distinct but exactly reproducible."""
    rng = np.random.default_rng([seed, k])
    if n < N:                                                  # safety; never hit at full scale
        return np.stack([rng.choice(n, N, replace=True) for _ in range(B)])
    return np.stack([rng.choice(n, N, replace=False) for _ in range(B)])


# ===========================================================================
# Counts + intervals + metrics
# ===========================================================================
def _p(df: pl.DataFrame, model: str, outcome: str) -> np.ndarray:
    return df.get_column(f"{model}_{outcome}_12m").to_numpy()


def _real(df: pl.DataFrame, outcome: str) -> np.ndarray:
    return df.get_column(f"realized_{outcome}").to_numpy().astype(np.float64)


def pool_counts_random(df: pl.DataFrame, idx: np.ndarray, model: str, outcome: str) -> dict:
    """Predicted count + Normal interval and realized count for every random pool, vectorised
    over the ``[B, N]`` membership matrix."""
    p = _p(df, model, outcome)
    r = _real(df, outcome)
    pm = p[idx]                                                # [B, N]
    mu = pm.sum(axis=1)
    sd = np.sqrt((pm * (1.0 - pm)).sum(axis=1))
    realized = r[idx].sum(axis=1)
    return {"pred": mu, "sd": sd, "lo": mu - Z90 * sd, "hi": mu + Z90 * sd, "realized": realized}


def pool_counts_char(df: pl.DataFrame, cell: np.ndarray, model: str, outcome: str) -> dict:
    """Predicted/realized counts per characteristic cell (an exhaustive partition)."""
    p = _p(df, model, outcome)
    r = _real(df, outcome)
    order = np.argsort(cell, kind="stable")
    cs = cell[order]
    uniq, start = np.unique(cs, return_index=True)
    pred = np.add.reduceat(p[order], start)
    var = np.add.reduceat((p * (1.0 - p))[order], start)
    realized = np.add.reduceat(r[order], start)
    n = np.diff(np.append(start, cs.shape[0]))
    return {"cell": uniq, "n": n, "pred": pred, "sd": np.sqrt(var),
            "lo": pred - Z90 * np.sqrt(var), "hi": pred + Z90 * np.sqrt(var),
            "realized": realized}


def _r2_rmse(pred: np.ndarray, realized: np.ndarray) -> tuple[float, float]:
    """R² (1 − SS_res/SS_tot about the realized mean) and RMSE of predicted vs realized."""
    resid = pred - realized
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sstot = float(np.sum((realized - realized.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / sstot) if sstot > 0 else float("nan")
    return r2, rmse


def _coverage(c: dict) -> float:
    return float(np.mean((c["realized"] >= c["lo"]) & (c["realized"] <= c["hi"])))


# ===========================================================================
# Figures — F4.1-style predicted-vs-realized scatter (random pools), per outcome × anchor
# ===========================================================================
def _figdir() -> Path:
    d = config.OUTPUTS / "figures" / "pool_level"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _oname(outcome: str) -> str:
    return "prepaid ≤ 12m" if outcome == "prepaid" else "60+ dpd ≤ 12m"


def _scatter_panel(counts, k, outcome, metrics, scheme, title_tail, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(counts)
    # Independent axes per panel: a shared y-axis is sized to whichever model is
    # drawn last and clips any model whose predictions fall outside that range —
    # e.g. the empirical matrix over-predicts pool counts to a near-constant far
    # above the well-calibrated models, and was clipped to an empty panel.
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.8))
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        c = counts[m]
        ax.scatter(c["realized"], c["pred"], s=8, alpha=0.35, edgecolors="none")
        # x-axis on the realised range (shared across panels); y-axis per model,
        # so an over-predicting model extends only upward, not sideways.
        xmax = float(c["realized"].max()) * 1.05 or 1.0
        ymax = max(float(c["realized"].max()), float(c["pred"].max())) * 1.05 or 1.0
        ax.plot([0, max(xmax, ymax)], [0, max(xmax, ymax)], ls="--", c="grey", lw=1)
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        mm = metrics[m][outcome][scheme]
        ax.set_title(f"{MODEL_LABELS[m]}\nR²={mm['r2']:.3f}  RMSE={mm['rmse']:.1f}")
        ax.set_xlabel("Realized count"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted count")
    fig.suptitle(f"Pool predicted vs realized — {_oname(outcome)}  ·  anchor Dec{k-1} (k={k}), "
                 + title_tail, y=1.04)
    fig.tight_layout()
    p = _figdir() / fname
    fig.savefig(f"{p}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{p}.pdf", bbox_inches="tight")
    plt.close(fig)
    return p


def fig_scatter(df: pl.DataFrame, idx: np.ndarray, k: int, outcome: str,
                metrics: dict, *, scheme: str = "random") -> Path:
    """F4.1-style scatter over the random pools (§4)."""
    models = [m for m in HEADLINE if f"{m}_{outcome}_12m" in df.columns]
    counts = {m: pool_counts_random(df, idx, m, outcome) for m in models}
    return _scatter_panel(counts, k, outcome, metrics, scheme,
                          f"{RANDOM_B} random pools of {RANDOM_N}",
                          f"F_m14_scatter_random_{outcome}_k{k}")


def fig_scatter_char(df: pl.DataFrame, cell: np.ndarray, k: int, outcome: str,
                     metrics: dict) -> Path:
    """Characteristic-bucket scatter — the interpretable cross-pool R² (point area ∝ pool size)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in HEADLINE if f"{m}_{outcome}_12m" in df.columns]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.8),
                             sharex=True, sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        c = pool_counts_char(df, cell, m, outcome)
        ax.scatter(c["realized"], c["pred"], s=np.sqrt(c["n"]) * 0.6, alpha=0.5,
                   edgecolors="none")
        lim = max(float(c["realized"].max()), float(c["pred"].max())) * 1.05 or 1.0
        ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        mm = metrics[m][outcome]["char"]
        ax.set_title(f"{MODEL_LABELS[m]}\nR²={mm['r2']:.3f}  RMSE={mm['rmse']:.1f}")
        ax.set_xlabel("Realized count"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted count")
    fig.suptitle(f"Pool predicted vs realized — {_oname(outcome)}  ·  anchor Dec{k-1} (k={k}), "
                 f"characteristic buckets (≥{MIN_CELL}/cell)", y=1.04)
    fig.tight_layout()
    p = _figdir() / f"F_m14_scatter_char_{outcome}_k{k}"
    fig.savefig(f"{p}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{p}.pdf", bbox_inches="tight")
    plt.close(fig)
    return p


# ===========================================================================
# Table writer — R²/RMSE by model × outcome × anchor (+ coverage)
# ===========================================================================
def _tabdir() -> Path:
    d = config.OUTPUTS / "tables" / "pool_level"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_table(per_anchor: dict) -> None:
    d = _tabdir()
    rows = []                       # flat records for csv/json (both schemes)
    for k in sorted(per_anchor):
        a = per_anchor[k]
        for scheme in ("random", "char"):
            for outcome in OUTCOMES:
                for m in MODELS:
                    if m not in a["metrics"]:
                        continue
                    mm = a["metrics"][m][outcome][scheme]
                    rows.append({"anchor": f"Dec{k-1}", "k": k, "scheme": scheme,
                                 "outcome": outcome, "model": m, "r2": mm["r2"],
                                 "rmse": mm["rmse"], "coverage90": mm["coverage"]})
    (d / "t_m14_pool_counts.json").write_text(json.dumps(
        {"rows": rows, "B": RANDOM_B, "N": RANDOM_N, "min_cell": MIN_CELL, "seed": SEED}, indent=2))

    csv = ["anchor,k,scheme,outcome,model,r2,rmse,coverage90"]
    for r in rows:
        csv.append(f"{r['anchor']},{r['k']},{r['scheme']},{r['outcome']},{r['model']},"
                   f"{r['r2']:.4f},{r['rmse']:.2f},{r['coverage90']:.3f}")
    (d / "t_m14_pool_counts.csv").write_text("\n".join(csv) + "\n")

    # Markdown — per scheme × outcome block, headline models as columns, anchors as rows.
    md = ["### T4.2 (M14) — pool count accuracy: R² / RMSE by model × outcome × anchor", "",
          f"*Per anchor, the t0-alive current loans. Random scheme: {RANDOM_B} pools of "
          f"{RANDOM_N} (seed {SEED}) — R² reflects level bias (homogeneous draws). "
          f"Characteristic scheme: FICO×rate×LTV buckets (≥{MIN_CELL}/cell) — the interpretable "
          f"cross-pool R². R²/RMSE of predicted vs realized 12-month counts; 90% interval "
          f"coverage in the .csv.*", ""]
    for scheme in ("char", "random"):
        sname = ("Characteristic buckets" if scheme == "char"
                 else f"Random pools ({RANDOM_B}×{RANDOM_N})")
        md += [f"#### {sname}", ""]
        for outcome in OUTCOMES:
            md += [f"**{_oname(outcome).capitalize()}**", "",
                   "| Anchor | " + " | ".join(f"{MODEL_LABELS[m]} R²" for m in HEADLINE)
                   + " | " + " | ".join(f"{MODEL_LABELS[m]} RMSE" for m in HEADLINE) + " |",
                   "|" + "---|" * (1 + 2 * len(HEADLINE))]
            for k in sorted(per_anchor):
                a = per_anchor[k]["metrics"]
                r2s = " | ".join(f"{a[m][outcome][scheme]['r2']:.3f}" if m in a else "—"
                                 for m in HEADLINE)
                rmses = " | ".join(f"{a[m][outcome][scheme]['rmse']:.1f}" if m in a else "—"
                                   for m in HEADLINE)
                md.append(f"| Dec{k-1} | {r2s} | {rmses} |")
            md.append("")
    (d / "t_m14_pool_counts.md").write_text("\n".join(md) + "\n")
    print(f"wrote {d/'t_m14_pool_counts.md'} (+ .csv/.json)")


# ===========================================================================
# Per-anchor driver
# ===========================================================================
def run_anchor(k: int, device, *, max_loans: int | None = None,
               chunk: int = PL.DEFAULT_CHUNK) -> dict:
    print(f"\n=== M14 anchor Dec{k-1} (k={k}) ===")
    tk = time.perf_counter()
    df, meta = per_loan_table(k, device, max_loans=max_loans, chunk=chunk)
    print(f"  population: {meta['n_alive']:,} alive → {meta['n_current']:,} current → "
          f"{meta['n_pop']:,} with buckets (dropped {meta['n_dropped_nullbucket']:,} null-bucket)")

    cell, cell_info = char_cell_ids(df)
    idx = random_pool_index(df.height, k)
    models = [m for m in MODELS if f"{m}_prepaid_12m" in df.columns]

    # ---- metrics for both schemes: random pools (§4 F4.1 scatter, homogeneous draws — R²
    #      mostly reflects level bias) and characteristic buckets (real cross-pool signal —
    #      the interpretable R²). Each: R², RMSE, 90% interval coverage. ----
    metrics: dict = {}
    for m in models:
        metrics[m] = {}
        for outcome in OUTCOMES:
            cr = pool_counts_random(df, idx, m, outcome)
            cc = pool_counts_char(df, cell, m, outcome)
            r2r, rmser = _r2_rmse(cr["pred"], cr["realized"])
            r2c, rmsec = _r2_rmse(cc["pred"], cc["realized"])
            metrics[m][outcome] = {
                "r2": r2r, "rmse": rmser, "coverage": _coverage(cr),       # random (back-compat)
                "random": {"r2": r2r, "rmse": rmser, "coverage": _coverage(cr),
                           "pred_mean": float(cr["pred"].mean()),
                           "realized_mean": float(cr["realized"].mean())},
                "char": {"r2": r2c, "rmse": rmsec, "coverage": _coverage(cc),
                         "n_pools": int(cc["cell"].shape[0]),
                         "pred_mean": float(cc["pred"].mean()),
                         "realized_mean": float(cc["realized"].mean())},
                "pred_mean": float(cr["pred"].mean()),
                "realized_mean": float(cr["realized"].mean())}

    # ---- realized reconciliation against the characteristic partition (Accept #2) ----
    recon = {}
    for outcome in OUTCOMES:
        cc = pool_counts_char(df, cell, models[0], outcome)        # realized is model-free
        part_sum = float(cc["realized"].sum())
        direct = float(_real(df, outcome).sum())
        recon[outcome] = {"partition_sum": part_sum, "direct_total": direct,
                          "abs_diff": abs(part_sum - direct), "exact": part_sum == direct,
                          "n_cells": int(cc["cell"].shape[0])}
        print(f"  reconcile {outcome}: Σcells={part_sum:.1f} == direct={direct:.1f} "
              f"(Δ={recon[outcome]['abs_diff']:.3g}) over {recon[outcome]['n_cells']} cells")

    # ---- reproducibility of the seeded random pools (Accept #1) ----
    idx2 = random_pool_index(df.height, k)
    repro = bool(np.array_equal(idx, idx2))
    print(f"  random-pool membership reproducible (seed {SEED}): {repro}")

    # ---- figures: random-pool scatter (§4 F4.1) + characteristic-bucket scatter ----
    figs = {}
    for o in OUTCOMES:
        figs[f"random_{o}"] = str(fig_scatter(df, idx, k, o, metrics, scheme="random"))
        figs[f"char_{o}"] = str(fig_scatter_char(df, cell, k, o, metrics))

    # ---- save per-pool count tables (parquet) for traceability / M15 reuse ----
    pdir = _tabdir()
    rand_records = {"pool": np.arange(idx.shape[0])}
    for m in models:
        for o in OUTCOMES:
            cr = pool_counts_random(df, idx, m, o)
            rand_records[f"{m}_{o}_pred"] = cr["pred"]
            rand_records[f"{m}_{o}_lo"] = cr["lo"]
            rand_records[f"{m}_{o}_hi"] = cr["hi"]
            rand_records[f"{o}_realized"] = cr["realized"]
    pl.DataFrame(rand_records).write_parquet(pdir / f"pools_random_k{k}.parquet")

    # headline: ensemble-vs-logit characteristic-bucket RMSE reduction per outcome (the
    # interpretable scheme; random-pool RMSE reported alongside).
    headline = {}
    for outcome in OUTCOMES:
        if "ensemble" in metrics and "logit" in metrics:
            le, ll = metrics["ensemble"][outcome]["char"]["rmse"], metrics["logit"][outcome]["char"]["rmse"]
            headline[outcome] = {"logit_rmse": ll, "ensemble_rmse": le,
                                 "rmse_reduction_pct": 100.0 * (ll - le) / ll if ll else None}

    for sch in ("char", "random"):
        for outcome in OUTCOMES:
            print(f"  R²({sch:>6}) {outcome:>7}: " + "  ".join(
                f"{m}={metrics[m][outcome][sch]['r2']:.3f}" for m in models))

    return {"k": k, "meta": meta, "cell_info": cell_info, "metrics": metrics,
            "reconciliation": recon, "repro": repro, "headline": headline,
            "figures": figs, "models": models, "wall_sec": time.perf_counter() - tk}


# ===========================================================================
# Accept verification
# ===========================================================================
def verify(per_anchor: dict) -> dict:
    ok = True
    checks = []

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        checks.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    print("\n[1] pool memberships reproducible (seeded)")
    for k in sorted(per_anchor):
        check(f"k={k}: random-pool membership identical across two seeded builds",
              per_anchor[k]["repro"])

    print("\n[2] realized counts reconcile with panel aggregates (characteristic partition)")
    for k in sorted(per_anchor):
        for outcome in OUTCOMES:
            r = per_anchor[k]["reconciliation"][outcome]
            check(f"k={k} {outcome}: Σ cells == direct total ({r['partition_sum']:.0f})",
                  r["exact"], f"Δ={r['abs_diff']:.3g}")

    print("\n[3] predicted-vs-realized scatter + R²/RMSE table — ensemble vs logit vs empirical "
          "at every anchor")
    for k in sorted(per_anchor):
        a = per_anchor[k]
        have_models = all(m in a["metrics"] for m in HEADLINE)
        figs_ok = all(Path(f"{p}.png").exists() and Path(f"{p}.pdf").exists()
                      for p in a["figures"].values())
        check(f"k={k}: empirical+logit+ensemble metrics present and all "
              f"{len(a['figures'])} scatter figures written", have_models and figs_ok)
    tab = config.OUTPUTS / "tables" / "pool_level" / "t_m14_pool_counts.md"
    check("R²/RMSE table written", tab.exists(), str(tab))

    print("\n[4] sanity — metrics finite, counts non-negative, coverage ∈ [0,1], population > 0")
    for k in sorted(per_anchor):
        a = per_anchor[k]
        fin = all(np.isfinite(a["metrics"][m][o]["r2"]) and a["metrics"][m][o]["rmse"] >= 0
                  and 0.0 <= a["metrics"][m][o]["coverage"] <= 1.0
                  and a["metrics"][m][o]["pred_mean"] >= 0 and a["metrics"][m][o]["realized_mean"] >= 0
                  for m in a["metrics"] for o in OUTCOMES)
        check(f"k={k}: all R²/RMSE finite, counts ≥ 0, coverage∈[0,1], n_pop={a['meta']['n_pop']:,}>0",
              fin and a["meta"]["n_pop"] > 0)
    # Informational (NOT pass/fail): the by-anchor ensemble-vs-logit RMSE gap is the regime
    # exhibit itself — nonlinearity helps in some regimes and not others (e.g. the frozen-macro
    # COVID miss dominates loan-level nonlinearity at k=2020), so no ordering is asserted here.
    print("\n  [info] ensemble vs logit pool-count RMSE by anchor (regime exhibit, §6):")
    for k in sorted(per_anchor):
        for o in OUTCOMES:
            h = per_anchor[k]["headline"].get(o)
            if h and h.get("rmse_reduction_pct") is not None:
                print(f"    k={k} {o:>7}: logit {h['logit_rmse']:.1f} → ensemble {h['ensemble_rmse']:.1f} "
                      f"({h['rmse_reduction_pct']:+.1f}% RMSE)")

    print(f"\n{'ALL M14 ACCEPT CRITERIA PASS' if ok else 'SOME CHECKS FAILED'} (variant={VARIANT})")
    return {"all_pass": ok, "checks": checks}


def run(windows: list[int], device, *, max_loans: int | None = None,
        chunk: int = PL.DEFAULT_CHUNK) -> dict:
    t0 = time.perf_counter()
    print(f"=== M14 pools  anchors={windows}  device={device}"
          f"{'  max_loans=' + format(max_loans, ',') if max_loans else ''} ===")

    per_anchor = {k: run_anchor(k, device, max_loans=max_loans, chunk=chunk) for k in windows}
    write_table(per_anchor)
    accept = verify(per_anchor)

    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "variant": VARIANT, "anchors": windows, "git_commit": T._git_commit(),
        "params": {"MIN_CELL": MIN_CELL, "RANDOM_B": RANDOM_B, "RANDOM_N": RANDOM_N,
                   "SEED": SEED, "fico_edges": FICO_EDGES, "ltv_edges": LTV_EDGES,
                   "rate_quantiles": RATE_QUANTILES, "horizon": HORIZON},
        "per_anchor": {k: {kk: vv for kk, vv in v.items() if kk != "figures"}
                       for k, v in per_anchor.items()},
        "figures": {k: v["figures"] for k, v in per_anchor.items()},
        "accept": accept, "wall_sec": time.perf_counter() - t0,
    }
    out = config.MODELS / "nn" / VARIANT / "pool_m14_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out}  [{summary['wall_sec']/60:.1f} min]")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="M14 pools + predictions + realized outcomes.")
    ap.add_argument("--k", type=int, default=None, help="single anchor window (default: all 5 key)")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--chunk", type=int, default=PL.DEFAULT_CHUNK, help="loans per roll-forward chunk")
    ap.add_argument("--max-loans", type=int, default=None, help="cap alive loans (smoke)")
    args = ap.parse_args()
    config.require_drive()
    device = (torch.device("cuda") if args.device == "cuda"
              else tc.resolve_device(args.device))
    windows = [args.k] if args.k else ANCHOR_WINDOWS
    run(windows, device, max_loans=args.max_loans, chunk=args.chunk)


if __name__ == "__main__":
    main()
