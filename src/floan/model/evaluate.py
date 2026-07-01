"""M11 — Evaluation suite (``02_LOAN_LEVEL §7``): one script, every model through it.

Scores the four model families assembled by M5/M10 — the **empirical transition matrix**
(the covariate-free floor), the **multinomial logit**, the **best single NN**, and the
**8-net ensemble** (key windows only) — on the *frozen, unthinned* per-window test slices and
produces the headline exhibits:

* **Table B** — the rolling results: test-year (2015…2025) × model out-of-sample NLL, the
  all-years **pooled** NLL (each test row predicted once by a model that never saw it), and
  the pooled NLL **broken out by origin state** (the gains concentrate in ``current`` rows —
  prepayment). Figure version: NLL by test year, one line per model.
* **AUC table** — per origin state × destination one-vs-rest AUC on the pooled test
  predictions (paper §3.4): logit vs best NN vs ensemble.
* **Calibration** — reliability diagrams for ``current→prepaid`` and ``current→dpd_30``,
  pooled and split 2020–21 vs other years.
* **Rate overlay** — predicted vs realized monthly aggregate transition rates over 2015–2025
  (a Phase-3 sanity input).

The empirical matrix is the only model whose *fit* is recomputed here (it is one weighted
``GROUP BY`` on the window's ``train_pool`` train slice — the importance weight ``1/p_keep``
makes the thinned pool's weighted counts an unbiased estimate of the unthinned panel matrix,
so it needs no panel access; cf. M5 §4 which read the panel directly). Every other model is
read from its committed run folder (``logit/full/k*``, ``nn/full/k*_d3_…``, ``ens_k*_s*``).

Identical-rows contract (the M11 Accept criteria): a window's test slice is loaded **once**
and every model is scored on that one encoding. The logit and NN per-window pipelines
(scaler+vocab, fit deterministically on the same train slice) are asserted byte-identical, so
the single encoding is faithful to each model's committed featurization. The pooled set is the
disjoint union of the 11 test slices (the ``period_ym`` test bounds tile the axis), verified by
a global uniqueness check on ``(loan, period_ym)``.

Outputs:
  ``outputs/tables/loan_level/table_b.{json,csv,md,tex}``     rolling NLL + pooled + by-origin
  ``outputs/tables/loan_level/auc.{json,csv,md,tex}``         one-vs-rest AUC (pooled / key)
  ``outputs/figures/loan_level/F_tableB_nll_by_year.{png,pdf}``
  ``outputs/figures/loan_level/F_calibration_*.{png,pdf}``
  ``outputs/figures/loan_level/F_rate_overlay.{png,pdf}``
  ``outputs/figures/loan_level/F_ensemble_size_curve.{png,pdf}``
  ``models/nn/full/evaluate_summary.json``                    all numbers + Accept evidence

Run (the GPU box; the full eval pool lives there):
    .venv/bin/python -m floan.model.evaluate --device cuda
    .venv/bin/python -m floan.model.evaluate --device cuda --windows 2015 2020   # subset (dev)
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

from floan.model import backtest as B  # LogitEmbNet (the committed full-scale logit parameterisation)
from floan.model import config
from floan.model import data as D
from floan.model import features as F
from floan.model import net as N
from floan.model import torch_common as tc
from floan.model import train as T

VARIANT = "full"
KEY_WINDOWS = (2015, 2019, 2020, 2023, 2025)     # §6: ensemble only on these 5
EVAL_BATCH = 32768
SEED = 0

STATES = F.STATES                                 # canonical 7-state order
ORIGIN_STATES = list(config.ORIGIN_STATES)        # the 4 transient origins
OI = {s: i for i, s in enumerate(ORIGIN_STATES)}  # origin → row index in the 4×7 matrix
SI = F.STATE_INDEX                                 # state → class index (0..6)
PREPAID, DPD30 = SI["prepaid"], SI["dpd_30"]


# ===========================================================================
# Empirical transition matrix — weighted GROUP BY on the train slice (§4)
# ===========================================================================
def empirical_matrix(k: int, alpha: float = 0.5) -> np.ndarray:
    """4×7 row-normalised, Laplace(``alpha``)-smoothed transition matrix for window ``k``,
    estimated from the **weighted** ``train_pool`` train slice. The current→current thinning
    weight (``1/p_keep``) makes ``Σ weight`` per (origin, dest) cell an unbiased estimate of
    the unthinned panel count (M5 §4), so the matrix matches the panel-derived one in
    expectation without needing panel access on the GPU box."""
    pool_dir, (lo, hi) = D.window_spec(VARIANT, k, "train")
    g = (pl.scan_parquet(str(pool_dir / "part-*.parquet"))
           .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
           .group_by("state", "state_next").agg(pl.col("weight").sum().alias("n"))
           .collect())
    counts = np.zeros((len(ORIGIN_STATES), F.N_CLASSES), dtype=np.float64)
    for r in g.iter_rows(named=True):
        if r["state"] in OI and r["state_next"] in SI:
            counts[OI[r["state"]], SI[r["state_next"]]] += r["n"]
    sm = counts + alpha
    return sm / sm.sum(axis=1, keepdims=True)


# ===========================================================================
# Per-window test scoring — load the frozen slice once, score every model on it
# ===========================================================================
@torch.no_grad()
def _resident_probs(model, Tres: dict, device) -> np.ndarray:
    """Softmax class probabilities ``[N, 7]`` (float32) for ``model`` on a GPU-resident split
    (``train.to_resident`` output). float64 softmax then cast — the eval-path numerics of
    ``torch_common``. Works for both :class:`net.MortgageMLP` and :class:`backtest.LogitEmbNet`
    (identical ``forward(cont, cat, binb)`` signature)."""
    model.eval()
    n = int(Tres["y"].shape[0])
    out = np.empty((n, F.N_CLASSES), dtype=np.float32)
    for s in range(0, n, EVAL_BATCH):
        sl = slice(s, s + EVAL_BATCH)
        logits = model(Tres["cont"][sl], Tres["cat"][sl].long(), Tres["bin"][sl])
        out[sl] = torch.softmax(logits.double(), dim=1).float().cpu().numpy()
    return out


def _load_nn(run: Path, device):
    arch = json.loads((run / "metrics.json").read_text())["architecture"]
    model = N.from_arch({kk: arch[kk] for kk in
                         ("n_continuous", "n_binary", "vocab_sizes", "emb_dims",
                          "hidden_dims", "n_classes", "dropout")}).to(device)
    model.load_state_dict(torch.load(run / "best_model.pt", map_location=device))
    return model


def _load_logit(run: Path, scaler: F.Scaler, vocab: F.Vocab, device):
    model = B.LogitEmbNet(len(scaler.cols), len(F.BINARY), list(vocab.vocab_sizes)).to(device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))
    return model


def score_window(k: int, device) -> dict:
    """Score every model on window ``k``'s frozen test slice. Returns per-row arrays (origin,
    y, label_ym, row-key hash) + per-model probability matrices, plus the identical-rows
    evidence (row count, key hash, pipeline-match flag)."""
    nn_run = config.MODELS / "nn" / VARIANT / B._nn_tag(k, dict(config.NN_SELECTED))
    logit_run = config.MODELS / "logit" / VARIANT / f"k{k}"

    # Pipeline chain-of-custody: logit & NN fit scaler+vocab on the SAME train slice
    # (deterministic), so the files must be byte-identical — then one encoding faithfully
    # serves both models (and every ensemble member).
    pipe_match = all(
        (logit_run / f).read_bytes() == (nn_run / f).read_bytes()
        for f in ("scaler.json", "vocab.json"))
    scaler, vocab = F.load_pipeline(nn_run)

    pool_dir, bounds = D.window_spec(VARIANT, k, "test")
    df = F.prepare_raw(D._masked_scan(pool_dir, bounds).collect())
    enc = F.encode_frame(df, scaler, vocab, encode="index")
    n = int(enc["y"].shape[0])
    y = np.asarray(enc["y"]).astype(np.int8)
    origin = (df.select(pl.col("state").replace_strict(
        list(OI), list(OI.values()), default=-1, return_dtype=pl.Int64))
        .to_numpy().reshape(-1)).astype(np.int8)
    label_ym = df.get_column("label_ym").to_numpy().astype(np.int32)
    # Stable per-row key for the global pooled-uniqueness check (M11 Accept #2).
    keyhash = (df.select(pl.struct(["Loan Identifier", "period_ym"]).hash(seed=0))
                 .to_numpy().reshape(-1).astype(np.uint64))

    Tres = T.to_resident(enc, device)
    del enc

    out = {"k": k, "n": n, "y": y, "origin": origin, "label_ym": label_ym,
           "keyhash": keyhash, "pipe_match": pipe_match,
           "test_key_hash": int(np.bitwise_xor.reduce(keyhash ^ y.astype(np.uint64))),
           "probs": {}}

    # Empirical matrix — gather the origin row, no GPU.
    M = empirical_matrix(k)
    emp = M[origin]                                  # [N, 7]
    out["probs"]["empirical"] = emp.astype(np.float32)

    # Logit + single NN — resident inference on the shared encoding.
    out["probs"]["logit"] = _resident_probs(_load_logit(logit_run, scaler, vocab, device),
                                            Tres, device)
    out["probs"]["nn"] = _resident_probs(_load_nn(nn_run, device), Tres, device)

    # Ensemble (key windows): mean of the 8 members' probabilities (paper Fig 7 rule).
    if k in KEY_WINDOWS:
        acc = np.zeros((n, F.N_CLASSES), dtype=np.float64)
        for s in range(B.E.N_MEMBERS):
            mrun = config.MODELS / "nn" / VARIANT / f"ens_k{k}_s{s}"
            acc += _resident_probs(_load_nn(mrun, device), Tres, device)
        out["probs"]["ensemble"] = (acc / B.E.N_MEMBERS).astype(np.float32)

    del Tres
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ===========================================================================
# Metric helpers
# ===========================================================================
def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean −log p[y] (the unweighted §1 out-of-sample NLL; the eval pool has w≡1)."""
    p = probs[np.arange(probs.shape[0]), y]
    return float(-np.log(np.clip(p, 1e-300, None)).mean())


def _auc_one_vs_rest(score: np.ndarray, pos: np.ndarray) -> float | None:
    """One-vs-rest AUC of ``score`` against the binary ``pos`` label; ``None`` if degenerate
    (one class only). Rank-based (Mann–Whitney U statistic with tie-corrected **average**
    ranks), fully vectorised — ``np.unique`` does the (C-level) sort and gives the per-value
    block sizes, so ranks for tied scores are their block midpoint without any Python loop
    (a hard requirement at ~80 M pooled rows)."""
    n = pos.shape[0]
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Average rank (1-based) of every row: block-start (0-based) + (block_size + 1)/2.
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts            # 0-based start index of each value's block
    avg_rank = start + (counts + 1) / 2.0
    sum_pos = avg_rank[inv[pos]].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# Structurally reachable destinations per origin (02 §1 monotone-delinquency rule) —
# the cells worth an AUC (others are ~0 by construction).
REACHABLE = {
    "current":    ["current", "dpd_30", "prepaid"],
    "dpd_30":     ["current", "dpd_30", "dpd_60", "prepaid"],
    "dpd_60":     ["current", "dpd_30", "dpd_60", "dpd_90plus", "prepaid"],
    "dpd_90plus": ["current", "dpd_30", "dpd_60", "dpd_90plus", "foreclosure", "prepaid"],
}


# ===========================================================================
# Table B + by-origin
# ===========================================================================
MODEL_LABELS = {"empirical": "Empirical matrix", "logit": "Logit",
                "nn": "Best NN", "ensemble": "Ensemble ×8"}
MODEL_ORDER = ["empirical", "logit", "nn", "ensemble"]


def build_table_b(per_window: dict, pooled: dict) -> dict:
    """Per-window NLL matrix + pooled all-years NLL + pooled NLL by origin state (§7)."""
    by_year = {}
    for k in sorted(per_window):
        w = per_window[k]
        row = {m: _nll(w["probs"][m], np.asarray(w["y"]).astype(np.int64))
               for m in w["probs"]}
        by_year[k] = row

    y = pooled["y"].astype(np.int64)
    pooled_nll = {m: _nll(pooled["probs"][m], y) for m in pooled["probs"]}
    # ensemble pooled NLL only over the key-window rows where it exists.
    if "ensemble" in pooled["probs_ens"]:
        ek = pooled["ens_mask"]
        pooled_nll["ensemble"] = _nll(pooled["probs_ens"]["ensemble"], y[ek])

    by_origin = {}
    for s, oi in OI.items():
        m = pooled["origin"] == oi
        ys = y[m]
        d = {mm: _nll(pooled["probs"][mm][m], ys) for mm in pooled["probs"]}
        if "ensemble" in pooled["probs_ens"]:
            em = pooled["origin"][pooled["ens_mask"]] == oi
            d["ensemble"] = _nll(pooled["probs_ens"]["ensemble"][em],
                                 y[pooled["ens_mask"]][em])
        by_origin[s] = {"n": int(m.sum()), **d}

    return {"by_year": by_year, "pooled": pooled_nll, "by_origin": by_origin,
            "n_pooled": int(y.shape[0]),
            "n_pooled_ensemble": int(pooled["ens_mask"].sum())}


# ===========================================================================
# AUC table
# ===========================================================================
def build_auc(pooled: dict) -> dict:
    """One-vs-rest AUC per (origin, destination) on the pooled predictions. Logit & best NN
    over all 11 windows; all three (+ensemble) over the key-window pooled rows for an
    apples-to-apples ensemble comparison (§3.4)."""
    y = pooled["y"].astype(np.int64)
    origin = pooled["origin"]
    ens_mask = pooled["ens_mask"]

    def table(models_probs: dict, row_mask: np.ndarray) -> list[dict]:
        # TODO(perf): re-slices each model's full [N,7] prob matrix per (origin,dest,model)
        # cell (`P[row_mask][m]`, a 2.4 GB copy ×~36) → ~14 min at full scale. Precompute
        # per-origin row indices once and index `P[idx, di]` directly. Correct, just slow.
        yy, oo = y[row_mask], origin[row_mask]
        rows = []
        for s, oi in OI.items():
            m = oo == oi
            if m.sum() == 0:
                continue
            ys = yy[m]
            for dest in REACHABLE[s]:
                di = SI[dest]
                pos = (ys == di)
                cell = {"origin": s, "destination": dest, "n": int(m.sum()),
                        "n_pos": int(pos.sum())}
                for name, P in models_probs.items():
                    cell[name] = _auc_one_vs_rest(P[row_mask][m][:, di], pos)
                rows.append(cell)
        return rows

    full = table({"logit": pooled["probs"]["logit"], "nn": pooled["probs"]["nn"]},
                 np.ones(y.shape[0], dtype=bool))
    key_models = {"logit": pooled["probs"]["logit"], "nn": pooled["probs"]["nn"]}
    # ensemble probs are stored only for the key-window rows; expand against the mask.
    ens_full = np.full((y.shape[0], F.N_CLASSES), np.nan, dtype=np.float32)
    ens_full[ens_mask] = pooled["probs_ens"]["ensemble"]
    key_models["ensemble"] = ens_full
    key = table(key_models, ens_mask)
    return {"full_pooled": full, "key_window_pooled": key,
            "key_windows": list(KEY_WINDOWS)}


# ===========================================================================
# Figures
# ===========================================================================
def _figdir() -> Path:
    d = config.OUTPUTS / "figures" / "loan_level"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ym_to_frac(ym: np.ndarray) -> np.ndarray:
    """YYYYMM int → fractional year for a continuous time axis."""
    yr = ym // 100
    mo = ym % 100
    return yr + (mo - 0.5) / 12.0


def fig_nll_by_year(table_b: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    years = sorted(table_b["by_year"])
    for m in MODEL_ORDER:
        ys = [table_b["by_year"][k].get(m) for k in years]
        xs = [k for k, v in zip(years, ys) if v is not None]
        vv = [v for v in ys if v is not None]
        ax.plot(xs, vv, marker="o", label=MODEL_LABELS[m])
    ax.set_xlabel("Test year"); ax.set_ylabel("Out-of-sample NLL")
    # No in-figure title: the LaTeX caption supplies it (this internal "Table B"
    # label does not correspond to any table number in the write-up).
    ax.set_xticks(years); ax.grid(alpha=0.3); ax.legend()
    p = _figdir() / "F_tableB_nll_by_year"
    fig.savefig(f"{p}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{p}.pdf", bbox_inches="tight")
    plt.close(fig)
    return p


def _reliability(ax, pred, label, model_name, nbins=12):
    """Quantile-binned reliability curve of ``pred`` vs the binary ``label``."""
    if pred.shape[0] < nbins * 5:
        return
    qs = np.quantile(pred, np.linspace(0, 1, nbins + 1))
    qs = np.unique(qs)
    idx = np.clip(np.searchsorted(qs[1:-1], pred), 0, len(qs) - 2)
    mp, mo = [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() >= 20:
            mp.append(float(pred[m].mean())); mo.append(float(label[m].mean()))
    ax.plot(mp, mo, marker="o", ms=4, label=model_name)


def fig_calibration(pooled: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    y = pooled["y"].astype(np.int64)
    origin = pooled["origin"]
    label_year = pooled["label_ym"] // 100
    cur = origin == OI["current"]
    covid = cur & np.isin(label_year, [2020, 2021])

    # Single 2x2 grid: rows = transition (prepayment / delinquency), columns =
    # regime (all years pooled vs the 2020–21 forbearance window). The bold row
    # transition is carried in each panel title so the two rows are self-evident;
    # the "Other years" panel is dropped (it tracks the pooled column closely).
    rows = [(PREPAID, "Prepaid"), (DPD30, "30+ DPD")]
    cols = [("Pooled", cur), ("2020–21 (COVID)", covid)]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.4))
    for r, (di, tname) in enumerate(rows):
        lim = max(0.02, float((y[cur] == di).mean()) * 4)
        for c, (cname, mask) in enumerate(cols):
            ax = axes[r][c]
            ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1, label="perfect")
            for mname in ("logit", "nn"):
                _reliability(ax, pooled["probs"][mname][mask][:, di],
                             (y[mask] == di).astype(np.float64), MODEL_LABELS[mname])
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_title(f"Current$\\to${tname}  ·  {cname}", fontweight="bold", fontsize=11)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
            ax.grid(alpha=0.3)
            if r == len(rows) - 1:
                ax.set_xlabel("Mean predicted probability")
            if c == 0:
                ax.set_ylabel("Observed frequency")
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    p = _figdir() / "F_calibration_current"
    fig.savefig(f"{p}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{p}.pdf", bbox_inches="tight")
    plt.close(fig)
    return [p]


def fig_rate_overlay(pooled: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = pooled["y"].astype(np.int64)
    cur = pooled["origin"] == OI["current"]
    ym = pooled["label_ym"][cur]
    months = np.unique(ym)
    x = _ym_to_frac(months)

    from matplotlib.ticker import PercentFormatter
    pretty = {"prepaid": "Prepaid", "dpd_30": "30+ DPD"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, (di, dname) in zip(axes, ((PREPAID, "prepaid"), (DPD30, "dpd_30"))):
        realized = np.array([float((y[cur][ym == mo] == di).mean()) for mo in months])
        ax.plot(x, realized, c="black", lw=1.8, label="realized")
        for mname in ("logit", "nn"):
            P = pooled["probs"][mname][cur][:, di]
            pred = np.array([float(P[ym == mo].mean()) for mo in months])
            ax.plot(x, pred, lw=1.3, alpha=0.85, label=f"predicted ({MODEL_LABELS[mname]})")
        ax.set_ylabel(f"Monthly Current$\\to${pretty[dname]} Rate",
                      fontweight="bold", fontsize=12)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        ax.grid(alpha=0.3); ax.legend(loc="upper right")
    axes[-1].set_xlabel("Transition (label) month")
    # No suptitle: the LaTeX caption supplies it.
    p = _figdir() / "F_rate_overlay"
    fig.savefig(f"{p}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{p}.pdf", bbox_inches="tight")
    plt.close(fig)
    return p


def fig_ensemble_curve() -> Path | None:
    """Out-of-sample-NLL-vs-ensemble-size curve (paper Fig 7), from the committed key-window
    ensemble summaries (M10b ``ensemble_k*_summary.json``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for k in KEY_WINDOWS:
        p = config.MODELS / "nn" / VARIANT / f"ensemble_k{k}_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        curve = s.get("test_curve", [])
        if curve:
            ax.plot([c["size"] for c in curve], [c["nll"] for c in curve],
                    marker="o", label=f"k={k}")
            plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.set_xlabel("Ensemble size"); ax.set_ylabel("Out-of-sample NLL")
    # No in-figure title: the LaTeX caption supplies it (and cites the paper figure).
    ax.grid(alpha=0.3); ax.legend()
    pp = _figdir() / "F_ensemble_size_curve"
    fig.savefig(f"{pp}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{pp}.pdf", bbox_inches="tight")
    plt.close(fig)
    return pp


# ===========================================================================
# Table writers (md + csv + tex + json) — table_a.py conventions
# ===========================================================================
def _fmt(x) -> str:
    return "—" if x is None else f"{x:.6f}"


def _fmt_auc(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def write_table_b(tb: dict) -> None:
    d = config.OUTPUTS / "tables" / "loan_level"
    d.mkdir(parents=True, exist_ok=True)
    (d / "table_b.json").write_text(json.dumps(tb, indent=2))

    cols = MODEL_ORDER
    hdr = "| Test year | " + " | ".join(MODEL_LABELS[m] for m in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    years = sorted(tb["by_year"])
    md = ["### Table B — rolling out-of-sample NLL (full export, frozen test slices)", "",
          hdr, sep]
    for k in years:
        r = tb["by_year"][k]
        md.append(f"| {k} | " + " | ".join(_fmt(r.get(m)) for m in cols) + " |")
    md.append(f"| **Pooled** | " + " | ".join(_fmt(tb["pooled"].get(m)) for m in cols) + " |")
    md += ["", f"*Pooled over {tb['n_pooled']:,} test rows (each predicted once); ensemble "
           f"column over the {tb['n_pooled_ensemble']:,} key-window rows. Empirical matrix = "
           "the covariate-free floor (weighted train-slice estimate).*", "",
           "### Pooled NLL by origin state", "",
           "| Origin | n | " + " | ".join(MODEL_LABELS[m] for m in cols) + " |",
           "|" + "---|" * (len(cols) + 2)]
    for s in ORIGIN_STATES:
        r = tb["by_origin"][s]
        md.append(f"| {s} | {r['n']:,} | " + " | ".join(_fmt(r.get(m)) for m in cols) + " |")
    (d / "table_b.md").write_text("\n".join(md) + "\n")

    csv = ["test_year," + ",".join(cols)]
    for k in years:
        r = tb["by_year"][k]
        csv.append(f"{k}," + ",".join("" if r.get(m) is None else f"{r[m]:.6f}" for m in cols))
    csv.append("pooled," + ",".join("" if tb["pooled"].get(m) is None else f"{tb['pooled'][m]:.6f}"
                                    for m in cols))
    (d / "table_b.csv").write_text("\n".join(csv) + "\n")

    tex = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
           "Test year & " + " & ".join(MODEL_LABELS[m] for m in cols) + r" \\", r"\midrule"]
    for k in years:
        r = tb["by_year"][k]
        tex.append(f"{k} & " + " & ".join(_fmt(r.get(m)) for m in cols) + r" \\")
    tex += [r"\midrule",
            r"\textbf{Pooled} & " + " & ".join(_fmt(tb["pooled"].get(m)) for m in cols) + r" \\",
            r"\bottomrule", r"\end{tabular}"]
    (d / "table_b.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {d/'table_b.md'} (+ .csv/.tex/.json)")


def write_auc(auc: dict) -> None:
    d = config.OUTPUTS / "tables" / "loan_level"
    (d / "auc.json").write_text(json.dumps(auc, indent=2))

    def block(rows, models, title):
        hdr = "| Origin | Destination | n_pos | " + " | ".join(models) + " |"
        out = [title, "", hdr, "|" + "---|" * (3 + len(models))]
        for r in rows:
            out.append(f"| {r['origin']} | {r['destination']} | {r['n_pos']:,} | "
                       + " | ".join(_fmt_auc(r.get(m)) for m in models) + " |")
        return out

    md = ["### AUC — one-vs-rest, pooled test predictions (§3.4)", ""]
    md += block(auc["full_pooled"], ["logit", "nn"],
                "#### All 11 windows pooled (logit vs best NN)")
    md += [""]
    md += block(auc["key_window_pooled"], ["logit", "nn", "ensemble"],
                f"#### Key windows pooled {auc['key_windows']} (logit vs NN vs ensemble)")
    (d / "auc.md").write_text("\n".join(md) + "\n")

    csv = ["scope,origin,destination,n,n_pos,logit,nn,ensemble"]
    for r in auc["full_pooled"]:
        csv.append(f"full,{r['origin']},{r['destination']},{r['n']},{r['n_pos']},"
                   f"{_fmt_auc(r.get('logit'))},{_fmt_auc(r.get('nn'))},")
    for r in auc["key_window_pooled"]:
        csv.append(f"key,{r['origin']},{r['destination']},{r['n']},{r['n_pos']},"
                   f"{_fmt_auc(r.get('logit'))},{_fmt_auc(r.get('nn'))},{_fmt_auc(r.get('ensemble'))}")
    (d / "auc.csv").write_text("\n".join(csv) + "\n")

    tex = [r"\begin{tabular}{llrrr}", r"\toprule",
           r"Origin & Destination & Logit & Best NN & Ensemble \\", r"\midrule"]
    for r in auc["key_window_pooled"]:
        o = r["origin"].replace("_", r"\_")            # escape outside the f-string expr (py<3.12)
        dst = r["destination"].replace("_", r"\_")
        tex.append(f"{o} & {dst} & "
                   f"{_fmt_auc(r.get('logit'))} & {_fmt_auc(r.get('nn'))} & "
                   f"{_fmt_auc(r.get('ensemble'))} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (d / "auc.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {d/'auc.md'} (+ .csv/.tex/.json)")


# ===========================================================================
# Driver
# ===========================================================================
def run(windows: list[int], device) -> dict:
    t0 = time.perf_counter()
    print(f"=== M11 evaluate  windows={windows}  device={device} ===")

    per_window = {}
    for k in windows:
        tk = time.perf_counter()
        per_window[k] = score_window(k, device)
        w = per_window[k]
        models = list(w["probs"])
        print(f"  k={k}: {w['n']:,} test rows, models={models}, "
              f"pipe_match={w['pipe_match']}  [{time.perf_counter()-tk:.0f}s]")

    # ---- Pool the per-window slices (disjoint period_ym tiling → each row once) ----
    keep = [k for k in config.TEST_YEARS if k in per_window]
    y = np.concatenate([per_window[k]["y"] for k in keep])
    origin = np.concatenate([per_window[k]["origin"] for k in keep])
    label_ym = np.concatenate([per_window[k]["label_ym"] for k in keep])
    keyhash = np.concatenate([per_window[k]["keyhash"] for k in keep])
    probs = {m: np.concatenate([per_window[k]["probs"][m] for k in keep])
             for m in ("empirical", "logit", "nn")}
    ens_mask = np.concatenate([np.full(per_window[k]["n"], k in KEY_WINDOWS, dtype=bool)
                               for k in keep])
    probs_ens = {}
    if ens_mask.any():
        probs_ens["ensemble"] = np.concatenate(
            [per_window[k]["probs"]["ensemble"] for k in keep if k in KEY_WINDOWS])
    pooled = {"y": y, "origin": origin, "label_ym": label_ym, "keyhash": keyhash,
              "probs": probs, "probs_ens": probs_ens, "ens_mask": ens_mask}

    # ---- Tables + figures ----
    table_b = build_table_b(per_window, pooled)
    write_table_b(table_b)
    auc = build_auc(pooled)
    write_auc(auc)
    figs = {"nll_by_year": str(fig_nll_by_year(table_b)),
            "calibration": [str(p) for p in fig_calibration(pooled)],
            "rate_overlay": str(fig_rate_overlay(pooled)),
            "ensemble_curve": str(fig_ensemble_curve() or "")}

    # ---- Accept evidence ----
    accept = verify(per_window, pooled, table_b, figs)

    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "variant": VARIANT, "windows": keep, "key_windows": list(KEY_WINDOWS),
        "git_commit": T._git_commit(),
        "table_b": table_b, "auc": auc, "figures": figs, "accept": accept,
        "wall_sec": time.perf_counter() - t0,
    }
    out = config.MODELS / "nn" / VARIANT / "evaluate_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}  [{summary['wall_sec']/60:.1f} min]")
    return summary


def verify(per_window: dict, pooled: dict, table_b: dict, figs: dict) -> dict:
    """Every M11 Accept criterion, with evidence."""
    ok = True
    checks = []

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        checks.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    print("\n[1] every model evaluated on identical per-window frozen test rows "
          "(row-count + hash)")
    for k in sorted(per_window):
        w = per_window[k]
        # All models scored on the single loaded encoding ⇒ identical rows by construction;
        # corroborate against each committed run's stored test row count + pipeline identity.
        logit_n = json.loads((config.MODELS / "logit" / VARIANT / f"k{k}" /
                              "metrics.json").read_text())["rows"]["test"]
        nn_n = json.loads((config.MODELS / "nn" / VARIANT /
                           B._nn_tag(k, dict(config.NN_SELECTED)) /
                           "metrics.json").read_text())["rows"]["test"]
        same = (w["n"] == logit_n == nn_n) and all(
            w["probs"][m].shape[0] == w["n"] for m in w["probs"])
        check(f"k={k}: {w['n']:,} rows == logit({logit_n:,}) == NN({nn_n:,}); "
              f"pipeline byte-identical={w['pipe_match']}; key_hash={w['test_key_hash']}",
              same and w["pipe_match"])

    print("\n[2] each test row appears exactly once in the pooled set")
    n_pool = pooled["y"].shape[0]
    n_sum = sum(per_window[k]["n"] for k in per_window)
    n_uniq = int(np.unique(pooled["keyhash"]).shape[0])
    # period_ym test bounds tile the axis disjointly (config) — corroborate via key uniqueness.
    check(f"pooled rows = Σ window rows ({n_pool:,} == {n_sum:,})", n_pool == n_sum)
    check(f"unique (loan, period_ym) keys == pooled rows ({n_uniq:,} == {n_pool:,})",
          n_uniq == n_pool, "no row counted twice across windows")

    print("\n[3] figures regenerate")
    for name, p in figs.items():
        paths = p if isinstance(p, list) else ([p] if p else [])
        present = all(Path(f"{x}.png").exists() and Path(f"{x}.pdf").exists() for x in paths)
        check(f"figure '{name}' written ({len(paths)} file(s))", present and bool(paths) or
              (name == "ensemble_curve" and not paths))

    print("\n[4] headline sanity — NN ≤ logit ≤ empirical pooled (covariates help)")
    pl_ = table_b["pooled"]
    check(f"pooled NN {pl_['nn']:.6f} ≤ logit {pl_['logit']:.6f} ≤ empirical {pl_['empirical']:.6f}",
          pl_["nn"] <= pl_["logit"] <= pl_["empirical"])

    print(f"\n{'ALL M11 ACCEPT CRITERIA PASS' if ok else 'SOME CHECKS FAILED'} (variant={VARIANT})")
    return {"all_pass": ok, "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description="M11 evaluation suite (Tables A/B, AUC, calibration).")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--windows", type=int, nargs="*", default=None,
                    help="subset of test years (default: all 11)")
    args = ap.parse_args()
    config.require_drive()
    device = (torch.device("cuda") if args.device == "cuda"
              else tc.resolve_device(args.device))
    windows = args.windows or list(config.TEST_YEARS)
    run(windows, device)


if __name__ == "__main__":
    main()
