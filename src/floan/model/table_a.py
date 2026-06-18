"""M9 — Table A: the tuning-window depth grid (paper Table 11 analogue, ``02_LOAN_LEVEL §7``).

Assembles, for window ``k = 2015``, the in-sample + out-of-sample NLL of every Phase-2 model:
empirical matrix, bucketed matrix, logit, augmented logit, NN×{1,3,5,7}, NN-5+dropout, and
the 8-net ensemble. Out-of-sample (test) NLL — the headline §1 metric and the selection
criterion — comes from each model's committed run (benchmarks via the M7 logit run's
``empirical_floor``; logit/augmented from the M7 run; NN cells from :mod:`grid`; ensemble
from :mod:`ensemble`). **In-sample** NLL is *not* stored by the trainers, so it is computed
here, uniformly, as the importance-weighted train-slice NLL (Σw·CE/Σw — the §6.4 unbiased
in-sample estimator, comparable to the unthinned-eval-pool test NLL).

Reading the in-sample column: it is *higher* than out-of-sample, which is **not** an
inverted overfitting signal — the train slice (labels ≤ 2013) spans the crisis-heavy
2000–2013 period (higher transition entropy) whereas the 2015 test year is calm. The
overfitting exhibit is therefore the depth trend *within* the in-sample column (more
capacity drives in-sample NLL down), not the in-vs-out gap; out-of-sample is the clean
decision metric.

Two documented gaps in the in-sample column (the ``§7`` deviation, permitted by the M9
Accept clause): (1) the **empirical/bucketed** matrices' in-sample NLL needs the unthinned
panel, which is not on the dev box — and a frequency estimator does not overfit (in≈out by
construction); (2) the **augmented logit**'s in-sample needs its spline/feature transform
reloaded — out-of-sample suffices for its "linear baseline" role. Both show ``—``.

Companion output: the full 14-cell depth×dropout×L2 grid with in+out NLL — the §8.1
overfitting ablation that exhibits the dropout-depth interaction.

Writes ``outputs/tables/loan_level/{table_a,grid}.{csv,md,tex}``.

Run:
    .venv/bin/python -m floan.model.table_a --device cuda
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from floan.model import config
from floan.model import data as D
from floan.model import features as F
from floan.model import logit as L
from floan.model import net as N
from floan.model import torch_common as tc
from floan.model import train as T

EVAL_BATCH = 16384


# ---------------------------------------------------------------------------
# In-sample scoring — weighted train-slice NLL, uniform across model families
# ---------------------------------------------------------------------------
@torch.no_grad()
def _weighted_nll(model, predict, enc: dict, device) -> float:
    """Σw·CE/Σw of ``model`` over a preloaded slice (the §6.4 unbiased estimator)."""
    return tc.evaluate_nll(model, T.eval_batches(enc, EVAL_BATCH), predict, device)["weighted_nll"]


def _nn_insample(run, enc: dict, device) -> float:
    arch = json.loads((run / "metrics.json").read_text())["architecture"]
    model = N.from_arch({k: arch[k] for k in
                         ("n_continuous", "n_binary", "vocab_sizes", "emb_dims",
                          "hidden_dims", "n_classes", "dropout")}).to(device)
    model.load_state_dict(torch.load(run / "best_model.pt", map_location=device))
    return _weighted_nll(model, N.make_nn_predict(), enc, device)


def _ensemble_insample(runs: list, enc: dict, device) -> float:
    """In-sample NLL of the mean-probability ensemble on the (weighted) train slice."""
    from floan.model import ensemble as E
    y, w = np.asarray(enc["y"]), np.asarray(enc["w"], dtype=np.float64)
    acc = np.zeros((y.shape[0], F.N_CLASSES), dtype=np.float64)
    for r in runs:
        acc += E._member_probs(r, enc, device)
    return E._nll(acc / len(runs), y, w)


def _logit_insample(logit_run, enc: dict, device) -> float:
    scaler, vocab = F.load_pipeline(logit_run)
    dims = L.design_width(scaler, vocab)
    model = L.LogitNet(dims["in_features"]).to(device)
    model.load_state_dict(torch.load(logit_run / "model.pt", map_location=device))
    return _weighted_nll(model, L.make_logit_predict(vocab), enc, device)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _fmt(x) -> str:
    return "—" if x is None else f"{x:.6f}"


def build(variant: str, k: int, device_name: str) -> dict:
    device = tc.resolve_device(device_name)
    grid = json.loads((config.MODELS / "nn" / variant / "grid_summary.json").read_text())
    ens = json.loads((config.MODELS / "nn" / variant / "ensemble_summary.json").read_text())
    logit_run = config.MODELS / "logit" / variant / f"k{k}"
    lm = json.loads((logit_run / "metrics.json").read_text())
    floor, lg, aug = lm["empirical_floor"], lm["logit"], lm["augmented_logit"]

    # Preload the train slice once per featurization (NN embeddings vs logit one-hot). The
    # scaler/vocab are deterministic on the train slice, so any NN run's pipeline serves all
    # NN cells; the logit is scored under its own committed pipeline.
    by_tag = {r["tag"]: r for r in grid["rows"]}
    nn_runs = {tag: config.MODELS / "nn" / variant / tag for tag in by_tag}
    nn_pipe_run = next(iter(nn_runs.values()))
    scaler, vocab = F.load_pipeline(nn_pipe_run)
    enc_nn = T.preload_split(variant, k, "train", scaler, vocab, None)
    lscaler, lvocab = F.load_pipeline(logit_run)
    enc_lg = T.preload_split(variant, k, "train", lscaler, lvocab, None)
    print(f"in-sample train slice: {enc_nn['y'].shape[0]:,} rows")

    def nn_cell(depth, dropout, wd):
        tag = f"k{k}_d{depth}_do{dropout:g}_wd{wd:g}"
        r = by_tag[tag]
        return {"in": _nn_insample(nn_runs[tag], enc_nn, device), "out": r["test_nll"],
                "val": r["val_nll"]}

    # Full 14-cell grid table (companion: the §8.1 dropout-depth ablation).
    grid_rows = []
    for r in sorted(grid["rows"], key=lambda r: (r["depth"], r["dropout"], r["weight_decay"])):
        ins = _nn_insample(nn_runs[r["tag"]], enc_nn, device)
        grid_rows.append({"depth": r["depth"], "dropout": r["dropout"],
                          "L2": r["weight_decay"], "in_sample": ins,
                          "val": r["val_nll"], "out_sample": r["test_nll"]})
        print(f"  grid d{r['depth']} do{r['dropout']:g} L2{r['weight_decay']:g}: "
              f"in={ins:.6f} val={r['val_nll']:.6f} out={r['test_nll']:.6f}")

    # Ensemble in/out.
    ens_runs = [config.MODELS / "nn" / variant / f"ens_k{k}_s{s}"
                for s in range(ens["n_members"])]
    ens_in = _ensemble_insample(ens_runs, enc_nn, device)
    sel_single = grid["selected"]["single_nn"]

    # Table A rows (paper-Table-11 order, §7).
    logit_in = _logit_insample(logit_run, enc_lg, device)
    rows = [
        {"model": "Empirical matrix", "in": None, "out": floor["empirical"]},
        {"model": "Bucketed matrix", "in": None, "out": floor["bucketed"]},
        {"model": "Logit", "in": logit_in, "out": lg["test_nll"]},
        {"model": "Augmented logit", "in": None, "out": aug["test_nll"]},
    ]
    for d in (1, 3, 5, 7):
        c = nn_cell(d, 0.0, 0.0)
        rows.append({"model": f"NN depth {d} (no reg)", "in": c["in"], "out": c["out"]})
    c5 = nn_cell(5, 0.2, 0.0)
    rows.append({"model": "NN-5 + dropout 0.2", "in": c5["in"], "out": c5["out"]})
    if (sel_single["depth"], sel_single["dropout"], sel_single["weight_decay"]) \
            not in [(5, 0.2, 0.0), (1, 0.0, 0.0), (3, 0.0, 0.0), (5, 0.0, 0.0), (7, 0.0, 0.0)]:
        cs = nn_cell(sel_single["depth"], sel_single["dropout"], sel_single["weight_decay"])
        rows.append({"model": f"NN best single (d{sel_single['depth']} "
                     f"do{sel_single['dropout']:g} L2{sel_single['weight_decay']:g})",
                     "in": cs["in"], "out": cs["out"]})
    rows.append({"model": f"Ensemble ×{ens['n_members']} (d{ens['config']['depth']} "
                 f"do{ens['config']['dropout']:g} L2{ens['config']['weight_decay']:g})",
                 "in": ens_in, "out": ens["ensemble_test_nll"]})

    table = {"window_k": k, "variant": variant, "pooled_empirical": floor["pooled_empirical"],
             "table_a": rows, "grid": grid_rows, "selected_single": sel_single,
             "ensemble_config": ens["config"]}
    _write(table)
    _print(table)
    return table


# ---------------------------------------------------------------------------
# Persist — csv + markdown + latex into outputs/tables/loan_level/
# ---------------------------------------------------------------------------
def _write(t: dict) -> None:
    d = config.OUTPUTS / "tables" / "loan_level"
    d.mkdir(parents=True, exist_ok=True)
    (d / "table_a.json").write_text(json.dumps(t, indent=2))

    a = t["table_a"]
    csv = ["model,in_sample_nll,out_sample_nll"] + \
          [f"{r['model']},{'' if r['in'] is None else f'{r['in']:.6f}'},{r['out']:.6f}" for r in a]
    (d / "table_a.csv").write_text("\n".join(csv) + "\n")

    md = [f"### Table A — tuning-window depth grid (k={t['window_k']}, {t['variant']} export)",
          "", "| Model | In-sample NLL | Out-of-sample NLL |", "|---|---|---|"]
    md += [f"| {r['model']} | {_fmt(r['in'])} | {_fmt(r['out'])} |" for r in a]
    md += ["", f"*Pooled empirical floor: {t['pooled_empirical']:.6f}. "
           "In-sample for the frequency benchmarks and augmented logit shown `—` "
           "(see module docstring). Note in-sample > out-of-sample here is **not** an "
           "inverted overfitting signal: the train slice (labels ≤ 2013) spans the "
           "crisis-heavy 2000–2013 period (higher transition entropy) while the 2015 test "
           "year is calm — the overfitting read is the depth trend *within* the in-sample "
           "column (more capacity → lower in-sample fit), not the in-vs-out gap.*", "",
           "### Companion grid — depth × dropout × L2 (in / val / out NLL)", "",
           "| Depth | Dropout | L2 | In-sample | Val | Out-of-sample |", "|---|---|---|---|---|---|"]
    md += [f"| {g['depth']} | {g['dropout']:g} | {g['L2']:g} | {g['in_sample']:.6f} | "
           f"{g['val']:.6f} | {g['out_sample']:.6f} |" for g in t["grid"]]
    (d / "table_a.md").write_text("\n".join(md) + "\n")

    tex = [r"\begin{tabular}{lrr}", r"\toprule",
           r"Model & In-sample NLL & Out-of-sample NLL \\", r"\midrule"]
    tex += [f"{r['model'].replace('&', r'\&')} & {_fmt(r['in'])} & {_fmt(r['out'])} \\\\" for r in a]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (d / "table_a.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {d / 'table_a.md'} (+ .csv/.tex/.json)")


def _print(t: dict) -> None:
    print(f"\n=== Table A (k={t['window_k']}, {t['variant']}) ===")
    print(f"  {'Model':<42} {'in-sample':>11} {'out-sample':>11}")
    for r in t["table_a"]:
        print(f"  {r['model']:<42} {_fmt(r['in']):>11} {_fmt(r['out']):>11}")
    print(f"  {'(pooled empirical floor)':<42} {'':>11} {t['pooled_empirical']:>11.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M9 Table A builder (tuning window).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    args = ap.parse_args()
    config.require_drive()
    build(args.variant, args.k, args.device)


if __name__ == "__main__":
    main()
