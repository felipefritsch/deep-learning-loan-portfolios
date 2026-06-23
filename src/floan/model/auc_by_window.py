"""M25 — AUC as the cross-period **display** metric (``ECONOMIC_ENGINE §7``).

NLL stays the estimation loss, the model-selection criterion, and the within-window comparison
metric (the spine of the flexibility-ladder attribution; ``evaluate.py`` Table B). NLL is,
however, unintuitive and **not comparable across years** (each test year is a different
difficulty). So **AUC is the display/visualization metric across time** — bounded [0, 1], a clean
"ranking quality" reading, comparable across windows. **AUC illustrates, NLL decides.**

This module is the **per-window AUC time-series driver** that the spec's integration table lists
under ``evaluate.py`` ("none for the metric path; M25 adds a per-window AUC time-series driver that
calls the existing AUC function per window — surface this, don't code around it"). It is kept as a
**standalone module** rather than appended to ``evaluate.py`` so the heavy M11 metric path is
provably untouched: every number here is produced by *calling* ``evaluate.py``'s own functions —

  * the frozen per-window test rows come from :func:`evaluate.score_window` (the *same* loader and
    the *same* slices the M11 suite uses — this is what makes every AUC trace to ``evaluate.py``);
  * each AUC is :func:`evaluate._auc_one_vs_rest` (the existing pooled-AUC function), applied per
    window instead of pooled.

Deliverable (the AUC mirror of the NLL Table B): a per-window AUC time series, 2015–2025, per
model (logit / best NN / ensemble) for the three diagnostic transitions where the nonlinearity
lives — ``current→prepaid`` (refi S-curve), ``dpd_90plus→foreclosure`` (the sparse deep-delinquency
cell where the net showed its edge), and ``current→dpd_30``. Empirical-matrix is excluded: it is
covariate-free, so its one-vs-rest AUC is ≈0.5 by construction. The ensemble exists only on the
key windows (``evaluate.KEY_WINDOWS``), so its series is plotted/tabulated with gaps elsewhere.

Outputs:
  ``outputs/tables/loan_level/auc_by_window.{json,csv,md,tex}``   per-transition AUC × test-year × model
  ``outputs/figures/loan_level/F_auc_by_window.{png,pdf}``        3-panel per-model time series
  ``models/nn/full/auc_by_window_summary.json``                   numbers + per-window trace + Accept

Run (the GPU box; the full eval pool + model artifacts live there — needs cuda, like ``evaluate``):
    .venv/bin/python -m floan.model.auc_by_window --device cuda
    .venv/bin/python -m floan.model.auc_by_window --device cuda --windows 2015 2020   # subset (dev)
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch

from floan.model import config
from floan.model import evaluate as E       # score_window + _auc_one_vs_rest (reused, never modified)
from floan.model import torch_common as tc
from floan.model import train as T          # _git_commit (provenance)

# The three diagnostic transitions (§7), each a structurally reachable cell (evaluate.REACHABLE).
TRANSITIONS = [("current", "prepaid"), ("dpd_90plus", "foreclosure"), ("current", "dpd_30")]
# Display models — the AUC mirror of Table B's covariate models (empirical excluded: AUC≈0.5).
MODELS = ["logit", "nn", "ensemble"]

NLL_DECIDES_NOTE = (
    "**AUC illustrates, NLL decides.** NLL is the estimation loss, the model-selection criterion, "
    "and the within-window comparison metric (Table B); AUC is a display-only ranking-quality view, "
    "bounded [0,1] and comparable across years. AUC is never a fitting or selection criterion.")


def _key(origin: str, dest: str) -> str:
    return f"{origin}->{dest}"


def _disp(origin: str, dest: str) -> str:
    return f"{origin}→{dest}"          # current→prepaid


# ===========================================================================
# Build — per-transition × test-year × model AUC from the frozen per-window slices
# ===========================================================================
def build_auc_by_window(per_window: dict) -> dict:
    """For each diagnostic transition, the one-vs-rest AUC per test year per model, read off the
    same ``score_window`` outputs the M11 suite uses. ``auc`` is ``None`` where the model is absent
    (ensemble off the key windows) or the cell is degenerate (no positives — ``_auc_one_vs_rest``)."""
    out: dict = {}
    for origin, dest in TRANSITIONS:
        per_year: dict = {}
        for k in sorted(per_window):
            w = per_window[k]
            mask = w["origin"] == E.OI[origin]
            di = E.SI[dest]
            pos = w["y"][mask].astype(np.int64) == di
            aucs = {}
            for m in MODELS:
                P = w["probs"].get(m)
                aucs[m] = None if P is None else E._auc_one_vs_rest(P[mask][:, di], pos)
            per_year[k] = {"n": int(mask.sum()), "n_pos": int(pos.sum()), "auc": aucs}
        out[_key(origin, dest)] = per_year
    return out


# ===========================================================================
# Figure — one 3-panel figure (the AUC mirror of evaluate.fig_nll_by_year)
# ===========================================================================
def fig_auc_by_window(table: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TRANSITIONS), figsize=(16, 4.8))
    for ax, (origin, dest) in zip(axes, TRANSITIONS):
        per_year = table[_key(origin, dest)]
        years = sorted(per_year)
        ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance (0.5)")
        for m in MODELS:
            ys = [per_year[k]["auc"][m] for k in years]
            xs = [k for k, v in zip(years, ys) if v is not None]
            vv = [v for v in ys if v is not None]
            if vv:
                ax.plot(xs, vv, marker="o", label=E.MODEL_LABELS[m])
        ax.set_title(_disp(origin, dest))
        ax.set_xlabel("Test year")
        ax.set_xticks(years)
        ax.tick_params(axis="x", labelrotation=90)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("One-vs-rest AUC")
    fig.suptitle("AUC by test year and model — diagnostic transitions (2015–2025)")
    fig.text(0.5, -0.04, "AUC illustrates, NLL decides — NLL is the estimation/selection metric "
             "(Table B); AUC is display-only.", ha="center", fontsize=9, style="italic")
    p = E._figdir() / "F_auc_by_window"
    fig.savefig(f"{p}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{p}.pdf", bbox_inches="tight")
    plt.close(fig)
    return p


# ===========================================================================
# Table writers (md + csv + tex + json) — evaluate.write_auc conventions
# ===========================================================================
def _csv_auc(x) -> str:
    return "" if x is None else f"{x:.4f}"


def write_auc_by_window(table: dict) -> None:
    d = config.OUTPUTS / "tables" / "loan_level"
    d.mkdir(parents=True, exist_ok=True)
    (d / "auc_by_window.json").write_text(json.dumps(table, indent=2))

    cols = MODELS
    md = ["### AUC by window — one-vs-rest, per-window frozen test slices (M25, ECONOMIC_ENGINE §7)",
          "", NLL_DECIDES_NOTE, ""]
    for origin, dest in TRANSITIONS:
        per_year = table[_key(origin, dest)]
        md += [f"#### {_disp(origin, dest)}", "",
               "| Test year | " + " | ".join(E.MODEL_LABELS[m] for m in cols) + " | n_pos |",
               "|" + "---|" * (len(cols) + 2)]
        for k in sorted(per_year):
            cell = per_year[k]
            md.append(f"| {k} | " + " | ".join(E._fmt_auc(cell["auc"][m]) for m in cols)
                      + f" | {cell['n_pos']:,} |")
        md.append("")
    (d / "auc_by_window.md").write_text("\n".join(md) + "\n")

    csv = ["transition,test_year,n,n_pos," + ",".join(cols)]
    for origin, dest in TRANSITIONS:
        per_year = table[_key(origin, dest)]
        for k in sorted(per_year):
            cell = per_year[k]
            csv.append(f"{_key(origin, dest)},{k},{cell['n']},{cell['n_pos']},"
                       + ",".join(_csv_auc(cell["auc"][m]) for m in cols))
    (d / "auc_by_window.csv").write_text("\n".join(csv) + "\n")

    tex = ["% M25 AUC by window — AUC illustrates, NLL decides (ECONOMIC_ENGINE §7)"]
    for origin, dest in TRANSITIONS:
        per_year = table[_key(origin, dest)]
        tex += [f"% {_disp(origin, dest)}",
                r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
                "Test year & " + " & ".join(E.MODEL_LABELS[m] for m in cols) + r" \\", r"\midrule"]
        for k in sorted(per_year):
            cell = per_year[k]
            tex.append(f"{k} & " + " & ".join(E._fmt_auc(cell["auc"][m]) for m in cols) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", ""]
    (d / "auc_by_window.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {d/'auc_by_window.md'} (+ .csv/.tex/.json)")


# ===========================================================================
# Driver
# ===========================================================================
def run(windows: list[int], device) -> dict:
    t0 = time.perf_counter()
    print(f"=== M25 auc-by-window  windows={windows}  device={device} ===")

    per_window = {}
    for k in windows:
        tk = time.perf_counter()
        per_window[k] = E.score_window(k, device)        # the SAME frozen rows the M11 suite scores
        w = per_window[k]
        print(f"  k={k}: {w['n']:,} test rows, models={list(w['probs'])}  "
              f"[{time.perf_counter()-tk:.0f}s]")

    table = build_auc_by_window(per_window)
    write_auc_by_window(table)
    figp = fig_auc_by_window(table)
    accept = verify(per_window, table, figp)

    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": "M25", "variant": E.VARIANT, "windows": sorted(per_window),
        "key_windows": list(E.KEY_WINDOWS),
        "transitions": [_disp(o, d) for o, d in TRANSITIONS], "models": MODELS,
        "git_commit": T._git_commit(),
        # trace-to-evaluate evidence: the per-window row count + test-key hash that score_window
        # stamps, so each AUC is pinned to the identical frozen test slice the M11 suite uses.
        "trace": {k: {"n": per_window[k]["n"], "test_key_hash": per_window[k]["test_key_hash"]}
                  for k in sorted(per_window)},
        "table": table, "figure": str(figp), "accept": accept,
        "wall_sec": time.perf_counter() - t0,
    }
    out = config.MODELS / "nn" / E.VARIANT / "auc_by_window_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}  [{summary['wall_sec']/60:.1f} min]")
    return summary


def verify(per_window: dict, table: dict, figp: Path) -> dict:
    ok = True
    checks = []

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and bool(passed)
        checks.append({"check": label, "passed": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    print("\n[1] every window traces to evaluate.score_window (n + test_key_hash recorded)")
    for k in sorted(per_window):
        w = per_window[k]
        check(f"k={k}: {w['n']:,} rows; key_hash={w['test_key_hash']}",
              w["n"] > 0 and isinstance(w["test_key_hash"], int))

    print("\n[2] all three diagnostic transitions present for every window")
    for origin, dest in TRANSITIONS:
        key = _key(origin, dest)
        present = key in table and all(k in table[key] for k in per_window)
        check(f"{_disp(origin, dest)} present for all {len(per_window)} windows", present)

    print("\n[3] 'AUC illustrates, NLL decides' note + figure written")
    md = (config.OUTPUTS / "tables" / "loan_level" / "auc_by_window.md").read_text()
    check("note present in auc_by_window.md", "AUC illustrates, NLL decides" in md)
    check("figure F_auc_by_window written",
          Path(f"{figp}.png").exists() and Path(f"{figp}.pdf").exists())

    print(f"\n{'ALL M25 ACCEPT CRITERIA PASS' if ok else 'SOME CHECKS FAILED'}")
    return {"all_pass": ok, "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="M25 — per-window AUC time series (display metric; ECONOMIC_ENGINE §7).")
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
