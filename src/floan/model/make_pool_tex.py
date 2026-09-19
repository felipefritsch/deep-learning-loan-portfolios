"""M15c — emit the two pool-level LaTeX tabulars (T4.2, T5.1) for chapter4 §4.3 from the committed
M14/M15b JSON tables. Pure formatting; no SSD models, no re-run of the pool pipeline.

  * ``table_t42.tex`` — characteristic-bucket pool accuracy: R² / RMSE by model × outcome × anchor
    (the *interpretable* panel; random-pool R² is uninformative by construction, see memo 02c §4).
  * ``table_t51.tex`` — characteristic-bucket economic error: mean |error| by model × metric × anchor
    with the ensemble-vs-logit reduction (the §5.3 headline lives in the price row).
  * ``table_horizon_regime.tex`` (chapter5) — the 5 ensemble-bearing anchors × H in {1,3,6,12}:
    logit/NN/ensemble mean |price error|, the ens-vs-logit %, and the ensemble dollar price error.
  * ``table_t51_full.tex`` (appendix1) — the full 11-anchor × H in {1,3,6,12} grid of mean |price error|.

M24 source: ``outputs/tables/pool_level/m24/t_m24_horizon_regime.json`` (regime table) and
``.../m24/t_m24_econ_errors.json`` (full grid); M14/M15b read ``.../pool_level/t_m1{4,5}_*.json``.

All are tabular-only (caption/label in the chapter .tex), Unicode-free, booktabs — matching the
loan-level ``table_b.tex`` convention. Writes into ``artifacts/generated/``.

Run:  .venv/bin/python -m floan.model.make_pool_tex
"""
from __future__ import annotations

import json

from floan.model import config
from floan.pipeline.config import REPO_ROOT

ANCHORS = ["Dec2014", "Dec2018", "Dec2019", "Dec2022", "Dec2024"]   # the 5 ensemble-bearing anchors
HEAD = ["empirical", "logit", "ensemble"]
FIGS = REPO_ROOT / "artifacts" / "generated"
SRC = config.OUTPUTS / "tables" / "pool_level"
M24 = SRC / "m24"                                                  # M24 horizon/full-grid econ-error tables
ANCHORS_FULL = [f"Dec{y}" for y in range(2014, 2025)]             # all 11 backtest anchors (k = 2015 … 2025)
HORIZONS = (1, 3, 6, 12)                                          # pricing horizons in months (ECONOMIC_ENGINE §2)


def _idx_t42():
    rows = json.loads((SRC / "t_m14_pool_counts.json").read_text())["rows"]
    d: dict = {}
    for r in rows:
        d.setdefault((r["scheme"], r["outcome"], r["anchor"], r["model"]), r)
    return d


def _idx_t51():
    j = json.loads((SRC / "t_m15_econ_errors.json").read_text())
    d: dict = {}
    for r in j["rows"]:
        d[(r["scheme"], r["metric"], r["anchor"], r["model"])] = r
    return d, j["headline"]


def _reduction(logit_v: float, ens_v: float) -> str:
    if logit_v <= 0:
        return "---"
    v = (1.0 - ens_v / logit_v) * 100.0
    return "$0\\%$" if round(v) == 0 else f"${v:+.0f}\\%$"     # avoid the "-0\%" artifact


def table_t42() -> str:
    d = _idx_t42()
    L = ["% Source: outputs/tables/pool_level/t_m14_pool_counts.json (M14, full run).",
         "% Characteristic-bucket panel (FICO x orig-rate-quartile x LTV, >=2000 loans/cell): the",
         "% interpretable cross-pool metric. Random-pool R^2 is uninformative by construction",
         "% (homogeneous ~1000-loan draws) and is discussed in text, not tabulated. For the rare",
         "% 60+ DPD outcome read RMSE, not R^2 (>=~16 buckets makes R^2 unstable). Caption in chapter4.tex.",
         "\\begin{tabular}{llrrrrrr}", "\\toprule",
         " & & \\multicolumn{3}{c}{$R^2$} & \\multicolumn{3}{c}{RMSE (loans)} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         "Outcome & Anchor & Emp. & Logit & Ens. & Emp. & Logit & Ens. \\\\", "\\midrule"]
    for oi, (out, lab) in enumerate((("prepaid", "Prepaid"), ("dpd60p", "60+ DPD"))):
        for ai, a in enumerate(ANCHORS):
            head_cell = f"\\multirow{{5}}{{*}}{{{lab}}}" if ai == 0 else ""
            r2 = " & ".join(f"{d[('char', out, a, m)]['r2']:.3f}" for m in HEAD)
            rm = " & ".join(f"{d[('char', out, a, m)]['rmse']:.0f}" for m in HEAD)
            L.append(f"{head_cell} & {a} & {r2} & {rm} \\\\")
        if oi == 0:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def table_t51() -> str:
    d, _head = _idx_t51()
    metrics = (("price", "\\shortstack[l]{Price error\\\\(per 100 face)}", "{:.3f}"),
               ("cpr",   "\\shortstack[l]{CPR error\\\\(pp)}",            "{:.2f}"),
               ("wal",   "\\shortstack[l]{WAL error\\\\(months)}",        "{:.1f}"))
    L = ["% Source: outputs/tables/pool_level/t_m15_econ_errors.json (M15b).",
         "% Characteristic-bucket panel. Each model cell is the mean |error| over the bucket pools;",
         "% errors = model-implied minus realized-SMM valuation through the same level-pay engine",
         "% (identical WAC/WAM/UPB, WAC-25bp curve, so price dispersion isolates the prepay model).",
         "% Last column = ensemble-vs-logit reduction in that row's mean |error| (the price row is the",
         "% headline). Reported per anchor, never pooled (the Dec2019 COVID anchor inverts). Caption in chapter4.tex.",
         "\\begin{tabular}{llrrrr}", "\\toprule",
         "Metric & Anchor & Empirical & Logit & Ensemble & Ens.\\,vs\\,Logit \\\\", "\\midrule"]
    for mi, (met, lab, fmt) in enumerate(metrics):
        for ai, a in enumerate(ANCHORS):
            head_cell = f"\\multirow{{5}}{{*}}{{{lab}}}" if ai == 0 else ""
            vals = " & ".join(fmt.format(d[("char", met, a, m)]["mean_abs_err"]) for m in HEAD)
            red = _reduction(d[("char", met, a, "logit")]["mean_abs_err"],
                             d[("char", met, a, "ensemble")]["mean_abs_err"])
            L.append(f"{head_cell} & {a} & {vals} & {red} \\\\")
        if mi < len(metrics) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def _anchor_label(a: str) -> str:
    """``Dec2014`` -> ``Dec\\,2014`` (thin space before the year)."""
    return f"{a[:3]}\\,{a[3:]}"


def _dollar_m(d: float) -> str:
    """Ensemble dollar price error as whole $M, sign as a math minus to match the regime table."""
    m = round(d / 1e6)
    return f"$-$\\${abs(m)}\\,M" if m < 0 else f"\\${m}\\,M"


def table_horizon_regime() -> str:
    grid = {(g["anchor"], g["h"]): g
            for g in json.loads((M24 / "t_m24_horizon_regime.json").read_text())["grid"]}
    L = ["% Generated from ssd_mirror/outputs/tables/pool_level/m24/t_m24_horizon_regime.md (M24).",
         "% Five ensemble-bearing anchors x horizons {1,3,6,12}. Tabular-only; caption/label in chapter5.tex.",
         "\\begin{tabular}{llrrrrr}", "\\toprule",
         "Anchor & $H$ & logit & NN & ens. & ens\\,$-$\\,logit & \\$ err (ens) \\\\", "\\midrule"]
    for ai, a in enumerate(ANCHORS):
        for hi, h in enumerate(HORIZONS):
            g = grid[(a, h)]
            mp = g["mae_price"]
            head = _anchor_label(a) if hi == 0 else ""
            red = f"{g['ensemble_vs_logit_pct']:+.1f}\\%"
            L.append(f"{head} & {h}\\,m & {mp['logit']:.3f} & {mp['nn']:.3f} & {mp['ensemble']:.3f} "
                     f"& {red} & {_dollar_m(g['dollar_price_err']['ensemble'])} \\\\")
        if ai < len(ANCHORS) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def table_t51_full() -> str:
    rows = json.loads((M24 / "t_m24_econ_errors.json").read_text())["rows"]
    d = {(r["anchor"], r["h"], r["model"]): r for r in rows if r["scheme"] == "char"}
    L = ["% Generated from ssd_mirror/.../m24/t_m24_econ_errors.md (M24). Full grid: 11 anchors x",
         "% horizons {1,3,6,12}; mean |price error| per 100 face. Caption/label in appendix1.tex.",
         "\\begin{tabular}{llrrrr}", "\\toprule",
         "Anchor & $H$ & empirical & logit & NN & ensemble \\\\", "\\midrule"]
    for ai, a in enumerate(ANCHORS_FULL):
        for hi, h in enumerate(HORIZONS):
            head = _anchor_label(a) if hi == 0 else ""
            cells = [f"{d[(a, h, m)]['price_mae']:.3f}" if (a, h, m) in d else "---"
                     for m in ("empirical", "logit", "nn", "ensemble")]
            L.append(f"{head} & {h}\\,m & " + " & ".join(cells) + " \\\\")
        if ai < len(ANCHORS_FULL) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    (FIGS / "table_t42.tex").write_text(table_t42())
    (FIGS / "table_t51.tex").write_text(table_t51())
    (FIGS / "table_horizon_regime.tex").write_text(table_horizon_regime())
    (FIGS / "table_t51_full.tex").write_text(table_t51_full())
    print(f"wrote {FIGS/'table_t42.tex'}")
    print(f"wrote {FIGS/'table_t51.tex'}")
    print(f"wrote {FIGS/'table_horizon_regime.tex'}")
    print(f"wrote {FIGS/'table_t51_full.tex'}")
