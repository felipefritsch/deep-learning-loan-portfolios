"""M9 — Accept-criteria verifier (``04_TASKS.md`` M9), with evidence printed.

Asserts, against the committed run artifacts (no recompute — pure reads of
``grid_summary.json`` / ``ensemble_summary.json`` / ``table_a.json`` / ``config.py``):

* **Table A complete for k=2015** — every paper-Table-11 row present (empirical, bucketed,
  logit, augmented logit, NN×{1,3,5,7}, NN-5+dropout, ensemble); out-of-sample NLL on every
  row; in-sample on the NN rows + ensemble (the overfitting exhibit).
* **Ensemble of 8 + size curve** — 8 members trained, an 8-point ensemble-size curve, and the
  ensemble's test NLL beats the mean single net (variance reduction, paper Fig 7).
* **Selected config frozen with rationale** — ``config.NN_SELECTED`` equals the grid's
  val-NLL argmin and the freeze block carries the rationale + deviation note.
* **Dropout-depth interaction** — the val-optimal depth at dropout 0 is no deeper than at
  dropout 0.2 (regularization shifts the optimum deeper, the paper's finding); the absolute
  optimum being shallower than the paper's 5 layers is the documented dev-scale deviation.
* **Reproducibility** — ensemble member seed 0 reproduces the grid's depth-3/dropout-0.2 cell
  bit-for-bit (determinism of the shared pipeline).

Run:
    .venv/bin/python -m floan.model.verify_m9
"""

from __future__ import annotations

import json

from floan.model import config

TABLE_DIR = config.OUTPUTS / "tables" / "loan_level"
NN_DIR = config.MODELS / "nn" / "dev"
_ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global _ok
    _ok = _ok and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def main() -> None:
    config.require_drive()
    grid = json.loads((NN_DIR / "grid_summary.json").read_text())
    ens = json.loads((NN_DIR / "ensemble_summary.json").read_text())
    table = json.loads((TABLE_DIR / "table_a.json").read_text())
    rows = {r["model"]: r for r in table["table_a"]}

    print("[1] Table A complete for k=2015 (paper-Table-11 rows, incl. ensemble)")
    must_have = ["Empirical matrix", "Bucketed matrix", "Logit", "Augmented logit",
                 "NN depth 1 (no reg)", "NN depth 3 (no reg)", "NN depth 5 (no reg)",
                 "NN depth 7 (no reg)", "NN-5 + dropout 0.2"]
    for m in must_have:
        check(f"row present: {m}", m in rows)
    check("ensemble row present", any(k.startswith("Ensemble") for k in rows))
    check("every row has out-of-sample NLL", all(r["out"] is not None for r in rows.values()))
    nn_ens_rows = [r for k, r in rows.items() if k.startswith(("NN", "Ensemble"))]
    check("every NN/ensemble row has in-sample NLL",
          all(r["in"] is not None for r in nn_ens_rows),
          f"{sum(r['in'] is not None for r in nn_ens_rows)}/{len(nn_ens_rows)}")
    check("table_a.{csv,md,tex} written",
          all((TABLE_DIR / f"table_a.{e}").exists() for e in ("csv", "md", "tex")))

    print("\n[2] Ensemble of 8 + ensemble-size curve (paper Fig 7)")
    check(f"8 members trained ({ens['n_members']})", ens["n_members"] == 8)
    check(f"8-point size curve ({len(ens['test_curve'])})", len(ens["test_curve"]) == 8)
    sizes = [p["size"] for p in ens["test_curve"]]
    check("curve sizes are 1..8", sizes == list(range(1, 9)))
    et, ms = ens["ensemble_test_nll"], ens["mean_single_nll"]["test_nll"]
    check(f"ensemble test NLL {et:.6f} < mean single {ms:.6f}", et < ms, f"Δ={ms - et:+.6f}")
    best_single_test = min(m["test_nll"] for m in ens["members"])
    print(f"  [info] ensemble test {et:.6f} vs best single member {best_single_test:.6f} "
          f"({'beats' if et < best_single_test else 'does not beat'})")

    print("\n[3] Selected config frozen in config.py with rationale")
    sel = grid["selected"]["single_nn"]
    check("config.NN_SELECTED matches grid val-NLL argmin",
          config.NN_SELECTED == {"depth": sel["depth"], "dropout": sel["dropout"],
                                 "weight_decay": sel["weight_decay"]},
          f"{config.NN_SELECTED}")
    check("ensemble built at the selected config",
          ens["config"]["depth"] == sel["depth"]
          and ens["config"]["dropout"] == sel["dropout"]
          and ens["config"]["weight_decay"] == sel["weight_decay"])
    src = (config.__file__)
    text = open(src).read()
    check("freeze block documents rationale + deviation",
          "NN_SELECTED" in text and "deviation" in text.lower() and "grid_summary" in text)

    print("\n[4] Dropout-depth interaction (regularization shifts the optimal depth)")
    by = {(r["depth"], r["dropout"], r["weight_decay"]): r["val_nll"] for r in grid["rows"]}
    depth_at = lambda do: min((d for d in (1, 3, 5, 7)), key=lambda d: by[(d, do, 0.0)])
    d0, d02 = depth_at(0.0), depth_at(0.2)
    check(f"val-optimal depth at dropout 0 ({d0}) ≤ at dropout 0.2 ({d02})", d0 <= d02,
          "regularization moves the optimum deeper — the paper's interaction")
    # L2 rescue at the anchor: small L2 recovers the depth-5 net toward the shallow optimum.
    d5_0, d5_l2 = by[(5, 0.2, 0.0)], by.get((5, 0.2, 1e-5))
    if d5_l2 is not None:
        print(f"  [info] depth-5 dropout-0.2: L2 0 → {d5_0:.6f}, L2 1e-5 → {d5_l2:.6f} "
              f"(L2 {'rescues' if d5_l2 < d5_0 else 'does not rescue'} the deep net)")
    print(f"  [note] absolute optimum is depth {sel['depth']} (< paper's 5) — the documented "
          f"dev-scale deviation; config.py records it.")

    print("\n[5] Reproducibility — ensemble member s0 == grid depth-3/dropout-0.2 cell")
    s0 = json.loads((NN_DIR / "ens_k2015_s0" / "metrics.json").read_text())["eval"]
    g3 = next(r for r in grid["rows"] if r["tag"] == "k2015_d3_do0.2_wd0")
    check(f"val NLL identical ({s0['val_nll']:.6f})", abs(s0["val_nll"] - g3["val_nll"]) < 1e-9)
    check(f"test NLL identical ({s0['test_nll']:.6f})", abs(s0["test_nll"] - g3["test_nll"]) < 1e-9)

    print(f"\n{'ALL M9 ACCEPT CRITERIA PASS' if _ok else 'SOME CHECKS FAILED'}")
    if not _ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
