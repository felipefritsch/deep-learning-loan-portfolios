"""M27a summary — rolling table (single-GRU test NLL + key AUC transitions, ensemble on key
windows w/ mean±sd) + pooled all-years NLL. Reads artifacts in m27a_gpu_runs/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floan.model import config as mc
from floan.model import evaluate as EV

RES = mc.OUTPUTS / "m27a_gpu_runs"
WINDOWS = list(range(2015, 2026))
KEY = [2015, 2019, 2020, 2023, 2025]
SEEDS = [0, 1, 2]
# the three display transitions (seq_train.TRANS)
KEY_TRANS = [("current", "dpd_30"), ("current", "prepaid"), ("dpd_90plus", "foreclosure")]

def load_json(k, s):
    return json.loads((RES / f"k{k}_s{s}.json").read_text())

def auc_of(auc_rows, o, d):
    for r in auc_rows:
        if r["origin"] == o and r["destination"] == d:
            return r["auc"]
    return None

def f(x):
    return "  —  " if x is None else f"{x:.4f}"

def ensemble_for(k):
    """Mean of softmax probs across 3 seeds -> ensemble test NLL + AUC; plus 3-seed mean±sd NLL."""
    probs, y, origin, nlls = [], None, None, []
    for s in SEEDS:
        z = np.load(RES / f"k{k}_s{s}.npz")
        probs.append(z["probs"].astype(np.float64))
        y = z["y"]; origin = z["origin"]
        nlls.append(load_json(k, s)["test_nll"])
    ens = np.mean(probs, axis=0)
    ens_nll = EV._nll(ens, y)
    ens_auc = {(o, d): EV._auc_one_vs_rest(ens[origin == EV.OI[o]][:, EV.SI[d]],
                                           (y[origin == EV.OI[o]] == EV.SI[d]))
               for o, d in KEY_TRANS}
    return {"mean": float(np.mean(nlls)), "sd": float(np.std(nlls, ddof=1)),
            "ens_nll": ens_nll, "ens_auc": ens_auc, "seeds": nlls}

def main():
    single = {k: load_json(k, 0) for k in WINDOWS}
    ens = {k: ensemble_for(k) for k in KEY}

    print("=" * 110)
    print("M27a ROLLING TABLE — single-GRU (seed 0) test NLL + key AUC transitions; ensemble on key windows")
    print("=" * 110)
    hdr = (f"{'test_yr':>7} | {'single NLL':>10} | {'AUC c->30':>9} {'AUC c->pp':>9} "
           f"{'AUC 90->fc':>10} | {'ens NLL':>8} | {'3-seed mean±sd':>16}")
    print(hdr); print("-" * 110)
    for k in WINDOWS:
        j = single[k]; a = j["test_auc"]
        row = (f"{k:>7} | {j['test_nll']:>10.6f} | "
               f"{f(auc_of(a,'current','dpd_30')):>9} {f(auc_of(a,'current','prepaid')):>9} "
               f"{f(auc_of(a,'dpd_90plus','foreclosure')):>10} | ")
        if k in ens:
            e = ens[k]
            row += f"{e['ens_nll']:>8.6f} | {e['mean']:.6f}±{e['sd']:.6f}"
        else:
            row += f"{'—':>8} | {'—':>16}"
        print(row)
    print("-" * 110)

    # pooled all-years (11 disjoint single-seed test years)
    num = sum(single[k]["test_nll"] * single[k]["n_test"] for k in WINDOWS)
    den = sum(single[k]["n_test"] for k in WINDOWS)
    print(f"\nPOOLED all-years NLL (Σ NLL_w·n_w / Σ n_w over {len(WINDOWS)} disjoint test years): "
          f"{num/den:.6f}   (total n = {den:,})")

    # ensemble AUC detail on key windows
    print("\nENSEMBLE per-transition AUC (key windows, 3-member mean-of-softmax):")
    print(f"  {'yr':>5} | {'c->30':>7} {'c->pp':>7} {'90->fc':>7} | ens_NLL   single(s0)NLL")
    for k in KEY:
        e = ens[k]
        print(f"  {k:>5} | {f(e['ens_auc'][('current','dpd_30')]):>7} "
              f"{f(e['ens_auc'][('current','prepaid')]):>7} "
              f"{f(e['ens_auc'][('dpd_90plus','foreclosure')]):>7} | "
              f"{e['ens_nll']:.6f}   {single[k]['test_nll']:.6f}")

    print(f"\nSanity: k2015 single-seed test NLL = {single[2015]['test_nll']:.6f} (expect ~0.0969)")

if __name__ == "__main__":
    main()
