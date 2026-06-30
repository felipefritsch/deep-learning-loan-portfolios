"""W3b (``ADR-002``) — hermetic tests for the FF + engineered-history Monte-Carlo pricing seam.

Synthetic only (CPU, no SSD / no GPU). Three things are proved:

  * **Feature recomputation == history.py oracle** — the load-bearing novelty. The per-step
    engineered-feature roll (:func:`seq_pricing_hist._roll_features` / ``_roll_incorporate``, seeded
    from the anchor's observed-history summary) must equal ``history._reference`` (the independent
    Polars oracle that ``history.spot_check`` validates against the leakage-safe SQL) recomputed on
    the full path at every feature-month. If the strict ``period < t`` convention or the episode /
    recency / ever logic drifts, this catches it.
  * **vectorized == reference** — :func:`seq_pricing_hist.simulate_paths_hist` reproduces the per-loan
    reference oracle bit-for-bit on CPU (same RNG streams, same window assembly).
  * **H=1 MC mean → direct** — the MC mean at H=1 converges to the FF net scored on the observed-
    history features (the regression-guard identity, in MC form), for an untrained model.
"""

from __future__ import annotations

import types

import numpy as np
import polars as pl
import torch

from floan.model import features as F
from floan.model import history as H
from floan.model import net as N
from floan.model import seq_pricing as MC
from floan.model import seq_pricing_hist as HP

SI = F.STATE_INDEX
STATES = F.STATES
CUR, D30, D60, D90 = SI["current"], SI["dpd_30"], SI["dpd_60"], SI["dpd_90plus"]
PRE = SI["prepaid"]
DELINQ = ("dpd_30", "dpd_60", "dpd_90plus")
HORIZON = 12


def _ym(mi: int) -> int:
    """Absolute month index -> yyyymm (inverse of history.py's ``mi``)."""
    return (mi // 12) * 100 + (mi % 12) + 1


# ===========================================================================
# 1) Feature recomputation == history.py oracle (the novel logic)
# ===========================================================================
def _oracle(states_by_mi: dict, t_mi: int) -> dict:
    """history._reference on a synthetic monthly (mi, state) series for feature-month ``t_mi``."""
    mis = sorted(states_by_mi)
    rows = pl.DataFrame({"period_ym": [_ym(m) for m in mis],
                         "state": [states_by_mi[m] for m in mis]})
    return H._reference(rows, _ym(t_mi))


def test_feature_recompute_matches_history_oracle():
    """Drive the actual roll (``_roll_features`` + ``_roll_incorporate``), seeded from the anchor's
    observed-history summary, and assert each step's four engineered features equal the independent
    ``history._reference`` oracle recomputed on the full path — across a trajectory that exercises
    re-delinquency (a second episode), recovery to current, and the never-delinquent tail."""
    t0_mi = 2015 * 12 + 11                      # Dec 2014 anchor month index
    # Observed pre-t0 monthly states (months t0_mi-4 .. t0_mi-1): one delinquency episode that cured.
    pre = {t0_mi - 4: "current", t0_mi - 3: "dpd_30", t0_mi - 2: "dpd_60", t0_mi - 1: "current"}
    # Simulated path states at months t0_mi .. t0_mi+H (index j = month t0_mi+j); origin current.
    path = [CUR, CUR, D30, D60, D90, CUR, CUR, CUR, D30, CUR, CUR, CUR, CUR]   # len H+1 = 13
    assert len(path) == HORIZON + 1

    # Seed from the anchor (feature-month t0) oracle — exactly what hist_seed derives from join_history.
    # _reference needs a record AT t0 (state = the origin, current); prior < t0 is the observed history.
    anchor = _oracle({**pre, t0_mi: "current"}, t0_mi)
    ever0 = bool(anchor["ever_delinquent"])
    last_mi0 = (t0_mi - anchor["months_since_last_delinq"]) if ever0 else HP._NEVER
    nepi0 = int(anchor["n_prior_delinq_episodes"])
    prev_cls0 = SI[anchor["prev_state"]]
    seed_li = {"ever": np.array(ever0), "last_mi": np.array(last_mi0),
               "nepi": np.array(nepi0), "prev_isd": np.array(prev_cls0 in HP.DELINQ_ROWS),
               "prev_vidx": np.array(prev_cls0)}                 # psvi=identity ⇒ vidx carries the class
    pred = types.SimpleNamespace(psvi=np.arange(F.N_CLASSES))

    states = np.array(path, dtype=np.int64)[None, :]            # [1, H+1]
    roll = HP._seed_roll(seed_li, (1,))
    full = dict(pre)                                            # full monthly series for the oracle

    for h in range(1, HORIZON + 1):
        tau_mi = t0_mi + (h - 1)
        prev_vidx, ever, msld_val, nepi_val = HP._roll_features(pred, states, h, t0_mi, seed_li, roll)
        ref = _oracle({**pre, **{t0_mi + j: STATES[path[j]] for j in range(h)}}, tau_mi)
        # prev_state class (psvi=identity so prev_vidx == class)
        assert int(prev_vidx.reshape(-1)[0]) == SI[ref["prev_state"]], f"prev_state h={h}"
        assert bool(ever.reshape(-1)[0]) == bool(ref["ever_delinquent"]), f"ever h={h}"
        assert int(nepi_val.reshape(-1)[0]) == int(ref["n_prior_delinq_episodes"]), f"nepi h={h}"
        if ref["months_since_last_delinq"] is None:
            assert not bool(ever.reshape(-1)[0]), f"msld None but ever set h={h}"
        else:
            assert int((tau_mi - roll["last_mi"]).reshape(-1)[0]) == int(ref["months_since_last_delinq"]), \
                f"msld h={h}"
        HP._roll_incorporate(states, h, t0_mi, roll)


# ===========================================================================
# Synthetic FFHistPredictor fixture (for the vectorized==reference + H=1 tests)
# ===========================================================================
_CONT_DEFAULT = {
    "Original Interest Rate": 5.0, "Current Interest Rate": 5.0, "Original UPB": 2.5e5,
    "Current Actual UPB": 2.0e5, "Original Loan Term": 360.0, "Loan Age": 24.0,
    "Remaining Months to Maturity": 336.0, "Original Loan-to-Value (LTV)": 75.0,
    "Original Combined LTV (CLTV)": 75.0, "Debt-to-Income (DTI)": 35.0, "fico_orig": 740.0,
    "fico_co": 745.0, "Mortgage Insurance Percentage": 0.0, "Number of Borrowers": 2.0,
    "Number of Units": 1.0, "pmms30": 4.5, "dgs10": 2.5, "slope_10y2y": 0.5, "unrate_nat": 5.0,
    "unrate_state": 5.0, "incentive": 0.5, "ltv_mtm": 70.0, "hpi_chg_12m": 0.03,
}
_CAT_LEVELS = {c: ["A", "B", "C"] for c in F.RAW_CATEGORICAL if c != "state"}


def _rows_aug(n, *, loans, period_ym, orig_ym, age, rem, upb, states, rng):
    """``n`` raw loan-month rows carrying every AUG feature column (base F.* + the four history
    columns), so the AUG Scaler/Vocab fit and ``encode_frame`` see a complete frame."""
    bc = lambda v: np.full(n, v) if np.isscalar(v) else np.asarray(v)
    cols = {"Loan Identifier": np.asarray(loans).astype(str), "period_ym": bc(period_ym).astype(np.int64),
            "orig_ym": bc(orig_ym).astype(np.int64), "weight": np.ones(n, np.float32),
            "state": list(states), "state_next": list(states)}
    for c, d in _CONT_DEFAULT.items():
        cols[c] = rng.uniform(0.85 * d, 1.15 * d, n) if d else rng.uniform(0.0, 1.0, n)
    cols["Loan Age"] = bc(age).astype(np.float64)
    cols["Remaining Months to Maturity"] = bc(rem).astype(np.float64)
    cols["Current Actual UPB"] = bc(upb).astype(np.float64)
    for c, levels in _CAT_LEVELS.items():
        cols[c] = rng.choice(levels, n)
    for c in F.BINARY:
        cols[c] = rng.integers(0, 2, n).astype(np.float32)
    # history columns
    cols["prev_state"] = rng.choice(STATES, n)
    cols["ever_delinquent"] = rng.integers(0, 2, n).astype(np.float32)
    cols["months_since_last_delinq"] = rng.integers(1, 36, n).astype(np.float64)
    cols["n_prior_delinq_episodes"] = rng.integers(0, 4, n).astype(np.float64)
    return pl.DataFrame(cols)


def _fixture(n_loans=6, *, t0=201412, seed=0):
    rng = np.random.default_rng(seed)
    fit = F.prepare_raw(_rows_aug(500, loans=np.arange(500), period_ym=t0, orig_ym=201001,
                                  age=20.0, rem=340.0, upb=2e5,
                                  states=[STATES[i % 7] for i in range(500)], rng=rng))
    scaler = F.Scaler.fit(fit, cols=H.AUG_CONTINUOUS)
    vocab = F.Vocab.fit(fit, cols=H.AUG_CATEGORICAL)
    torch.manual_seed(seed)
    nb = len(F.BINARY) + len(H.HIST_BINARY)
    model = N.build(scaler, vocab, depth=2, dropout=0.0, n_binary=nb)
    pred = HP.FFHistPredictor(model, scaler, vocab, device="cpu")
    loans = [f"L{i}" for i in range(n_loans)]
    base = F.prepare_raw(_rows_aug(n_loans, loans=loans, period_ym=t0, orig_ym=201001, age=24.0,
                                   rem=rng.integers(180, 356, n_loans).astype(float),
                                   upb=rng.uniform(8e4, 4e5, n_loans),
                                   states=["current"] * n_loans, rng=rng))
    seed_all = {"ever": rng.integers(0, 2, n_loans).astype(bool),
                "last_mi": np.full(n_loans, HP._t0_mi(t0) - 5, np.int64),
                "nepi": rng.integers(0, 3, n_loans).astype(np.int64),
                "prev_isd": rng.integers(0, 2, n_loans).astype(bool),
                "prev_vidx": rng.integers(0, vocab.vocab_size("prev_state"), n_loans).astype(np.int64)}
    return pred, base, t0, seed_all


def test_vectorized_equals_reference():
    """``simulate_paths_hist`` (batched) reproduces the per-loan reference oracle bit-for-bit on CPU
    (identical RNG streams + window assembly): a clean differential gate for the production path."""
    pred, base, t0, seed_all = _fixture(seed=3)
    d = HP._differential_check_hist(pred, base, k=2015, t0=t0, horizon=HORIZON, n_paths=64,
                                    seed=0, seed_all=seed_all)
    assert d["bit_exact"], d


def test_h1_mc_mean_converges_to_direct():
    """At H=1 every path scores the identical anchor row (engineered features from the seed, no
    sampling yet), so the MC mean over paths → those deterministic probabilities (post M20) to
    Monte-Carlo tolerance — the regression-guard identity in MC form, for an untrained model."""
    pred, base, t0, seed_all = _fixture(seed=4)
    sim = HP.simulate_paths_hist(pred, base, k=2015, t0=t0, horizon=1, n_paths=8000, seed=0,
                                 seed_all=seed_all)
    # Independent "direct": encode the h=1 row (advance_frame h=1 == base) with the seed features.
    fcont, fcat, fbin = pred.encode_future(base, t0, 1)
    origin = HP.MX._origin_classes(base)
    t0_mi = HP._t0_mi(t0)
    cont, cat, binb = pred.assemble(fcont[0], fcat[0], fbin[0], origin, seed_all["prev_vidx"],
                                    seed_all["ever"], (t0_mi - seed_all["last_mi"]).astype(np.float64),
                                    seed_all["nepi"].astype(np.float64))
    direct = MC._apply_post(pred.score(cont, cat, binb), origin, None, zero_impossible=True)
    err = np.abs(sim["h1_dist"] - direct).max()
    assert err < 0.05, err
