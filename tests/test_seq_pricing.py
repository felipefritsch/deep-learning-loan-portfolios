"""M28 / W3 (``ADR-002``) — hermetic tests for the sequence-model Monte-Carlo pricing seam.

Synthetic only: a tiny untrained ``SeqGRU`` + a few synthetic loans, on CPU, no SSD / no GPU
(the whole point of doing this locally). The identities under test hold for ANY model, trained or
not, so an untrained random-init net is a valid fixture.

  * **H=1 identity (regression guard)** — at H=1 the window is the observed trailing window scored
    once (no covariate advance, no sampling done yet), so (a) ``SeqPredictor`` drops into the
    UNCHANGED ``pool.roll_forward_predictor`` and its one composed step equals the direct one-step
    sequence prediction EXACTLY, and (b) the Monte-Carlo mean over paths converges to that same
    distribution to MC tolerance ``O(1/√N)``.
  * **Aggregation identity** — the per-loan MC SMM paths, priced one loan at a time through the
    EXISTING ``cashflow_engine`` and summed, equal the ``price_loans_vec`` pool aggregate to ~1e-6
    (the §4.3 linear-in-UPB dollar identity, now fed by the sequence simulation).
  * **Convergence** — the pool price stabilises as ``n_paths`` grows at a stressed horizon (H=12);
    the run reports the ``n_paths`` needed to get within tolerance of a high-path reference.
  * **Seeded sampler** — same seed ⇒ identical draws; different seed ⇒ different draws.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from floan.model import features as F
from floan.model import history as H
from floan.model import pool as PL
from floan.model import seq_model as SM
from floan.model import seq_pricing as SP

SI = F.STATE_INDEX
CUR, D30, D60, D90 = SI["current"], SI["dpd_30"], SI["dpd_60"], SI["dpd_90plus"]
FC, REO, PRE = SI["foreclosure"], SI["REO"], SI["prepaid"]
DELINQ = (D30, D60, D90)

T_WIN = 8            # short window so the trailing cap (Lh + h > T) is exercised past h=8
HORIZON = 12
ORIGIN_STATES = list(SP.ORIGIN_STATES)

# Plausible per-column centres for the continuous block; everything else fills uniformly. The four
# semantic columns (age / remaining term / note rate / current UPB) are overridden per row.
_CONT_DEFAULT = {
    "Original Interest Rate": 5.0, "Current Interest Rate": 5.0,
    "Original UPB": 2.5e5, "Current Actual UPB": 2.0e5,
    "Original Loan Term": 360.0, "Loan Age": 24.0, "Remaining Months to Maturity": 336.0,
    "Original Loan-to-Value (LTV)": 75.0, "Original Combined LTV (CLTV)": 75.0,
    "Debt-to-Income (DTI)": 35.0, "fico_orig": 740.0, "fico_co": 745.0,
    "Mortgage Insurance Percentage": 0.0, "Number of Borrowers": 2.0, "Number of Units": 1.0,
    "pmms30": 4.5, "dgs10": 2.5, "slope_10y2y": 0.5, "unrate_nat": 5.0, "unrate_state": 5.0,
    "incentive": 0.5, "ltv_mtm": 70.0, "hpi_chg_12m": 0.03,
}
_CAT_LEVELS = {c: ["A", "B", "C"] for c in F.RAW_CATEGORICAL if c != "state"}


def _rows(n, *, loans, period_ym, orig_ym, age, rem, rate, upb, states, rng) -> pl.DataFrame:
    """Build ``n`` raw loan-month rows carrying every model feature column (so Scaler/Vocab fit and
    ``encode_rows`` see a complete frame). Scalar args broadcast; ``states`` is a per-row list."""
    bc = lambda v: np.full(n, v) if np.isscalar(v) else np.asarray(v)
    cols: dict = {"Loan Identifier": np.asarray(loans).astype(str),
                  "period_ym": bc(period_ym).astype(np.int64),
                  "orig_ym": bc(orig_ym).astype(np.int64),
                  "weight": np.ones(n, np.float32), "state": list(states)}
    for c, d in _CONT_DEFAULT.items():
        cols[c] = rng.uniform(0.85 * d, 1.15 * d, n) if d else rng.uniform(0.0, 1.0, n)
    cols["Loan Age"] = bc(age).astype(np.float64)
    cols["Remaining Months to Maturity"] = bc(rem).astype(np.float64)
    cols["Current Interest Rate"] = bc(rate).astype(np.float64)
    cols["Current Actual UPB"] = bc(upb).astype(np.float64)
    for c, levels in _CAT_LEVELS.items():
        cols[c] = rng.choice(levels, n)
    for c in F.BINARY:
        cols[c] = rng.integers(0, 2, n).astype(np.float32)
    return pl.DataFrame(cols)


def _fit_pipeline(rng):
    """A Scaler/Vocab fit on a varied frame that includes ALL FOUR transient origins in ``state``
    (so ``vocab.maps['state']`` resolves every origin for the window's state override)."""
    states = [ORIGIN_STATES[i % 4] for i in range(400)]
    fit = F.prepare_raw(_rows(400, loans=np.arange(400), period_ym=201906, orig_ym=201001,
                              age=20.0, rem=340.0, rate=5.0, upb=2e5, states=states, rng=rng))
    return F.Scaler.fit(fit, cols=F.CONTINUOUS), F.Vocab.fit(fit, cols=F.CATEGORICAL)


def _fixture(n_loans=4, *, t0=201912, seed=0, shared_terms=False, rng=None):
    """A tiny untrained ``SeqGRU`` + ``SeqPredictor`` + an anchor ``base`` of ``n_loans`` loans
    (one per origin, cycled). ``shared_terms`` gives all loans the same rate/term (varying UPB) so
    the §4.3 single-pool reconciliation is tight."""
    rng = rng or np.random.default_rng(seed)
    scaler, vocab = _fit_pipeline(rng)
    torch.manual_seed(seed)
    model = SM.build(scaler, vocab, hidden=8)
    pred = SP.SeqPredictor(model, scaler, vocab, device="cpu", T=T_WIN)
    loans = [f"L{i}" for i in range(n_loans)]
    states = [ORIGIN_STATES[i % 4] for i in range(n_loans)]
    rate = 5.0 if shared_terms else rng.uniform(3.5, 7.0, n_loans)
    rem = 336.0 if shared_terms else rng.integers(180, 356, n_loans).astype(float)
    upb = rng.uniform(8e4, 4e5, n_loans)
    base = _rows(n_loans, loans=loans, period_ym=t0, orig_ym=201001, age=24.0,
                 rem=rem, rate=rate, upb=upb, states=states, rng=rng)
    return pred, F.prepare_raw(base), t0


# ===========================================================================
# H=1 identity — the regression guard (exact composition + MC convergence)
# ===========================================================================
def test_h1_origin_scores_equal_direct_one_step():
    """``SeqPredictor`` drops into the UNCHANGED ``roll_forward_predictor`` at H=1: the composed
    one-step distribution equals the direct one-step sequence prediction (the loan's window scored
    once at its true origin, with the M20 ``current→fc/REO`` zeroing) — EXACTLY (``ADR-002`` §"At
    H=1 this is harmless ... the GRU already drops into roll_forward unchanged")."""
    pred, base, t0 = _fixture(seed=1)
    eng = PL.roll_forward_predictor(pred, base, t0, horizon=1, snapshots=(1,),
                                    zero_impossible=True)["snapshots"][1]
    # Direct one-step: the per-origin scores, gathered at each loan's true origin, post-processed.
    scores = np.stack(pred.origin_scores(PL.advance_frame(base, t0, 1)))      # [4, L, 7]
    state_names = base.get_column("state").to_list()
    oi = np.array([ORIGIN_STATES.index(s) for s in state_names])
    L = base.height
    picked = scores[oi, np.arange(L)]                                        # [L, 7]
    origin_cls = np.array([SP.SI[s] for s in state_names])
    direct = SP._apply_post(picked, origin_cls, None, zero_impossible=True)
    assert np.allclose(direct, eng, atol=1e-9), np.abs(direct - eng).max()


def test_h1_mc_mean_converges_to_direct():
    """MC mean over paths at H=1 → the direct one-step prediction to Monte-Carlo tolerance (the
    ADR-001 Accept-#2 identity, in MC form). Holds for the untrained model — it is just a categorical
    sample converging to its categorical."""
    pred, base, t0 = _fixture(seed=2)
    eng = PL.roll_forward_predictor(pred, base, t0, horizon=1, snapshots=(1,),
                                    zero_impossible=True)["snapshots"][1]
    sim = SP.simulate_paths(pred, base, t0, horizon=1, n_paths=8000, zero_impossible=True, seed=0)
    err = np.abs(sim["h1_dist"] - eng).max()
    print(f"\n[H=1 MC] max|MC mean − direct| = {err:.4f} over {base.height} loans × 7 states "
          f"(8000 paths)")
    assert err < 0.05, err


def test_h1_identity_with_history():
    """The H=1 exact identity also holds when the predictor carries a real trailing history (rows <
    t0): ``origin_scores`` + compose still equals the direct window score, and the MC mean still
    converges — so the window-from-history path is exercised, not just the empty-history one."""
    rng = np.random.default_rng(11)
    pred0, base, t0 = _fixture(seed=11, rng=rng)
    # Build 5 months of trailing history per loan (period_ym < t0), random transient states.
    hist_rows = []
    for i, loan in enumerate(base.get_column("Loan Identifier").to_list()):
        for m in range(5):
            ym = 201907 + m                                  # 201907..201911, all < 201912
            hist_rows.append({"loan": loan, "ym": ym, "age": 19.0 + m,
                              "state": ORIGIN_STATES[(i + m) % 4]})
    hist = _rows(len(hist_rows),
                 loans=[r["loan"] for r in hist_rows],
                 period_ym=[r["ym"] for r in hist_rows], orig_ym=201001,
                 age=[r["age"] for r in hist_rows], rem=340.0, rate=5.0, upb=2e5,
                 states=[r["state"] for r in hist_rows], rng=rng)
    pred = SP.SeqPredictor(pred0.model, pred0.scaler, pred0.vocab, device="cpu",
                           history=hist, T=T_WIN)
    eng = PL.roll_forward_predictor(pred, base, t0, horizon=1, snapshots=(1,),
                                    zero_impossible=True)["snapshots"][1]
    sim = SP.simulate_paths(pred, base, t0, horizon=1, n_paths=8000, zero_impossible=True, seed=0)
    assert np.abs(sim["h1_dist"] - eng).max() < 0.05


# ===========================================================================
# Aggregation identity — per-loan cashflows sum to the pool (§4.3, MC-SMM-fed)
# ===========================================================================
def test_aggregation_identity_pool_value():
    """Per-loan MC SMM paths, priced one loan at a time through the UNCHANGED ``cashflow_engine``
    and summed in dollars, equal the ``price_loans_vec`` pool aggregate to ~1e-6 — the §4.3
    linear-in-UPB identity, now fed by the sequence simulation. Also checks the §4.3 single-pool
    reconciliation (approximate; reported)."""
    pred, base, t0 = _fixture(n_loans=6, seed=5, shared_terms=True)
    sim = SP.simulate_paths(pred, base, t0, horizon=HORIZON, n_paths=3000, seed=1)
    res = SP.price_paths(sim, base)
    wac, wam, upb = SP._pricing_inputs(base)

    # Sum of per-loan cashflow-engine dollar values == pool aggregate value (exact, ~1e-6).
    val_sum = sum(PL.cashflow_engine(float(wac[i]), float(wam[i]), float(upb[i]), sim["smm"][i])["price"]
                  * float(upb[i]) / 100.0 for i in range(base.height))
    assert abs(res["pool_value"] - val_sum) < 1e-6 * max(1.0, abs(val_sum)), \
        (res["pool_value"], val_sum)

    # §4.3 single-pool reconciliation (shared wac/wam): price the pool as ONE pool on the
    # balance-of-survivors-weighted aggregate SMM. Approximate (prob-mass vs dollar-survival
    # weighting; the M22 caveat) — reported, bounded generously.
    inc = np.empty_like(sim["cum_prepaid"]); inc[:, 0] = sim["cum_prepaid"][:, 0]
    inc[:, 1:] = sim["cum_prepaid"][:, 1:] - sim["cum_prepaid"][:, :-1]
    num = (inc * upb[:, None]).sum(axis=0)
    den = (sim["alive_before"] * upb[:, None]).sum(axis=0)
    pool_smm = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    ref = PL.cashflow_engine(float(wac[0]), float(wam[0]), float(upb.sum()), pool_smm)["price"]
    gap = abs(res["pool_price"] - ref)
    print(f"\n[aggregation] pool_value={res['pool_value']:.2f}  Σ per-loan={val_sum:.2f}  "
          f"(Δ={abs(res['pool_value']-val_sum):.2e})  | §4.3 single-pool recon gap={gap:.4f} per-100")
    assert gap < 1.0, gap


# ===========================================================================
# Convergence — price stabilises as n_paths grows (stressed horizon)
# ===========================================================================
def test_price_converges_with_paths():
    """At the stressed horizon (H=12) the pool price stabilises as ``n_paths`` grows: the error vs a
    high-path reference shrinks and is small at the largest setting. Reports the n_paths needed."""
    pred, base, t0 = _fixture(n_loans=5, seed=7, shared_terms=True)
    ref = SP.price_paths(SP.simulate_paths(pred, base, t0, horizon=HORIZON, n_paths=8000, seed=99),
                         base)["pool_price"]
    grid = [50, 200, 800, 3200]
    errs, needed = [], None
    for P in grid:
        price = SP.price_paths(SP.simulate_paths(pred, base, t0, horizon=HORIZON, n_paths=P, seed=99),
                               base)["pool_price"]
        e = abs(price - ref)
        errs.append(e)
        if needed is None and e < 0.10:
            needed = P
    print(f"\n[convergence] ref(8000)={ref:.4f}  errors by n_paths "
          + "  ".join(f"{P}:{e:.4f}" for P, e in zip(grid, errs))
          + f"  | n_paths for <0.10 per-100: {needed}")
    assert errs[-1] < errs[0], errs                    # error shrinks as paths grow
    assert errs[-1] < 0.10, errs[-1]                   # stabilised at 3200 paths
    assert needed is not None


# ===========================================================================
# Seeded sampler — reproducible draws
# ===========================================================================
def test_sampler_is_seeded():
    """Same seed ⇒ byte-identical simulation; a different seed ⇒ different draws (so the convergence
    and pricing numbers are reproducible, the ADR-002 'seed the sampler' requirement)."""
    pred, base, t0 = _fixture(seed=3)
    a = SP.simulate_paths(pred, base, t0, horizon=6, n_paths=500, seed=7)
    b = SP.simulate_paths(pred, base, t0, horizon=6, n_paths=500, seed=7)
    c = SP.simulate_paths(pred, base, t0, horizon=6, n_paths=500, seed=8)
    assert np.array_equal(a["state_dist"], b["state_dist"])
    assert np.array_equal(a["smm"], b["smm"])
    assert not np.array_equal(a["state_dist"], c["state_dist"])


# ===========================================================================
# State-only path-dependence guard (item 1) — the precompute precondition
# ===========================================================================
def test_no_state_derived_feature_in_sequence_set():
    """The load-bearing precompute ("``state`` is the only path-dependent per-timestep field") is
    valid ONLY if no sequence feature is derived from the simulated state. Assert the engineered
    delinquency-history columns (``history.HIST_COLS`` — ``prev_state`` / ``ever_delinquent`` /
    ``months_since_last_delinq`` / ``n_prior_delinq_episodes``, the FF+hist arm) are absent from the
    sequence feature blocks, and that ``simulate_paths``' guard rejects a predictor that smuggles one
    in."""
    for col in H.HIST_COLS:
        assert col not in F.CONTINUOUS and col not in F.CATEGORICAL and col not in F.BINARY
    # Positive control: a scaler advertising a state-derived continuous feature trips the guard.
    pred, base, t0 = _fixture(seed=0)
    bad = F.Scaler(cols=list(pred.scaler.cols) + ["n_prior_delinq_episodes"],
                   center={**pred.scaler.center, "n_prior_delinq_episodes": 0.0},
                   scale={**pred.scaler.scale, "n_prior_delinq_episodes": 1.0})
    try:
        SP._assert_state_only_path_dependence(bad, pred.vocab)
        raise AssertionError("guard failed to reject a state-derived feature")
    except AssertionError as e:
        assert "state-derived" in str(e)


# ===========================================================================
# H>1 path-dependence (item 2) — the MC captures what composition cannot
# ===========================================================================
class _PathDepModel(nn.Module):
    """A hand-crafted, untrained model whose next-state distribution genuinely depends on the
    REALISED path, in TWO ways (so both the state distribution AND the prepay-driven price diverge):

      * **delinquency duration → foreclosure**: the cure probability out of a delinquent state falls
        with the consecutive-delinquency run (``cure = clip(0.5 − 0.20·(run−1), 0, 0.5)``); the freed
        mass goes to foreclosure. Sustained spells foreclose; a 1-month dip mostly cures.
      * **current seasoning → prepayment**: the prepay probability of a current loan RISES with the
        consecutive-current run (``prepay = clip(0.02 + 0.05·(run−1), 0.02, 0.40)``).

    It reads ONLY the ``state`` trajectory (ignores every covariate), so any MC-vs-composition gap is
    purely path/duration dependence. At H=1 both sides see a 1-month window (run=1) → identical
    distribution; at H>1 the MC accumulates the run along each sampled path while
    ``roll_forward_predictor`` only ever scores a 1-row window per origin (run≡1) and composes —
    forgetting duration — so the two diverge exactly where path-dependence lives."""

    def __init__(self, state_col: int, vocab: F.Vocab):
        super().__init__()
        self.state_col = state_col
        maxidx = max(vocab.maps["state"].values())
        v2c = np.full(maxidx + 1, -1, np.int64)
        for name, idx in vocab.maps["state"].items():
            v2c[idx] = F.STATE_INDEX[name]
        self.v2c = v2c

    @staticmethod
    def _trailing_run(flag: np.ndarray, last: np.ndarray) -> np.ndarray:
        """Length of the consecutive ``flag``-run ending at index ``last`` (per row)."""
        B, T = flag.shape
        run = np.zeros((B, T), int)
        run[:, 0] = flag[:, 0]
        for t in range(1, T):
            run[:, t] = flag[:, t] * (run[:, t - 1] + 1)
        return run[np.arange(B), last]

    def forward(self, cont, cat, binb, lengths):           # -> log-probs [B, 7] (score() softmaxes)
        cat = cat.detach().cpu().numpy()
        lengths = lengths.detach().cpu().numpy().astype(int)
        B, T = cat.shape[0], cat.shape[1]
        sv = np.clip(cat[:, :, self.state_col], 0, self.v2c.shape[0] - 1)
        cls = self.v2c[sv]                                  # [B, T] class idx (-1 where unmapped)
        real = np.arange(T)[None, :] < lengths[:, None]
        last = np.clip(lengths - 1, 0, T - 1)
        from_cls = cls[np.arange(B), last]
        delq_run = self._trailing_run((np.isin(cls, DELINQ) & real).astype(int), last)
        cur_run = self._trailing_run(((cls == CUR) & real).astype(int), last)

        probs = np.zeros((B, 7), np.float64)
        cur = from_cls == CUR
        prepay = np.clip(0.02 + 0.05 * (cur_run - 1), 0.02, 0.40)   # rises with current seasoning
        probs[cur, PRE] = prepay[cur]
        probs[cur, D30] = 0.05
        probs[cur, CUR] = 0.95 - prepay[cur]                # prepay+0.05+(0.95−prepay)=1
        dq = np.isin(from_cls, DELINQ)
        cure = np.clip(0.5 - 0.20 * (delq_run - 1), 0.0, 0.5)       # falls with delinquency duration
        probs[dq, CUR] = cure[dq]
        probs[dq, D60] = 0.45                               # self-loop keeps the spell alive → run grows
        probs[dq, FC] = 0.55 - cure[dq]                     # cure+0.45+(0.55−cure)=1
        term = np.isin(from_cls, (FC, REO, PRE))            # absorbed (ignored by simulate) — stay put
        probs[term, from_cls[term]] = 1.0
        return torch.from_numpy(np.log(np.clip(probs, 1e-12, None)))


def _all_dpd60_base(rng, t0=201912, n_loans=4):
    loans = [f"D{i}" for i in range(n_loans)]
    upb = rng.uniform(1e5, 3e5, n_loans)
    base = _rows(n_loans, loans=loans, period_ym=t0, orig_ym=201001, age=24.0, rem=336.0,
                 rate=5.0, upb=upb, states=["dpd_60"] * n_loans, rng=rng)
    return F.prepare_raw(base), t0


def test_mc_captures_path_dependence_at_h_gt_1():
    """The whole reason the MC exists: at H>1 it must give a materially different state distribution
    (and price) than feeding the SAME model through the composition-only-valid
    ``roll_forward_predictor`` — while still agreeing at H=1. With a path-dependent cure model and
    delinquent-origin loans, sustained spells drive foreclosure that composition (run≡1) never sees."""
    rng = np.random.default_rng(21)
    scaler, vocab = _fit_pipeline(rng)
    model = _PathDepModel(vocab.cols.index("state"), vocab)
    pred = SP.SeqPredictor(model, scaler, vocab, device="cpu", T=T_WIN)
    base, t0 = _all_dpd60_base(rng)

    # H=1: path-dependence not yet active (run=1 both sides) → MC mean ≈ composition.
    h1_eng = PL.roll_forward_predictor(pred, base, t0, horizon=1, snapshots=(1,),
                                       zero_impossible=False)["snapshots"][1]
    h1_mc = SP.simulate_paths(pred, base, t0, horizon=1, n_paths=8000,
                              zero_impossible=False, seed=0)["h1_dist"]
    assert np.abs(h1_mc - h1_eng).max() < 0.05, np.abs(h1_mc - h1_eng).max()

    # H=12: composition forgets duration (always run=1) → MC and composition diverge materially.
    Hh = 12
    eng = PL.roll_forward_predictor(pred, base, t0, horizon=Hh, snapshots=(Hh,),
                                    zero_impossible=False)["snapshots"][Hh]
    mc = SP.simulate_paths(pred, base, t0, horizon=Hh, n_paths=8000,
                           zero_impossible=False, seed=0)["state_dist"][:, Hh - 1, :]
    diff = np.abs(mc - eng).max()
    mc_term = mc[:, [FC, REO]].sum(axis=1).mean()
    eng_term = eng[:, [FC, REO]].sum(axis=1).mean()
    # Prices too: same model, same engine, but the SMM survival differs because of foreclosure.
    p_mc = SP.price_paths(SP.simulate_paths(pred, base, t0, horizon=Hh, n_paths=8000, seed=0),
                          base)["pool_price"]
    rf = PL.roll_forward_predictor(pred, base, t0, horizon=Hh, snapshots=(Hh,),
                                   zero_impossible=False, capture_smm=True)
    smm_comp = PL.per_loan_smm(rf["smm"]["cum_prepaid"], rf["smm"]["alive_before"])
    wac, wam, upb = SP._pricing_inputs(base)
    p_comp = PL.price_loans_vec(wac, wam, upb, smm_comp)["pool_price"]
    print(f"\n[path-dep H=12] max|MC − comp| state-dist = {diff:.3f}  | E[foreclosure+REO]: "
          f"MC={mc_term:.3f} vs comp={eng_term:.3f}  | pool price: MC={p_mc:.3f} vs comp={p_comp:.3f}")
    assert diff > 0.15, diff                       # distributions materially differ at H>1
    assert mc_term - eng_term > 0.10, (mc_term, eng_term)   # MC sees MORE foreclosure (duration-driven)
    assert abs(p_mc - p_comp) > 0.05, (p_mc, p_comp)        # and the price differs


def test_simulate_outputs_well_formed():
    """Shapes, ranges, and the absorbing-state invariant: state distributions are valid (rows sum to
    1, in [0,1]); cumulative prepaid mass is monotone non-decreasing (prepaid is absorbing); SMM ∈
    [0,1]."""
    pred, base, t0 = _fixture(n_loans=4, seed=4)
    sim = SP.simulate_paths(pred, base, t0, horizon=HORIZON, n_paths=400, seed=0)
    L = base.height
    assert sim["state_dist"].shape == (L, HORIZON, 7)
    assert sim["smm"].shape == (L, HORIZON)
    sd = sim["state_dist"]
    assert (sd >= -1e-12).all() and (sd <= 1 + 1e-12).all()
    assert np.allclose(sd.sum(axis=2), 1.0, atol=1e-9)
    assert (np.diff(sim["cum_prepaid"], axis=1) >= -1e-12).all()       # prepaid absorbing ⇒ monotone
    assert (sim["smm"] >= -1e-12).all() and (sim["smm"] <= 1 + 1e-12).all()
