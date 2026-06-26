# ADR-002: A `SeqPredictor` + Monte-Carlo path simulation for H>1 sequence-model pricing

**Status:** Proposed
**Date:** 2026-06-26
**Deciders:** Felipe (author); supervisor (asked for sequence-model pricing + Monte-Carlo paths)
**Implemented by:** W3 / M28 in `writeup/memos/final_week_plan.md`; extends ADR-001
**Supersedes:** the deferral flagged in ADR-001 §"To revisit" / Action Item 5

## Context

ADR-001 introduced one `Predictor` seam so the roll-forward and cashflow engine score any
learner with identical engine math, and made `horizon` a parameter. That seam assumes the
predictor is a function of **point-in-time covariates at month h**: `roll_forward` advances a
single feature frame, scores it, composes the resulting transition matrix, and repeats —
exact under a first-order Markov model, "no Monte Carlo required" (`chapter2`,
`subsec:meth-rollforward`).

The sequence models (GRU `seq_model.py`, transformer `seq_transformer.py`) break that
assumption by construction: each prediction consumes a **trajectory** (the trailing-T monthly
feature vectors), not a point-in-time row. At H=1 this is harmless — one composed step equals
the direct one-step prediction, so the GRU already drops into `roll_forward` unchanged
(ADR-001; `chapter2`). At H>1 it does not: the month-(h+1) distribution depends on the
*realised path* through months 1..h, which matrix composition averages away. Pricing a
sequence model across horizons therefore needs a different roll-forward — **simulate paths**
rather than compose matrices.

This is the exact item ADR-001 deferred ("if M26 needs per-loan history, the signature must
carry sequence context — flagged as a likely ADR-002"). M26/M27a succeeded and the supervisor
asked for sequence-model pricing across horizons, so the decision is now live.

Forces: (1) preserve the controlled comparison — the cashflow/valuation backbone must stay the
**same `cashflow_engine`** used for every other model, so a price difference is attributable to
the learner, not the engine; (2) Karpathy surgical discipline — add a second roll-forward, do
not fork the engine; (3) the H=1 identity must still hold exactly (regression guard), now in
MC form (MC mean → the direct prediction as paths → ∞).

Constraint: must not perturb the validated matrix-composition path used by
empirical/logit/GBT/net/ensemble (M13–M24). Sequence pricing is **additive**, not a rewrite.

## Decision

Add **one new predictor adapter and one new roll-forward**, leaving the engine untouched:

1. **`SeqPredictor`** — conforms to the same injection contract as `TorchPredictor`/`GBTPredictor`,
   but its scoring unit is a *per-loan history* (the trailing-T sequence ending at the simulated
   month), not a single advanced frame. It wraps a fitted `SeqGRU`/`SeqTransformer` + scaler/vocab.
2. **`simulate_paths(...)`** — a Monte-Carlo roll-forward that replaces matrix *composition* for
   sequence models only. For each loan, draw `N` trajectories: at each month h, build the
   trailing-T window from the path so far, score the next-state distribution with `SeqPredictor`,
   sample the next state, advance covariates deterministically (age/term/seasonality evolve, UPB
   and the macro block frozen at t0 — identical evolution rules to `advance_frame`), and absorb
   terminal states. Aggregate the `N` paths to the per-loan state distribution and SMM path.
3. **Reuse everything downstream unchanged** — the per-loan SMM path feeds the **same**
   `cashflow_engine` (a one-loan pool, per ADR-001 §M22), then aggregates to the pool. The
   calibrator seam (ADR-001 §M23) applies to each month's raw scores before sampling, exactly as
   it applies before assembly in the composition path.

This is "**add a second roll-forward behind the same predictor/cashflow contract**" rather than
"**make the matrix engine path-dependent**" or "**fork the cashflow engine per model**".

## Options Considered

### Option A: `SeqPredictor` + a separate `simulate_paths` MC roll-forward, same cashflow engine (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Med — one new MC roll-forward; the predictor and cashflow contracts are reused |
| Diff size | Additive — zero change to `compose`/`absorb_rows`/`cashflow_engine`/`evaluate.py` |
| Comparison control | **Strong** — same valuation backbone as every other model; only the state-distribution *generator* differs (composition vs simulation), which is the genuine modelling difference |
| Forward fit | Clean — GRU and transformer share the one MC path via the same `SeqPredictor` |
| Risk | MC variance + cost; controlled by a paths-convergence check and the H=1 MC↔direct identity guard |

**Pros:** keeps the cashflow/aggregation identity and closed-form tests intact; one MC path serves
both sequence architectures; the composition path for the other five models is untouched.
**Cons:** Monte-Carlo sampling introduces variance and per-loan GPU inference cost at H>1 (mitigations below).

### Option B: Force the sequence model into matrix composition (score it as if point-in-time)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Diff size | Tiny |
| Comparison control | **Broken** — composing a per-step sequence score discards the path dependence that is the entire point of the model; H>1 prices would be wrong, not approximate |
| Risk | High — silently misprices; defeats the contribution |

**Cons:** mathematically invalid beyond H=1; would report a path-dependent model as if Markov.

### Option C: Fork a sequence-specific cashflow/valuation engine

| Dimension | Assessment |
|-----------|------------|
| Complexity | High — a parallel engine |
| Comparison control | **Worst** — two engines are two chances to diverge; the closed-form tests must be re-proven per copy (the ADR-001 Option C objection) |

**Cons:** duplicates the exact code whose identity across models is the contribution.

## Trade-off Analysis

The dominant force is the same as ADR-001: the price comparison is only meaningful if the
**valuation backbone is identical** across models. Option A preserves that — the only thing that
changes for the sequence models is *how the per-loan state distribution / SMM path is produced*
(simulation instead of composition), which is the real, intended modelling difference, not an
engine difference. Option B is the cheap path but is wrong above H=1. Option C reintroduces the
duplication ADR-001 already rejected.

The cost of A is Monte-Carlo variance and compute. Both are bounded: variance by a paths
convergence check (raise `N` until the price moves < a set tolerance at the stressed anchors,
where path dependence is largest), and cost by simulating on a **subsample of loans per anchor**
(the pricing exhibit is at ≥3 regime anchors, not the full panel) with batched GPU inference over
paths. The H=1 case is the regression guard: the MC mean must converge to the direct one-step
prediction, tying the new path back to the validated engine.

## Consequences

**Easier:**
- Sequence-model pricing across H ∈ {1,3,6,12} lands in the existing T5.1/F5.2 as one more model
  column, via the same `cashflow_engine` and aggregation as every other learner.
- The transformer reuses the GRU's MC path with no extra pricing code (same `SeqPredictor`).
- Loan- and pool-level sequence prices reconcile by the same aggregation identity (one-loan pool),
  so the ADR-001 §M22 reconciliation test extends for free.

**Harder / to watch:**
- **MC variance:** report the convergence check (`N` vs price stability) so the priced numbers are
  defensible; seed the sampler for reproducibility.
- **Cost:** per-loan, per-path, per-month GPU inference is the heavy step; cap loans-per-anchor and
  batch paths; macro frozen at t0 means no scenario branching (keeps it tractable).
- **Frozen-t0 honesty:** the same conditional-on-anchor caveat as the composition path
  (`chapter2`) applies, and is arguably sharper under simulation — state it.

**To revisit:**
- If MC cost proves prohibitive within the week, the fallback (per `final_week_plan.md` risk
  ladder) is to keep sequence models as the **loan-level (H=1)** headline and leave H>1 pricing as
  future work — the existing conclusions paragraph already covers it. This ADR is then "Accepted,
  not yet implemented."

## Action Items

1. [ ] M28 — implement `SeqPredictor` (wraps `SeqGRU`/`SeqTransformer` + scaler/vocab; scores a
   trailing-T window).
2. [ ] M28 — implement `simulate_paths(...)`: per-loan MC roll-forward reusing `advance_frame`
   covariate evolution and the terminal-state absorption rules; returns per-loan state
   distribution + SMM path.
3. [ ] M28 — H=1 regression guard: MC mean over paths converges to the direct one-step prediction
   to MC tolerance (the ADR-001 Accept-#2 identity, in MC form).
4. [ ] M28 — paths-convergence check (`N` sweep at a stressed anchor) + seeded sampler; document `N`.
5. [ ] M28 — feed the per-loan SMM path through the existing `cashflow_engine`; extend the loan→pool
   aggregation-identity test to the sequence models.
6. [ ] M28 — apply the ADR-001 `Calibrator` to each month's raw scores before sampling (no-op when
   disabled).
7. [ ] Produce GRU (+transformer) columns in T5.1/F5.2 at ≥3 regime anchors × H ∈ {1,3,6,12}.
