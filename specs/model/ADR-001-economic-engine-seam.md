# ADR-001: One injected `Predictor` seam for a model-, horizon-, and calibration-agnostic economic engine

**Status:** Proposed
**Date:** 2026-06-19
**Deciders:** Felipe (author); supervisor sign-off on the pricing-target framing (already given)
**Implemented by:** M21–M23 in `04_TASKS.md`; specified in `ECONOMIC_ENGINE.md`

## Context

The post-supervision plan makes the cashflow/pricing engine the critical path and asks it to vary along three axes the current code does not expose:

- **Model** — score the net today, GBT later, and a sequence model (RNN/transformer) if the M26 spike succeeds.
- **Horizon** — produce pool/loan outcomes at H ∈ {1, 3, 6, 12} months, not the fixed 12 of `03_POOL_LEVEL §1`.
- **Calibration** — compare raw vs per-window-calibrated levels, per horizon (M23).

The machinery already exists and is *already* mostly agnostic: `pool.compose`, `pool.absorb_rows`, `pool.assemble_matrix`, `pool.cashflow_engine`, and the NLL/AUC math in `evaluate.py` do not depend on which learner or horizon produced the probabilities. The coupling is concentrated in one place: the roll-forward's scoring path is torch-specific (`pool._forward_probs(model, cont, cat, binb)` → `pool._origin_scores(...)` → `pool._chunk_matrices(..., torch_models, ...)`), and the horizon is a literal `1..12` inside `pool.roll_forward`.

Forces: (1) the dissertation's whole credibility rests on **byte-identical frozen test rows and identical engine math across models** — any per-model code path threatens that control; (2) the author follows Karpathy surgical-change discipline — minimum diff, no speculative abstraction; (3) `06_GBT_BASELINE §5` already pre-committed to "one model-agnostic predictor seam in `pool.py`" for M19, so this decision formalizes and generalizes an existing intent rather than inventing one.

Constraint: this must not become a rewrite. The engine is validated (M13–M15 closed-form unit tests pass); the change must preserve those guarantees and keep `H=1` exactly equal to `evaluate.py`.

## Decision

Introduce **one injection seam** — a `Predictor` protocol whose sole method returns the per-origin 7-vectors at the covariates already advanced to month *h* — and drive `roll_forward` through it. Make **horizon a parameter** of `roll_forward`, and thread an **optional `Calibrator`** transform applied to predictor output before matrix assembly. Everything downstream (`assemble_matrix`, `absorb_rows`, `compose`, `cashflow_engine`, `evaluate.py`) is untouched. Loan-level pricing is not new machinery: it is `cashflow_engine` called per loan (a one-loan pool).

This is "**inject one seam**" rather than "**branch per model**" or "**duplicate per model**".

## Options Considered

### Option A: One injected `Predictor` protocol + horizon parameter + calibrator transform (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low — extract one protocol from the existing `_origin_scores`; parameterize one loop bound; add one optional transform |
| Diff size | Small and surgical — engine math files unchanged; only the scoring seam and the loop bound move |
| Comparison control | **Strongest** — identical engine math by construction; only `Predictor.origin_scores` differs across models |
| Forward fit (sequence models) | Clean — `SeqPredictor` conforms to the same protocol; no engine change when B lands |
| Risk | A protocol indirection adds one call layer; mitigated by the `H=1 == evaluate.py` regression guard |

**Pros:** matches `06 §5`'s pre-committed seam; the model-specific surface shrinks to one method; horizon and calibration become a parameter and a transform, not new code paths; loan-level pricing falls out of the existing engine.
**Cons:** introduces a protocol abstraction for (today) two implementations — justified only because a third (sequence model) is on the roadmap and GBT (M19) already needs it.

### Option B: Conditional branches inside the existing torch path

| Dimension | Assessment |
|-----------|------------|
| Complexity | Med — `if torch / if gbt / if seq` inside `_origin_scores`, `_chunk_matrices`, `roll_forward` |
| Diff size | Grows with every model; touches the hot loop repeatedly |
| Comparison control | Weaker — model-specific branches inside shared code invite subtle divergence (dtype, ordering, weighting) |
| Forward fit | Poor — each new learner reopens the engine and its tests |
| Risk | High — the thing most likely to break the "identical engine math" claim the dissertation rests on |

**Pros:** no new protocol; smallest possible diff for exactly two models.
**Cons:** every added learner edits validated engine code; branch drift is precisely the failure the frozen-test-row discipline exists to prevent.

### Option C: Separate roll-forward/cashflow implementation per model family

| Dimension | Assessment |
|-----------|------------|
| Complexity | High — parallel engines |
| Diff size | Large; duplicated logic |
| Comparison control | **Worst** — two engines are two chances to diverge; closed-form tests must be re-proven per copy |
| Forward fit | Poor — N learners → N engines |
| Risk | High — duplication of the exact code whose identity across models is the contribution |

**Pros:** each engine can be tuned to its learner.
**Cons:** defeats the purpose; the comparison is only meaningful if the engine is the same object for every model.

## Trade-off Analysis

The dominant force is **preserving the controlled comparison**: a net-vs-GBT-vs-sequence price difference is only attributable to the *learner* if the composition and cashflow code is provably identical across them. Option A enforces that structurally — there is one engine, and the only thing that varies is the injected predictor. Options B and C both reintroduce model-specific code into the shared path, which is the precise risk the frozen-test-row protocol was built to eliminate. The cost of A — one protocol indirection — is small and is already a sunk decision from `06 §5` (the M19 GBT seam). Horizon-as-parameter vs. horizon-specific code paths is the same argument in miniature: a parameter cannot diverge across horizons, three code paths can.

The only real objection to A is "abstraction for two implementations." That is answered by the roadmap: GBT (M19) and the sequence model (M26/M27) are the third and fourth implementations, both already planned, both needing exactly this seam.

## Consequences

**Easier:**
- Adding GBT (M19) and the sequence model (M27) to pricing becomes "write a `Predictor`," with zero engine change.
- The 1/3/6/12-month sweep is one parameter; the calibrated-vs-raw sweep is one boolean.
- Loan-level and pool-level prices reconcile by construction (both sum the same per-loan cashflows), giving a free aggregation-identity unit test.

**Harder / to watch:**
- One indirection layer in the hot loop — guard with the `H=1 == evaluate.py` exact-match regression test so any seam regression is caught immediately.
- The calibrator must act on **raw per-origin scores before** `assemble_matrix`/`absorb_rows` (it calibrates model outputs, not composed/absorbed chains) — specified in `ECONOMIC_ENGINE §4`.

**To revisit:**
- If the sequence-model spike (M26) needs per-loan *history* (not just point-in-time covariates), the `Predictor.origin_scores(frame_h)` signature must carry sequence context — flagged now as a likely **ADR-002** at M26 go/no-go, not pre-built here (no speculative generality).

## Action Items

1. [ ] M21 — extract `Predictor` protocol from `pool._origin_scores`; wrap the torch path as `TorchPredictor`; make `horizon` a parameter of `roll_forward`.
2. [ ] M21 — add the `H=1 == evaluate.py` exact-match regression test (the M13 Accept-#2 check, parameterized).
3. [ ] M23 — add the optional `Calibrator` seam, applied to raw per-origin scores before assembly; no-op identity when disabled.
4. [ ] M22 — loan-level pricing as `cashflow_engine` per one-loan pool; add the loan→pool aggregation-identity test.
5. [ ] Defer the sequence-context signature question to ADR-002 at the M26 go/no-go.
