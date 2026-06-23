# M21 — Predictor seam + horizon parameter + calibrator hook: notes + verification

**Status:** implemented + **locally green** (pytest + CPU smoke at the COVID anchor k=2020).
Branch `m21-predictor-seam` (off `m20c-verify-gbt-live`, which carries M20's `pool.py`).
Spec: `ECONOMIC_ENGINE.md §3,§4`; ADR: `ADR-001-economic-engine-seam.md`; task: `04_TASKS.md` M21.
Scope: **seam + horizon + calibrator HOOK only** — *not* M22 loan-level pricing, *not* M23 calibration fitting.

## 1. What was implemented (`pool.py`, `config.py`, `tests/test_pool.py`)

- **Predictor seam (ADR-001 §3.1).** `class Predictor(Protocol).origin_scores(frame_h) -> list[(n,7)]`.
  - `TorchPredictor(model, scaler, vocab, device)` — `origin_scores` = the moved `_chunk_matrices`
    encode + the **verbatim** former `_origin_scores` body (`_origin_scores_tensors`; only
    `model→self.model`, the unused `device` arg dropped). Byte-identical to the pre-seam scoring
    **by construction** (and now **by test** — see §3 Part A).
  - `EmpiricalPredictor(emp)` — feature-free; the **stub second Predictor** for the model-agnostic
    Accept (a real model, no throwaway code). `EnsemblePredictor(members)` — shared-encode mean.
  - The free `_origin_scores` function is **removed**; `_chunk_matrices` repointed at
    `TorchPredictor._origin_scores_tensors` (shared single encode preserved).
- **Calibrator seam (§3.2).** `class Calibrator(Protocol)` + `identity_calibrator` no-op default;
  threaded into the new engine, applied to the **raw** per-origin scores **before** assembly (§4 item 2).
- **The agnostic engine (§3.3).** `roll_forward_predictor(predictor, base, t0, *, horizon, snapshots,
  calibrator, absorb, zero_impossible, chunk, capture_h1)` — ONE injected predictor; per step
  `origin_scores → calibrate → assemble_matrix → _zero_impossible → absorb_rows → compose`; H∈{1,3,6,12}
  are **snapshots of a single roll to 12** (intermediates free). Bounded-memory chunks. **No change** to
  `compose`/`absorb_rows`/`assemble_matrix`/`cashflow_engine`.
- **`config.HORIZONS = (1,3,6,12)`** (spec §8).
- **Legacy `roll_forward` (M14/M15) untouched** except: (i) torch scoring now flows through the seam
  (byte-identical); (ii) a behaviour-preserving `zero_impossible: bool = True` kwarg — default keeps the
  M20 production path M14/M15 bank; `False` exposes the raw path for the regression guard.
- **`verify_h1` restored + staged + legacy-guarded** (see §2); `run_horizons` + `--smoke/--horizons/--xcheck`
  CPU driver (standing rule 3).

## 2. `verify_h1` — two independent proofs, both vs `_score_direct` (the un-refactored reference)

- **Part A — legacy regression guard** (the *old* `verify_h1` assertion, kept): legacy
  `roll_forward(…, horizon=1, zero_impossible=False)` h=1 == `_score_direct` **byte-identical**, per the
  user's requirement — the repoint must be *provably* non-altering, not non-altering by construction,
  because M14/M15 results are banked.
- **Part B — staged seam check** (localises a divergence instead of needing a bisection):
  **(a)** h=1 input frame == eval input frame · **(b)** predictor per-origin output (gathered at each loan's
  true origin) == `_score_direct` · **(c)** composed h=1 (raw, `zero_impossible=False`) == `_score_direct`
  EXACTLY. Reports the first failing stage.
- **Bit-exactness on the raw path is RESTORED** (ADR-001 action item 2): M20's zeroing had broken the
  `current`-origin bit-match; running the identity un-zeroed restores Δ=0.
- The expensive committed-run cross-check vs `evaluate.score_window` is **gated behind `--xcheck`**
  (rides a GPU/full session, like M20b). `XCHECK_TOL = 1e-5` is **unchanged** — gated, not relaxed.

## 3. Verification evidence (standing rule 1 — paste the numbers)

**(0) `pytest tests/test_pool.py` — 18 green** (12 pre-existing + 6 new M21: snapshots==compose,
identity-calibrator no-op, calibrator-before-assemble, absorb first-passage, empirical-stub-flows,
zero_impossible flag). Pure NumPy/polars stub, no SSD/GPU. Green **before and after** the change.

**(1) Staged h=1 + legacy guard** — `pool --k 2020 --device cpu --max-loans 10000 --verify` (Mac, SSD; 42s):
```
Part A: legacy roll_forward (raw path) byte-identical vs _score_direct
   [PASS] empirical · [PASS] logit · [PASS] nn · [PASS] ensemble   (all 10,000 rows, byte-identical=True)
Part B: staged seam check (a)frame -> (b)predictor -> (c)composed
   [PASS] empirical · [PASS] logit · [PASS] nn · [PASS] ensemble   (a=b=c=True for every model)
```
k=2020 is a **key window** (COVID anchor), so this covers the **8-net ensemble** too — the repoint is
proven non-altering for the banked ensemble path, not just logit/nn.

**(2) Multi-horizon plumbing** — `pool --k 2020 --device cpu --smoke --horizons` (50k cap; 23s):
```
rolling 50,000 loans to H=12 (single roll), snapshots [1, 3, 6, 12]
          logit: H=[1,3,6,12] produced  shapes_ok=True valid=True  E[prepaid] h1=0.0111 h3=0.0349 h6=0.0709 h12=0.1347
 empirical_stub: H=[1,3,6,12] produced  shapes_ok=True valid=True  E[prepaid] h1=0.0137 h3=0.0404 h6=0.0792 h12=0.1519
 peak RSS 2,314 MB  (single roll to 12, 4 free snapshots)
```
Four snapshots from ONE roll; valid distributions; E[prepaid] monotone in H (cumulative prepayment grows);
the **empirical stub flows through the identical engine** (model-agnostic). Bounded memory — a 50k smoke
slice, **not** the full alive-loan population (that ≥3-anchor × H grid is **M24**).

## 4. Gate status

- [x] Predictor protocol + `TorchPredictor` verbatim extraction; old call site repointed.
- [x] `horizon` parameter / H∈{1,3,6,12} snapshots of a single roll.
- [x] optional `calibrator` (identity default, before assembly) + optional `absorb`.
- [x] no change to `compose`/`absorb_rows`/`assemble_matrix`/`cashflow_engine`.
- [x] `verify_h1` bit-exact on the raw path restored; tolerance unchanged.
- [x] legacy regression guard (repoint provably non-altering) — Part A above.
- [x] `--smoke` CPU path; pytest green before and after.
- [ ] `--xcheck` committed-run cross-check (≤1e-5) — rides the next GPU session (non-gating for M21).

**Next (Thursday plan):** M22 loan-level pricing (thin `cashflow_engine`-per-loan wrapper + small-pool
exact tests), then M24's full-scale H×regime grid on the GPU pod.
