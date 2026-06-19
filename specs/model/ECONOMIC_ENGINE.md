# ECONOMIC_ENGINE — Model-, horizon-, and calibration-agnostic pricing engine

> **Goal:** generalize the executed Phase-3 engine of `03_POOL_LEVEL.md` along three axes the supervisor
> prioritized — **model** (net / GBT / future sequence model, via one injected seam), **horizon**
> (H ∈ {1, 3, 6, 12} months, not the fixed 12 of `03 §1`), and **calibration** (raw vs per-window-calibrated
> levels, reported per horizon) — and add **loan-level** pricing alongside pool-level. This spec is the binding
> contract for tasks **M20–M27** in `04_TASKS.md` and is governed by **ADR-001** (the seam decision).
>
> **It does not restate the engine math.** Composition is `03 §3`; the cashflow engine is `03 §5`; pool
> construction is `03 §2`; the NLL/AUC/calibration evaluator is `02 §7`. Those remain the description of the
> underlying machinery. This spec specifies only the **deltas** and the **new invariants**, and **supersedes**
> `03 §1`'s "Horizon: 12 months" and its implicit single-model (torch) assumption.

---

## 1. Where this sits (and what it reuses)

The Phase-3 engine (M13–M15) is built and closed-form-validated. This spec consumes it **unchanged** and adds
a thin agnostic layer. Reused as-is:

- `pool.compose`, `pool.absorb_rows`, `pool.assemble_matrix`, `pool.onehot_origin` — the composition algebra (`03 §3`);
- `pool.cashflow_engine(wac, wam, upb, smm, …)` — the level-pay pass-through (`03 §5`), already taking an arbitrary-length SMM vector;
- `pool.advance_frame(base, t0, h)` — deterministic covariate evolution per horizon step (`03 §3`);
- `smm_paths.py` (per-loan → pool SMM aggregation) and `economics.py` (T4.2/T5.1/F5.2 builders);
- `evaluate.py` — NLL / AUC / calibration math (`02 §7`), unchanged;
- the export, eval pool, window masks and frozen test slices of `02 §3` / `00_OVERVIEW §3.2`.

New code is confined to: a `Predictor` protocol + `TorchPredictor` adapter (extracted from `pool._origin_scores`),
a `horizon` parameter on `pool.roll_forward`, an optional `Calibrator` seam, a loan-level pricing wrapper over
`cashflow_engine`, and the M20 mask correction.

## 2. Locked decisions (binding)

1. **One `Predictor` protocol is the only model-specific surface** (ADR-001). `roll_forward` calls
   `predictor.origin_scores(frame_h) → list[(n_origin, 7)]` and nothing else model-specific. `TorchPredictor`
   wraps today's path; `GBTPredictor` (M19) and `SeqPredictor` (M27) conform without engine change.
2. **Horizon is a parameter, H ∈ {1, 3, 6, 12}.** `H = 1` is the direct one-step prediction and **must equal
   `evaluate.py`'s output exactly** (the M13 Accept-#2 identity, now parameterized) — the engine's regression guard.
3. **Calibration is an injected transform, identity when disabled** (`03`/`06 §4` machinery). Applied to the
   **raw per-origin scores before** `assemble_matrix`/`absorb_rows` — it calibrates model outputs, not composed
   or absorbed chains (see §4).
4. **Loan-level pricing reuses `cashflow_engine` per loan** (a one-loan "pool"); it is not new machinery. It
   **must reconcile to the pool level by the aggregation identity** of §4.
5. **The impossible-cell mask correction (M20) is applied uniformly to every model** — it is a pipeline bug, not
   a GBT artifact — and is **gated before any pricing run** (§3.5).
6. **Macro frozen at t0; deterministic composition, not simulation** — inherited from `03 §3` unchanged. Macro
   scenario paths remain out of scope (future-work hook).

## 3. Interfaces (the contract; rationale in ADR-001)

### 3.1 Predictor seam

```python
class Predictor(Protocol):
    def origin_scores(self, frame_h: pl.DataFrame) -> list[np.ndarray]:
        """One (n_origin, 7) probability block per non-terminal origin (current, dpd_30, dpd_60,
        dpd_90plus), at covariates already advanced to month h. Rows sum to 1, float64."""
```

`TorchPredictor(model, scaler, vocab, device)` wraps the existing `pool._forward_probs`/`_origin_scores`.
`GBTPredictor(booster, vocab)` and `SeqPredictor(...)` are added by M19/M27 with no change to the engine.

### 3.2 Calibrator seam

```python
class Calibrator(Protocol):
    def __call__(self, probs: np.ndarray, origin: int) -> np.ndarray: ...   # identity when disabled
```

Per-window temperature scaling fit on the **val** slice only (`06 §4`); per-class isotonic + renormalise as fallback.

### 3.3 Roll-forward (horizon parameterized, predictor injected)

```python
def roll_forward(predictor: Predictor, base, t0, *, horizon: int,
                 calibrator: Calibrator | None = None, absorb: int | None = None):
    p   = onehot_origin(...)                       # (n, 7)
    smm = np.empty((n, horizon))
    for h in range(1, horizon + 1):
        frame_h = advance_frame(base, t0, h)       # EXISTING (03 §3): age/term/seasonality evolve, macro@t0
        scores  = predictor.origin_scores(frame_h) # the ONLY model-specific call
        if calibrator:
            scores = [calibrator(s, o) for o, s in enumerate(scores)]
        P_h = assemble_matrix(scores)              # EXISTING
        if absorb is not None:
            P_h = absorb_rows(P_h, absorb)         # EXISTING first-passage trick (03 §3)
        p = compose(p, P_h)                        # EXISTING
        smm[:, h - 1] = prepay_increment(p)
    return p, smm
```

### 3.4 Loan-level pricing wrapper

`cashflow_engine` called per loan (note rate / remaining term / UPB + that loan's SMM path), then aggregate.
No new engine; the wrapper only marshals per-loan inputs and reuses `03 §5`.

### 3.5 Mask correction (M20)

Two disjoint fixes (per the supervision brief §7), applied in the predictor/QA path, **uniformly across models**:

- **Permit** the four reporting-gap-legal skip-bucket cells — `current→dpd_60`, `current→dpd_90plus`,
  `dpd_30→dpd_90plus`, `dpd_90plus→REO` — in `backtest._structural_allow` (`02 §7`). These are rare-but-realised;
  the mask wrongly forbade them.
- **Zero** the two mechanically-impossible cells — `current→foreclosure`, `current→REO` — inside the
  roll-forward predictor **before** any levels-consuming step. Immaterial to NLL (~1e-9) but high-LGD, so
  severity-amplified in pricing.

## 4. Resolved open design questions (these were intentionally open in the roadmap)

1. **SMM extrapolation beyond H.** The cashflow engine prices a full amortization schedule, so it needs an SMM
   value past the modelled horizon. Use the **constant terminal-SMM** extrapolation already in `03 §5.2`:
   `SMM(h > H) = SMM(H)`. For short horizons this is explicit and conservative (e.g. H = 1 ⇒ flat SMM at the
   one-month hazard). State the assumption in every short-horizon table; it is the horizon-analogue of the
   frozen-macro caveat, not a new modelling choice.
2. **Calibrator × absorb ordering.** Calibrate **first**, on the raw per-origin `(n, 7)` scores, **then**
   `assemble_matrix` → `absorb_rows` → `compose`. Calibration is a statement about the model's one-step output
   distribution; applying it after absorption would distort the first-passage construction and break the
   `H = 1 == evaluate.py` identity. Fixed in §3.3.
3. **Loan ↔ pool aggregation identity.** Because predicted counts are Σ of per-loan probabilities (`03 §4`) and
   the cashflow engine is linear in UPB, **pool cashflow = Σ loan cashflows** by construction. This gives the
   M22 unit test: loan-level prices summed over a pool reconcile to the pool-level price to ~1e-6. Any
   divergence is a wiring bug, not a modelling difference.
4. **First-passage × horizon.** `absorb_rows` is horizon-agnostic; "ever reaches 60+/90+ within H" uses the same
   absorbing trick for any H. No special-casing per horizon.

## 5. Multi-horizon semantics (the new analytical axis)

Horizon is a second axis alongside regime, not just more outputs:

- **H = 1** — the clean **calibration test**: minimal roll-forward, no compounding, minimal macro-path
  dependence. The most direct read of whether the level is right.
- **H = 3, 6, 12** — composed; this is where frozen-t0 macro inputs and any calibration drift **compound**. The
  COVID-inversion-in-dollars (`03 §5`) is expected to deepen with H.

The headline new exhibit is the **horizon × regime grid**: T4.2/T5.1/F5.2 (`03 §4–5`) gained an H dimension —
a *time series of the economic value of nonlinearity*, now also a *function of horizon*.

## 6. Calibration as a per-horizon result (M23)

Run reliability diagrams on raw outputs per horizon; decide and document per `06 §4`. Report **calibrated-vs-raw
price error by H** — the expectation (compounding) is that the gap widens with H, making "does calibration earn
its keep" a *result indexed by horizon* rather than a single yes/no. The calibrator is the §3.2 seam; disabled
runs use the identity (regression-checked).

## 7. AUC as the cross-period display metric (M25)

**NLL remains the estimation loss, the model-selection criterion, and the within-window comparison metric** — it
is the proper scoring rule and the spine of the flexibility-ladder attribution. But NLL is unintuitive and not
comparable across years (each test year is a different difficulty). So **AUC is the display/visualization metric
across time**: bounded [0, 1], a clean "ranking quality" reading, comparable across windows.

Deliverable: a **per-window AUC time series, 2015–2025, per model** for the diagnostic transitions where the
nonlinearity lives — `current→prepaid` (refi S-curve, ≈0.65→0.75 logit→net) and `dpd_90plus→foreclosure` (logit
below chance 0.42 → net 0.67), plus `current→dpd_30`. The AUC math already exists in `evaluate.py` (pooled);
M25 computes it per window and plots the series — the AUC mirror of the NLL Table B. **AUC to illustrate, NLL to
decide** — state this explicitly so AUC is never read as a fitting or selection criterion.

## 8. Integration points (modules)

| Module | Change |
|---|---|
| `pool.py` | extract `Predictor` protocol + `TorchPredictor` from `_origin_scores`; add `horizon` param + optional `calibrator`/`absorb` to `roll_forward`; add the §3.5 zeroing in the predictor path; add the loan-level pricing wrapper over `cashflow_engine`. |
| `backtest.py` | M20 mask correction in `_structural_allow` (permit the four legal cells), applied to **all** models; re-score nets under the corrected mask. |
| `smm_paths.py` / `economics.py` | thread `horizon`; extend T4.2/T5.1/F5.2 builders with the H dimension; add loan-level price-error outputs. |
| `evaluate.py` | **none for the metric path**; M25 adds a per-window AUC time-series driver that calls the existing AUC function per window (surface this, don't code around it). |
| `config.py` | record the horizon set {1,3,6,12}, the anchor set, and the calibration decision per window. |

## 9. Evaluation additions (exhibits)

- **Horizon × regime grid:** T4.2 (count R²/RMSE) and T5.1 (CPR/WAL/price error) with an H dimension, ≥3 regime
  anchors × H ∈ {1,3,6,12}; F5.2 price-error buckets per H.
- **Loan-level pricing:** per-loan WAL/price-error distribution alongside the pool-level T5.1, with the §4
  aggregation-identity check reported.
- **Calibrated-vs-raw by horizon** (M23): the per-H price-error table.
- **AUC-by-window time series** (M25): per key transition, per model, 2015–2025.

All fold into `writeup/memos/03_pool.md` and the Chapter-4 tables via the M11b/M15 writeup-sync pattern — no new
memo. The COVID-inversion-in-dollars is shown to deepen with H.

## 10. Definition of done

- `roll_forward(..., horizon=1)` equals `evaluate.py` exactly; the same engine prices net and (later) GBT/sequence
  models with only the injected `Predictor` differing.
- 1/3/6/12-month pool **and** loan-level pricing exist at ≥3 regime anchors; loan↔pool aggregation identity passes
  to ~1e-6; closed-form cashflow tests (`03 §5`) still pass.
- M20 mask correction applied to all models; residual impossible mass < 1e-4 every window; the two-cell zeroing
  leaves NLL unchanged (~1e-9) while removing the phantom high-LGD mass.
- Calibrated-vs-raw price error reported by horizon; calibration decision documented (applied/skipped with evidence).
- AUC-by-window time series exist for the key transitions, with the "AUC illustrates, NLL decides" note.

Implemented by **M20–M27** (`04_TASKS.md`). Governed by **ADR-001**. Sequence-model forward-compatibility: a
`SeqPredictor` conforms to §3.1; if it needs per-loan history rather than point-in-time covariates, the signature
change is **ADR-002** at the M26 go/no-go, not pre-built here.
