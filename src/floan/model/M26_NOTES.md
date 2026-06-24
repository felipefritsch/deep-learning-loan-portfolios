# M26 — minimal sequence-model spike: outcome record (authoritative)

**Task** (`specs/model/04_TASKS.md` §M26): conditional on M26a's signal, a *discriminating*
(not competitive) sequence-model spike — does **learned** memory (a GRU over the trailing
sequence) beat the **engineered** one-step memory of M26a's FF+history net? De-risked the data
path first (step 2a), then ran a three-arm, apples-to-apples comparison at k=2015 (step 2b).

## DECISION — M26 delivers the complete "value of memory" finding; **M27 = GO (staged).**

Learned memory beats engineered one-step memory, but **modestly**: the GRU's edge over
FF+history is −2.67×10⁻³ val-NLL (−10.6σ) — real and significant, yet only ~30% of the total
memory gain (the cheap engineered one-step features already capture ~70%). The extra signal is
the **multi-step path into deep delinquency** (the GRU's AUC edge concentrates there). M27 is
justified but should be **staged by tractability**: loan-level rolling first (**M27a**), h=1
pool pricing (clean — no roll-forward path problem); **multi-month sequence pricing is
future-work** (needs per-loan path simulation, not point-in-time covariates).

## Result (k=2015, dev, 3 seeds; train 300k uniform train_pool + HT weights; FULL 3.0M val / 3.1M test)

| arm | val-NLL (mean ± sd) | test-NLL (mean ± sd) |
|-----|--------------------:|---------------------:|
| FF baseline (current-state only) | 0.099362 ± 3.15×10⁻⁴ | 0.106136 ± 3.03×10⁻⁴ |
| FF + history (engineered one-step) | 0.093240 ± 2.87×10⁻⁴ | 0.100404 ± 2.33×10⁻⁴ |
| **GRU (learned sequence)** | **0.090574 ± 2.08×10⁻⁴** | **0.097964 ± 6.09×10⁻⁴** |
| *M26a full-data FF (banked)* | *base 0.096190 / hist 0.089308* | — |

All three arms share one training sample and the identical loop/early-stop protocol; scored on
the same full frozen slices via `evaluate._nll` / `evaluate._auc_one_vs_rest` (the FF-net metric
path, not reimplemented). The only difference is architecture.

### The two findings (val-NLL Δ, in pooled-seed-sd σ = RMS of the two arms' seed sds)

- **GRU − FF baseline = −0.008788 = −32.9σ** (pooled sd 2.67×10⁻⁴) — a sequence architecture
  clearly beats plain current-state conditioning.
- **GRU − FF + history = −0.002666 = −10.6σ** (pooled sd 2.51×10⁻⁴) — **learned memory beats
  engineered one-step memory**, real but modest.

### Decomposition of the memory gain

`ff_base → ff_hist` = −0.006122 (engineered one-step) ≈ **70%** of `ff_base → gru` = −0.008788
(total). The GRU adds the remaining **~30%** (−0.002666) — the part of the trajectory the
hand-crafted one-step summaries (prev_state, months-since-delinq, ever-delinq, episode count)
cannot express.

### Per-transition AUC (val, mean over seeds)

| transition | FF base | FF + history | GRU |
|---|---:|---:|---:|
| current → prepaid | 0.6807 | 0.6825 | **0.6989** |
| dpd_90plus → foreclosure | 0.6172 | 0.6300 | **0.6981** |
| current → dpd_30 | 0.8366 | **0.8951** | 0.8885 |

`ff_hist` **edges the GRU on `current→dpd_30`** (0.895 vs 0.889 — the one-step "was recently
delinquent" cue is exactly what predicts re-delinquency). The GRU wins **`current→prepaid`** and,
decisively, **`dpd_90plus→foreclosure`** (0.617 → 0.630 → **0.698**) — where the *multi-step path*
into deep delinquency carries signal the one-step summary misses. That deep-delinquency cell is
the same place the net showed its edge over GBT/logit, and is where a full sequence model earns
its keep.

### Corroboration + caveats

- **Test tracks val in lockstep** across all three arms (base 0.099→0.106, hist 0.093→0.100,
  gru 0.0906→0.0980) — the ranking is not val-fitting.
- **Caveats:** single dev window (k=2015); 300k train subsample (FF arms land slightly worse
  than their full-data banked numbers — `ff_base` 0.0994 vs 0.0962, `ff_hist` 0.0932 vs 0.0893
  — the expected subsample gap, sanity-checked); the GRU's **test sd (6.1×10⁻⁴) is ~3× its val
  sd** (2.1×10⁻⁴), so the headline is val (the selection metric) with test as corroboration.

## Bug record — the FF+history eval misalignment (found + fixed in this task)

First pass: `ff_hist` collapsed to ~0.46 val-NLL (≈ marginal) across all seeds while `ff_base`
and the GRU were healthy. Root cause: **`history.join_history` used a streaming hash-join, which
does not preserve left-row order.** The spike scored `ff_hist` probs (from the post-join,
*reordered* frame) against `val_y` / `val_origin` taken from the original `val_df` order → every
row's prob was paired with the wrong label → garbage NLL (and ~0.5 AUC). `ff_base`/GRU have no
join, so they were unaffected; a direct retrain that scored against the encoded frame's *own* `y`
converged fine (val → 0.097), which is what localized it. Diagnostic confirmed a 200k val frame
came out fully permuted by the join.

**Fix:** `join_history` now carries an explicit row index across the join and sorts back, keeping
the streaming memory bound (commit `5ac734d`). **M26a is UNAFFECTED** — its augmented net scored
`y` from the *same* joined frame (and NLL is order-invariant when each prediction keeps its own
label), so the M26a numbers stand. Sanity gate after the fix: `ff_hist` 0.0932 < `ff_base` 0.0994
and ≈ the full-data 0.089 — so the GRU−ff_hist delta is trustworthy.

## Artifacts

- Harness: `src/floan/model/seq_spike.py` (3-arm, `--calibrate` projection); data path
  `sequence.py`, model `seq_model.py`, loop `seq_train.py`; tests `tests/test_sequence.py`.
- Summary of record: `models/nn/dev/seq_spike_summary_fixed.json` (corrected ff_hist + the saved
  ff_base/gru per-seed); the original (pre-fix) run is `seq_spike_summary.json`.
- Timing (Mac CPU): GRU full-slice scoring ~700s/seed (3.0M val + 3.1M test, chunked); full
  3-seed run ≈ 50 min — tractable on the Mac, no pod needed for the dev spike.

## M27 framing (the staged go)

- **M27a (GO):** loan-level rolling sequence model first — the supervisor's explicit
  loan-level ask, and the cleanest extension of this spike.
- **h=1 pool pricing (GO):** drops straight into the `Predictor` seam (`ADR-001`) — no
  roll-forward path problem at h=1 (it equals the direct prediction).
- **Multi-month sequence pricing (FUTURE-WORK):** pricing at H>1 needs per-loan *path
  simulation* (the sequence model consumes history, not point-in-time covariates), i.e. an
  `ADR-002` on the seam signature before it can feed the cashflow engine. Deferred.
