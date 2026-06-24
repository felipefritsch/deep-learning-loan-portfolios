# M26a — Markov-assumption probe: outcome record (authoritative)

**Task** (`specs/model/04_TASKS.md` §M26a): test path-dependence **without** a sequence
model — add history-summary features to the **existing** feed-forward net at the tuning
window (k=2015, dev), reuse the existing loader/net/eval path, and compare val-NLL against
the current-state-only net with the M12 seed-noise band.

## DECISION — **SIGNAL: the first-order Markov assumption is rejected at dev scale → M26 is GO.**

The augmented net improves val-NLL by **+6.88×10⁻³**, which is **≈29.5σ** of the baseline
seed-noise band — not a 1σ wisp. The improvement holds in lockstep on the **frozen test**
slice (so it is real generalization, not val overfitting). Per the pre-registered rule
(≥2σ ⇒ signal/escalate; 1σ–2σ ⇒ suggestive/future-work; <1σ ⇒ current-state sufficient),
this clears the bar by a wide margin.

> **Do NOT re-run the probe — the result is banked here.** Reproduction commands are listed
> at the bottom for provenance only; the run folders + `markov_probe_summary.json` are the
> evidence of record.

## Result (k=2015, dev, 3 seeds, CPU; identical training protocol — only the feature set differs)

| seed | baseline val-NLL | augmented val-NLL | Δ val (aug−base) | baseline test-NLL | augmented test-NLL |
|-----:|-----------------:|------------------:|-----------------:|------------------:|-------------------:|
| 0    | 0.095975 | 0.089209 | −0.006766 | 0.102595 | 0.096301 |
| 1    | 0.096155 | 0.089355 | −0.006801 | 0.102975 | 0.096654 |
| 2    | 0.096438 | 0.089360 | −0.007078 | 0.103250 | 0.096346 |
| **mean** | **0.096190** | **0.089308** | **−0.006882** | **0.102940** | **0.096433** |

- **Baseline seed sd (the M12 band):** 2.334×10⁻⁴ → 1σ = 2.33×10⁻⁴, 2σ = 4.67×10⁻⁴.
  Augmented seed sd is even tighter (8.56×10⁻⁵); both arms are highly stable.
- **Improvement = +6.882×10⁻³ val-NLL = +29.48σ.** Paired Δ per seed: mean −6.882×10⁻³,
  **sd 1.71×10⁻⁴** (every seed gives essentially the same gap).
- **Not val overfitting:** test-NLL falls **0.102940 → 0.096433** (−6.5×10⁻³) in lockstep
  with val — the augmented net generalizes the gain to the frozen, unthinned test rows.

## Leakage guard — proven, not asserted

The four features are derived from the **panel** (the only complete per-loan monthly state
series — `train_pool` thins current→current, `eval_pool` floors the label month) via a
single DuckDB window SELECT (`history._history_select_sql`). Three pillars:

1. **Structural strict precedence.** Every aggregate runs over the window frame
   `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` — it *ends at the row before t*, so a
   row's own state can never enter its own features; `prev_state = LAG(state)` is strictly
   t−1. (Not a hand-applied filter — a property of the frame.)
2. **`state_next` (the target) is referenced nowhere** — the SQL reads only `state` (last
   month's reported status, known at t) and `period`.
3. **Exact `(Loan Identifier, period_ym)` join** keeps the existing window masks binding, so
   every model row only ever receives history through t−1 (mask-invariant).

**Spot-check evidence** (`history.spot_check`, run on real data): loan **725909295135 @
t=201304** — a `…→dpd_90plus(201302)→cured-to-current` episode — yields `prev_state=current`,
`ever_delinquent=1`, `months_since_last_delinq=2`, `n_prior_delinq_episodes=1`, with **cached
== independent pre-t oracle**. The **flip-`state_next` test PASSED**: mutating every
`state_next` for the loan left all 238 rows' features byte-identical (empirical proof the
target is never read). Hermetic versions of all of this are in `tests/test_history.py`
(causal correctness, partition isolation, strict precedence, flip-invariant).

## Features & the signal's shape

| feature | block | derivation (at t, using only period < t) |
|---|---|---|
| `prev_state` (S₍ₜ₋₁₎) | categorical (embed) | `LAG(state)`; first record → UNK |
| `months_since_last_delinq` | continuous | `mi_t − max{mi_j : j<t, state_j∈DPD}`; never → null → scaler centre |
| `ever_delinquent` | binary | `1` if any j<t had state∈DPD |
| `n_prior_delinq_episodes` | continuous | # DPD-episode entries before t |

Delinquency set = **DPD only** `{dpd_30, dpd_60, dpd_90plus}` (the spec's "delinquency
episodes"; foreclosure/REO are terminal credit events, kept out so the signal stays
delinquency-specific). `prev_state` still records the *full* prior state regardless.

**The signal is one-step concentrated.** AUC (val, mean over seeds — illustrative; NLL
decides) moves most on the transition that the current state alone cannot disambiguate:

| transition | baseline AUC | augmented AUC |
|---|---|---|
| `current → dpd_30` | 0.8414 | **0.9015** |
| `dpd_90plus → foreclosure` | 0.6436 | 0.6741 |
| `current → prepaid` | 0.7068 | 0.7094 |

Mechanism: whether a `current` loan was *recently cured* vs *always-current*, and the
*direction of travel* into a DPD state, sharply changes the next-transition odds — exactly
the memory a first-order (current-state-only) Markov model discards. The bulk of the lift is
plausibly carried by `prev_state` (one step of memory), which is precisely what M26's minimal
GRU/LSTM spike is built to capture; the longer-range summaries can be decomposed there.

## Implementation notes (for M26 and reuse)

- **Opt-in, additive, byte-identical when off.** `--augment` threads through `train.py`
  (fit via `history.fit_window_aug`, `preload_split` joins history + extends the binary
  block, `net.build` sizes `n_binary`); `features.encode_frame` gained `binary_cols`;
  `net.build` gained `n_binary`. **`data.py` untouched** — the core loader stays pristine.
- **Cache build is per-shard** (`history.build`): the all-shards windowed COPY OOMs the 5.5
  GiB DuckDB limit; loans are shard-stable (`shard = hash(loan) % N_SHARDS`), so
  `PARTITION BY loan` within a shard is exact. 37.7 M rows cached in 56 s under
  `processed/training/dev/history/k2015/`.
- **Join is lazy/streaming** (`engine="streaming"`) — no resident history frame, so peak
  memory stays bounded across the 6 runs.
- **CPU only.** `train.py`'s loss/eval loop accumulates in float64, which MPS does not
  support; CUDA is the only GPU path. The probe ran on CPU per the brief (89.5 min).

## Artifacts

- Run folders: `models/nn/dev/mp_{base,aug}_s{0,1,2}/` (scaler/vocab/metrics/best weights).
- Summary of record: `models/nn/dev/markov_probe_summary.json` (run on the working tree of
  commit `250042f` + the then-uncommitted M26a code, now committed).
- Code: `history.py`, `markov_probe.py`, `tests/test_history.py`, and the additive
  `features.py` / `net.py` / `train.py` edits.

Reproduction (provenance only — **do not re-run; banked above**):

```
python -m floan.model.history build       --variant dev --k 2015
python -m floan.model.history spot-check   --variant dev --k 2015
python -m floan.model.markov_probe --seeds 0 1 2 --device cpu
```
