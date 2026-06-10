# 03 — Phase 3: Pool-Level Analysis

> **Goal:** show that the loan-level model differences *matter economically* by aggregating to pools — the paper's §5 argument: pool cashflows are sums of loan-level events, so a better loan-level transition model prices pools better, and linear misspecification compounds at the pool level (especially for prepayment). Everything here consumes the **frozen Phase-2 models**; no re-fitting.

---

## 1. Setup

- **Anchors follow the rolling windows:** for window k, anchor `t0 = Dec(k−1)` — loans alive (origin state non-terminal) at t0, predicted over test year k by **window k's frozen model** (trained on labels ≤ Dec(k−2)). Predictions use only loan information available at t0. (Subtlety to state in the writeup: window k's net was early-stopped on year k−1 val NLL, so the anchor-time *fit* is honest but the stopping epoch saw year k−1 aggregates — a mild, standard concession.) Run **all 11 anchors if scoring cost allows; minimum 3 spanning regimes** — chosen from the key windows that have an ensemble (`02 §6`): anchors 2019-12 (k=2020, COVID), 2022-12 (k=2023, rate spike), 2024-12 (k=2025, latest).
- **Horizon:** 12 months. **Pool outcome variables:** number of loans prepaying within the horizon; number reaching `dpd_60+` (or `90+`) within the horizon. (UPB-weighted versions optional, second pass.)
- **Models compared:** empirical matrix, logit, best single NN, ensemble — same window, identical roll-forward harness.

## 2. Pool construction (`pool.py`)

Two complementary schemes, per the May-8 notes and the paper:

1. **Characteristic buckets:** partition the t0-alive `current` loans by FICO bucket × original-rate bucket × LTV bucket (e.g. 4×4×3 = 48 pools; merge thin cells, target ≥ 2,000 loans per pool). Optionally add vintage-year as a fourth axis in a second cut.
2. **Random pools:** B = 500 pools of N = 1,000 loans drawn at random from the same population — the paper's device for a scatter of predicted vs realized without bucketing artifacts.

## 3. Roll-forward of the loan-level model (the part the notes flagged as confusing)

For loan *i* alive at t0, the 12-month outcome probability comes from **composing monthly transition matrices**:

```
For each loan i:
  p ← one-hot(state_{t0})                                   # 1×7 state distribution
  for h = 1..12:
      build features x_{t0+h-1}(i): static fields frozen at t0;
      loan age += h−1; remaining months −= h−1; seasonality follows calendar;
      macro block (incentive, unemployment, HPI-derived features) frozen at t0 (no macro forecasting — document this);
      P_h ← model's 7×7 matrix for loan i at month h               # rows for terminal states = identity
      p ← p · P_h
  P(prepaid within 12m | i) = p[prepaid];  P(60+ dpd within 12m | i) = absorbed mass (see below)
```

- **Deterministic composition, not simulation, is the default:** since terminal states are absorbing and we track the full 7-vector, the product of monthly matrices gives exact horizon probabilities under the model — no Monte Carlo error. (The paper's MC machinery exists for path-dependent covariates/correlated macro scenarios, which are out of scope here.)
- For "ever reaches 60+ dpd within 12m", make `dpd_60` temporarily absorbing in the composed chain (standard first-passage trick) — otherwise cures leak the event.
- **Covariate evolution assumption** (state explicitly in the writeup): only deterministic features (age, remaining term, calendar seasonality) evolve; current UPB and the entire macro block frozen at t0. Predictions are therefore *conditional on the t0 macro environment* — the conservative, assumption-light choice (macro scenario paths are a future-work hook, not in scope).
- Implementation: vectorize — batch-score all alive loans once per horizon step per origin state (12 × few forward passes over ~ millions of rows; GPU or chunked CPU both fine).

## 4. Pool predictions & evaluation

- **Predicted count** per pool = Σ_i P(event within 12m | i) (the notes' formula). Predicted **distribution** (for the random pools): Poisson-binomial, approximated Normal(μ=Σp, σ²=Σp(1−p)) — loans are conditionally independent given covariates under the model (paper's assumption; flag it).
- **Realized count** per pool from the panel over `(t0, t0+12]`.
- **Metrics & figures:**
  - F4.1 scatter predicted vs realized counts (random pools), 45° line, per model — the paper's flagship pool exhibit; report R² and RMSE.
  - T4.2 table: R²/RMSE by model × outcome (prepay, 60+ dpd) × anchor; relative RMSE reduction NN-ensemble vs logit is the headline number. With all 11 anchors, add the figure: pool-level RMSE by anchor year per model (the pool-level mirror of loan-level Table B).
  - F4.3 characteristic-bucket bar/heatmap: signed prediction error by FICO×rate bucket per model — *where* does the linear model break (expect: high-incentive buckets).
  - Coverage check: % of random pools whose realized count falls in the model's 90% interval.

## 5. Optional extension (only if time allows)

Paper §5.2-style portfolio exercise: rank pools by predicted 12-month prepay (or delinquency), form best/worst decile portfolios at t0, compare realized event rates NN vs logit. Cheap once §4 exists; a nice economic-significance paragraph. **Not** part of acceptance.

## 6. Memo

`writeup/memos/03_pool.md`, ~3 pages: roll-forward method (with the covariate-freezing assumption stated), F4.1, T4.2, F4.3, and a closing paragraph linking pool-level accuracy to MBS valuation (cite paper §5). Seeds the dissertation's final results chapter.

**Acceptance for Phase 3:** roll-forward harness validated on a toy chain with known closed-form (unit test); composed 1-month probabilities at h=1 match `evaluate.py` outputs exactly; predicted-vs-realized artifacts exist for both pool schemes × both outcomes × ≥2 models (ensemble + logit) at ≥3 anchors spanning regimes; memo committed.
