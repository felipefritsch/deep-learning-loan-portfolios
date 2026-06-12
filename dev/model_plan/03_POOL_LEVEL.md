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

## 5. Economic translation: from counts to CPR, WAL, and price (the contribution exercise)

This section is the dissertation's main **beyond-replication contribution**: it converts the statistical pool-level comparison of §4 into the units an MBS investor prices in. The chain is counts → prepayment speed → cashflows → weighted-average life and price.

1. **Pool prepayment speed.** From the §3 roll-forward, each loan has a monthly prepayment hazard path over the horizon; aggregate UPB-weighted to a pool **SMM path** $\mathrm{SMM}_{\mathcal{P}}(h),\ h=1..12$ (and the realized analogue from the panel). Report annualized CPR for headline tables: $\mathrm{CPR} = 1-(1-\mathrm{SMM})^{12}$.
2. **Cashflow engine** (`pool.py` addition, ~100 lines): deterministic level-pay pass-through — given a pool's WAC/WAM/UPB and a monthly SMM vector (model or realized for h=1..12; constant extrapolation of the terminal SMM thereafter), generate scheduled amortization + prepayments, and compute **WAL** and **price** by discounting at a fixed yield (par-anchored: discount at WAC less a fixed servicing spread, same curve for every model — so price differences isolate the prepayment-model component). Unit tests against closed forms: zero-prepay = annuity formula; constant-SMM has a closed-form survival schedule.
3. **Error metrics per pool × model × anchor:** CPR error (pp), **WAL error (months)**, **price error (per 100 face)** — model-implied vs realized-SMM valuation. Artifacts: **T5.1** (summary table: mean |error| and signed bias by model × outcome × anchor; the headline number is the ensemble-vs-logit reduction in mean |price error|) and **F5.2** (price-error by characteristic bucket — expect the linear model's errors to concentrate in high-incentive cells, the pricing mirror of `02`'s S-curve).
4. **Stated caveats (one paragraph, not a limitation section):** macro frozen at t0 → this is the *prepayment-model* component of valuation error at a fixed discount curve, not full OAS; credit events enter as zero-recovery terminations in the engine's first pass (sensitivity with a recovery haircut optional); conditional-independence inherited from §4.

> **Regime-conditional accuracy (costless contribution exhibit).** The §4 by-anchor table/figure (T4.2 extension) is itself original — a *time series of the economic value of nonlinearity* across regimes (2015 calm → 2020 COVID → 2023 rate shock). Frame it as such in the memo, not as a robustness afterthought.

**Deferred (not in scope):** the paper-§5.2-style portfolio decile exercise (rank pools, form best/worst portfolios). Rationale (written out in the methodology overview §"Why pricing rather than portfolio sorts" — the memo cites it): sorts test only the *ranking* of the same per-loan probabilities, pricing tests the *levels*; our probabilities are calibration-checked, the model's intended consumer prices in levels, and ranking vs calibration are each already tested upstream (AUC / reliability diagrams) — so the sort would duplicate F5.2's information without the magnitudes. Noted as future work in the memo.

## 6. Memo

`writeup/memos/03_pool.md`, ~4 pages: roll-forward method (with the covariate-freezing assumption stated), F4.1, T4.2 (incl. the by-anchor regime exhibit), F4.3, the §5 economic translation (T5.1, F5.2) with its caveat paragraph, and a closing **contributions** paragraph that names what is original here versus Sirignano et al.: prime conforming credit box; 2015–2025 test regimes incl. COVID forbearance and the 2022–23 rate shock; the rolling-window protocol; and the CPR/WAL/price translation. Seeds the dissertation's final results chapter.

**Acceptance for Phase 3:** roll-forward harness validated on a toy chain with known closed-form (unit test); composed 1-month probabilities at h=1 match `evaluate.py` outputs exactly; predicted-vs-realized artifacts exist for both pool schemes × both outcomes × ≥2 models (ensemble + logit) at ≥3 anchors spanning regimes; cashflow engine passes its closed-form unit tests and T5.1/F5.2 exist for ensemble vs logit at the same anchors; memo committed with the contributions paragraph.
