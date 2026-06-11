# Memo 01 — Exploratory Analysis & Motivation

*Phase 1 of the modelling plan (`dev/model_plan/01_EDA.md`); seeds the Data and
Motivation chapters. All figures regenerate from `dev/analysis/` against the Stage-4
DuckDB panel views; tables in `outputs/tables/eda/`, figures in `outputs/figures/eda/`.*

---

## 1. Dataset scope and integrity

The assembled panel is the **complete Fannie Mae Single-Family Loan Performance**
release, 2000–2025: **3,312,456,883 loan-months across 57,562,668 loans**, spanning
reporting months `2000-01 … 2025-12` over 104 acquisition-vintage partitions
(coverage table T1.1). Each loan-month carries the Sirignano-style seven-state target
(`state`, `state_next`), leakage-safe calendar keys (`period_ym`, `orig_ym`), and a
loan-keyed `shard`.

Three deviations from Sirignano, Sadhwani & Giesecke (2021) frame every result that
follows and are stated here once (`00_OVERVIEW §4`):

- **Credit box.** This is the *public* GSE dataset — 30-year fixed-rate, conventional,
  conforming, LTV ≤ 97 — not the paper's CoreLogic private-label universe. No
  ARMs, no subprime: a cleaner book, so delinquency base rates are far lower than the
  paper's and the *strength* of the nonlinearities, not just their shape, is the open
  question.
- **Terminal states.** `foreclosure`, `REO`, and `prepaid` are derived from Zero
  Balance codes in a loan's final month and are absorbing — there is no ongoing
  "in foreclosure" monthly state. Origins are effectively `{current, dpd_30, dpd_60,
  dpd_90plus}`; the empirical transition matrix is **4×7**, not 7×7.
- **Macro granularity & period.** Rates are national, unemployment and house prices
  are by state (`05_MACRO_DATA.md`) — the channels are all present but within-state
  local variation is not. The 2000–2025 window includes the **COVID forbearance**
  episode, whose delinquency codes behave unusually (F2.3); we adopt no special
  handling but flag it.

## 2. The dependent variable: severe class imbalance and regime shifts

The state distribution (T1.3) is dominated by `current`:

| state | share of loan-months |
|---|---|
| current | **96.28 %** |
| prepaid | 1.25 % |
| dpd_30 | 1.15 % |
| dpd_90plus | 0.98 % |
| dpd_60 | 0.31 % |
| REO | 0.014 % |
| foreclosure | 0.0067 % |

This ≈96 % `current` mass is *the* reason Phase 2 thins `current→current` rows by a
keep-probability `p_keep` and importance-weights the loss (`1/p_keep`) so estimated
probabilities stay unbiased; the rare credit-event states (foreclosure/REO at
~1-in-10,000) demand Laplace smoothing in the empirical benchmark and Wilson intervals
on every empirical rate in this memo.

Pooled over time (T2.1), the one-month rates out of `current` are `→ prepaid` 1.27 %
and `→ dpd_30` 0.64 %. But the *pooling* hides the story: **F2.2** (monthly transition
rates, 2000–2025) shows the 2003 and 2020–21 refinancing waves in `current→prepaid`,
the 2008–11 default wave in `current→dpd_30` and `dpd_90+→foreclosure/REO`, and the
sharp **2022–23 prepay collapse** as rates spiked. These regime shifts — not a single
COVID dummy — are why Phase 2 uses an expanding-window **rolling backtest** with one
test year per window.

## 3. Nonlinearity: model-free hazard curves

Each curve below is one DuckDB aggregation over the full panel, collapsed to
equal-population buckets with Wilson 95 % CIs (`eda.hazard_curve`). They are the
model-free analogues of the paper's nonlinearity exhibits.

- **F3.1 — Prepayment vs loan age (the seasoning hump).** `current→prepaid` rises
  from **0.27 %/mo at origination** to a hump peak of **1.56 %/mo at ≈17 months**, then
  *burns out* to **1.11 %/mo by ≈52 months**. (A secondary late-life rise to ~1.65 %/mo
  at 15+ years reflects deep-in-the-money survivors prepaying, pooled across calendar
  time.) The early hump alone is strongly non-monotone — a linear age term is
  indefensible.
- **F3.2 — Prepayment vs rate incentive (the refinancing S-curve).** With
  `incentive = current_rate − pmms30(t)`, the prepay rate is flat and low
  (**0.4–0.6 %/mo**) while out-of-the-money, turns up sharply through zero, peaks at
  **2.71 %/mo near +1.5 pp**, and **burns out to 2.16 %/mo by +2.7 pp** — a ~7× range
  with an inflection and a turning point. This is the paper's flagship nonlinearity and
  it reproduces cleanly here.
- **F3.3 — Delinquency vs FICO (convex decay).** `current→dpd_30` falls from
  **3.06 %/mo at FICO ≈623** to **0.13 %/mo at FICO ≈815** (a ~24× span), and the
  decline is convex — the mean |slope| in the low-FICO half is ~3.7× that in the
  high-FICO half. A linear FICO term would badly misfit both tails.

## 4. Interaction: the case against an additive model

- **F3.4 — FICO × LTV (the headline exhibit).** On a FICO×LTV grid (equal-population
  bands per axis), the
  `current→dpd_30+` rate ranges from **0.11 %/mo** (best FICO, lowest LTV) to
  **2.69 %/mo** (worst FICO, highest LTV). The decisive fact for model choice: a
  linear-*additive* model forces the FICO effect to be constant across LTV, but the
  empirical **additive FICO gradient** (worst- minus best-FICO rate) *grows* from
  **1.73 pp at low LTV to 2.48 pp at high LTV (+43 %)** — equivalently, the LTV effect
  is **~8.5× larger for low-FICO than high-FICO** loans. Additivity is misspecified;
  the interaction is exactly the low-FICO/high-LTV corner that drives credit losses.
- **F3.5 — Incentive × FICO (burnout / credit constraint).** Splitting the F3.2 S-curve
  by FICO tercile, the in-the-money (+1.5 pp) refi response rises with credit quality —
  **2.36 %/mo (low FICO) → 3.01 %/mo (high FICO)** — because credit-constrained
  borrowers cannot act on the incentive. The incentive *slope* depends on FICO: a
  second interaction, in the prepayment channel.
- **F3.6 — Vintage effects.** Holding loan age fixed at 12–36 months to strip out
  seasoning, the `current→dpd_30` hazard by origination year isolates a clean cohort
  effect: the **2007 vintage peaks at 1.68 %/mo** (2006: 1.25 %, 2008: 1.14 %) against
  **0.47 %/mo** for calm cohorts (2003/04/12/13) — a **×3.6** bubble-vintage premium.
  Vintage is a feature in its own right, not reducible to contemporaneous covariates.

## 5. The incentive variable and its panel-derived proxy

The headline incentive uses the downloaded Freddie Mac **PMMS 30-yr** series
(`macro_national`, lag 0). As a Phase-2 robustness alternative, `mkt_rate.py` builds a
**panel-derived proxy** — the average origination rate over loans originated each
calendar month (`processed/macro/mkt_rate.parquet`, 324 months 1999-01…2025-12; the
average is over *loans*, not loan-months, and no month fell below the thin-population
fill threshold). **F4.1** validates one against the other: the proxy tracks PMMS with
**correlation 0.991, RMSE 0.198 pp, MAE 0.147 pp**, and a mean gap of essentially zero
— an unbiased, tight tracker. The PMMS-based incentive is the primary feature; the
proxy is the documented robustness check. F4.2/F4.3 (M2b) provide the macro-context and
state-dispersion exhibits.

## 6. Sampling sanity checks

The loan-keyed `shard = hash(loan_id) % 256` underpins leakage-safe Phase-2 splits, so
its balance is checked directly. **F5.1:** rows per shard are uniform to
**CV 0.26 %** and exact loans per shard to **CV 0.21 %** (224,854 ± ~1,500) across all
256 shards (loan-disjoint by construction). *(Note: `approx_count_distinct` grouped over
the 256 shards proved unreliable — it reported a spurious ~30 % spread that the exact
`count(DISTINCT)` flatly contradicts — so F5.1 uses the exact distinct.)* **F5.2:** the
per-shard origination-year composition matches the panel marginal — the across-shard CV
is **≤ 2.3 % for every vintage above 0.5 % of the book (median 1.4 %)**; only the
negligible 1999 cohort (0.2 % of loan-months) reaches 5.7 %, pure small-sample noise.
The shards are **not** vintage-clustered, the failure mode the May-8 notes flagged,
because the hash keys on the loan id rather than the vintage.

## 7. Modelling hypothesis (carried into Phase 2)

The evidence is consistent and points one way: **monthly transition risk depends on the
covariates nonlinearly** (the seasoning hump F3.1, the refinancing S-curve F3.2, convex
FICO decay F3.3) **and interactively** (FICO×LTV F3.4, incentive×FICO F3.5, vintage
F3.6). A linear-additive multinomial logit should therefore **underfit**, and most
severely on **prepayment** — the channel with both the sharpest single-variable
nonlinearity (the incentive S-curve) and a first-order interaction with credit. Phase 2
tests exactly this: empirical matrix → multinomial logit → deep nets, on a strictly
out-of-sample rolling backtest, with the prepayment NLL the place to look for the
deep model's largest edge.
