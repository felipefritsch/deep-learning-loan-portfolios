# 01 — Phase 1: Exploratory Analysis & Motivation

> **Goal:** establish (a) the scope and quality of the assembled panel, (b) the base rates of the seven-state transitions, and (c) *visual evidence of nonlinearity and variable interactions* — the motivation for deep learning over linear models, mirroring the argument in Sirignano et al. §1 and their Figures on variable interactions. Everything here runs out-of-core via the Stage-5 DuckDB views; no model fitting yet.

All figures go to `outputs/figures/eda/` (PNG + PDF), all tables to `outputs/tables/eda/` (CSV + LaTeX via `to_latex`), each produced by one script in `dev/analysis/` so regeneration is one command. The memo (`writeup/memos/01_eda.md`) embeds the highlights.

---

## 1. Panel scope & integrity (tables)

- **T1.1 Coverage table:** per acquisition vintage — loan count, loan-month count, first/last reporting period, % censored. Source: Stage 1 manifest + a DuckDB `GROUP BY acq_quarter`. Confirms the uniform release cutoff visually.
- **T1.2 Covariate summary:** mean / std / p1-p50-p99 / null-rate for every `feature_spec` continuous column; level counts for categoricals. Largely a re-presentation of the Stage 6 QA output — do not recompute from scratch if `outputs/` already has it.
- **T1.3 State distribution:** loan-month counts and shares by `state`, overall and by calendar year. Quantifies the imbalance (`current` ≈ 95%+ of rows) that motivates stratified sampling and importance weighting in Phase 2.

## 2. Transition structure (the dependent variable)

- **T2.1 Empirical transition matrix (full panel, pooled over time):** 4 origin states × 7 destinations, counts and row-normalized frequencies. This is *also* the Phase-2 benchmark — compute it once here on the full panel for description, recompute on the training slice only in Phase 2 for honest benchmarking.
- **F2.2 Transition rates over calendar time:** monthly series of `current→prepaid`, `current→dpd_30`, `dpd_90plus→foreclosure/REO` rates, 2000–2025. Expect: 2003 refi wave, 2008–11 default wave, 2020–21 refi/forbearance spike, 2022–23 prepay collapse. This figure motivates the regime-shift discussion and the rolling backtest.
- **F2.3 COVID-era check:** distribution of `Current Loan Delinquency Status` codes by month for 2019-06 → 2022-06. Forbearance shows up as unusual delinquency-status persistence with later "cures." Decide (and document) whether any special handling is warranted; default is none, but the writeup must acknowledge it.
- **F2.4 Cure/roll heatmap:** for each origin state, destination shares over time (stacked area per origin). Shows roll-rate cyclicality.

## 3. Nonlinearity & interaction evidence (the motivation figures)

Empirical hazard curves — bucket a covariate, plot the empirical one-month transition rate per bucket with CIs. These are model-free analogues of the paper's nonlinearity exhibits:

- **F3.1 Prepayment vs loan age** (`current→prepaid` rate by loan-age bucket): the classic seasoning hump — strongly non-monotone, indefensible as a linear term.
- **F3.2 Prepayment vs rate incentive** (see §4): S-shaped around zero incentive — the paper's flagship nonlinearity.
- **F3.3 Delinquency vs FICO** (`current→dpd_30` by FICO bucket): convex decay.
- **F3.4 Interaction heatmap — FICO × LTV:** `current→dpd_30plus` rate on a 2-D grid. If the FICO gradient steepens at high LTV (it will), a linear-additive model is misspecified — this is the single most persuasive motivation figure.
- **F3.5 Interaction — incentive × FICO (or × loan size):** prepayment response to incentive by credit/size bucket; documents burnout/credit constraints on refinancing.
- **F3.6 Vintage effects:** default hazard by orig-year cohort at fixed loan age (2006–07 vintages stand out).

Implementation note: each figure is a single DuckDB aggregation (`GROUP BY` bucket) over the panel — seconds to minutes, no sampling needed. Use equal-population buckets (deciles/ventiles), not equal-width.

## 4. Macro tables, incentive variable & mkt-rate proxy

The macro set (PMMS, unemployment nat+state, FMHPI nat+state, 10y Treasury) is downloaded and built per **`05_MACRO_DATA.md`** into `processed/macro/{macro_national,macro_state}.parquet` — tiny time-keyed tables joined at export time (Phase 2); the panel is never rewritten. EDA additions:

- **Panel-derived proxy (kept as robustness):** `mkt_rate(t) = avg(Original Interest Rate) over loans with orig_ym = t`, materialized as `processed/macro/mkt_rate.parquet`. Months with thin origination counts (panel edges) → forward-fill and flag.
- **F4.1 Validation figure:** `mkt_rate(t)` overlaid on the actual PMMS 30-yr series (now downloaded, not eyeballed). Report the tracking error in the memo; the headline incentive variable is `current_rate − pmms30(t)`, with the proxy version as a robustness check in Phase 2.
- **F4.2 Macro panel figure:** small-multiples of pmms30, national unemployment + interquartile range across states, national FMHPI + state dispersion, 10y Treasury, 2000–2025. Doubles as the writeup's macro-context exhibit.
- **F4.3 State-dispersion check:** unemployment and HPI-drawdown ranges across states in 2009 vs 2019 — motivates *state-level* (vs national-only) joins.

## 5. Sampling sanity checks (carry-over from the May 8 notes)

- **F5.1 Shard balance:** rows and loans per `shard` (should be near-uniform; loan-disjoint by construction). One bar chart, one assertion.
- **F5.2 Vintage × shard mix:** confirm shards are not vintage-clustered (the hash is on loan id, so they won't be — but the notes flagged exactly this failure mode, so show it).

## 6. Memo

`writeup/memos/01_eda.md`, ~3–4 pages: dataset scope paragraph (with the public-subset caveat from `00_OVERVIEW §4`), the imbalance table, F2.2, F3.1/F3.2/F3.4 with one paragraph each, the incentive-proxy validation, and a closing paragraph stating the modelling hypothesis: *transition risk depends on covariates nonlinearly and interactively; linear models should underfit prepayment most severely.* This memo seeds the dissertation's Data and Motivation chapters.

**Acceptance for Phase 1:** every table/figure regenerates from `dev/analysis/` scripts against the DuckDB views in bounded memory; `mkt_rate.parquet` exists and is validated; memo committed.
