# Memo 03 — Pool-level results and the economic translation (Phase 3)

*Phase 3 of the modelling plan (`specs/model/03_POOL_LEVEL.md`, §1–§6); the final results
chapter (§4.3) in embryo, and the dissertation's main beyond-replication contribution. Consumes the
**frozen** Phase-2 models — no re-fitting. Tables in `outputs/tables/pool_level/`
(`t_m14_pool_counts.*` for §2–§4, `t_m15_econ_errors.*` for §5; per-pool detail in
`econ_errors_k*.parquet`), figures in `outputs/figures/pool_level/`. Mechanical notes + Accept
evidence: `src/floan/model/{M13,M14,M15}_NOTES.md`. Builds on memo 02c §4 (the M14 addendum) and shares its
shock-type-dependence framing. Read alongside `02b` (loan-level) — the point of this memo is that the
loan-level NLL differences, which `02c §3` flagged as a ~1e-4 ensemble edge, **earn their keep at pool
aggregation** in most regimes.*

---

## 1. Roll-forward method (the bridge from loan-level transitions to pool outcomes)

Pool outcomes are sums of loan-level events, so a per-loan 12-month event probability is needed first.
For each loan alive at a window anchor `t0 = Dec(k−1)`, window-k's frozen model is rolled forward by
**composing its monthly 7×7 transition matrices**: `p ← one-hot(state_t0)`, then `p ← p · P_h` for
`h = 1..12`, reading off `p[prepaid]` for prepayment. Because terminal states are absorbing and the full
7-vector is tracked, the matrix product gives the **exact** horizon probability under the model — no
Monte-Carlo error (the toy-chain unit test matches the closed form; `h=1` reproduces `evaluate.py`'s
loan-level probabilities bit-for-bit, M13 Accept). For "ever reaches 60+ DPD within 12m" the target
states are made temporarily absorbing (the standard first-passage trick) so cures do not leak the event.

**Covariate-freezing assumption (stated up front, because it governs every result below).** Only the
*deterministic* covariates evolve along the chain — loan age, remaining term, calendar seasonality. The
entire **macro block is frozen at t0**: rate incentive, unemployment, HPI-derived features. Predictions
are therefore *conditional on the t0 macro environment*; there is no macro forecasting. This is the
conservative, assumption-light choice — macro scenario paths are a future-work hook, not in scope — and
it is the single assumption that the COVID anchor below is designed to stress.

Two pool schemes (§2), both on the t0-alive `current` loans with non-null FICO/rate/LTV: **characteristic
buckets** (FICO × original-rate-quartile × LTV, ≥2,000 loans/cell, ~36–40 surviving cells) — the
interpretable cross-pool partition a structurer would use; and **random pools** (B=500 of N=1,000) — the
paper's bucketing-free scatter device. Five regime-spanning anchors, each scored by its own frozen
window-k models: **Dec2014** (calm), **Dec2018** (late cycle), **Dec2019** (entering the COVID
forbearance shock), **Dec2022** (rate spike), **Dec2024**. Realized counts reconcile with panel
aggregates to Δ=0 over the exhaustive partition at every anchor (M14 Accept); random-pool membership is
seed-reproducible and was re-verified bit-for-bit in M15a.

## 2. Pool-level accuracy (F4.1, T4.2, F4.3) — the regime exhibit

**F4.1 (`F_m14_scatter_random_prepaid_k*`)** is the flagship scatter: predicted vs realized 12-month
prepayment counts, 500 random pools, per model. At Dec2024 the *level* improvement is stark — the
covariate-free empirical matrix predicts ≈2.5× too many prepayments (its flat CPR floor ignores that the
2024 cohort is deep out-of-the-money), the logit closes most of the gap, the ensemble sits nearest the
45° line (RMSE 82.9 → 28.0 → 11.9 loans). The **R² is negative even for the ensemble**: random pools are
near-homogeneous 1,000-loan draws, so cross-pool realized variance is tiny and R² is dominated by a small
level bias. The random scheme tests *level calibration*, not discrimination — the discriminating metric
lives in the characteristic buckets (so the T4.2 R² panel, and the chapter, read the **char** scheme;
random-pool R² is not tabulated, per 02c §4).

**T4.2 (`t_m14_pool_counts`, characteristic buckets)** — prepaid R² / RMSE by model × anchor, the
**by-anchor regime exhibit**, *empirical / logit / ensemble*:

| Anchor | prepaid R² (E/L/Ens) | prepaid RMSE (E/L/Ens) | reading |
|---|---|---|---|
| Dec2014 (calm)       | 0.575 / 0.980 / 0.981 | 1517 / 328 / 320  | covariate models dominate |
| Dec2018 (late cycle) | 0.744 / 0.607 / 0.791 | 1118 / 1387 / 1011 | ensemble best |
| Dec2019 (COVID)      | **0.647** / 0.537 / **0.449** | 3054 / 3500 / 3817 | **inversion** — empirical best, ensemble worst |
| Dec2022 (rate spike) | **−4.616** / 0.742 / **0.848** | 3303 / 708 / 542 | empirical collapses; ensemble best |
| Dec2024              | **−1.455** / 0.651 / **0.939** | 2627 / 991 / 415 | empirical collapses; ensemble best |

**The headline finding is not "nonlinearity wins" but shock-type dependence: which model prices a pool
best depends on whether the shock lands in a covariate-captured channel.** A **rate/prepayment-incentive
shock** (2022–24) is exactly the channel the rate-incentive covariates encode, so the covariate-free
empirical matrix is catastrophic (R² −4.6, −1.5) and the ensemble is strongest (0.85, 0.94; RMSE 415 vs
the logit's 991 at Dec2024). A **forbearance/credit shock** (2020) inverts this: at Dec2019 the
assumption-free empirical floor is *best* (R² 0.647, lowest RMSE 3054) and the **ensemble is worst**
(0.449) — it bakes in hardest a pre-COVID prepayment signal that March 2020 abruptly invalidates. The
ensemble beats the logit on prepaid in **4 of 5** anchors, substantially so off-COVID, and loses only at
the 2020 break — so its value is real and *regime-conditional* at pool aggregation, unlike the flat
~1e-4 loan-level NLL edge of `02c §3`.

**F4.3** is the characteristic-bucket count exhibit (`F_m14_scatter_char_prepaid_k*`, point area ∝ pool
size): the per-bucket predicted-vs-realized scatter from which the T4.2 R²/RMSE are computed; signed
per-bucket error is its pricing analogue F5.2 (§5). For the rare **60+ DPD** outcome, RMSE is the
reliable metric (with ~16 buckets R² is unstable across all models) — the ensemble is the only model
with positive 60+ DPD char-bucket R² at Dec2022 (0.635) and Dec2024 (0.261).

## 3. Economic translation: CPR, WAL, price (T5.1, F5.2) — the contribution

The cashflow engine (`pool.cashflow_engine`) is a deterministic level-pay pass-through: given a pool's
WAC/WAM/UPB and a monthly SMM path, it amortises month by month (scheduled principal + SMM-driven
prepayment), carries the gross note coupon, and discounts at `y = WAC − 25 bp` to a **price** (per 100
face) and **WAL** (years). The discount curve is a pool property, **identical across models**, so all
price dispersion isolates the prepayment-model component (§5.2). It is hermetic and **matches both closed
forms to ~1e-8** (zero-prepay = annuity; constant-SMM = the survival schedule `B_h = UPB·φ_h·(1−λ)^h`),
plus a par identity and a price-monotone-in-speed check (`test_pool.py`, 11/11). Errors per pool are
**model-implied minus realized-SMM** valuation through the same engine: CPR error (pp, on the annualised
12-mo-mean SMM), WAL error (months), price error (per 100 face).

**T5.1 (`t_m15_econ_errors`, characteristic buckets) — the headline, per anchor.** The ensemble's
reduction in mean |price error| vs the logit, with the **absolute** mean |price error| (per 100 face)
for both models so the magnitude is concrete:

| Anchor | Logit \|price err\| | Ensemble \|price err\| | reduction |
|---|---|---|---|
| Dec2014 (calm)       | 0.148 | 0.121 | **+18%** |
| Dec2018 (late cycle) | 0.560 | 0.418 | **+25%** |
| Dec2019 (COVID)      | 0.583 | 0.587 | **−1%** (inverts) |
| Dec2022 (rate spike) | 0.254 | 0.176 | **+31%** |
| Dec2024              | 0.233 | 0.122 | **+48%** |

**Reported per anchor, never pooled** — a single pooled number would average the COVID inversion away and
read as a flat improvement, which is precisely the wrong conclusion. The signature is the same as the
counts (§2): the ensemble prices better off-COVID and under the rate shock, and inverts at the 2020
forbearance break. The aggregate magnitudes are economically modest — fractions of a point per 100 of
face, as expected when the discount curve is fixed and pools diversify idiosyncratic timing — so the
contribution is *where* the error concentrates, not the pooled level.

**F5.2 (`F5.2_price_error_buckets_k*`) — where the linear model misprices.** Signed price error across
FICO × rate-quartile cells (UPB-weighted, LTV collapsed), per model. At the **Dec2022** rate-spike anchor
the pattern is sharpest: the empirical matrix misprices everywhere (−0.7 to −1.2 per 100, over-predicting
prepayment uniformly); the logit is benign in low-rate cells but its error **concentrates in the
high-incentive top rate quartile (−0.35 to −0.61 per 100)** — its linear-in-incentive form over-fires
prepayment on high-coupon loans whose incentive had in fact collapsed after the 2022 rate rise — and the
ensemble compresses exactly those cells to **−0.15 to −0.32**. This is the pricing mirror of the
prepayment S-curve (memo 01 / `02 §`'s single-variable nonlinearity): the linear model breaks where the
hazard is most curved, and that is where dollars of mispricing collect. (At Dec2024 the logit's error is
more uniform and the ensemble's residual concentrates in the highest-incentive corner — the concentration
is itself regime-dependent, sharpest under the rate shock.)

**Why price is not simply average CPR** (a point that shows the engine is doing real work, not relabelling
CPR). Price depends on the *timing* of returned principal — the full monthly SMM path, hence the WAL —
not on the average speed alone. The two can rank models differently: at **Dec2014** the logit's mean CPR
error is the *smaller* (2.34 vs the ensemble's 3.28 pp), yet the ensemble still prices the buckets better
(+18% lower price error), because it places its prepayments at more accurate *months*. The engine is
internally coherent — across all pools and anchors `sign(price error) == sign(WAL error)` in **99.4–100%**
of cases (slower prepay ⇒ longer life ⇒ higher price on these premium pass-throughs). Crucially, the
price-vs-CPR sign disagreements occur **only where the CPR error is itself negligible** (median 0.5–1.6 pp
at the disagreements vs 2–4.5 pp where they agree): when average speeds nearly match, timing is all that
remains, and only the price/WAL errors can see it.

## 4. Stated caveats (one paragraph, not a limitations section)

These are **prepayment-model** valuation errors at a fixed discount curve with the macro environment
frozen at t0 — not full option-adjusted spreads. The one regime where the frozen-macro assumption bites
hardest is the one it is built to expose: at **Dec2019 every model under-predicts the realized 33% CPR by
~17 pp**, because all are blind to the March-2020 rate collapse and refi wave that no t0 covariate could
anticipate (empirical 15.2%, logit 15.6%, ensemble 13.8% vs realized 33.0%). Far from a flaw, this is the
cleanest statement of the design's boundary, and exactly why the value of nonlinearity must be read as
regime-conditional. Credit events enter the engine's first pass as zero-recovery terminations (a recovery
haircut is an optional sensitivity, not run); conditional independence of loans given covariates is
inherited from §4 / the paper.

## 5. Contributions (what is original here vs Sirignano et al.)

Sirignano, Sadhwani & Giesecke (2021) established the deep-learning transition model and its loan- and
pool-level machinery on subprime-heavy private-label data through ~2014. This study's original
contributions, which this memo's exhibits substantiate, are: **(i)** the same modelling question posed on
a **prime conforming credit box** (Fannie Mae Single-Family, 2000–2025), where the architecture search
selects a *shallower* optimum (3 layers vs the paper's 5) — the nonlinearity premium is real but
structurally simpler; **(ii)** **eleven test years the literature has not examined, 2015–2025**, including
the COVID-era forbearance episode and the 2022–23 rate shock — regimes that post-date the paper's sample
entirely; **(iii)** a **rolling eleven-window backtest** in place of the single temporal split, whose
output is itself a finding — a *time series of the value of nonlinearity* that **widens under the rate
shock, compresses, and inverts at the forbearance break** (the §2/§3 regime exhibit); and **(iv)** an
**economic translation to CPR, WAL, and price** — restating the model comparison in the units an MBS
investor actually prices in, with the per-anchor ensemble-vs-logit price-error reduction (up to +48%
off-COVID) and the bucket-level mispricing map (F5.2) as the headline economic results, rather than
log-likelihood alone.

**Deferred to future work (not in scope):** the paper's §5.2-style portfolio-decile sort (rank pools,
form best/worst portfolios). Rationale (methodology §"Why pricing rather than portfolio sorts"): a sort
tests only the *ranking* of the same per-loan probabilities, whereas pricing tests their *levels*; the
ranking is already tested upstream (AUC) and the calibration via reliability diagrams, so a sort would
duplicate F5.2's information without the dollar magnitudes. A second future-work hook is **macro scenario
paths** along the roll-forward (relaxing the frozen-t0 assumption of §1), which would convert the COVID
inversion from a boundary condition into a testable forecast.
