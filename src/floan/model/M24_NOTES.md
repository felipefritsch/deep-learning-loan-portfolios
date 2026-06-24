# M24 — Rolling pricing grid (horizon × regime): notes, verification + corrections

**Status:** built, run at full scale (all 11 anchors × H{1,3,6,12}, raw torch), **verified, and
corrected**. Branch `m24-pricing-grid` (off `m23-calibration`). Code: commit **`e351cf1`** (build) +
**`03c0ddc`** (verification fixes).
Spec: `ECONOMIC_ENGINE.md §5,§9`; `03_POOL_LEVEL §2–5`; task `04_TASKS.md` M24.
Scope: full-scale H×regime grid for the torch models (logit / best-NN / 8-net ensemble), **raw**
(M23: calibration is a no-op for the torch path, T≈1 → single pass, no calibrated second compose).
**No GBT** (M19 not built).

> **AUTHORITATIVE FOR THE WRITEUP.** Use the numbers in §3–§4 below. The **original-run** headline was
> **smoke-contaminated at k2020** and its loan→pool gap was a **NaN-WAM dilution artifact** — both are
> superseded here. Do **not** lift numbers from the first run's tables/logs.

## 1. What M24 produces
Per anchor, one `capture_smm` roll to H=12 (snapshots {1,3,6,12} are free) feeds three exhibits:
- **T5.1** — pool CPR/WAL/**price** error by model × anchor × H; headline = ensemble/NN-vs-logit
  mean|price-error| reduction. **Pool-level** (`economics.per_pool_econ` → `pool.cashflow_engine`).
- **T4.2** — pool count R²/RMSE by {prepaid, 60+} × anchor × H (60+ first-passage captured off the
  same roll — `cum_dpd60p`).
- **F5.2** — signed price-error by characteristic bucket, per anchor × H (44 heatmaps).
- **Loan-level** — full-population per-loan WAL/price distribution (`pool.price_loans_vec`,
  vectorised) + the §4.3 loan→pool aggregation reconciliation.
Resumable/atomic per anchor (`.done`); tables assemble from whatever anchors have finished. Runtime:
**~55 min** for all 11 on one A4500 (~4–7 min/anchor; ensemble-5 in ~32 min). Foundations green this
run: M21 seam `--xcheck` ≤2.4e-7 (≤1e-5) at full k2020 scale; M20b [5] ensemble impossible-mass
~3.5–9e-6 (<1e-4) all 5 key windows.

## 2. Bugs found in verification + fixed (commit `03c0ddc`)
**(a) Smoke contamination — table-builder glob.** `pricing_grid._load_completed` globbed
`econ_k*.parquet`, which **also matched `econ_k2020_smoke.parquet`** (the 50k local-smoke file present
on disk when tables were assembled). The smoke subsample's char pools were concatenated into k2020,
inflating its mean|price-error| ~0.006–0.010/100 (k2020 only — only it had a smoke run). **Fix:** match
`econ_k20[0-9][0-9].parquet` (anchors are 4-digit years), which cannot match `*_smoke`. Same hardening
for every k-globbed artifact (`loan_*`, `counts_*`).

**(b) NaN-WAM dilution — `price_loans_vec` pool aggregate.** Loans with null `Remaining Months to
Maturity` (a real ~1–3% UPB share per anchor) got `n=round(nan→0)=0` ⇒ priced at value 0 but their UPB
stayed in the pool denominator, **diluting `pool_price` down** by ~the NaN-WAM UPB share; a single
NaN-**WAC** loan went further and turned the whole pool aggregate NaN (k2017/2024/2025). **Fix:** an
`isfinite` guard on WAC **and** WAM — non-finite-term loans are dropped from **both** the value sum and
the UPB sum (mirroring how the pool-level reference forms its UPB-weighted WAC/WAM over finite entries),
so the loan→pool aggregate is over priceable loans only and stays apples-to-apples. Their per-loan
price stays NaN (the distribution filters it). Guard unit-tested (`test_price_loans_vec_drops_nonfinite_wac_wam`).
*Pool-level T5.1/T4.2/headline are independent of `price_loans_vec` (they price via
`cashflow_engine`) — Fix (b) does not touch them; verified byte-identical (§5).*

## 3. VERIFIED headline — USE THESE (supersede the original run)
**COVID inversion, k2020 (anchor Dec 2019 → rolls through the 2020 shock).** Ensemble-vs-logit mean
|price-error| reduction (char scheme), and the value-of-nonlinearity **inverts and deepens with H**:

| H | logit mean\|perr\| | ensemble | ens-vs-logit | $ price-err (ens) = Σ(perr·UPB)/100 |
|---|---|---|---|---|
| 1 | 0.1822 | 0.1030 | **+43.5%** | $77M |
| 3 | 0.2068 | 0.2600 | **−25.7%** | $346M |
| 6 | 0.4313 | 0.4841 | **−12.2%** | $619M |
| 12 | 0.5827 | 0.5868 | **−0.7%** | $734M |

At H=1 nonlinearity helps (+43.5%); by H≥3 it **hurts** (ensemble error > logit) as the frozen-t0 macro
assumption compounds through COVID — a true sign flip (ens < logit at H1, ens > logit at H3), and the
dollar misprice deepens **$77M → $734M**.

**Off-COVID value of nonlinearity** (anchors k2015/2019/2023/2025): NN/ensemble cut pool price error by
up to **~62% (ensemble) / ~65% (NN)** at the longer horizons — substantial and regime-broad. *Nuance
(traceable): some **short-horizon (H=1)** cells favour logit (e.g. k2015 H1 NN −48.7%); the value of
nonlinearity is a longer-horizon phenomenon. The "~18–65%" range is the dominant positive band.*

> **SUPERSEDED — do NOT use** (smoke-contaminated original run): k2020 H3 **−31.1%**, dollars
> **$84M → $793M**. The corrected, de-contaminated figures are the table above (~−25.7%, $77M→$734M).

## 4. Corrected §4.3 loan→pool reconciliation story
**The aggregation identity holds TIGHTLY.** After Fix (b), the loan→pool recon gap (loan-level
`pool_price` vs the single-aggregate-pool valuation) is small everywhere — **mean|gap| 0.016–0.098 /100**,
worst single cell **0.31/100** — across all 11 anchors:

| anchor | max\|gap\| | mean\|gap\| | | anchor | max\|gap\| | mean\|gap\| |
|---|---|---|---|---|---|---|
| k2015 | 0.152 | 0.087 | | k2021 | **0.315** | 0.094 |
| k2016 | 0.081 | 0.049 | | k2022 | **0.262** | 0.098 |
| k2017 | 0.052 | 0.033 | | k2023 | 0.104 | 0.067 |
| k2018 | 0.067 | 0.038 | | k2024 | 0.185 | 0.078 |
| k2019 | 0.029 | 0.016 | | k2025 | 0.148 | 0.060 |
| k2020 | 0.103 | 0.058 | | | | |

The residual is **structured, not noise**: largest in the **NN-at-H=1** cell for the **rate-sensitive
2021–22 vintages** (k2021/k2022 — ultra-low-coupon pools whose per-loan refi behaviour is most
heterogeneous; their empirical/logit gaps are only 0.03–0.09). That is exactly where a single
UPB-weighted-mean-WAC/WAM pool diverges most from true loan-level pricing.

> **Writeup line:** *the loan→pool aggregation holds tightly (residual ≤0.31/100, mostly ≤0.1); the
> value of loan-level pricing is the **per-loan error distribution** and the residual **concentrating
> where per-loan prepayment heterogeneity is highest** (NN, short horizon, rate-sensitive vintages) —
> NOT "loan-level recovers a 1–3 pt aggregation gap."* The 1–3 pt "gap" in the original run was a
> NaN-WAM **dilution artifact** (it tracked the NaN-WAM UPB share anchor-for-anchor), removed by Fix (b).

## 5. Verification evidence (standing rule 1 — paste the numbers)
- **Headline byte-identical after the loan-level correction.** The all-11 loan-level re-run (Fix b)
  rewrote **only** `loan_k*.parquet`; `econ_k*.parquet` + the assembled tables were never touched —
  **md5: 14/14 files identical** before/after (11 econ + `t_m24_econ_errors.parquet` +
  `t_m24_horizon_regime.json` + `t_m24_pool_counts.parquet`). So §3 is provably unchanged by the §4 fix.
- **k2020 T5.1 == clean recompute** from `econ_k2020.parquet` to **Δ=0.00000** every cell (smoke
  contamination gone).
- **All 11 recon gaps** finite and small (table §4); the 3 formerly-NaN anchors (k2017/2024/2025) now
  finite.
- **pytest:** 35 green (`test_pricing_grid.py` incl. the NaN-WAC/WAM guard + the H=12==M15 regression
  guard; `test_pool.py`).

## 6. Authoritative artifacts
`<SSD>/outputs/tables/pool_level/m24/` — `t_m24_econ_errors.{md,json,parquet}` (T5.1),
`t_m24_pool_counts.{json,parquet}` (T4.2), `t_m24_horizon_regime.{md,json}` (headline + COVID
$-inversion), per-anchor `econ_k{k}.parquet` / `loan_k{k}.parquet` / `counts_k{k}.parquet`, and
`<SSD>/outputs/figures/pool_level/F5.2_*_h*.{png,pdf}` (44). (Data lives on the SSD, gitignored.)

## 7. Deferred (post-deadline; pure cleanup, gates nothing)
- The benign `RuntimeWarning: invalid value in divide` in `price_loans_vec` (the `wac==0` lane
  `np.where` discards — output is correct; wrap in `np.errstate` to silence).
- The verify-`[6]` GBT thread oversubscription + a thread-capped live re-print (see `M20_NOTES §5`).
