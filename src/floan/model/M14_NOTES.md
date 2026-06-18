# M14 — Pools + predictions + realized outcomes (`03_POOL_LEVEL §2 / §4`)

`pools.py`: consume the M13 roll-forward engine to build **pool-level** predicted counts (both
schemes of `§2`), attach the **realized** 12-month outcomes from the panel, and produce the
predicted-vs-realized scatter + R²/RMSE table that is the statistical half of the Phase-3
pool exhibit. Economic translation (CPR/WAL/price) is M15 — not started here.

## 1. Design

- **Anchors:** the five ensemble-bearing key windows (`02 §6`) k ∈ {2015, 2019, 2020, 2023,
  2025} → t0 ∈ {2014-12, 2018-12, 2019-12, 2022-12, 2024-12}. Covers the spec **minimum**
  (2019-12, 2022-12, 2024-12) and adds 2014-12 (calm) + 2018-12 (late-cycle) for regime span.
  Every anchor carries an ensemble, so "ensemble vs logit vs empirical at every anchor run"
  holds at all five. (The non-key windows have no ensemble and so cannot serve that headline;
  the five key windows already span calm → COVID → rate-shock.)
- **Population (both schemes):** the t0-alive `current` loans (`§2` — characteristic buckets
  partition the current loans, random pools draw "from the same population"), restricted to
  non-null FICO/rate/LTV so one population serves both schemes and the reconciliation is exact.
  Dropped fraction is FICO-only and ≈0.1% (e.g. k=2020: 92 of 118 k → here 557 of 718 k at full
  scale, 0.08%).
- **Outcomes**, horizon-aligned with the M13 chains:
  - **prepaid ≤ 12m** — model: composed `p[prepaid]`; realized: `state_next == prepaid` at any
    of the 12 monthly transitions from t0.
  - **60+ dpd ≤ 12m** — model: first-passage `p_fp[dpd_60]`; realized:
    `state_next ∈ {dpd_60, dpd_90plus, foreclosure, REO}` (monotone delinquency ⇒ any deeper
    state passes through 60+).
  The realized window is exactly `config.test_bounds(k) = [Dec(k−1), Dec(k))` — the same 12
  transitions the roll-forward composes (the panel is read via the unthinned **eval pool**,
  which is the panel slice for the fixed eval-loan block; there is no separate panel on the
  modelling SSD).
- **Characteristic buckets (`§2.1`):** FICO fixed `[680,720,760]` (4 bands) × original-rate
  per-anchor quartiles (4) × LTV fixed `[80,90]` (3) = 48 cells; cells below `MIN_CELL=2000`
  loans are merged into a single catch-all so the partition stays exhaustive (exact realized
  reconciliation). At full scale ~40 surviving cells/anchor.
- **Random pools (`§2.2`):** B=500 pools of N=1000 current loans, drawn without replacement
  within a pool, independent across pools, seeded on `(SEED=0, k)` ⇒ reproducible.
- **Predicted count / interval (`§4`):** count = Σ_i p_i; 90% interval = μ ± 1.645·σ with the
  Poisson-binomial Normal approximation μ=Σp, σ²=Σp(1−p).

## 2. Accept criteria — evidence

Full run (`pools.py --device cuda`, all 5 anchors, **5.5 min**, GPU box) — `ALL M14 ACCEPT
CRITERIA PASS`. Summary `models/nn/full/pool_m14_summary.json`.

**(1) Pool memberships reproducible (seeded)** — for every anchor, two independent seeded
builds of the 500×1000 random-pool membership matrix are bit-identical
(`np.array_equal`): k=2015/2019/2020/2023/2025 all PASS. (Unit test
`test_random_pool_reproducible_and_anchor_distinct`: same `(seed,k)` → identical, different
anchor → different.)

**(2) Realized counts reconcile with panel aggregates** — Σ realized over the exhaustive
characteristic partition equals an independent direct total over the population, **Δ=0** for
both outcomes at every anchor:

| Anchor | prepaid Σcells == direct | 60+ Σcells == direct |
|---|---|---|
| Dec2014 | 65,244 (Δ0) | 3,754 (Δ0) |
| Dec2018 | 75,724 (Δ0) | 3,756 (Δ0) |
| Dec2019 | 165,574 (Δ0) | 27,936 (Δ0) |
| Dec2022 | 40,891 (Δ0) | 5,395 (Δ0) |
| Dec2024 | 49,947 (Δ0) | 5,896 (Δ0) |

**(3) Predicted-vs-realized scatter + R²/RMSE table for ensemble vs logit vs empirical at
every anchor** — all three models present and all four scatter figures written per anchor:
- `outputs/figures/pool_level/F_m14_scatter_random_{prepaid,dpd60p}_k{k}.{png,pdf}` (§4 F4.1
  random-pool scatter, 5 anchors) and the companion `F_m14_scatter_char_*` (characteristic
  buckets, point area ∝ pool size).
- `outputs/tables/pool_level/t_m14_pool_counts.{md,csv,json}` — R²/RMSE by model × outcome ×
  anchor for **both** schemes; per-pool counts/intervals in `pools_random_k{k}.parquet`.

**(4) Sanity** — all R²/RMSE finite, predicted/realized counts ≥ 0, 90% coverage ∈ [0,1],
population > 0 at every anchor (n_pop 493k–718k). PASS.

## 3. Results worth a memo note (M15)

Characteristic-bucket **prepaid** R² (the interpretable cross-pool metric):

| Anchor | Empirical | Logit | Ensemble |
|---|---|---|---|
| Dec2014 (calm) | 0.575 | 0.980 | **0.981** |
| Dec2018 | 0.744 | 0.607 | **0.791** |
| Dec2019 (COVID) | **0.647** | 0.537 | 0.449 |
| Dec2022 (rate spike) | −4.616 | 0.742 | **0.848** |
| Dec2024 | −1.455 | 0.651 | **0.939** |

- The **value of nonlinearity is regime-conditional** (the `§6` exhibit): the ensemble cuts
  characteristic-bucket prepay RMSE vs logit by **+58% (Dec2024)**, **+27% (Dec2018)**, **+23%
  (Dec2022)** — but at the **COVID anchor (Dec2019) it is *worse* than the covariate-free
  empirical floor** (R² 0.449 vs 0.647; characteristic-bucket prepay RMSE +9% vs logit and +25%
  vs the empirical floor — random-pool RMSE +4% vs logit). The cause is the
  frozen-t0 macro assumption (`§3`): predictions are made at Dec-2019 incentives, blind to the
  March-2020 rate collapse, so the loan-level refi nonlinearity fires on the *wrong* signal and
  the broad refi wave is best tracked by the flat historical average. This is the headline
  caveat for the M15 memo, not a bug. (Fuller two-regime / shock-type framing — forbearance
  shock 2020 vs rate/prepay-incentive shock 2022–24 — and the §4.3 write-up feed:
  `writeup/memos/02c_robustness_caveats.md §4`.)
- **60+ dpd** counts are harder for all models (smaller event base); the ensemble is the only
  model with a positive characteristic-bucket R² at Dec2022 (0.635) and Dec2024 (0.261).

## 4. Artefacts

- Code: `pools.py` (population/realized/bucketing/counts/metrics/figures/driver),
  `test_pools.py` (8 hermetic unit tests — bucketing partition+merge, reconciliation,
  seeded reproducibility, count/interval math, R²/RMSE/coverage closed forms).
- Outputs (SSD): `outputs/tables/pool_level/t_m14_pool_counts.{md,csv,json}`,
  `outputs/tables/pool_level/pools_random_k{2015,2019,2020,2023,2025}.parquet`,
  `outputs/figures/pool_level/F_m14_scatter_{random,char}_{prepaid,dpd60p}_k*.{png,pdf}`
  (40 figures), `models/nn/full/pool_m14_summary.json` (all numbers + Accept evidence).
