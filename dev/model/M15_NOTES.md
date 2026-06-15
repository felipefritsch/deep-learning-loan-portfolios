# M15a — cashflow engine + monthly pool SMM paths (`03_POOL_LEVEL §5.1–§5.2`)

First of three M15 sub-sessions. Two parallel pieces: **(B)** the deterministic level-pay
pass-through cashflow engine (hermetic, the time-variable gate) and **(A)** the UPB-weighted
monthly pool SMM paths from the roll-forward. Error metrics + T5.1/F5.2 are **M15b** (not here).

## (B) Cashflow engine — `pool.cashflow_engine` (§5.2)

~100-line deterministic level-pay pass-through: given a pool `(WAC, WAM, UPB)` and a monthly SMM
vector (constant-extrapolated past its end), recompute the level payment on the surviving balance
each month → scheduled amortisation + SMM-driven prepayment → per-month cashflow, then **price**
(per 100 face) and **WAL** (years). Cashflows carry the gross note coupon `WAC`; discount at
`y = WAC − servicing_spread` (25 bp). A positive spread makes a premium pass-through whose price
*falls* as prepayment speeds up — with the curve fixed (`WAC` is a pool property, same for every
model) all price dispersion isolates the prepayment model (§5.2).

**Hermetic closed-form tests** (`test_pool.py`, 11/11 pass, all to ~1e-8 — the M13-toy-chain
analogue):
- **par identity** — `servicing_spread=0` ⇒ price ≡ 100 for *any* SMM (zero/constant/random),
  by telescoping `Σ cf_h/(1+c)^h = B_0 − B_n/(1+c)^n`, `B_n=0`. Strongest check: exercises
  amortisation + prepay + discounting at once.
- **zero-prepay annuity** — `smm≡0` ⇒ constant level payment; price = closed-form annuity PV.
- **constant-SMM survival schedule** — `smm≡λ` ⇒ `B_h = UPB·φ_h·(1−λ)^h` (φ = no-prepay
  scheduled-balance fraction); balance/cashflow/price/WAL reconstructed independently match.
- plus price strictly decreasing in CPR (premium erosion) and the constant-extrapolation identity.

Sanity (4.25% WAC, 330-mo, 25 bp): price 101.95 → 100.34 and WAL 10.4 → 1.4 yr as CPR 5% → 50%.

## (A) Monthly pool SMM paths — `smm_paths.py` + `pool.roll_forward(capture_smm=…)`

`roll_forward` extended to capture, per loan per step, the composed cumulative-prepaid mass and
the transient (alive) mass (default off — M13/M14 behaviour unchanged; the 6 existing engine tests
still pass). `smm_paths.py` aggregates these UPB-weighted to a pool monthly path
`SMM_P(h) = Σ_i w_i·Δ_h(i) / Σ_i w_i·alive_i(h−1)` (`w_i` = Current Actual UPB at t0, frozen per
the §3 covariate-freezing assumption; a balance-of-survivors-weighted conditional monthly prepay
rate), with the realized analogue from the panel over the same `[t0, t0+12)` transitions. Output:
`outputs/tables/pool_level/smm_paths_k{k}.parquet`, long over `(scheme, pool, model)` with
`smm_h01..smm_h12` + engine-ready `wac` (annual decimal) / `wam` (months) / `upb` / `n_loans`.

**Pools are exactly M14's** (no rebuild): same population (t0-alive `current`, non-null
fico/rate/ltv, sorted), same seeded `pools.char_cell_ids` / `random_pool_index`. Verified by the
**membership cross-check** — recomputed random-pool realized prepaid counts match the committed
`pools_random_k{k}.parquet` `prepaid_realized` column **exactly (max|Δ|=0 over 500 pools)** at
every anchor. Null Remaining-Months on ≈0.3% of current loans (M14 only drops nulls on
fico/rate/ltv) are kept in the pool (0-weight / excluded from the WAM mean only — they don't touch
the SMM path).

**Run:** `nohup caffeinate -i .venv/bin/python -u dev/model/smm_paths.py --models
empirical,ensemble --device cpu` (CPU background; ~20 min, 5 anchors), log
`logs/m15/smm_paths.log`. Models scored: **empirical + ensemble + realized** (logit pending — see
below).

**Sanity — mean CPR by anchor (random pools), model vs realized** (the regime story of M14, now
in prepayment-speed units):

| Anchor | Empirical | Ensemble ×8 | Realized | reading |
|---|---|---|---|---|
| Dec2014 (calm)        | 17.1% | 15.7% | 13.4% | all close, ensemble between |
| Dec2018 (late-cycle)  | 15.6% | 7.9%  | 12.2% | empirical overshoots, ensemble undershoots, realized between |
| Dec2019 (COVID)       | 15.2% | 13.8% | **33.0%** | frozen-t0 macro misses the Mar-2020 refi wave (both models blind to the rate collapse) — the headline caveat |
| Dec2022 (rate spike)  | 16.3% | **3.0%** | **4.0%** | ensemble tracks the prepay collapse; the covariate-free empirical floor stays stuck high |
| Dec2024               | 15.2% | **5.6%** | **6.0%** | ensemble ≈ realized, empirical floor too high |

All SMM ∈ [0,1]; empirical is a ~flat covariate-free CPR floor (constant over h), the ensemble
adapts to the regime. (The economic-error read — CPR/WAL/price error vs realized-SMM valuation,
T5.1/F5.2 — is M15b.)

## Logit — recompute pending (merged in M15b)

The frozen **logit/full** checkpoints (k=2015/2019/2020/2023/2025) were **never synced off the GPU
pod** (only NN + ensemble were); `logit/dev/k2015` is the only survivor, and the pod is terminated.
So M15a's SMM run covers empirical + ensemble + realized; logit is added later.

`refit_logit.py` recomputes them **memory-safely on the Mac** (the pod's whole-split-resident
`backtest.fit_logit` overflows 16 GB on the 36–63 M-row slices): it reuses the saved scaler/vocab
from the NN run folders (byte-identical to logit's), streams the train slice via `WindowLoader`,
and fits the identical `LogitEmbNet` / weighted-CE / hyperparameters / L2 grid. The logit is convex
(+ embedding zero-init + L2), so the recompute reproduces M14's logit up to optimiser noise.
**Self-gating:** fit k=2015 first, validate its 12-month random-pool counts against the committed
`logit_*_pred` to ≤1e-3 prepaid rel-RMSE; only if it passes does it fit the other 4 windows and
**merge** logit's SMM path into `smm_paths_k*.parquet` (`smm_paths --add-logit`). If k=2015 misses,
it STOPS for a GPU-pod fallback (`backtest.fit_logit`). Launched detached; the validation + merge
land in M15b.

## Side-fix — Python 3.9 import compatibility

M11–M14 ran on the pod (Python ≥3.12); this Mac is 3.9.6. `evaluate.py` used a 3.12-only
backslash-in-f-string (in the AUC `.tex` writer) that crashed `import` of the whole Phase-3 chain.
Fixed surgically (escape computed outside the f-string expression) — LaTeX-string only, no
numerics. `table_a.py` has the same construct but is not in the M15 import chain (left as-is).

## Artefacts
- Code: `pool.py` (`cashflow_engine` + `roll_forward(capture_smm/which)`), `smm_paths.py`,
  `refit_logit.py`, `test_pool.py` (+5 engine tests), `evaluate.py` (3.9 fix).
- Outputs (SSD): `outputs/tables/pool_level/smm_paths_k{2015,2019,2020,2023,2025}.parquet`
  (empirical/ensemble/realized; logit merged in M15b), `models/nn/full/smm_paths_summary.json`,
  `logs/m15/smm_paths.log`.
