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

**Run:** `nohup caffeinate -i .venv/bin/python -u -m floan.model.smm_paths --models
empirical,ensemble --device cpu` (CPU background; ~20 min, 5 anchors), log
`logs/m15/smm_paths.log`. Models scored: **empirical + ensemble + realized** in the first pass;
**logit merged later** from the recovered M14 checkpoints (`--add-logit`, see below) — all four
models now present.

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
and fits the identical `LogitEmbNet` / weighted-CE / hyperparameters / L2 grid. **Self-gating:**
fit k=2015 first, validate its 12-month random-pool counts against the committed `logit_*_pred` to
≤1e-3 prepaid rel-RMSE; only if it passes does it fit the other 4 windows and **merge** logit's SMM
path; if k=2015 misses, STOP for a GPU-pod fallback.

**RESOLVED — the original M14 logit/full checkpoints were recovered (no re-fit needed).** They
were never lost, just never synced off the pod's persistent volume; all 11 windows were rsync'd to
`models/logit/full/` (k2015 val_nll 0.09981142 — bit-matches the committed `backtest_summary.json`;
seed 0; variant full). The Block-D gate (recovered logit's 12-month random-pool counts vs the
committed `pools_random_k*.parquet` `logit_*_pred`) **PASSES at all 5 anchors to ~1e-8** (prepaid
rel-RMSE 1.9e-8–1.2e-7, max|Δ| rounds to 0.0000; ~5 orders of magnitude inside the 1e-3 bar) — i.e.
the recovered checkpoints reproduce M14 to re-scoring float noise. Logit's SMM path was then merged
into `smm_paths_k*.parquet` (`smm_paths --add-logit`); **all four models — empirical / logit /
ensemble / realized — are now present and row-aligned** (verified: identical pool sets + identical
per-pool WAC/WAM/UPB across models; SMM ∈ [0,1]) at every key anchor. Logit CPR by anchor: Dec2014
12.2%, Dec2018 7.2%, Dec2019 15.6% (vs realized 33.0% — same frozen-macro COVID miss), Dec2022 4.5%
(vs realized 4.0%), Dec2024 5.3% (vs realized 6.0%).

*Dead-end recorded for posterity:* the memory-safe streaming re-fit (`refit_logit.py`) reproduced
M14's k=2015 **val NLL** (0.099951 vs 0.09981) but its 12-month pool counts diverged **5.7%**
(`logit/full/refit_gate_k2015.json`) — because the logit is early-stopped and weighted-CE is
dominated by `current→current`, so it is insensitive to the small prepay-hazard tail the pool count
is built from; streaming's batch order ≠ M14's resident global-shuffle → matching NLL, different
early-stop point. The recovery made this moot, and it's a clean illustration that *matching loss ≠
matching the tail an economic exhibit depends on*. The ensemble-vs-logit T5.1/F5.2 headline is
**M15b** (now unblocked — all four SMM paths exist).

## Side-fix — Python 3.9 import compatibility

M11–M14 ran on the pod (Python ≥3.12); this Mac is 3.9.6. `evaluate.py` used a 3.12-only
backslash-in-f-string (in the AUC `.tex` writer) that crashed `import` of the whole Phase-3 chain.
Fixed surgically (escape computed outside the f-string expression) — LaTeX-string only, no
numerics. `table_a.py` has the same construct but is not in the M15 import chain (left as-is).

## Artefacts
- Code: `pool.py` (`cashflow_engine` + `roll_forward(capture_smm/which)`), `smm_paths.py`,
  `refit_logit.py`, `test_pool.py` (+5 engine tests), `evaluate.py` (3.9 fix).
- Outputs (SSD): `outputs/tables/pool_level/smm_paths_k{2015,2019,2020,2023,2025}.parquet`
  (**all 4 models**: empirical/logit/ensemble/realized, row-aligned), `models/nn/full/`
  `smm_paths_summary.json` + `smm_paths_summary_merge_logit.json`, `logs/m15/{smm_paths,
  logit_gate,logit_merge}.log`. Recovered logit checkpoints: `models/logit/full/k*/`.

---

# M15b — CPR / WAL / price errors + T5.1 + F5.2 (`03_POOL_LEVEL §5.3`)

Pure post-processing of M15a (no re-fit, no re-score, no GPU; runs in seconds on the Mac).
`economics.py` reads the merged `smm_paths_k*.parquet` (4 models × {char, random} pools, each with
the pool's WAC/WAM/UPB + 12-mo SMM path), runs each model's path **and** the realized path through the
M15a `pool.cashflow_engine` (same WAC/WAM/UPB, same WAC−25bp curve → price dispersion isolates the
prepayment model, §5.2), and forms per-pool errors = **model − realized-SMM valuation**:
CPR error (pp, on the annualised 12-mo-mean SMM, the M15a convention), WAL error (months), price error
(per 100 face). Sign convention: a model that under-predicts prepay on the premium pass-through
over-values it (`price_err>0`), dates cashflows late (`wal_err>0`), `cpr_err<0`.

## Headline — char-bucket ensemble-vs-logit reduction in mean |price error| (per anchor)

| Anchor | Logit mean&#124;price err&#124; | Ensemble | Reduction | reading |
|---|---|---|---|---|
| Dec2014 (calm)       | 0.148 | 0.121 | **+18%** | ensemble ahead |
| Dec2018 (late-cycle) | 0.560 | 0.418 | **+25%** | ensemble ahead |
| Dec2019 (COVID)      | 0.583 | 0.587 | **−1%**  | **inverts** — frozen-t0 macro misses Mar-2020 refi wave (both miss together: CPR bias ≈ −17pp for *every* model) |
| Dec2022 (rate spike) | 0.254 | 0.176 | **+31%** | ensemble ahead |
| Dec2024              | 0.233 | 0.122 | **+48%** | ensemble ahead |

**Reported per anchor, never pooled** (a pooled mean would average the COVID inversion away). This is the
**same regime signature as the M14 counts** (`M14_NOTES §3`: ensemble beats logit off-COVID, inverts at
Dec-2019) now in *valuation* units — the economic translation of T4.2. Char buckets are the headline
(the interpretable cross-pool metric, as in T4.2). Random pools show the same off-COVID/COVID pattern but
attenuated (Dec2014 +10%, Dec2018 +14%, Dec2019 −8%, Dec2022 +17%, Dec2024 +4%): near-homogeneous draws
are a level-bias near-tie at the aggregate — **the value of nonlinearity lives in the cross-bucket
discrimination, not the population aggregate** (at random-pool aggregate CPR, logit even edges the ensemble
at Dec2014/Dec2022; cf. M14's "R² reflects level bias (homogeneous draws)").

## F5.2 — price error by characteristic bucket (where the linear model breaks)

`F5.2_price_error_buckets_k{k}.{png,pdf}`: UPB-weighted signed price error over FICO × original-rate-quartile
cells (LTV collapsed), empirical/logit/ensemble side by side. The expected S-curve-mirror concentration is
**clearest at Dec2022**: logit's largest errors sit in the high-incentive top-rate-quartile column
(−0.35 to −0.61 — it over-fires prepay on high-coupon loans whose incentive vanished after the 2022 spike),
and the ensemble compresses exactly those cells (−0.15 to −0.32) → the +31% headline. Empirical is a
uniform large miss at every anchor (the covariate-free CPR floor: ~−0.7 to −1.25 per 100 at Dec2022).
At Dec2024 the logit error is more uniform (broad under-firing) and the ensemble's residual concentrates in
the highest-incentive corner — i.e. the concentration is regime-dependent, sharpest under the rate shock.

## Integrity checks (no SSD/GPU; `test_economics.py` 3/3 + `test_pool.py` 11/11 still green)

- **Engine internally coherent:** per-pool `sign(price_err) == sign(wal_err)` at **99.4–100%** of pools at
  every anchor (slower prepay ⇒ higher price ⇒ longer WAL — the two PV-of-timing measures are sign-locked);
  every pool prices ≥ 100 (premium by the 25-bp strip); all CPR/WAL/price finite.
- **CPR (mean-SMM) vs price are *not* sign-locked** — they diverge on ~15–28% of pools off-COVID, and the
  divergences sit **exactly where `|CPR error|` is small** (median 0.5–1.6pp at mismatch vs 2–4.5pp at
  coherent pools). This is correct, not a bug: CPR summarises *average speed*, price summarises the
  *present value of prepayment timing*; when the level error is tiny, timing flips the price/WAL sign while
  the mean SMM can't see it. (A clean small argument for pricing-over-speed as the consumer metric — feeds
  the §"why pricing" framing, but stated only as an observation here.)
- **Tests:** `test_economics.py` — zero error when model path == realized; the premium under/over-prediction
  sign convention end to end; the T5.1 `aggregate` mean|err|/bias/headline-reduction arithmetic.

## M15b artefacts
- Code: `economics.py` (engine driver + T5.1 + F5.2 + per-pool detail + summary; `per_pool_econ` takes an
  optional in-memory frame for hermetic tests), `test_economics.py`.
- Outputs (SSD): `outputs/tables/pool_level/t_m15_econ_errors.{md,csv,json}` (T5.1),
  `econ_errors_k{2015,2019,2020,2023,2025}.parquet` (per-pool price/WAL/CPR + errors),
  `outputs/figures/pool_level/F5.2_price_error_buckets_k*.{png,pdf}` (5 anchors),
  `models/nn/full/pool_m15b_summary.json` (headlines + Accept evidence).
- **Accept (Phase-3 §5):** T5.1/F5.2 exist for ensemble vs logit at **5** regime-spanning anchors (≥3
  required) with the headline |price-error| reduction stated per anchor → **PASS**. Memo (M15c) not written.
