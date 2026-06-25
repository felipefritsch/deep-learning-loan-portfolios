# M27b — h=1 pool pricing for the GRU (sequence model in the economic comparison)

**Status:** built + run at full scale (all 11 anchors single-GRU + seed-{0,1,2} ensemble on the 5 key
windows), **H=1 identity-checked (bit-exact, every window)**, priced through the unchanged M24 engine.
Branch `m27a-seq-rolling` (off M27a). The Mac-side merge into the full M24 FF exhibits (T5.1 /
horizon-regime / F5.2) and the **full 11-anchor FF×GRU H=1 comparison** are now done — see **§7**.
Spec: ADR-001 (the Predictor seam; §87 flagged exactly this — the sequence model needs *history*,
not point-in-time covariates); `ECONOMIC_ENGINE §3–5`; M24 (`M24_NOTES.md`) is the FF reference.

**H=1 ONLY.** The deterministic matrix composition is exact at one step. Beyond one step the GRU
would need its trailing-12-month window advanced into the *unobserved* future, so multi-month GRU
pricing is invalid here (future-work Monte-Carlo). Every result below is the one-month horizon —
the GRU's **weakest** horizon (its value-of-memory edge, M27a, is multi-month), framed honestly.

## 1. What M27b adds (surgical; one new Predictor, zero engine change)
- **`SeqPredictor`** (`seq_predict.py`) on the `pool.Predictor` seam: `origin_scores(frame_h)` returns
  the four transient-origin one-step distributions of the window-`k` GRU, scored on each loan's
  trailing-12-month **observed** history ending at the anchor `t0 = Dec(k−1)` (window-`k`
  scaler/vocab from the M27a cache). The origin override is the sequence analogue of
  `TorchPredictor`'s single-row state override: hold the trailing history fixed, set the
  **prediction-point (last real timestep) state** to each origin → row `M₁[s,:] = GRU(history,
  last=s)`. Anchor sequences are pre-built once per anchor (chunked `sequence.build_split`); the
  per-step engine call is then a gather.
- Everything downstream is **reused unchanged**: `assemble_matrix` / `_zero_impossible` /
  `roll_forward_predictor` (the ADR-001 seam) → `smm_paths.pool_smm_paths` / `paths_to_frame` →
  `economics.per_pool_econ` (`horizons=(1,)`). Only the injected predictor differs — the
  controlled-comparison guarantee.
- **Retrain, not reload.** The M27a GPU run saved only predictions (no weights), so each window's GRU
  is retrained with the frozen M27a config (hidden 48, Adam 1e-3 / wd 1e-5, ReduceLROnPlateau,
  early-stop patience 5, batch 4096, seed 0) and the `state_dict` is **persisted** to
  `outputs/m27b_gpu_runs/` (reruns + the identity check are now reproducible). Retrain reproduces
  the M27a val-NLL (e.g. k2015 **0.089469** vs M27a 0.0893; k2020 0.0954). GRU **ensemble** (5 key
  windows) = mean-of-softmax over seeds {0,1,2}, mirroring the FF mean-of-members rule.

## 2. H=1 identity check (built-in regression guard) — PASS, bit-exact, every window
The SeqPredictor's one-step output, **composed at H=1 through the engine**
(`roll_forward_predictor(horizon=1, zero_impossible=False)`, the raw path), equals the GRU's
`evaluate` one-step prediction on the same loans (`seq_train._val_probs`, the M27a metric path) —
asserted on a 20,000-loan anchor sample **before** pricing each window:

| anchor | k2015 | k2016 | k2017 | k2018 | k2019 | k2020 | k2021 | k2022 | k2023 | k2024 | k2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `max|Δ|` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

`max|Δ| = 0.00e+00` (≪ 1e-6) for all 11 — the sequence model flows through the **same** composition +
cashflow math as logit/NN/ensemble; the H=1 pool price is the GRU's per-row prediction, not a
re-implementation. This is the parameterized M13 Accept-#2 / `pool.verify_h1` stage (c), for the GRU.

## 3. GRU H=1 pool pricing — all 11 anchors (errors = GRU − realized, `pool.cashflow_engine`)
Char-bucket scheme; mean `|price error|` per 100 face (bias in parens), CPR error (pp), WAL error (mo).
Same pools / realized reference / engine as M24 — reproduced independently on the pod (the FF
checkpoints + M24 econ grid live on the SSD/Mac; the **pool structure + realized SMM are
model-independent** and rebuilt from `eval_pool`/`train_pool`).

| k | anchor | model | n_pop | char \|perr\| | (bias) | rand \|perr\| | char \|CPRe\| pp | char \|WALe\| mo |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 2015 | Dec2014 | gru     | 493,614 | 0.2456 | −0.226 | 0.3485 | 5.37 | 16.54 |
| 2015 | Dec2014 | gru_ens | 493,614 | 0.2590 | −0.243 | 0.3671 | 5.55 | 17.41 |
| 2016 | Dec2015 | gru     | 513,304 | 0.1018 | −0.020 | 0.2285 | 1.46 | 7.12 |
| 2017 | Dec2016 | gru     | 538,052 | 0.1150 | +0.022 | 0.2462 | 1.87 | 7.86 |
| 2018 | Dec2017 | gru     | 564,182 | 0.1649 | −0.157 | 0.2675 | 2.15 | 11.36 |
| 2019 | Dec2018 | gru     | 589,880 | 0.1667 | −0.132 | 0.2536 | 1.66 | 12.35 |
| 2019 | Dec2018 | gru_ens | 589,880 | 0.1969 | −0.179 | 0.2690 | 1.97 | 14.53 |
| 2020 | Dec2019 | gru     | 606,845 | **0.1495** | −0.122 | 0.2029 | 2.64 | 9.98 |
| 2020 | Dec2019 | gru_ens | 606,845 | 0.1529 | −0.127 | 0.2025 | 2.63 | 10.21 |
| 2021 | Dec2020 | gru     | 640,171 | 0.1849 | −0.079 | 0.0920 | 5.41 | 11.21 |
| 2022 | Dec2021 | gru     | 694,628 | 0.0991 | +0.009 | 0.2185 | 1.86 | 6.16 |
| 2023 | Dec2022 | gru     | 717,544 | 0.3225 | −0.322 | 0.3286 | 2.27 | 23.37 |
| 2023 | Dec2022 | gru_ens | 717,544 | 0.2673 | −0.267 | 0.3026 | 1.81 | 19.36 |
| 2024 | Dec2023 | gru     | 719,225 | 0.1002 | −0.030 | 0.2580 | 0.73 | 7.43 |
| 2025 | Dec2024 | gru     | 716,863 | 0.1547 | −0.101 | 0.2469 | 1.36 | 12.61 |
| 2025 | Dec2024 | gru_ens | 716,863 | 0.1663 | −0.109 | 0.2504 | 1.50 | 13.60 |

(Anchor = `Dec(k−1)`. Price bias is **mostly negative** = the GRU **over**-predicts month-1
prepayment → prices the premium pass-through slightly low. The two worst anchors — **k2023/Dec2022
0.3225** (bias −0.322) and **k2015/Dec2014 0.2456** — are rate-*transition* vintages: Dec2022 rolls
into the 2023 refi shutdown, and the frozen-t0 macro leaves the GRU (trained through 2022)
over-predicting prepay it cannot see collapse — the same frozen-macro failure mode `M24_NOTES` flags
for the FF models at COVID. The calmest anchors (k2016 0.102, k2022 0.099, k2024 0.100) price to
~0.1/100. Seed-ensembling barely moves H=1 — low seed variance at one step.)

## 4. GRU vs logit / NN / ensemble — the H=1 comparison (COVID anchor, on-pod FF numbers)
The only FF H=1 numbers on the pod are the committed, de-contaminated **k2020** (anchor Dec2019, the
COVID anchor) char-scheme mean `|price error|` (`M24_NOTES §3`):

| model (Dec2019, H=1, char) | mean \|price err\| /100 | vs logit |
|---|--:|--:|
| logit (M24)        | 0.1822 | — |
| **GRU single**     | **0.1495** | **−18%** |
| GRU ensemble (3 seeds) | 0.1529 | −16% |
| FF ensemble (M24, 8 nets) | 0.1030 | −43% |

**Read — competitive, not winning (and the gap is confounded).** At H=1 the GRU **beats the linear
logit** (0.1495 vs 0.1822, **−18%**) but **trails the full-data 8-net FF ensemble** (0.1495 vs
0.1030; the GRU is +45% over the ensemble). Two reasons not to over-read the ensemble gap, both
honest:
1. **The comparison is confounded.** The GRU is trained on a **1.5M** uniform sample with a **3-seed**
   average; the FF ensemble is **full-data** with **8 nets**. The H=1 price gap mixes architecture
   with data-budget and ensemble-width — it is **not** a clean architecture verdict.
2. **H=1 is the GRU's weakest horizon by construction.** The sequence model's value-of-memory edge
   (M27a: gru−ff_hist −0.001 pooled, concentrated at the COVID regime shift) lives in the
   **multi-month** trajectory — exactly the horizon a deterministic single-step composition cannot
   give it.

So the clean architecture claim stays the **loan-level** M27a result (the GRU matches/edges the
matched FF arms one-month-ahead, confound-free, same 1.5M/seed protocol); the H=1 **pricing** is
**competitive-not-winning** and chiefly **motivates the multi-month Monte-Carlo** pricing
(future work), where the sequence model's edge actually lives. We do **not** claim H=1 pricing parity
or advantage for the GRU.

> **Caveat — full FF side-by-side is a Mac merge.** The per-anchor FF (logit/best-NN/8-net-ensemble)
> H=1 grid lives in `<SSD>/outputs/tables/pool_level/m24/econ_k*.parquet` (not on the pod). The
> `econ_gru_k*.parquet` emitted here carry the **same schema** (`scheme/pool/model/h/price/cpr/wal/
> *_err/...`, model=`gru`/`gru_ens`, h=1), the **same pools** and the **same realized reference**, so
> dropping them next to the M24 econ artifacts extends T5.1 / F5.2 / the horizon-regime table with a
> GRU column directly — no recompute. **→ Merge done (this commit): the full per-anchor FF×GRU H=1
> grid + the authoritative read are in §7.**

## 5. Artifacts
`<SSD>/outputs/tables/pool_level/m27b/` — `econ_gru_k{2015..2025}.parquet` (single GRU, H=1) +
`econ_gru_ens_k{2015,2019,2020,2023,2025}.parquet` (seed-ensemble) + the matching
`smm_paths_*` frames + `m27b_summary.{parquet,json}`. GRU weights:
`<SSD>/outputs/m27b_gpu_runs/k{k}_s{seed}.pt`. Run log: `reports/m27b_run.log` (~126 min, one 4090).

## 6. Reproduce
```
python scripts/m27b/price_h1.py --device cuda                 # all 11 anchors + 5 ensembles
python scripts/m27b/price_h1.py --device cuda --anchors 2020  # one anchor
python scripts/m27b/price_h1.py --device cuda --anchors 2016 --max-loans 20000   # wiring smoke
```
Weights persist, so a rerun loads them and skips training (identity check + pricing only).

## 7. Mac merge — full 11-anchor FF × GRU H=1 comparison (this commit)
The M24 FF econ grid (`<SSD>/m24/econ_k*.parquet`) and the pod's GRU econ were merged on the Mac by
`m27b_merge.py`, which **reuses the M24 table builders**: the pure reductions
`pricing_grid._t51_agg` / `_horizon_regime_grid` and `economics.fig_price_error_buckets`, extracted
surgically and verified `frame_equal` against the committed M24 tables (304-row T5.1, 44-row
horizon-regime) — **M24 originals untouched**, 35 pytest green. A **pool-alignment gate** ran first:
for every (k, scheme, pool) at H=1 the pod's `econ_gru*` join to M24's `econ_k*` matches the
**model-independent** fields — `n_loans` **exact**; `wac/wam/upb/cpr_real/wal_real` bit-identical
(max|Δ|=0); `price_real` to 4e-14 (engine-recompute roundoff). Apples-to-apples confirmed.

Mean `|price error|` per 100 face, H=1, by anchor (lower = better; **bold** = best in row). ens×8 and
GRU×3 exist only at the 5 key windows.

**Characteristic buckets** (headline scheme)

| anchor | logit | best-NN | ens×8 | GRU | GRU×3 |
|---|--:|--:|--:|--:|--:|
| Dec2014 | **0.137** | 0.215 | 0.203 | 0.246 | 0.259 |
| Dec2015 | 0.200 | 0.158 | — | **0.102** | — |
| Dec2016 | 0.249 | 0.222 | — | **0.115** | — |
| Dec2017 | **0.112** | 0.184 | — | 0.165 | — |
| Dec2018 | 0.191 | **0.148** | 0.173 | 0.167 | 0.197 |
| Dec2019 (COVID) | 0.182 | **0.098** | 0.103 | 0.149 | 0.153 |
| Dec2020 | 0.378 | **0.126** | — | 0.185 | — |
| Dec2021 | 0.245 | 0.200 | — | **0.099** | — |
| Dec2022 | 0.259 | 0.170 | **0.143** | 0.323 | 0.267 |
| Dec2023 | 0.276 | 0.180 | — | **0.100** | — |
| Dec2024 | 0.252 | 0.155 | 0.161 | **0.155** | 0.166 |
| **Pooled (key-5 matched)** | 0.206 | **0.156** | **0.156** | 0.208 | 0.208 |
| _Pooled (all 11)_ | 0.228 | 0.168 | 0.156¹ | 0.164 | 0.208¹ |

**Random pools (500×1000)**

| anchor | logit | best-NN | ens×8 | GRU | GRU×3 |
|---|--:|--:|--:|--:|--:|
| Dec2014 | **0.256** | 0.298 | 0.292 | 0.349 | 0.367 |
| Dec2015 | 0.295 | 0.255 | — | **0.228** | — |
| Dec2016 | 0.262 | 0.288 | — | **0.246** | — |
| Dec2017 | **0.244** | 0.295 | — | 0.268 | — |
| Dec2018 | 0.274 | 0.256 | 0.267 | **0.254** | 0.269 |
| Dec2019 | 0.198 | **0.197** | 0.205 | 0.203 | 0.203 |
| Dec2020 | 0.267 | 0.095 | — | **0.092** | — |
| Dec2021 | 0.290 | 0.241 | — | **0.218** | — |
| Dec2022 | 0.312 | 0.219 | **0.212** | 0.329 | 0.303 |
| Dec2023 | 0.340 | 0.312 | — | **0.258** | — |
| Dec2024 | 0.261 | 0.250 | 0.254 | **0.247** | 0.250 |
| **Pooled (key-5 matched)** | 0.260 | **0.244** | 0.246 | 0.276 | 0.278 |
| _Pooled (all 11)_ | 0.273 | 0.246 | 0.246¹ | 0.245 | 0.278¹ |

¹ ens×8 and GRU×3 exist only at the 5 key windows, so their "all" column = their key-5 value.

**Read (authoritative; supersedes the single-anchor §4 lead).**
- **Primary = the key-5 matched set** (Dec2014/18/19/22/24 — the only windows where *every* model is
  scored). Char: GRU **0.208 ≈ logit 0.206** (a wash) and **trails best-NN 0.156 and the FF ensemble
  0.156** (≈+33%). On the matched set the GRU is competitive with the linear baseline but **does not**
  reach the FF nonlinear models at H=1.
- **The all-11 "GRU beats logit ~28% (0.228→0.164)" is real but window-selection-sensitive** —
  present as *context, not headline*. The 6 non-key windows happen to favour the GRU, and the
  **GRU-vs-NN ranking flips** between sets (all-11: GRU 0.164 < NN 0.168; key-5: GRU 0.208 ≫ NN
  0.156). Lead with the matched set.
- **GRU is high-variance across regimes.** Calm windows price to ~0.10 (Dec2015/21/23); the two
  rate-*transition* vintages blow up — **Dec2022 0.323**, **Dec2014 0.246** — the frozen-t0-macro
  failure mode `M24_NOTES` flags for the FF models at COVID.
- **H=1 error is bias-dominated** — a negative price bias (over-predicts 1-month prepay) at **9/11**
  anchors — so **seed-bagging barely helps** (GRU×3 ≈ GRU, often a touch worse): averaging seeds
  cannot remove a shared directional bias.
- **Headline: competitive-not-dominant at H=1; the memory edge is multi-month.** The GRU clears the
  linear baseline broadly and matches it on the matched set, but trails the FF nonlinear models
  one-month-ahead — its hypothesised value (memory of the delinquency path) can only express over a
  multi-month roll. **H{3,6,12} are left blank (future-work Monte-Carlo).** This dovetails with M24's
  own finding that the **value of nonlinearity is a longer-horizon phenomenon** (several H=1 cells
  favour the simpler model).

Merged artifacts (SSD, **gitignored**): `<SSD>/outputs/tables/pool_level/m27b/`
`t_m27b_econ_errors.{md,json,parquet}` (T5.1 + GRU), `t_m27b_horizon_regime.{md,json}`,
`t_m27b_h1_compare.{md,parquet}`, and `<SSD>/outputs/figures/pool_level/F5.2_*_m27b_h01.{png,pdf}`
(×11 anchors). Built by `m27b_merge.py` (`python -m floan.model.m27b_merge`).
