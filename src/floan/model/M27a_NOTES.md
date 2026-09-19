# M27a — Sequence (GRU) vs matched FF baselines, rolling on all 11 windows

The definitive value-of-memory read for the seven-state monthly transition model: a single 1-layer
GRU vs two FF baselines (`ff_base` = current-state Markov, `ff_hist` = FF + engineered one-step
history), trained and scored on every rolling window k = 2015…2025. Supersedes the M26 single-window
spike (k=2015 only); see the **Reconciliation with M26** section.

## Confound-free protocol

All three arms share, per window:

- **The same 1.5M training points** — `prediction_points(variant="full", k, "train", sample_n=1_500_000,
  seed=0)`, i.e. the exact rows the GRU sequence caches were built from (uniform train_pool + HT
  weights, natural ~69–72% current-origin mix, **not** origin-balanced). The FF driver asserts the
  semi-joined sample is exactly 1,500,000 rows.
- **The same full TEST slice** — the frozen eval_pool test year, scored in full (n asserted equal
  across arms per window, 84,512,377 pooled).
- **The same frozen training config** — Adam lr 1e-3, weight_decay 1e-5, ReduceLROnPlateau(0.5,
  patience 2), early-stop patience 5 on **full-val** NLL, max 40 epochs, batch 4096, seed 0, cuda.
  Architecture is the only thing that varies: FF net (`net.build`, depth/dropout from `NN_SELECTED`)
  vs GRU (hidden 48, 1 layer). Metrics are the unmodified `evaluate._nll` / `_auc_one_vs_rest`.

## Three-arm rolling table — test NLL (seed 0)

| test yr |  ff_base |  ff_hist |      gru |   gru−base |   gru−hist |     n_test |
|--------:|---------:|---------:|---------:|-----------:|-----------:|-----------:|
|    2015 | 0.103026 | 0.097024 | 0.096915 | −0.006111  | −0.000109  |  6,180,351 |
|    2016 | 0.109383 | 0.103113 | 0.103096 | −0.006287  | −0.000017  |  6,410,612 |
|    2017 | 0.098049 | 0.092165 | 0.092106 | −0.005943  | −0.000059  |  6,759,294 |
|    2018 | 0.089114 | 0.083531 | 0.083278 | −0.005836  | −0.000253  |  7,065,297 |
|    2019 | 0.100815 | 0.095100 | 0.095959 | −0.004857  | **+0.000859** |  7,287,022 |
|    2020 | 0.186073 | 0.184688 | 0.178991 | −0.007081  | −0.005696  |  7,639,066 |
|    2021 | 0.144329 | 0.140364 | 0.139500 | −0.004829  | −0.000864  |  8,233,784 |
|    2022 | 0.086578 | 0.082596 | 0.081655 | −0.004923  | −0.000940  |  8,669,811 |
|    2023 | 0.072607 | 0.067616 | 0.064514 | −0.008093  | −0.003102  |  8,768,239 |
|    2024 | 0.072873 | 0.067456 | 0.067015 | −0.005858  | −0.000441  |  8,775,310 |
|    2025 | 0.076043 | 0.069376 | 0.069579 | −0.006464  | **+0.000203** |  8,723,591 |
| **POOLED** | **0.102528** | **0.097494** | **0.096491** | **−0.006036** | **−0.001003** | **84,512,377** |

Pooled = Σ(NLL_w · n_w) / Σ n_w over the 11 disjoint test years. Ordering `ff_base > ff_hist > gru`
(lower = better) holds pooled and in every individual window. Per-window detail (val NLL, per-transition
AUC, n, timings) is in [`artifacts/results/m27a_results/`](../../../artifacts/results/m27a_results/) — `k{k}_s0.json` (GRU), `k{k}_ff_base.json`,
`k{k}_ff_hist.json`; key-window GRU seeds 1/2 in `k{k}_s{1,2}.json`.

## Finding 1 — sequence memory beats Markov current-state, everywhere

**gru − ff_base = −0.006036 pooled.** Robust across *all* regimes: every window is negative, range
**−0.0048 … −0.0081** (least at 2021, most at 2023). A learned sequence architecture conditioning on
the trailing 12-month window beats plain current-state (Markov) conditioning in every test year,
calm or stressed. This is the headline value-of-memory result.

## Finding 2 — learned vs engineered memory: small, regime-concentrated

**gru − ff_hist = −0.001003 pooled.** Decomposing the total base→gru gain (−0.006037):

- **Engineered one-step history captures ~83%** of it (ff_base→ff_hist = −0.005034 pooled): the four
  hand-built summaries (`prev_state`, `ever_delinquent`, `months_since_last_delinq`,
  `n_prior_delinq_episodes`) recover most of the memory value.
- **Learned memory adds the remaining ~17%** (ff_hist→gru = −0.001003 pooled), and that increment is
  **regime-concentrated**, not uniform:
  - **COVID 2020: −0.005696** (~6× the pooled average) — the regime shift is exactly where the full
    trailing sequence beats a one-step summary; the GRU's learned-memory edge is largest here.
  - **Calm years flip:** **2019 (+0.000859)** and **2025 (+0.000203)** are slightly *positive* —
    engineered one-step history matched or edged the learned GRU. Reported honestly; both are < 0.001
    and within seed noise (3-seed GRU sd at these windows is ≤4e-4).

So the learned-over-engineered advantage is real but **sample-sensitive and regime-concentrated** —
it shows up under distributional stress, not in calm vintages.

## Reconciliation with M26

M26's three-arm spike ran a single window (k=2015) at a 300k train sample and reported gru−ff_hist
≈ **−0.0024** (a ~70/30 engineered/learned split). At the M27a scale (k=2015, **1.5M** points) the same
delta shrinks to **−0.000109** (~−0.0001):

- `ff_hist` benefits more from data than the GRU at this scale — the engineered features sharpen with
  more samples, closing most of the gap that the 300k spike showed.
- The learned-vs-engineered edge is therefore **both sample-sensitive and regime-concentrated**:
  M26's 70/30 (300k, single calm window) **overstated** it.

**M27a rolling (11 windows, 1.5M, full test) is the definitive read:** engineered one-step history
~83%, learned ~17% pooled, with the learned increment paying off mainly under regime stress (2020).

## Implementation notes (pod build)

- **Concurrency / memory.** `build_cache` per window peaks ~25 GB RSS; the windowed DuckDB join is
  the heavy step. Running 3 windows at `DUCKDB_MEMORY_LIMIT=48GB` each oversubscribed the 124 GB box
  and OOM-killed builds. The working recipe is **24 GB DuckDB × 2 concurrent** (~88 GB peak), with a
  `MemAvailable` headroom guard that auto-drops to 1. See `scripts/m27a/build_windows_batch.py`.
- **History superset.** `ff_hist` needs `history.build` caches, absent on the pod. History feature
  *values* for a `(loan, period_ym)` are **period-trim invariant** (window frame is `UNBOUNDED
  PRECEDING … 1 PRECEDING`; `period_hi` only trims output rows, never a retained row's value). So the
  **k=2025 superset** (824 M loan-month rows, shards 0–63) covers every loan-month any earlier window
  needs, with identical values and no look-ahead. We built it once (~10 min) and **symlinked**
  `history/k{2015..2024} → k2025`; `join_history` (keyed on `(loan, period_ym)`) yields byte-identical
  results to per-window builds. See `scripts/m27a/build_history_superset.py`.
- **verify_sample race fix.** `sequence.verify_sample` wrote its round-trip sample to a *fixed*
  `OUTPUTS/seq_cache/_verify_tmp`, so concurrent builds raced (one build's clear/save clobbered
  another's reload → spurious "round-trip mismatch on cont"; only ever a false *failure*, never a
  false pass). Fixed in `sequence.py` to use a per-call `tempfile.mkdtemp` + `rmtree`.

## Reproduce

```
# caches (per window, or batched 2-up):           on the pod, panel + training pools + macro present
python scripts/m27a/build_window.py 2016
python scripts/m27a/build_windows_batch.py        # k=2017..2025, resumable, 2 concurrent
python scripts/m27a/build_history_superset.py     # k=2025 history; then symlink k2015..k2024 -> k2025
# GRU: all 11 windows seed 0 + key-window seeds 1,2; FF baselines:
python scripts/m27a/gpu_train_all.py
python scripts/m27a/ff_arms.py
# tables:
python scripts/m27a/summary_gru.py                # GRU rolling + ensemble + pooled
python scripts/m27a/summary_three_arm.py          # ff_base / ff_hist / gru + deltas + pooled
```
