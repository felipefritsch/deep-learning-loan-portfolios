# Deep-Learning Models of Loan Portfolios and Asset-Backed Securities

Research project and dissertation for the University of Oxford MSc in Mathematical
and Computational Finance.

## What this project is

Residential and commercial loans are one of the largest asset classes in the world (\$20T+).
A loan's value — and the value of any asset-backed security written on a pool of loans — is
driven by how its borrowers move month to month between **paying, delinquency, default, and
prepayment**. This project models that movement directly and asks how much modern machine
learning buys you over classical credit models, both statistically and in the price an investor
actually pays.

Concretely, it estimates a **seven-state monthly transition model** — the conditional
distribution of next month's state given the loan's current state and covariates — on the
**Fannie Mae Single-Family Loan Performance** dataset (~800 GB, ~100 quarterly vintages,
2000–2025; 57.6M loans, 3.3B loan-month observations). The state machine, following
Sadhwani, Giesecke & Sirignano (2021):

```
current → 30 / 60 / 90+ days delinquent → foreclosure → REO        (+ prepaid)
```

Everything is estimated **strictly out of sample** under an expanding-window rolling backtest,
one window per test year **2015–2025**, so regime shifts (the 2020 COVID forbearance episode,
the 2022–23 rate shock) appear as legible test-year effects rather than hinging on one split.

## The three questions, and the model ladder

The estimators are arranged as a nested **ladder of flexibility** — each rung relaxes one
restriction on how next-month state depends on covariates, so the gap between two adjacent rungs
*attributes* predictive power to a specific freedom rather than being a leaderboard position:

| Rung | Model | What it adds |
|---|---|---|
| floor | **empirical transition matrix** | covariate-free base rates |
| | **multinomial logit** | linear, additive covariate effects |
| | **spline-logit** | univariate nonlinear shapes (no interactions) |
| | **gradient-boosted trees** | arbitrary nonlinearity + interactions, piecewise-constant |
| headline | **deep neural network** | the same, on a smooth surface (learned interactions) |
| | **ensemble** | variance reduction over independently trained nets |

Three research questions sit on top:

1. **Is the dependence nonlinear and interactive?** (the central claim of the reference paper) —
   measured by the gap from logit to the flexible learners, decomposed into shapes vs interactions
   vs smoothness.
2. **Does the path matter — is the first-order Markov assumption right?** Tested by adding
   engineered borrower-history features, then by *learned* memory: a **GRU** and a **transformer**
   read each loan's trailing-12-month sequence. (The reference paper hand-engineers history; the
   learned sequence models are the genuine extension here.)
3. **What is the economic value?** The fitted monthly models are composed across horizons (1–12
   months), carried to the **pool** level, and run through a pass-through cashflow engine, so the
   comparison is restated in errors of prepayment speed (CPR), weighted-average life (WAL), and
   **price**. Sequence models, being path-dependent, are priced by **Monte-Carlo path simulation**.

## Repository layout

The code is an installable package (**`floan`** = "f" + *loan*) under a `src/` layout; modules
import each other by package path. Each code/results directory carries its own README with
per-file detail.

```
.
├── pyproject.toml              # package metadata, dependencies, pytest config
├── src/floan/
│   ├── pipeline/               # data pipeline: inventory → parquet → clean → panel → sample → QA   (README)
│   ├── analysis/               # Phase-1 exploratory tables (T*) and figures (F*)                   (README)
│   └── model/                  # Phases 2–3: the model ladder, sequence models, pools, pricing      (README)
├── tests/                      # hermetic unit suite (synthetic data; no SSD/GPU/network)
├── scripts/                    # runnable drivers (backtest, GBT sweep, sequence-model pipeline)     (README)
├── specs/{pipeline,model}/     # the executable specifications (schema, stages, sequenced tasks, ADRs)
├── writeup/
│   ├── latex/                  # the dissertation text (chapters, main.tex)
│   └── memos/                  # interim result memos behind each chapter
├── docs/                       # source material: dataset glossary/tutorial PDFs, the reference paper
└── reports/                    # disposable DuckDB catalog (gitignored)
```

> A note on naming: some of the codebase was built task-by-task against `specs/model/04_TASKS.md`,
> so a few comments reference task codes (e.g. "M8", "M27a"). The directory READMEs translate every
> file into **what it does**, not which task built it.

## Installation

Requires Python ≥ 3.9. `pip install -e .` installs `floan` and its dependencies (DuckDB, Polars,
PyArrow, NumPy, pandas, matplotlib, scikit-learn, PyTorch, LightGBM). On a CPU-only machine install
the CPU torch wheel first so the `torch==2.8.0` pin doesn't pull CUDA:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Data

The raw dataset is **not** in this repository. It lives on an external SSD rooted at
`/Volumes/SSD Felipe/dissertation/`, in an immutable-raw layout `raw/ → interim/ → processed/`
(plus `models/`, `outputs/`, `logs/`). All paths derive from a single `ROOT` in
`src/floan/pipeline/config.py`; every data/model command calls `require_drive()` and fails fast
if the drive is not mounted. `raw/` is never written. The unit suite needs none of this.

## Usage

Always run modules with `python -m` so package imports resolve.

```bash
# Data pipeline (idempotent, per-quarter, memory-bounded) — see specs/pipeline/HOWTO_RUN.md
python -m floan.pipeline.run inventory      # catalogue + integrity gate (run first)
python -m floan.pipeline.run pipeline       # convert → clean → panel, end-to-end per quarter
python -m floan.pipeline.run sample --cutoff-ym 201501   # balanced sample + train-only scaler

# Exploratory analysis (one entry point per table/figure)
python -m floan.analysis.t2_1_transition_matrix          # pooled empirical transition matrix
python -m floan.analysis.f3_2_prepay_incentive           # the refinancing S-curve

# Modelling: the ladder and the rolling backtest
python -m floan.model.benchmarks            # empirical transition-matrix floor, all windows
python -m floan.model.train                 # train the deep net on the tuning window
python -m floan.model.gbt                   # gradient-boosted-tree baseline
python -m floan.model.backtest              # roll the frozen config across all 11 windows
python -m floan.model.evaluate              # per-window NLL / AUC / calibration tables
```

The sequence-model (GRU / transformer) training and pricing pipeline lives in `scripts/` (it runs
on a GPU and needs the per-window sequence caches built first) — see `scripts/README.md`.

## Testing

The unit suite is **hermetic** — synthetic data only, no SSD/GPU/network:

```bash
python -m pytest
```

GitHub Actions runs it on every push to `main` and every PR. PyTorch-dependent tests skip cleanly
if `torch` is absent.

## Documentation

- **`specs/pipeline/`, `specs/model/`** — the executable specifications (schema, stages, sequenced
  tasks with acceptance criteria, the economic-engine spec, and the architecture decision records
  `ADR-001`/`ADR-002`).
- **`writeup/`** — the dissertation text and the result memos behind each chapter.
- **`docs/`** — dataset glossary/tutorial PDFs and the reference paper.
- **Directory READMEs** — per-file detail for `pipeline/`, `analysis/`, `model/`, and `scripts/`.
