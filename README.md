# Deep-Learning Models of Loan Portfolios and Asset-Backed Securities

Research project and dissertation for the University of Oxford MSc in Mathematical
and Computational Finance.

[![tests](https://github.com/felipefritsch/deep-learning-loan-portfolios/actions/workflows/tests.yml/badge.svg)](https://github.com/felipefritsch/deep-learning-loan-portfolios/actions/workflows/tests.yml)

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

## Key findings

- On 84.5 million held-out loan-months, the deep transition model reduces pooled negative
  log-likelihood by **6.2% versus multinomial logit** and **9.0% versus an empirical transition
  matrix**. It beats both baselines in every test year and origin state.
- Most of the discrimination gain comes from nonlinear prepayment and delinquency effects:
  Current→Prepaid AUC rises from **0.65 to 0.75**, while 90+ DPD→Foreclosure rises from
  **0.48 to 0.62**.
- The first-order Markov assumption is rejected. Four engineered history summaries capture about
  **88% of the total memory gain**; a GRU captures the remainder, concentrated in stress regimes.
- At the pool level, model value is regime- and horizon-dependent. The ensemble cuts the logit's
  mean absolute price error by **31%–48%** in the 2022–24 rate-shock windows, but slightly
  underperforms it at the COVID anchor when frozen macro covariates become stale.

These are research results, not production performance claims. The full methodology, limitations,
tables, and figures are in the [submitted dissertation](artifacts/1101489_dissertation.pdf).

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
├── AGENTS.md                   # repository-wide instructions and research invariants for Codex
├── CODEX_CASE_STUDY.md         # one real agent-assisted optimization, reconstructed and evidenced
├── LICENSE                     # source-available, non-commercial research-use terms
├── src/floan/
│   ├── pipeline/               # data pipeline: inventory → parquet → clean → panel → sample → QA
│   ├── analysis/               # Phase-1 exploratory tables (T*) and figures (F*)                   (README)
│   └── model/                  # Phases 2–3: the model ladder, sequence models, pools, pricing      (README)
├── tests/                      # hermetic unit suite (synthetic data; no SSD/GPU/network)
├── scripts/                    # runnable drivers (backtest, GBT sweep, sequence-model pipeline)     (README)
├── specs/{pipeline,model}/     # executable specifications, acceptance tests, and ADRs
├── artifacts/                  # final dissertation and compact result artifacts (Git LFS)
└── reports/                    # disposable DuckDB catalog (gitignored)
```

> A note on naming: some of the codebase was built task-by-task against `specs/model/04_TASKS.md`,
> so a few comments reference task codes (e.g. "M8", "M27a"). The directory READMEs translate every
> file into **what it does**, not which task built it.

## Quick start

Requires Python ≥ 3.9. `pip install -e .` installs `floan` and its dependencies (DuckDB, Polars,
PyArrow, NumPy, pandas, matplotlib, scikit-learn, PyTorch, LightGBM). On a CPU-only machine install
the CPU torch wheel first so the `torch==2.8.0` pin doesn't pull CUDA:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
python -m pytest
```

## Data

The raw dataset is **not** in this repository. It lives on an external SSD rooted at
`/Volumes/SSD Felipe/dissertation/`, in an immutable-raw layout `raw/ → interim/ → processed/`
(plus `models/`, `outputs/`, `logs/`). All paths derive from a single `ROOT` in
`src/floan/pipeline/config.py`; every data/model command calls `require_drive()` and fails fast
if the drive is not mounted. `raw/` is never written. The unit suite needs none of this.

Primary external references:

- [Fannie Mae Single-Family Loan Performance Data](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data)
- [Fannie Mae glossary and file layout](https://capitalmarkets.fanniemae.com/sites/g/files/koqyhd216/files/2024-07/crt-file-layout-and-glossary.pdf)
- [Sadhwani, Giesecke & Sirignano (2021), *Deep Learning for Mortgage Risk*](https://doi.org/10.1093/jjfinec/nbaa025)

## Usage

Always run modules with `python -m` so package imports resolve.

```bash
# Data pipeline (idempotent, per-quarter, memory-bounded) — see specs/pipeline/CODEX_WORKFLOW.md
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

## Codex and Claude-assisted development

I used Codex extensively to implement and review this research codebase, as well as Claude Code in a smaller capacity. The workflow was
specification-driven: I defined the statistical design, data invariants, and acceptance criteria;
Codex worked on bounded tasks; and I reviewed the diffs and validated them with synthetic tests,
reconciliation checks, frozen out-of-sample keys, and model/run manifests. The repository is set up
for the same workflow today:

- [`AGENTS.md`](AGENTS.md) and [`src/floan/pipeline/AGENTS.md`](src/floan/pipeline/AGENTS.md) encode
  the memory, leakage, data-integrity, and verification constraints Codex must follow.
- [`specs/pipeline/03_CODEX_TASKS.md`](specs/pipeline/03_CODEX_TASKS.md) and
  [`specs/model/04_TASKS.md`](specs/model/04_TASKS.md) turn the research plan into small tasks with
  explicit acceptance criteria.
- [`specs/pipeline/CODEX_WORKFLOW.md`](specs/pipeline/CODEX_WORKFLOW.md) documents the operating
  loop. Tests remain synthetic and hermetic; full-data checks run against the external data lake.

This framing is intentional: Codex accelerated implementation and verification, while the research
questions, modelling decisions, interpretation, and final dissertation remain my responsibility.
See the [Codex case study](CODEX_CASE_STUDY.md) for a concrete, evidenced example.

## Planned extension

A small synthetic end-to-end demo is a **TBD deliverable**. It will exercise ingestion, state
construction, temporal splitting, one baseline fit, and evaluation without requiring the external
800 GB dataset. It is intentionally documented but not implemented in the current repository.

## Documentation

- **`specs/pipeline/`, `specs/model/`** — the executable specifications (schema, stages, sequenced
  tasks with acceptance criteria, the economic-engine spec, and the architecture decision records
  `ADR-001`/`ADR-002`).
- **`artifacts/`** — the final dissertation and compact results; large binaries use Git LFS.
- **Directory READMEs** — per-file detail for `pipeline/`, `analysis/`, `model/`, and `scripts/`.

Third-party papers, Fannie Mae manuals, university guidance, meeting notes, briefs, and private
working material are intentionally not distributed in this repository.

## License

Copyright © 2026 Felipe Fritsch. The repository is source-available for personal review,
educational study, recruiting evaluation, and non-commercial academic research. Redistribution,
commercial use, production deployment, and published derivative works require prior written
permission. See [LICENSE](LICENSE).
