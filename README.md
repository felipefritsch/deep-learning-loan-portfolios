# Deep-Learning Models of Loan Portfolios and Asset-Backed Securities

Research project and dissertation for the University of Oxford MSc in Mathematical
and Computational Finance.

The commercial and residential loan market is one of the largest asset classes, with
over \$20 trillion in value. Banks hold large loan portfolios and investment funds trade
asset-backed securities (ABS) whose cashflows are a function of how many underlying loans
are **current, delinquent, prepaid, or defaulted**. Each loan carries a high-dimensional
feature set that can be used to model its likelihood of becoming delinquent or defaulting.

This project develops and trains a deep-learning model to predict mortgage **delinquency
and prepayment**, benchmarked against simpler statistical models (empirical transition
matrices and logistic regression) and a gradient-boosted-tree baseline. Models are trained
and evaluated on the **Fannie Mae Single-Family Loan Performance** dataset (~800 GB, ~100
acquisition vintages, 2000–2025; 30M+ loans, billions of monthly observations).

The core model is a **Sirignano-style seven-state monthly transition model** over the states

```
current → dpd_30 → dpd_60 → dpd_90plus → foreclosure → REO        (+ prepaid)
```

estimated per rolling backtest window, with cashflow/valuation error measured at the loan
pool level.

## Repository layout

The code is an installable Python package (`floan`) under a `src/` layout; modules import
each other by package path (no `sys.path` manipulation).

```
.
├── pyproject.toml          # package metadata, dependencies, pytest config
├── src/floan/
│   ├── pipeline/           # data pipeline: inventory → parquet → clean → panel → sample → QA
│   │   ├── config.py       # single ROOT / REPO_ROOT anchor; require_drive()
│   │   ├── schema.py       # 113-field positional schema + seven-state target derivation
│   │   ├── run.py          # stage dispatch entry point
│   │   └── s1_inventory … s6_qa
│   ├── analysis/           # Phase-1 EDA tables (T*) and figures (F*) + shared helpers
│   └── model/              # Phases 2–3: benchmarks, logit, nets, GBT, pools, economics, backtest
├── tests/                  # hermetic unit suite (synthetic data; no SSD/GPU/network)
├── scripts/                # shell helpers (backtest sweep, depth check, SSD backup)
├── specs/
│   ├── pipeline/           # data-pipeline specification (00_OVERVIEW … HOWTO_RUN)
│   └── model/              # analysis & modelling specification (00_OVERVIEW … 06_GBT_BASELINE)
├── writeup/                # dissertation text (latex/) and interim result memos (memos/)
├── docs/                   # source material: dataset glossary/tutorial PDFs, reference paper
└── reports/                # disposable DuckDB catalog (lake.duckdb) — gitignored
```

## Installation

Requires Python ≥ 3.9.

```bash
pip install -e .
```

This installs the `floan` package and its dependencies (DuckDB, Polars, PyArrow, NumPy,
pandas, matplotlib, scikit-learn, PyTorch, LightGBM). The pinned `torch==2.8.0` resolves to
a CUDA build on the default index; on a CPU-only machine install the CPU wheel first so the
pin is satisfied without pulling CUDA:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Data

The raw dataset is **not** in this repository. It lives on an external SSD, rooted at
`/Volumes/SSD Felipe/dissertation/`, in the standard immutable-raw layout
`raw/ → interim/ → processed/` (plus `models/`, `outputs/`, `logs/`). All paths derive from a
single `ROOT` in `src/floan/pipeline/config.py`; every stage calls `require_drive()` and fails
fast if the drive is not mounted. `raw/` is treated as immutable — no code writes there. Only
the disposable DuckDB catalog `reports/lake.duckdb` lives on the internal disk.

Pipeline and model commands therefore require the mounted SSD and the dataset; the unit suite
(below) does not.

## Usage

Always run modules with `python -m` (not by file path), so package imports resolve.

**Pipeline** — the stages are idempotent and per-quarter; data is streamed via
DuckDB/Polars/PyArrow (a full quarter is never loaded into memory):

```bash
python -m floan.pipeline.run inventory   # Stage 1: vintage inventory + QA gate (run first)
python -m floan.pipeline.run convert     # Stage 2: CSV → projected/typed Parquet
python -m floan.pipeline.run clean       # Stage 3: standardisation transform
python -m floan.pipeline.run panel       # Stage 4: seven-state transition panel
python -m floan.pipeline.run sample      # Stage 5: balanced sample + train-only scaler
python -m floan.pipeline.run qa          # Stage 6: reconciliation / QA report
python -m floan.pipeline.run pipeline    # Stages 2→3→4 end-to-end, per quarter
```

See `specs/pipeline/HOWTO_RUN.md` for the operator guide.

**Analysis (Phase 1 EDA)** — each table/figure script is its own entry point, e.g.:

```bash
python -m floan.analysis.t1_1_coverage
python -m floan.analysis.t2_1_transition_matrix
python -m floan.analysis.f2_2_transition_rates
```

**Modelling (Phases 2–3)** — benchmarks, models, and the rolling backtest, e.g.:

```bash
python -m floan.model.benchmarks         # M5 empirical transition-matrix benchmark
python -m floan.model.train              # train a single net on the tuning window
python -m floan.model.gbt                # M16 LightGBM GBT baseline
python -m floan.model.backtest           # rolling-window backtest across all windows
```

Convenience shell wrappers live in `scripts/` (e.g. `scripts/run_backtest.sh`,
`scripts/run_gbt_sweep.sh`); they compute the repo root automatically and invoke the package.

## Testing

The unit suite is **hermetic** — synthetic data only, no SSD, no GPU, no network — so it runs
anywhere:

```bash
python -m pytest
```

It is configured in `pyproject.toml` (`testpaths = ["tests"]`). A few model tests that import
PyTorch skip cleanly if `torch` is not installed. GitHub Actions runs the suite on every push
to `main` and every pull request (`.github/workflows/tests.yml`).

## Documentation

- **`specs/pipeline/`** and **`specs/model/`** — the executable specifications (schema, stages,
  tasks with acceptance criteria, macro-data spec).
- **`writeup/`** — the dissertation text (`latex/`) and interim result memos (`memos/`).
- **`docs/`** — dataset glossary/tutorial PDFs and the reference paper.
- **`CLAUDE.md`** files — working guidelines and the binding repo map for each area.
