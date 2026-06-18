# Memo — Test-coverage review and a prioritized improvement backlog

*A cross-cutting review of the automated tests across `dev/pipeline/`, `dev/analysis/`,
and `dev/model/` (not tied to a single milestone). Catalogues what is tested today,
where the highest-risk gaps are, and proposes a ranked backlog of tests plus the
missing test infrastructure. Companion to the milestone `*_NOTES.md` files, which
carry per-task **Accept** evidence; this memo is about the standing unit suite.*

---

## 0. Status — implemented (2026-06-18)

The §4 backlog and the §5 infrastructure below have since been **implemented**; §1–§7
are retained as the original review that motivated the work. Current state:

- **Suite: 45 → 94 hermetic tests**, run as one command (`python -m pytest`, green in
  ~3 s), order-independent. New files, in the existing hand-rolled-runner style:
  - *P0* — `dev/model/test_config.py`, `dev/pipeline/test_s4_panel.py`,
    `dev/pipeline/test_s5_sample.py`
  - *P1* — `dev/pipeline/test_s3_clean.py`, `dev/model/test_export.py`,
    `dev/pipeline/test_s1_inventory.py`
  - *P2* — `dev/analysis/test_eda_common.py`, `dev/model/test_benchmarks.py`,
    `dev/analysis/test_mkt_rate.py`
- **Infra:** root `pyproject.toml` (`[tool.pytest.ini_options] testpaths=["dev"]`) for
  one-command discovery, and `.github/workflows/tests.yml` running the suite on pull
  requests and pushes to `main` (CPU-only PyTorch wheel + `dev/pipeline/requirements.txt`).
- **Deviations from the original proposal:**
  - Torch is handled by a per-file module-level skip guard (`try: import torch / except:
    pytest.skip(..., allow_module_level=True)`) on the three files whose modules import
    torch at load (`test_pool`, `test_pools`, `test_economics`), **not** a
    `requires_torch` marker — a marker cannot prevent a collection-time import error. CI
    installs the CPU torch wheel so those run rather than skip.
  - A small `sys.modules` shim at the top of each pipeline/analysis/model test file
    resolves the `dev/model` vs `dev/pipeline` duplicate `config.py` name clash within a
    single pytest session (verified across collection orders).
- **Still open (deferred, non-blocking):** the `pytest-cov` floor (§5.2), a shared
  `conftest.py` for the synthetic-frame builders (§5.4), and folding
  `verify_logit_equiv.py`/`verify_m9.py` into discoverable `test_*` (§4 item 10).

## 1. Summary

The suite covers the **numerically subtle cores** of the modelling stack very well —
matrix composition, the cashflow engine, the feature scaler/vocab, the macro lag
convention, and pool aggregation are all verified hermetically against closed forms
or brute force. But coverage is **narrow**: of the pipeline's six stages only
`schema.py` is tested, and several modelling/analysis modules with real branching
logic (`config.py`'s window bounds, `s4_panel`'s `state_next`/`censored` derivation,
`s5_sample`'s scaler round-trip) have **no tests at all**. There is also **no test
infrastructure** — no `pyproject.toml`/`pytest.ini`, no `conftest.py`, no
`pytest-cov`, and no CI workflow — so the suite is run by hand, file by file, and
nothing guards against regressions on push. *(This describes the state at review time;
§0 records what has since been implemented.)*

The good news: every existing test is **hermetic** (synthetic data, no SSD, no GPU,
no network), so the suite is cheap to run and CI is entirely feasible. The
recommendations below extend that same style to the untested logic rather than
introducing heavyweight integration machinery.

## 2. Inventory of current tests

All six test files share one hand-rolled runner (`if __name__ == "__main__": …` that
calls every `test_*` function and prints `ok`), and are also plain pytest-discoverable.
Verification style is uniformly **closed-form or brute-force ground truth** on
synthetic frames — a strong pattern worth preserving.

| Module under test | Test file | ~Tests | What's covered |
|---|---|---|---|
| `dev/pipeline/schema.py` | `dev/pipeline/test_schema.py` | 8 | 113-col layout, `parse_mmyyyy`, `scrub_credit_score`, `fico_orig_expr` coalescing, seven-state `derive_state` |
| `dev/model/features.py` + `data.py` | `dev/model/test_features.py` | 9 | `Scaler` (train-only, log1p/robust, zero-scale guard, null→0), `Vocab` (UNK/min-count prune), one-hot drop-first, `WindowLoader` (every row once, weights, reproducibility) |
| `dev/model/macro_features.py` | `dev/model/test_macro_features.py` | 6 | `sub_months` arithmetic, per-variable lag convention, `ltv_mtm`/`incentive`/`hpi_chg_12m` exact formulas, territory→national fallback |
| `dev/model/pool.py` | `dev/model/test_pool.py` | 11 | `compose` vs `p0·P^H`, matrix structure/mass, first-passage vs brute force, cashflow-engine closed forms (par identity, annuity, constant-SMM survival, price monotonicity) |
| `dev/model/pools.py` | `dev/model/test_pools.py` | 8 | char-cell partition/thin-merge, realized reconciliation, random-pool reproducibility/no-replacement, R²/RMSE/coverage |
| `dev/model/economics.py` | `dev/model/test_economics.py` | 3 | zero-error identity, §5.3 premium sign convention, T5.1 aggregation arithmetic |

**Total: ~45 hermetic unit tests across 6 files**, covering 6 of the project's modules.

There are also two **validation scripts** (not unit tests): `dev/model/verify_logit_equiv.py`
(PyTorch logit vs sklearn) and `dev/model/verify_m9.py`. These run against real
artifacts and assert numeric agreement; they are useful but are not part of a
discoverable, hermetic suite.

## 3. Gap analysis

### 3.1 Pipeline (Stages 1–6) — only `schema.py` is tested

`s1_inventory.py`, `s2_to_parquet.py`, `s3_clean.py`, `s4_panel.py`, `s5_sample.py`,
and `s6_qa.py` have **zero tests**, yet each contains pure, deterministic logic that
is easy to exercise on small synthetic frames and is the kind of thing that breaks
silently:

- **`s4_panel._finalize`** (`dev/pipeline/s4_panel.py:124`) derives `state_next` only
  when the next row is the **same loan** (`_next_loan == "Loan Identifier"`) and marks
  `censored` only for a known, **non-absorbing** state with no observed next month.
  This is the seven-state target's heart and is governed by a cross-batch carry-over
  in `build_quarter`; an off-by-one at the batch boundary would silently mislabel
  transitions. Untested.
- **`s4_panel`'s `_STATE_SQL`** (`:57`) is a hand-maintained SQL `CASE` that must stay
  bit-for-bit equivalent to `schema.derive_state` (the Polars version). Today that
  parity is only checked at runtime inside `validate()` on real data — there is no
  unit test pinning the two implementations together, so they can drift.
- **`s5_sample.fit_scaler`/`apply_scaler`** (`dev/pipeline/s5_sample.py:136`, `:173`)
  implement the train-only standardization (log1p + median/IQR for skewed dollars,
  mean/std otherwise, zero-scale guard, null→0). A fit/apply **round-trip** on a known
  frame, plus a check that `cutoff_ym` actually isolates the train slice, would lock
  down behaviour the model training depends on. `training_mask` (`:72`) is a one-liner
  with a subtle strict-`<` leakage rule and a `None`→`"TRUE"` branch — trivially
  testable, currently untested.
- **`s3_clean._transform_batch`/`_stage1_exprs`/`_stage2_exprs`** — date parsing,
  Y/N→bool, categorical maps, `period_ym`/`orig_ym` calendar keys, and the five
  missingness flags, all pure on a bounded `pl.DataFrame`.
- **`s1_inventory`** — pure helpers (`_quarter_key`, `_all_quarters_between`,
  `_fingerprint`, `_coerce`) and the statistical `_apply_cross_file_checks`
  (modal release cut-off + relative-volume outlier) run on synthetic manifest rows.

### 3.2 Model — config and wiring

- **`config.py`** (`dev/model/config.py`): `train_bounds`/`val_bounds`/`test_bounds`
  (`:74`–`:86`) must **tile** the `period_ym` axis with **no overlap** for every
  window `k = 2015…2025`; `label_year` (`:95`) and the `LABEL_YM_SQL` shift (`:102`)
  encode the Dec→Jan rollover (`+89` in YYYYMM units); `sql_period_mask` (`:89`)
  emits the `>= lo AND < hi` predicate. These are pure integer functions — the single
  cheapest, highest-value gap to close, since a boundary error here leaks data
  between train/val/test in *every* backtest window.
- **`export.py`**: `OUTPUT_COLS` ordering, the current→current thinning weight
  `1/P_KEEP_CURRENT` (= 20.0), and the label-month shift — schema/weight correctness
  the whole training export rests on.
- **`benchmarks.py`** `_weighted_terciles`, **`smm_paths.py`**, **`evaluate.py`**,
  **`backtest.py`** — the per-window loop and evaluation alignment are currently
  guarded only by integration-level `Accept` checks, not unit tests.

### 3.3 Analysis utilities

- **`eda_common.py`**: `wilson_ci` (edge cases `n=0`, `p∈{0,1}`), `equal_pop_edges`
  (weighted quantile bucketing, tie handling), `hazard_curve` (bucket count, rates in
  `[0,1]`). These feed *every* F3.x nonlinearity figure, so a bug propagates widely.
- **`mkt_rate.py`**: `_month_grid` (Jan↔Dec rollover, gap-free sequence) and the
  thin-month forward-fill flagging.

## 4. Prioritized recommendations (proposed test backlog)

Each item is a proposed new hermetic test file in the existing style (synthetic
frames, closed-form assertions, same runner). Ordered by risk × ease.

### P0 — high risk, pure logic, direct model-correctness impact
1. **`dev/model/test_config.py`** — for `config.py`. Assert: for all `k`,
   `train/val/test_bounds` are contiguous and non-overlapping and that
   `val`/`test` are exactly one and two Decembers wide; `label_year` handles Dec
   rollover; `sql_period_mask` emits the correct `>= / <` operators.
   *Accept:* the three ranges tile `[PANEL_PERIOD_MIN, _dec(k))` with no gap/overlap
   for every `k ∈ TEST_YEARS`.
2. **`dev/pipeline/test_s4_panel.py`** — for `_finalize` and the `_STATE_SQL`↔
   `schema.derive_state` parity. Assert: `state_next` is null at a loan boundary and
   equals the next row's state within a loan; `censored` is true **only** for known,
   non-absorbing terminal rows; build a small DuckDB frame and a Polars frame from the
   same synthetic rows and assert the SQL and Polars state derivations agree on every
   row (the parity `validate()` only checks at runtime today).
   *Accept:* a hand-built 3-loan fixture with a prepay, a censor, and a normal roll
   produces the expected `state_next`/`censored`; SQL vs Polars match exactly.
3. **`dev/pipeline/test_s5_sample.py`** — for `fit_scaler`/`apply_scaler`/`training_mask`.
   Assert: fit→apply round-trip standardizes a known column to ~mean 0 / unit scale;
   log columns go through `log1p` first; a constant column hits the zero-scale guard
   (scale=1.0); `training_mask(None)=="TRUE"` and `training_mask(c)=="period_ym < c"`;
   fitting with a `cutoff_ym` uses only rows below it.
   *Accept:* round-trip recovers known centers/scales; cutoff isolation verified on a
   two-period fixture.

### P1 — data-transform correctness
4. **`dev/pipeline/test_s3_clean.py`** — `_transform_batch` on a small synthetic batch:
   MMYYYY→date, Y/N→bool (and `7`/`9`/blank→null), categorical code→label maps,
   `period_ym`/`orig_ym` computation, and the five `*_missing` flags.
5. **`dev/model/test_export.py`** — `OUTPUT_COLS` order/completeness, thinning weight
   `1/P_KEEP_CURRENT`, and the `LABEL_YM_SQL` month shift on representative months
   (incl. December).
6. **`dev/pipeline/test_s1_inventory.py`** — pure helpers (`_quarter_key`,
   `_all_quarters_between`, `_fingerprint`, `_coerce`) and `_apply_cross_file_checks`
   (modal cut-off detection + relative-volume outlier) on synthetic manifest rows.

### P2 — utilities & breadth
7. **`dev/analysis/test_eda_common.py`** — `wilson_ci` (`n=0`, `p∈{0,1}`),
   `equal_pop_edges`, `hazard_curve`.
8. **`dev/model/test_benchmarks.py`** — `_weighted_terciles` weighted-quantile edges.
9. **`dev/analysis/test_mkt_rate.py`** — `_month_grid` rollover; forward-fill flagging.
10. Fold the existing `verify_logit_equiv.py` / `verify_m9.py` checks into discoverable
    `test_*` entry points (kept skippable when artifacts/torch are absent).

## 5. Infrastructure recommendations

The suite is hermetic and CPU-only, so the missing scaffolding is low-cost and
high-leverage:

1. **Add a `pyproject.toml` `[tool.pytest.ini_options]`** at the repo root —
   `testpaths = ["dev"]`, the default `test_*.py` discovery, and a marker (e.g.
   `requires_torch`) so torch-dependent loader tests can be skipped where torch is
   absent. This makes `python -m pytest` a one-command run instead of file-by-file.
2. **Adopt `pytest-cov`** with an **initial non-blocking** coverage report
   (`--cov=dev --cov-report=term-missing`). `.gitignore` already lists `.coverage`
   and `htmlcov/`, so no new ignores are needed. Set a floor only once the P0/P1
   tests land, to avoid a vanity gate on day one.
3. **Add a minimal GitHub Actions workflow** (`.github/workflows/tests.yml`) that
   installs `dev/pipeline/requirements.txt` + `dev/model/requirements.txt` and runs
   the hermetic suite on push/PR. No SSD or GPU is required, so CI is genuinely
   green-able; gate it on the pure-logic tests and mark anything needing torch as
   optional.
4. **Consider a single `conftest.py`** housing the shared synthetic-frame builders
   (`_synthetic_pool`, `_nat`/`_state`/`_loan`, `_frame`) that are currently
   re-implemented per test file — a small DRY win, not a prerequisite.

## 6. Out of scope (deliberately)

End-to-end pipeline runs (Stage 1→6 on a real quarter), full GPU training, and the
~800 GB data dependency stay **integration** concerns, governed by the per-task
**Accept** criteria in `dev/model_plan/04_TASKS.md` and the pipeline `HOWTO_RUN.md`,
not by this unit suite. The recommendations above keep every proposed test hermetic
so the suite stays fast and runnable without the SSD.

## 7. Verification of this review

- The "current tests" table was cross-checked against the actual files
  (`dev/pipeline/test_schema.py`, `dev/model/test_{macro_features,features,pool,pools,economics}.py`).
- Every module and line reference cited in §3 was read directly (e.g.
  `s4_panel.py:57,124`, `s5_sample.py:72,136,173`, `config.py:74–102`).
- Infra absence confirmed: no `pyproject.toml`/`pytest.ini`/`setup.cfg`/`tox.ini`/
  `conftest.py`/`.coveragerc` and no `.github/workflows/` exist in the repo.
- At review time the hermetic suite could not be executed in this container (`polars`,
  `numpy`, `torch`, `pytest` were not installed). It has since been run after installing
  the SSD-free deps — `pip install pytest numpy -r dev/pipeline/requirements.txt` plus a
  CPU torch wheel, then `python -m pytest` → **94 passed** (see §0). Each file's built-in
  `python dev/.../test_*.py` runner also still works. Note the model/analysis deps live
  in `dev/analysis/requirements.txt` (there is no `dev/model/requirements.txt`).
