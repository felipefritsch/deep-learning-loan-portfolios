# src/floan/model — the model ladder, sequence models, pools & pricing

Phases 2–3 of the project: estimate the seven-state monthly transition model with a ladder of
learners, evaluate them strictly out of sample across the 11 rolling windows (2015–2025), then carry
the fitted models to the pool level and into prices. Every model is scored through the **same**
loss/eval path so a difference between two models is attributable to the model, not to preprocessing.

Run any module as `python -m floan.model.<name>`. Below, files are grouped by role and described by
**what they do** (the codebase was built task-by-task, so some docstrings cite task codes like "M8";
those are just build provenance).

### Configuration, features, data
| File | What it does |
|---|---|
| `config.py` | the 11 rolling backtest windows (train/val/test label-month masks), the export knobs (thinning keep-rate, eval shard block), and the selected frozen configs |
| `export.py` | builds the shared training export — one thinned, importance-weighted **train pool** + a frozen, unthinned **eval pool** (loan-disjoint by shard) used by every model |
| `features.py` | the feature pipeline: train-only standardisation, embedding vocabularies (with a reserved unseen/missing slot), the one-hot path for the logit, and missingness indicators |
| `data.py` | the shard-streaming loader (yields every row once per epoch in shuffled order, carrying importance weights) |
| `torch_common.py` | shared PyTorch pieces: the importance-weighted cross-entropy loss and the streaming negative-log-likelihood / AUC evaluator used by all neural models |

### External macro data
| File | What it does |
|---|---|
| `fetch_macro.py` | fetches the macro series snapshot (30-yr mortgage rate, Treasury yields, national/state unemployment, house-price indices) |
| `build_macro.py` | builds the two macro Parquet tables (national + state) from the fetched series |
| `macro_features.py` | joins macro to the loan panel and derives the economic covariates (refinancing incentive, mark-to-market LTV, 12-month house-price change) |

### The model ladder (estimators)
| File | What it does |
|---|---|
| `benchmarks.py` | the **empirical transition matrix** — the covariate-free base-rate floor (plus a bucketed variant), scored on every window |
| `logit.py` | the **multinomial logistic regression** — the linear baseline, implemented as a zero-hidden-layer network so it shares the nets' loss/eval exactly |
| `refit_logit.py` | a memory-safe **streaming** re-fit of the full-scale logit (avoids materialising the design matrix) |
| `verify_logit_equiv.py` | a one-time equivalence receipt that the backtest's reimplemented logit matches a standard library fit |
| `net.py` | the **deep neural network** — an MLP over standardised continuous + embedded categorical + binary inputs; the nonlinear headline model |
| `train.py` | the single-window neural-net training loop (Adam, dropout, early stopping on the validation year, resumable from checkpoint) |
| `grid.py` | the **architecture/regularisation grid search** (depth × dropout × L2) on the tuning window |
| `table_a.py` | emits the architecture-search results table (the depth/dropout/ensemble grid) |
| `verify_m9.py` | verifies the grid's acceptance criteria and prints the evidence |
| `ensemble.py` | the **deep-net ensemble** (independently initialised nets, averaged) plus the ensemble-size curve |
| `gbt.py` | the **gradient-boosted-tree baseline** — a single 7-class softmax LightGBM with native categoricals; the non-neural flexible learner |

### Evaluation & robustness
| File | What it does |
|---|---|
| `evaluate.py` | the evaluation suite: per-window and pooled NLL, NLL-by-origin-state, one-vs-rest AUC, and calibration — every model through one code path on identical frozen test rows |
| `auc_by_window.py` | AUC as the **cross-period display metric**: the 2015–2025 per-window time series for the diagnostic transitions |
| `robustness.py` | robustness of the model *comparison*: seed variance, ranking stability across regimes, tuning-window sensitivity, and the layer-width sweep |

### Memory / path-dependence (the sequence-model thread)
| File | What it does |
|---|---|
| `history.py` | engineered borrower-history summaries (previous state, months-since-last-delinquency, ever-delinquent, prior-episode count) for the Markov-assumption test |
| `markov_probe.py` | the **Markov-assumption probe**: does history beyond the current state predict next-state transitions? (adds the history features to the existing net and measures the gain) |
| `sequence.py` | materialises per-loan **sequence tensors** (the trailing-T monthly feature windows) and shards them to disk — the shared data path for the GRU and transformer |
| `seq_model.py` | the **GRU** sequence model (a 1-layer gated recurrent unit over the trailing 12 months) — learned memory |
| `seq_transformer.py` | the **transformer** sequence model (a small self-attention encoder over the same window) — the attention counterpart to the GRU |
| `seq_train.py` | the sequence-to-one training loop / smoke for the sequence models |
| `seq_spike.py` | the three-arm comparison (current-state-only vs engineered-history vs GRU) and its go/no-go read |

### Pool-level prediction & economic translation
| File | What it does |
|---|---|
| `pool.py` | the **roll-forward harness**: composes the frozen monthly model to a horizon by matrix multiplication (with a first-passage variant for "ever 60+ DPD"), behind a model-agnostic predictor seam, and the pass-through **cashflow engine** |
| `pools.py` | forms characteristic-bucket and random pools at each window anchor; predicted vs realised event counts |
| `smm_paths.py` | the UPB-weighted monthly pool **SMM/CPR paths** (prepayment speed) read off the roll-forward |
| `economics.py` | the **economic translation**: CPR / weighted-average-life / **price** errors per pool × model × anchor |
| `calibration.py` | the per-horizon calibration decision and (per-window temperature-scaling) calibrator fit |
| `pricing_grid.py` | the rolling **pricing grid** — the horizon × regime price-error exhibit (1/3/6/12 months across anchors) |
| `make_pool_tex.py` | emits the pool-level LaTeX tables (pool accuracy, economic errors) for the dissertation |

### Sequence-model pricing
| File | What it does |
|---|---|
| `seq_predict.py` | one-month-horizon pool pricing for the GRU, dropped onto the same predictor seam as the per-month models |
| `seq_pricing.py` | **Monte-Carlo path simulation** for pricing the sequence models at horizons > 1 (per `ADR-002`): draw trajectories, score the trailing window each month, sample, evolve covariates, feed the same cashflow engine; reduces to the exact roll-forward at H=1 |
| `seq_pricing_merge` (`m27b_merge.py`) | merges the sequence-model H=1 pricing column into the rolling pricing exhibits |

### Committed result artifacts
- `artifacts/results/m27a_results/`, `artifacts/results/m27a_results_10m/` — the per-window,
  per-seed result JSONs (test NLL, AUC) and
  summary tables for the **sequence-model rolling comparison** at the 1.5M and 10M training-sample
  scales. See those folders' README for the schema.

The full task sequence and acceptance criteria are in `specs/model/04_TASKS.md`; the economic-engine
design is in `specs/model/ECONOMIC_ENGINE.md` and the decision records `ADR-001` (the predictor seam)
and `ADR-002` (the Monte-Carlo sequence-pricing seam).
