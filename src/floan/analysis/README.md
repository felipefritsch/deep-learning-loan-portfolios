# src/floan/analysis — exploratory data analysis (Phase 1)

Model-free evidence about the dataset and the structure the later models exploit: one script per
table (`T*`) or figure (`F*`), each its own `python -m floan.analysis.<name>` entry point. Outputs
land on the SSD (`outputs/tables/eda/`, `outputs/figures/eda/`). Spec: `specs/model/01_EDA.md`.

### Coverage & target structure
| File | What it shows |
|---|---|
| `t1_1_coverage.py` | panel coverage table (loans, loan-months, span) |
| `t1_2_covariates.py` | covariate summary (ranges, missingness, types) |
| `t1_3_state_dist.py` | the seven-state distribution (the `current`-state dominance the thinning addresses) |
| `t2_1_transition_matrix.py` | the empirical transition matrix, full panel pooled over time |
| `f2_2_transition_rates.py` | transition rates over calendar time (the 2003/2020 refi waves, the 2008–11 default wave) |
| `f2_3_covid_check.py` | COVID-era delinquency-code sanity check (forbearance handling) |
| `f2_4_roll_rates.py` | cure / roll-rate areas between delinquency buckets |

### Nonlinearity & interactions (the case against a linear model)
| File | What it shows |
|---|---|
| `f3_1_prepay_age.py` | prepayment vs loan age — the seasoning hump |
| `f3_2_prepay_incentive.py` | prepayment vs refinancing incentive — the S-curve |
| `f3_3_delinq_fico.py` | delinquency vs FICO — convex decay |
| `f3_4_fico_ltv_heatmap.py` | the FICO × LTV interaction heatmap |
| `f3_5_incentive_fico.py` | incentive × FICO — refi response by credit quality |
| `f3_6_vintage.py` | default hazard by origination cohort (vintage effects) |

### Macro covariates & rolling design
| File | What it shows |
|---|---|
| `mkt_rate.py` | builds the panel-derived market-rate proxy (`processed/macro/mkt_rate.parquet`) |
| `f4_1_mkt_rate_validation.py` | validates that proxy against the Freddie Mac PMMS survey rate |
| `f4_2_macro_panel.py` | small-multiples of the macro panel (rates, unemployment, HPI) |
| `f4_3_state_dispersion.py` | cross-state dispersion check on the macro block |
| `f5_1_shard_balance.py` | loan-keyed shard balance (the backtest partition) |
| `f5_2_vintage_shard.py` | vintage × shard mix |

`eda_common.py` holds the shared helpers (DuckDB connection to the panel views, figure/table save
conventions).
