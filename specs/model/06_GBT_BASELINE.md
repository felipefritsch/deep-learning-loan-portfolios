# 06 — GBT Baseline: Gradient-Boosted Trees as a Non-Neural Flexible Learner

> **Goal:** add **one** gradient-boosted-tree learner (LightGBM, multiclass softmax) to the loan-level comparison of `02_LOAN_LEVEL.md` and the pool roll-forward of `03_POOL_LEVEL.md`, as a *parallel flexible learner alongside the net* — not a new nested rung. The net and GBT have comparable capacity but opposite inductive bias (smooth compositions vs axis-aligned partitions), so a net-vs-GBT comparison isolates **which kind of flexibility** the prime-conforming transition surface rewards — sharpening the dissertation's nonlinearity claim beyond "flexible beats linear." GBT consumes the **same** export, loader, evaluation, roll-forward and cashflow machinery as every other model; the only new code is the trainer (`gbt.py`) and a one-line predict adapter. Nothing in Phases 1–3 is re-run. This spec is implemented by tasks **M16–M19** in `04_TASKS.md`.

---

## 1. Where GBT sits (and what it reuses)

GBT is a fifth model in the existing hierarchy — `{empirical matrix, logit, best NN, ensemble, **GBT**}` — added because it is the field-standard strong **non-neural** baseline on tabular data (the *Elements of Statistical Learning* lineage already cited in the DL-foundations appendix), and because the original paper (`sirignano2021`) benchmarks only against linear/logistic specifications, never a tree ensemble — so the net-vs-GBT race on this panel is a genuine gap, not a re-tread.

It reuses, unchanged:
- the shared **train pool** (thinned, importance-weighted) and **eval pool** (unthinned, fixed shard block) of `02 §3` — GBT reads the *same* Parquet the net reads;
- the **window masks** and **frozen test slices** of `00_OVERVIEW §3.2` — identical rows, identical splits;
- `evaluate.py` (`02 §7`) — GBT predictions are an `(n, 7)` array that flows through the existing NLL / AUC / calibration path with no change to the evaluator;
- `pool.py`'s composition engine (`03 §3`) and cashflow engine (`03 §5`) — GBT enters via a model-agnostic predictor (M19).

New code is confined to `dev/model/gbt.py` (trainer + predict adapter) and a single model-agnostic-predictor seam in `pool.py`.

## 2. Locked decisions (binding — these preserve the controlled comparison)

1. **Single 7-class softmax model, origin `state` as a categorical input feature — NOT four per-origin models.** This matches the net/logit design (`02 §1`): every model conditions on the same origin set as a feature, so a net-vs-GBT gap is attributable to the *learner*, not to a different problem decomposition. The four-model alternative would confound learner family with decomposition and break the control. (Tree-specific reassurance: with origin as a categorical feature of cardinality 4, a single LightGBM model *contains* the four-separate-models solution as a reachable special case — split on origin at the root — but is free to pool structure across origins where the data justify it. Going single forfeits nothing and gains data-adaptive pooling under the severe origin imbalance.)
2. **Feature set byte-identical to the net's input**, minus standardisation. GBT consumes the same `feature_spec` columns, the same `UNK`-routed categoricals, the same missingness indicators (`02 §2`). It skips the per-window scaler only because trees are scale-invariant — the *columns* are identical, so the comparison isolates inductive bias.
3. **HT importance weights on the train pool only; val and test are unweighted and unthinned** (`00_OVERVIEW §6.3–6.4`). Weights enter LightGBM's per-row gradient/Hessian natively, keeping the thinned-sample loss an unbiased estimate of the full-sample loss — exactly as the net's weighted cross-entropy does.
4. **CPU, not GPU** — a deliberate, documented deviation from `00_OVERVIEW §3.1`'s GPU-training default. GBT at ~30 features is CPU-bound (leaf-wise growth is sequential; the histogram work a GPU parallelises is small at low feature count — LightGBM GPU speedup is often ≤1× here). GBT runs on the local Mac or a cheap CPU/low-end pod; the ~$13–15 of GPU credit is reserved for any net re-runs. Architecturally GBT is closer to `benchmarks.py` (a non-neural baseline) than to `net.py`, but heavy enough to warrant its own module.
5. **Tune once on k=2015, freeze, roll** (`02 §6` / M9 protocol). One dev-scale grid on the tuning window selects the config; it is frozen and looped over all 11 windows by `backtest.py`. Per-window val serves early stopping only.
6. **Calibration is conditional** (§4) — decided by the reliability diagrams, not applied by default.

## 3. Model specification (`gbt.py`)

LightGBM `train` with:
- `objective="multiclass"`, `num_class=7`, `metric="multi_logloss"` — softmax outputs sum to one, which the NLL headline and the pool roll-forward both require (one-vs-all `multiclassova` is wrong here: per-class sigmoids need not normalise).
- origin `state` (+ the other categoricals) declared via `categorical_feature`; LightGBM handles NaN natively but the net's missingness-indicator columns are still passed so the feature set matches.
- `sample_weight` = the HT weights on the **train** Dataset; the **val** Dataset carries no weight.
- `early_stopping` on the window's val multi-logloss (patience comparable to the net's).
- memory/throughput flags as needed: `max_bin` 255 (drop to 63 if tight), `two_round=True`, `save_binary` the constructed Dataset and reuse across windows, `free_raw_data=True`. The thinned per-window train slice (≤ the full-export row count, ~2 GB binned) fits in RAM — materialising it is in-scope and expected (the "no full `.collect()`" pipeline invariant protects the 3.3 B-row raw panel, not the thinned training export).

**Tuning grid (k=2015, dev scale, pruned — anchor and freeze):** `num_leaves ∈ {31,63,127}`, `learning_rate ∈ {0.05,0.1}`, `min_sum_hessian_in_leaf` (sweep — the HT weights inflate summed leaf Hessians, so the default is miscalibrated to this objective), `feature_fraction ∈ {0.7,1.0}`, `lambda_l2 ∈ {0,1}`. Select on the 2014 val NLL. Mirror M9/M10's dev-scale-first, then re-confirm at full scale before the rolling loop.

**Numerical cross-check (the GBT analogue of M7's sklearn check):** LightGBM's reported `multi_logloss` on the eval slice must equal `evaluate.py`'s NLL on the same `(n,7)` predictions to ~1e-6 — confirming the two NLLs are the same object before any model comparison is read.

## 4. Calibration (conditional)

GBT softmax probabilities can be miscalibrated (boosting tends to over-confidence, worst on the rare credit-event classes that drive cashflows), and the pool valuation consumes **levels**. So: run the existing reliability diagrams on the **raw** GBT outputs first (`02 §7` path, reused). If they sit on the diagonal, do nothing and document it. If they deviate, fit per-window post-hoc calibration on the window's **val** slice only (never train, never test) — **temperature scaling** first (one parameter, preserves AUC/ranking), per-class **isotonic** + renormalise as fallback — and apply it before every levels-consuming step, including inside the M19 roll-forward predictor. Refit per window to stay consistent with the rolling design. This is the one extra step GBT may need that the nets did not; it mirrors the calibration `fuster2022` applies to tree-based mortgage-default probabilities.

## 5. Integration points (modules)

| Module | Change |
|---|---|
| `dev/model/gbt.py` | **new** — trainer (weighted multiclass softmax, early stopping, run-folder writer) + `predict_proba(booster, X) → (n,7)` adapter + optional per-window calibrator. |
| `backtest.py` | add GBT to the per-window loop (frozen config; per-window vocab + early stopping); **no change to the window/masking logic**. |
| `evaluate.py` | **none** — GBT predictions flow through the existing NLL/AUC/calibration path as a fifth model column. |
| `pool.py` | one **model-agnostic predictor** seam (a callable returning the 7-vector at evolved covariates), then pass the net or GBT predictor; the composition and cashflow engines are otherwise untouched. |
| `config.py` | record the frozen GBT config alongside the net's. |

## 6. Evaluation additions

GBT becomes a column/line in the existing exhibits, no new exhibit types:
- **Table A** (tuning-window grid, `02 §7`): a GBT row beside logit / NN depths / ensemble.
- **Table B** (rolling 2015–2025 + pooled all-years) and **NLL-by-origin-state**: a GBT column.
- **AUC table** (one-vs-rest by origin→destination): logit vs best NN vs ensemble **vs GBT** — read especially in the sparse deep-delinquency cells where the net showed its largest edge (`90+→Foreclosure`).
- **Calibration**: GBT reliability diagrams (raw, and calibrated if §4 applies).
- **Pool level** (`03 §4–5`): GBT columns in the count R²/RMSE table (T4.2), the economic-error table (T5.1), and the price-error bucket map (F5.2), at the same regime anchors.

GBT results fold into **memo 2b** (loan-level) and **memo 03_pool**, and into the corresponding chapter tables, by the M11b/M15 writeup-sync pattern — not a new memo.

## 7. Known result framing (pre-register both ways; state in the memo)

The point of GBT is that the outcome is informative in every direction:
- **GBT ≈ net (a tie):** "flexibility is what matters; architecture is secondary" — reinforces the headline that 3 layers suffice in prime conforming credit (`sirignano2021` needed 5 on subprime-heavy private-label data).
- **net > GBT:** "the *smooth* structure of mortgage behaviour — the incentive S-curve, the seasoning hump, convex FICO decay — rewards smooth learners," the sharper claim, since trees fit piecewise-constant staircases and the target's economically important dimensions are smooth (`grinsztajn2022`'s smoothing mechanism).
- **GBT > net:** legitimate, but note that the tabular-ML metafeatures here (huge N, high size-to-feature ratio, severe class imbalance — `mcelfresh2023`) all favour GBT, so attributing a GBT win specifically to *smoothness* rather than those would need the ablation; report it as such rather than over-claiming.

Whichever way it lands, the per-origin NLL/AUC breakdown is the diagnostic: it shows where each learner's flexibility pays off, in exactly the sparse transitions the loan-level chapter already highlights.

## 8. Definition of done

- `gbt.py` exists; the frozen config is selected on k=2015 and recorded in `config.py` with rationale.
- GBT appears as a column in Table A, Table B (+ pooled), NLL-by-origin, and the AUC table, on identical frozen test rows, with no change to `evaluate.py`.
- Calibration decision documented (applied-with-evidence or skipped-with-evidence).
- GBT columns exist in T4.2 / T5.1 / F5.2 at the regime anchors, via the model-agnostic `pool.py` predictor, with the engine logic unchanged.
- Every GBT fit writes a self-describing run folder (`00_OVERVIEW §6.6`); the CPU/GPU deviation (§2.4) and the result framing (§7) are stated in the memo.

Implemented by **M16–M19** (`04_TASKS.md`).

---

### References to add to the bibliography
LightGBM (`ke2017`); the tabular DL-vs-GBT benchmark set — `grinsztajn2022`, `shwartzziv2022`, `mcelfresh2023`, `borisov2022`, `gorishniy2021`; the net-vs-tree credit precedent `albanesi2019` (Albanesi & Vamossy). `fuster2022`, `gu2020`, `hastie2009`, `sirignano2021` are already in the bibliography.
