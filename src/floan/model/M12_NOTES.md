# M12 — Robustness of the model comparison (`02_LOAN_LEVEL §8`, Phase-2 gate)

The rolling backtest (§1) is the main design, so robustness here is not a new model: it asks
whether the headline **empirical < logit < NN** ordering is an artefact of one seed, one
tuning window, or the paper-inherited layer widths. Five readouts via `robustness.py`
(`seedvar | ranking | k2019 | width | perm | report | all`); none re-tunes on a test slice.
Artefacts in `outputs/tables/loan_level/{robustness,width_sweep,perm_importance}.*` and
`models/nn/dev/k2019_sensitivity_summary.json`.

## 1. Seed variance (§8.3) — no training
The M10b full-scale k=2015 ensemble is **8 independent draws of the frozen config**
(d3/do0.2/L2=1e-5, seeds 0–7; random-init + shuffle-order diversity) — a ready-made seed
study at full scale, so nothing was retrained. Single-net **test NLL 0.102925 ± 2.61e-04**
(range [0.102634, 0.103409]). The NN−logit gap on the same window is **+0.005321**
(logit 0.108246), i.e. **20.4× the seed sd**. The architecture's advantage is an order of
magnitude above run-to-run noise — the comparison is not a lucky draw.

## 2. Ranking stability (§8.4) — no training
From Table B, the per-test-year ordering is **invariant across all 11 windows**: the single
NN is best in 11/11 and beats the logit in 11/11, the logit beats the empirical matrix in
11/11 — `NN < logit < empirical` holds **without exception**, including the COVID-2020 window
(NN +0.0071 over logit, highest-entropy slice) and the 2022–23 rate-spike / prepay collapse
(2023 +0.0056). NN−logit margin ranges +0.0045–+0.0098 (mean +0.0068), narrowest in 2022. On
the 5 ensemble-bearing key windows the ensemble is ≤ the single net in 4/5 (2023 a
+2.2e-5 within-seed-noise tie) — ensembling sharpens but never reorders. The ranking is a
property of the problem, not of a regime (the regime-shift exhibit replacing a fixed COVID
split). Paragraph drafted in `robustness.md`.

## 3. Tuning-window sensitivity (§8.5) — k=2019, dev scale
Re-ran M9's **same pruned 14-cell** depth×dropout×L2 grid on the later window **k=2019** (dev),
written to `k2019_sensitivity_summary.json` (NOT the k=2015 `grid_summary.json` that table_a /
ensemble depend on). Val-NLL argmin: **depth 3, dropout 0.2, L2 0** (val 0.088072, test
0.100845). The tuning architecture decision — **depth 3, dropout 0.2 — is unchanged** two
years out (the d3/do0.2 cells occupy the top of the k=2019 ranking; depth-3 is the val winner
at both L2=0 and L2=1e-5). The L2 sub-choice lands at 0 vs the frozen 1e-5, but M10b already
showed the L2 axis is within noise (2.3e-5), so this does not move the architecture. The
selected config is not regime-specific.

## 4. Width sweep (appendix completeness check) — k=2015, dev scale
The 200/140 widths were inherited from Sirignano et al., not tuned. Holding the selected
config fixed (`net.depth_to_hidden` gained a `width_mult`; default 1.0 reproduces the paper
widths **bit-identically**, so every pre-M12 run is unchanged) and sweeping half/paper/double:

| Width | Hidden | Params | Val NLL | Test NLL |
|---|---|---|---|---|
| Half ×0.5 | [100,70,70] | 110,675 | 0.096269 | 0.102815 |
| Paper ×1 | [200,140,140] | 166,405 | 0.095931 | **0.102579** |
| Double ×2 | [400,280,280] | 349,265 | **0.095734** | 0.102711 |

**Conclusion (one line):** halving or doubling the widths moves out-of-sample NLL by only
**2.36e-04 across a 3.2× parameter range**; the paper width is itself the test-NLL best and
within 1.96e-04 (val) of the ×2 optimum — the inherited widths leave no accuracy on the table,
width is not a material tuning axis here. Table → `width_sweep.{csv,md,tex,json}` (appendix).

## 5. Permutation importance (§8.6, optional)
Deployed tuning-window net over its frozen test slice (6.18 M rows, baseline NLL 0.103125);
ΔNLL when each feature column is shuffled. The origin transition `state` dominates (ΔNLL
+0.157 — expected, it is the §1 conditioning variable), then the seasoning block (loan age
+0.036, remaining term +0.023) and balance; among the credit/rate covariates the EDA hazard
drivers lead — incentive +0.0023, FICO +0.0021, original rate +0.0015, mark-to-market LTV
+0.0005. Consistent with the EDA. Table → `perm_importance.{csv,md,json}`.

## Code / evidence
- `robustness.py` (new); `net.py` (+`width_mult`, default-1.0 bit-identical); `train.py`
  (threads `--width-mult` into build + cfg + metrics).
- Trained: `models/nn/dev/k2019_d*` (14 cells), `models/nn/dev/width_k2015_w{0.5,1,2}`.
  Reused: `models/nn/full/ensemble_k2015_summary.json` (seeds), `outputs/.../table_b.json`.
- Log: `logs/m12/robustness_all.log`. All Accept criteria verified (seed sd ≪ gap; stability
  paragraph; k=2019 documented; width table + one-line conclusion). **Phase-2 gate.**
