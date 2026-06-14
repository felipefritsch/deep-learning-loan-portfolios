# Memo 02c — Robustness caveats (M12)

*Phase 2 §8 of the modelling plan (`dev/model_plan/02_LOAN_LEVEL.md`); feeds §4.2 (rolling
backtest) and the supplementary appendix of the results chapter. M12 verified all four
Accept criteria (seed variance, ranking stability, k=2019 sensitivity, width sweep) — see
`outputs/tables/loan_level/robustness.{json,md}`. This memo records three caveats to surface
explicitly when writing up §4.2 and the appendix; none is a failure, each is a point a
careful reader could otherwise poke. Integrity confirmed: M12 did not overwrite the k=2015
`grid_summary.json` (still M9-dated) that `table_a` and the ensemble depend on; all M12
outputs are separate files.*

---

## 1. k=2019 reselects the architecture, but L2 lands at 0 (not the deployed 1e-5)

Re-running M9's pruned 14-cell depth×dropout×L2 grid on the later window k=2019 (dev scale)
reselects **depth 3, dropout 0.2** on that window's own validation NLL — the architecture
decision is unchanged on a regime two years after the tuning window. The one difference is
the L2 sub-choice: k=2019 picks **L2 = 0** (val 0.088072, test 0.100845), whereas the
deployed config uses L2 = 1e-5.

*Write-up point:* present k=2019 as confirming the **architecture**, and justify the L2
mismatch in one line — M10b already showed the L2 axis is within noise, so L2 = 0 vs 1e-5
does not move the model. Don't let the L2 difference read as instability of the selection.
Full ranking: `models/nn/dev/k2019_sensitivity_summary.json`.

## 2. The width sweep is dev-scale, generalised to the full-scale deployed net

Half / paper / double of the Sirignano 200/140 widths move out-of-sample NLL by only
**2.36e-04** across a 3.2× parameter range; the paper width is itself the test-NLL best.
Conclusion stands — the inherited widths leave no accuracy on the table — but it is a
**dev-scale** completeness check, while the deployed network is full-scale.

*Write-up point:* state the scale explicitly in the appendix caption; frame it as a
completeness/sanity check on an inherited hyperparameter, not a full-scale tuning result.

## 3. The ensemble's edge is ~1e-4 — do not pool it as a headline

The 8-net ensemble improves per-window NLL by only ~1e-4 and is ≤ the single net in 4 of
the 5 key windows (the fifth a within-seed-noise tie). This is consistent with §4.1's
"ensemble helps, with diminishing returns."

*Write-up point:* this ties directly to the §4.2 per-window caveat (see the M11b prompt
addition). The ensemble was fit only on the 5 key windows (2015/2019/2020/2023/2025), not
all 11, so **do not report a pooled all-years ensemble figure** beside the pooled
network/logit numbers — that pools incomparable row coverage, and a ~1e-4 edge does not
warrant a pooled headline. Keep the ensemble comparison per-window (or pooled over exactly
the matched 5 windows); restrict the pooled all-years comparison to logit vs the single net.
