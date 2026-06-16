# TRIM_PLAN.md — Appendix C (Deep-Learning Foundations)

**Suggestion only.** Nothing in `writeup/latex/appendix3.tex` has been deleted or
rewritten. The only additions to that file are inline `% SUGGESTED TRIM:` comment
lines, one per section, which this plan reconciles and expands. The revision and
the voice are yours.

**Guiding rule (yours).** *Keep* sections that defend a specific Chapter-2
decision — they earn their place by making a choice principled rather than
restating it. *Compress / cut* pure textbook recap an examiner would treat as
padding. The crosswalk table is keep.

**Scope note (see TERMINOLOGY_AUDIT.md).** The course's formal reading list is
three textbooks — ESL (Hastie 2009), Deep Learning (Goodfellow 2016), and RL
(Sutton & Barto 2018). Everything the appendix cites beyond those is supporting
literature the crosswalk maps in. Most appendix topics (universal approximation,
backprop, SGD, Adam, dropout, ℓ2, early stopping, batch/layer norm, Glorot init)
are covered by Goodfellow 2016; the **NTK section is the one block not grounded
in any reading-list text** (Jacot 2018 post-dates Goodfellow 2016).

| Section | Label | Recommendation | Rationale |
|---|---|---|---|
| Intro | `app:dlfoundations` | **Keep, tighten** | Frames the appendix. Trim sentences that only *announce* it is background ("without pausing to justify them", "It is deliberately background") rather than doing work. |
| Function classes & universal approximation | `sec:dlf-approx` | **Keep** (compress UAT) | Defends the deep hypothesis class over the logit and ends on the load-bearing point — the logit is the zero-hidden-layer boundary case, which makes the hierarchy a controlled comparison. Compress the textbook statement of the theorem to ~1 sentence. |
| Backprop & SGD | `sec:dlf-sgd` | **Keep core** (compress history) | Earns its place: it ties the two-level shuffle (Sec. `sec:meth-shards`) and the Horvitz–Thompson weights (Eq. `eq:weightednll`) to the requirement that minibatch gradients be unbiased. Compress the backprop / Robbins–Monro / Bottou lineage to a clause. |
| Adaptive optimisers | `sec:dlf-adam` | **Keep** | Defends a real Ch-2 control: Adam is used *identically* for the logit baseline so the network-vs-logit gap is architecture, not optimiser. Already short; leave essentially as is. |
| Regularisation, ensembling, normalisation | `sec:dlf-reg` | **Mixed** | KEEP the batch/layer-norm **omission** defence — it justifies a Ch-2 design choice (per-window standardisation instead, no train-time batch statistics leaking across the temporal split). COMPRESS the ridge / dropout / bagging recap to one sentence each. |
| Non-convexity & the NTK | `sec:dlf-ntk` | **Compress (cut if tight)** | The weakest-earning section and the only one **not grounded in the reading list** (see TERMINOLOGY_AUDIT.md §b). Keep at most the one sentence linking NTK / initialisation to the seed-and-ensemble robustness checks already run; cut the kernel-regression exposition if space is tight. |
| Architectures beyond feedforward | `sec:dlf-arch` | **Keep argument** (compress survey) | KEEP "why a plain feedforward net is the right tool here" — it defends the Ch-2 Markov framing (`sec:meth-model`) against the alternatives. COMPRESS the ResNet/LSTM/Transformer/GAN tour to a single sentence; the citations can stay in the crosswalk. |
| ML in credit & asset pricing | `sec:dlf-lineage` | **Keep** | Literature positioning, not textbook padding — and currently the *only* place the broader literature is positioned (see INTRO_GAPS.md). The Gu–Kelly–Xiu "levels, not ranks" link to the pool-level pricing exercise (`sec:meth-pool`) earns its place. |
| Course-to-dissertation crosswalk (+ Table `tab:dlcrosswalk`) | `sec:dlf-crosswalk` | **Keep** | Explicitly retained. The table is the appendix's load-bearing exhibit: it is what turns "background" into "evidence the methods trace to the course." |

## Bottom line
The appendix is strongest where it converts a Chapter-2 mechanic into a justified
choice — the shuffle and HT weights under SGD, Adam as a shared control, the
deliberate batch-norm omission, the feedforward-vs-alternatives argument, and the
crosswalk. It is weakest where it restates standard results for their own sake —
the universal-approximation exposition, the **NTK section**, and the architecture
survey. Compressing those three and keeping the rest preserves every claim that
does defensive work while removing what reads as textbook padding. If only one
cut is made, make it the NTK section: it is both the least load-bearing and the
least defensible on syllabus grounds.
