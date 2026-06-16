# TERMINOLOGY_AUDIT.md — Task 5 (suggestion only; no prose changed)

**Baseline / course materials in the workspace.** The Deep Learning course
reading list (`docs/bib_refs/AY25-26_MATH_MScMCF_Deep Learning_HT.xlsx`) contains
**three textbooks only**: ESL (Hastie, Tibshirani & Friedman, 2009/2017), *Deep
Learning* (Goodfellow, Bengio & Courville, 2016), and *Reinforcement Learning: An
Introduction* (Sutton & Barto, 1998/2018). No lecture slides or notes are present
in the workspace; the second baseline is therefore the crosswalk table
`tab:dlcrosswalk` in `appendix3.tex`. No edits were made to any chapter or
appendix. I did **not** re-flag anything already covered by an inserted
`% SUGGESTED TRIM:` comment except to add a distinct scope point.

Honest summary up front: **genuine same-concept-different-name mismatches are
few**, because the text already bridges terminology deliberately (it equates
"weight decay", "ℓ2 penalty", and "ridge regularisation"; it names the logit a
"zero-hidden-layer network" on purpose). The real finding is in part (b): one
block (NTK) is not grounded in the reading list.

---

## (a) Terminology mismatches — thesis term vs course term

Only genuine same-concept cases. Finance/domain terms (mortgage, CPR, WAL,
prepayment, incentive, mark-to-market LTV, …) and precise technical terms are
deliberately **not** flagged.

| # | Thesis term | Course term (Goodfellow 2016) | Where | Recommendation | Confidence |
|---|---|---|---|---|---|
| 1 | "rectified-linear activations $\sigma(u)=\max(u,0)$" | **ReLU / rectified linear unit** | `chapter2.tex:259` | Spelled-out form is correct; the abbreviation **ReLU** never appears. Consider introducing "ReLU" once in parentheses so it matches the course's standard name. Purely cosmetic. | uncertain — confirm |
| 2 | hidden unit as a learned "soft feature" | **learned feature / representation** | `chapter2.tex:267` | Already in scare quotes as informal. Optional: align to "learned feature/representation" (Goodfellow's term). Not an error. | uncertain — confirm |
| 3 | multinomial logit as a "zero-layer / zero-hidden-layer network" | **softmax regression** | `chapter2.tex:225,236`; `appendix3.tex:53` | This is a *deliberate* framing that makes the model hierarchy nest cleanly, not a mismatch. Flagged only for completeness; recommend **keep as is**. | keep (low priority) |

Nothing else rises to a genuine mismatch. The appendix's bridging passages
("$\ell_2$ penalty (weight decay) is ridge regularisation in the sense of
Hastie") are exactly what an examiner wants and should stay.

---

## (b) Out-of-scope techniques/results (examiner "beyond syllabus" risk)

Grounding test: is the technique covered by one of the three reading-list
textbooks (primarily Goodfellow 2016 for the DL material)?

| Technique | Where | In reading list? | Assessment |
|---|---|---|---|
| **Neural Tangent Kernel** (`jacot2018`) | `appendix3.tex` §`sec:dlf-ntk`; crosswalk row "NTK / global convergence" | **No** — Jacot 2018 post-dates Goodfellow 2016 and is not in ESL or Sutton & Barto | **Primary flag.** This is the one block invoked as *support* (not merely surveyed) that has no grounding in any course text. The existing `% SUGGESTED TRIM` already says compress; the **added** point here is the scope one: if kept, frame NTK explicitly as background beyond the taught syllabus, or cut it (see TRIM_PLAN.md). |
| Transformers / attention (`vaswani2017`) | §`sec:dlf-arch` (surveyed, **not used**); crosswalk | No (2017, post-dates Goodfellow 2016) | Lower risk — explicitly surveyed and excluded by the Markov framing. Fine to keep as a one-line "not used"; do not expand. |
| Layer normalisation (`ba2016`) | §`sec:dlf-reg` (named, **not used**) | No (2016 arXiv, not in textbook) | Lower risk — named only to say it is deliberately omitted. Fine. |
| ResNet (`he2016`), LSTM (`hochreiter1997`), GAN (`goodfellow2014gan`) | §`sec:dlf-arch` (surveyed, **not used**) | Partial / borderline (LSTM & GAN appear in Goodfellow 2016; ResNet is era-adjacent) | In scope enough; surveyed-and-excluded. No action. |
| Universal approximation, backprop, SGD, Adam/RMSprop, dropout, ℓ2/weight decay, early stopping, batch norm, Glorot init | throughout `chapter2.tex` / `appendix3.tex` | **Yes** — all in Goodfellow 2016 | In scope. No action. |
| Horvitz–Thompson / inverse-probability weighting (`ht1952`) | `chapter2.tex:188` | No (survey-sampling, not the DL course) | **Not** flagged as out-of-scope: it is core statistics, correctly cited to its primary source, and is what makes the thinned-sample estimator unbiased. Legitimately cross-disciplinary, not padding. |

**Bottom line:** the only material an examiner could reasonably call beyond the
taught syllabus is the **NTK section**. Everything else either sits inside
Goodfellow 2016 or is explicitly surveyed-and-excluded. This reinforces (does not
duplicate) the existing NTK trim suggestion. *Uncertain — confirm* whether the
course lectures (not in the workspace) covered NTK; if they did, downgrade this
flag.
