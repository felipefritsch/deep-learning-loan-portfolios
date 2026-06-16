# INTRO_GAPS.md — Task 6 (analysis only; nothing written into chapter1.tex)

What Chapter 1 (`writeup/latex/chapter1.tex`) is thin on, grounded in the current
text. Sections present: §`sec:intro-object` (11–50), §`sec:intro-design` (52–67),
§`sec:intro-contributions` (69–102), §`sec:intro-outline` (104–115).

**Strengths to preserve.** The modelling-object framing (§object, "a mortgage is
a small dynamical system", 14–50) is excellent and distinctive; the four
contributions (76–102) are crisp and genuinely original. The gaps below are
additive — none requires touching that material.

---

## Gap 1 — Literature positioning (largest gap)
Chapter 1 cites **only** `sirignano2021` (43, and implicitly at 81). The broader
literature it builds on — option-theoretic mortgage models
(`deng2000`, `schwartz1989`) and the ML-credit / ML-asset-pricing turn
(`khandani2010`, `fuster2022`, `gukellyxiu2020`) — appears in Chapters 2–3 and
Appendix C but **nowhere in the introduction**. An examiner expects the intro to
situate the contribution in its field.

*Suggestion:* add one short paragraph (4–6 sentences) after §object or inside
§design: the option-theoretic lineage (Deng–Quigley–Van Order; Schwartz–Torous),
the move to machine learning in credit (Khandani; Fuster et al.) and asset
pricing (Gu–Kelly–Xiu), and Sirignano et al. as the deep-learning antecedent this
work tests. **All five sources are already in `refs.bib`** — no new research
needed.

## Gap 2 — The research question is never stated explicitly
It is distributed and implicit: "the central empirical claim of \citet{sirignano2021}
… which this dissertation tests" (43–50) and contribution 1 (77–85, "asks whether
the deep-learning advantage survives in the clean GSE credit box"). The reader
must assemble it.

*Suggestion:* one or two crisp sentences stating the question(s) outright — e.g.
"Does the deep-learning advantage documented on subprime-heavy private-label data
survive in the prime conforming credit box, and in which market regimes is it
largest?" — placed at the end of §object or the start of §design.

## Gap 3 — The empirical setting and scale are absent from the intro
The introduction never names the dataset (Fannie Mae SFLP), its scale
(3.31 bn loan-months / 57.6 m loans), or its span (2000–2025). Contribution 2
(86–89) even invokes "eleven test years … 2015–2025" without saying what data
they come from. The reader learns the setting only in Chapter 3.

*Suggestion:* one sentence in §design naming the dataset, its scale, and the
2015–2025 rolling-test window.

## Gap 4 — Motivation / real-world stakes are light
§object motivates the modelling *primitive* elegantly but the broader "why this
matters" (size of the agency-MBS market, the pricing/risk-management/policy
stakes of getting prepayment and default right) is barely stated beyond a brief
clause on cashflows and MBS pricing (19–25, 48–50).

*Suggestion:* 2–3 sentences on the economic stakes — why transition risk on
agency mortgages matters to investors, hedgers, and policymakers.

## Gap 5 — Outline omits the appendices and conflicts with §design ordering
§outline (104–115) lists Chapters 2–5 but not the three appendices
(Supplementary, Reproducibility, **Deep-Learning Foundations**). It also presents
chapters methodology-first, which conflicts with §design's "exploratory-first"
phrasing (see REORDER_ANALYSIS.md).

*Suggestion:* add one clause pointing to the appendices; reconcile the ordering
language with whatever you decide in REORDER_ANALYSIS.md.

## Gap 6 (optional) — No results preview
Many intros preview headline findings. Contribution 1 already previews the
3-vs-5-layer result (83–85); you could add a one-sentence preview of the others
(network beats logit in all eleven test years; regime-dependent price-error
reductions) if you want the intro to advertise the payoff.

---

**Priority order:** Gap 1 (literature positioning) → Gap 3 (name the data) →
Gap 2 (explicit research question) → Gap 4 (stakes) → Gap 5 (outline) → Gap 6
(optional preview). Gaps 1–3 are what an examiner is most likely to mark.
Nothing here has been written into `chapter1.tex`.
