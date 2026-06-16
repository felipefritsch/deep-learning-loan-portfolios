# AI-Use Self-Audit — MCF Dissertation

**Policy basis:** Oxford Mathematical Institute, *Guidance for students on the use of
generative AI in summatively assessed work* (covers MCF/MMSC/MFoCS dissertations),
read with the University's AI-study guidance and the Declaration of Authorship.

**Core principle (memorise this):** the department *encourages* AI use as an extended
toolkit, but (a) you must **declare how** you used it, and (b) you take **full
responsibility** for everything submitted — AI errors cost marks even when cited.
The test for any task: *"would it be inappropriate to ask another person to do this
part for me?"* If yes, AI may not do it either.

Work top to bottom. Tick each box only when the statement is **true**, not aspirational.

---

## A. The declaration (mandatory — this is the one hard requirement)

- [ ] `ai_declaration.tex` is included in `main.tex` and every `[CONFIRM]` marker has been
      verified and replaced with a true statement.
- [ ] The declaration states **how** the tools were interacted with (spec-driven, iterative,
      reviewed per task) — not just *that* they were used.
- [ ] It names the tool(s) and the rough period of use.
- [ ] Anything taken **verbatim** from AI (text or maths) is flagged in quotation marks and
      cited as a "personal communication" — or there is no such verbatim material (preferred).

## B. Permissible uses — confirm each was *declared*

These are allowed **with declaration**. For each, confirm it's covered by the declaration.

- [ ] **Code generation / translation** (pipeline `dev/pipeline/`, models `dev/model/`,
      EDA `dev/analysis/`): declared as AI-assisted, iterative, student-reviewed.
- [ ] **Literature search**: if AI was used to find sources, you personally verified each
      reference exists, is relevant, and is correctly described; the discussion is yours.
- [ ] **Grammar/spelling only** (if any tool like this was used): declared; no domain content
      was introduced by it.
- [ ] **List/bibliography formatting** (e.g. `refs.bib` tidy-ups): fine; declared if used.

## C. Impermissible uses — confirm NONE occurred (this is the real risk area)

Go file-by-file. These are the things that turn "tool use" into misconduct.

- [ ] **No substantive AI-written prose.** Re-read with this lens: `chapter1.tex` (intro),
      `conclusions.tex`, the literature review, and each chapter's framing. Every argument and
      sentence is in your own words. *If any passage was AI-drafted, rewrite it yourself now.*
- [ ] **No AI "improvement" beyond spelling/grammar** — nothing where AI supplied domain
      knowledge, restructured an argument, or strengthened the logic of your exposition.
- [ ] **No AI interpretation of data or maths.** The readings in the memos
      (`02b_loan_level`, `02c_robustness_caveats`, `03_pool`, etc.) and in chapters 3–5 —
      what the coefficients/curves/errors *mean* — are your own analysis, not AI-generated.
- [ ] **No AI-produced plots.** Every figure in `writeup/latex/figs/` was rendered by *your*
      (AI-assisted) plotting code from data you generated. You did **not** hand data to an AI
      and receive a chart/table back, and no data in any diagram was altered by AI.
- [ ] **No undeclared AI code.** All AI-assisted code is covered by the declaration (Part A/B).

## D. Data security

- [ ] No confidential or licensed third-party data was pasted into a third-party AI tool.
      (Fannie Mae public data → fine. Confirm nothing under a restrictive licence was shared.)
- [ ] Per the project invariant, raw data stays on the SSD; AI was given *code/specs*, not
      bulk licensed data.

## E. Ownership test (what the examiner / viva actually probes)

- [ ] You can explain, from first principles, **every modelling choice** (seven-state target,
      window/early-stopping scheme, pool roll-forward, cashflow engine) without reference to AI.
- [ ] You can reproduce or re-derive any result if asked, and you understand each figure.
- [ ] You could defend the claim that **you** are the author and driver of the intellectual
      content, with AI as a tool — not the reverse.

---

### Notes / exposure log
*(Optional: jot any borderline cases here so you can describe them accurately in the
declaration rather than discovering them under questioning.)*

- …
