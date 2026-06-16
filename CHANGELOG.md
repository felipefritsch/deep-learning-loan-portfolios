# CHANGELOG.md — surgical fix pass (refs.bib, Ch. 2/3, Ch. 1/4, Appendix C)

Date: 2026-06-15. Scope: `writeup/latex/`. Build verified after all edits.

## Build status (clean)
`latexmk -pdf main.tex` from a cleaned tree: **exit 0**, **57 pages**,
**0 `!` errors**, **0 undefined references/citations** (grep of `main.log`),
**0 bibtex warnings** (`main.blg`; `warning$ -- 0`), **28 bib entries used**.
Every `\cite` key in `appendix3.tex` resolves.

Line numbers in "before" refer to the pre-edit files; exact before→after text is
quoted so each change is unambiguous.

---

## TASK 1 — refs.bib corrections

| Key | Change | Before → After |
|---|---|---|
| (header) | replaced stale comment | `% Dissertation bibliography. Keys currently cited in chapter3.tex:` / `% sirignano2021, deng2000, schwartz1989, fanniemae2024. Add as chapters grow.` → `% Dissertation bibliography.` |
| `gukellyxiu2020` | added DOI | (no doi) → `doi = {10.1093/rfs/hhaa009}` |
| `fuster2022` | added DOI | (no doi) → `doi = {10.1111/jofi.13090}` |
| `bai2025` | added DOI; dropped spurious issue | `number = {1}` (no doi) → `doi = {10.1146/annurev-statistics-112723-034423}`, `number` removed |
| `kingma2015` | added arXiv note (type already `@inproceedings`, no DOI) | (no note) → `note = {arXiv:1412.6980}` |
| `ba2016` | retyped + field | `@article … journal = {arXiv preprint arXiv:1607.06450}` → `@misc … howpublished = {arXiv preprint arXiv:1607.06450}` |
| `ioffe2015` | added series/volume | (none) → `series = {PMLR}, volume = {37}` |
| `glorot2010` | added series/volume | (none) → `series = {PMLR}, volume = {9}` |

Not in correction scope, left untouched: `sirignano2021`, `deng2000`,
`schwartz1989`, `dropout`, `ht1952`, `fanniemae2024`, and the books.

### STEP A confirmations (no edit needed)
1. **Entry types for no-DOI items are all correct** — `@inproceedings`:
   `kingma2015`, `jacot2018`, `he2016`, `vaswani2017`, `goodfellow2014gan`,
   `ioffe2015`, `glorot2010`; `@misc`: `ba2016`, `tieleman2012`, `fanniemae2024`.
   None is a malformed `@article` with empty volume/pages.
2. **`tieleman2012` (RMSprop) is a well-formed `@misc` lecture cite** — Tieleman &
   Hinton, Coursera "Neural Networks for Machine Learning," Lecture 6.5, 2012, no
   pages/DOI. **Resolved; no change, no flag.**
3. **`jacot2018`** is `@inproceedings`, no DOI, and **retains its verified NeurIPS
   2018 page range 8571–8580**. STEP A's "(no volume/pages)" was noted; since the
   pages are verified-correct they were kept (see "Needs Felipe" if you prefer the
   page-less NeurIPS style).

### Per-entry verification table (every key)
"Verified-against-source" = web-checked this pass (or, for books, against the
course reading list / workspace PDF). All 28 entries carry a status; **none is
left-as-is-unverifiable**.

| Key | Status this pass | Verified against |
|---|---|---|
| sirignano2021 | left as-is | JFinEc 19(2):313–368, 2021 (Oxford Academic) |
| deng2000 | left as-is | Econometrica 68(2):275–307, 2000 |
| schwartz1989 | left as-is | J. Finance 44(2):375–392, 1989 |
| dropout | left as-is | JMLR 15:1929–1958, 2014 |
| ht1952 | left as-is | JASA 47(260):663–685, 1952 |
| fanniemae2024 | left as-is | `@misc` official Fannie Mae SFLP data page; not re-verified as a journal cite |
| goodfellow2016 | left as-is | Course reading list + workspace PDF (MIT Press, 2016) |
| hastie2009 | left as-is | Course reading list (Springer, 2nd ed, 2009) |
| suttonbarto2018 | left as-is | Course reading list (MIT Press, 2nd ed, 2018) |
| cybenko1989 | left as-is | MCSS 2(4):303–314, 1989 |
| hornik1991 | left as-is | Neural Networks 4(2):251–257, 1991 |
| rumelhart1986 | left as-is | Nature 323:533–536, 1986 |
| robbins1951 | left as-is | Ann. Math. Statist. 22(3):400–407, 1951 |
| bottou2018 | left as-is | SIAM Review 60(2):223–311, 2018 |
| kingma2015 | **corrected** | ICLR 2015; arXiv:1412.6980; no DOI |
| tieleman2012 | confirmed | Coursera lecture 6.5, 2012 (no pages/DOI) |
| ioffe2015 | **corrected** | ICML 2015 / PMLR 37:448–456 |
| ba2016 | **corrected** | arXiv:1607.06450, 2016 |
| jacot2018 | left as-is | NeurIPS 2018:8571–8580; no DOI |
| he2016 | left as-is | CVPR 2016:770–778 |
| hochreiter1997 | left as-is | Neural Computation 9(8):1735–1780, 1997 |
| glorot2010 | **corrected** | AISTATS 2010 / PMLR 9:249–256 |
| vaswani2017 | left as-is | NeurIPS 2017:5998–6008 |
| goodfellow2014gan | left as-is | NeurIPS 2014:2672–2680 |
| gukellyxiu2020 | **corrected** | RFS 33(5):2223–2273, 2020 (DOI added) |
| khandani2010 | left as-is | JBF 34(11):2767–2787, 2010 |
| fuster2022 | **corrected** | J. Finance 77(1):5–47, 2022 (DOI added) |
| bai2025 | **corrected** | Ann. Rev. Stat. Appl. 12:209–232, 2025 (DOI added; issue dropped) |

The six standard refs you named (`cybenko1989`, `hornik1991`, `rumelhart1986`,
`robbins1951`, `bottou2018`, `hochreiter1997`) were **actually web-checked this
pass**, not assumed.

---

## TASK 2 — Chapter 2, the 96.3% relabel (`chapter2.tex`, class-thinning ¶, orig. 173–176)
**Before:** "\emph{Current}-to-\emph{Current} transitions are $96.3\%$ of
loan-months (Table~\ref{tab:stateshares}) and individually nearly uninformative."
**After:** "The \emph{Current} state accounts for $96.3\%$ of loan-months
(Table~\ref{tab:stateshares}), and \emph{Current}$\to$\emph{Current} transitions
are individually nearly uninformative."
The sentence is now true (96.3% is the Current-**state** occupancy share, which is
what `tab:stateshares` reports) and the citation is correct. A 7-line
`% TODO (Felipe -- choose): …` was inserted directly above, offering the
alternative (state the true conditional C→C **transition** figure 98.08% and cite
a transition-probability table T2.1; unconditional C→C ≈ 94.4%). The
class-thinning logic that follows is unchanged.

## TASK 3 — cosmetic removals
- **`chapter3.tex` 3–11:** removed the stale "Drop this file into your
  dissertation root…" + "Required packages…" setup header (packages already in
  `main.tex`). The `% Chapter 3 — Data` banner is retained.
- **`chapter1.tex` 85:** removed trailing `%RESULTS-DEPENDENT: confirm phrasing
  after M10–M12.`; prose "…structurally simpler in prime credit." retained.
- **`chapter4.tex` 98–99:** removed the two-line `%RESULTS-DEPENDENT: revisit
  wording after M12 (seed variance, tuning-window sensitivity on k=2019)…`.

## TASK 4 — Appendix C trim suggestions (no prose changed)
- Inserted one `% SUGGESTED TRIM:` comment after each of the nine section labels in
  `appendix3.tex` (`app:dlfoundations`, `sec:dlf-approx`, `sec:dlf-sgd`,
  `sec:dlf-adam`, `sec:dlf-reg`, `sec:dlf-ntk`, `sec:dlf-arch`, `sec:dlf-lineage`,
  `sec:dlf-crosswalk`). Appendix prose is byte-for-byte unchanged.
- **`TRIM_PLAN.md`** created (per-section keep/compress/cut; crosswalk = keep).

## TASK 5 — `TERMINOLOGY_AUDIT.md` created
Baseline: the Deep Learning course reading list
(`docs/bib_refs/AY25-26_MATH_MScMCF_Deep Learning_HT.xlsx` — three textbooks) +
the crosswalk. Few genuine term mismatches (the text already bridges weight
decay/ℓ2/ridge). Primary out-of-scope flag: **NTK** (Jacot 2018, not in any
reading-list text). No prose changed.

## TASK 6 — `REORDER_ANALYSIS.md` + `INTRO_GAPS.md` created
Bidirectional Ch2↔Ch3 dependency map; a modest, defensible case for Data-before-
Methodology (see Needs Felipe). Chapter 1 gaps: literature positioning, explicit
research question, dataset/scale, motivation, outline. Nothing written into the
chapters.

---

## Needs Felipe
1. **Task 2 alternative.** The default fix (Current-state occupancy share) is
   applied. The alternative — quote the conditional C→C transition figure
   (≈98.08%) and cite a transition-probability table — needs a decision; note the
   document does **not** yet print T2.1 as a labelled table, so that route also
   requires adding/labelling that table. (`% TODO` is in place at the spot.)
2. **REORDER recommendation.** Lean **yes** to moving Data (Ch. 3) before
   Methodology (Ch. 2), or take the lighter alternative (make the intro's §design/
   §outline consistent with the current order). Your call — see REORDER_ANALYSIS.md.
3. **`chapter1.tex:5` header note** still reads "Update the results-dependent
   sentences … (marked %RESULTS-DEPENDENT)." Both markers it refers to are now
   removed, so this header line is mildly stale. Left untouched because it is not
   itself one of the two markers you listed — delete it if you want.
4. **`jacot2018` pages.** Verified NeurIPS 2018 pages 8571–8580 were retained.
   Strip them only if you prefer the page-less modern-NeurIPS citation style.
5. **Unverifiable citation fields: none.** `fanniemae2024` is a `@misc` data-source
   page (official SFLP URL), not a journal claim, so it was not re-verified as a
   citation.
6. **Journal-title "The".** `gukellyxiu2020` and `fuster2022` keep the correct
   full titles ("The Review of Financial Studies", "The Journal of Finance") vs the
   shorthand in your ground-truth note; left as the correct official names.

---

# Structural pass (option 2 — no reorder) + intro scaffold — 2026-06-16

Option 2 selected: **chapter order unchanged** (Methodology Ch. 2 before Data
Ch. 3; no `\include` reorder; `chapter3.tex` eq-ref sentences ~222/231 left as
correct backward refs). Scope: insert AI-assisted intro-draft blocks, remove the
NTK appendix section, three small fixes. Surgical; build re-verified.

## Build status (clean)
`latexmk -pdf main.tex` from a cleaned tree: **exit 0**, **58 pages** (was 57),
**0 `!` errors**, **0 undefined references/citations**, **0 bibtex warnings**
(`main.blg` `warning$ -- 0`), **26 bib entries used** (was 28 — `jacot2018` and
`glorot2010` removed, see PART 2).

## §design — reverted; framing fix deferred
An earlier run had rewritten §design (`sec:intro-design`) to resolve the
exploratory-first framing. Per instruction that edit was **reverted to its
original wording** (`git checkout chapter1.tex`); the framing fix is **deferred**
to a later intro revision — see Needs Felipe #2.

## PART A — intro-draft blocks inserted (AI-assisted scaffold, verbatim)
Inserted verbatim; **no existing sentence altered**. Each insertion point is
preceded by the marker `% DRAFT (AI-assisted) — intro additions, to be revised
into final voice.` (placed at **both** locations, since the blocks land in two
sections).

| Block | Placement | Opening words | Cites |
|---|---|---|---|
| 1 Stakes/motivation | new ¶ at END of §object (`sec:intro-object`) | "Agency mortgage-backed securities are among the largest…" | — |
| 2 Literature positioning | new ¶ after block 1, end of §object | "This study sits at the confluence of two literatures…" | `schwartz1989`, `deng2000`, `khandani2010`, `fuster2022`, `gukellyxiu2020`, `sirignano2021` |
| 3 Research question | new ¶ after block 2, end of §object | "Two questions organise the work…" | — |
| 4 Name the data | new ¶ within §design (`sec:intro-design`), existing sentences untouched | "These questions are answered on the Fannie Mae… 3.31 billion loan-months / 57.6M loans / 2000–2025…" | — |

**No new bib keys introduced** — every key cited by the blocks already existed in
`refs.bib` and was already cited elsewhere (entry count went 28→26 purely from the
PART-2 removals, confirming `schwartz1989`/`deng2000` were already in use).

## PART 2 — NTK section removed
- Deleted `\section{Why gradient descent works: non-convexity and the NTK}`
  (`sec:dlf-ntk`) — heading, prose, and its `% SUGGESTED TRIM` comment — from
  `appendix3.tex`.
- Crosswalk table `tab:dlcrosswalk`:
  - **"NTK / global convergence"** row → **removed** (as instructed).
  - **"Convex vs.\ non-convex"** row → **removed** (not re-pointed). The
    optimisation sense of convexity survives in **no** section; the `convex` hits
    in `chapter3.tex` are the geometric "convex decay" of the FICO hazard, a
    different concept — nowhere to re-point.
  - **"Parameter initialisation"** row → **removed** (not re-pointed).
    Weight-initialisation / training-dynamics conditioning survives nowhere; the
    remaining `initialised` mentions (`appendix3.tex` ensembling, `chapter2.tex`)
    refer to *ensemble members independently initialised*, a different idea.
- `refs.bib` removals (now orphaned): **`jacot2018`** (only in NTK prose + NTK
  row) and **`glorot2010`** (only in NTK prose + Parameter-initialisation row).
  `hastie2009` (in the removed convex row) is **kept** — still cited at the
  "$\ell_2$, dropout, ensembles" row.
- Verified: no `\ref{sec:dlf-ntk}` and no `\cite{jacot2018}`/`\cite{glorot2010}`
  remain in any `.tex` source.

## PART 3 — small fixes
1. `chapter2.tex` — removed the 7-line `% TODO (Felipe -- choose): …` comment
   above the class-thinning ¶; the occupancy-share sentence ("The \emph{Current}
   state accounts for $96.3\%$ of loan-months…") is unchanged.
2. `chapter1.tex:4–5` — removed the stale header note "Update the
   results-dependent sentences as Phases 2–3 conclude (marked %RESULTS-DEPENDENT)."
   ; the "Ported from … (June 2026)." line is retained. (Resolves Needs-Felipe #3
   of the prior pass.)
3. `chapter2.tex` — inserted "(ReLU)" once at first introduction of the
   activation: "rectified-linear activations **(ReLU)** $\sigma(u)=\max(u,0)$".

## Needs Felipe
1. **Intro blocks are AI-assisted scaffold** — the four inserted blocks are marked
   `% DRAFT (AI-assisted)` and need revision into your final voice before
   submission.
2. **§design exploratory-first contradiction is deferred (by design).** §design
   still frames the study exploratory-first (Chapter `chap:data`, then models)
   while the kept order is Methodology-before-Data. To be resolved in the later
   intro revision, not this pass.
3. **Glorot / initialisation note dropped with the NTK cut.** The document's only
   parameter-initialisation mention left with the NTK section. If you want the
   topic represented, add a one-line initialisation note elsewhere (e.g.
   `appendix3.tex` §`sec:dlf-reg` or the methodology training paragraph); the
   crosswalk no longer lists it.
4. **Two crosswalk rows removed, not re-pointed.** "Convex vs. non-convex" and
   "Parameter initialisation" were removed because neither concept survives in
   another section (see PART 2). If you'd rather keep either, it needs a
   destination section to point at.
