# REORDER_ANALYSIS.md — Task 6 (analysis only; nothing reordered or rewritten)

Question: should **Data (Chapter 3)** come before **Methodology (Chapter 2)**?
Current order is Methodology → Data. Line numbers below are from the current
(post-fix) source.

---

## (a) Cross-reference dependency map (both directions)

Cross-chapter `\ref`/`\eqref` only. Shared `\citet{sirignano2021}` calls are
external (to a paper, not a chapter) and are excluded.

### Methodology → Data  (`chapter2.tex` references labels defined in `chapter3.tex`)

| chapter2.tex line | Reference | What it needs from Data | In current order |
|---|---|---|---|
| 17 | `\ref{sec:data-outcome}` | "the seven states defined in Section …" — the whole state space the model predicts | **forward ref** (Data is later) |
| 98 | `\ref{sec:data-variables}` | the loan-level field list the features are built from | **forward ref** |
| 99 | `\ref{subsec:data-external}` | the external macro covariates | **forward ref** |
| 182 | `\ref{tab:stateshares}` | the 96.3% Current-state share motivating class thinning | **forward ref** |

Four references, all currently **forward** — including the methodology's opening
sentence (line 17), which defines the supervised object in terms of a state space
not introduced until the next chapter.

### Data → Methodology  (`chapter3.tex` references labels defined in `chapter2.tex`)

| chapter3.tex line | Reference | What it needs from Methodology | In current order |
|---|---|---|---|
| 222 | `\eqref{eq:incentive}` | the refinancing-incentive formula | **backward ref** (Methodology is earlier) |
| 231 | `\eqref{eq:mtmltv}` | the mark-to-market LTV formula | **backward ref** |
| 368 | `\ref{sec:meth-model}` | why the matrix is 4×7 not 7×7 | **backward ref** |
| 376 | `\ref{subsec:meth-empirical}` | the empirical-matrix floor benchmark | **backward ref** |
| 391 | `\ref{sec:meth-design}` | importance-weighted thinning | **backward ref** |
| 392 | `\ref{subsec:meth-empirical}` | Laplace smoothing of the empirical benchmark | **backward ref** |

Six references, all currently **backward**.

**Reading:** the coupling is genuinely bidirectional. Today Methodology
forward-points into Data 4× (the model is described before its data substrate);
Data back-points into Methodology 6× (cleanly). Neither chapter is self-contained.

---

## (b) Sentences that assume the other chapter came first

### If the order is swapped (Data → Methodology), these `chapter3.tex` sentences become forward refs and would read awkwardly / want rework
- **`chapter3.tex:222`** — "…the prevailing market mortgage rate from which the *rate incentive* of Equation~\eqref{eq:incentive} is computed." Would reference a formula defined a chapter later. **Rework or relocate** (e.g., state the incentive informally here and keep the formula in Methodology, or define the two feature equations in Data).
- **`chapter3.tex:231`** — "…the mark-to-market loan-to-value ratio of Equation~\eqref{eq:mtmltv} … is constructed." Same issue. **Rework or relocate.**
- **`chapter3.tex:368,376,391,392`** — the 4×7 matrix, empirical-benchmark, thinning, and Laplace-smoothing asides become forward refs but read fine as "as the methodology will set out"; **no rewrite strictly required.**

These are the only two substantively awkward spots (222, 231). Everything else
flips automatically.

### In the current order (Methodology → Data), these `chapter2.tex` sentences already forward-ref Data
- **`chapter2.tex:16–17`** — the methodology's first sentence defines the object "over the seven states defined in Section~\ref{sec:data-outcome}" before the reader has met them.
- **`chapter2.tex:97–99`** — features are "described in Sections~\ref{sec:data-variables} and~\ref{subsec:data-external}" before those variables exist for the reader.
- **`chapter2.tex:174–182`** — class thinning cites the state-share table (`tab:stateshares`) that lives in the later chapter.

So the current order is **not** free of forward references; it has 3 (incl. the
opening sentence). Swapping trades these 3 away and introduces 2 new ones (222,
231).

---

## (c) Bottom line (not defaulting to "no change")

**There is a modest, defensible case for moving Data before Methodology**, and it
turns on narrative rather than mechanics:

1. **The introduction already frames the work as exploratory-first.** `chapter1.tex:58–63` (§design): "*First*, an exploratory phase establishes the dataset's scope and produces model-free evidence of nonlinearity (Chapter~\ref{chap:data}). *Second*, loan-level models are estimated (Chapters~\ref{chap:methodology} and~\ref{chap:results})." Yet the document orders Methodology (ch2) before Data (ch3), and §outline (`chapter1.tex:107–111`) restates methodology-first. The chapter order contradicts the stated phase order.
2. **The EDA's rhetoric needs to precede the model.** `chapter3.tex:462–463`: "*Before any model is fitted*, the case that nonlinear modelling is needed can be made by counting alone." The argument "evidence of nonlinearity ⇒ therefore nonlinear models" is strongest when the evidence (Data/EDA) comes before the model hierarchy (Methodology).
3. **Convention.** Empirical-finance theses — and Sirignano et al. itself — typically present data before methodology.

**Cost of swapping is small and enumerable:** reword/relocate exactly two
sentences (`chapter3.tex:222, 231`, the incentive and mtm-LTV equation
references); the methodology's 3 current forward refs become clean backward refs;
the 4 remaining Data→Methodology asides become innocuous forward refs.

**Recommendation:** lean **yes — move Data (Ch. 3) before Methodology (Ch. 2)**,
because it removes the contradiction with the intro's own phase ordering and
strengthens the "evidence → model" narrative, at a cost of two reworded
sentences. This is your call, not an automatic verdict — flagged in CHANGELOG.md
"Needs Felipe."

**Lower-effort alternative (if you prefer not to reorder):** keep the order and
instead make the **intro consistent with it** — revise `chapter1.tex:58–63` and
107–111 so §design and §outline both describe methodology-first, removing the
"exploratory-first" phrasing. That is a 1–2 sentence intro edit instead of a
chapter swap, and it resolves the only hard inconsistency. It does **not** fix
the methodology's opening forward-dependence on the state definitions.
