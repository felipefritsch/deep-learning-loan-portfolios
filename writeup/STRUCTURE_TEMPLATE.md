# Dissertation Structure Template

Distilled from two high-marking Oxford MCF sample dissertations:

- **Kozyra, *Deep learning approach to hedging*** (2018 Natixis prize) — the **structural** model:
  model-heavy, ~34pp body, disciplined body/appendix split, theory quoted not re-proved.
- **Evaluating Credit Portfolios under IFRS 9*** (2023, industry) — the **subject** model:
  credit/default, macro scenarios, strong backtest→counterfactual→forecast arc. But a
  *cautionary* example on length: ~43pp body, **zero appendices**, everything crammed inline,
  references sloppy. Copy its empirical arc, not its page discipline.

**Page rule (decisive):** the 40-page limit excludes front matter, references, **and
appendices**. So the strategy is: keep the *argument* in ≤40pp; push everything bulky or
reproducibility-related to appendices that don't count.

---

## What to copy from each

**From Kozyra (do this):**
- Body = argument + evidence; appendix = reproducibility. He put all figures/tables/interpretation
  in-body, and offloaded code + bulky sample-path figures to appendices.
- **Quote theory, don't re-prove it.** Foundational results stated as numbered Def/Prop/Thm with
  *precise external locators* (e.g. "see [Sirignano et al.], Prop. X, p. Y"). Only prove your *own*
  derivations in-body. This is the single biggest page-saver for a model-heavy thesis.
- **Standardised result table, repeated.** He compared 6 models with one P/L figure + one
  Mean/Std/VaR table each. You should fix one metrics table format and reuse it everywhere.
- Lean front matter: 1-paragraph abstract + one outline paragraph; no padding.

**From Credit Portfolios (do this):**
- The empirical **validation arc**: build factor/model → fit → **backtest on a held-out window** →
  **counterfactual/stress** (their "Without COVID") → **forward scenarios**, each with its own
  figure. This maps almost perfectly onto your loan-level + pool-level work.
- Domain/regulatory framing up front so a non-specialist examiner can follow.
- A **glossary / notation section** in front matter (excluded from page count) — worth it given
  your seven-state notation.
- **Validation shown, not asserted** (calibration/diagnostic figures, honest "the model under-reacts
  to COVID" admissions woven in).

**From Credit Portfolios (avoid this):**
- Zero appendices → a crammed, over-length body. You have the opposite opportunity.
- Incomplete/placeholder references. Keep `refs.bib` clean.

---

## Proposed chapter + page budget (≈40pp body)

You already have this skeleton (`chapter1–4`, `conclusions`, `appendix1–3`). Suggested allocation:

| Section | Pages | Contents |
|---|---|---|
| **Front matter** (excl.) | — | Title · Acknowledgements · Abstract (1 para) · **Notation table** · Contents · AI declaration |
| **Ch.1 Introduction & contributions** | ~3 | Motivation (mortgage default risk; why deep learning; Sirignano as anchor) · **explicit bulleted contributions** · one outline paragraph |
| **Ch.2 Methodology** | ~9 | Seven-state monthly transition framework · the model (multinomial logit baseline → NN/ensemble) · features & estimation · training protocol *summarised* (full detail → appendix) · **quote** Sirignano + universal-approximation results |
| **Ch.3 Data & exploratory evidence** | ~7 | Fannie Mae panel described compactly · key summary stats + the 4–6 EDA figures that *motivate* features (FICO/LTV/incentive) · full 113-col dictionary → appendix |
| **Ch.4 Loan-level results** | ~9 | Fit · **standardised per-state metrics table** (NLL/AUC/calibration), baseline vs ensemble · calibration figures · by-vintage / by-regime (incl. COVID, 2022–23 rate shock) |
| **Ch.5 Pool-level valuation** (or §4.x) | ~7 | Roll-forward + cashflow engine (closed forms → appendix) · CPR/WAL/price-error results · per-anchor regime exhibit · economic translation |
| **Ch.6 Conclusion** | ~2 | Findings recap · **limitations** (woven earlier too) · future work |
| **References** (excl.) | — | Numeric `[n]`, alphabetised; keep complete |

≈37–38pp → leaves a safety buffer under 40.

## Appendices (excluded from the 40pp — use generously)

- **A. Data dictionary** — 113-column Fannie schema, seven-state mapping logic, derived columns
  (`period_ym`, `orig_ym`, `censored`, `shard`).
- **B. Features & preprocessing** — full feature list, missingness indicators, train-only scaler discipline.
- **C. Training & compute** — hyperparameter tables, ensemble config, early-stopping window, GPU/throughput.
- **D. Additional results** — full per-vintage tables, extra calibration plots, the without-COVID
  counterfactual, by-anchor regime tables.
- **E. Derivations** — cashflow-engine closed forms, any analytic results (with the in-body version citing these).
- **F. Code** — key pipeline/model excerpts + pointer to the repo (Kozyra-style listings).

---

## Conventions checklist

- [ ] Abstract = one paragraph that doubles as a contributions roadmap.
- [ ] One explicit **contributions** list in Ch.1 (Sirignano replication on Fannie + pool-level
      valuation + 2015–2025 regime/COVID analysis).
- [ ] Notation table in front matter (saves redefining the seven states repeatedly in-body).
- [ ] Every borrowed theorem cited with a precise locator; only your own derivations proved in-body.
- [ ] One fixed metrics-table format, reused across every model/regime comparison.
- [ ] Figures: short descriptive captions with the specifics embedded (date range, what each colour is).
- [ ] Limitations stated honestly *where they arise*, not only in the conclusion.
- [ ] Anything bulky or reproducibility-only → appendix, never the body.
