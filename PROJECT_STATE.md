# PROJECT_STATE.md

State audit of the dissertation as of 2026-06-15. Read-only pass over the LaTeX
source (`writeup/latex/`), the planning markdown (`dev/pipeline_plan/`,
`dev/model_plan/`), the result/figure/run artifacts mirrored in `ssd_mirror/`,
and the compiled `writeup/latex/main.pdf`. Nothing was modified.

Bottom line: **all four chapters are drafted and every number/table/figure they
print traces to a real output file**; the document compiles (`main.pdf`, 50 pp,
no undefined references/citations in `main.log`). What is missing is the *back
matter and front matter* — Conclusions, both appendices, abstract,
acknowledgements, dedication — most of which the drafted prose already
forward-references. The artifacts those empty appendices are supposed to hold
already exist on disk; they have simply not been written into the `.tex`.

Path note: paths under `ssd_mirror/` are the committed mirror of the SSD
`outputs/`–`models/` trees; the dissertation's `\input`/`\includegraphics`
targets live in `writeup/latex/figs/`.

---

## 1. Status per component

| Component | File | Status |
|---|---|---|
| Abstract | `writeup/latex/abstract.tex` | **empty** — only `% TODO: ~1 page` inside an empty `abstract` environment (lines 1–4). Its stated "quarry," `writeup/methodology/methodology_overview.tex`, has an **empty** `abstract` block too (lines 40–42), so there is no draft to port. |
| Acknowledgements | `writeup/latex/acknowlegements.tex` | **empty** — `% TODO: supervisor, family, etc.` (lines 1–3). |
| Dedication | `writeup/latex/dedication.tex` | **empty** — `% TODO — or delete the \include{dedication} line` (lines 1–3). `main.tex` line 70 still `\include{dedication}`. |
| Ch. 1 Introduction | `writeup/latex/chapter1.tex` | **drafted**, complete (modelling object, design, four contributions, outline). Two stale `%RESULTS-DEPENDENT` comment markers (lines 5, 85). |
| Ch. 2 Methodology | `writeup/latex/chapter2.tex` | **drafted**, complete (transition model, sharding, features, rolling design, model hierarchy, evaluation, pool framework, reproducibility). |
| Ch. 3 Data | `writeup/latex/chapter3.tex` | **drafted**, complete (source, eligibility, variables, outcome, summary stats, EDA, Sirignano comparison). Five embedded tables + 7 figures, all present. |
| Ch. 4 Results | `writeup/latex/chapter4.tex` | **drafted**, complete — §4.1 tuning window, §4.2 rolling backtest, §4.3 pool-level + economic translation. All 6 `\input` tables and 9 figures present. One stale `%RESULTS-DEPENDENT` marker (line 98). |
| Conclusions | `writeup/latex/conclusions.tex` | **stub** — chapter heading + `% TODO after Phase 3:` only (lines 1–4). Forward-referenced by Ch. 1 outline (`chapter1.tex` line 115). |
| Appendix A — "Supplementary tables and figures" (`app:supplementary`) | `writeup/latex/appendix1.tex` | **stub** — heading + `% TODO: width-sweep table (M12), full Table A grid, per-window diagnostics` (lines 1–4). **Forward-referenced** by `chapter2.tex` line 266 and `chapter4.tex` line 223. |
| Appendix B — "Reproducibility" (`app:reproducibility`) | `writeup/latex/appendix2.tex` | **stub** — heading + `% TODO: pipeline summary, run-folder/manifest conventions, hardware/software (torch 2.8.0), data snapshot dates` (lines 1–4). **Forward-referenced** by `chapter2.tex` line 453. |

Forward-references that point at content not yet in the document (detail in §4):

- `chapter2.tex` line 266 — width sweep "(Appendix~\ref{app:supplementary})" → Appendix A empty.
- `chapter4.tex` lines 219–223 — four robustness exhibits "(Appendix~\ref{app:supplementary})" → Appendix A empty.
- `chapter2.tex` line 453 — "Appendix~\ref{app:reproducibility} records the computing environment and data snapshot conventions" → Appendix B empty.
- `chapter1.tex` line 115 — "Chapter~\ref{chap:conclusions} concludes" → Conclusions stub.

---

## 2. Claim-vs-evidence

Every number printed in Chapters 3–4 was checked against the artifacts.
**All trace.** The verification below records *where* each headline figure comes
from, then lists the genuine gaps: claims the prose makes that are **not backed by
any exhibit in the document** (even though the underlying data exists on disk).

### 2a. Verified — printed numbers that trace to an output file

- Panel totals "3,312,456,883 loan-months … 57,562,668 loans … 104 … vintages"
  (`chapter3.tex` lines 51–54, 391) → loan-months exact in
  `ssd_mirror/outputs/qa_report.md` line 5 and as the column sum of
  `ssd_mirror/outputs/tables/eda/T1.1_coverage.csv`; 104 files in
  `ssd_mirror/outputs/inventory_summary.md`.
- Seven state shares in Table `tab:stateshares` (`chapter3.tex` lines 414–417:
  Current 96.28 / Prepaid 1.25 / 30DPD 1.15 / 90+DPD 0.98 / 60DPD 0.31 / REO
  0.014 / Foreclosure 0.0067 %) → exact in
  `ssd_mirror/outputs/tables/eda/T1.3_state_distribution_overall.csv`.
- "out of Current … 1.27% to Prepaid and 0.64% to 30DPD" (`chapter3.tex` lines
  426–427) → `T2.1_transition_probs.csv` (0.012724, 0.0064189).
- Continuous-covariate summary Table `tab:covariate-summary` →
  `figs/T1.2_covariate_summary.tex` (source
  `ssd_mirror/outputs/tables/eda/T1.2_covariate_summary_continuous.tex`).
- Market-rate proxy "correlation 0.991, RMSD 0.198 pp, MAD 0.147 pp"
  (`chapter3.tex` lines 280–282) → `ssd_mirror/outputs/tables/eda/mkt_rate_proxy.csv`
  + `F4.1_mkt_rate_validation_series.csv`.
- Incentive-curve peak "2.71% near +1.5 pp" (`chapter3.tex` line 484) →
  `F3.2_prepay_incentive_curve.csv` (rate 0.02709 at value 1.514).
- §4.1 grid: selected config val 0.09602 / test 0.10319, the L2 rescue
  0.09767→0.09630 (`chapter4.tex` lines 56–62) →
  `ssd_mirror/models/nn/dev/grid_summary.json`.
- §4.1 ensemble test 0.10274, avg single 0.10359, best single 0.10307
  (`chapter4.tex` lines 63–68) → `ssd_mirror/models/nn/dev/ensemble_summary.json`.
- §4.1 full-scale depth check "depth 3 … 0.09572 against depth 5's 0.09591"
  (`chapter4.tex` lines 90–91) → `ssd_mirror/models/nn/full/k2015_d3_do0.2_wd1e-05/metrics.json`
  (val 0.095723) and `k2015_d5_do0.2_wd0/metrics.json` (val 0.095912).
- Table A (`figs/table_a.tex`), Table B + by-origin (`figs/table_b.tex`,
  `table_b_origin.tex`), AUC (`figs/auc_table.tex`) → committed copies of
  `ssd_mirror/outputs/tables/loan_level/table_a/table_b/auc.*`. Spot-checks all
  agree: pooled NLL 0.111477 / 0.108249 / 0.101468; "2.9% / 9.0% / two-thirds"
  arithmetic holds; "84.5 M" = 82,817,018+788,362+220,753+686,244 = 84,512,377;
  2023 ensemble tie "+2.2e-5" = 0.069318−0.069296; AUC 0.653→0.748 and
  0.482→0.623 match `auc_table.tex` rows.
- §4.3 char-bucket R²/RMSE (`figs/table_t42.tex`) and CPR/WAL/price errors
  (`figs/table_t51.tex`) → `ssd_mirror/outputs/tables/pool_level/t_m14_pool_counts.*`
  and `t_m15_econ_errors.*`. Every inline value in §4.3 (e.g. 0.575/0.980/0.981;
  −4.62/−1.46; price +18/+25/−1/+31/+48%) matches.
- §4.3 random-pool "RMSE 82.9 → 28.0 → 11.9" at Dec2024 (`chapter4.tex` lines
  253, 288) → `ssd_mirror/models/nn/full/pool_m14_summary.json` (anchor 2025).
- §4.3 "realised 33% CPR" at Dec2019 under-predicted by ~17 pp (`chapter4.tex`
  lines 386–388) → `smm_paths_summary.json` anchor 2020 (realized 0.330;
  empirical 0.152, ensemble 0.138) and CPR-error column of `table_t51.tex`.

### 2b. Gaps — claims with no backing exhibit in the dissertation

These are not fabricated numbers; the data exists. But the *document* asserts
them without printing the exhibit, or backs them only with a JSON/parquet/memo
rather than a thesis table.

1. **The four §4.2 robustness exhibits** (`chapter4.tex` lines 219–223): "seed
   variance … stability of the model ranking … re-tuning sensitivity on the
   k=2019 window … width sweep (Appendix~\ref{app:supplementary})." None are in
   the document — **Appendix A is empty**. The evidence exists and would back
   them: `ssd_mirror/outputs/tables/loan_level/robustness.{json,md}`,
   `width_sweep.{tex,csv,json,md}`, `perm_importance.{md,csv,json}`,
   `ssd_mirror/models/nn/dev/k2019_sensitivity_summary.json`.
2. **The §4.1 / Ch. 2 width sweep** "half/paper/double width sweep … (Appendix
   A)" (`chapter2.tex` line 266) — same empty appendix; data in
   `width_sweep.tex`. Note `writeup/memos/02c_robustness_caveats.md` §2 flags
   this sweep as **dev-scale** generalised to the full-scale net — that caveat
   should appear in the appendix caption and is not yet anywhere in the `.tex`.
3. **Sign-agreement statistic** "sign of the price error agrees with the sign of
   the weighted-average-life error in 99.4–100% of cases" (`chapter4.tex` lines
   373–374). Not in any committed `.tex`/`.csv` table; traces only to
   `ssd_mirror/models/nn/full/pool_m15b_summary.json` /
   `econ_errors_k*.parquet` (per-pool) and `writeup/memos/03_pool.md` §3 /
   `dev/model/M15_NOTES.md` (M15b integrity check). Backed, but by a
   memo/JSON, not a dissertation exhibit.
4. **Random-pool RMSE and "≈2.5× too many prepayments"** (`chapter4.tex` lines
   252–253) trace to `pool_m14_summary.json`, which by design is **not
   tabulated** in the thesis (random-pool R² is "discussed in the text").
   Reader has no in-document table to check these against.

---

## 3. Internal inconsistencies

1. **Current→Current share mislabelled** (genuine). `chapter2.tex` lines
   173–176: "*Current*-to-*Current* transitions are 96.3% of loan-months
   (Table~\ref{tab:stateshares})." The 96.3% is the share of loan-months whose
   **state is Current** (`T1.3` = 96.28%, the figure in `tab:stateshares`), not
   the Current→**Current transition** share. The conditional C→C transition
   probability is 98.08% (`T2.1_transition_probs.csv`), and the *unconditional*
   C→C share of all loan-months is ≈ 0.9628 × 0.9808 ≈ **94.4%**. So the
   sentence quotes the state-occupancy number, labels it as a transition share,
   and cites a state-share table for a transition claim. The class-thinning
   logic it motivates is unaffected, but the number/label/citation triple is
   internally inconsistent.
2. **Stale `refs.bib` header comment** (cosmetic). `writeup/latex/refs.bib`
   lines 1–2 say keys cited are "sirignano2021, deng2000, schwartz1989,
   fanniemae2024," but `chapter2.tex` also cites `dropout` (line 273) and
   `ht1952` (line 188). Both entries **exist** in the `.bib` (lines 34, 43), so
   no citation breaks — the comment is just out of date.
3. **Stale "drop this file in" authoring header** (cosmetic). `chapter3.tex`
   lines 1–12 still carry setup instructions ("add `\include{chapter3}` to
   main.tex … Required packages …") for packages already loaded in `main.tex`
   (lines 14–17). Harmless, but contradicts the assembled state.
4. **Stale `%RESULTS-DEPENDENT` markers.** `chapter1.tex` line 85 and
   `chapter4.tex` line 98 mark wording to be "revisited after M10–M12." M12 is
   complete (`writeup/memos/02c_robustness_caveats.md`), the surrounding prose
   now traces, but the reminders remain in place — flagging text the author
   intended to re-verify.

---

## 4. Open threads

1. **Conclusions chapter unwritten.** `writeup/latex/conclusions.tex` is a stub;
   its own TODO lists what it owes: "headline NLL/AUC results, regime stability,
   pool-level economic translation, limitations (frozen macro, state-level
   granularity, final-revision series), future work (portfolio sorts, macro
   scenarios)." `chapter1.tex` line 115 forward-references it. All inputs exist
   (`writeup/memos/02b_loan_level.md`, `02c_robustness_caveats.md`, `03_pool.md`).
2. **Appendix A (`app:supplementary`) unwritten** but promised twice
   (`chapter2.tex` line 266; `chapter4.tex` line 223). Owes: width-sweep table,
   full Table A grid, per-window diagnostics, seed variance, ranking stability,
   k=2019 re-tuning. Source artifacts all present (see §2b item 1).
3. **Appendix B (`app:reproducibility`) unwritten** but promised
   (`chapter2.tex` line 453). Owes: pipeline summary, run-folder/manifest
   conventions, hardware/software (`torch 2.8.0`), data snapshot dates. Inputs:
   `ssd_mirror/processed/training/{dev,full}/manifest.json`,
   `ssd_mirror/outputs/inventory_manifest.csv`, the `git_commit`/manifest-hash
   fields inside every `metrics.json`, and `dev/pipeline_plan/HOWTO_RUN.md`.
4. **Explicit future-work items, promised but homeless** (their destination,
   Conclusions, is empty):
   - *Portfolio-decile sort* deferred — argued in `chapter2.tex` lines 414–440
     ("Why pricing rather than portfolio sorts") and `writeup/memos/03_pool.md`
     §5; the conclusions TODO lists it but the chapter is a stub.
   - *Macro scenario paths* along the roll-forward (relax the frozen-`t0`
     assumption) — `writeup/memos/03_pool.md` §5 only; not yet in any chapter.
5. **Front matter:** abstract, acknowledgements, dedication all empty (§1). The
   "remaining work is prose" status is confirmed by
   `dev/model_plan/RUN_SCHEDULE.md` (M1–M15 complete; wrap = prose) and
   `dev/model_plan/04_TASKS.md` (M15 Phase-3 gate passed; "memo (M15c) not
   written" per `dev/model/M15_NOTES.md` close).

---

## 5. Suggested next actions (ordered by downstream unblocking)

1. **Write the Conclusions chapter** (`conclusions.tex`). Unblocks the most:
   closes the Ch. 1 forward-reference, and is the only place the two promised
   future-work notes (portfolio sort, macro scenarios) have a home. Inputs are
   ready in memos `02b`/`02c`/`03_pool`; no new computation.
2. **Populate Appendix A** (`appendix1.tex`) from existing M12 artifacts —
   `\input` `width_sweep.tex`, add the robustness/ranking/k=2019 readouts from
   `robustness.md` and `k2019_sensitivity_summary.json`. Removes **five** dangling
   forward-references at once (`chapter2.tex` 266; `chapter4.tex` 223 ×4). Carry
   the dev-scale caveat from memo `02c` §2 into the caption.
3. **Populate Appendix B** (`appendix2.tex`) from the manifests/run-records/env
   above. Low effort, removes the `chapter2.tex` 453 reference; also lets §2b
   item 3 (sign-agreement) and item 4 (random-pool numbers) be cited to a
   reproducibility/methods note if not tabulated.
4. **Write the Abstract** (`abstract.tex`, ~1 page). All results now exist; the
   quarry is empty, so draft fresh from Ch. 1 contributions + Ch. 4 headlines.
5. **Acknowledgements; decide on the dedication.** Fill `acknowlegements.tex`;
   either fill `dedication.tex` or delete `main.tex` line 70 per its own TODO.
6. **Fix the Ch. 2 96.3% C→C labelling** (§3 item 1) — small wording/citation
   correction; either restate as the Current state-occupancy share or quote the
   true C→C transition figure with the right table.
7. **Housekeeping:** refresh the stale `refs.bib` header comment, remove the
   `chapter3.tex` setup header and the two `%RESULTS-DEPENDENT` markers once the
   wording is confirmed.

---

## Needs Felipe to confirm

- **Flattened figure provenance.** The thesis copies `figs/F4.1_scatter_random_prepaid.pdf`
  and `figs/F5.2_price_error_buckets.pdf` are single files, but the source renders
  are per-anchor (`F_m14_scatter_random_prepaid_k2025.pdf`,
  `F5.2_price_error_buckets_k{2015,2019,2020,2023,2025}.pdf`). The prose says F4.1
  is the **Dec2024** anchor (= `k2025`) and F5.2 is the **Dec2022** anchor (=
  `k2023`). I cannot verify from the flattened files which anchor each contains —
  confirm F4.1 = k2025 and F5.2 = k2023.
- **Sign-agreement / random-pool / CPR inline stats** (§2b items 3–4) live only
  in memos + JSON/parquet, not in any dissertation table. Confirm that is
  acceptable for the prose, or whether a small appendix table should back them.
- **Dedication page**: keep (and write) or drop `main.tex` line 70?
- I did **not** open `dev/pipeline_plan/{00–03}` or `dev/model_plan/{00,01,02,03,05}`
  line-by-line beyond what was needed to resolve the above; if you want the
  status cross-checked against the *pipeline* spec's acceptance criteria too, say
  so and I'll extend §1.
