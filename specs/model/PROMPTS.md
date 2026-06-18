# Copy-Paste Prompts for Claude Code (one task per conversation)

> Usage: `/clear` (fresh conversation) → paste the next prompt → review its evidence → confirm the commit happened → move on. After the gate tasks (M3, M12, M15) and every GPU session, **you** run `dev/tools/backup_ssd.sh` and `git push`.

---

## Phase 1 — EDA

### M1 ✅ (done)

```
Read dev/model_plan/00_OVERVIEW.md, 01_EDA.md, and 04_TASKS.md. Execute task M1 only. Before coding, state your assumptions and a short plan. When done, verify every Accept criterion for M1 explicitly, pasting the evidence (commands + outputs), then commit M1. Do not start M2. The SSD is mounted.
```

### M2 ✅ (done)

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/01_EDA.md. Task M1 is complete and committed. Execute task M2 only. Before coding, state your assumptions and a short plan. Verify every Accept criterion for M2 with evidence (commands + outputs), then commit M2. Do not start M2b. The SSD is mounted.
```

### M2b — macro data (uses the standalone spec) ✅ (done)

```
Read dev/model_plan/05_MACRO_DATA.md and execute its tasks MD1–MD4 in order, one at a time, verifying each task's Accept block before advancing to the next. The FMHPI master file is already at /Volumes/SSD Felipe/dissertation/raw/macro/2026-06-10/. Commit after each MD task passes. Do not start M3. The SSD is mounted.
```

*(If the session gets long or stuck, fall back to one MD task per conversation: "Read dev/model_plan/05_MACRO_DATA.md. MD1–MD2 are complete and committed. Execute MD3 only…")*

### M3 — Phase-1 gate ✅ (done)

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/01_EDA.md. Tasks M1, M2, and M2b are complete and committed. Execute task M3 only. Before coding, state your assumptions and a short plan. Verify every Accept criterion for M3 with evidence (commands + outputs), then commit M3. The SSD is mounted.
```

**→ You: run `dev/tools/backup_ssd.sh`, then `git push`.**

---

## Phase 2 — Loan-level models

### M4 - `dev/model/ scaffolding + export✅

```
Read dev/model_plan/00_OVERVIEW.md, 02_LOAN_LEVEL.md, and 04_TASKS.md. Phase 1 (M1–M3) is complete and committed. Execute task M4 only. Before coding, state your assumptions and a short plan. Verify every Accept criterion for M4 with evidence (commands + outputs), then commit M4. Do not start M5. The SSD is mounted.
```

### M5 — Empirical matrix benchmark (all windows)  ✅

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M4 are complete and committed. Execute task M5 only. State assumptions and a short plan first. Verify every Accept criterion for M5 with evidence, then commit M5. Do not start M6. The SSD is mounted.
```

### M6 - Feature pipeline + loader ✅

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M5 are complete and committed. Execute task M6 only. State assumptions and a short plan first. Verify every Accept criterion for M6 with evidence (run the unit tests and paste results), then commit M6. Do not start M7. The SSD is mounted.
```

### M7 - Logit ✅

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M6 are complete and committed. Execute task M7 only. State assumptions and a short plan first. Verify every Accept criterion for M7 with evidence (including the sklearn cross-check numbers), then commit M7. Do not start M8. The SSD is mounted.
```

### M8a — build + CPU smoke test (no GPU needed) ✅

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M7 are complete and committed. Execute the BUILD portion of task M8 only — I do not have the GPU yet, so everything today is CPU-only. Build net.py and train.py per the spec (5-layer net, dropout 0.2, early stopping, checkpointing, run folders) and verify via the --smoke CPU path (<=100k rows): training runs end-to-end without error, loss decreases, a checkpoint + metrics.json + scaler + manifest hash + window id land in the run folder, and training resumes correctly from a checkpoint (kill and restart mid-run to prove it). Defer to the GPU session ONLY the dev-export fit and its Accept criteria (val NLL improves on logit; no OOM at scale) — list these explicitly as deferred at the end. Also prepare and show me the exact commands I will run on the GPU box tomorrow (env setup, data path assumptions, the training command). Commit as "M8a: NN training pipeline + CPU smoke test". Do not start M9. The SSD is mounted.
```

### M8b — the GPU run (after the box is set up and the dev export uploaded) ✅

> **Result (RTX 4090, AMP):** k=2015 dev fit, best epoch 20 of 26 (early-stopped). **val NLL 0.097671 < logit 0.100081** (Δ +0.00241, beats) — test NLL 0.104845 < logit 0.109430. No OOM (≤0.6 GB / 24 GB). Throughput **~1.55 M train rows/s, ~2.0 s train / 2.3 s per epoch** over the 3.15 M-row k=2015 train slice → M10 sizing: a 100 M-row train slice ≈ 1.1 min/epoch, ≈ 27 min per single-net window fit. Surfaced + fixed a cuda-only resume bug (`map_location` moved the CPU RNG state to GPU); resume now reproduces the uninterrupted trajectory bit-for-bit on GPU.

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. M8a (build + CPU smoke test) is complete and committed — do NOT rebuild or rerun the smoke test. The GPU box is running and the M4 dev export is uploaded; this environment IS the GPU box [adjust if running Claude Code locally instead: "I will run commands on the box and paste outputs"]. Execute the deferred GPU portion of M8 only: fit the 5-layer net on the dev export (tuning window k=2015, early stopping on its val slice). Verify the remaining M8 Accept criteria with evidence: no OOM, val NLL improves on the M7 logit (paste both numbers), complete run folder, resumable from checkpoint. Record per-epoch wall-clock throughput (rows/sec and min/epoch) — it sets the M10 full-export size. Commit as "M8b: dev-scale GPU fit". Do not start M9.
```

**→ You: after the GPU session, sync checkpoints back to `models/`, run `backup_ssd.sh`, note the per-epoch throughput.** ✅

### M9 - Depth/regularization grid (tuning window) + Table A ✅

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M8 are complete and committed; the dev-scale GPU pipeline works. Execute task M9 only (depth/regularization grid on the tuning window + Table A; prune the grid sensibly). State the pruned grid you propose before running. Verify every Accept criterion for M9 with evidence, freeze the winning config in config.py, then commit M9. Do not start M10. The SSD is mounted.
```

### M10a — full-scale export (runs on the MAC — needs the panel lake on the SSD) ✅

> **Result:** `full` variant = **64-shard (25% of loans) train block + 12-shard (≈4.7%) eval block**, `p_keep=0.05` (config.VARIANTS now carries explicit `train_shard_lt`/`eval_shard_lt`). **train_pool 74,419,387 rows** (2.0 GB, weighted 814,236,181 — reconciles: 38.94 M cur→cur ×20 + 35.48 M ×1.0) → k=2015 tuning slice **33.5 M** (10.6× the dev 3.15 M, so M10b's depth-3-vs-5 re-check has real signal); **eval_pool 90,515,851 rows** (755 MB, ~7.5 M unthinned rows/test-year). **Dir total 2.7 GB** to rsync. Held the eval block to 12 shards (not the design-intent 51 ≈ 385 M / 3.2 GB) — NLL SE ~1e-4 is already ~20× below the NN-vs-logit gap, so 20% buys no CI and costs 11×+ensemble scoring. All M4 Accept checks PASS at full scale (per-label-year reconcile, eval loan-disjoint+unthinned, k=2015/k=2025 masks, macro lags+fallbacks null-free, idempotent re-run reproduces shard hashes bit-for-bit). Committed "M10a: full-scale export".

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M9 are complete and committed. This is the Mac-side half of task M10: the FULL-SCALE EXPORT ONLY (the GPU box cannot see the panel lake; the rolling loop runs there next session). Propose the full-export size (~50–100M train-pool rows) using the M8b throughput numbers, state assumptions, then build it with export.py into processed/training/full/ with its manifest. Verify the M4-style Accept criteria on this export (row counts reconcile per label year, eval pool unthinned + loan-disjoint, window masks correct, macro joins + fallback indicators present, idempotent re-run). Report the resulting directory size (it gets rsynced to the GPU box) — if the full eval pool makes upload or scoring intractable, propose a smaller shard block with rationale. Commit as "M10a: full-scale export". Do not run the loop. The SSD is mounted.
```

**→ You: deploy a pod, update `~/.ssh/config`, re-create the symlink, `git pull` + activate venv on the pod, then rsync `processed/training/full` up (same command as the dev upload with `dev` → `full`).**

### M10b — depth check + rolling loop (runs on the POD) ✅

> **Result (RunPod RTX A4500 20 GB — not M8b's 4090):** First full-scale fit ran at ~0.35 M rows/s (GPU ~10 %, CPU/transfer-bound) → ~40 h loop. Added a **GPU-resident fast path** to `train.py` (`--gpu-resident`, cuda-only; whole split on-GPU, indexed on-device) → **0.80 M rows/s, 2.3×**, with **bit-identical per-epoch NLLs** vs the naive path (epoch 0: train 0.147757 / val 0.097586 on both) — a post-M9 train.py change made a non-event by that equivalence. **Depth re-check** on full k=2015 (33.5 M, 10.6× dev): **depth 3 still beats depth 5** (val 0.095723 vs 0.095912, gap +1.66e-4 ≫ the within-d3 L2 noise) — froze `NN_SELECTED = d3/dropout0.2/L2 1e-5` (val argmin) with rationale. **Logit equivalence receipt**: the loop's new embedding-sum logit reproduces the sklearn-validated M7 logit to **<1e-4 across the L2 grid** (wd=0 anchor 5.0e-5) after zero-init embeddings (`models/logit/logit_equiv_receipt.json`). **Loop** (tmux, per-epoch checkpointing, ~20 h): 11×{logit, NN} + 5×ensemble. **NN beats logit 11/11 windows** (mean test NLL 0.1026 vs 0.1094, all regimes incl. COVID-2020 & the 2022–23 spike); **ensemble ≤ mean single-member 5/5** (Fig-7 variance reduction; beats deployed single net 4/5, k2023 a +2.2e-5 seed-noise tie). **All M10 Accept PASS** (`logs/m10b/verify_final2.log`): run folders, per-window val-only early stopping, scaler/vocab locality, ensemble vs single, §7 base-rate + structural impossible-transition QA (mass <1e-3, tracking realized — the panel's rare `dpd_90plus→REO` jumps noted). Full write-up: `dev/model/M10b_NOTES.md`. Committed "M10b: rolling loop".

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. M10a (full-scale export) is complete and committed, and the export is uploaded to this box at processed/training/full/ (via the ROOT symlink). This is the GPU half of M10. Two preliminaries before the loop: (1) M9 selected depth 3 on the 3M-row dev slice and flagged it as scale-sensitive — re-fit depth 3 vs depth 5 (dropout 0.2; add the L2 1e-5 variant if cheap) on the full-scale tuning window k=2015, take the val-NLL winner, update the frozen config with rationale, and use it for the loop; (2) this card may differ from M8b's RTX 4090 — re-measure throughput on that first full-scale fit and confirm the loop budget is tractable before launching it. Then run backtest.py over all 11 windows: per-window logit + best single NN everywhere, ensemble of 8 on the 5 key windows (2015/2019/2020/2023/2025). Run long fits inside tmux with per-window checkpointing so preemption or disconnects only cost the current epoch. Verify every Accept criterion for M10 with evidence, then commit as "M10b: rolling loop". Do not start M11.
```

### M11 ✅

> **Result:** `evaluate.py` scores all four model families (empirical / logit / best NN /
> 8-net ensemble) through one path on the frozen unthinned test slices. **All M11 Accept
> PASS** (`models/nn/full/evaluate_summary.json`, 17.8 min): every model on identical
> per-window rows (row-count == committed metrics + byte-identical scaler/vocab per window),
> each of **84,512,377** pooled test rows appearing exactly once (Σ window rows == unique
> (loan, period_ym)), figures regenerate. **Table B** (`outputs/tables/loan_level/table_b.*`):
> NN beats logit **11/11 windows**; pooled **NN 0.101468 < logit 0.108249 < empirical
> 0.111477** (NN −9.0 % vs floor, logit only −2.9 % → the nonlinearity is ~⅔ of the achievable
> gain). By origin, the NN beats the floor in **all 4** origins while the logit is at/below the
> floor in every delinquent origin. **AUC** (`auc.*`): NN's edge concentrates in prepayment
> (current→prepaid 0.653→0.748) and the deep-delinquency events the logit can't rank
> (dpd_90+→foreclosure 0.482→0.623). Calibration + rate-overlay figures confirm the NN tracks
> the prepay waves and is better-calibrated; both models miss only the 2020 COVID dpd_30 spike.
> Ensemble ≤ mean single member 5/5, ≤ deployed single net 4/5 (2023 a seed-noise tie).
> **Note (perf):** the AUC builder re-slices the full [84.5M×7] prob matrix per cell (~14 min
> of the 17.8); numbers verified correct — the vectorized `_auc_one_vs_rest` (np.unique
> average-ranks) matches the original Python-loop AUC **and** sklearn `roc_auc_score` to **|Δ|
> = 0.0** on a 386,760-row tie-bearing slice. A `# TODO(perf)` marks the re-slice for a later
> speedup. Memos `02a_benchmarks.md` + `02b_loan_level.md` committed. Committed "M11:
> evaluation suite + memos 2a/2b".

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M10 are complete and committed. Execute task M11 only (evaluation suite: Tables A & B, AUC, calibration, rate overlay; memos 2a and 2b to writeup/memos/). Verify every Accept criterion for M11 with evidence, then commit M11. Do not start M12. The SSD is mounted.
```

### M11b — Thesis draft sync (run on the MAC, SSD mounted) ✅

```
Read dev/model_plan/04_TASKS.md (task M11b). Tasks through M11 are complete and committed, and models/ + outputs/ are synced back to the SSD. Execute M11b only: (1) verify every number quoted in writeup/latex/chapter4.tex §4.1 against the run-folder metrics.json and table_a artifacts, correcting any discrepancy and reporting the diff; (2) copy the EDA and loan-level figure/table artifacts from outputs/ into writeup/latex/figs/ and replace the corresponding \figplaceholder/\tabplaceholder blocks in chapters 3–4 with real \includegraphics/\input (leave the Phase-3 placeholders). Recompile main.tex with latexmk and verify zero undefined references. Commit M11b. The SSD is mounted.

Additionally, in §4.2 (the rolling backtest): the eight-network ensemble was fit only on the 5 key windows (2015/2019/2020/2023/2025), not all 11. So do NOT report a pooled all-years ensemble figure alongside the pooled network/logit numbers — that pools incomparable row coverage. Present the ensemble-vs-network comparison per-window on the windows where the ensemble exists (or pooled over exactly those matched 5 windows), while the pooled all-years comparison stays restricted to logit vs the single network, which exist on all 11. Adjust the Table B and AUC-table captions/text accordingly if they currently imply a pooled ensemble number.
```

### M12 — Phase-2 gate ✅

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M11 are complete and committed. Execute task M12 only (seed variance, ranking stability, tuning-window sensitivity on k=2019, the width sweep at the selected config — half/paper/double widths, dev scale, appendix artifact — and optional permutation importance). Verify every Accept criterion for M12 with evidence, then commit M12. The SSD is mounted.
```

**→ You: run `dev/tools/backup_ssd.sh`, then `git push`.**

---

## Phase 3 — Pool-level analysis

### M13 ✅

```
Read dev/model_plan/00_OVERVIEW.md, 03_POOL_LEVEL.md, and 04_TASKS.md. Phase 2 (M1–M12) is complete and committed. Execute task M13 only (roll-forward harness incl. the first-passage variant and the toy-chain unit test). Verify every Accept criterion for M13 with evidence, then commit M13. Do not start M14. The SSD is mounted.
```

### M14 ✅

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/03_POOL_LEVEL.md. Tasks through M13 are complete and committed. Execute task M14 only. Verify every Accept criterion for M14 with evidence, then commit M14. Do not start M15. The SSD is mounted.
```

### M15 — Phase-3 gate ✅

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/03_POOL_LEVEL.md. Tasks through M14 are complete and committed. Execute task M15 only: the economic-translation exercise of 03_POOL_LEVEL §5 (cashflow engine with closed-form unit tests, CPR/WAL/price errors per pool × model × anchor, T5.1 + F5.2) and the pool memo with the contributions paragraph. The portfolio decile exercise is deferred — future-work note only, do not implement it. Then swap the remaining Phase-3 placeholders in writeup/latex/chapter4.tex §4.3 for the real artifacts (copy into writeup/latex/figs/, replace \figplaceholder/\tabplaceholder, recompile main.tex cleanly — requires the artifacts synced to the SSD, so run this part on the Mac if this session is on the GPU box). Verify every Accept criterion for M15 with evidence, then commit M15. The SSD is mounted.

Additionally: (1) reuse M14's pools — the same 5 key anchors (2014-12, 2018-12, 2019-12, 2022-12, 2024-12), the same characteristic and random pool schemes, the same model set (empirical/logit/nn/ensemble), and the saved per-pool artifacts under outputs/tables/pool_level/ (pools_random_k*.parquet); do NOT rebuild pools differently — so T5.1's price errors trace pool-for-pool to T4.2's counts. (2) Report CPR/WAL/price errors per pool × model × ANCHOR; do not collapse the headline price-error reduction into a single pooled number that averages over the COVID-anchor inversion — present it per-anchor or with the regime spread. (3) Before writing the memo, read writeup/memos/02c_robustness_caveats.md §4 and dev/model/M14_NOTES.md §3; keep the price-error interpretation consistent with that shock-type narrative (forbearance shock 2020 → covariate models misled, empirical robust; rate/prepay shock 2022-24 → covariates essential), and attribute any Dec-2019 pricing degradation to the frozen-t0 macro assumption (03 §3) — document it, do not tune it away. Build the contributions paragraph on this regime-conditional value-of-nonlinearity finding, coherent with the §4.2 loan-level and §4.3 pool-level story.
```

**→ You: run `dev/tools/backup_ssd.sh`, then `git push`. Plan complete.**

---

### M15 — split into sub-sessions (use these instead of the single block above if running M15 across several `/clear`s)

> All Mac, no pod. Run M15a → M15b → M15c, each in its own conversation, verifying/committing before the next. The frozen-t0 macro caveat and the per-anchor (never pooled-over-COVID) discipline propagate through all three.

#### M15a — cashflow engine + monthly pool SMM paths (internal parallel block) ✅

```
Read dev/model_plan/04_TASKS.md (M15) and dev/model_plan/03_POOL_LEVEL.md §5. Tasks through M14 are complete and committed; this is the first of three M15 sub-sessions — execute M15a only, do not start M15b. Two pieces, run in parallel:

(A) SMM-path regeneration [CPU-heavy — launch in the BACKGROUND first]: pool.py's roll_forward currently returns only the 12-month distribution; extend it to capture the per-step prepayment hazard and aggregate UPB-weighted to a pool monthly SMM path SMM_P(h), h=1..12, for models {empirical, logit, ensemble} and the realized analogue from the panel, over the SAME 5 key anchors and the SAME characteristic + random pool schemes as M14 (reuse outputs/tables/pool_level/pools_random_k*.parquet for membership; do NOT rebuild pools). Save to outputs/tables/pool_level/smm_paths_k*.parquet. Run over all 5 anchors on CPU as a background job (nohup, log to logs/m15/smm_paths.log) — launch it BEFORE coding (B), and print the exact launch command + how to confirm it is running and the log is growing.

(B) Cashflow engine [code while (A) runs]: add the ~100-line deterministic level-pay pass-through to pool.py per §5.2 — given pool WAC/WAM/UPB + a monthly SMM vector (constant extrapolation past h=12), produce scheduled amortization + prepayments and compute WAL and price (discount at WAC minus a fixed servicing spread, same curve for every model so price differences isolate the prepayment model). Write hermetic unit tests against the two closed forms — zero-prepay = annuity, constant-SMM = closed-form survival schedule — to ~1e-8; these need no real SMM paths, so test on synthetic vectors independent of (A).

When (A) finishes, sanity-check the SMM paths (CPR in plausible ranges, model vs realized) and confirm the engine tests pass, then commit as "M15a: cashflow engine + monthly pool SMM paths". Do not compute error metrics or T5.1/F5.2 (that is M15b). The SSD is mounted.
```

#### M15b — CPR/WAL/price errors + T5.1 + F5.2 ✅

```
Read dev/model_plan/04_TASKS.md (M15) and 03_POOL_LEVEL.md §5. M15a (cashflow engine + monthly SMM paths) is complete and committed. Execute M15b only. Using the engine and the saved smm_paths_k*.parquet, compute per pool × model × anchor: CPR error (pp), WAL error (months), price error (per 100 face) — model-implied vs realized-SMM valuation, on the SAME 5 anchors / pools as M14. Produce T5.1 (mean |error| and signed bias by model × outcome × anchor; the headline is the ensemble-vs-logit reduction in mean |price error|) and F5.2 (price error by characteristic bucket — expect the linear model's errors to concentrate in high-incentive cells). Report the headline price-error reduction PER ANCHOR — do NOT collapse to a single pooled number that averages over the COVID-anchor inversion; expect the same regime signature as the counts (ensemble beats logit off-COVID, inverts at Dec-2019 under frozen-t0 macro). Verify T5.1/F5.2 exist for ensemble vs logit at ≥3 regime-spanning anchors with the headline reduction stated, then commit as "M15b: CPR/WAL/price errors + T5.1 + F5.2". Do not write the memo (M15c). The SSD is mounted.
```

#### M15c — pool memo + §4.3 LaTeX swap (Phase-3 gate) ✅

```
Read dev/model_plan/04_TASKS.md (M15), 03_POOL_LEVEL.md §5–6, writeup/memos/02c_robustness_caveats.md §4, and dev/model/M14_NOTES.md §3. M15a and M15b are complete and committed. Execute M15c only. (1) Write writeup/memos/03_pool.md (~4 pages): roll-forward method with the covariate-freezing assumption stated; F4.1, T4.2 (incl. the by-anchor regime exhibit), F4.3, T5.1, F5.2; the frozen-t0 macro caveat paragraph; and a closing contributions paragraph naming what is original vs Sirignano et al. (prime conforming credit box; 2015–2025 test regimes incl. COVID forbearance and the 2022–23 rate shock; the rolling-window protocol; the CPR/WAL/price translation) — keep the narrative consistent with 02c §4 and M14_NOTES §3 (shock-type regime dependence). The portfolio-decile exercise is deferred: future-work note only, do not implement it. (2) Swap the remaining Phase-3 placeholders in writeup/latex/chapter4.tex §4.3 for the real artifacts (M11b pattern: copy into writeup/latex/figs/, replace \figplaceholder/\tabplaceholder, recompile main.tex with latexmk, zero undefined references). Verify every M15 Accept criterion with evidence (engine matches both closed forms to ~1e-8; T5.1/F5.2 for ensemble vs logit at ≥3 anchors with the headline |price-error| reduction; memo committed with all exhibits + caveat + contributions; §4.3 placeholders swapped and main.tex recompiles clean), then commit as "M15c: pool memo + §4.3 swap (Phase-3 gate)". The SSD is mounted.

Additionally, in the memo prose: (1) wherever you state a headline ensemble-vs-logit price-error reduction (%), also give the absolute mean |price error| (per 100 face) for both models at that anchor — the % alone is ambiguous, so pair it with the absolute (read the real figures from T5.1) to make the economic magnitude concrete. (2) Add a short paragraph on why price is not a monotone function of average CPR: price depends on prepayment timing (the full SMM path + WAL), not just mean speed — note that the CPR-vs-price sign divergences occur only where |CPR error| is tiny. This pre-empts the "why isn't price just CPR?" question and shows the cashflow engine is doing real work.
```

**→ You (after M15c): run `dev/tools/backup_ssd.sh`, then `git push`. Plan complete.** ✅

---

## If something fails mid-task

```
Do not restart the task. Debug the failing Accept criterion in place: show me the exact error, your diagnosis, and the minimal fix before applying it.
```
