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

### M10b — depth check + rolling loop (runs on the POD)

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. M10a (full-scale export) is complete and committed, and the export is uploaded to this box at processed/training/full/ (via the ROOT symlink). This is the GPU half of M10. Two preliminaries before the loop: (1) M9 selected depth 3 on the 3M-row dev slice and flagged it as scale-sensitive — re-fit depth 3 vs depth 5 (dropout 0.2; add the L2 1e-5 variant if cheap) on the full-scale tuning window k=2015, take the val-NLL winner, update the frozen config with rationale, and use it for the loop; (2) this card may differ from M8b's RTX 4090 — re-measure throughput on that first full-scale fit and confirm the loop budget is tractable before launching it. Then run backtest.py over all 11 windows: per-window logit + best single NN everywhere, ensemble of 8 on the 5 key windows (2015/2019/2020/2023/2025). Run long fits inside tmux with per-window checkpointing so preemption or disconnects only cost the current epoch. Verify every Accept criterion for M10 with evidence, then commit as "M10b: rolling loop". Do not start M11.
```

### M11

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M10 are complete and committed. Execute task M11 only (evaluation suite: Tables A & B, AUC, calibration, rate overlay; memos 2a and 2b to writeup/memos/). Verify every Accept criterion for M11 with evidence, then commit M11. Do not start M12. The SSD is mounted.
```

### M12 — Phase-2 gate

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M11 are complete and committed. Execute task M12 only (seed variance, ranking stability, tuning-window sensitivity on k=2019, the width sweep at the selected config — half/paper/double widths, dev scale, appendix artifact — and optional permutation importance). Verify every Accept criterion for M12 with evidence, then commit M12. The SSD is mounted.
```

**→ You: run `dev/tools/backup_ssd.sh`, then `git push`.**

---

## Phase 3 — Pool-level analysis

### M13

```
Read dev/model_plan/00_OVERVIEW.md, 03_POOL_LEVEL.md, and 04_TASKS.md. Phase 2 (M1–M12) is complete and committed. Execute task M13 only (roll-forward harness incl. the first-passage variant and the toy-chain unit test). Verify every Accept criterion for M13 with evidence, then commit M13. Do not start M14. The SSD is mounted.
```

### M14

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/03_POOL_LEVEL.md. Tasks through M13 are complete and committed. Execute task M14 only. Verify every Accept criterion for M14 with evidence, then commit M14. Do not start M15. The SSD is mounted.
```

### M15 — Phase-3 gate

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/03_POOL_LEVEL.md. Tasks through M14 are complete and committed. Execute task M15 only: the economic-translation exercise of 03_POOL_LEVEL §5 (cashflow engine with closed-form unit tests, CPR/WAL/price errors per pool × model × anchor, T5.1 + F5.2) and the pool memo with the contributions paragraph. The portfolio decile exercise is deferred — future-work note only, do not implement it. Verify every Accept criterion for M15 with evidence, then commit M15. The SSD is mounted.
```

**→ You: run `dev/tools/backup_ssd.sh`, then `git push`. Plan complete.**

---

## If something fails mid-task

```
Do not restart the task. Debug the failing Accept criterion in place: show me the exact error, your diagnosis, and the minimal fix before applying it.
```
