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

### M4 ✅

```
Read dev/model_plan/00_OVERVIEW.md, 02_LOAN_LEVEL.md, and 04_TASKS.md. Phase 1 (M1–M3) is complete and committed. Execute task M4 only. Before coding, state your assumptions and a short plan. Verify every Accept criterion for M4 with evidence (commands + outputs), then commit M4. Do not start M5. The SSD is mounted.
```

### M5

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M4 are complete and committed. Execute task M5 only. State assumptions and a short plan first. Verify every Accept criterion for M5 with evidence, then commit M5. Do not start M6. The SSD is mounted.
```

### M6

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M5 are complete and committed. Execute task M6 only. State assumptions and a short plan first. Verify every Accept criterion for M6 with evidence (run the unit tests and paste results), then commit M6. Do not start M7. The SSD is mounted.
```

### M7

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M6 are complete and committed. Execute task M7 only. State assumptions and a short plan first. Verify every Accept criterion for M7 with evidence (including the sklearn cross-check numbers), then commit M7. Do not start M8. The SSD is mounted.
```

### M8 — first GPU task

*(Before this prompt: choose the GPU provider, get the box running, upload the M4 dev export. Develop/debug locally with `--smoke` first; only then spend GPU time.)*

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M7 are complete and committed. Execute task M8 only. Build and verify everything locally via the --smoke CPU path first; tell me when the code is ready for the cloud-GPU run and what exact commands to run on the GPU box. Verify every Accept criterion for M8 with evidence, then commit M8. Do not start M9. The SSD is mounted.
```

**→ You: after the GPU session, sync checkpoints back to `models/`, run `backup_ssd.sh`, note the per-epoch throughput (it sets the M10 full-export size).**

### M9

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M8 are complete and committed; the dev-scale GPU pipeline works. Execute task M9 only (depth/regularization grid on the tuning window + Table A; prune the grid sensibly). State the pruned grid you propose before running. Verify every Accept criterion for M9 with evidence, freeze the winning config in config.py, then commit M9. Do not start M10. The SSD is mounted.
```

### M10

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M9 are complete and committed; the frozen config is in config.py. Execute task M10 only: full-scale export, then the rolling loop via backtest.py — logit + best NN on all 11 windows, ensemble of 8 on the 5 key windows (2015/2019/2020/2023/2025). Propose the full-export size from the M8 throughput numbers before running. Verify every Accept criterion for M10 with evidence, then commit M10. Do not start M11. The SSD is mounted.
```

### M11

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M10 are complete and committed. Execute task M11 only (evaluation suite: Tables A & B, AUC, calibration, rate overlay; memos 2a and 2b to writeup/memos/). Verify every Accept criterion for M11 with evidence, then commit M11. Do not start M12. The SSD is mounted.
```

### M12 — Phase-2 gate

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M11 are complete and committed. Execute task M12 only (seed variance, ranking stability, tuning-window sensitivity on k=2019, optional permutation importance). Verify every Accept criterion for M12 with evidence, then commit M12. The SSD is mounted.
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
Read dev/model_plan/04_TASKS.md and dev/model_plan/03_POOL_LEVEL.md. Tasks through M14 are complete and committed. Execute task M15 only (pool memo; the portfolio exercise is optional — ask me before doing it). Verify every Accept criterion for M15 with evidence, then commit M15. The SSD is mounted.
```

**→ You: run `dev/tools/backup_ssd.sh`, then `git push`. Plan complete.**

---

## If something fails mid-task

```
Do not restart the task. Debug the failing Accept criterion in place: show me the exact error, your diagnosis, and the minimal fix before applying it.
```
