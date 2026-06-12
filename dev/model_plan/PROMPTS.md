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

### M8a — build + CPU smoke test (no GPU needed)

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

### M10

```
Read dev/model_plan/04_TASKS.md and dev/model_plan/02_LOAN_LEVEL.md. Tasks through M9 are complete and committed; the frozen config is in config.py. Execute task M10 only: full-scale export, then the rolling loop via backtest.py — logit + best NN on all 11 windows, ensemble of 8 on the 5 key windows (2015/2019/2020/2023/2025). Two preliminaries before the loop: (1) the M9 config was selected on the 3M-row dev slice and flagged depth-3-vs-5 as scale-sensitive — re-fit depth-3 and depth-5 (dropout 0.2; add the L2 1e-5 variant if cheap) on the full-scale tuning window k=2015, pick the val-NLL winner, update the frozen config with rationale, and use it for the loop; (2) this box may not be the M8b RTX 4090 — re-measure throughput on the first full-scale fit and use that (not the M8b numbers) to sanity-check the export size you propose before running. Verify every Accept criterion for M10 with evidence, then commit M10. Do not start M11. The SSD is mounted.
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
