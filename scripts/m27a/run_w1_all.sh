#!/usr/bin/env bash
# W1 scale-up driver — connection-independent (launch under nohup so it survives an SSH/VSCode
# disconnect). Runs the remaining M27a three-arm pipeline at S = $SEQ_TRAIN_N end to end:
#   build the rest of the per-window sequence caches -> GATE on all 11 caches present at the right
#   train_n -> train+score (GRU + FF arms) -> summary tables.
#
# k2015 is built separately (the interactive smoke build); this driver builds 2016 + 2017..2025,
# then waits at the gate until ALL 11 caches (incl. k2015) have meta.json with train_n=$SEQ_TRAIN_N
# before any training starts. The k2015 builder is itself detached, so the whole pipeline survives
# a disconnect. Every trainer skips finished (k,seed)/(k,arm) artifacts, so a mid-run disconnect
# costs at most the in-flight run; re-running this script resumes.
set -euo pipefail

export SEQ_TRAIN_N="${SEQ_TRAIN_N:-10000000}"
REPO=/workspace/repo
PY=/workspace/.venv/bin/python
OUTPUTS=/workspace/dissertation/outputs
CACHE="$OUTPUTS/seq_cache/full"
LOGDIR="${W1_LOGDIR:-/tmp/claude-0/-workspace/8123ac67-065e-4829-bd5c-469cf941fa66/scratchpad}"
WINDOWS="2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025"
GATE_TIMEOUT_S="${W1_GATE_TIMEOUT_S:-7200}"   # backstop wait for the externally-built k2015

cd "$REPO"
mkdir -p "$LOGDIR"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# echo the cache's train_n, or return 1 if the meta is absent/unreadable
meta_train_n() {
  local meta="$CACHE/k$1/meta.json"
  [ -f "$meta" ] || return 1
  "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['train_n'])" "$meta" 2>/dev/null
}
# 0 iff k$1's cache exists at exactly $SEQ_TRAIN_N
meta_ok() {
  local tn; tn=$(meta_train_n "$1") || return 1
  [ "$tn" = "$SEQ_TRAIN_N" ]
}

log "=== W1 driver start  SEQ_TRAIN_N=$SEQ_TRAIN_N  PY=$PY ==="
log "cache root: $CACHE"
log "stage logs under: $LOGDIR (w1_*.log)"

# ---- Stage 1: build the remaining per-window caches (k2015 is built externally) -------------
log "STAGE build: build_window 2016  -> $LOGDIR/w1_build_k2016.log"
"$PY" -u scripts/m27a/build_window.py 2016        >> "$LOGDIR/w1_build_k2016.log" 2>&1

log "STAGE build: build_windows_batch 2017..2025  -> $LOGDIR/w1_build_batch.log"
"$PY" -u scripts/m27a/build_windows_batch.py      >> "$LOGDIR/w1_build_batch.log" 2>&1

log "STAGE build: build_history_superset (no-op if present)  -> $LOGDIR/w1_build_history.log"
"$PY" -u scripts/m27a/build_history_superset.py   >> "$LOGDIR/w1_build_history.log" 2>&1

# Fail fast if any window THIS driver built is missing/wrong (the batch swallows per-window
# failures and exits 0, so verify here rather than hang at the gate for 2h).
for k in 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025; do
  if ! meta_ok "$k"; then
    log "BUILD FAIL: k$k cache absent or train_n != $SEQ_TRAIN_N after build stage (flip-test/base-rate QA?). Aborting before training."
    exit 1
  fi
done
log "build OK: k2016..k2025 all present at train_n=$SEQ_TRAIN_N"

# ---- Gate: ALL 11 caches must be present at $SEQ_TRAIN_N before training starts -------------
# 2016..2025 are confirmed above; wait for the externally-built k2015 (bounded by GATE_TIMEOUT_S).
log "GATE: waiting for all 11 caches at train_n=$SEQ_TRAIN_N (k2015 may be finishing externally)"
deadline=$(( $(date +%s) + GATE_TIMEOUT_S ))
while ! meta_ok 2015; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "GATE TIMEOUT: k2015 cache not ready at train_n=$SEQ_TRAIN_N after ${GATE_TIMEOUT_S}s. Aborting."
    exit 1
  fi
  log "GATE wait: k2015 not ready yet, sleeping 15s"
  sleep 15
done
for k in $WINDOWS; do
  meta_ok "$k" || { log "GATE FAIL: k$k not at train_n=$SEQ_TRAIN_N"; exit 1; }
  log "  gate ok: k$k train_n=$SEQ_TRAIN_N"
done
log "GATE PASS: all 11 caches at train_n=$SEQ_TRAIN_N — starting training"

# ---- Stage 2: train + score (resumable per (k,seed) / (k,arm)) ------------------------------
log "STAGE train: gpu_train_all.py (GRU: 11 windows seed0 + key-window seeds 1,2)  -> $LOGDIR/w1_gpu_train_all.log"
"$PY" -u scripts/m27a/gpu_train_all.py            >> "$LOGDIR/w1_gpu_train_all.log" 2>&1

log "STAGE train: ff_arms.py (ff_base / ff_hist, 11 windows)  -> $LOGDIR/w1_ff_arms.log"
"$PY" -u scripts/m27a/ff_arms.py                  >> "$LOGDIR/w1_ff_arms.log" 2>&1

# ---- Stage 3: summary tables ----------------------------------------------------------------
log "STAGE summary: summary_three_arm.py  -> $LOGDIR/w1_summary_three_arm.log"
"$PY" -u scripts/m27a/summary_three_arm.py        >> "$LOGDIR/w1_summary_three_arm.log" 2>&1

log "STAGE summary: summary_gru.py  -> $LOGDIR/w1_summary_gru.log"
"$PY" -u scripts/m27a/summary_gru.py              >> "$LOGDIR/w1_summary_gru.log" 2>&1

log "=== W1 driver COMPLETE — three-arm + GRU summary tables in the two summary logs above ==="
log "NOTE: run scripts/backup_ssd.sh + git push after this completes (standing rule 5)."
