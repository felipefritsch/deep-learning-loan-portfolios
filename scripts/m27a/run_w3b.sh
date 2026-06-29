#!/usr/bin/env bash
# W3b — Monte-Carlo H>1 sequence-model pricing driver (pod GPU run), hardened for UNATTENDED
# overnight resume. If it dies, just re-launch this same script: it skips finished anchors,
# reloads saved 10M weights instead of retraining, reuses the persisted convergence N, and
# commits+pushes each anchor as it lands. A disk watchdog stops it gracefully before the overlay
# can deadlock at 0 bytes (which killed a prior run).
#
# Launch (survives SSH/VSCode disconnect):
#   nohup scripts/m27a/run_w3b.sh > /workspace/logs/w3b.out 2>&1 &
# Re-launch after a crash/restart: identical command — it resumes.
set -uo pipefail          # NOT -e: we manage per-anchor failures explicitly so we can still
                          # commit completed work and emit a clean summary.

# ---------------------------------------------------------------------------
# Self-heal preflight — recreate the ROOT symlink (lives on the wiped overlay /) and re-export the
# overlay-safe cache redirects, so a re-launch after a pod restart needs no hand-recovery.
# ---------------------------------------------------------------------------
[ -e "/Volumes/SSD Felipe" ] || { mkdir -p /Volumes && ln -s /workspace "/Volumes/SSD Felipe"; }

REPO=/workspace/repo
PY=/workspace/.venv/bin/python
OUTPUTS=/workspace/dissertation/outputs
BRANCH=w3b-seq-pricing
MC="$REPO/src/floan/model/m27b_mc"          # committed per-anchor artifacts (json + parquets)
NFILE="$MC/chosen_n.txt"                     # persisted convergence N (resume skips the sweep)

# Keep ALL scratch on /workspace (the large network volume), NOT the small overlay root /.
# torch/triton/CUDA compile caches + tmp accumulate on / (~30 GB base image) and tip it to 0 bytes,
# after which every write hits ENOSPC and the run dies mid-window.
export TMPDIR="${TMPDIR:-$OUTPUTS/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/workspace/.nv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.mpl}"
export LOGDIR="${LOGDIR:-/workspace/logs}"

cd "$REPO"
mkdir -p "$LOGDIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$MC"

STOP="$LOGDIR/w3b.STOP"                       # presence => halt; contents => reason
STATUS="$LOGDIR/w3b.STATUS"                   # final morning-check summary
rm -f "$STOP"                                 # a fresh launch clears any stale stop flag
ANCHORS="2020 2023 2025 2019 2015"            # priority order, COVID (2020) first to fix N
MIN_FREE_KB=3145728                           # 3 GiB overlay floor (1K blocks)

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [w3b] $*"; }

# ---------------------------------------------------------------------------
# Disk watchdog — stop the run gracefully with margin (don't wait for the 0-byte deadlock).
# Completed anchors are already committed between anchors, so a stop here loses at most the
# in-flight (incomplete) anchor, which a resume re-does.
# ---------------------------------------------------------------------------
watchdog() {
  while [ ! -f "$STOP" ]; do
    local avail; avail=$(df -P / | awk 'NR==2{print $4}')
    if [ -n "$avail" ] && [ "$avail" -lt "$MIN_FREE_KB" ]; then
      echo "low overlay disk: ${avail}K free < ${MIN_FREE_KB}K floor" > "$STOP"
      log "WATCHDOG: overlay below 3 GiB (${avail}K) — signalling stop + killing in-flight driver"
      pkill -f "floan.model.seq_price_mc" 2>/dev/null
      break
    fi
    sleep 120
  done
}
watchdog & WATCH_PID=$!
trap 'kill "$WATCH_PID" 2>/dev/null' EXIT

# echo the persisted/derivable convergence N, or return 1 if not yet known (fresh COVID run).
read_N() {
  if [ -s "$NFILE" ]; then cat "$NFILE"; return 0; fi
  local f n
  for f in "$MC/k2020_gru_h.json" "$MC/k2020_xf_h.json"; do
    if [ -f "$f" ]; then
      n=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['meta']['n_paths'])" "$f" 2>/dev/null) \
        && [ -n "$n" ] && { echo "$n" > "$NFILE"; echo "$n"; return 0; }
    fi
  done
  return 1
}

anchor_done() { [ -f "$MC/k$1_gru_h.json" ] && [ -f "$MC/k$1_xf_h.json" ]; }

commit_anchor() {
  local k=$1
  anchor_done "$k" || { log "k$k not fully complete — nothing to commit"; return; }
  git add "$MC"/k${k}_gru_h.json "$MC"/k${k}_xf_h.json \
          "$MC"/k${k}_gru_econ.parquet "$MC"/k${k}_xf_econ.parquet \
          "$MC"/k${k}_gru_smm_paths.parquet "$MC"/k${k}_xf_smm_paths.parquet \
          "$NFILE" 2>/dev/null
  if git diff --cached --quiet; then log "k$k nothing new staged"; return; fi
  git commit -q -m "W3b: anchor k$k priced (gru+xf, H={1,3,6,12}, N=$(cat "$NFILE" 2>/dev/null))"
  if git push -q origin "$BRANCH"; then log "k$k committed + pushed to $BRANCH"
  else log "k$k commit done but PUSH FAILED (will retry on next anchor / final push)"; fi
}

# ---------------------------------------------------------------------------
# Per-anchor loop — one driver process per anchor (clean resume granularity; ~30s CUDA-init each,
# negligible vs the multi-hour 10M retrains). COVID first with no --n-paths => convergence sweep
# fixes N; every later anchor passes the fixed N (no sweep).
# ---------------------------------------------------------------------------
log "=== W3b start  anchors=[$ANCHORS]  archs=[gru xf]  seed=0  device=cuda  branch=$BRANCH ==="
log "overlay: $(df -h / | awk 'NR==2{print $4" free ("$5" used)"}')   N-file: $([ -s "$NFILE" ] && cat "$NFILE" || echo unset)"

for k in $ANCHORS; do
  [ -f "$STOP" ] && { log "stop flag set — halting before k$k"; break; }
  if anchor_done "$k"; then log "k$k already complete (both arms) — skip"; continue; fi

  alog="$LOGDIR/w3b_k${k}.log"
  if N=$(read_N); then
    log "k$k — pricing with fixed N=$N (retrain only if 10M weights absent)"
    "$PY" -u -m floan.model.seq_price_mc --anchors "$k" --archs gru xf --device cuda --seed 0 \
          --n-paths "$N" 2>&1 | tee "$alog"
  else
    log "k$k (COVID) — convergence sweep will fix N"
    "$PY" -u -m floan.model.seq_price_mc --anchors "$k" --archs gru xf --device cuda --seed 0 \
          2>&1 | tee "$alog"
  fi

  # 1) disk stop takes precedence (watchdog killed the driver) — don't mislabel it a guard failure.
  if [ -f "$STOP" ]; then log "halting at k$k — $(cat "$STOP")"; break; fi
  # 2) guard / OOM / crash: the driver prints '!!! W3b k{k} FAILED: ...' (guards + OOM are caught
  #    per-anchor); a bare Traceback means a top-level crash. Either way: STOP, do not skip ahead.
  if grep -qiE "W3b k${k} FAILED|out of memory|CUDA error|^Traceback" "$alog"; then
    reason=$(grep -iE "W3b k${k} FAILED|out of memory|CUDA error" "$alog" | tail -1)
    echo "guard/OOM/crash at k$k: ${reason:-see $alog}" > "$STOP"
    log "GUARD/OOM/CRASH at k$k — STOP (no skip-ahead). $reason"
    break
  fi

  read_N >/dev/null 2>&1 || true   # persist N right after COVID
  commit_anchor "$k"
  [ -f "$STOP" ] && break
done

# ---------------------------------------------------------------------------
# Completion signal — unambiguous morning-check summary, a verified final push, and (only on a
# fully-clean run) a pod self-stop to halt idle GPU billing.
# ---------------------------------------------------------------------------
done_list=""; missing=""
for k in $ANCHORS; do if anchor_done "$k"; then done_list="$done_list $k"; else missing="$missing $k"; fi; done
n_done=$(echo $done_list | wc -w)

if [ -f "$STOP" ]; then
  summary="W3b STOPPED: $(cat "$STOP") | anchors complete:${done_list:- none} (${n_done}/5) | remaining:${missing:- none}"
elif [ -z "$missing" ]; then
  summary="W3b COMPLETE: all 5 anchors done (${done_list# })"
else
  summary="W3b ENDED: ${n_done}/5 anchors complete (${done_list# }); remaining:${missing}"
fi
{ echo "[$(ts)] $summary"; echo "ship gate (>=3 anchors): $([ "$n_done" -ge 3 ] && echo PASS || echo NOT-YET) (${n_done}/5)"; } | tee "$STATUS"

# Verified final push (catch-all; per-anchor pushes already ran). Capture the exit code: we must
# NOT self-stop if results are not safely on the remote.
if git push origin "$BRANCH"; then push_ok=true; log "final push OK"
else push_ok=false; log "FINAL PUSH FAILED — leaving pod UP so results are not stranded"; fi

# Self-stop ONLY on a fully-clean run: no STOP flag (no guard/OOM/disk-watchdog abort), every anchor
# complete, AND the push succeeded. Use `stop` (not remove/terminate): GPU billing halts while
# /workspace and the pod persist for restart + inspection. Any failure path leaves the pod UP.
if [ ! -f "$STOP" ] && [ -z "$missing" ] && [ "$push_ok" = true ]; then
  if [ -n "${RUNPOD_POD_ID:-}" ]; then
    log "clean full run, all results pushed — stopping pod $RUNPOD_POD_ID to halt GPU billing"
    runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop FAILED (not authed?) — pod left UP, stop it manually"
  else
    log "RUNPOD_POD_ID unset — cannot self-stop; stop the pod manually"
  fi
else
  log "NOT self-stopping (STOP=$([ -f "$STOP" ] && echo yes || echo no) missing=[${missing:- none}] push_ok=$push_ok) — pod left UP for inspection"
fi
log "=== $summary ==="
