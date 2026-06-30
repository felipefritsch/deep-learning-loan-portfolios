#!/usr/bin/env bash
# finish_overnight.sh — one-shot overnight closer for W3b. Owns the SINGLE pod stop (last), so it
# runs run_w3b.sh with W3B_NO_AUTOSTOP=1 (run_w3b finishes its work but does NOT stop the pod), then
# the matched-comparator pass, then commits/pushes, and only on a fully-clean run stops the pod to
# halt GPU billing. ANY failure (GPU wedge, guard, OOM, push error, auth) sets the STOP flag and
# leaves the pod UP for inspection — work already committed between stages is never stranded.
#
# Sequence:
#   1. W3B_NO_AUTOSTOP=1 run_w3b.sh — finish k2015 (reload k2015_s0.pt, train the transformer, price
#      both at the persisted N), skip the four done anchors; run_w3b commits+pushes k2015 itself.
#   2. seq_price_compare on all 5 anchors — matched 150k subsample, H={1,3,6,12}, composition path.
#   3. commit + push the comparator code, matched tables, subsample-ID files (+ econ/json) — verify exit 0.
#   4. stop LAST — only if every guard passed AND the push succeeded.
#
# Launch (survives SSH/VSCode disconnect):
#   nohup scripts/m27a/finish_overnight.sh > /workspace/logs/finish_overnight.out 2>&1 &
# Re-launch after a crash/restart: identical command — run_w3b skips finished anchors, the comparator
# driver skips nothing heavy (its guards re-verify), and the stop only fires on a clean finish.
set -uo pipefail          # NOT -e: failures are managed explicitly so we STOP cleanly, never half-stop.

# ---------------------------------------------------------------------------
# Self-heal preflight — recreate the ROOT symlink (on the wiped overlay /) and the overlay-safe cache
# redirects, mirroring run_w3b.sh, so a relaunch after a pod restart needs no hand-recovery.
# ---------------------------------------------------------------------------
[ -e "/Volumes/SSD Felipe" ] || { mkdir -p /Volumes && ln -s /workspace "/Volumes/SSD Felipe"; }
# Re-point git at the PAT store on the volume — the file persists, but the --global config that
# names it lives on the wiped overlay (same situation as the symlink above), so re-point every pod.
git config --global credential.helper 'store --file=/workspace/.secrets/git-credentials'

REPO=/workspace/repo
PY=/workspace/.venv/bin/python
OUTPUTS=/workspace/dissertation/outputs
BRANCH=w3b-seq-pricing
MC="$REPO/src/floan/model/m27b_mc"            # per-anchor artifacts (sequence + comparator)
ANCHORS="2015 2019 2020 2023 2025"            # all five key anchors (comparator order is COVID-first inside the loop)

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

STOP="$LOGDIR/w3b.STOP"                        # the shared STOP flag (run_w3b uses the same path)
STATUS="$LOGDIR/finish_overnight.STATUS"       # final morning-check summary

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [finish] $*"; }
# fail: record the reason in STOP, emit the morning summary, leave the pod UP, exit non-zero.
fail() {
  echo "$*" > "$STOP"
  { echo "[$(ts)] finish_overnight FAILED: $* | pod left UP for inspection"; } | tee "$STATUS"
  log "FAILURE: $* — leaving pod UP (no stop)"
  exit 1
}

log "=== finish_overnight start  branch=$BRANCH  anchors=[$ANCHORS] ==="
log "overlay: $(df -h / | awk 'NR==2{print $4\" free (\"$5\" used)\"}')"

# ---------------------------------------------------------------------------
# Stage 1 — finish k2015 via run_w3b.sh (no autostop; run_w3b commits+pushes k2015 itself)
# ---------------------------------------------------------------------------
log "STAGE 1: run_w3b.sh (W3B_NO_AUTOSTOP=1) — finish k2015, skip the 4 done -> $LOGDIR/finish_w3b.out"
W3B_NO_AUTOSTOP=1 bash scripts/m27a/run_w3b.sh > "$LOGDIR/finish_w3b.out" 2>&1
log "run_w3b.sh exit=$?"
[ -f "$STOP" ] && fail "run_w3b set STOP: $(cat "$STOP")"
for k in $ANCHORS; do
  [ -f "$MC/k${k}_gru_h.json" ] && [ -f "$MC/k${k}_xf_h.json" ] \
    || fail "sequence anchor k$k incomplete after run_w3b (missing gru/xf h.json)"
done
log "STAGE 1 OK: all 5 sequence anchors complete (gru+xf)"

# ---------------------------------------------------------------------------
# Stage 2 — matched-comparator pass on all 5 anchors (identical 150k subsample, H={1,3,6,12})
# ---------------------------------------------------------------------------
log "STAGE 2: seq_price_compare --anchors $ANCHORS --device cuda -> $LOGDIR/finish_compare.out"
"$PY" -u -m floan.model.seq_price_compare --anchors $ANCHORS --device cuda --seed 0 --target-n 150000 \
      > "$LOGDIR/finish_compare.out" 2>&1
log "seq_price_compare exit=$?"
if grep -qiE "out of memory|CUDA error|CUDA_ERROR|device-side assert|cudaError|device.*unavailable" \
        "$LOGDIR/finish_compare.out"; then
  fail "comparator pass hit a CUDA/OOM error (see finish_compare.out)"
fi

# Per-anchor guard verification — every anchor must have a committed-ready compare.json whose
# population + pool-alignment + H=1-identity guards all passed (the driver writes the json ONLY after
# the guards pass; a guard failure raises before the write, so an absent json == a failed anchor).
if ! "$PY" - "$MC" $ANCHORS <<'PYEOF'
import json, sys
from pathlib import Path
mc, anchors = sys.argv[1], sys.argv[2:]
bad = []
for k in anchors:
    p = Path(mc) / f"k{k}_compare.json"
    if not p.exists():
        bad.append(f"k{k}: no compare.json (anchor failed before write)"); continue
    d = json.loads(p.read_text())
    if d.get("error"):
        bad.append(f"k{k}: {d['error']}"); continue
    if not d.get("priced"):
        bad.append(f"k{k}: nothing priced"); continue
    g = d.get("guards", {})
    if not g.get("pool_alignment", {}).get("pass"):
        bad.append(f"k{k}: pool_alignment not pass ({g.get('pool_alignment', {}).get('reason')})")
    if not g.get("population", {}).get("pass"):
        bad.append(f"k{k}: population guard not pass ({g.get('population', {}).get('reason')})")
    for m, r in g.get("h1_identity", {}).items():
        if not r.get("pass"):
            bad.append(f"k{k}/{m}: H=1 identity failed (max|Δ|={r.get('max_abs')})")
if bad:
    print("COMPARATOR GUARDS FAILED:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("all comparator guards passed for anchors", anchors)
PYEOF
then
  fail "comparator guard verification failed (see finish_compare.out / above)"
fi
log "STAGE 2 OK: all 5 comparator anchors priced + guards passed"

# ---------------------------------------------------------------------------
# Stage 3 — commit + push the comparator code, matched tables, subsample-ID files (+ econ/json)
# ---------------------------------------------------------------------------
log "STAGE 3: commit + push comparator artifacts to $BRANCH"
git add scripts/m27a/finish_overnight.sh \
        src/floan/model/seq_price_compare.py tests/test_seq_price_compare.py 2>/dev/null || true
git add "$MC"/k*_compare_econ.parquet "$MC"/k*_compare.json \
        "$MC"/k*_matched.md "$MC"/k*_matched.parquet "$MC"/k*_subsample.parquet 2>/dev/null || true
if git diff --cached --quiet; then
  log "nothing new staged (already committed on a prior run?) — proceeding to push"
else
  git commit -q -m "W3b: matched-comparator pricing (empirical/logit/FF-current-state/ensemble) at H={1,3,6,12} on the identical seeded 150k subsample; per-anchor matched tables, subsample IDs, and population/pool-alignment/H1 guards"
  log "committed comparator artifacts"
fi
if git push origin "$BRANCH"; then
  log "comparator push OK (exit 0)"
else
  fail "comparator push FAILED — results committed locally but not on the remote, pod left UP"
fi

# ---------------------------------------------------------------------------
# Stage 4 — stop LAST (reached only on a fully-clean run: no STOP, all guards passed, push exit 0)
# ---------------------------------------------------------------------------
{ echo "[$(ts)] finish_overnight COMPLETE: k2015 finished (gru+xf), 5 comparator anchors priced + guards passed + pushed to $BRANCH"; } | tee "$STATUS"
log "STAGE 4: clean run — stopping pod to halt GPU billing (stop, not terminate)"
# runpodctl authenticates from $RUNPOD_API_KEY; source the persisted secret from the volume HERE
# (a post-restart nohup shell will not have inherited it). The secret never appears in logs/args.
[ -f /workspace/.secrets/runpod.env ] && . /workspace/.secrets/runpod.env
if [ -z "${RUNPOD_POD_ID:-}" ]; then
  log "RUNPOD_POD_ID unset (secrets missing?) — cannot self-stop; stop the pod manually"
elif [ -z "${RUNPOD_API_KEY:-}" ]; then
  log "RUNPOD_API_KEY unset (secrets missing?) — cannot self-stop; stop the pod manually"
elif runpodctl stop pod "$RUNPOD_POD_ID"; then
  log "pod stop requested OK — GPU billing halting; /workspace + pod persist for restart"
else
  log "runpodctl stop FAILED (key unauthorised?) — pod left UP, stop it manually"
fi
log "=== finish_overnight DONE ==="
