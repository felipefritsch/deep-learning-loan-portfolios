#!/usr/bin/env bash
# m27a/W2 — unattended runner for the transformer 4th-arm rolling evaluation.
# Trains+scores scripts/m27a/gpu_train_transformer.py over all 11 windows (+ key seeds),
# then publishes the result JSONs to the repo, commits, pushes, and drops a done-marker.
# All scratch is forced onto the persistent /workspace volume (the container overlay is
# ephemeral and is wiped on pod restart). Resumable: finished (window,seed) runs are SKIPped.
set -euo pipefail

cd /workspace/repo

# --- scratch: keep everything on /workspace (persists across overlay resets) -------------
export TMPDIR=/workspace/tmp
export XDG_CACHE_HOME=/workspace/.cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor
export TRITON_CACHE_DIR=/workspace/.triton
export CUDA_CACHE_PATH=/workspace/.nv
export MPLCONFIGDIR=/workspace/.mpl
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"

PY=/workspace/.venv/bin/python
OUTDIR=/workspace/dissertation/outputs                 # mc.OUTPUTS via /Volumes/SSD Felipe symlink
RESULTS="$OUTDIR/m27a_gpu_runs"                         # where gpu_train_transformer.py writes k*_xf_s*
TRACKED=src/floan/model/m27a_results_10m               # tracked-in-repo metrics dir
SENTINEL="=== ALL TRANSFORMER RUNS COMPLETE ==="
RUNLOG="$OUTDIR/finish_xf.run.log"
DONE="$OUTDIR/XF_DONE.txt"
mkdir -p "$OUTDIR"

echo "[finish_xf] start  repo=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"

# --- train + score (k2015 seed0 runs first → a NaN/OOM aborts early; pipefail propagates) -
# set -e + pipefail: if the python aborts, the script stops here — no commit, no done-marker.
"$PY" -u scripts/m27a/gpu_train_transformer.py 2>&1 | tee "$RUNLOG"

if ! grep -qF "$SENTINEL" "$RUNLOG"; then
    echo "[finish_xf] ERROR: python exited 0 but completion sentinel not found — aborting" >&2
    exit 1
fi

# --- publish results to the repo ---------------------------------------------------------
shopt -s nullglob
xf=( "$RESULTS"/k*_xf_s*.json )
shopt -u nullglob
if (( ${#xf[@]} == 0 )); then
    echo "[finish_xf] ERROR: no k*_xf_s*.json in $RESULTS — aborting" >&2
    exit 1
fi
cp "${xf[@]}" "$TRACKED"/
git add "$TRACKED"

if git diff --cached --quiet; then
    echo "[finish_xf] no result changes to commit (already published) — continuing to verify/push"
else
    git commit -m "m27a/W2: transformer 4th-arm rolling results"
fi

git push origin HEAD

# --- verify remote sync ------------------------------------------------------------------
git fetch origin
HEAD_SHA="$(git rev-parse HEAD)"
ORIGIN_SHA="$(git rev-parse origin/main)"
if [ "$HEAD_SHA" = "$ORIGIN_SHA" ]; then
    HEAD_CHECK="HEAD == origin/main  OK  ($HEAD_SHA)"
else
    HEAD_CHECK="HEAD != origin/main  MISMATCH  (HEAD=$HEAD_SHA origin/main=$ORIGIN_SHA)"
fi

# --- per-window transformer test NLL table + done-marker ---------------------------------
"$PY" - "$TRACKED" <<'PY' > "$DONE"
import json, sys, glob, os
tracked = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(tracked, "k*_xf_s*.json"))):
    d = json.load(open(f))
    rows.append((d["k"], d["seed"], d["test_nll"], d.get("val_nll"), d.get("eff_epochs")))
rows.sort()
print("m27a/W2 — transformer 4th-arm rolling evaluation: per-window test NLL")
print(f"{len(rows)} (window,seed) runs")
print()
print(f"{'window':>7} {'seed':>4} {'test_nll':>12} {'val_nll':>12} {'eff_ep':>7}")
for k, s, tn, vn, ee in rows:
    vn_s = f"{vn:.6f}" if isinstance(vn, (int, float)) else "—"
    ee_s = str(ee) if ee is not None else "—"
    print(f"{k:>7} {s:>4} {tn:>12.6f} {vn_s:>12} {ee_s:>7}")
PY
{
    echo
    echo "$HEAD_CHECK"
    echo
    echo "SAFE TO STOP"
} >> "$DONE"

echo "[finish_xf] DONE — wrote $DONE"
cat "$DONE"
