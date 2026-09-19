#!/usr/bin/env bash
# m27a/W2 — unattended runner for the transformer 4th-arm rolling evaluation.
# Drives gpu_train_transformer.py's run_one() over all 11 windows (+ key-window seeds 1,2),
# ONE FRESH PROCESS PER RUN so RAM is fully reclaimed between windows (the container has a
# ~58 GiB cgroup cap; a single long-lived process accumulates and gets OOM-killed mid-roll).
# Resumable: run_one SKIPs finished (window,seed) pairs. All scratch is forced onto /workspace
# (the overlay is ephemeral). On full completion: publish JSONs, commit, push, drop done-marker.
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
export PYTHONPATH=/workspace/repo/scripts/m27a${PYTHONPATH:+:$PYTHONPATH}
OUTDIR=/workspace/dissertation/outputs                 # mc.OUTPUTS via /Volumes/SSD Felipe symlink
RESULTS="$OUTDIR/m27a_gpu_runs"                         # where run_one writes k*_xf_s*
TRACKED=artifacts/results/m27a_results_10m               # tracked-in-repo metrics dir
DONE="$OUTDIR/XF_DONE.txt"
mkdir -p "$OUTDIR"

echo "[finish_xf] start $(date -u +%FT%TZ)  repo=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"

# --- canonical (window,seed) plan straight from the trainer module -----------------------
mapfile -t PAIRS < <("$PY" -c "from gpu_train_transformer import WINDOWS,KEY,EXTRA_SEEDS
runs=[(k,0) for k in WINDOWS]+[(k,s) for k in KEY for s in EXTRA_SEEDS]
print('\n'.join(f'{k} {s}' for k,s in runs))")
echo "[finish_xf] ${#PAIRS[@]} (window,seed) runs planned"

# --- run each pair in a FRESH process (k2015 seed0 first; a NaN/OOM aborts the script) ----
# set -e: a killed/failed run stops here — no commit, no done-marker (no false 'SAFE TO STOP').
for p in "${PAIRS[@]}"; do
    k=${p% *}; s=${p#* }
    echo "[finish_xf] >>> k${k} seed${s}  ($(date -u +%T))"
    "$PY" -u -c "from gpu_train_transformer import run_one; run_one(${k}, ${s})"
done

# --- completion = every planned artifact present -----------------------------------------
missing=0
for p in "${PAIRS[@]}"; do
    k=${p% *}; s=${p#* }
    [ -f "$RESULTS/k${k}_xf_s${s}.json" ] || { echo "[finish_xf] MISSING k${k}_xf_s${s}.json" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "[finish_xf] incomplete — not publishing" >&2; exit 1; }
echo "=== ALL TRANSFORMER RUNS COMPLETE ==="

# --- publish results to the repo ---------------------------------------------------------
cp "$RESULTS"/k*_xf_s*.json "$TRACKED"/
git add "$TRACKED"
if git diff --cached --quiet; then
    echo "[finish_xf] no result changes to commit (already published)"
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
"$PY" - "$TRACKED" <<'PYEOF' > "$DONE"
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
PYEOF
{
    echo
    echo "$HEAD_CHECK"
    echo
    echo "SAFE TO STOP"
} >> "$DONE"

echo "[finish_xf] DONE — wrote $DONE"
cat "$DONE"
