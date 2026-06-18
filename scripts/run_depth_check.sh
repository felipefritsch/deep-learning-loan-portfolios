#!/bin/bash
# M10b preliminary (1): full-scale k=2015 depth re-check — depth 3 vs 5 at dropout 0.2,
# plus the L2=1e-5 variant at each depth (M9 found a small L2 rescues the deeper net).
# Resumable: a finished cell (metrics.json) is skipped by train.py; an interrupted cell
# resumes from its last epoch checkpoint. Resident GPU path (--gpu-resident).
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
C="--variant full --k 2015 --device cuda --amp --gpu-resident"
for spec in "3 0.2 0" "5 0.2 0" "5 0.2 1e-5" "3 0.2 1e-5"; do
  set -- $spec
  echo "######## depth=$1 dropout=$2 wd=$3  $(date -u +%H:%M:%S) ########"
  "$PY" -m floan.model.train $C --depth $1 --dropout $2 --weight-decay $3 || { echo "CELL FAILED: $spec"; exit 1; }
done
echo "######## DEPTH CHECK COMPLETE $(date -u +%H:%M:%S) ########"
