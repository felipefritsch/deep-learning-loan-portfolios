#!/usr/bin/env bash
# W3b — Monte-Carlo H>1 sequence-model pricing driver (pod GPU run).
# Mirrors run_w1_all.sh's overlay-protection export block (TMPDIR + all torch/triton/CUDA/mpl
# compile-cache redirects under /workspace, NEVER the small overlay /), then runs the W3b driver
# COVID-first across the priority anchors, both arms (gru + xf), seed 0, on cuda. The COVID anchor
# (k2020) runs first so its paths-convergence sweep fixes N, which the driver reuses for the rest.
#
# Launch under nohup so it survives an SSH/VSCode disconnect:
#   nohup scripts/m27a/run_w3b.sh > /workspace/logs/w3b.out 2>&1 &
set -euo pipefail

REPO=/workspace/repo
PY=/workspace/.venv/bin/python
OUTPUTS=/workspace/dissertation/outputs

# Keep ALL scratch on /workspace (the large network volume), NOT the small overlay root /.
# A full overlay killed an earlier run: torch/triton/CUDA compile caches + tmp accumulate on /
# and tip it to 0 bytes, after which every write hits ENOSPC and the run dies mid-window.
export TMPDIR="${TMPDIR:-$OUTPUTS/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/workspace/.nv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.mpl}"
export LOGDIR="${LOGDIR:-/workspace/logs}"

cd "$REPO"
mkdir -p "$LOGDIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"

# Priority anchors, COVID-first (the driver also reorders COVID-first internally and reuses the
# COVID-chosen N for the rest). seed 0, both arms, no --n-paths => convergence sweep at k2020.
"$PY" -u -m floan.model.seq_price_mc \
    --anchors 2020 2023 2025 2019 2015 \
    --archs gru xf \
    --device cuda \
    --seed 0
