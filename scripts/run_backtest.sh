#!/bin/bash
# M10b — the rolling backtest loop over all 11 windows. Resumable: finished fits are
# skipped, interrupted ones resume from their last epoch checkpoint. Runs in tmux so a
# disconnect costs nothing. Verify + base-rate QA run automatically at the end.
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m floan.model.backtest --device cuda --amp
