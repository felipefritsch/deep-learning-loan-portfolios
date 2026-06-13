#!/bin/bash
# M10b — the rolling backtest loop over all 11 windows. Resumable: finished fits are
# skipped, interrupted ones resume from their last epoch checkpoint. Runs in tmux so a
# disconnect costs nothing. Verify + base-rate QA run automatically at the end.
set -u
cd /workspace/repo/dev/model
exec /workspace/repo/.venv/bin/python backtest.py --device cuda --amp
