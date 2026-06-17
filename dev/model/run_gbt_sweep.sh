#!/usr/bin/env bash
# M17 GBT sweep supervisor (06_GBT_BASELINE / M17).
#
# Runs the 10-window GBT-only backtest, auto-resuming on transient failure. The dominant
# failure mode is an SSD drop+reconnect: each python run fails fast via require_drive() while
# the drive is gone, the loop waits, and it resumes the instant the drive is back — no manual
# action beyond reconnecting the cable. The fit is idempotent on (window, config=msh1000):
# completed windows skip in seconds (their atomic metrics.json is the completion marker), so a
# re-run only fits what's missing. Retries are BOUNDED so a deterministic failure (e.g. a true
# OOM on the biggest window) can't infinite-loop — it surfaces in the log instead.
#
# k=2015 is already complete (promoted from the full-scale reconfirm fit at msh=1000); these
# are the 10 remaining windows, key/regime-first (2019,2020,2023,2025) then the six fillers.
#
# Launch (detached, no-sleep, unbuffered, append log):
#   nohup caffeinate -dims bash dev/model/run_gbt_sweep.sh \
#     >> "/Volumes/SSD Felipe/dissertation/logs/gbt_sweep.log" 2>&1 &
set -u

REPO="/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"
MODELS="/Volumes/SSD Felipe/dissertation/models/gbt/full"
WINDOWS="2019 2020 2023 2025 2016 2017 2018 2021 2022 2024"
SWEEP_WINDOWS="2016 2017 2018 2019 2020 2021 2022 2023 2024 2025"   # the 10 to complete (2015 promoted)
MAX_ATTEMPTS=30
SLEEP_SECS=90

cd "$REPO" || exit 1

done_count() {
  local n=0 k
  for k in $SWEEP_WINDOWS; do
    [ -f "$MODELS/k$k/metrics.json" ] && n=$((n + 1))
  done
  echo "$n"
}

for a in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== supervisor attempt $a/$MAX_ATTEMPTS  $(date '+%F %T')  ($(done_count)/10 windows done) ==="
  .venv/bin/python -u dev/model/backtest.py --device cpu --gbt-only --windows $WINDOWS --gbt-threads 8
  rc=$?
  n=$(done_count)
  echo "=== attempt $a exited rc=$rc; $n/10 windows complete  $(date '+%F %T') ==="
  if [ "$n" -ge 10 ]; then
    echo "=== supervisor: ALL 10 sweep windows complete — done ==="
    break
  fi
  echo "=== supervisor: $((10 - n)) windows remain (likely SSD gone or a window died); resuming in ${SLEEP_SECS}s ==="
  sleep "$SLEEP_SECS"
done

echo "=== supervisor finished  $(date '+%F %T')  ($(done_count)/10 windows done) ==="
