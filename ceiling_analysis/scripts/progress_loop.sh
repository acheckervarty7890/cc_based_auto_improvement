#!/usr/bin/env bash
# Snapshot + push every 30 minutes for as long as the analysis is running.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
while true; do
  flock -w 300 /tmp/ceiling_analysis_git.lock ceiling_analysis/scripts/progress_snapshot.sh \
    >> ceiling_analysis/logs/progress_loop.log 2>&1
  sleep 1800
done
