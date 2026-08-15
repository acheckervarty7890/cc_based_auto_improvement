#!/usr/bin/env bash
# Wait for the single diagnostic job to finish, then run the full pipeline. The full run
# skips any job whose CSV already exists, so the diagnostic's result is kept, and it picks
# up the malloc_trim fix the diagnostic process started too early to have.
cd /workspace/cc_based_auto_improvement || exit 1
while kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "[relaunch $(date -Is)] diagnostic job (pid $1) finished"
set -a; . ./HFtokn.txt; set +a
export KAGGLE_CONFIG_DIR=/workspace/cc_based_auto_improvement/kaggle/
export AGENTIC_REDTEAM_MAX_MEMORY="0=6GiB,cpu=7GiB"
export WORK_DIR=/workspace/cc_based_auto_improvement/results/devsamples_kfold
export NO_FETCH=1   # both caches are filled and verified
echo "[relaunch $(date -Is)] starting run_devsamples_kfold.sh"
exec bash run_devsamples_kfold.sh
