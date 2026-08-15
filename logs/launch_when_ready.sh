#!/usr/bin/env bash
# Wait for the pre-warm downloads (Kaggle restore + gemma snapshot) to finish, then
# start run_devsamples_kfold.sh. Chained rather than run in parallel because the run's
# own fetch writes into the same cache dir the pre-warm is still staging into.
cd /workspace/cc_based_auto_improvement || exit 1
for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    echo "[launcher $(date -Is)] pid $pid finished"
done
set -a; . ./HFtokn.txt; set +a
export KAGGLE_CONFIG_DIR=/workspace/cc_based_auto_improvement/kaggle/
# 8 GB GPU / 15 GB RAM against ~30 GB of executed gemma-3-27b weights: the remainder
# goes to disk either way, but pinning stops accelerate handing the CPU every byte that
# is free at load time and leaving nothing for the process itself.
export AGENTIC_REDTEAM_MAX_MEMORY="0=6GiB,cpu=7GiB"
export WORK_DIR=/workspace/cc_based_auto_improvement/results/devsamples_kfold
echo "[launcher $(date -Is)] starting run_devsamples_kfold.sh"
exec bash run_devsamples_kfold.sh
