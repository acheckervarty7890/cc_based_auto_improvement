#!/usr/bin/env bash
set -u

# Extract and publish the RED-TEAM activations of the two gemma-3-27b runs whose caches were
# never synced back off their cloud boxes:
#
#   experiment9_cloud              high-stakes            -> kaggle anku7890/hs-gemma27b-*
#   experiment_instruction_cloud_1 instruction-following  -> kaggle anku7890/instructions-gemma27b-*
#
# Both are driven by scripts/publish_kaggle_redteam_activations.py `sync`, which walks the run's
# units (base split first, then each (arm, iteration)) and extracts + uploads each one BEFORE
# starting the next. Every blob is written through as soon as it is computed and every unit's
# existence on Kaggle is the resume signal, so this script is safe to kill and re-run: it picks
# up at the first unit that is not yet published, and recomputes only blobs missing from disk.
#
# WHY A WORKTREE PER EXPERIMENT. An experiment's probes, configs, postprocessed dumps and base
# training data live on its own branch; this script runs from experiment11_cloud, which has
# none of them. `git worktree add --detach` gives each one a read-only checkout — nothing is
# written there. The activation caches and staged archives stay under this repo (results_*/,
# which .gitignore already covers).
#
# COST. 1470 conversations for hs + 1992 for instructions + both base splits, each one forward
# through google/gemma-3-27b-it truncated to layers 0..32, then ~31 GB of archives uploaded
# (the per-iteration datasets are self-contained, so a conversation shared by two iterations is
# uploaded twice — see the script's module docstring on why the iterations cannot be unioned).
#
# Usage:
#   mkdir -p logs
#   nohup bash run_publish_redteam_acts.sh > logs/publish_redteam_acts.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

PY=.venv_claude/bin/python
WT_ROOT="${WT_ROOT:-$HOME/wt}"

# Kaggle credentials. KAGGLE_CONFIG_DIR must name the DIRECTORY holding kaggle.json — the API
# joins the filename on itself and os.makedirs a wrong path, so pointing at the file fails
# silently. Checked here rather than at first use: the first upload is an hour of forwards in.
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$PWD/kaggle}"
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ ! -f "$KAGGLE_CONFIG_DIR/kaggle.json" ]; then
    echo "ERROR: no $KAGGLE_CONFIG_DIR/kaggle.json and no KAGGLE_API_TOKEN." >&2
    exit 1
fi

# Pin accelerate's per-device budget. Left unset, tuberlens passes max_memory=None and
# accelerate infers from whatever is free AT LOAD TIME, then silently spills to DISK — the
# 48-264 s/sample regime. Measured on this box (8 GB VRAM / 31 GB RAM): the truncated model
# fits with no disk offload at this split.
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=6GiB,cpu=25GiB}"

# tuberlens' extraction batch size, which is also this script's chunk size. Left at 1, and
# that is a measurement, not the default going unexamined. A/B on the SAME 20 conversations,
# same box: BATCH_SIZE=1 -> 33.7 s/sample, BATCH_SIZE=4 -> 68.7 s/sample. Batching is 2x
# SLOWER here. The box cannot hold the truncated model (28.8 GiB) plus working set in 31 GB of
# RAM, so every forward faults a slice of the weights back in at ~16 MB/s of random page reads
# — with the CPU 95% idle, that fault traffic IS the per-sample cost, and a bigger batch's
# intermediates simply evict more of the weights the next forward needs. Batching only wins
# where the model stays resident; measure before raising it on another box.
export BATCH_SIZE="${BATCH_SIZE:-1}"

# Instantiate a 1-layer stub in place of gemma-3's vision tower, which text-only conversations
# never execute. Worth 0.83 GB of resident weights, which on this box is the difference between
# faulting and not; the text stack — and therefore every activation — is untouched, verified
# bit-identical against blobs computed without it.
export AGENTIC_REDTEAM_TRIM_VISION="${AGENTIC_REDTEAM_TRIM_VISION:-1}"

run_experiment () {  # $1 = experiment name, $2 = branch, $3 = logfile
    local exp="$1" branch="$2" log="$3"
    local wt="$WT_ROOT/$exp"

    if [ ! -d "$wt/probes" ]; then
        echo ">>> $(date -Is)  creating worktree for $branch at $wt"
        git worktree add --detach "$wt" "origin/$branch" || return 1
    fi

    echo ">>> $(date -Is)  START $exp ($branch)  -> $log"
    local rc=0
    # --no-release-between-units: keep the extraction model loaded across the pack+upload.
    # The default drops it to free RAM, which is the right trade on a box that holds the model
    # comfortably — here it is not. This box cannot cache all 28.8 GiB of weights, so a fresh
    # load starts cold and the first samples of a unit cost 100+ s each while the working set
    # faults in, settling to ~9 s once warm. Paying that warm-up once beats paying it 14 times;
    # keeping the pages mapped also makes them harder to evict while the archive streams past.
    $PY scripts/publish_kaggle_redteam_activations.py sync \
        --experiment "$exp" --experiment-root "$wt" --no-release-between-units \
        >> "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo ">>> $(date -Is)  FAILED $exp (exit $rc) — see $log" >&2
        tail -n 15 "$log" >&2
    else
        echo ">>> $(date -Is)  DONE  $exp"
    fi
    return "$rc"
}

overall=0
# Sequential, never parallel: both arms of both runs share one GPU and one extraction model,
# and two live writers into one cache dir can tear a blob.
run_experiment hs           experiment9_cloud              logs/publish_redteam_acts_hs.log           || overall=1
run_experiment instructions experiment_instruction_cloud_1 logs/publish_redteam_acts_instructions.log || overall=1

echo ">>> $(date -Is)  all experiments finished (overall exit $overall)"
exit "$overall"
