#!/usr/bin/env bash
# Two retraining experiments on the human-harm gemma-3-27b (L32) probe, run in one pass.
#
#   (i)  scripts/eval_kfold_cv.py       k-fold CV with the eval set as BOTH train and test
#                                       -> the in-distribution ceiling
#   (ii) scripts/dev_sample_retrain.py  iteration-3 retrain + 0/2/8/16/30 dev samples per
#                                       eval split, both attacker arms
#                                       -> the dose-response curve between the pipeline
#                                          and that ceiling
#
# ORDER IS DELIBERATE. The CV runs first: it needs only the eval activations, loads no
# model, and finishes in minutes, so the run has a headline number banked before the long
# part starts. The dev-sample arm then does its one model-loading step (120 dev
# conversations through gemma-3-27b) and the ten retrain+eval jobs.
#
# WHAT COMES OFF KAGGLE, AND WHAT DOES NOT
#   eval activations     4 blobs, ~4.6 GB          scripts/fetch_kaggle_eval_activations.py
#   red-team activations ~1.6k blobs, ~10 GB       publish_kaggle_redteam_activations.py restore
#   dev activations      120, NOT published        computed here, once (stage `extract`)
#
# The red-team blobs are the ones REPUBLISHED after the contrastive-pair shortening
# (commit 98b5011): 105 conversations were rewritten and so carry new content-addressed
# keys. `restore` writes only what is missing, and PRUNE_STALE then deletes blobs no dump
# on this branch references — which is the cheap way round. Wiping the cache first would
# re-transfer ~1500 valid blobs to replace 105.
#
# NO MODEL IS LOADED except by stage `extract`, and only if a dev blob is missing.
#
# Usage:
#   export KAGGLE_CONFIG_DIR=/path/to/dir-holding-kaggle.json
#   nohup bash run_devsamples_kfold.sh > logs/run_devsamples_kfold.out 2>&1 &
#
#   NO_FETCH=1 bash run_devsamples_kfold.sh        # caches already filled
#   PARTS="cv" bash run_devsamples_kfold.sh        # experiment (i) alone
#   PARTS="dev" bash run_devsamples_kfold.sh       # experiment (ii) alone
#   DRY_RUN=1 bash run_devsamples_kfold.sh         # print the plan, compute nothing
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

PY="${PY:-$REPO_ROOT/.venv_claude/bin/python}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/results/devsamples_kfold}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/results_hu_harm_gemma27b_batch_ablation}"
LOG="${LOG:-$REPO_ROOT/devsamples_kfold.log}"
PARTS="${PARTS:-cv dev}"
K="${K:-5}"
DEV_LEVELS="${DEV_LEVELS:-0 2 8 16 30}"
SEEDS="${SEEDS:-42}"
ITERATION="${ITERATION:-3}"
NO_FETCH="${NO_FETCH:-0}"
PRUNE_STALE="${PRUNE_STALE:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ -x "$PY" ]] || { echo "ERROR: no interpreter at $PY" >&2; exit 2; }
mkdir -p "$WORK_DIR" "$CACHE_DIR/base_activations" "$CACHE_DIR/eval_activations" logs

# Credentials are checked HERE rather than at first use: the fetch is the very first
# thing that needs them, but an unauthenticated KaggleApi.authenticate() ends in exit(1)
# rather than an exception, so a bad path fails in a way that is hard to read from a log.
if [[ "$NO_FETCH" != "1" ]]; then
    if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
        kj="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
        [[ -f "$kj" ]] || kj="$HOME/.config/kaggle/kaggle.json"
        if [[ ! -f "$kj" ]]; then
            echo "ERROR: no Kaggle credentials. Set KAGGLE_CONFIG_DIR to the DIRECTORY" >&2
            echo "       holding kaggle.json (not the file), or export KAGGLE_API_TOKEN." >&2
            echo "       Or pass NO_FETCH=1 if both caches are already filled." >&2
            exit 2
        fi
        echo ">>> kaggle credentials: $kj"
    fi
    "$PY" -c "import kaggle" 2>/dev/null || {
        echo "ERROR: the 'kaggle' package is not installed: $PY -m pip install kaggle" >&2
        exit 2
    }
fi

# Same reasoning for the HuggingFace token, one stage later. Only `extract` loads a
# model, and gemma-3-27b-it is gated, so tuberlens' hf_login() raises "No HuggingFace
# token found" — AFTER the ~15 GB of Kaggle downloads and the CV have run. Check it
# here so a box with no token says so in the first second.
if [[ " $PARTS " == *" dev "* && -z "${HF_TOKEN:-}${HUGGINGFACE_TOKEN:-}" ]]; then
    echo "ERROR: no HuggingFace token. The dev-sample extraction loads the gated" >&2
    echo "       google/gemma-3-27b-it, so export HF_TOKEN (or HUGGINGFACE_TOKEN)" >&2
    echo "       with an account that has accepted the Gemma licence." >&2
    echo "       PARTS=\"cv\" needs no token — it runs off cached activations." >&2
    exit 2
fi

echo "=== dev-sample + k-fold experiments ==="
echo "  work dir  : $WORK_DIR"
echo "  cache     : $CACHE_DIR"
echo "  parts     : $PARTS"
echo "  k         : $K"
echo "  dev levels: $DEV_LEVELS   seeds: $SEEDS   iteration: $ITERATION"
echo "  log       : $LOG"
echo

DRY_FLAG=()
[[ "$DRY_RUN" == "1" ]] && DRY_FLAG=(--dry-run)

run () {  # echo the command, then run it through the shared log
    echo; echo ">>> $(date -Is)  $*"; echo
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

# --- caches ---------------------------------------------------------------------------
if [[ "$NO_FETCH" != "1" ]]; then
    run "$PY" -u scripts/fetch_kaggle_eval_activations.py \
        --concept hu_harm \
        --cache-dir "$CACHE_DIR/eval_activations" "${DRY_FLAG[@]}" \
        || { echo "ERROR: eval activation fetch failed" >&2; exit 1; }

    # Red-team blobs are only needed by part (ii); skip the ~10 GB pull for a CV-only run.
    if [[ " $PARTS " == *" dev "* ]]; then
        run "$PY" -u scripts/publish_kaggle_redteam_activations.py restore \
            --iterations "$ITERATION" \
            --cache-dir "$CACHE_DIR/base_activations" "${DRY_FLAG[@]}" \
            || { echo "ERROR: red-team activation restore failed" >&2; exit 1; }

        if [[ "$PRUNE_STALE" == "1" ]]; then
            run "$PY" -u scripts/prune_stale_redteam_activations.py \
                --cache-dir "$CACHE_DIR/base_activations" "${DRY_FLAG[@]}" \
                || echo "WARN: prune failed; stale blobs cost disk, not correctness"
        fi
    fi
fi

# --- (i) k-fold CV on the eval set ----------------------------------------------------
if [[ " $PARTS " == *" cv "* ]]; then
    run "$PY" -u scripts/eval_kfold_cv.py \
        --work-dir "$WORK_DIR/kfold_cv" \
        --eval-cache "$CACHE_DIR/eval_activations" \
        --k "$K" \
        || { echo "ERROR: k-fold CV failed" >&2; exit 1; }
fi

# --- (ii) dev samples added to the red-team retrain -----------------------------------
if [[ " $PARTS " == *" dev "* ]]; then
    run "$PY" -u scripts/dev_sample_retrain.py \
        --work-dir "$WORK_DIR/dev_samples" \
        --cache-dir "$CACHE_DIR" \
        --iteration "$ITERATION" \
        --dev-levels $DEV_LEVELS \
        --seeds $SEEDS "${DRY_FLAG[@]}" \
        || { echo "ERROR: dev-sample retrain failed" >&2; exit 1; }
fi

# The failsafe committer watches for this file and stops (and, with --close-script, powers
# the box off) once it appears. Written LAST and only on success, so a failed part leaves
# the committer running and the box up.
if [[ "$DRY_RUN" != "1" ]]; then
    date -Is > "$WORK_DIR/ALL_DONE"
    echo
    echo ">>> $(date -Is)  all requested parts finished; wrote $WORK_DIR/ALL_DONE"
fi
