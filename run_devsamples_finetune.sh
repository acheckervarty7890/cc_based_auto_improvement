#!/usr/bin/env bash
# Sequential (train-then-finetune) counterpart to the mixed dev-sample run.
#
#   (i)  scripts/dev_sample_finetune.py   stage 1 = base + red-team, stage 2 = dev only
#   (ii) scripts/dev_sample_retrain.py    the ORIGINAL mixed run, extended to the same
#                                         weight-init seeds
#
# (ii) is here because the comparison is the point. The mixed run of 2026-08-15 was a
# single seed, and the sequential-vs-mixed gaps this is measuring are of the same order
# as this setup's seed noise (~0.008 AUROC) — so without matching seeds on both sides the
# comparison table is unreadable. The two parts run in SEQUENCE, never concurrently: each
# holds ~12 GB of activations at its peak and this box has 15.9 GB.
#
# No model is loaded by either part. Every activation comes from the caches the
# 2026-08-15 run left behind (red-team blobs + the 120 dev conversations).
#
# Usage:
#   nohup bash run_devsamples_finetune.sh > logs/run_devsamples_finetune.out 2>&1 &
#
#   PARTS="seq"  bash run_devsamples_finetune.sh    # the new experiment alone
#   PARTS="mix"  bash run_devsamples_finetune.sh    # the mixed-run seed extension alone
#   SEEDS="42"   bash run_devsamples_finetune.sh    # single seed, ~30 min
#   DRY_RUN=1    bash run_devsamples_finetune.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

PY="${PY:-$REPO_ROOT/.venv_claude/bin/python}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/results/devsamples_kfold}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/results_hu_harm_gemma27b_batch_ablation}"
LOG="${LOG:-$REPO_ROOT/devsamples_finetune.log}"
PARTS="${PARTS:-seq mix}"
SEEDS="${SEEDS:-42 7 13}"
DEV_LEVELS="${DEV_LEVELS:-0 2 8 16 30}"
ITERATION="${ITERATION:-3}"
DRY_RUN="${DRY_RUN:-0}"

[[ -x "$PY" ]] || { echo "ERROR: no interpreter at $PY" >&2; exit 2; }
mkdir -p "$WORK_DIR" logs

DRY_FLAG=()
[[ "$DRY_RUN" == "1" ]] && DRY_FLAG=(--dry-run)

echo "=== dev-sample finetune (sequential) vs mixed ==="
echo "  work dir : $WORK_DIR"
echo "  parts    : $PARTS"
echo "  seeds    : $SEEDS   iteration: $ITERATION"
echo

run () {
    echo; echo ">>> $(date -Is)  $*"; echo
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

# --- (i) sequential: red-team first, then finetune on dev -----------------------------
if [[ " $PARTS " == *" seq "* ]]; then
    run "$PY" -u scripts/dev_sample_finetune.py \
        --work-dir "$WORK_DIR/finetune" \
        --cache-dir "$CACHE_DIR" \
        --iteration "$ITERATION" \
        --seeds $SEEDS "${DRY_FLAG[@]}" \
        || { echo "ERROR: sequential finetune failed" >&2; exit 1; }
fi

# --- (ii) the mixed run, on the same seeds --------------------------------------------
# Same --work-dir as 2026-08-15 on purpose: `run` skips any (arm, N, seed) whose CSV
# already exists, so this adds the new seeds and leaves the seed-42 results untouched.
if [[ " $PARTS " == *" mix "* ]]; then
    run "$PY" -u scripts/dev_sample_retrain.py \
        --work-dir "$WORK_DIR/dev_samples" \
        --cache-dir "$CACHE_DIR" \
        --iteration "$ITERATION" \
        --dev-levels $DEV_LEVELS \
        --seeds $SEEDS \
        --stages run analyze "${DRY_FLAG[@]}" \
        || { echo "ERROR: mixed seed extension failed" >&2; exit 1; }

    # Re-join the comparison now that the mixed side has the extra seeds.
    if [[ " $PARTS " == *" seq "* && "$DRY_RUN" != "1" ]]; then
        run "$PY" -u scripts/dev_sample_finetune.py \
            --work-dir "$WORK_DIR/finetune" \
            --cache-dir "$CACHE_DIR" \
            --seeds $SEEDS \
            --stages analyze
    fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
    date -Is > "$WORK_DIR/FINETUNE_DONE"
    echo; echo ">>> $(date -Is)  finished; wrote $WORK_DIR/FINETUNE_DONE"
fi
