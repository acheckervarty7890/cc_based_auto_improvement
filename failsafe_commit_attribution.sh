#!/usr/bin/env bash
#
# failsafe_commit_attribution.sh — keep run_attribution_hu_harm_gemma27b.sh's progress on
# the remote so a wiped container costs minutes, not hours.
#
# The sibling failsafe_commit.sh checkpoints *iterative_retrain* runs, and keys off
# artifacts that exist only there (phase markers, probe_iter*.pkl, a comparison CSV). The
# attribution sweep has none of those. Its resume state is one append-only sidecar per arm —
# <attribution>/<arm>_iter3_loo_progress.jsonl, one fsync'd row per finished pass — so this
# poller is correspondingly simpler: watch the attribution dir, and commit whenever it
# changes.
#
# WHAT IT COMMITS
#   results_hu_harm_gemma27b_batch_ablation/attribution/**   the progress sidecars, the
#       LOO .npz cubes, the verification JSON and the noise-floor JSON  (~5 MB total)
#   logs/attribution_*                                        the run logs
#
# WHAT IT DOES NOT COMMIT: activations. Every add carries `:(exclude)**/*.pt`, because the
# blobs are ~12 GB, are published on Kaggle already, and are a recompute-only input rather
# than resume state. A stray .pt in the commit set would be the one mistake that makes these
# checkpoints unusable.
#
# WHY force-add: .gitignore has both `results*` and `*.log`, which is exactly the material
# that has to survive. Same reasoning as failsafe_commit.sh.
#
# BRANCH MODEL: commits and pushes onto whatever branch is already checked out. It never
# switches branches.
#
# EXIT: Ctrl-C / SIGTERM makes a final commit via the trap, so stopping it by hand does not
# strand the last interval's work.
#
# USAGE (alongside the runner, second terminal or second nohup):
#   nohup bash failsafe_commit_attribution.sh > logs/failsafe_attribution.out 2>&1 &
#
#   INTERVAL=300  seconds between polls (default 300)
#   NO_PUSH=1     commit locally only (e.g. no credentials on the box)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

INTERVAL="${INTERVAL:-300}"
ATTRIB_DIR=results_hu_harm_gemma27b_batch_ablation/attribution
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

mkdir -p logs "$ATTRIB_DIR"

say() { echo ">>> $(date -Is)  $*"; }

snapshot() {  # $1 = commit subject suffix
    git add -f "$ATTRIB_DIR" ':(exclude)**/*.pt' 2>/dev/null || true
    git add -f logs/attribution_*.log logs/attribution_*.out 2>/dev/null || true
    if git diff --cached --quiet; then
        return 1
    fi
    local n_pass
    n_pass=$(cat "$ATTRIB_DIR"/*_loo_progress.jsonl 2>/dev/null | wc -l)
    git commit -q -m "failsafe(attribution): $1 — ${n_pass} LOO passes done @ $(date -Is)"
    if [ -z "${NO_PUSH:-}" ]; then
        git push -q origin "$BRANCH" || say "push failed (committed locally; will retry next poll)"
    fi
    return 0
}

finish() {
    say "stopping — final snapshot"
    snapshot "final" && say "committed" || say "nothing new to commit"
    exit 0
}
trap finish INT TERM

say "watching $ATTRIB_DIR on branch '$BRANCH', every ${INTERVAL}s (NO_PUSH=${NO_PUSH:-unset})"

# Poll rather than inotify: the signal we want is "a pass landed", which is a file append,
# and a 5-minute granularity already bounds the loss to ~10 passes. Simpler, and it works
# the same on a container without inotify tooling installed.
last=""
while true; do
    now=$(find "$ATTRIB_DIR" logs -maxdepth 1 -newermt "-${INTERVAL} seconds" \
              \( -name '*.jsonl' -o -name '*.npz' -o -name '*.json' -o -name 'attribution_*' \) \
              2>/dev/null | sort | tr '\n' ' ')
    stamp=$(cat "$ATTRIB_DIR"/*_loo_progress.jsonl 2>/dev/null | wc -l)
    if [ -n "$now" ] || [ "$stamp" != "$last" ]; then
        if snapshot "checkpoint"; then
            say "checkpointed (${stamp} passes)"
        fi
        last="$stamp"
    fi
    sleep "$INTERVAL"
done
