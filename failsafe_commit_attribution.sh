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

# Stage one path, loudly. Errors are REPORTED, never swallowed: the first version of
# this script ran `git add -f "$ATTRIB_DIR" :(exclude)**/*.pt 2>/dev/null || true`, which
# worked on the box it was written on and silently added nothing on the box it ran on —
# so 30+ checkpoints committed the logs and none of the results, and the failure was
# invisible precisely because stderr was discarded. A checkpointer that cannot save the
# thing it exists to save has to say so.
stage() {
    local f="$1" err
    [ -e "$f" ] || return 0
    case "$f" in *.pt) return 0 ;; esac          # activations: multi-GB, on Kaggle already
    case "$f" in *_features.npz) return 0 ;; esac # 35 MB each, not needed downstream
    if ! err=$(git add -f -- "$f" 2>&1); then
        say "WARNING: could not stage $f: $err"
        return 1
    fi
    return 0
}

snapshot() {  # $1 = commit subject suffix
    local f
    # Explicit per-file globs rather than a directory plus an :(exclude) pathspec. The
    # results are the payload, so they are staged first and each failure is named.
    for f in "$ATTRIB_DIR"/*.jsonl "$ATTRIB_DIR"/*.npz "$ATTRIB_DIR"/*.json \
             logs/attribution_*.log logs/attribution_*.out \
             logs/failsafe_attribution.out; do
        stage "$f"
    done
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

# Self-test at startup: stage once and report whether the RESULTS (not just the logs)
# actually made it into the index. The previous failure mode was silent for hours, so
# the state of the thing being protected is checked in minute one, out loud.
for f in "$ATTRIB_DIR"/*.jsonl "$ATTRIB_DIR"/*.npz "$ATTRIB_DIR"/*.json; do stage "$f"; done
staged_results=$(git diff --cached --name-only -- "$ATTRIB_DIR" | wc -l)
on_disk=$(ls -1 "$ATTRIB_DIR"/*.jsonl "$ATTRIB_DIR"/*.npz "$ATTRIB_DIR"/*.json 2>/dev/null | wc -l)
tracked=$(git ls-files -- "$ATTRIB_DIR" | wc -l)
if [ "$on_disk" -gt 0 ] && [ "$staged_results" -eq 0 ] && [ "$tracked" -eq 0 ]; then
    say "FATAL: $on_disk result file(s) on disk, none stageable and none tracked."
    say "       Checkpointing would silently protect nothing. Fix git first."
    exit 1
fi
say "self-test: $on_disk result file(s) on disk, $staged_results newly staged, $tracked already tracked"

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
