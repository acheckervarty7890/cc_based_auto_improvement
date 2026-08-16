#!/usr/bin/env bash
#
# failsafe_commit_arch.sh — keep scripts/arch_sweep.py's progress on the remote so a
# wiped container costs one fit, not the whole sweep.
#
# Sibling of failsafe_commit_heldout.sh, same reasoning, different payload and a
# different denominator: this sweep is 8 architectures x 5 seeds x 2 arms = 80 fits,
# written into ONE sidecar for both arms (rows carry `arm` and `architecture`), so the
# heldout poller's /20 denominator and globs would mislabel every checkpoint.
#
# The resume state is results_hu_harm_gemma27b_batch_ablation/arch_sweep/
# arch_progress.jsonl: one fsync'd row per finished (arm, architecture, seed) fit
# carrying that probe's fp32 logits on all 866 eval rows. Adam fits are minutes each and
# the whole sweep is hours; this watches that file and commits whenever it grows.
#
# WHAT IT COMMITS
#   <out-dir>/*                          progress sidecar and the final JSON
#   scripts/arch_sweep.py                the sweep
#   scripts/nonlinear_ceiling.py         the Tier-2 ceiling measurement
#   scripts/test_probe_architectures.py  the architecture checks
#   src/agentic_redteam/probe_architectures.py   the heads themselves
#   logs/arch_sweep.log, logs/nonlinear_ceiling.log
#   failsafe_commit_arch.sh, logs/failsafe_<TAG>.out
#
# WHAT IT DOES NOT COMMIT: activations (`*.pt`, ~11 GB across the eval blobs and the
# per-conversation red-team cache). They are recompute-only inputs, are published on
# Kaggle, and one stray blob would make these checkpoints unpushable.
# Nor kaggle/kaggle.json — it holds live API credentials.
#
# WHY force-add: .gitignore carries both `results*` and `*.log`, which is exactly the
# material that has to survive. Same reasoning as the sibling scripts.
#
# BRANCH MODEL: commits onto whatever branch is checked out, but PUSHES to
# TARGET_BRANCH (default experiment11_cloud) via `HEAD:$TARGET_BRANCH`. The push is a
# plain fast-forward: no --force, ever. The self-test refuses to start unless
# origin/$TARGET_BRANCH is an ancestor of HEAD, so this can only advance that branch.
#
# EXIT: Ctrl-C / SIGTERM makes a final commit via the trap. NOTE bash defers a trap
# until the current foreground command finishes, so a SIGTERM during the poll `sleep`
# is not acted on until that sleep elapses.
#
# USAGE:
#   nohup bash failsafe_commit_arch.sh > logs/failsafe_arch.out 2>&1 &
#
#   INTERVAL=1800       seconds between polls (default 1800 = 30 min)
#   ODIR=...            results dir to watch
#   TAG=...             commit-subject tag AND the logs/failsafe_<TAG>.out it commits
#   TOTAL=...           denominator shown in the commit subject (default 80)
#   TARGET_BRANCH=...   remote branch to push to (default experiment11_cloud)
#   NO_PUSH=1           commit locally only

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

INTERVAL="${INTERVAL:-1800}"
ODIR="${ODIR:-results_hu_harm_gemma27b_batch_ablation/arch_sweep}"
TAG="${TAG:-arch}"
TOTAL="${TOTAL:-80}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
TARGET_BRANCH="${TARGET_BRANCH:-experiment11_cloud}"

PAYLOAD_CODE=(
    scripts/arch_sweep.py
    scripts/nonlinear_ceiling.py
    scripts/mlp_readout_cosine.py
    scripts/test_probe_architectures.py
    scripts/attribution_refit.py
    src/agentic_redteam/probe_architectures.py
    failsafe_commit_arch.sh
    logs/arch_sweep.log
    logs/nonlinear_ceiling.log
    logs/mlp_readout_cosine.log
)

mkdir -p logs "$ODIR"

say() { echo ">>> $(date -Is)  $*"; }

# Stage one path, loudly. Errors are REPORTED, never swallowed — an ancestor of this
# script discarded stderr and silently committed nothing but logs for hours.
stage() {
    local f err
    f="$1"
    [ -e "$f" ] || return 0
    case "$f" in *.pt) return 0 ;; esac          # activations: multi-GB, on Kaggle
    case "$f" in */kaggle.json) return 0 ;; esac # credentials
    if ! err=$(git add -f -- "$f" 2>&1); then
        say "WARNING: could not stage $f: $err"
        return 1
    fi
    return 0
}

# One row per finished (arm, architecture, seed) fit. The redirection itself fails when
# the sidecar does not exist yet (the first minutes of a run), so guard on the file
# rather than letting bash write to stderr each poll.
n_fits() {
    local p="$ODIR/arch_progress.jsonl"
    [ -f "$p" ] && wc -l < "$p" || echo 0
}

snapshot() {  # $1 = commit subject suffix
    local f n
    for f in "$ODIR"/* "${PAYLOAD_CODE[@]}" "logs/failsafe_${TAG}.out"; do
        stage "$f"
    done
    if git diff --cached --quiet; then
        return 1
    fi
    n=$(n_fits)
    git commit -q -m "failsafe(${TAG}): $1 — ${n}/${TOTAL} fits done @ $(date -Is)"
    if [ -z "${NO_PUSH:-}" ]; then
        git push -q origin "HEAD:$TARGET_BRANCH" \
            || say "push to $TARGET_BRANCH failed (committed locally; retry next poll)"
    fi
    return 0
}

finish() {
    say "stopping — final snapshot"
    snapshot "final" && say "committed" || say "nothing new to commit"
    exit 0
}
trap finish INT TERM

say "watching $ODIR, committing on '$BRANCH', pushing to origin/$TARGET_BRANCH, every ${INTERVAL}s (NO_PUSH=${NO_PUSH:-unset})"

# Refuse to run at all unless the push can only fast-forward the target. A checkpointer
# that force-pushes is worse than none: these commits land unattended every 30 min, and
# the target is a shared experiment branch.
if [ -z "${NO_PUSH:-}" ]; then
    git fetch -q origin "$TARGET_BRANCH" 2>/dev/null \
        || say "WARNING: could not fetch origin/$TARGET_BRANCH; using the cached ref"
    if ! git rev-parse --verify -q "origin/$TARGET_BRANCH" >/dev/null; then
        say "FATAL: origin/$TARGET_BRANCH does not exist."
        exit 1
    fi
    if ! git merge-base --is-ancestor "origin/$TARGET_BRANCH" HEAD; then
        say "FATAL: origin/$TARGET_BRANCH is NOT an ancestor of HEAD — pushing would"
        say "       need a merge or a force. Reconcile the branches first."
        exit 1
    fi
    say "fast-forward check: origin/$TARGET_BRANCH is an ancestor of HEAD ($(git rev-list --count "origin/$TARGET_BRANCH"..HEAD) commit(s) ahead)"
fi

# Self-test in minute one: the previous failure mode in this repo was a checkpointer
# that protected nothing and said so to no one.
for f in "$ODIR"/* "${PAYLOAD_CODE[@]}"; do stage "$f"; done
staged=$(git diff --cached --name-only | wc -l)
on_disk=$(ls -1d "$ODIR"/* "${PAYLOAD_CODE[@]}" 2>/dev/null | wc -l)
if [ "$on_disk" -eq 0 ]; then
    say "FATAL: nothing matching the payload globs exists yet — wrong cwd, or the run never started."
    exit 1
fi
say "self-test: $on_disk payload file(s) on disk, $staged staged, $(n_fits) fit(s) recorded"

last=""
while true; do
    stamp=$(n_fits)
    changed=$(find "$ODIR" -maxdepth 1 -newermt "-${INTERVAL} seconds" 2>/dev/null \
                  | sort | tr '\n' ' ')
    if [ -n "$changed" ] || [ "$stamp" != "$last" ]; then
        if snapshot "checkpoint"; then
            say "checkpointed (${stamp}/${TOTAL} fits)"
        fi
        last="$stamp"
    fi
    sleep "$INTERVAL"
done
