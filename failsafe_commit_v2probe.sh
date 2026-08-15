#!/usr/bin/env bash
#
# failsafe_commit_v2probe.sh — keep scripts/v2_probe_on_new_v3.py's progress on the
# remote so a wiped container costs one fit, not the whole run.
#
# Sibling of failsafe_commit_vintage.sh, same reasoning, different payload. That one
# reports "N vintage fits done" off vintage_progress.jsonl and regenerates the vintage
# SUMMARY.md; this run writes different sidecars into the same directory and would be
# mis-described by those counters, hence a separate poller rather than a wider glob.
#
# The resume state is two append-only sidecars, one per arm —
# <vintage-dir>/{gptoss120b,deepseekv4pro}_v2probe_on_new_v3.progress.jsonl, one fsync'd
# row per finished (arm, seed) refit carrying that seed's logits on the held-out
# new-in-v3 rows. 20 fits at ~5-7 min each is ~2 h, so this watches those files and
# commits whenever they grow.
#
# WHAT IT COMMITS
#   <vintage-dir>/*v2probe_on_new_v3*        the two progress sidecars + the final JSON
#   scripts/v2_probe_on_new_v3.py            the code that produced them
#   logs/v2probe_on_new_v3.log               the run log (per-seed mislabel rates)
#   failsafe_commit_v2probe.sh, logs/failsafe_v2probe.out
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
# TARGET_BRANCH (default experiment11_cloud) via `HEAD:$TARGET_BRANCH`. The two differ
# because this payload belongs to the experiment11_cloud run while the working tree is
# checked out elsewhere, and switching branches mid-run would swap the probes and
# red-team dumps out from under the sweep — it reassembles per arm, so the second arm
# would read a different tree than the first.
#
# The push is a plain fast-forward: no --force, ever. The self-test refuses to start
# unless origin/$TARGET_BRANCH is an ancestor of HEAD, so this can only ever advance
# that branch and never rewrite someone else's commits. If the check fails, or if the
# branch moves under us mid-run, the push is rejected and reported rather than forced.
#
# EXIT: Ctrl-C / SIGTERM makes a final commit via the trap. NOTE bash defers a trap
# until the current foreground command finishes, so a SIGTERM during the poll `sleep`
# is not acted on until that sleep elapses — use SIGKILL if you need it to stop now
# and are willing to skip the final snapshot.
#
# USAGE:
#   nohup bash failsafe_commit_v2probe.sh > logs/failsafe_v2probe.out 2>&1 &
#
#   INTERVAL=1200       seconds between polls (default 1200 = 20 min)
#   TARGET_BRANCH=...   remote branch to push to (default experiment11_cloud)
#   NO_PUSH=1           commit locally only

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

INTERVAL="${INTERVAL:-1200}"
VDIR=results_hu_harm_gemma27b_batch_ablation/vintage
LOG=logs/v2probe_on_new_v3.log
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
TARGET_BRANCH="${TARGET_BRANCH:-experiment11_cloud}"

mkdir -p logs "$VDIR"

say() { echo ">>> $(date -Is)  $*"; }

# Stage one path, loudly. Errors are REPORTED, never swallowed — the sibling script's
# first version discarded stderr and silently committed nothing but logs for hours.
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

# One row per finished (arm, seed) refit, summed across both arms' sidecars.
n_fits() { cat "$VDIR"/*_v2probe_on_new_v3.progress.jsonl 2>/dev/null | wc -l; }

snapshot() {  # $1 = commit subject suffix
    local f n
    for f in "$VDIR"/*v2probe_on_new_v3* scripts/v2_probe_on_new_v3.py \
             "$LOG" failsafe_commit_v2probe.sh logs/failsafe_v2probe.out; do
        stage "$f"
    done
    if git diff --cached --quiet; then
        return 1
    fi
    n=$(n_fits)
    git commit -q -m "failsafe(v2probe): $1 — ${n}/20 seed refits done @ $(date -Is)"
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

say "watching $VDIR, committing on '$BRANCH', pushing to origin/$TARGET_BRANCH, every ${INTERVAL}s (NO_PUSH=${NO_PUSH:-unset})"

# Refuse to run at all unless the push can only fast-forward the target. A checkpointer
# that force-pushes is worse than none: these commits land unattended every 20 min, and
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
for f in "$VDIR"/*v2probe_on_new_v3* scripts/v2_probe_on_new_v3.py; do stage "$f"; done
staged=$(git diff --cached --name-only | wc -l)
on_disk=$(ls -1 "$VDIR"/*v2probe_on_new_v3* scripts/v2_probe_on_new_v3.py 2>/dev/null | wc -l)
if [ "$on_disk" -eq 0 ]; then
    say "FATAL: nothing matching the payload globs exists yet — wrong cwd, or the run never started."
    exit 1
fi
say "self-test: $on_disk payload file(s) on disk, $staged staged, $(n_fits) refit(s) recorded"

last=""
while true; do
    stamp=$(n_fits)
    changed=$(find "$VDIR" -maxdepth 1 -newermt "-${INTERVAL} seconds" \
                  -name '*v2probe_on_new_v3*' 2>/dev/null | sort | tr '\n' ' ')
    if [ -n "$changed" ] || [ "$stamp" != "$last" ]; then
        if snapshot "checkpoint"; then
            say "checkpointed (${stamp}/20 refits)"
        fi
        last="$stamp"
    fi
    sleep "$INTERVAL"
done
