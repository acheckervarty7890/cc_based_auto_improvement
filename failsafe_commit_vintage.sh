#!/usr/bin/env bash
#
# failsafe_commit_vintage.sh — keep run_vintage_hs_gemma27b.sh's progress on the remote so
# a wiped container costs one fit, not the sweep.
#
# The sibling failsafe_commit.sh checkpoints *iterative_retrain* runs and keys off
# artifacts that exist only there (phase markers, probe_iter*.pkl, a comparison CSV). The
# vintage sweep has none of those. Its resume state is two things — the append-only
# progress sidecar (one fsync'd row per finished (arm, vintage, seed)) and the per-fit
# classifier checkpoints under fits/ — so this poller is correspondingly simpler: watch
# the vintage dir, regenerate SUMMARY.md, and commit whenever anything changed.
#
# WHAT IT COMMITS
#   results_hs_gemma27b_batch_ablation/vintage/*.jsonl   the progress sidecar (resume state)
#   results_hs_gemma27b_batch_ablation/vintage/*.json    the per-arm membership reports
#   results_hs_gemma27b_batch_ablation/vintage/*.csv     the per-fit and aggregate tables
#   results_hs_gemma27b_batch_ablation/vintage/SUMMARY.md  regenerated every poll
#   results_hs_gemma27b_batch_ablation/vintage/fits/*.pt   trained classifiers, 21 KB each
#   logs/vintage_hs*.log                                  the run log
#
# WHY THE fits/*.pt ARE AN EXCEPTION. The sibling checkpointer excludes `*.pt` outright,
# because there the only .pt files are multi-GB activation blobs that are published on
# Kaggle anyway. Here fits/ holds 5376-float classifier state dicts — genuine resume state
# that saves a ~4-minute refit each. They are staged through a hard SIZE GUARD
# (MAX_PT_BYTES, 5 MB) rather than by trusting the directory, so if an activation blob ever
# lands in that tree it is skipped and named instead of pushing gigabytes to the remote.
#
# WHAT IT NEVER COMMITS: the activation caches (base_activations/, eval_activations/) —
# ~24 GB, on Kaggle already, and a recompute-only input rather than resume state. They are
# never named by any glob here. Nor kaggle/ or HFtokn.txt, which hold credentials.
#
# WHY force-add: .gitignore has `results*`, which is exactly the material that has to
# survive. Same reasoning as failsafe_commit.sh.
#
# BRANCH MODEL: commits and pushes onto whatever branch is already checked out. It never
# switches branches.
#
# EXIT: Ctrl-C / SIGTERM makes a final commit via the trap, so stopping it by hand does not
# strand the last interval's work.
#
# USAGE (alongside the sweep):
#   nohup bash failsafe_commit_vintage.sh > logs/failsafe_vintage.out 2>&1 &
#
#   INTERVAL=1800  seconds between polls (default 1800 — every 30 minutes)
#   NO_PUSH=1      commit locally only (e.g. no credentials on the box)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

INTERVAL="${INTERVAL:-1800}"
MAX_PT_BYTES="${MAX_PT_BYTES:-5000000}"
VINTAGE_DIR=results_hs_gemma27b_batch_ablation/vintage
PY=.venv_claude/bin/python
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

mkdir -p logs "$VINTAGE_DIR"

say() { echo ">>> $(date -Is)  $*"; }

# Stage one path, loudly. Errors are REPORTED, never swallowed: an earlier version of the
# sibling script ran `git add -f ... 2>/dev/null || true`, which worked on the box it was
# written on and silently added nothing on the box it ran on — so 30+ checkpoints committed
# the logs and none of the results, and the failure was invisible precisely because stderr
# was discarded. A checkpointer that cannot save the thing it exists to save has to say so.
stage() {
    local f="$1" err sz
    [ -e "$f" ] || return 0
    case "$f" in
        *.pt)
            sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
            if [ "$sz" -gt "$MAX_PT_BYTES" ]; then
                say "WARNING: skipping $f — ${sz} bytes exceeds MAX_PT_BYTES=${MAX_PT_BYTES}"
                return 0
            fi
            ;;
    esac
    if ! err=$(git add -f -- "$f" 2>&1); then
        say "WARNING: could not stage $f: $err"
        return 1
    fi
    return 0
}

stage_all() {
    local f
    # Explicit per-file globs rather than a directory plus an :(exclude) pathspec. The
    # results are the payload, so they are staged first and each failure is named.
    for f in "$VINTAGE_DIR"/*.jsonl "$VINTAGE_DIR"/*.json "$VINTAGE_DIR"/*.csv \
             "$VINTAGE_DIR"/SUMMARY.md "$VINTAGE_DIR"/fits/*.pt \
             logs/vintage_hs*.log logs/failsafe_vintage.out; do
        stage "$f"
    done
}

n_fits() { wc -l < "$VINTAGE_DIR/vintage_progress.jsonl" 2>/dev/null || echo 0; }

snapshot() {  # $1 = commit subject suffix
    # Regenerate the write-up first, so whatever is on the remote explains itself without
    # anyone having to re-read the sidecar. It reads only the sidecar, so it is correct
    # mid-run and costs milliseconds.
    $PY scripts/vintage_summary_md.py >/dev/null 2>&1 || say "WARNING: SUMMARY.md regen failed"
    stage_all
    if git diff --cached --quiet; then
        return 1
    fi
    git -c user.name="$(git config user.name || echo failsafe)" \
        -c user.email="$(git config user.email || echo failsafe@local)" \
        commit -q -m "failsafe(vintage): $1 — $(n_fits) fit(s) @ $(date -Is)"
    if [ -z "${NO_PUSH:-}" ]; then
        git push -q origin "$BRANCH" || say "push failed (committed locally; will retry next poll)"
    fi
    return 0
}

finish() {
    # The poll parks in `sleep $INTERVAL`, and bash defers a trap until the current
    # FOREGROUND child returns — so signalling this script while it sleeps would leave
    # the final snapshot waiting up to a full interval (30 min at the default). The loop
    # backgrounds the sleep and `wait`s on it instead, which a signal does interrupt;
    # this kills it so the trap body runs immediately.
    [ -n "${SLEEP_PID:-}" ] && kill "$SLEEP_PID" 2>/dev/null
    say "stopping — final snapshot"
    snapshot "final" && say "committed" || say "nothing new to commit"
    exit 0
}
trap finish INT TERM

say "watching $VINTAGE_DIR on branch '$BRANCH', every ${INTERVAL}s (NO_PUSH=${NO_PUSH:-unset})"

# Self-test at startup: stage once and report whether the RESULTS (not just the logs)
# actually made it into the index. The failure mode this guards against was silent for
# hours, so the state of the thing being protected is checked in minute one, out loud.
stage_all
staged_results=$(git diff --cached --name-only -- "$VINTAGE_DIR" | wc -l)
on_disk=$(ls -1 "$VINTAGE_DIR"/*.jsonl "$VINTAGE_DIR"/*.json "$VINTAGE_DIR"/*.csv 2>/dev/null | wc -l)
tracked=$(git ls-files -- "$VINTAGE_DIR" | wc -l)
if [ "$on_disk" -gt 0 ] && [ "$staged_results" -eq 0 ] && [ "$tracked" -eq 0 ]; then
    say "FATAL: $on_disk result file(s) on disk, none stageable and none tracked."
    say "       Checkpointing would silently protect nothing. Fix git first."
    exit 1
fi
say "self-test: $on_disk result file(s) on disk, $staged_results newly staged, $tracked already tracked"

# Poll rather than inotify: the signal we want is "a fit landed", which is a file append,
# and the interval already bounds the loss. Simpler, and it works the same on a container
# without inotify tooling installed. The fit count is the primary trigger; the mtime sweep
# catches the phase where fits/ is filling but no progress row has been written yet.
last=""
while true; do
    stamp=$(n_fits)
    touched=$(find "$VINTAGE_DIR" logs -maxdepth 2 -newermt "-${INTERVAL} seconds" \
                  \( -name '*.jsonl' -o -name '*.json' -o -name '*.csv' -o -name '*.pt' \
                     -o -name 'vintage_hs*' \) 2>/dev/null | head -1)
    if [ -n "$touched" ] || [ "$stamp" != "$last" ]; then
        if snapshot "checkpoint"; then
            say "checkpointed (${stamp} fits)"
        fi
        last="$stamp"
    fi
    # Backgrounded + waited, not a plain `sleep`, so SIGTERM reaches the trap now rather
    # than at the end of the interval. See finish().
    sleep "$INTERVAL" &
    SLEEP_PID=$!
    wait "$SLEEP_PID" 2>/dev/null
done
