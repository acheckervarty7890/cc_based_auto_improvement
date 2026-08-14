#!/usr/bin/env bash
#
# failsafe_commit_vintage.sh — keep the red-team vintage sweep's progress on the remote
# so a wiped container costs one fit, not the whole run.
#
# Sibling of failsafe_commit_attribution.sh, same reasoning, different payload. The
# vintage sweep's resume state is one append-only sidecar —
# results_hu_harm_gemma27b_batch_ablation/vintage/vintage_progress.jsonl, one fsync'd
# row per finished (arm, vintage, seed) fit — so this poller watches that directory and
# commits whenever it grows.
#
# WHAT IT COMMITS
#   results_hu_harm_gemma27b_batch_ablation/vintage/**   progress sidecar, per-arm JSON,
#       the gate + triage results, the CSVs, SUMMARY.md and the run logs (~1 MB)
#   scripts/attribution_vintage*.py, scripts/vintage_*.py   the code that produced them
#
# It also REGENERATES SUMMARY.md before each commit, so whatever lands on the remote
# explains itself: what a vintage is, how many fits are done, and the current
# mean +/- sd table. A checkpoint nobody can read without the session context is only
# half a backup.
#
# WHAT IT DOES NOT COMMIT: activations (`*.pt`, ~11 GB across the eval blobs and the
# per-conversation red-team cache). They are recompute-only inputs, are published on
# Kaggle, and one stray blob would make these checkpoints unpushable.
# Nor kaggle/kaggle.json — it holds live API credentials.
#
# WHY force-add: .gitignore carries both `results*` and `*.log`, which is exactly the
# material that has to survive. Same reasoning as the sibling scripts.
#
# BRANCH MODEL: commits and pushes onto whatever branch is already checked out.
#
# EXIT: Ctrl-C / SIGTERM makes a final commit via the trap.
#
# USAGE:
#   nohup bash failsafe_commit_vintage.sh > logs/failsafe_vintage.out 2>&1 &
#
#   INTERVAL=1200  seconds between polls (default 1200 = 20 min)
#   NO_PUSH=1      commit locally only

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

INTERVAL="${INTERVAL:-1200}"
VDIR=results_hu_harm_gemma27b_batch_ablation/vintage
PY=.venv_claude/bin/python
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

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

n_fits() { wc -l < "$VDIR/vintage_progress.jsonl" 2>/dev/null || echo 0; }

snapshot() {  # $1 = commit subject suffix
    local f n
    # Refresh the human-readable status first, so it is part of the same commit as the
    # data it describes rather than trailing it by one interval.
    "$PY" scripts/vintage_summary_md.py >/dev/null 2>&1 \
        || say "WARNING: SUMMARY.md regeneration failed"
    for f in "$VDIR"/*.jsonl "$VDIR"/*.json "$VDIR"/*.csv "$VDIR"/*.md "$VDIR"/*.log \
             scripts/attribution_vintage.py scripts/attribution_vintage_gate.py \
             scripts/vintage_triage.py scripts/vintage_summary_md.py \
             failsafe_commit_vintage.sh logs/failsafe_vintage.out; do
        stage "$f"
    done
    if git diff --cached --quiet; then
        return 1
    fi
    n=$(n_fits)
    git commit -q -m "failsafe(vintage): $1 — ${n} vintage fits done @ $(date -Is)"
    if [ -z "${NO_PUSH:-}" ]; then
        git push -q origin "$BRANCH" || say "push failed (committed locally; retry next poll)"
    fi
    return 0
}

finish() {
    say "stopping — final snapshot"
    snapshot "final" && say "committed" || say "nothing new to commit"
    exit 0
}
trap finish INT TERM

say "watching $VDIR on branch '$BRANCH', every ${INTERVAL}s (NO_PUSH=${NO_PUSH:-unset})"

# Self-test in minute one: the previous failure mode in this repo was a checkpointer
# that protected nothing and said so to no one.
for f in "$VDIR"/*.jsonl "$VDIR"/*.json "$VDIR"/*.csv; do stage "$f"; done
staged=$(git diff --cached --name-only -- "$VDIR" | wc -l)
on_disk=$(ls -1 "$VDIR"/*.jsonl "$VDIR"/*.json "$VDIR"/*.csv 2>/dev/null | wc -l)
tracked=$(git ls-files -- "$VDIR" | wc -l)
if [ "$on_disk" -gt 0 ] && [ "$staged" -eq 0 ] && [ "$tracked" -eq 0 ]; then
    say "FATAL: $on_disk result file(s) on disk, none stageable and none tracked."
    exit 1
fi
say "self-test: $on_disk result file(s) on disk, $staged newly staged, $tracked already tracked"

last=""
while true; do
    stamp=$(n_fits)
    changed=$(find "$VDIR" -maxdepth 1 -newermt "-${INTERVAL} seconds" \
                  \( -name '*.jsonl' -o -name '*.json' -o -name '*.csv' -o -name '*.log' \) \
                  2>/dev/null | sort | tr '\n' ' ')
    if [ -n "$changed" ] || [ "$stamp" != "$last" ]; then
        if snapshot "checkpoint"; then
            say "checkpointed (${stamp} fits)"
        fi
        last="$stamp"
    fi
    sleep "$INTERVAL"
done
