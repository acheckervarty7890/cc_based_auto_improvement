#!/usr/bin/env bash
#
# failsafe_commit_arch_ablation.sh — periodically commit + push the output of
# scripts/arch_cluster_ablation.py so a ~13-hour run survives the container
# being wiped.
#
# WHY A SEPARATE SCRIPT FROM failsafe_commit.sh
#   That one is wired to the iterative_retrain pipeline: it takes --config and
#   --probe-out-dir per stage, watches redteam_done_iter*.marker and probe_iter*.pkl,
#   and calls a stage finished when its comparison CSV appears. The ablation has
#   none of those — it writes one CSV per (architecture, arm) job into
#   <work-dir>/results/ and nothing else. So the finish condition and the commit
#   set are different, and this is a much simpler poller.
#
# WHAT IT WATCHES
#   <work-dir>/results/*.csv — one per finished job. The count only ever grows
#   (stage_run skips jobs whose CSV exists), so it doubles as a progress counter.
#   A new CSV triggers an immediate checkpoint commit; a periodic snapshot runs
#   regardless so clusters.json and the log are captured between jobs too.
#
#   Failed jobs write <arch>__<run>.FAILED instead of a CSV and are counted
#   separately — they are results too, and a run that fails every job would
#   otherwise look like a run that had not started.
#
# WHEN IT EXITS
#   With --expect N, once N jobs have landed (CSV or FAILED) it takes a final
#   commit and exits 0. Get N from the run's own dry-run:
#       python scripts/arch_cluster_ablation.py --work-dir DIR --dry-run --stages run
#   Without --expect it polls until you kill it. Either way the EXIT trap takes a
#   final snapshot, so nothing is lost by stopping it.
#
# WHAT IS COMMITTED
#   Everything under the repo root, force-added, so .gitignore'd run outputs
#   (*.log, the results CSVs) ARE captured. That is the point.
#
# WHAT IS EXCLUDED  (and why)
#   *.pt / *.pth    Activation blobs. The eval cache alone is ~20 GB and every
#                   byte of it is re-downloadable from Kaggle.
#   archive/        99 GB of past experiment output in the local checkout. Never
#                   present on a fresh cloud clone, excluded so that running this
#                   script locally by mistake cannot try to commit it.
#   kaggle.json     Kaggle API credentials.
#   .env            Holds API keys; .gitignore excludes it deliberately and a
#                   force-add would push the secret.
#   .venv*/         Rebuildable, GBs, tens of thousands of files.
#   __pycache__/, *.pyc   Build noise.
#   >30 MB files    Backstop by SIZE for whatever the name rules did not
#                   anticipate. GitHub rejects a push with any file over 100 MB
#                   outright, which would strand every later snapshot too.
#
#   The exclusions are applied TWICE on purpose: directory pathspecs keep the big
#   trees out of the index in the first place (staging .venv then unstaging it
#   would still walk every file), and a `git reset` of the file globs afterwards
#   is the belt-and-suspenders pass — an unanchored `:(exclude,glob)**/*.pt` on
#   the ADD side silently drops every file rather than just the .pt ones.
#
# BRANCH MODEL
#   Commits and pushes onto the branch ALREADY checked out; it never creates or
#   switches branches. One run per branch.
#
# USAGE
#   nohup bash failsafe_commit_arch_ablation.sh --work-dir results/arch_abl \
#         --expect 240 > /tmp/failsafe_arch.out 2>&1 &
#
#   bash failsafe_commit_arch_ablation.sh --work-dir results/arch_abl --once
#   bash failsafe_commit_arch_ablation.sh --work-dir results/arch_abl --no-push
#
#   Stop it with Ctrl-C or `kill` — the trap takes one final snapshot first.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR=""
REMOTE="origin"
POLL_INTERVAL=60      # seconds between result-count checks
PERIODIC_INTERVAL=900 # seconds between snapshots taken regardless of progress
EXPECT=0              # 0 = poll until killed
DO_PUSH=1
ONCE=0
MAX_SIZE_MB=30

usage() {
    grep '^# ' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Flags:
  --work-dir DIR        arch_cluster_ablation.py --work-dir (REQUIRED)
  --repo DIR            repo checkout (default: $REPO_ROOT)
  --expect N            exit after N jobs land; 0 = poll forever (default: $EXPECT)
  --poll-interval SEC   result-count poll cadence (default: $POLL_INTERVAL)
  --periodic-interval SEC  snapshot cadence regardless of progress (default: $PERIODIC_INTERVAL)
  --remote NAME         git remote to push to (default: $REMOTE)
  --no-push             commit locally only
  --once                take one snapshot and exit
  --max-size-mb N       skip any file larger than N MB (default: $MAX_SIZE_MB; 0 = no limit)
  -h, --help            show this help
EOF
}

# A value-taking flag must be given its value here, not left to `shift 2`. With only
# one positional left `shift 2` is a no-op that returns non-zero, so under `set -u`
# (no -e) the loop would spin on the same argument forever rather than erroring.
need_arg() {
    [[ $# -ge 2 && -n "${2:-}" ]] || { echo "ERROR: $1 needs a value" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir)          need_arg "$@"; WORK_DIR="$2"; shift 2;;
        --repo)              need_arg "$@"; REPO_ROOT="$2"; shift 2;;
        --expect)            need_arg "$@"; EXPECT="$2"; shift 2;;
        --poll-interval)     need_arg "$@"; POLL_INTERVAL="$2"; shift 2;;
        --periodic-interval) need_arg "$@"; PERIODIC_INTERVAL="$2"; shift 2;;
        --remote)            need_arg "$@"; REMOTE="$2"; shift 2;;
        --no-push)           DO_PUSH=0; shift;;
        --once)              ONCE=1; shift;;
        --max-size-mb)       need_arg "$@"; MAX_SIZE_MB="$2"; shift 2;;
        -h|--help)           usage; exit 0;;
        *) echo "Unknown arg: $1" >&2; usage; exit 2;;
    esac
done

log() { echo "[failsafe-arch $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

for v in EXPECT POLL_INTERVAL PERIODIC_INTERVAL MAX_SIZE_MB; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || { log "ERROR: --${v,,} wants a whole number, got '${!v}'"; exit 2; }
done
MAX_SIZE_BYTES=$(( MAX_SIZE_MB * 1024 * 1024 ))

[[ -n "$WORK_DIR" ]] || { log "ERROR: --work-dir is required"; usage; exit 2; }
[[ -d "$REPO_ROOT" ]] || { log "ERROR: no such directory: $REPO_ROOT"; exit 1; }
cd "$REPO_ROOT" || exit 1
git rev-parse --git-dir >/dev/null 2>&1 || { log "ERROR: $REPO_ROOT is not a git repo"; exit 1; }
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Resolve --work-dir relative to the repo, matching how the ablation resolves it.
[[ "$WORK_DIR" = /* ]] || WORK_DIR="$REPO_ROOT/$WORK_DIR"
RESULTS_DIR="$WORK_DIR/results"

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
if [[ -z "$BRANCH" || "$BRANCH" == "HEAD" ]]; then
    log "ERROR: detached HEAD — check out a branch first"; exit 1
fi

ADD_EXCLUDES=(
    ':(exclude).venv'
    ':(exclude).venv_claude'
    ':(exclude)archive'
    ':(exclude)__pycache__'
)
RESET_EXCLUDES=(
    ':(glob)**/*.pt'
    ':(glob)**/*.pth'
    ':(glob)**/kaggle.json'
    ':(glob)**/.env'
    ':(glob)**/*.pyc'
    ':(glob)**/__pycache__/**'
)

# Jobs that have landed: a finished job leaves a CSV, a failed one a .FAILED.
# Counted rather than diffed because stage_run is resumable — the set only grows.
count_done() {
    local csv failed
    csv=$(find "$RESULTS_DIR" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
    failed=$(find "$RESULTS_DIR" -maxdepth 1 -name '*.FAILED' 2>/dev/null | wc -l)
    echo "$(( csv + failed )) $csv $failed"
}

# Exclusion pass by SIZE rather than by name — the backstop for whatever the two
# pathspec passes did not anticipate. Reads the size from DISK, not the index, so it
# cannot fire on deletions (--diff-filter=d already drops those).
LAST_OVERSIZE=""
drop_oversize() {
    (( MAX_SIZE_BYTES > 0 )) || return 0
    local f sz n=0 report=""
    while IFS= read -r -d '' f; do
        [[ -f "$f" ]] || continue
        sz=$(stat -c %s -- "$f" 2>/dev/null) || continue
        (( sz > MAX_SIZE_BYTES )) || continue
        git reset -q -- "$f" >/dev/null 2>&1 || true
        n=$(( n + 1 ))
        report+="$(printf '\n    %s (%s MB)' "$f" "$(( (sz + 1048575) / 1048576 ))")"
    done < <(git diff --cached --name-only -z --diff-filter=d)

    (( n > 0 )) || { LAST_OVERSIZE=""; return 0; }
    # Full list when it changes, one-liner when it is the same files again —
    # otherwise a single stranded blob reprints forever.
    if [[ "$report" != "$LAST_OVERSIZE" ]]; then
        log "skipping $n file(s) over ${MAX_SIZE_MB} MB:$report"
        LAST_OVERSIZE="$report"
    else
        log "skipping the same $n file(s) over ${MAX_SIZE_MB} MB"
    fi
}

PUSHED_UPSTREAM=0
snapshot() {
    local reason="$1"
    git add -f -A -- . "${ADD_EXCLUDES[@]}" >/dev/null 2>&1
    git reset -q -- "${RESET_EXCLUDES[@]}" >/dev/null 2>&1 || true
    drop_oversize

    if git diff --cached --quiet; then
        return 0   # nothing changed since the last snapshot
    fi

    local msg="failsafe(arch): $reason @ $(date '+%Y-%m-%dT%H:%M:%S')"
    if ! git commit -q -m "$msg" >/dev/null 2>&1; then
        log "WARN: git commit failed ($reason)"; return 0
    fi
    log "committed: $msg  ($(git rev-parse --short HEAD))"

    [[ "$DO_PUSH" -eq 1 ]] || return 0
    local push_args=("$REMOTE" "$BRANCH")
    [[ "$PUSHED_UPSTREAM" -eq 0 ]] && push_args=(-u "$REMOTE" "$BRANCH")
    if git push -q "${push_args[@]}" >/dev/null 2>&1; then
        PUSHED_UPSTREAM=1
        log "pushed to $REMOTE/$BRANCH"
    else
        sleep 5
        if git push -q "${push_args[@]}" >/dev/null 2>&1; then
            PUSHED_UPSTREAM=1
            log "pushed to $REMOTE/$BRANCH on retry"
        else
            log "WARN: push failed; commit is local, will retry next snapshot"
        fi
    fi
}

FINALIZED=0
finalize() {
    [[ "$FINALIZED" -eq 1 ]] && return
    FINALIZED=1
    log "finalizing: capturing latest state before exit"
    snapshot "final snapshot"
    log "done."
}
trap 'finalize; exit 0' INT TERM
trap 'finalize' EXIT

log "repo:      $REPO_ROOT"
log "branch:    $BRANCH$([[ "$DO_PUSH" -eq 1 ]] && echo " -> $REMOTE" || echo " (local only)")"
log "work dir:  $WORK_DIR"
log "expect:    $([[ "$EXPECT" -eq 0 ]] && echo "unset — polls until killed" || echo "$EXPECT jobs")"
log "cadence:   poll ${POLL_INTERVAL}s, snapshot ${PERIODIC_INTERVAL}s"
log "excluded:  *.pt, *.pth, kaggle.json, .env, .venv*/, archive/, __pycache__/, *.pyc"
log "size cap:  $([[ "$MAX_SIZE_MB" -eq 0 ]] && echo "none" || echo "${MAX_SIZE_MB} MB per file")"
[[ -d "$RESULTS_DIR" ]] || log "note: $RESULTS_DIR does not exist yet — the run has not reached stage 'run'"

snapshot "startup snapshot"
if [[ "$ONCE" -eq 1 ]]; then
    FINALIZED=1   # the startup snapshot already covered it; skip the EXIT-trap repeat
    log "done (--once)."
    exit 0
fi

# Sleep in short chunks. bash defers a trap until the running foreground command
# returns, so a plain `sleep $PERIODIC_INTERVAL` would delay the final snapshot by up
# to that long after a `kill` — and `kill` is the normal way to stop a nohup'd poller.
interruptible_sleep() {
    local left="$1"
    while (( left > 0 )); do
        (( left < 5 )) && { sleep "$left"; return; }
        sleep 5
        left=$(( left - 5 ))
    done
}

read -r LAST_DONE _ _ <<<"$(count_done)"
log "polling: $LAST_DONE job(s) already present"
SINCE_SNAPSHOT=0
while true; do
    interruptible_sleep "$POLL_INTERVAL"
    SINCE_SNAPSHOT=$(( SINCE_SNAPSHOT + POLL_INTERVAL ))
    read -r done csv failed <<<"$(count_done)"

    if (( done > LAST_DONE )); then
        log "progress: $done job(s) done ($csv ok, $failed failed)$([[ "$EXPECT" -gt 0 ]] && echo " of $EXPECT")"
        snapshot "checkpoint — $done job(s) done"
        LAST_DONE=$done
        SINCE_SNAPSHOT=0
    elif (( SINCE_SNAPSHOT >= PERIODIC_INTERVAL )); then
        snapshot "periodic snapshot — $done job(s) done"
        SINCE_SNAPSHOT=0
    fi

    if (( EXPECT > 0 && done >= EXPECT )); then
        log "all $EXPECT job(s) landed ($csv ok, $failed failed) — finishing"
        break
    fi
done
finalize
exit 0
