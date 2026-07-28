#!/usr/bin/env bash
#
# failsafe_commit.sh — checkpoint iterative_retrain run(s) to git so they can be
# RESUMED after the container is wiped.
#
# MULTI-STAGE: one poller covers a whole sequence of runs (e.g. both arms of an
# ablation launched by run_deepseekv4_memo_ablation.sh). You give it N stages —
# each a (config, probe-out-dir, log-file) triple, in launch order — and it polls
# stage 1's artifacts until that run FINISHES, then automatically hands itself
# over to stage 2, and so on. No second poller, no manual restart.
#
# "Finished" = the stage's comparison CSV (config output.comparison_csv) exists,
# which cli.py writes exactly ONCE, after the last iteration's eval — so it is a
# true end-of-run signal, not a per-iteration one. As a backstop (e.g. a run
# launched WITHOUT --eval, which never writes a CSV) a stage also counts as
# finished when its log tail contains cli.py's closing "Final probe:" line.
# Artifacts of already-finished stages stay in the commit set, so later
# checkpoints keep carrying them.
#
# Branch model: this commits/pushes onto the branch that is ALREADY checked out —
# it never switches branches. Create a fresh local branch per experiment before
# launching (e.g. `git checkout -b failsafe/<experiment>`); the failsafe records
# checkpoints on top of it. One experiment (all its stages) per branch.
#
# This is a STANDALONE poller: you launch the training yourself (as in
# run_deepseekv4_memo_ablation.sh), then start this alongside it. It watches the
# ACTIVE stage's --probe-out-dir for red-team **phase markers**
# (redteam_done_iter*.marker) and newly retrained probes (probe_iter*.pkl) — the
# exact points cli.py's --resume keys off — and, on every new one, force-adds all
# resume-critical artifacts + logs, commits, and pushes to origin on the current
# branch. A periodic fallback snapshot (default every 15 min) also captures
# red-team successes that accumulate in the JSONL *between* markers, so an
# interrupt mid-phase still leaves those (dedup-safe) successes on the remote for
# --resume to reload.
#
# Why force-add: .gitignore ignores *.log AND results* — i.e. both the logs and
# the entire results dir that holds the JSONL successes. Those are exactly what
# resume needs, so every add here uses `git add -f`.
#
# What is NOT committed: the base/eval activation caches (*.pt, multiple GB each —
# eval ~1.6G, base ~4G). They are a recompute-only optimization, not resume state,
# so they are excluded both by leaving their dirs out of the commit set AND by a
# hard `**/*.pt` exclude on every add (the DEFAULT base-cache location is *inside*
# --probe-out-dir, so the exclude guards configs that don't override it elsewhere).
#
# Exit: once the LAST stage finishes, the poller makes a final commit and exits 0.
# Stopping it early by hand (Ctrl-C / kill) also makes a final commit via the trap.
#
# Defaults target the deepseek-v4-flash memo ablation, in the order
# run_deepseekv4_memo_ablation.sh launches them:
#   stage 1 (nomemo)  configs/deepseekv4_hu_harm_llama70b50_nomemo.md
#                     probes/llama70b50_deepseekv4_nomemo
#                     logs/run_hh_llama70b50_deepseekv4_nomemo.log
#   stage 2 (memo)    configs/deepseekv4_hu_harm_llama70b50_memo.md
#                     probes/llama70b50_deepseekv4_memo
#                     logs/run_hh_llama70b50_deepseekv4_memo.log
#
# Typical use on the remote box (two terminals / two nohups):
#
#   # 0) create this experiment's branch (the failsafe pushes onto whatever is
#   #    checked out — it does NOT create or switch branches for you)
#   git checkout -b failsafe/deepseekv4-memo-ablation
#
#   # 1) launch BOTH arms (the runner runs them sequentially)
#   nohup bash run_deepseekv4_memo_ablation.sh > logs/run_deepseekv4_memo_ablation.out 2>&1 &
#
#   # 2) launch the ONE failsafe — it follows arm A, then arm B, then exits.
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &
#
# Single-run (old) usage is unchanged — pass one triple:
#   bash failsafe_commit.sh --config configs/foo.md --probe-out-dir probes/foo \
#                           --log-file logs/foo.log
#
# Custom multi-stage: repeat the three flags once per stage, IN ORDER. The Nth
# --config pairs with the Nth --probe-out-dir and the Nth --log-file:
#   bash failsafe_commit.sh \
#       --config configs/a.md --probe-out-dir probes/a --log-file logs/a.log \
#       --config configs/b.md --probe-out-dir probes/b --log-file logs/b.log
#
# To RESUME on a fresh container (check out the same branch you pushed to):
#   git fetch origin && git checkout failsafe/deepseekv4-memo-ablation
#   # then re-run the SAME runner/iterative_retrain.py command with --resume
#   # (default); it picks up from the latest probe_iterN.pkl and skips red-team
#   # phases whose markers were committed. Restarting this failsafe on that branch
#   # continues appending checkpoints — and it SKIPS stages whose comparison CSV
#   # is already present, so it resumes polling at the stage that was in flight.

set -uo pipefail

# --------------------------------------------------------------------------- #
# Defaults — one entry per stage, in launch order.
# --------------------------------------------------------------------------- #
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIGS=(
    "configs/deepseekv4_hu_harm_llama70b50_nomemo.md"
    "configs/deepseekv4_hu_harm_llama70b50_memo.md"
)
PROBE_DIRS=(
    "probes/llama70b50_deepseekv4_nomemo"
    "probes/llama70b50_deepseekv4_memo"
)
LOG_FILES=(
    "logs/run_hh_llama70b50_deepseekv4_nomemo.log"
    "logs/run_hh_llama70b50_deepseekv4_memo.log"
)

REMOTE="origin"
BRANCH=""   # set at startup to the CURRENTLY checked-out branch — never switched
PY="${REPO_ROOT}/.venv_claude/bin/python"

POLL_INTERVAL=30       # seconds between marker checks
PERIODIC_INTERVAL=900  # seconds between fallback snapshots (captures in-progress JSONL)

usage() {
    grep '^# ' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Flags (--config / --probe-out-dir / --log-file are REPEATABLE — one per stage,
in launch order; the first use of any of them clears the built-in defaults):
  --config PATH            iterative_retrain config for a stage
  --probe-out-dir DIR      that stage's --probe-out-dir
  --log-file PATH          that stage's run log (committed; also a finish backstop)
  --remote NAME            git remote (default: $REMOTE)
  --poll-interval SEC      marker poll cadence (default: $POLL_INTERVAL)
  --periodic-interval SEC  fallback snapshot cadence (default: $PERIODIC_INTERVAL)
  -h, --help               show this help
EOF
}

# First user-supplied stage flag wipes the defaults, so a custom invocation never
# silently inherits the deepseek arms.
USER_STAGES=0
clear_defaults_once() {
    if [[ "$USER_STAGES" -eq 0 ]]; then
        CONFIGS=(); PROBE_DIRS=(); LOG_FILES=(); USER_STAGES=1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)            clear_defaults_once; CONFIGS+=("$2");    shift 2;;
        --probe-out-dir)     clear_defaults_once; PROBE_DIRS+=("$2"); shift 2;;
        --log-file)          clear_defaults_once; LOG_FILES+=("$2");  shift 2;;
        --remote)            REMOTE="$2"; shift 2;;
        --poll-interval)     POLL_INTERVAL="$2"; shift 2;;
        --periodic-interval) PERIODIC_INTERVAL="$2"; shift 2;;
        -h|--help)           usage; exit 0;;
        *) echo "Unknown arg: $1" >&2; usage; exit 2;;
    esac
done

log() { echo "[failsafe $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

N_STAGES="${#CONFIGS[@]}"
if [[ "$N_STAGES" -eq 0 ]]; then
    log "ERROR: no stages configured"; exit 1
fi
if [[ "${#PROBE_DIRS[@]}" -ne "$N_STAGES" || "${#LOG_FILES[@]}" -ne "$N_STAGES" ]]; then
    log "ERROR: stage flags must be given the same number of times — got ${#CONFIGS[@]} --config, ${#PROBE_DIRS[@]} --probe-out-dir, ${#LOG_FILES[@]} --log-file"
    exit 1
fi

# --------------------------------------------------------------------------- #
# Resolve output paths from each config (config.py resolves them to absolute,
# relative to the config file). We commit the results DIRECTORIES wholesale so
# every _fp/_fn JSONL, sidecar (.summaries.jsonl/.iteration_memos.jsonl), marker,
# probe, contrastive cache, postprocessed snapshot and comparison CSV is captured
# without hard-coding filenames.
# --------------------------------------------------------------------------- #
if [[ ! -x "$PY" ]]; then
    log "ERROR: venv python not found at $PY"; exit 1
fi

RESULTS_DIRS=(); CSV_PATHS=(); BASE_ACT_DIRS=(); EVAL_ACT_DIRS=()
for cfg in "${CONFIGS[@]}"; do
    readarray -t RESOLVED < <("$PY" - "$cfg" <<'PYEOF'
import sys
from pathlib import Path
from agentic_redteam.config import load_config
c = load_config(sys.argv[1])
def emit(p):
    print(str(Path(p).resolve()) if p else "")
emit(c.output.jsonl_path.parent)                 # results dir (holds successes + sidecars)
emit(c.output.base_activation_cache_dir)         # base activations (optional)
emit(c.output.activations_cache_dir)             # eval activations (optional)
emit(c.output.comparison_csv)                    # comparison csv (optional)
PYEOF
)
    if [[ ${#RESOLVED[@]} -lt 1 || -z "${RESOLVED[0]}" ]]; then
        log "ERROR: could not resolve output paths from $cfg"; exit 1
    fi
    RESULTS_DIRS+=("${RESOLVED[0]}")
    BASE_ACT_DIRS+=("${RESOLVED[1]:-}")   # logged only — NOT committed (GBs of *.pt)
    EVAL_ACT_DIRS+=("${RESOLVED[2]:-}")   # logged only — NOT committed (GBs of *.pt)
    csv="${RESOLVED[3]:-}"
    if [[ -z "$csv" ]]; then
        # cli.py's fallback when output.comparison_csv is unset and --comparison-csv
        # isn't passed: <--results-dir>/iter_run_comparison.csv, default results/.
        csv="${REPO_ROOT}/results/iter_run_comparison.csv"
        log "NOTE: $cfg sets no output.comparison_csv — assuming $csv for stage-finish detection"
    fi
    CSV_PATHS+=("$csv")
done

# Paths force-added on every commit (only those that exist are added) — the union
# over ALL stages, so a checkpoint taken during stage 2 still carries stage 1's
# artifacts. The base and eval activation caches are deliberately absent: they are
# recompute-only (multi-GB *.pt), and any *.pt that lives *inside* one of these
# dirs (e.g. the default <probe-out-dir>/base_activation_cache) is filtered out by
# the per-path *.pt exclude applied in do_commit.
COMMIT_PATHS=()
for i in $(seq 0 $((N_STAGES - 1))); do
    COMMIT_PATHS+=("${PROBE_DIRS[$i]}")     # probe_iter*.pkl, redteam_done_*.marker,
                                            #   contrastive_cache.jsonl, redteam_postprocessed_iter*.jsonl
                                            #   (NOT the *.pt activation blobs)
    COMMIT_PATHS+=("${RESULTS_DIRS[$i]}")   # *_fp.jsonl / *_fn.jsonl successes +
                                            #   .summaries/.iteration_memos sidecars + comparison csv
done
COMMIT_PATHS+=("logs")                      # run log(s) — *.log is gitignored, force-added

# Never stage activation blobs (*.pt/*.pth), wherever they land. The exclude
# pathspec is built PER PATH in do_commit and anchored to that path
# (":(exclude,glob)<path>/**/*.pt"): an UNANCHORED exclude like
# ":(exclude,glob)**/*.pt" silently drops *every* file, not just the .pt ones.

# --------------------------------------------------------------------------- #
# Commit/push onto the branch that is CURRENTLY checked out — never switch. You
# create a fresh local branch per experiment before launching this; the failsafe
# just records checkpoints on top of it. (This means it does touch the same
# working branch the run was launched from, so keep one experiment per branch.)
# --------------------------------------------------------------------------- #
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
if [[ -z "$BRANCH" || "$BRANCH" == "HEAD" ]]; then
    log "ERROR: detached HEAD (or not a git repo) — check out a branch first"; exit 1
fi

log "stages:        $N_STAGES (polled in order; hand-off when a stage's comparison CSV lands)"
for i in $(seq 0 $((N_STAGES - 1))); do
    log "  [$((i + 1))] config=${CONFIGS[$i]}"
    log "      probe-out-dir=${PROBE_DIRS[$i]}"
    log "      results=${RESULTS_DIRS[$i]}"
    log "      log=${LOG_FILES[$i]}"
    log "      finish-signal=${CSV_PATHS[$i]}"
    log "      NOT committed: activation caches (*.pt) — base=${BASE_ACT_DIRS[$i]:-<none>} eval=${EVAL_ACT_DIRS[$i]:-<none>}"
done
log "branch:        $BRANCH -> $REMOTE   (current branch; not switched)"

# --------------------------------------------------------------------------- #
# Commit helper: force-add the resume-critical paths and push. No-op when
# nothing changed. Never aborts the poller on a transient git/network failure.
# --------------------------------------------------------------------------- #
PUSHED_UPSTREAM=0
do_commit() {
    local reason="$1"
    local p
    for p in "${COMMIT_PATHS[@]}"; do
        # Anchor the exclude to $p; an unanchored **/*.pt exclude drops everything.
        [[ -e "$p" ]] && git add -f "$p" \
            ":(exclude,glob)${p}/**/*.pt" ":(exclude,glob)${p}/**/*.pth" \
            >/dev/null 2>&1
    done
    # Belt-and-suspenders: unstage any activation blob that slipped through, so a
    # commit can never carry GBs. (Unanchored reset pathspec DOES work here.)
    git reset -q -- ':(glob)**/*.pt' ':(glob)**/*.pth' >/dev/null 2>&1 || true
    if git diff --cached --quiet; then
        return 0   # nothing new to checkpoint
    fi
    local msg="failsafe: $reason @ $(date '+%Y-%m-%dT%H:%M:%S')"
    if ! git commit -q -m "$msg" >/dev/null 2>&1; then
        log "WARN: git commit failed ($reason)"; return 0
    fi
    log "committed: $msg"
    # Push (set upstream on first successful push); retry once on failure.
    local push_args=("$REMOTE" "$BRANCH")
    [[ "$PUSHED_UPSTREAM" -eq 0 ]] && push_args=(-u "$REMOTE" "$BRANCH")
    if git push -q "${push_args[@]}" >/dev/null 2>&1; then
        PUSHED_UPSTREAM=1
        log "pushed to $REMOTE/$BRANCH ($(git rev-parse --short HEAD))"
    else
        sleep 5
        if git push -q "${push_args[@]}" >/dev/null 2>&1; then
            PUSHED_UPSTREAM=1
            log "pushed to $REMOTE/$BRANCH on retry ($(git rev-parse --short HEAD))"
        else
            log "WARN: push failed; commit is local, will retry on next checkpoint"
        fi
    fi
}

# Signature of the ACTIVE stage's resume checkpoints on disk: sorted marker +
# probe filenames. A change means a red-team phase finished or a probe was
# retrained.
checkpoint_signature() {  # $1 = probe-out-dir
    { ls -1 "$1"/redteam_done_iter*.marker 2>/dev/null
      ls -1 "$1"/probe_iter*.pkl 2>/dev/null; } \
      | sort | sha1sum | cut -d' ' -f1
}

# Has a stage finished? Primary signal: its comparison CSV exists — cli.py writes
# it once, after the final iteration's eval. Backstop for runs launched without
# --eval (no CSV is ever written): cli.py's closing "Final probe:" line in the log.
stage_finished() {  # $1 = csv path, $2 = log file
    [[ -n "$1" && -f "$1" ]] && return 0
    [[ -n "$2" && -f "$2" ]] && tail -n 40 "$2" 2>/dev/null | grep -q '^Final probe: ' && return 0
    return 1
}

# --------------------------------------------------------------------------- #
# Final commit on any exit (Ctrl-C, SIGTERM, or the last stage finishing).
# --------------------------------------------------------------------------- #
FINALIZED=0
finalize() {
    [[ "$FINALIZED" -eq 1 ]] && return
    FINALIZED=1
    log "finalizing: capturing latest state before exit"
    do_commit "final snapshot"
    log "done."
}
trap 'finalize; exit 0' INT TERM
trap 'finalize' EXIT

# --------------------------------------------------------------------------- #
# Poll loop — one active stage at a time, advancing on finish.
# --------------------------------------------------------------------------- #
STAGE=0
# On a restart, skip stages that already finished so we resume on the in-flight one.
while [[ "$STAGE" -lt "$N_STAGES" ]] && stage_finished "${CSV_PATHS[$STAGE]}" "${LOG_FILES[$STAGE]}"; do
    log "stage $((STAGE + 1)) (${CONFIGS[$STAGE]}) already finished — skipping"
    STAGE=$((STAGE + 1))
done
if [[ "$STAGE" -ge "$N_STAGES" ]]; then
    log "all $N_STAGES stage(s) already finished — committing final state and exiting"
    finalize
    exit 0
fi

last_sig="$(checkpoint_signature "${PROBE_DIRS[$STAGE]}")"
last_periodic="$(date +%s)"
# Commit whatever already exists at startup (e.g. resuming the failsafe itself).
do_commit "startup snapshot"

log "polling stage $((STAGE + 1))/$N_STAGES (${PROBE_DIRS[$STAGE]}) every ${POLL_INTERVAL}s (periodic snapshot every ${PERIODIC_INTERVAL}s)"
while true; do
    sleep "$POLL_INTERVAL"

    # 1) New marker / retrained probe in the ACTIVE stage → checkpoint commit.
    sig="$(checkpoint_signature "${PROBE_DIRS[$STAGE]}")"
    if [[ "$sig" != "$last_sig" ]]; then
        do_commit "checkpoint (new marker/probe, stage $((STAGE + 1)))"
        last_sig="$sig"
        last_periodic="$(date +%s)"
    else
        # 2) Fallback: periodically snapshot in-progress successes/logs.
        now="$(date +%s)"
        if (( now - last_periodic >= PERIODIC_INTERVAL )); then
            do_commit "periodic snapshot (stage $((STAGE + 1)))"
            last_periodic="$now"
        fi
    fi

    # 3) Active stage finished (its comparison CSV landed) → checkpoint it and
    #    hand over to the next stage. The finished stage's paths stay in
    #    COMMIT_PATHS, so later commits keep carrying its artifacts.
    if stage_finished "${CSV_PATHS[$STAGE]}" "${LOG_FILES[$STAGE]}"; then
        do_commit "stage $((STAGE + 1))/$N_STAGES finished (${CONFIGS[$STAGE]})"
        STAGE=$((STAGE + 1))
        if [[ "$STAGE" -ge "$N_STAGES" ]]; then
            log "last stage finished — exiting after final commit"
            exit 0   # EXIT trap runs finalize()
        fi
        log "handing over to stage $((STAGE + 1))/$N_STAGES: ${CONFIGS[$STAGE]} (${PROBE_DIRS[$STAGE]})"
        last_sig="$(checkpoint_signature "${PROBE_DIRS[$STAGE]}")"
        last_periodic="$(date +%s)"
    fi
done
