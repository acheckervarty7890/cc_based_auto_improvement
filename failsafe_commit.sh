#!/usr/bin/env bash
#
# failsafe_commit.sh — checkpoint iterative_generate run(s) to git so they can be
# RESUMED after the container is wiped.
#
# THIS IS THE dev_new_scaffolding VERSION. The scaffold it watches is the
# generate -> score -> retrain -> guide loop (scripts/iterative_generate.py), not the
# old red-team loop, and the resume state is different: there are no
# redteam_done_iter*.marker files any more. What --resume keys off now is
#   <probe-out-dir>/probe_iter{N}.pkl   the iteration to restart at
#   <run-dir>/batches.jsonl             which batches are already generated / scored
#   <run-dir>/guidance.jsonl            the directions (and the memo) for an iteration
# so those are exactly what this poller commits, and what it watches for change.
#
# MULTI-STAGE: one poller covers a whole sequence of runs. Here it is configured for
# the THREE arms launched by run_gemma27b_generate_arms.sh (high-stakes, human harm,
# instruction following). You give it N stages — each a (config, probe-out-dir,
# log-file) triple, in launch order — and it polls stage 1's artifacts until that run
# FINISHES, then hands itself over to stage 2, and so on. No second poller, no manual
# restart.
#
# "Finished" = the stage's comparison CSV (config output.comparison_csv) exists, which
# cli.py writes exactly ONCE, after the last iteration's eval — a true end-of-run
# signal, not a per-iteration one. As a backstop (e.g. a run launched WITHOUT --eval,
# which never writes a CSV) a stage also counts as finished when its log tail carries
# cli.py's closing "AUROC ledger:" line. Artifacts of already-finished stages stay in
# the commit set, so later checkpoints keep carrying them.
#
# Branch model: this commits/pushes onto the branch that is ALREADY checked out — it
# never switches branches. Create a fresh local branch per experiment before launching;
# for this experiment that branch is generator_experiment_1. One experiment (all its
# stages) per branch.
#
# This is a STANDALONE poller: you launch the training yourself (as in
# run_gemma27b_generate_arms.sh), then start this alongside it. It watches the ACTIVE
# stage for a new probe_iter*.pkl OR a grown batches.jsonl / guidance.jsonl — the exact
# points --resume keys off — and, on every change, force-adds all resume-critical
# artifacts + the stripped logs, commits, and pushes to origin on the current branch. A
# periodic fallback snapshot (default every 40 min) covers the long stretch inside one
# batch's generate-and-fit, so an interrupt mid-batch still leaves the run's state on
# the remote.
#
# Why force-add: .gitignore ignores *.log AND results* — i.e. both the logs and the
# entire results dir that holds batches.jsonl, guidance.jsonl, runlog.jsonl,
# auroc_history.csv and accepted_iter*.jsonl. Those are exactly what resume needs, so
# every add here uses `git add -f`.
#
# What is NOT committed: the activation caches (*.pt, multiple GB each). In these arms
# they live in cache_gen_gemma27b_*/ , outside every committed path, but the hard
# `**/*.pt` exclude on each add is kept anyway — the DEFAULT base-cache location is
# *inside* --probe-out-dir, so a config that drops output.base_activation_cache_dir
# would otherwise put gigabytes into the commit set. They are recompute-only by design
# (and for these arms the eval and dev blobs are re-fetchable from Kaggle), not resume
# state.
#
# Exit: once the LAST stage finishes, the poller makes a final commit and exits 0.
# Stopping it early by hand (Ctrl-C / kill) also makes a final commit via the trap.
#
# Defaults target the THREE arms launched by run_gemma27b_generate_arms.sh, in order:
#   stage 1 (high-stakes)   configs/gen_gemma27b_highstakes.md
#                           probes/gen_gemma27b_highstakes
#                           logs/run_gen_gemma27b_highstakes.log
#   stage 2 (human harm)    configs/gen_gemma27b_hu_harm.md
#                           probes/gen_gemma27b_hu_harm
#                           logs/run_gen_gemma27b_hu_harm.log
#   stage 3 (instructions)  configs/gen_gemma27b_instructions.md
#                           probes/gen_gemma27b_instructions
#                           logs/run_gen_gemma27b_instructions.log
#
# NOTE each arm runs FIVE iterations and fits six probes per iteration, so a stage is
# hours long: the per-batch and per-probe checkpoints are what make a wiped container
# resumable.
#
# Typical use on the remote box (two nohups):
#
#   # 0) check out this experiment's branch (the failsafe pushes onto whatever is
#   #    checked out — it does NOT create or switch branches for you).
#   git fetch origin && git checkout generator_experiment_1
#
#   # 1) launch ALL THREE arms (the runner runs them sequentially)
#   nohup bash run_gemma27b_generate_arms.sh > logs/run_generate_arms.out 2>&1 &
#
#   # 2) launch the ONE failsafe — it follows arm 1, then 2, then 3, then exits.
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &
#
# Single-run usage — pass one triple:
#   bash failsafe_commit.sh --config configs/foo.md --probe-out-dir probes/foo \
#                           --log-file logs/foo.log
#
# Custom multi-stage: repeat the three flags once per stage, IN ORDER. The Nth --config
# pairs with the Nth --probe-out-dir and the Nth --log-file:
#   bash failsafe_commit.sh \
#       --config configs/a.md --probe-out-dir probes/a --log-file logs/a.log \
#       --config configs/b.md --probe-out-dir probes/b --log-file logs/b.log
#
# To RESUME on a fresh container (check out the same branch you pushed to):
#   git fetch origin && git checkout generator_experiment_1
#   # then re-run the SAME runner; --resume (default) picks up from the latest
#   # probe_iterN.pkl, skips batches already scored in batches.jsonl and restores the
#   # directions from guidance.jsonl. Restarting this failsafe on that branch continues
#   # appending checkpoints — and it SKIPS stages whose comparison CSV is already
#   # present, so it resumes polling at the stage that was in flight.

set -uo pipefail

# --------------------------------------------------------------------------- #
# Defaults — one entry per stage, in launch order.
# --------------------------------------------------------------------------- #
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIGS=(
    "configs/gen_gemma27b_highstakes.md"
    "configs/gen_gemma27b_hu_harm.md"
    "configs/gen_gemma27b_instructions.md"
)
PROBE_DIRS=(
    "probes/gen_gemma27b_highstakes"
    "probes/gen_gemma27b_hu_harm"
    "probes/gen_gemma27b_instructions"
)
LOG_FILES=(
    "logs/run_gen_gemma27b_highstakes.log"
    "logs/run_gen_gemma27b_hu_harm.log"
    "logs/run_gen_gemma27b_instructions.log"
)

REMOTE="origin"
BRANCH=""   # set at startup to the CURRENTLY checked-out branch — never switched
PY="${REPO_ROOT}/.venv_claude/bin/python"

CLOSE_ON_FINISH=0              # --close-on-finish: run CLOSE_SCRIPT once every stage has finished
CLOSE_SCRIPT="close_this.sh"   # --close-script: what to run then (e.g. stop the cloud container)

POLL_INTERVAL=30        # seconds between checkpoint checks
PERIODIC_INTERVAL=2400  # 40 min between fallback snapshots.
                        #   The probe/batches/guidance poll above is the PRIMARY trigger and is
                        #   unaffected by this — a newly generated or scored batch, or a fresh
                        #   probe_iter*.pkl, still commits within POLL_INTERVAL seconds. This
                        #   only bounds how much a wipe can cost when none of those has landed
                        #   for a while, which under these arms is the inside of a single
                        #   generate-then-fit step. batches.jsonl is append-only and every
                        #   reader takes the newest row per (iteration, batch) key, so a re-run
                        #   after resume simply re-appends what was lost.

usage() {
    grep '^# ' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Flags (--config / --probe-out-dir / --log-file are REPEATABLE — one per stage,
in launch order; the first use of any of them clears the built-in defaults):
  --config PATH            iterative_generate config for a stage
  --probe-out-dir DIR      that stage's --probe-out-dir
  --log-file PATH          that stage's run log (committed stripped; also a finish backstop)
  --remote NAME            git remote (default: $REMOTE)
  --poll-interval SEC      checkpoint poll cadence (default: $POLL_INTERVAL)
  --periodic-interval SEC  fallback snapshot cadence (default: $PERIODIC_INTERVAL)
  --close-on-finish        after the LAST stage finishes and the final commit is
                           confirmed on the remote, run --close-script to shut the
                           box down. Never fires on Ctrl-C/SIGTERM, on a crashed
                           run, or while HEAD is unpushed.
  --close-script PATH      what --close-on-finish runs (default: $CLOSE_SCRIPT)
  -h, --help               show this help
EOF
}

# First user-supplied stage flag wipes the defaults, so a custom invocation never
# silently inherits the three built-in arms.
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
        --close-on-finish)   CLOSE_ON_FINISH=1; shift;;
        --close-script)      CLOSE_SCRIPT="$2"; shift 2;;
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
# relative to the config file). We commit each run DIRECTORY wholesale so
# batches.jsonl, guidance.jsonl, runlog.jsonl, auroc_history.csv,
# accepted_iter*.jsonl and the comparison CSV are all captured without
# hard-coding filenames.
# --------------------------------------------------------------------------- #
if [[ ! -x "$PY" ]]; then
    log "ERROR: venv python not found at $PY"; exit 1
fi

RUN_DIRS=(); CSV_PATHS=(); BASE_ACT_DIRS=(); EVAL_ACT_DIRS=()
for cfg in "${CONFIGS[@]}"; do
    readarray -t RESOLVED < <("$PY" - "$cfg" <<'PYEOF'
import sys
from pathlib import Path
from agentic_redteam.config import load_config
c = load_config(sys.argv[1])
def emit(p):
    print(str(Path(p).resolve()) if p else "")
emit(c.output.run_dir)                           # batches/guidance/runlog/history/accepted
emit(c.output.base_activation_cache_dir)         # training-side activations (optional)
emit(c.output.activations_cache_dir)             # eval activations (optional)
emit(c.output.comparison_csv)                    # comparison csv (optional)
PYEOF
)
    if [[ ${#RESOLVED[@]} -lt 1 || -z "${RESOLVED[0]}" ]]; then
        log "ERROR: could not resolve output paths from $cfg"; exit 1
    fi
    RUN_DIRS+=("${RESOLVED[0]}")
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

# Paths force-added on every commit (only those that exist are added) — the union over
# ALL stages, so a checkpoint taken during stage 3 still carries stages 1 and 2. The
# activation caches are deliberately absent: they are recompute-only (multi-GB *.pt),
# and any *.pt that lands *inside* one of these dirs is filtered out by the per-path
# *.pt exclude applied in do_commit.
COMMIT_PATHS=()
for i in $(seq 0 $((N_STAGES - 1))); do
    COMMIT_PATHS+=("${PROBE_DIRS[$i]}")   # probe_iter*.pkl (the iteration checkpoint) and
                                          #   candidates/probe_iter{i}_batch{k}.pkl
                                          #   (NOT resume state, but small and useful for
                                          #    re-reading a batch's candidate afterwards)
    COMMIT_PATHS+=("${RUN_DIRS[$i]}")     # batches.jsonl + guidance.jsonl (the resume state),
                                          #   runlog.jsonl, auroc_history.csv,
                                          #   accepted_iter*.jsonl, eval_comparison.csv
done
COMMIT_PATHS+=("logs_archive")            # STRIPPED run log(s) — see strip_logs below.
                                          #   NOT logs/: a raw run log is mostly tqdm
                                          #   carriage-return spam and has previously grown
                                          #   past GitHub's hard 100 MB per-file limit, at
                                          #   which point EVERY push failed and origin
                                          #   silently stopped being a resume point.

# Never stage activation blobs (*.pt/*.pth), wherever they land. The exclude pathspec is
# built PER PATH in do_commit and anchored to that path (":(exclude,glob)<path>/**/*.pt"):
# an UNANCHORED exclude like ":(exclude,glob)**/*.pt" silently drops *every* file, not just
# the .pt ones.

# --------------------------------------------------------------------------- #
# Commit/push onto the branch that is CURRENTLY checked out — never switch.
# --------------------------------------------------------------------------- #
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
if [[ -z "$BRANCH" || "$BRANCH" == "HEAD" ]]; then
    log "ERROR: detached HEAD (or not a git repo) — check out a branch first"; exit 1
fi

log "stages:        $N_STAGES (polled in order; hand-off when a stage's comparison CSV lands)"
for i in $(seq 0 $((N_STAGES - 1))); do
    log "  [$((i + 1))] config=${CONFIGS[$i]}"
    log "      probe-out-dir=${PROBE_DIRS[$i]}"
    log "      run-dir=${RUN_DIRS[$i]}"
    log "      log=${LOG_FILES[$i]}"
    log "      finish-signal=${CSV_PATHS[$i]}"
    log "      NOT committed: activation caches (*.pt) — base=${BASE_ACT_DIRS[$i]:-<none>} eval=${EVAL_ACT_DIRS[$i]:-<none>}"
done
log "branch:        $BRANCH -> $REMOTE   (current branch; not switched)"
if [[ "$CLOSE_ON_FINISH" -eq 1 ]]; then
    # Checked NOW, not at the end: the end is hours away and unattended, and a missing
    # close script discovered then means the box just stays up burning money.
    if [[ ! -f "$CLOSE_SCRIPT" ]]; then
        log "ERROR: --close-on-finish given but $CLOSE_SCRIPT does not exist"; exit 1
    fi
    log "close-on-finish: ENABLED -> $CLOSE_SCRIPT (only after ALL stages finish AND HEAD is on $REMOTE/$BRANCH)"
else
    log "close-on-finish: disabled (box stays up when the last stage finishes)"
fi

# --------------------------------------------------------------------------- #
# Commit helper: force-add the resume-critical paths and push. No-op when
# nothing changed. Never aborts the poller on a transient git/network failure.
# --------------------------------------------------------------------------- #
PUSHED_UPSTREAM=0
PUSH_FAIL_STREAK=0
LAST_PUSHED_SHA=""

# Write a tqdm-free copy of each stage's log into logs_archive/, and commit THAT instead
# of the raw file. Size: a raw log is ~99% carriage-return progress bars and has reached
# 115 MB in ~19 h on this project — over GitHub's hard 100 MB cap, from which point every
# push is rejected and origin freezes without the run noticing. Deltas: the stripped log
# is append-only text, which git deltas almost perfectly, so a checkpoint costs roughly
# the lines added since the previous one. The raw log stays on disk untouched.
strip_logs() {
    local f base
    mkdir -p logs_archive 2>/dev/null || return 0
    for f in "${LOG_FILES[@]}"; do
        [[ -f "$f" ]] || continue
        base="$(basename "${f%.log}")"
        tr -d '\r' < "$f" \
            | grep -av "^Epoch .*:.*|" \
            | grep -av "Processing batches:" \
            | grep -av "it/s\]$" \
            > "logs_archive/${base}.stripped.log" 2>/dev/null || true
    done
}

do_commit() {
    local reason="$1"
    local p
    strip_logs
    for p in "${COMMIT_PATHS[@]}"; do
        # Anchor the exclude to $p; an unanchored **/*.pt exclude drops everything.
        [[ -e "$p" ]] && git add -f "$p" \
            ":(exclude,glob)${p}/**/*.pt" ":(exclude,glob)${p}/**/*.pth" \
            >/dev/null 2>&1
    done
    # Belt-and-suspenders: unstage any activation blob that slipped through, so a commit
    # can never carry GBs. (Unanchored reset pathspec DOES work here.)
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
    #
    # The push output is CAPTURED and logged on failure. Sending it to /dev/null with only
    # "WARN: push failed" recorded is how a hard, never-self-healing rejection (a file over
    # GitHub's size limit) once looked exactly like a transient network blip for three
    # hours. A failsafe whose failures are indistinguishable from noise is not a failsafe.
    local push_args=("$REMOTE" "$BRANCH")
    [[ "$PUSHED_UPSTREAM" -eq 0 ]] && push_args=(-u "$REMOTE" "$BRANCH")
    local push_out rc
    push_out="$(git push "${push_args[@]}" 2>&1)"; rc=$?
    if [[ $rc -ne 0 ]]; then
        sleep 5
        push_out="$(git push "${push_args[@]}" 2>&1)"; rc=$?
    fi
    if [[ $rc -eq 0 ]]; then
        PUSHED_UPSTREAM=1
        PUSH_FAIL_STREAK=0
        LAST_PUSHED_SHA="$(git rev-parse --short HEAD)"
        log "pushed to $REMOTE/$BRANCH ($LAST_PUSHED_SHA)"
        return 0
    fi

    PUSH_FAIL_STREAK=$((PUSH_FAIL_STREAK + 1))
    log "ERROR: push #$PUSH_FAIL_STREAK FAILED — origin is NOT a resume point right now."
    log "ERROR:   local HEAD $(git rev-parse --short HEAD); last pushed ${LAST_PUSHED_SHA:-<never>}"
    while IFS= read -r line; do
        [[ -n "$line" ]] && log "ERROR:   git: $line"
    done <<< "$(printf '%s\n' "$push_out" | grep -aiv '^remote: *$' | head -6)"
    if printf '%s' "$push_out" | grep -qi "exceeds\|file size limit\|GH001\|pre-receive hook declined"; then
        log "ERROR:   This will NOT fix itself — a committed file is over the remote's"
        log "ERROR:   size limit. Every later checkpoint will fail too until history is"
        log "ERROR:   rewritten to drop it. Stop and fix now; do not let the run continue"
        log "ERROR:   believing it is checkpointed."
    fi
}

# Signature of the ACTIVE stage's resume state on disk. Three sources, all cheap:
#   - probe_iter*.pkl filenames  -> an iteration's union retrain finished
#   - batches.jsonl size         -> a batch was generated, or scored (a scored batch is a
#                                   SECOND appended row, so the size grows either way)
#   - guidance.jsonl size        -> the judge wrote the next iteration's directions
# Sizes, not hashes: these files are append-only, so a change is always a growth, and
# stat'ing them costs nothing at a 30 s cadence. This replaces the old scaffold's
# redteam_done_iter*.marker poll, which has no equivalent here.
checkpoint_signature() {  # $1 = probe-out-dir, $2 = run-dir
    { ls -1 "$1"/probe_iter*.pkl 2>/dev/null
      stat -c '%n %s' "$2/batches.jsonl" "$2/guidance.jsonl" 2>/dev/null; } \
      | sort | sha1sum | cut -d' ' -f1
}

# Has a stage finished? Primary signal: its comparison CSV exists — cli.py writes it
# once, after the final iteration's eval. Backstop for runs launched without --eval (no
# CSV is ever written): cli.py's closing "AUROC ledger:" line in the log. (The literal
# line begins with a non-ASCII delta, so match the ASCII tail of it.)
stage_finished() {  # $1 = csv path, $2 = log file
    [[ -n "$1" && -f "$1" ]] && return 0
    [[ -n "$2" && -f "$2" ]] && tail -n 40 "$2" 2>/dev/null | grep -aq 'AUROC ledger: ' && return 0
    return 1
}

# --------------------------------------------------------------------------- #
# Final commit on any exit (Ctrl-C, SIGTERM, or the last stage finishing).
# --------------------------------------------------------------------------- #
FINALIZED=0
ALL_STAGES_DONE=0   # set ONLY on normal completion of the last stage — see maybe_close_box

# Shut the box down, but only when it is genuinely safe to lose it. Three guards, each
# preventing an unrecoverable failure — the box is destroyed, not paused:
#
#   1. ALL_STAGES_DONE. Set only where the poller decides every stage finished. So a
#      Ctrl-C, a SIGTERM, or the poller dying for any other reason runs finalize() and
#      exits WITHOUT closing. Killing the failsafe must never kill the box.
#   2. HEAD is on the remote. Compared against the remote-TRACKING ref, which only our
#      own successful pushes advance, so this is a real "the results are off this
#      machine" check rather than a "we tried" one.
#   3. The commit set is what do_commit stages: probes, batches.jsonl, guidance.jsonl and
#      the rest of the run dir. Activation caches are deliberately NOT pushed, so closing
#      the box does lose those — recompute-only by design, and for these arms the eval and
#      dev blobs come back from Kaggle anyway.
#
# A crashed run never reaches ALL_STAGES_DONE either: the poller just keeps polling, so the
# box stays up for inspection. That is the intended asymmetry — an unfinished run is worth
# more than the machine time.
maybe_close_box() {
    [[ "$CLOSE_ON_FINISH" -eq 1 ]] || return 0
    if [[ "$ALL_STAGES_DONE" -ne 1 ]]; then
        log "NOT closing the box: the poller is exiting without every stage having finished."
        return 0
    fi
    local head remote_head
    head="$(git rev-parse HEAD 2>/dev/null)"
    # --verify -q: a plain `git rev-parse badref` ECHOES THE REF BACK, which would then be
    # compared against a sha and (harmlessly) never match — but it also lands a nonsense
    # string in the log. --verify makes a missing ref produce empty output instead.
    remote_head="$(git rev-parse --verify -q "$REMOTE/$BRANCH^{commit}" 2>/dev/null)"
    if [[ -z "$remote_head" || "$head" != "$remote_head" ]]; then
        local shown="${remote_head:0:7}"
        log "REFUSING to close the box: HEAD ${head:0:7} is NOT on $REMOTE/$BRANCH (remote-tracking ref: ${shown:-<none>})."
        log "REFUSING:   the run's results exist only on this machine. Push by hand, then close it."
        return 0
    fi
    log "all stages finished and ${head:0:7} is on $REMOTE/$BRANCH — closing the box via $CLOSE_SCRIPT"
    bash "$CLOSE_SCRIPT" 2>&1 | while IFS= read -r line; do log "close: $line"; done
    log "close request sent."
}

finalize() {
    [[ "$FINALIZED" -eq 1 ]] && return
    FINALIZED=1
    log "finalizing: capturing latest state before exit"
    do_commit "final snapshot"
    log "done."
    maybe_close_box
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
    ALL_STAGES_DONE=1
    finalize
    exit 0
fi

last_sig="$(checkpoint_signature "${PROBE_DIRS[$STAGE]}" "${RUN_DIRS[$STAGE]}")"
last_periodic="$(date +%s)"
# Commit whatever already exists at startup (e.g. resuming the failsafe itself).
do_commit "startup snapshot"

log "polling stage $((STAGE + 1))/$N_STAGES (${PROBE_DIRS[$STAGE]}, ${RUN_DIRS[$STAGE]}) every ${POLL_INTERVAL}s (periodic snapshot every ${PERIODIC_INTERVAL}s)"
while true; do
    sleep "$POLL_INTERVAL"

    # 1) New probe / new or re-scored batch / new guidance in the ACTIVE stage → checkpoint.
    sig="$(checkpoint_signature "${PROBE_DIRS[$STAGE]}" "${RUN_DIRS[$STAGE]}")"
    if [[ "$sig" != "$last_sig" ]]; then
        do_commit "checkpoint (new probe/batch/guidance, stage $((STAGE + 1)))"
        last_sig="$sig"
        last_periodic="$(date +%s)"
    else
        # 2) Fallback: periodically snapshot logs and anything else in flight.
        now="$(date +%s)"
        if (( now - last_periodic >= PERIODIC_INTERVAL )); then
            do_commit "periodic snapshot (stage $((STAGE + 1)))"
            last_periodic="$now"
        fi
    fi

    # 3) Active stage finished (its comparison CSV landed) → checkpoint it and hand over
    #    to the next stage. The finished stage's paths stay in COMMIT_PATHS, so later
    #    commits keep carrying its artifacts.
    if stage_finished "${CSV_PATHS[$STAGE]}" "${LOG_FILES[$STAGE]}"; then
        do_commit "stage $((STAGE + 1))/$N_STAGES finished (${CONFIGS[$STAGE]})"
        STAGE=$((STAGE + 1))
        if [[ "$STAGE" -ge "$N_STAGES" ]]; then
            log "last stage finished — exiting after final commit"
            ALL_STAGES_DONE=1
            exit 0   # EXIT trap runs finalize() -> maybe_close_box()
        fi
        log "handing over to stage $((STAGE + 1))/$N_STAGES: ${CONFIGS[$STAGE]} (${PROBE_DIRS[$STAGE]})"
        last_sig="$(checkpoint_signature "${PROBE_DIRS[$STAGE]}" "${RUN_DIRS[$STAGE]}")"
        last_periodic="$(date +%s)"
    fi
done
