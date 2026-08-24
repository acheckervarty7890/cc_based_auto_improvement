#!/usr/bin/env bash
#
# failsafe_commit.sh — checkpoint iterative_retrain run(s) to git so they can be
# RESUMED after the container is wiped.
#
# MULTI-STAGE: one poller covers a whole sequence of runs. Here it is configured
# for the SINGLE stage launched by run_gemma27b_hu_harm_evaldesc_measured.sh
# (experiment23 arm 3 with a rewritten eval.data_description).
# You give it N stages —
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
# run_gemma27b_hu_harm_evaldesc_measured.sh), then start this alongside it. It watches the
# ACTIVE stage's --probe-out-dir for red-team **phase markers**
# (redteam_done_iter*.marker) and newly retrained probes (probe_iter*.pkl) — the
# exact points cli.py's --resume keys off — and, on every new one, force-adds all
# resume-critical artifacts + logs, commits, and pushes to origin on the current
# branch. A periodic fallback snapshot (default every 40 min) also captures
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
# Defaults target the ONE arm launched by run_gemma27b_hu_harm_evaldesc_measured.sh:
# experiment23's memo-ladder arm 3 (human-harm concept, gemma-3-27b-it L32, 10-member
# ensemble, attacker openai/gpt-oss-120b) re-run with `eval.data_description` rewritten:
#   stage 1  configs/gptoss120b_hu_harm_gemma27b_ens10_devval_s3_itermemo150_evaldesc_measured.md
#            probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc_measured
#            logs/run_hu_harm_gemma27b_gptoss120b_s3_evaldesc_measured.log
#
# NOTE this arm runs TEN iterations, so the stage is long: the poller's per-marker and
# per-probe checkpoints matter more here than they did on the 5-iteration runs.
#
# Typical use on the remote box (two terminals / two nohups):
#
#   # 0) check out this experiment's branch (the failsafe pushes onto whatever is
#   #    checked out — it does NOT create or switch branches for you). For this
#   #    experiment that branch is experiment24_cloud.
#   git fetch origin && git checkout experiment24_cloud
#
#   # 1) launch the arm
#   nohup bash run_gemma27b_hu_harm_evaldesc_measured.sh > logs/run_evaldesc_measured.out 2>&1 &
#
#   # 2) launch the failsafe — it follows the stage, then exits.
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
#   git fetch origin && git checkout experiment24_cloud
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
    "configs/gptoss120b_hu_harm_gemma27b_ens10_devval_s3_itermemo150_evaldesc_measured.md"
)
PROBE_DIRS=(
    "probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc_measured"
)
LOG_FILES=(
    "logs/run_hu_harm_gemma27b_gptoss120b_s3_evaldesc_measured.log"
)

REMOTE="origin"
BRANCH=""   # set at startup to the CURRENTLY checked-out branch — never switched
PY="${REPO_ROOT}/.venv_claude/bin/python"

CLOSE_ON_FINISH=0       # --close-on-finish: run CLOSE_SCRIPT once every stage has finished
CLOSE_SCRIPT="close_this.sh"   # --close-script: what to run then (here: stops the Salad container)

POLL_INTERVAL=30        # seconds between marker checks
PERIODIC_INTERVAL=2400  # 40 min between fallback snapshots (captures in-progress JSONL).
                        #   The marker/probe poll above is the PRIMARY checkpoint trigger and
                        #   is unaffected by this — a finished red-team phase or a fresh
                        #   probe_iter*.pkl still commits within POLL_INTERVAL seconds. This
                        #   only bounds how much in-flight JSONL a wipe can cost when neither
                        #   has landed for a while, which under these arms' schedule (10
                        #   sessions x 5 rounds per error type) is the long stretch inside a
                        #   rotation. 40 min of successes is the exposure; the JSONL is
                        #   append-only and dedup-by-canonical-text, so a re-run after resume
                        #   simply re-adds what was lost.

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
  --close-on-finish        after the LAST stage finishes and the final commit is
                           confirmed on the remote, run --close-script to shut the
                           box down. Never fires on Ctrl-C/SIGTERM, on a crashed
                           run, or while HEAD is unpushed.
  --close-script PATH      what --close-on-finish runs (default: $CLOSE_SCRIPT)
  -h, --help               show this help
EOF
}

# First user-supplied stage flag wipes the defaults, so a custom invocation never
# silently inherits the memo-ablation arms.
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
COMMIT_PATHS+=("logs_archive")              # STRIPPED run log(s) — see strip_logs below.
                                            #   NOT logs/: a raw run log is mostly tqdm
                                            #   carriage-return spam and grew to 115 MB in
                                            #   ~19 h, past GitHub's hard 100 MB per-file
                                            #   limit, at which point EVERY push failed and
                                            #   origin silently stopped being a resume point.

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
if [[ "$CLOSE_ON_FINISH" -eq 1 ]]; then
    # Checked NOW, not at the end: the end is hours or days away and unattended, and a
    # missing close script discovered then means the box just stays up burning money.
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

# Write a tqdm-free copy of each stage's log into logs_archive/, and commit THAT
# instead of the raw file.
#
# Two reasons, both learned the hard way on 2026-08-19. Size: the raw arm-1 log
# reached 115 MB in ~19 h — over GitHub's hard 100 MB per-file cap — and from that
# moment every push was rejected, so origin froze at 07:15 while probe_iter4 and
# two hours of red-team JSONLs existed only on the box. Deltas: the raw log is
# ~99% carriage-return progress bars, and a gzipped snapshot is binary, so each
# checkpoint would store a fresh multi-MB blob. The stripped log is append-only
# text, which git deltas almost perfectly — a checkpoint costs roughly the lines
# added since the previous one.
#
# The raw log stays on disk untouched; it is simply not git's problem.
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
    #
    # The push output is CAPTURED and logged on failure. It used to be sent to
    # /dev/null with only "WARN: push failed" recorded, which is how a hard,
    # never-self-healing rejection ("File ... is 109.25 MB; this exceeds GitHub's
    # file size limit of 100.00 MB") looked exactly like a transient network blip
    # for three hours. A failsafe whose failures are indistinguishable from noise
    # is not a failsafe.
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
ALL_STAGES_DONE=0   # set ONLY on normal completion of the last stage — see maybe_close_box

# Shut the box down, but only when it is genuinely safe to lose it.
#
# Three guards, and every one of them exists because the failure it prevents is
# unrecoverable — the box is destroyed, not paused:
#
#   1. ALL_STAGES_DONE. Set only where the poller decides every stage finished. So a
#      Ctrl-C, a SIGTERM, or the poller dying for any other reason runs finalize() and
#      exits WITHOUT closing. Killing the failsafe must never kill the box.
#   2. HEAD is on the remote. Compared against the remote-TRACKING ref, which only our
#      own successful pushes advance, so this is a real "the results are off this
#      machine" check rather than a "we tried" one. A push can fail for hours (the
#      100 MB GitHub limit did exactly that on 2026-08-19) — closing then would
#      destroy the only copy of the run.
#   3. The commit set is what do_commit stages: probes, markers, JSONL successes and
#      sidecars. Activation caches are deliberately NOT pushed, so closing the box does
#      lose those — they are recompute-only by design, which is the whole reason they
#      are excluded.
#
# A crashed run never reaches ALL_STAGES_DONE either: the poller just keeps polling, so
# the box stays up for inspection. That is the intended asymmetry — an unfinished run is
# worth more than the machine time.
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
            ALL_STAGES_DONE=1
            exit 0   # EXIT trap runs finalize() -> maybe_close_box()
        fi
        log "handing over to stage $((STAGE + 1))/$N_STAGES: ${CONFIGS[$STAGE]} (${PROBE_DIRS[$STAGE]})"
        last_sig="$(checkpoint_signature "${PROBE_DIRS[$STAGE]}")"
        last_periodic="$(date +%s)"
    fi
done
