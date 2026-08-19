#!/usr/bin/env bash
set -e

# ONE-ARM variant of run_gemma27b_hs_ens3.sh — the ensemble-of-3, 5-iteration dev-validation
# run on the HIGH-STAKES concept, for a single attacker instead of both.
#
# Identical in every knob to the two-arm runner; it just runs one arm and exits. Use it when
# you want the arm's own curve rather than the attacker comparison — with only one arm there
# is no ablation, so any result here is about the pipeline, not about the attacker.
#
# WHY THIS EXISTS NOW: main's `perf(retrain): stage activations on the GPU for the probe fits`
# (67c9ddd) stages the merged train/validation activations on the fit device once, after
# _release_model() has emptied the card. The per-epoch host->device copy of ~13 GB disappears
# and the fits go from ~3-4 epochs/min to ~158 — measured ~45x end-to-end on a live
# 978-train/290-val retrain. The change is PLACEMENT ONLY: same indices, same order, same
# values, verified to reproduce an existing probe member-for-member including best_epoch. So
# a run started before the patch and continued after it is still one experiment.
#
# THAT MAKES --resume THE DEFAULT WORTH USING. The CLI resumes from the latest probe_iterN.pkl
# in --probe-out-dir, then any finished (iteration, error_type) phase markers, then the rounds
# recorded in <jsonl>.rounds_done.jsonl. Because the patched fits are bit-identical to the
# unpatched ones, resuming a part-finished run loses nothing scientifically and keeps the
# hours already spent. Pass --fresh only if you want the arm rebuilt from iteration 0.
#
#   --fresh does NOT clean the output dirs. Red-team successes append to the same
#   <arm>_probing_{fp,fn}.jsonl, so a fresh run over a used dir mixes both attempts' successes
#   into one file. Move probes/hs_gemma27b_<arm>_ens3 and results_hs_gemma27b_<arm>_ens3 aside
#   first if you want a clean slate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   export AGENTIC_REDTEAM_MAX_MEMORY="0=22GiB,cpu=45GiB"
#   mkdir -p logs
#   nohup bash run_gemma27b_hs_ens3_one_arm.sh > logs/run_gemma27b_hs_ens3_one_arm.out 2>&1 &
#
#   bash run_gemma27b_hs_ens3_one_arm.sh --arm deepseekv4pro      # the other arm
#   bash run_gemma27b_hs_ens3_one_arm.sh --fresh                  # rebuild from iteration 0
#
# Checkpointing: failsafe_commit.sh takes one stage explicitly —
#   bash failsafe_commit.sh --config configs/<arm>_hs_gemma27b_ens3.md \
#       --probe-out-dir probes/hs_gemma27b_<arm>_ens3 \
#       --log-file logs/run_hs_gemma27b_<arm>_ens3.log
# (its built-in defaults are the TWO-arm stages; passing any stage flag clears them).

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

ARM="gptoss120b"
RESUME_FLAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --arm)   ARM="$2"; shift 2 ;;
        --fresh) RESUME_FLAG="--no-resume"; shift ;;
        -h|--help) grep '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done
case "$ARM" in
    gptoss120b|deepseekv4pro) ;;
    *) echo "ERROR: --arm must be gptoss120b or deepseekv4pro (got '$ARM')" >&2; exit 2 ;;
esac

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

PY=.venv_claude/bin/python
CONFIG="configs/${ARM}_hs_gemma27b_ens3.md"
PROBE_DIR="probes/hs_gemma27b_${ARM}_ens3"
LOGFILE="logs/run_hs_gemma27b_${ARM}_ens3.log"
BASE_DATA=data/highstakes_llama70b_50.jsonl
EVAL_DIR=eval_sets/highstakes            # the RE-CUT eval set — never dev_samples/highstakes,
                                         #   which is what the fit early-stops against
SHARED_CACHE="results_hs_gemma27b_devval"   # shared with experiment18 on purpose

[ -f "$CONFIG" ]    || { echo "ERROR: config missing: $CONFIG" >&2; exit 1; }
[ -f "$BASE_DATA" ] || { echo "ERROR: base training data missing: $BASE_DATA" >&2; exit 1; }
[ -d "$EVAL_DIR" ]  || { echo "ERROR: eval split dir missing: $EVAL_DIR" >&2; exit 1; }

if [ -z "${AGENTIC_REDTEAM_MAX_MEMORY:-}" ]; then
    echo ">>> WARNING: AGENTIC_REDTEAM_MAX_MEMORY is unset; on a 24 GB GPU / 62 GB host use" >&2
    echo "             export AGENTIC_REDTEAM_MAX_MEMORY=\"0=22GiB,cpu=45GiB\"" >&2
fi

# The GPU-staging patch needs headroom on the card for the merged train+val activations on top
# of whatever the fit itself uses; it falls back to host-resident tensors if the copy fails, so
# a busy GPU costs speed, not correctness. Warn if something else is already holding the card.
if command -v nvidia-smi >/dev/null 2>&1; then
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    if [ "${used:-0}" -gt 2000 ]; then
        echo ">>> WARNING: ${used} MiB already allocated on GPU 0 — is another run still alive?" >&2
        echo "             The fits will fall back to host-resident activations if staging OOMs." >&2
    fi
fi

echo ">>> $(date -Is)  staging local activations into $SHARED_CACHE"
$PY scripts/stage_local_hs_activations.py --config "$CONFIG" || {
    echo "ERROR: staging failed; see above. The eval side can fall back to the configs' kaggle:" >&2
    echo "       section, but fix this rather than paying for a 27B eval extraction." >&2
    exit 1
}
mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"

OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — OpenRouter unusable

echo ">>> $(date -Is)  START $CONFIG -> $PROBE_DIR   (log: $LOGFILE)  ${RESUME_FLAG:---resume (default)}"
rc=0
$PY scripts/iterative_retrain.py "$CONFIG" \
    --iterations 5 \
    --base-training-data "$BASE_DATA" \
    --probe-out-dir "$PROBE_DIR" \
    --eval --eval-dataset-dir "$EVAL_DIR" \
    --seed 42 \
    $RESUME_FLAG \
    > "$LOGFILE" 2>&1 || rc=$?

if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
    echo ">>> $(date -Is)  ABORTED $CONFIG — OpenRouter unusable (exit $rc)." >&2
    tail -n 5 "$LOGFILE" >&2
    echo ">>> Fix credits/key, then re-run (it resumes by default)." >&2
    exit "$rc"
elif [ "$rc" -ne 0 ]; then
    echo ">>> $(date -Is)  FAILED  $CONFIG (exit $rc) — see $LOGFILE" >&2
    exit "$rc"
fi
echo ">>> $(date -Is)  DONE  $CONFIG"
