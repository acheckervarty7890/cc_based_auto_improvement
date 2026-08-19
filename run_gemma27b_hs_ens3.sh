#!/usr/bin/env bash
set -e

# ENSEMBLE-OF-3 dev-validation run of the two batch-submission attacker arms on the
# HIGH-STAKES concept with a google/gemma-3-27b-it (L32) probe.
#
# Same two arms, same attacker/judge/preprocessing knobs, same validation source and the same
# prompt body as run_gemma27b_hs_devval.sh (experiment18_cloud). Exactly TWO things change:
#
#   probe.ensemble_size: 1 -> 3   (in the configs) three independently-seeded probes per fit,
#                                 score-averaged; members use the pinned ENSEMBLE_SEEDS[:3]
#                                 = (3699, 14431, 23529), so --seed still moves only the data
#   --iterations: 3 -> 5          (below) two more red-team/retrain cycles, because
#                                 experiment18 peaked at iter2 and fell back at iter3 — with a
#                                 fixed dev set the per-iteration checkpoints stay comparable,
#                                 so the extra cycles answer whether that was a turn or noise
#
# EXPECT THIS TO TAKE LONGER THAN experiment18. Activations are extracted once and shared
# across members, so the ensemble costs three head fits over one extraction — but each member
# early-stops independently and scores all 1908 dev rows every epoch, and there are now 6 fits
# per arm instead of 4. The red-team phases are unchanged. experiment18 took ~3h10m per arm.
#
# THE EVAL SET IS NOT THE ONE experiment9 SCORED. eval_sets/highstakes was re-cut (4408 rows:
# 2984/604/86/734); the splits experiment9 reported are now dev_samples/highstakes (1908 rows).
# So no number from experiment9 or earlier is comparable to this run's. It IS directly
# comparable to experiment18's, which scored the same re-cut set with the same dev set.
#
#   ARM 1 (gpt-oss-120b):    configs/gptoss120b_hs_gemma27b_ens3.md
#                            -> results_hs_gemma27b_gptoss120b_ens3/
#                               probes/hs_gemma27b_gptoss120b_ens3
#   ARM 2 (deepseek-v4-pro): configs/deepseekv4pro_hs_gemma27b_ens3.md
#                            -> results_hs_gemma27b_deepseekv4pro_ens3/
#                               probes/hs_gemma27b_deepseekv4pro_ens3
#
# The judge (openai/gpt-5.1), the preprocessing model, the probe (gemma-3-27b-it L32), the base
# data (data/highstakes_llama70b_50.jsonl) and every scheduling knob are held fixed, so any
# delta between the two comparison CSVs is attributable to the attacker.
#
# ACTIVATIONS. The cache dirs are SHARED WITH experiment18 (results_hs_gemma27b_devval/) —
# activations depend on the probe model/layer/splits/seed/transforms, not on the ensemble size
# or the iteration count — so on this box staging is a pure no-op and nothing is copied,
# rebuilt, downloaded or recomputed. Both the eval blobs and the dev blob are STAGED before
# either arm starts, by
# scripts/stage_local_hs_activations.py:
#   * eval  — the four <split>-acts_full.pt blobs are hard-linked into the shared cache dir, so
#             prefetch_eval_activations validates them and downloads nothing, and
#             _assign_cached_activations attaches them before get_performances can load the 27B.
#   * dev   — assembled from the per-split dev blobs into the single content-keyed blob
#             _dev_activation_cache_path derives, so the first fit does not push 1908 rows
#             through gemma-3-27b. There is no Kaggle fetch path for this one.
# If the local blobs are absent (a fresh box), staging is skipped, the configs' kaggle: section
# fetches the eval side — which needs credentials, checked below — and the dev side is computed
# on the box. The BASE split (50 rows) and every red-team conversation are computed either way.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   export AGENTIC_REDTEAM_MAX_MEMORY="0=22GiB,cpu=45GiB"
#   mkdir -p logs
#   nohup bash run_gemma27b_hs_ens3.sh > logs/run_gemma27b_hs_ens3.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it —
# its defaults already point at these two arms in this order.

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

PY=.venv_claude/bin/python
BASE_DATA=data/highstakes_llama70b_50.jsonl
EVAL_DIR=eval_sets/highstakes            # the RE-CUT eval set — never dev_samples/highstakes,
                                         #   which is what the fit early-stops against
SHARED_CACHE="results_hs_gemma27b_devval"   # shared with experiment18 on purpose

[ -f "$BASE_DATA" ] || { echo "ERROR: base training data missing: $BASE_DATA" >&2; exit 1; }
[ -d "$EVAL_DIR" ]  || { echo "ERROR: eval split dir missing: $EVAL_DIR" >&2; exit 1; }

# The memory pin is not this script's to choose, but an unpinned gemma-3-27b load lets
# accelerate hand the CPU a budget equal to whatever RAM happens to be free, which is what
# OOM-kills the fit later. Warn rather than fail — a box with different RAM wants a different pin.
if [ -z "${AGENTIC_REDTEAM_MAX_MEMORY:-}" ]; then
    echo ">>> WARNING: AGENTIC_REDTEAM_MAX_MEMORY is unset; on a 24 GB GPU / 62 GB host use" >&2
    echo "             export AGENTIC_REDTEAM_MAX_MEMORY=\"0=22GiB,cpu=45GiB\"" >&2
fi

# --- stage the activations this box already has -------------------------------------------
# Idempotent: a no-op once staged, and it only ever places blobs that pass the same
# model/layer/row-count validation the Kaggle download side runs.
echo ">>> $(date -Is)  staging local activations into $SHARED_CACHE"
staged=0
$PY scripts/stage_local_hs_activations.py \
    --config configs/gptoss120b_hs_gemma27b_ens3.md && staged=1 || true

if [ "$staged" -ne 1 ]; then
    echo ">>> local activations not fully staged — the eval side will come from Kaggle and the"
    echo "    dev side will be computed on this box (1908 rows through gemma-3-27b, once)."
    # Kaggle credentials checked HERE rather than at first use: the first eval is hours into
    # arm 1, and an unauthenticated KaggleApi.authenticate() ends in exit(1), not an exception.
    if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
        kaggle_json="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
        [ -f "$kaggle_json" ] || kaggle_json="$HOME/.config/kaggle/kaggle.json"
        if [ ! -f "$kaggle_json" ]; then
            echo "ERROR: no Kaggle credentials and no staged eval blobs. Set KAGGLE_CONFIG_DIR to" >&2
            echo "       the DIRECTORY holding kaggle.json (not the file), or export KAGGLE_API_TOKEN." >&2
            exit 1
        fi
        echo ">>> kaggle credentials: $kaggle_json"
    fi
    $PY -c "import kaggle" 2>/dev/null || {
        echo "ERROR: the 'kaggle' package is not installed: .venv_claude/bin/pip install kaggle" >&2
        exit 1
    }
fi

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (shared by both arms)"

# --- run one arm ---------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    $PY scripts/iterative_retrain.py "$1" \
        --iterations 5 \
        --base-training-data "$BASE_DATA" \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir "$EVAL_DIR" \
        --seed 42 \
        > "$3" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the key is
        # dead. Do NOT start the next arm — it would fail identically and write a comparison
        # CSV from probes trained on nothing.
        echo ">>> $(date -Is)  ABORTED $1 — OpenRouter unusable (exit $rc)." >&2
        tail -n 5 "$3" >&2
        echo ">>> Fix credits/key, then re-run with --resume to continue this arm." >&2
        exit "$rc"
    elif [ "$rc" -ne 0 ]; then
        echo ">>> $(date -Is)  FAILED  $1 (exit $rc) — see $3" >&2
        exit "$rc"
    fi
    echo ">>> $(date -Is)  DONE  $1"
}

run_arm configs/gptoss120b_hs_gemma27b_ens3.md    probes/hs_gemma27b_gptoss120b_ens3    logs/run_hs_gemma27b_gptoss120b_ens3.log
run_arm configs/deepseekv4pro_hs_gemma27b_ens3.md probes/hs_gemma27b_deepseekv4pro_ens3 logs/run_hs_gemma27b_deepseekv4pro_ens3.log

echo ">>> $(date -Is)  both arms finished."
