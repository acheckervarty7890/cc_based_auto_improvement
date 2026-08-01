#!/usr/bin/env bash
set -e

# Attacker ablation on the HIGH-STAKES concept with a google/gemma-3-27b-it (L32) probe.
# Two arms, IDENTICAL in every knob except attacker.models, run sequentially and fully
# isolated. NEITHER arm uses contrastive label guidance (no preprocessing.concept_description
# / label_guidance) — that is held off in both, so the attacker model is the only variable:
#
#   ARM 1 (gpt-oss-120b):   configs/gptoss120b_hs_gemma27b_noguidance.md
#                           -> results_hs_gemma27b_gptoss120b_noguidance/
#                              probes/hs_gemma27b_gptoss120b_noguidance
#   ARM 2 (deepseek-v4-pro): configs/deepseekv4pro_hs_gemma27b_noguidance.md
#                           -> results_hs_gemma27b_deepseekv4pro_noguidance/
#                              probes/hs_gemma27b_deepseekv4pro_noguidance
#
# The judge (openai/gpt-5.1), the preprocessing model (openai/gpt-5.1), the probe
# (gemma-3-27b-it L32), the base data (data/hs_ls_200.jsonl) and every scheduling knob are
# held fixed, so any delta in the comparison CSVs is attributable to the attacker.
#
# ACTIVATIONS. The shared cache dir (results_hs_gemma27b_attacker_ablation/) starts empty on
# a clean cloud box; arm 1 fills it and arm 2 hits it, because those blobs depend only on the
# probe model / layer / seed / base data / eval splits / transforms — NOT on the attacker.
# The redteam_acts_* per-conversation cache written into the same dir is content-keyed with a
# frozen LLM, so the two arms' distinct successes get distinct keys.
#
# The EVAL half is not computed at all: both configs carry a `kaggle:` section pointing at
# anku7890/{split}gemmaevalpt, so arm 1's first eval downloads ~20 GB of precomputed
# gemma-3-27b activations (validated against the probe's model/layer and each split's row
# count) straight into eval_activations/ instead of running full splits through a 27B model.
# That needs credentials — see the KAGGLE_CONFIG_DIR check below. The BASE split (~116 MB) is
# still computed locally by arm 1, as is every red-team conversation.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hs_attacker_ablation.sh > logs/run_gemma27b_hs_attacker_ablation.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it,
# pointed at these two arms in this order.

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# Kaggle credentials for the precomputed eval activations (configs' `kaggle:` section).
# Checked HERE rather than at first use: the first eval is hours into arm 1, and an
# unauthenticated KaggleApi.authenticate() ends in exit(1), not an exception.
if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
    kaggle_json="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
    [ -f "$kaggle_json" ] || kaggle_json="$HOME/.config/kaggle/kaggle.json"
    if [ ! -f "$kaggle_json" ]; then
        echo "ERROR: no Kaggle credentials. Set KAGGLE_CONFIG_DIR to the DIRECTORY holding" >&2
        echo "       kaggle.json (not the file), or export KAGGLE_API_TOKEN." >&2
        exit 1
    fi
    echo ">>> kaggle credentials: $kaggle_json"
fi
.venv_claude/bin/python -c "import kaggle" 2>/dev/null || {
    echo "ERROR: the 'kaggle' package is not installed: .venv_claude/bin/pip install kaggle" >&2
    exit 1
}

SHARED_CACHE="results_hs_gemma27b_attacker_ablation"   # shared, arm-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (starting empty — computed by arm 1, reused by arm 2)"

# --- run one arm ---------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 3 \
        --base-training-data data/hs_ls_200.jsonl \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_datasets \
        > "$3" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the
        # key is dead. Do NOT start the next arm — it would fail identically and
        # write a comparison CSV from probes trained on nothing.
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

run_arm configs/gptoss120b_hs_gemma27b_noguidance.md    probes/hs_gemma27b_gptoss120b_noguidance    logs/run_hs_gemma27b_gptoss120b_noguidance.log
run_arm configs/deepseekv4pro_hs_gemma27b_noguidance.md probes/hs_gemma27b_deepseekv4pro_noguidance logs/run_hs_gemma27b_deepseekv4pro_noguidance.log

echo ">>> $(date -Is)  both arms finished."
