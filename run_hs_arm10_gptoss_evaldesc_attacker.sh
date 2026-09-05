#!/usr/bin/env bash
set -e

# ARM 10 of the SELF-GENERATED-BASE memo experiment on the HIGH-STAKES concept.
#
# ARM 10 = ARM 2 + `attacker.show_eval_data_description: true`, and nothing else bar the
# output paths. Verified before launch: the two configs diff on exactly that key, and the
# rendered attacker system prompt differs by exactly the added
# `## The data this probe is evaluated on` section (2550 -> 4016 chars, additions only).
#
#                                 attacker        base data      eval.data_description
#   ARM 1 (done)                  gpt-oss-120b    gptoss_50                     unset
#   ARM 2 (done)                  gpt-oss-120b    gptoss_50         the four hs kinds
#   ARM 10 (THIS)                 gpt-oss-120b    gptoss_50         the four hs kinds,
#                                                                   ALSO SHOWN TO THE
#                                                                   ATTACKER DIRECTLY
#
#   ARM 2 -> ARM 10   does telling the ATTACKER the eval kinds outright do more than
#                     the memo relay does?
#
# WHY RE-ASK ON GPT-OSS. ARM 9 already ran this exact knob on nemotron (arm 8 -> arm 9) and
# returned +0.0107 mean eval AUROC against arm 8's own within-arm sd of 0.0122 — under 1 sd,
# no effect demonstrated. But the resampling grids (aef81990, 9611f65b) show WHY that arm was
# a poor place to ask: the nemotron BASE arm is the least stable in the whole experiment —
# draw sd 0.0362 (90% grid) / 0.0594 (80%), five to ten times every other arm's, on a mean of
# ~0.78 against ~0.89-0.90 elsewhere. GPT-OSS is the tightest pair in the grid (draw sd
# 0.0054 / 0.0126), so an effect of the size arm 9 hinted at would actually be visible here.
#
# Note the grids also showed all EIGHT base -> +evaldesc gaps positive (4 pairs x 2 grids),
# but three of four pairs under 1 sd and only nemotron clearing 3 sd on one grid. This arm
# tests the SECOND rung (memo relay -> direct channel), not that first one.
#
# `judge.eval_scope_check` is on (its default) in BOTH arms 2 and 10, so the scope gate is
# INHERITED, not introduced here. The judge's LABELLING function is identical in the two arms,
# so success rate, clone rate, red-team labels and the eval CSV are all directly comparable.
#
# EXPECT the ~2-3% attempt tax arm 9 exhibited: told the eval data contains doctor-patient
# dialogues and tool-calling agents, that attacker began using `patient`/`doctor`/`tool_calls`
# as message ROLES, which the chat template rejects. Handled (token_budget fails open,
# ProbeScoringError reports back), biases AGAINST this arm, and worth measuring again on a
# different attacker.
#
# Usage:
#   nohup bash run_hs_arm10_gptoss_evaldesc_attacker.sh > logs/run_hs_arm10.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

# Credentials live in .env (git-ignored), same as every earlier arm on this box.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY (attacker, judge and preprocessing are all provider: openrouter)}"

# tuberlens' hf_login() RAISES without a token even for a fully cached model, and the first
# red-team model load is well past the initial train and the iter0 eval — so check it here.
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"; export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in hf_token.txt)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Kaggle creds for the precomputed eval activations (config `kaggle:` section). An
# unauthenticated KaggleApi.authenticate() ends in exit(1), not an exception, so check now
# rather than hours in. (On this box the blobs are already warm and the fetch is lazy.)
if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
    kaggle_json="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
    [ -f "$kaggle_json" ] || kaggle_json="$HOME/.config/kaggle/kaggle.json"
    [ -f "$kaggle_json" ] || { echo "ERROR: no Kaggle credentials (KAGGLE_CONFIG_DIR must name the DIRECTORY holding kaggle.json)" >&2; exit 1; }
fi
.venv_claude/bin/python -c "import kaggle" 2>/dev/null || { echo "ERROR: 'kaggle' not installed" >&2; exit 1; }

# Transfer layer only — cannot change a number the run produces. Guarded on the import, since
# huggingface_hub RAISES when HF_HUB_ENABLE_HF_TRANSFER=1 and the package is missing.
if .venv_claude/bin/python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
fi

# Placement only — fixes WHERE the frozen extraction LLM's weights live, not what they
# compute. Unpinned, accelerate infers the budget from whatever is free at load time, and one
# unlucky reload of the ~40 here spills the executed tail to DISK (48-264 s/sample against
# ~2.8 resident). Both vars: the AGENTIC_ one is authoritative on load_extraction_model,
# tuberlens' MAX_MEMORY reaches every other tuberlens load (get_performances included).
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"

# Arm 8's rotation hit a lot of transient OpenRouter noise on nemotron; keep the same
# tolerance the arm-8 runner used.
export OPENROUTER_MAX_CONSECUTIVE_ERRORS="${OPENROUTER_MAX_CONSECUTIVE_ERRORS:-40}"

BASE="data/highstakes_gptoss_50.jsonl"
[ -f "$BASE" ] || { echo "ERROR: base training data not found: $BASE" >&2; exit 1; }
echo ">>> base training data: $BASE ($(wc -l < "$BASE") rows, written by openai/gpt-oss-120b)"

# Shared, arm-independent activation cache (experiment18/19, reused by arms 1-8). No cache key
# mentions the memo knobs, the eval-data description, who is shown it, view_limit,
# sessions_per_model or the iteration count — so the four EVAL blobs (~46 GB) and the 1908-row
# DEV blob are already warm, and the NEMOTRON BASE blob was computed by arm 7 and reused by
# arm 8, so this arm recomputes none of them.
mkdir -p results_hs_gemma27b_devval/base_activations results_hs_gemma27b_devval/eval_activations

OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"
LOG=logs/run_hs_gemma27b_gptoss120b_gptossbase_evaldesc_attacker.log

echo ">>> $(date -Is)  START arm 10 (gpt-oss, eval-desc ALSO shown to the attacker)  -> $LOG"
rc=0
# --iterations 10 matches arms 7 and 8 exactly: the contrast is about what crosses an
# iteration boundary, so the number of boundaries must not vary.
# NOT passing --ensemble-size or --dev-data: both are flag > config, both are properties all
# the arms share, and both live in the configs so the arms can be diffed. --test-size /
# --split-field are moot under validation.dev_data.
.venv_claude/bin/python scripts/iterative_retrain.py \
    configs/gptoss120b_hs_gemma27b_gptossbase_evaldesc_attacker.md \
    --iterations 10 \
    --base-training-data "$BASE" \
    --probe-out-dir probes/hs_gemma27b_gptoss120b_gptossbase_evaldesc_attacker \
    --eval --eval-dataset-dir eval_sets/highstakes \
    >> "$LOG" 2>&1 || rc=$?

if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
    echo ">>> $(date -Is)  ABORTED arm 10 — OpenRouter unusable (exit $rc)." >&2
    tail -n 5 "$LOG" >&2
    echo ">>> Fix credits/key, then re-run this script; --resume is on by default." >&2
    exit "$rc"
elif [ "$rc" -ne 0 ]; then
    echo ">>> $(date -Is)  FAILED arm 10 (exit $rc) — see $LOG" >&2
    exit "$rc"
fi
echo ">>> $(date -Is)  DONE arm 10."
