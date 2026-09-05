#!/usr/bin/env bash
set -e

# ARM 9 of the SELF-GENERATED-BASE memo experiment on the HIGH-STAKES concept.
#
# ARM 9 = ARM 8 + `attacker.show_eval_data_description: true`, and nothing else bar the
# output paths. Verified: the two configs diff on exactly that one key plus jsonl_path /
# run_id / comparison_csv, and the rendered attacker system prompt differs by exactly the
# added `## The data this probe is evaluated on` section (2550 -> 4016 chars).
#
#                                 attacker              base data        eval.data_description
#   ARM 7 (done)                  nemotron-3-ultra      nemotron_50                    unset
#   ARM 8 (done)                  nemotron-3-ultra      nemotron_50        the four hs kinds
#   ARM 9 (THIS)                  nemotron-3-ultra      nemotron_50        the four hs kinds,
#                                                                          ALSO SHOWN TO THE
#                                                                          ATTACKER DIRECTLY
#
#   ARM 7 -> ARM 8   does telling the MEMO-WRITER the eval kinds change what the loop finds?
#   ARM 8 -> ARM 9   does telling the ATTACKER them outright do more than the memo relay does?
#
# In arms 2/4/6/8 the description reaches the attacker only laundered through a memo, and only
# from round 1 — round 0 of every iteration is always written by a model that has never been
# told what the eval data looks like. This arm opens the DIRECT channel: the same text,
# verbatim, in the attacker's own system prompt, present from the first batch of round 0.
#
# `judge.eval_scope_check` is on (its default) in BOTH arms 8 and 9, so the scope gate is
# INHERITED, not introduced here — arm 8 recorded one `violated_constraint` row in 639
# false-positive attempts. The judge's LABELLING function is identical in the two arms, so
# success rate, clone rate, red-team labels and the eval CSV are all directly comparable.
#
# Usage:
#   nohup bash run_hs_arm9_evaldesc_attacker.sh > logs/run_hs_arm9.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): failsafe_commit.sh alongside it.

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

BASE="data/highstakes_nemotron_50.jsonl"
[ -f "$BASE" ] || { echo "ERROR: base training data not found: $BASE" >&2; exit 1; }
echo ">>> base training data: $BASE ($(wc -l < "$BASE") rows, written by nvidia/nemotron-3-ultra-550b-a55b)"

# Shared, arm-independent activation cache (experiment18/19, reused by arms 1-8). No cache key
# mentions the memo knobs, the eval-data description, who is shown it, view_limit,
# sessions_per_model or the iteration count — so the four EVAL blobs (~46 GB) and the 1908-row
# DEV blob are already warm, and the NEMOTRON BASE blob was computed by arm 7 and reused by
# arm 8, so this arm recomputes none of them.
mkdir -p results_hs_gemma27b_devval/base_activations results_hs_gemma27b_devval/eval_activations

OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"
LOG=logs/run_hs_gemma27b_nemotron_nemobase_evaldesc_attacker.log

echo ">>> $(date -Is)  START arm 9 (nemotron, eval-desc ALSO shown to the attacker)  -> $LOG"
rc=0
# --iterations 10 matches arms 7 and 8 exactly: the contrast is about what crosses an
# iteration boundary, so the number of boundaries must not vary.
# NOT passing --ensemble-size or --dev-data: both are flag > config, both are properties all
# the arms share, and both live in the configs so the arms can be diffed. --test-size /
# --split-field are moot under validation.dev_data.
.venv_claude/bin/python scripts/iterative_retrain.py \
    configs/nemotron_hs_gemma27b_nemobase_evaldesc_attacker.md \
    --iterations 10 \
    --base-training-data "$BASE" \
    --probe-out-dir probes/hs_gemma27b_nemotron_nemobase_evaldesc_attacker \
    --eval --eval-dataset-dir eval_sets/highstakes \
    >> "$LOG" 2>&1 || rc=$?

if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
    echo ">>> $(date -Is)  ABORTED arm 9 — OpenRouter unusable (exit $rc)." >&2
    tail -n 5 "$LOG" >&2
    echo ">>> Fix credits/key, then re-run this script; --resume is on by default." >&2
    exit "$rc"
elif [ "$rc" -ne 0 ]; then
    echo ">>> $(date -Is)  FAILED arm 9 (exit $rc) — see $LOG" >&2
    exit "$rc"
fi
echo ">>> $(date -Is)  DONE arm 9."
