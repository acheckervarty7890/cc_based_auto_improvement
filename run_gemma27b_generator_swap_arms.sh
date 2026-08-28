#!/usr/bin/env bash
set -e

# =============================================================================
# generator_experiment_1 — FIVE GENERATOR-SWAP ARMS of the generate -> score ->
# retrain -> guide loop (scripts/iterative_generate.py), each against a probe on
# google/gemma-3-27b-it layer 32, each for FIVE iterations.
#
#   ARM 1N  high-stakes            nemotron-3-ultra-550b   configs/gen_gemma27b_highstakes_nemotron.md
#   ARM 2N  human harm             nemotron-3-ultra-550b   configs/gen_gemma27b_hu_harm_nemotron.md
#   ARM 1G  high-stakes            gpt-oss-120b            configs/gen_gemma27b_highstakes_gptoss.md
#   ARM 2G  human harm             gpt-oss-120b            configs/gen_gemma27b_hu_harm_gptoss.md
#   ARM 3G  instruction following  gpt-oss-120b            configs/gen_gemma27b_instructions_gptoss.md
#
# THE ONLY THING THAT VARIES from the already-run arms 1/2/3 is generator.models. Each
# config is its parent arm's file with one line changed and the run_dir/comparison_csv
# renamed; base training data, dev set, eval splits, judge, probe, loop and both system
# prompts are identical. Together with arm 3N (nemotron on instructions, already run)
# this fills in the generator x concept grid:
#
#            llama-3.3-70b     nemotron-3-ultra     gpt-oss-120b
#   high-stakes   arm 1  (done)     ARM 1N              ARM 1G
#   human harm    arm 2  (done)     ARM 2N              ARM 2G
#   instructions  arm 3  (done)     arm 3N (done)       ARM 3G
#
# COST PER ARM, unchanged from the first three: 5 generator calls (+ up to 2 top-ups
# each) per iteration, a cache warm over the new samples, 5 candidate fits and 1 union
# retrain — six probe fits per iteration, 30 per arm; up to 250 generated samples. The
# 27B extraction LLM is loaded only for newly generated conversations: the eval splits
# and the dev set come precomputed from Kaggle, and each concept's base blob is already
# in the shared cache dir from its parent arm.
#
# NOISE FLOOR: no arm sets probe.ensemble_size, so every fit is a SINGLE probe (+/-0.005
# dev AUROC between near-identical training sets) and min_auroc_gain is 0.0 — some
# accepted batches will be noise. Same as arms 1/2/3, deliberately, so the ledgers are
# readable against each other.
#
# Usage (credentials come from .env if it exists, else from the environment):
#   mkdir -p logs
#   nohup bash run_gemma27b_generator_swap_arms.sh > logs/run_generator_swap_arms.out 2>&1 &
#   nohup bash failsafe_commit.sh \
#       --config configs/gen_gemma27b_highstakes_nemotron.md   --probe-out-dir probes/gen_gemma27b_highstakes_nemotron   --log-file logs/run_gen_gemma27b_highstakes_nemotron.log \
#       --config configs/gen_gemma27b_hu_harm_nemotron.md      --probe-out-dir probes/gen_gemma27b_hu_harm_nemotron      --log-file logs/run_gen_gemma27b_hu_harm_nemotron.log \
#       --config configs/gen_gemma27b_highstakes_gptoss.md     --probe-out-dir probes/gen_gemma27b_highstakes_gptoss     --log-file logs/run_gen_gemma27b_highstakes_gptoss.log \
#       --config configs/gen_gemma27b_hu_harm_gptoss.md        --probe-out-dir probes/gen_gemma27b_hu_harm_gptoss        --log-file logs/run_gen_gemma27b_hu_harm_gptoss.log \
#       --config configs/gen_gemma27b_instructions_gptoss.md   --probe-out-dir probes/gen_gemma27b_instructions_gptoss   --log-file logs/run_gen_gemma27b_instructions_gptoss.log \
#       > logs/failsafe_commit_swap.out 2>&1 &
#
# Run ONE arm only, or reorder them: comment out the run_arm calls at the bottom.
# =============================================================================

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

# --- credentials ------------------------------------------------------------------------------
# .env on this box holds OPENROUTER_API_KEY, HF_TOKEN, KAGGLE_CONFIG_DIR and
# AGENTIC_REDTEAM_MAX_MEMORY. Sourced only for variables that are not already exported, so an
# explicit `export OPENROUTER_API_KEY=... ; bash run_...sh` still wins.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY (or put it in .env) — both the generator and the judge are provider: openrouter}"

# --- per-request timeout ----------------------------------------------------------------------
# openrouter_client's default is 60 s. nemotron-3-ultra is a reasoning model measured at ~50 s
# for a batch of 10 under this experiment's prompt, and a slow route can be minutes; a timeout
# is a LOST BATCH (the exception is captured on BatchGeneration.error and the iteration continues
# with one batch fewer). 900 s costs nothing when the model answers quickly.
export OPENROUTER_TIMEOUT_S="${OPENROUTER_TIMEOUT_S:-900}"
echo ">>> OPENROUTER_TIMEOUT_S=$OPENROUTER_TIMEOUT_S"

# --- Kaggle credentials -----------------------------------------------------------------------
# Checked HERE rather than at first use: both the eval splits AND the dev set are pulled from
# Kaggle, and an unauthenticated KaggleApi.authenticate() ends in exit(1) — a SystemExit, not an
# exception. No credentials means no run at all, not a slow run.
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

# --- HuggingFace token ------------------------------------------------------------------------
# tuberlens' LLMModel.load calls hf_login(), which RAISES when no token is set — even when the
# gemma weights are already in the local HF cache and no download happens.
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in .env / hf_token.txt) — tuberlens hf_login() raises without one, even for a fully cached model}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"

# --- pin the extraction model's memory budget --------------------------------------------------
# PLACEMENT ONLY — cannot change a single number the run produces. Unpinned accelerate infers the
# budget from whatever is FREE AT LOAD TIME, and this loop loads the 27B model once per iteration
# (the cache warm over newly generated samples) x 5 iterations x 5 arms; one unlucky reload
# silently spills the executed tail to DISK (48-264 s/sample against ~2.8 s/sample resident).
# Sized for the 24 GiB card and ~62 GiB of host RAM on this box.
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
echo ">>> max_memory pinned: $AGENTIC_REDTEAM_MAX_MEMORY (placement only — does not change results)"

export PROBE_FUSED_ENSEMBLE="${PROBE_FUSED_ENSEMBLE:-1}"
echo ">>> ensemble path: PROBE_FUSED_ENSEMBLE=$PROBE_FUSED_ENSEMBLE (inert while ensemble_size is unset)"

# No clobber guard on the per-arm output/probe dirs ON PURPOSE: --resume is on by default and is
# how a wiped container picks a run back up. To start genuinely fresh, move the per-arm dirs aside.

OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is durably unusable"

run_arm () {  # $1 = config, $2 = base training data, $3 = eval dir, $4 = probe-out-dir, $5 = logfile
    echo ">>> $(date -Is)  START $1  -> $4   (log: $5)"
    local rc=0
    # --iterations is NOT passed: loop.iterations is 5 in every config, where the arms can be
    # diffed against each other and against arms 1/2/3. Same reasoning for --dev-data,
    # --ensemble-size, --eval-max-samples and the transform flags. --eval IS passed: it is a
    # per-launch choice, and it writes the comparison CSV the failsafe uses as a finish signal.
    .venv_claude/bin/python scripts/iterative_generate.py "$1" \
        --base-training-data "$2" \
        --probe-out-dir "$4" \
        --eval --eval-dataset-dir "$3" \
        > "$5" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the key is dead.
        # Do NOT start the next arm — it would fail identically and spend hours of GPU producing
        # empty batches.
        echo ">>> $(date -Is)  ABORTED $1 — OpenRouter unusable (exit $rc)." >&2
        tail -n 5 "$5" >&2
        echo ">>> Fix credits/key, then re-run this script; --resume picks the arm back up." >&2
        exit "$rc"
    elif [ "$rc" -ne 0 ]; then
        echo ">>> $(date -Is)  FAILED  $1 (exit $rc) — see $5" >&2
        exit "$rc"
    fi
    echo ">>> $(date -Is)  DONE  $1"
}

# ARM 1N — high-stakes, nemotron generator
run_arm configs/gen_gemma27b_highstakes_nemotron.md \
        data/highstakes_llama70b_50.jsonl \
        eval_sets/highstakes \
        probes/gen_gemma27b_highstakes_nemotron \
        logs/run_gen_gemma27b_highstakes_nemotron.log

# ARM 2N — human harm, nemotron generator
run_arm configs/gen_gemma27b_hu_harm_nemotron.md \
        data/hu_harm_llama70b_50.jsonl \
        eval_sets/hu_ha \
        probes/gen_gemma27b_hu_harm_nemotron \
        logs/run_gen_gemma27b_hu_harm_nemotron.log

# ARM 1G — high-stakes, gpt-oss generator
run_arm configs/gen_gemma27b_highstakes_gptoss.md \
        data/highstakes_llama70b_50.jsonl \
        eval_sets/highstakes \
        probes/gen_gemma27b_highstakes_gptoss \
        logs/run_gen_gemma27b_highstakes_gptoss.log

# ARM 2G — human harm, gpt-oss generator
run_arm configs/gen_gemma27b_hu_harm_gptoss.md \
        data/hu_harm_llama70b_50.jsonl \
        eval_sets/hu_ha \
        probes/gen_gemma27b_hu_harm_gptoss \
        logs/run_gen_gemma27b_hu_harm_gptoss.log

# ARM 3G — instruction following, gpt-oss generator
run_arm configs/gen_gemma27b_instructions_gptoss.md \
        data/instructions_llama70b_50.jsonl \
        eval_sets/instructions \
        probes/gen_gemma27b_instructions_gptoss \
        logs/run_gen_gemma27b_instructions_gptoss.log

echo ">>> $(date -Is)  all five generator-swap arms finished."
