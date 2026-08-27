#!/usr/bin/env bash
set -e

# =============================================================================
# generator_experiment_1 — THREE ARMS of the generate -> score -> retrain -> guide
# loop (scripts/iterative_generate.py, the dev_new_scaffolding scaffold), each against
# a probe on google/gemma-3-27b-it layer 32, each for FIVE iterations.
#
#   ARM 1  high-stakes            data/highstakes_llama70b_50.jsonl    eval_sets/highstakes
#   ARM 2  human harm             data/hu_harm_llama70b_50.jsonl       eval_sets/hu_ha
#   ARM 3  instruction following  data/instructions_llama70b_50.jsonl  eval_sets/instructions
#
# THE ONLY THING THAT VARIES IS THE CONCEPT. All three configs carry byte-identical
# generator, judge and loop blocks, copied verbatim from configs/example_generate.md
# (llama-3.3-70b generator, gpt-5.1-chat judge, n_batches 5 x batch_size 10,
# concurrency 5, memo 400 words, min_auroc_gain 0.0, exhausted_gain 0.002), and the
# generator/judge system prompts are byte-identical to that file's — verified, not
# assumed. So the three arms are directly readable against each other, and each is
# readable against example_generate.md's shape.
#
# WHAT EACH ARM COSTS. Per iteration: 5 generator calls (+ up to 2 top-ups each), a
# cache warm over the new samples, 5 candidate fits (one per batch, each scored on the
# whole dev set) and 1 union retrain — six probe fits per iteration, 30 per arm. The
# 27B extraction LLM is loaded ONLY to activate newly generated conversations: the eval
# splits and the dev set come precomputed from Kaggle (each config's kaggle: block), and
# the base training blob is computed once per arm and cached.
#
# VOLUME: 5 batches x 10 samples x 5 iterations = up to 250 generated samples per arm,
# of which only the accepted batches join the training set.
#
# NOISE FLOOR — read the ledger with this in mind. No arm sets probe.ensemble_size, so
# every fit is a SINGLE probe, matching example_generate.md. A single fit's dev AUROC
# moves by roughly +/-0.005 between near-identical training sets, and min_auroc_gain is
# 0.0, so some accepted batches will be noise rather than signal. If that dominates,
# the lever is probe.ensemble_size in the configs (it applies to all six fits an
# iteration does, so it multiplies the fit cost by that factor).
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   export KAGGLE_CONFIG_DIR=/path/to/dir/holding/kaggle.json   # the DIRECTORY, not the file
#   export HF_TOKEN=...            # or put it in hf_token.txt
#   mkdir -p logs
#   nohup bash run_gemma27b_generate_arms.sh > logs/run_generate_arms.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside
# it — its built-in stage list already points at these three arms, in this order:
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &
#
# Run ONE arm only, or reorder them: comment out the run_arm calls at the bottom.
# =============================================================================

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (both the generator and the judge are provider: openrouter)}"

# --- Kaggle credentials -----------------------------------------------------------------------
# Checked HERE rather than at first use. The dev prefetch happens before the first fit, but the
# eval prefetch does not run until the iter0 eval, and an unauthenticated KaggleApi.authenticate()
# ends in exit(1) — a SystemExit, not an exception. Both the eval splits AND the dev set are
# pulled from Kaggle in this experiment, so no credentials means no run at all, not a slow run.
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
# tuberlens' LLMModel.load calls hf_login(), which RAISES ValueError("No HuggingFace token
# found") when no token is set — even when the gemma weights are already in the local HF cache
# and no download happens. Without this check the run gets past the Kaggle prefetch and dies at
# the first extraction. The token only has to EXIST; an expired one logs a warning and the
# cached load proceeds.
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in hf_token.txt) — tuberlens hf_login() raises without one, even for a fully cached model}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"

# --- pin the extraction model's memory budget --------------------------------------------------
# PLACEMENT ONLY — this cannot change a single number the run produces. It fixes WHERE the frozen
# extraction LLM's weights live, not what they compute.
#
# Why pin at all: every tuberlens load uses device_map="auto", and UNPINNED accelerate infers the
# budget from whatever is FREE AT LOAD TIME. This loop loads the model once per iteration (the
# cache warm over newly generated samples) x 5 iterations x 3 arms, so one unlucky reload silently
# shifts the layer split and spills the executed tail to DISK — measured elsewhere in this repo at
# 48-264 s/sample against ~2.8 s/sample resident.
#
# Sized for a 24 GiB card (leaving ~2 GiB for the fit's activation staging and fragmentation) and
# ~57 GiB of host RAM. ADJUST IF THIS BOX IS DIFFERENT.
#
# Both vars deliberately: AGENTIC_REDTEAM_MAX_MEMORY is authoritative on this repo's
# load_extraction_model path, while tuberlens' own MAX_MEMORY reaches EVERY tuberlens load.
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
echo ">>> max_memory pinned: $AGENTIC_REDTEAM_MAX_MEMORY (placement only — does not change results)"

# --- ensemble fit/score path -------------------------------------------------------------------
# Left at tuberlens' default (fused ON). No arm sets probe.ensemble_size, so every fit is a single
# probe and this setting is inert as far as the fit goes — it is exported explicitly so the log
# records what the run used, and so raising ensemble_size in a config later does not silently
# change paths. Set PROBE_FUSED_ENSEMBLE=0 before launching to force the sequential fit and score.
export PROBE_FUSED_ENSEMBLE="${PROBE_FUSED_ENSEMBLE:-1}"
echo ">>> ensemble path: PROBE_FUSED_ENSEMBLE=$PROBE_FUSED_ENSEMBLE (inert while ensemble_size is unset)"

# No clobber guard on the per-arm output/probe dirs ON PURPOSE: --resume is on by default and is
# how a wiped container picks a run back up, which requires the existing probe-out-dir and its
# probe_iter*.pkl plus the run dir's batches.jsonl / guidance.jsonl. To start genuinely fresh,
# move the per-arm dirs aside first (or pass --no-resume, which leaves the files in place but
# reuses nothing from them).

OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is durably unusable"

run_arm () {  # $1 = config, $2 = base training data, $3 = eval dir, $4 = probe-out-dir, $5 = logfile
    echo ">>> $(date -Is)  START $1  -> $4   (log: $5)"
    local rc=0
    # --iterations is NOT passed: loop.iterations is 5 in every config, where the arms can be
    # diffed against each other. Passing it here would override all three at once and silently
    # mask a config edit. Same reasoning for --dev-data, --ensemble-size, --eval-max-samples and
    # the transform flags — every one of them is a config property these arms share or vary
    # deliberately, and flag > config precedence would hide that.
    #
    # --eval IS passed: it is a per-launch choice (do we score the eval splits at all), and it is
    # also what writes the comparison CSV the failsafe uses as this stage's finish signal.
    .venv_claude/bin/python scripts/iterative_generate.py "$1" \
        --base-training-data "$2" \
        --probe-out-dir "$4" \
        --eval --eval-dataset-dir "$3" \
        > "$5" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the key is dead.
        # Do NOT start the next arm — it would fail identically and spend hours of GPU producing
        # empty batches and a comparison CSV over probes that learned nothing.
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

# ARM 1 — high-stakes
run_arm configs/gen_gemma27b_highstakes.md \
        data/highstakes_llama70b_50.jsonl \
        eval_sets/highstakes \
        probes/gen_gemma27b_highstakes \
        logs/run_gen_gemma27b_highstakes.log

# ARM 2 — human harm
run_arm configs/gen_gemma27b_hu_harm.md \
        data/hu_harm_llama70b_50.jsonl \
        eval_sets/hu_ha \
        probes/gen_gemma27b_hu_harm \
        logs/run_gen_gemma27b_hu_harm.log

# ARM 3 — instruction following
run_arm configs/gen_gemma27b_instructions.md \
        data/instructions_llama70b_50.jsonl \
        eval_sets/instructions \
        probes/gen_gemma27b_instructions \
        logs/run_gen_gemma27b_instructions.log

echo ">>> $(date -Is)  all three arms finished."
