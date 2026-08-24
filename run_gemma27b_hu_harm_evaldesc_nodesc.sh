#!/usr/bin/env bash
set -e

# ONE ARM: experiment23's memo-ladder ARM 3 re-run with `probe.description` REMOVED.
# HUMAN-HARM concept, attacking a 10-MEMBER DEEP ENSEMBLE over google/gemma-3-27b-it (L32),
# for TEN iterations, with the probe fit validated against a HELD-OUT DEV SET
# (dev_samples/hu_ha).
#
#                                cross_iteration_memos   eval.data_description   probe.description
#   exp23 ARM 1  control                         false                   unset       one-liner
#   exp23 ARM 2  memo                             true                   unset       one-liner
#   exp23 ARM 3  memo + evaldesc                  true      the four data kinds      one-liner
#   THIS RUN     ditto, no probe description      true      the four data kinds       ABSENT
#
# THE QUESTION. `probe.description` is the concept text that reaches the ATTACKER, the
# JUDGE'S CLASSIFICATION PROMPT and both SUMMARIZERS. Removing it leaves the four data kinds
# in `eval.data_description` as the only concept text anywhere in the run — and those reach
# the two summarizers ONLY. So: with the concept defined nowhere but the memo-writer's own
# context, does arm 3's behaviour survive?
#
# WHAT IS AND IS NOT COMPARABLE TO EXPERIMENT23. Dropping the description moves the JUDGE'S
# classification prompt, i.e. the labelling function. So:
#   - COMPARABLE: the eval comparison CSVs. The eval splits carry their own labels and
#     evaluate_probe never reads a probe description.
#   - NOT COMPARABLE: success rate, clone rate and the red-team training labels — those are
#     defined by a judge that is now prompted differently.
# There is no no-description CONTROL yet (exp23's three arms all carry the one-liner), so the
# red-team-side numbers here are this run's own baseline, not a rung on that ladder.
#
# THE SCHEDULE, carried unchanged from experiment23:
#
#     rounds: 5                 4 round-memo updates per error type per iteration
#     sessions_per_model: 3
#     max_turns: 5              batch size
#     concurrency: 3            >= sessions_per_model x len(models)
#     batch_target: 30          inert: a round produces at most 3 x 5 = 15 attempts
#
# VOLUME: 15 attempts/round x 5 rounds = 75 per error type per iteration, x2 error types =
# 150/iteration, x10 iterations = ~1500 attempts.
#
# WHERE IT WRITES:
#   configs/gptoss120b_hu_harm_gemma27b_ens10_devval_s3_itermemo150_evaldesc_nodesc.md
#     -> results_hu_harm_gemma27b_gptoss120b_s3_evaldesc_nodesc/
#        probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc_nodesc
#
# ACTIVATIONS. The shared cache dir (results_hu_harm_gemma27b_batch_ablation/) is the one
# experiments 11/16/17/20/21/22/23 wrote, and no cache key mentions the memo knobs, the
# eval-data description, THE PROBE DESCRIPTION, view_limit, sessions_per_model, ensemble_size
# or the iteration count. So on a box that ran any of those, the eval blobs, the base blob and
# the dev blob are already warm and the only new extraction is this run's own novel red-team
# conversations. The output and probe dirs are NOT shared.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hu_harm_evaldesc_nodesc.sh > logs/run_evaldesc_nodesc.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it —
# its built-in stage list already points at this arm:
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# Kaggle credentials for the precomputed eval activations (the config's `kaggle:` section).
# Checked HERE rather than at first use: the first eval is hours in, and an unauthenticated
# KaggleApi.authenticate() ends in exit(1), not an exception.
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

# HuggingFace token. tuberlens' LLMModel.load calls hf_login(), which RAISES
# ValueError("No HuggingFace token found") when no token is set — even though the gemma
# weights are already in the local HF cache and no download happens. So the run gets all the
# way past the initial train and the iter0 eval before dying at the first red-team model load.
# Check it here instead. (The token only has to EXIST; an expired one logs a warning and the
# cached load proceeds.)
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in hf_token.txt) — tuberlens hf_login() raises without one, even for a fully cached model}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"

# --- pin the extraction model's memory budget -----------------------------------------------
# PLACEMENT ONLY — this cannot change a single number the run produces. It fixes WHERE the
# frozen extraction LLM's weights live, not what they compute.
#
# Why pin at all: every tuberlens load uses device_map="auto", and UNPINNED accelerate infers
# the budget from whatever is FREE AT LOAD TIME. The model is reloaded on every red-team
# rotation and every retrain — 10 iterations x 2 error types of them here — so one unlucky
# reload silently shifts the split and spills the executed tail to DISK. Measured elsewhere in
# this repo at 48-264 s/sample against ~2.8 s/sample resident.
#
# Sized for a 24 GiB card (leave ~2 GiB for the fit's activation staging and fragmentation)
# and ~57 GiB of host RAM. ADJUST IF THIS BOX IS DIFFERENT.
#
# Both vars deliberately: AGENTIC_REDTEAM_MAX_MEMORY is authoritative on this repo's
# load_extraction_model path, while tuberlens' own MAX_MEMORY reaches EVERY tuberlens load —
# including get_performances, which this repo cannot pass model_kwargs to.
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
echo ">>> max_memory pinned: $AGENTIC_REDTEAM_MAX_MEMORY (placement only — does not change results)"

# --- ensemble fit/score path ------------------------------------------------------------------
# Defaults to 0 = SEQUENTIAL, matching experiment21/22/23, so this run's probes are fit on the
# same path those runs' were and the CSVs stay directly readable against them. That costs
# wall-clock: the fused path stacks the 10 members and steps them under vmap (measured 3.8x on
# a comparable shape), and this run does 10 retrains.
#
# Export PROBE_FUSED_ENSEMBLE=1 before launching to take that speedup. What it costs is exact
# comparability of the probes themselves with experiment21/22/23, which ran sequential: the
# fused path changes the floating-point reduction order, which is a 4th-decimal effect on AUROC
# (no prediction flipped when it was measured) but not bit-identity.
export PROBE_FUSED_ENSEMBLE="${PROBE_FUSED_ENSEMBLE:-0}"
if [ "$PROBE_FUSED_ENSEMBLE" = "0" ]; then
    echo ">>> ensemble: SEQUENTIAL fit and scoring (PROBE_FUSED_ENSEMBLE=0 — matches experiment21/22/23, slower)"
else
    echo ">>> ensemble: FUSED fit and scoring (PROBE_FUSED_ENSEMBLE=$PROBE_FUSED_ENSEMBLE — faster; 4th-decimal drift vs experiment21/22/23)"
fi

SHARED_CACHE="results_hu_harm_gemma27b_batch_ablation"   # shared, arm-independent activation cache

# No clobber guard on the output/probe dirs ON PURPOSE: --resume is on by default and is how a
# wiped container picks a run back up, which requires the existing probe-out-dir and its
# markers. To start genuinely fresh, move those dirs aside first.

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (warm already if this box ran experiment 11/16/17/20/21/22/23;"
echo ">>>   otherwise filled by this run — eval blobs come from Kaggle, no 27B forwards.)"

# --- run -------------------------------------------------------------------------------------
OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"

CONFIG=configs/gptoss120b_hu_harm_gemma27b_ens10_devval_s3_itermemo150_evaldesc_nodesc.md
PROBE_DIR=probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc_nodesc
LOG=logs/run_hu_harm_gemma27b_gptoss120b_s3_evaldesc_nodesc.log

echo ">>> $(date -Is)  START $CONFIG  -> $PROBE_DIR   (log: $LOG)"
rc=0
# --iterations 10: ten red-team → retrain → eval cycles, i.e. NINE iteration boundaries for
# the cross-iteration memo to cross. Same as experiment23, so the curves are the same length.
#
# NOT passing --ensemble-size or --dev-data here on purpose. Both flags OVERRIDE the config
# (precedence is flag > config), and both are properties of the probe, so they live in the
# config — probe.ensemble_size: 10 and validation.dev_data — where it can be diffed against
# experiment23's arms. Setting either here would silently mask a config edit.
# --test-size / --split-field are likewise absent: dev_data makes retrain.py ignore them.
.venv_claude/bin/python scripts/iterative_retrain.py "$CONFIG" \
    --iterations 10 \
    --base-training-data data/hu_harm_llama70b_50.jsonl \
    --probe-out-dir "$PROBE_DIR" \
    --eval --eval-dataset-dir eval_sets/hu_ha \
    > "$LOG" 2>&1 || rc=$?

if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
    # The circuit breaker stopped the run: OpenRouter is out of credits or the key is dead.
    echo ">>> $(date -Is)  ABORTED $CONFIG — OpenRouter unusable (exit $rc)." >&2
    tail -n 5 "$LOG" >&2
    echo ">>> Fix credits/key, then re-run with --resume to continue." >&2
    exit "$rc"
elif [ "$rc" -ne 0 ]; then
    echo ">>> $(date -Is)  FAILED  $CONFIG (exit $rc) — see $LOG" >&2
    exit "$rc"
fi
echo ">>> $(date -Is)  DONE  $CONFIG"
