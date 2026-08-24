#!/usr/bin/env bash
set -e

# TWO ARMS: experiment23's memo-ladder ARM 3 re-run twice with `eval.data_description`
# narrowed to a SINGLE eval split — a DIFFERENT split in each arm. HUMAN-HARM concept,
# attacking a 10-MEMBER DEEP ENSEMBLE over google/gemma-3-27b-it (L32), for TEN iterations,
# with the probe fit validated against a HELD-OUT DEV SET (dev_samples/hu_ha).
#
#                              cross_iteration_memos   eval.data_description   probe.description
#   exp23 ARM 1  control                       false                   unset       one-liner
#   exp23 ARM 2  memo                           true                   unset       one-liner
#   exp23 ARM 3  memo + evaldesc                true    four kinds, 1 line ea.     one-liner
#   ARM 3a       ONE-kind evaldesc              true       ant_hh only             one-liner
#   ARM 3b       ONE-kind evaldesc              true    balanced_refusal only      one-liner
#
# THE QUESTION. exp23's arm 3 named all four eval_sets/hu_ha splits, so its memos were steered
# to SPREAD over them. Each arm here names ONE, so there is nothing to spread over and the
# steering inverts from breadth to depth. Between 3a and 3b the ONLY thing that varies is
# which split — the two configs are byte-identical apart from that block and their output
# paths, and llm_judge is shared, so the pair isolates the choice of split.
#
# WHY THESE TWO SPLITS. They are the ends of the hu_ha eval set on the dimension that matters
# for a probe. In eval_ant_hh no surface cue separates the classes: both sides run the same
# length and register misleads in both directions (crude-but-harmless is not-harmful,
# polite-but-damaging is harmful). In eval_balanced_refusal one cue separates them almost
# perfectly: the non-harmful side is nearly all bare one-line refusals. So the pair asks
# whether pointing the memo-writer at a split with a clean surface cue steers the attacker
# into that cue's failure mode — a harmful reply worded as a refusal — or merely into
# producing more of what the cue already handles.
#
# THE EVAL IS UNCHANGED IN BOTH ARMS — the probe is still scored on all four hu_ha splits and
# each arm's comparison CSV still reports all four. Only what the memo-writer is told differs,
# which is what makes the cross-split effect readable: if describing one split lifts only that
# split, that is steering; if it lifts all four, it is something else.
#
# COMPARABILITY TO EXPERIMENT23 IS FULL, on every metric. `probe.description` is unchanged in
# both arms — the same one-line definition every hu_harm arm since experiment17 has carried —
# so the ATTACKER prompt and the JUDGE'S CLASSIFICATION PROMPT are byte-identical to arm 3's
# and the labelling function has not moved. Read each arm against
# results_hu_harm_gemma27b_gptoss120b_s3_evaldesc/ (exp23 arm 3), and against each other.
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
#   configs/gptoss120b_hu_harm_gemma27b_ens10_devval_s3_itermemo150_evaldesc_anthh.md
#     -> results_hu_harm_gemma27b_gptoss120b_s3_evaldesc_anthh/
#        probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc_anthh
#
# ACTIVATIONS. The shared cache dir (results_hu_harm_gemma27b_batch_ablation/) is the one
# experiments 11/16/17/20/21/22/23 wrote, and no cache key mentions the memo knobs, the
# eval-data description, view_limit, sessions_per_model, ensemble_size
# or the iteration count. So on a box that ran any of those, the eval blobs, the base blob and
# the dev blob are already warm and the only new extraction is this run's own novel red-team
# conversations. The output and probe dirs are NOT shared.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hu_harm_evaldesc_arms.sh > logs/run_evaldesc_arms.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it —
# its built-in stage list already points at these two arms, in this order:
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
# rotation and every retrain — 10 iterations x 2 error types x 2 arms of them here — so one unlucky
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
# a comparable shape), and this runs 10 retrains per arm, 20 in all.
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

# No clobber guard on the per-arm output/probe dirs ON PURPOSE: --resume is on by default and
# is how a wiped container picks a run back up, which requires the existing probe-out-dir and
# its markers. To start genuinely fresh, move the per-arm dirs aside first.

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (warm already if this box ran experiment 11/16/17/20/21/22/23;"
echo ">>>   otherwise filled by this run — eval blobs come from Kaggle, no 27B forwards.)"

# --- run one arm ------------------------------------------------------------------------------
OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    # --iterations 10: ten red-team → retrain → eval cycles, i.e. NINE iteration boundaries for
    # the cross-iteration memo to cross. Same as experiment23, so the curves are the same
    # length, and identical in both arms — the arms differ in one config block, nothing else.
    #
    # NOT passing --ensemble-size or --dev-data here on purpose. Both flags OVERRIDE the config
    # (precedence is flag > config), and both are properties of the probe both arms share, so
    # they live in the configs — probe.ensemble_size: 10 and validation.dev_data — where the
    # arms can be diffed against each other. Setting either here would silently mask a config
    # edit. --test-size / --split-field are likewise absent: dev_data makes retrain.py ignore
    # them.
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 10 \
        --base-training-data data/hu_harm_llama70b_50.jsonl \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_sets/hu_ha \
        > "$3" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the key is dead.
        # Do NOT start the next arm — it would fail identically and write a comparison CSV from
        # probes trained on nothing.
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

# ARM 3a — eval_ant_hh described
run_arm configs/gptoss120b_hu_harm_gemma27b_ens10_devval_s3_itermemo150_evaldesc_anthh.md \
        probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc_anthh \
        logs/run_hu_harm_gemma27b_gptoss120b_s3_evaldesc_anthh.log

# ARM 3b — eval_balanced_refusal described
run_arm configs/gptoss120b_hu_harm_gemma27b_ens10_devval_s3_itermemo150_evaldesc_refusal.md \
        probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc_refusal \
        logs/run_hu_harm_gemma27b_gptoss120b_s3_evaldesc_refusal.log

echo ">>> $(date -Is)  both arms finished."
