#!/usr/bin/env bash
set -e

# PROBE-DESCRIPTION experiment on the HUMAN-HARM concept, attacking a 10-MEMBER DEEP
# ENSEMBLE over a google/gemma-3-27b-it (L32) probe, for 5 iterations, with the probe fit
# validated against a HELD-OUT DEV SET.
#
# Both arms are experiment20's ARM 2 (cross_iteration_memos on at a 150-word budget,
# view_limit 8, batch_submissions) with TWO changes, and they differ from each other in
# ONE thing — the attacker model:
#
#                          attacker                  probe description   ensemble fit/score
#   [experiment20 ARM 2]   openai/gpt-oss-120b       one line            fused
#   ARM 1                  openai/gpt-oss-120b       full definition     sequential
#   ARM 2                  deepseek/deepseek-v4-pro  full definition     sequential
#
# 1. THE PROBE CARRIES A DEFINITION OF THE CONCEPT. `probe.description` in both configs
#    spells out what counts as harm to human well-being and what does not, and it now
#    reaches the JUDGE as well as the attacker (llm_judge `_concept_block` puts it in the
#    judge's system prompt; `_concept_context_line` puts it in both summarizers). Until
#    now the judge got the two label strings and nothing else, so what `harmful_to_human`
#    meant was whatever the judge model thought that call. The judge's label decides
#    `success` AND is what the retrain trains on, so this is the labelling function
#    itself, not a prompt tweak.
#
#    CONSEQUENCE FOR COMPARISON, stated once here and repeated in both configs: success
#    rates, clone rates and the red-team training labels are NOT row-for-row comparable
#    with experiment20's — a move in them is a move in what "harmful" means. The EVAL
#    numbers are: the eval splits carry their own fixed labels and never touch the judge,
#    so the comparison CSVs still read directly against experiment20's and the
#    experiment17 control's.
#
# 2. THE ENSEMBLE IS FIT AND SCORED SEQUENTIALLY (PROBE_FUSED_ENSEMBLE=0, exported below).
#    Wall-clock only: same members, same seeds, same averaged score, same pickle. It
#    cannot move a number this run reports.
#
# THE ATTACKER IS THE VARIABLE BETWEEN THE ARMS. gpt-oss-120b is the model experiment17
# and experiment20 held fixed; deepseek-v4-pro is experiment17's second attacker. Judge
# (openai/gpt-5.1), preprocessing model, probe, dev set, base data, eval splits, memo
# knobs, view_limit and every scheduling knob are identical across the two, so any delta
# between their CSVs is the attacker.
#
# VOLUME, identical in both arms: 10 sessions x 5 conversations x 5 rounds ~= 250 attempts
# per error type per ITERATION, over 5 iterations.
#
#   ARM 1  configs/gptoss120b_hu_harm_gemma27b_ens10_devval_itermemo150_view8_probedesc.md
#          -> results_hu_harm_gemma27b_gptoss120b_probedesc/
#             probes/hu_harm_gemma27b_gptoss120b_probedesc
#   ARM 2  configs/deepseekv4pro_hu_harm_gemma27b_ens10_devval_itermemo150_view8_probedesc.md
#          -> results_hu_harm_gemma27b_deepseekv4pro_probedesc/
#             probes/hu_harm_gemma27b_deepseekv4pro_probedesc
#
# ACTIVATIONS. The shared cache dir (results_hu_harm_gemma27b_batch_ablation/) is the same
# one experiment17/20 wrote, and no cache key mentions the probe description, the memo
# knobs, view_limit, ensemble_size or the fused/sequential choice — a description is not
# an input to any forward pass. So on a box that ran either of those, the eval blobs, the
# base blob and the dev blob are already warm and the only new extraction is this run's
# own novel red-team conversations.
#
# The per-arm output and probe dirs are NOT shared: their successes are found under a
# different attacker and a different labelling function.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hu_harm_probedesc.sh > logs/run_gemma27b_hu_harm_probedesc.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it —
# its built-in stage list already points at these two arms in this order, and it takes a
# fallback snapshot every 40 min on top of committing at every marker/probe:
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &

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

# HuggingFace token. tuberlens' LLMModel.load calls hf_login(), which RAISES
# ValueError("No HuggingFace token found") when no token is set — even though the gemma
# weights are already in the local HF cache and no download happens. So the run gets all the
# way past the initial train and the iter0 eval before dying at the first red-team model load,
# ~1 min in. Check it here instead. (The token only has to EXIST; an expired one logs a
# warning and the cached load proceeds.)
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in hf_token.txt) — tuberlens hf_login() raises without one, even for a fully cached model}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"

# --- pin the extraction model's memory budget -----------------------------------------------
# PLACEMENT ONLY — this cannot change a single number the run produces. It fixes WHERE the
# frozen extraction LLM's weights live, not what they compute, so the activations, the fits
# and the eval are identical with or without it. That is why setting it does not break
# comparability with the experiment17 control, which ran unpinned.
#
# Why pin at all: every tuberlens load uses device_map="auto", and UNPINNED accelerate infers
# the budget from whatever is FREE AT LOAD TIME. The model is reloaded on every red-team
# rotation and every retrain, so on a tight box one unlucky reload — anything else still
# holding GPU memory, torch's allocator holding reserved blocks — silently shifts the split
# and spills the executed tail to DISK. Measured elsewhere in this repo at 48-264 s/sample
# against ~2.8 s/sample resident, i.e. the difference between a run that finishes and one
# that does not.
#
# Sized for this box: a 24 GiB card (leave ~2 GiB for the fit's activation staging and
# fragmentation) and 57 GiB of host RAM (leave room for the merged activation tensors, which
# retrain.py's OOM analysis warns are the other half of the budget).
#
# Both vars, deliberately: AGENTIC_REDTEAM_MAX_MEMORY is authoritative on this repo's
# load_extraction_model path, while tuberlens' own MAX_MEMORY reaches EVERY tuberlens load —
# including get_performances, which this repo cannot pass model_kwargs to. Here get_performances
# is a pure cache hit (the `kaggle:` blobs), so MAX_MEMORY is belt-and-braces for the case where
# a blob is ever missing.
export AGENTIC_REDTEAM_MAX_MEMORY="0=22GiB,cpu=45GiB"
export MAX_MEMORY="0=22GiB,cpu=45GiB"
echo ">>> max_memory pinned: $AGENTIC_REDTEAM_MAX_MEMORY (placement only — does not change results)"

# --- fit and score the ensemble SEQUENTIALLY -------------------------------------------------
# WALL-CLOCK ONLY. PROBE_FUSED_ENSEMBLE=0 puts both halves of the deep ensemble back on the
# paths they took before fusion existed: `retrain._train_with_cached_base_activations` fits one
# ProbeFactory.build per seed instead of stepping the members together under vmap, and
# `EnsembleProbe._mean_proba` calls each member's predict_proba instead of one stacked forward.
# Same members, same repo-pinned seeds, same averaged score, same pickle — the fused path is an
# optimization with a fallback, so turning it off can only cost time, never change a result.
# One switch governs both halves by construction (ensemble.fusion_enabled), so this cannot end
# up half reverted.
export PROBE_FUSED_ENSEMBLE=0
echo ">>> ensemble: SEQUENTIAL fit and scoring (PROBE_FUSED_ENSEMBLE=0 — slower, identical results)"

SHARED_CACHE="results_hu_harm_gemma27b_batch_ablation"   # shared, arm-independent activation cache

# No clobber guard on the per-arm output/probe dirs ON PURPOSE: --resume is on by default and
# is how a wiped container picks a run back up, which requires the existing probe-out-dir and
# its markers. (The SHARED cache dir is likewise meant to persist and grow across both arms and
# across re-runs — and across the control's run before them.) To start genuinely fresh, move
# the per-arm dirs aside first.

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (warm already if this box ran experiment17 or 20;"
echo ">>>   otherwise filled by arm 1 and reused by arm 2. No cache key mentions the probe"
echo ">>>   description, the memo knobs, view_limit or the fused/sequential choice.)"

# --- run one arm ---------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    # --iterations 5: five red-team → retrain → eval cycles. Four iteration BOUNDARIES, which
    # is what this experiment is about — the control carried nothing across them, both arms here
    # carry the memo.
    # Same value the control ran, so the comparison CSVs line up row-for-row.
    # NOT passing --ensemble-size or --dev-data here on purpose. Both flags OVERRIDE the
    # configs (precedence is flag > config), and both knobs are properties of the probe both
    # arms (and the control) share, so they live in the configs — probe.ensemble_size: 10 and
    # validation.dev_data — where the arms can be diffed against each other. Setting either
    # here would silently mask a config edit.
    # --test-size / --split-field are likewise absent: dev_data makes retrain.py ignore them.
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 5 \
        --base-training-data data/hu_harm_llama70b_50.jsonl \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_sets/hu_ha \
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

run_arm configs/gptoss120b_hu_harm_gemma27b_ens10_devval_itermemo150_view8_probedesc.md \
        probes/hu_harm_gemma27b_gptoss120b_probedesc \
        logs/run_hu_harm_gemma27b_gptoss120b_probedesc.log

run_arm configs/deepseekv4pro_hu_harm_gemma27b_ens10_devval_itermemo150_view8_probedesc.md \
        probes/hu_harm_gemma27b_deepseekv4pro_probedesc \
        logs/run_hu_harm_gemma27b_deepseekv4pro_probedesc.log

echo ">>> $(date -Is)  both arms finished."
