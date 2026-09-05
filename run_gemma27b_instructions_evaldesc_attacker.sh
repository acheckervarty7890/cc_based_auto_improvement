#!/usr/bin/env bash
set -e

# ARM 9 — a THIRD arm on the llama70b instruction pair: everything from arm 6 (memo +
# eval.data_description) plus `attacker.show_eval_data_description: true`, which renders that
# same text directly into the ATTACKER's system prompt from round 0 instead of leaving it to
# reach the attacker second-hand through a memo.
#
#   arm 5  memo only                                     0.8216
#   arm 6  + description to the judge's summarizers       0.8542
#   arm 9  + the attacker is shown it too                 <- this run
#
# arm 6 -> arm 9 isolates the DELIVERY CHANNEL: same text, same judge, same base data, same
# schedule, byte-identical probe.description and system prompts. Verified by loading both
# configs and diffing every field: they differ in exactly one key plus the output paths.
# judge.eval_scope_check stays off, so the labelling function does not move and every metric
# stays comparable to arms 5 and 6.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   nohup bash run_gemma27b_instructions_evaldesc_attacker.sh > logs/run_arm9.out 2>&1 &
#
# Checkpointing: failsafe_commit.sh with this arm's single stage —
#   nohup bash failsafe_commit.sh \
#     --config configs/llama70b_instructions_gemma27b_l70base_evaldesc_attacker.md \
#     --probe-out-dir probes/instructions_gemma27b_llama70b_l70base_evaldesc_attacker \
#     --log-file logs/run_instructions_gemma27b_llama70b_l70base_evaldesc_attacker.log \
#     > logs/failsafe_arm9.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# The two base files are this experiment's variable — check them BEFORE the hours-long first
# arm rather than letting train_initial_probe discover one missing at arm 3.
LLAMA70B_BASE="data/instructions_llama70b_50.jsonl"
for base in "$LLAMA70B_BASE"; do
    if [ ! -f "$base" ]; then
        echo "ERROR: base training data not found: $base" >&2
        echo "       It comes from the generator_experiment_1 branch:" >&2
        echo "       git show origin/generator_experiment_1:$base > $base" >&2
        exit 1
    fi
    echo ">>> base training data: $base ($(wc -l < "$base") rows)"
done

# Kaggle credentials for the precomputed eval AND dev activations (configs' `kaggle:` section).
# Checked HERE rather than at first use: the dev prefetch runs before iteration 0 trains, and an
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
# ValueError("No HuggingFace token found") when no token is set — even though the gemma weights
# are already in the local HF cache and no download happens. So the run gets all the way past
# the initial train before dying at the first red-team model load. Check it here instead. (The
# token only has to EXIST; an expired one logs a warning and the cached load proceeds.)
if [ -z "${HF_TOKEN:-}" ] && [ -f hf_token.txt ]; then
    HF_TOKEN="$(tr -d '[:space:]' < hf_token.txt)"
    export HF_TOKEN
fi
: "${HF_TOKEN:?export HF_TOKEN (or put it in hf_token.txt) — tuberlens hf_login() raises without one, even for a fully cached model}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo ">>> HF token: present (${#HF_TOKEN} chars)"

# --- weight download path ---------------------------------------------------------------------
# TRANSFER LAYER ONLY — this cannot change a single number the run produces. It fixes how fast
# the frozen gemma-3-27b-it shards arrive, not what they contain.
#
# GUARDED on the import: huggingface_hub RAISES when HF_HUB_ENABLE_HF_TRANSFER=1 and the package
# is missing, which would turn a speedup into a dead run on any box that lacks it.
if .venv_claude/bin/python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    echo ">>> hf_transfer: enabled (parallel shard download; transfer layer only)"
else
    echo ">>> hf_transfer: NOT installed — using the default single-threaded download."
    echo ">>>   .venv_claude/bin/pip install hf_transfer  to speed up a cold HF cache."
fi

# --- pin the extraction model's memory budget -----------------------------------------------
# PLACEMENT ONLY — this cannot change a single number the run produces. It fixes WHERE the
# frozen extraction LLM's weights live, not what they compute.
#
# Why pin at all: every tuberlens load uses device_map="auto", and UNPINNED accelerate infers
# the budget from whatever is FREE AT LOAD TIME. The model is reloaded on every red-team
# rotation and every retrain — 10 iterations x 2 error types x 4 arms of them here — so one
# unlucky reload silently shifts the split and spills the executed tail to DISK. Measured
# elsewhere in this repo at 48-264 s/sample against ~2.8 s/sample resident.
#
# Sized for a 24 GiB card (leave ~2 GiB for the fit's activation staging and fragmentation) and
# ~57 GiB of host RAM. ADJUST IF THIS BOX IS DIFFERENT.
#
# Both vars deliberately: AGENTIC_REDTEAM_MAX_MEMORY is authoritative on this repo's
# load_extraction_model path, while tuberlens' own MAX_MEMORY reaches EVERY tuberlens load —
# including get_performances, which this repo cannot pass model_kwargs to.
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
echo ">>> max_memory pinned: $AGENTIC_REDTEAM_MAX_MEMORY (placement only — does not change results)"

# --- ensemble fit/score path ------------------------------------------------------------------
# Defaults to 0 = SEQUENTIAL, as in arms 1-4 and every earlier instruction experiment, so this run's probes
# are fit on the same path those runs' were (one ProbeFactory.build per seed, one predict_proba
# per member). That costs wall-clock: the fused path stacks the 10 members and steps them under
# vmap (measured 3.8x on a comparable shape), and this run does 40 retrains.
#
# Export PROBE_FUSED_ENSEMBLE=1 before launching to take that speedup. It is safe WITHIN this
# experiment — one switch governs both the fit and the scoring path (ensemble.fusion_enabled),
# so all four arms move together and stay comparable to each other. What it costs is exact
# comparability of the probes themselves with arms 1-4 and experiment_instruction_cloud_4..7,
# which ran sequential: the fused path changes the floating-point reduction order, which is a 4th-decimal
# effect on AUROC (no prediction flipped when it was measured) but not bit-identity.
export PROBE_FUSED_ENSEMBLE="${PROBE_FUSED_ENSEMBLE:-0}"
if [ "$PROBE_FUSED_ENSEMBLE" = "0" ]; then
    echo ">>> ensemble: SEQUENTIAL fit and scoring (PROBE_FUSED_ENSEMBLE=0 — matches arms 1-4, slower)"
else
    echo ">>> ensemble: FUSED fit and scoring (PROBE_FUSED_ENSEMBLE=$PROBE_FUSED_ENSEMBLE — faster; 4th-decimal drift vs those runs)"
fi
# Assert the setting actually reached tuberlens' pydantic settings (populated at import).
.venv_claude/bin/python - <<'PYEOF' || exit 1
import sys
from agentic_redteam.ensemble import fusion_enabled
import os
want = os.environ.get("PROBE_FUSED_ENSEMBLE", "0") != "0"
got = fusion_enabled()
print(f">>> ensemble fusion_enabled() = {got} (wanted {want})")
sys.exit(0 if got == want else 1)
PYEOF

SHARED_CACHE="results_instructions_gemma27b_shared"   # shared, arm-independent activation cache

# No clobber guard on the per-arm output/probe dirs ON PURPOSE: --resume is on by default and is
# how a wiped container picks a run back up, which requires the existing probe-out-dir and its
# markers. To start genuinely fresh, move the per-arm dirs aside first.

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (eval + dev blobs warm already if this box ran"
echo ">>>   arms 1-4 or experiment_instruction_cloud_1/_3/_4/_5/_6/_7; otherwise from Kaggle."
echo ">>>   The two 50-row base blobs are new: arm 5 computes one, arm 7 the other.)"

# --- run one arm ---------------------------------------------------------------------------
OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = base training data, $4 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (base: $3, log: $4)"
    local rc=0
    # --iterations 10: ten red-team → retrain → eval cycles, i.e. NINE iteration boundaries for
    # the memo to cross. Identical in all four arms — the contrast is about what crosses a
    # boundary, so the number of boundaries must not vary.
    #
    # NOT passing --ensemble-size or --dev-data here on purpose. Both flags OVERRIDE the configs
    # (precedence is flag > config), and both are properties of the probe all four arms share,
    # so they live in the configs — probe.ensemble_size: 10 and validation.dev_data — where the
    # arms can be diffed against each other. Setting either here would silently mask a config
    # edit. --test-size / --split-field are likewise absent: dev_data makes retrain.py ignore
    # them.
    #
    # --base-training-data IS passed here, and unlike the earlier runs it is NOT the same for
    # every arm: it is the self-generated-base variable, so it pairs with the attacker.
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 10 \
        --base-training-data "$3" \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_sets/instructions \
        > "$4" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the key is dead.
        # Do NOT start the next arm — it would fail identically and write a comparison CSV from
        # probes trained on nothing.
        echo ">>> $(date -Is)  ABORTED $1 — OpenRouter unusable (exit $rc)." >&2
        tail -n 5 "$4" >&2
        echo ">>> Fix credits/key, then re-run with --resume to continue this arm." >&2
        exit "$rc"
    elif [ "$rc" -ne 0 ]; then
        echo ">>> $(date -Is)  FAILED  $1 (exit $rc) — see $4" >&2
        exit "$rc"
    fi
    echo ">>> $(date -Is)  DONE  $1"
}

run_arm configs/llama70b_instructions_gemma27b_l70base_evaldesc_attacker.md \
        probes/instructions_gemma27b_llama70b_l70base_evaldesc_attacker \
        "$LLAMA70B_BASE" \
        logs/run_instructions_gemma27b_llama70b_l70base_evaldesc_attacker.log

echo ">>> $(date -Is)  arm 9 finished."
