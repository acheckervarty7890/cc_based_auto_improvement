#!/usr/bin/env bash
set -e

# DEEPSEEK-SELF-GENERATED-BASE experiment on the HUMAN-HARM concept, attacking a 10-MEMBER
# DEEP ENSEMBLE over a google/gemma-3-27b-it (L32) probe, for TEN iterations, with the probe
# fit validated against a HELD-OUT DEV SET (dev_samples/hu_ha).
#
# TWO ARMS, ONE ATTACKER (deepseek/deepseek-v4-pro). This is experiment25 re-run with a
# different attacker, on that attacker's OWN generated base:
#
#                                            cross_iteration_memos   eval.data_description
#   ARM 1  memo (150-word budget)                              true                   unset
#   ARM 2  memo + eval-data description                        true      the four data kinds
#
#   ARM 1 -> ARM 2   what does telling the memo-writer which KINDS of conversation the probe
#                    is scored on buy, on top of carrying a hand-off memo across the
#                    iteration boundary?
#
# ATTACKER *AND* BASE MOVE TOGETHER VS EXPERIMENT25, to preserve the self-generated-base
# property (the model that wrote the initial probe's training data is the one attacking it):
#
#   experiment25   openai/gpt-oss-120b       data/hu_harm_gptoss_50.jsonl
#   this run       deepseek/deepseek-v4-pro  data/hu_harm_deepseekv4pro_50.jsonl
#
# Both 25 harmful_to_human / 25 not_harmful_to_human, same schema, both from
# generator_experiment_1 (commit 8136e5ec, "per-generator 50-row bases, to make each arm
# single-source"), each verified disjoint from its own generator's 600-row set. Passed here
# rather than in the configs because it is identical in both arms.
#
# CONSEQUENCE FOR READING THE RESULTS: absolute AUROCs are NOT comparable to experiment25's
# (different base => different initial probe). What IS comparable is the ARM 1 -> ARM 2 gap
# and the shape of each curve. experiment25's answer: both arms started at 0.8781 and
# finished within 0.0013 (0.9034 / 0.9021) — the knob moved the PATH, not the destination,
# while roughly doubling red-team yield (250 vs 153 successes). This run tests replication.
#
# WHY NO CONTROL ARM. experiment23 already measured control -> memo on this concept, probe
# and schedule. The contrast this run exists for is the top rung, so the budget goes there.
#
# HOW TO READ THE RESULTS. ARM 1 vs ARM 2 WITHIN this run is a clean contrast — the two
# configs differ by exactly one key (eval.data_description), probe.description is
# byte-identical, so the judge labels the same way in both and success rate, clone rate,
# red-team labels and eval CSVs are all comparable. Against EXPERIMENT23's numbers, only the
# SHAPE of the ARM 1 -> ARM 2 gap is comparable: a different base means a different initial
# probe, hence different absolute AUROCs and success rates throughout.
#
# THE SCHEDULE, identical in both arms and unchanged from experiment23:
#
#     rounds: 5                 4 round-memo updates per error type per iteration
#     sessions_per_model: 3
#     max_turns: 5              batch size
#     concurrency: 3            >= sessions_per_model x len(models)
#     batch_target: 30          inert: a round can produce at most 3 x 5 = 15 attempts
#
# VOLUME per arm: 15 attempts/round x 5 rounds = 75 per error type per iteration, x2 error
# types = 150/iteration, x10 iterations = ~1500 attempts.
#
# ARMS AND WHERE THEY WRITE:
#
#   ARM 1  configs/deepseekv4pro_hu_harm_gemma27b_dsbase_itermemo150.md
#          -> results_hu_harm_gemma27b_deepseekv4pro_dsbase_itermemo150/
#             probes/hu_harm_gemma27b_deepseekv4pro_dsbase_itermemo150
#   ARM 2  configs/deepseekv4pro_hu_harm_gemma27b_dsbase_itermemo150_evaldesc.md
#          -> results_hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc/
#             probes/hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc
#
# ACTIVATIONS. The shared cache dir (results_hu_harm_gemma27b_batch_ablation/) is the one
# experiments 11/16/17/20/21/22/23/25 wrote. No cache key mentions the memo knobs, the
# eval-data description, view_limit, sessions_per_model, ensemble_size or the iteration
# count, so on a box that ran any of those the EVAL blobs and the 290-row DEV blob are
# already warm. The BASE blob is keyed on a hash of the base data file, so the new deepseek
# base gets its OWN key: arm 1 computes it once (50 rows) and arm 2 reuses it. Nothing
# existing is invalidated. The per-arm output and probe dirs are NOT shared.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hu_harm_dsbase_arms.sh > logs/run_dsbase_arms.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it —
# its built-in stage list already points at these two arms in this order:
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# The base data is this experiment's variable — check it BEFORE the hours-long first arm
# rather than letting train_initial_probe discover it missing.
BASE_DATA="data/hu_harm_deepseekv4pro_50.jsonl"
if [ ! -f "$BASE_DATA" ]; then
    echo "ERROR: base training data not found: $BASE_DATA" >&2
    echo "       It comes from the generator_experiment_1 branch:" >&2
    echo "       git show origin/generator_experiment_1:$BASE_DATA > $BASE_DATA" >&2
    exit 1
fi
echo ">>> base training data: $BASE_DATA ($(wc -l < "$BASE_DATA") rows, generated by deepseek/deepseek-v4-pro)"

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
# way past the initial train and the iter0 eval before dying at the first red-team model
# load. Check it here instead. (The token only has to EXIST; an expired one logs a warning
# and the cached load proceeds.)
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
# On a fresh box the HF cache is empty and the ~54 GB of shards come down the default
# single-threaded python path, measured here at ~213 MB/min = ~4 h before the first activation
# is extracted. hf_transfer parallelizes the same download.
#
# GUARDED on the import: huggingface_hub RAISES when HF_HUB_ENABLE_HF_TRANSFER=1 and the
# package is missing, which would turn a speedup into a dead run on any box that lacks it.
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
# rotation and every retrain — 10 iterations x 2 error types x 2 arms of them here — so one
# unlucky reload silently shifts the split and spills the executed tail to DISK. Measured
# elsewhere in this repo at 48-264 s/sample against ~2.8 s/sample resident.
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
# same path those runs' were. That costs wall-clock: the fused path stacks the 10 members and
# steps them under vmap (measured 3.8x on a comparable shape), and this run does 20 retrains.
#
# Export PROBE_FUSED_ENSEMBLE=1 before launching to take that speedup. It is safe WITHIN this
# experiment — one switch governs both the fit and the scoring path (ensemble.fusion_enabled),
# so both arms move together and stay comparable to each other. What it costs is exact
# comparability of the probes themselves with experiment21/22/23, which ran sequential: the
# fused path changes the floating-point reduction order, which is a 4th-decimal effect on
# AUROC (no prediction flipped when it was measured) but not bit-identity.
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
echo ">>> activation cache: $SHARED_CACHE (eval + dev blobs warm already if this box ran"
echo ">>>   experiment 11/16/17/20/21/22/23/25; the deepseek base blob is new, computed by arm 1.)"

# --- run one arm ---------------------------------------------------------------------------
OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    # --iterations 10: ten red-team → retrain → eval cycles, i.e. NINE iteration boundaries
    # for the memo to cross. Identical in both arms — the contrast is about what crosses a
    # boundary, so the number of boundaries must not vary.
    #
    # NOT passing --ensemble-size or --dev-data here on purpose. Both flags OVERRIDE the
    # configs (precedence is flag > config), and both are properties of the probe both arms
    # share, so they live in the configs — probe.ensemble_size: 10 and validation.dev_data —
    # where the arms can be diffed against each other. Setting either here would silently
    # mask a config edit.
    # --test-size / --split-field are likewise absent: dev_data makes retrain.py ignore them.
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 10 \
        --base-training-data "$BASE_DATA" \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_sets/hu_ha \
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

run_arm configs/deepseekv4pro_hu_harm_gemma27b_dsbase_itermemo150.md \
        probes/hu_harm_gemma27b_deepseekv4pro_dsbase_itermemo150 \
        logs/run_hu_harm_gemma27b_deepseekv4pro_dsbase_itermemo150.log

run_arm configs/deepseekv4pro_hu_harm_gemma27b_dsbase_itermemo150_evaldesc.md \
        probes/hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc \
        logs/run_hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc.log

echo ">>> $(date -Is)  both arms finished."
