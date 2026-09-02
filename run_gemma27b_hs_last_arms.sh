#!/usr/bin/env bash
set -e

# SELF-GENERATED-BASE memo experiment on the HIGH-STAKES concept, attacking a SINGLE linear
# probe over google/gemma-3-27b-it (L32), for TEN iterations, with the probe fit validated
# against a HELD-OUT DEV SET (dev_samples/highstakes).
#
# FOUR ARMS, TWO ATTACKERS, run SEQUENTIALLY on one box. This is experiment25 (gpt-oss) and
# experiment26 (deepseek) carried over from human-harm to high-stakes, with all four arms in
# one branch:
#
#                                 attacker              base data        eval.data_description
#   ARM 1                         gpt-oss-120b          gptoss_50                       unset
#   ARM 2                         gpt-oss-120b          gptoss_50            the four hs kinds
#   ARM 3                         deepseek-v4-pro       deepseekv4pro_50                unset
#   ARM 4                         deepseek-v4-pro       deepseekv4pro_50     the four hs kinds
#
#   ARM 1 -> ARM 2  and  ARM 3 -> ARM 4   what does telling the memo-writer which KINDS of
#                    conversation the probe is scored on buy, on top of carrying a hand-off
#                    memo across the iteration boundary — and does it do the same thing under
#                    two different attackers?
#
# `cross_iteration_memos` is ON in all four. experiment23 already measured control -> memo, so
# the budget goes to the top rung, exactly as experiment25/26 spent theirs.
#
# THE BASE DATA IS PER-ATTACKER, and that is the design property carried over: the model that
# wrote the initial probe's training data is the model that then attacks it.
#
#   ARMS 1-2   data/highstakes_gptoss_50.jsonl        25 high-stakes / 25 low-stakes
#   ARMS 3-4   data/highstakes_deepseekv4pro_50.jsonl 25 high-stakes / 25 low-stakes
#
# Both from generator_experiment_1 (commit 8136e5ec, "per-generator 50-row bases, to make each
# arm single-source"), each verified to have zero overlap with its own generator's 600-row set,
# with each other, and with eval_sets/highstakes and dev_samples/highstakes. It is passed HERE
# rather than in the configs because it pairs with the attacker, and the pairing is what this
# runner exists to keep straight.
#
# HOW TO READ THE RESULTS. WITHIN an attacker (1 vs 2, 3 vs 4) the contrast is clean: those two
# configs differ by exactly one key (eval.data_description) plus their output paths,
# probe.description is byte-identical, so the judge labels the same way in both and success
# rate, clone rate, red-team labels and eval CSVs are all comparable. ACROSS attackers only the
# SHAPE of the gap is comparable — a different base means a different initial probe, hence
# different absolute AUROCs throughout. Nothing here is comparable to experiment25/26's numbers:
# different concept, different eval splits, different probe size.
#
# A SINGLE PROBE, NOT experiment25/26's 10-MEMBER ENSEMBLE. `probe.ensemble_size: 1` in all four
# configs — the ordinary single-probe path, not a one-member ensemble. It is a cost decision:
# the high-stakes dev set is 1908 rows (~19.6 GB of gemma activations) against hu_ha's 290, is
# resident for the whole fit and is scored every epoch, and this run does 40 retrains.
#
# THE SCHEDULE, identical in all four arms and unchanged from experiment25/26:
#
#     rounds: 5                 4 round-memo updates per error type per iteration
#     sessions_per_model: 3
#     max_turns: 5              batch size
#     concurrency: 3            >= sessions_per_model x len(models)
#     batch_target: 30          inert: a round can produce at most 3 x 5 = 15 attempts
#
# VOLUME per arm: 15 attempts/round x 5 rounds = 75 per error type per iteration, x2 error
# types = 150/iteration, x10 iterations = ~1500 attempts. Four arms of that, one after another.
#
# ARMS AND WHERE THEY WRITE:
#
#   ARM 1  configs/gptoss120b_hs_gemma27b_gptossbase_itermemo150.md
#          -> results_hs_gemma27b_gptoss120b_gptossbase_itermemo150/
#             probes/hs_gemma27b_gptoss120b_gptossbase_itermemo150
#   ARM 2  configs/gptoss120b_hs_gemma27b_gptossbase_itermemo150_evaldesc.md
#          -> results_hs_gemma27b_gptoss120b_gptossbase_evaldesc/
#             probes/hs_gemma27b_gptoss120b_gptossbase_evaldesc
#   ARM 3  configs/deepseekv4pro_hs_gemma27b_dsbase_itermemo150.md
#          -> results_hs_gemma27b_deepseekv4pro_dsbase_itermemo150/
#             probes/hs_gemma27b_deepseekv4pro_dsbase_itermemo150
#   ARM 4  configs/deepseekv4pro_hs_gemma27b_dsbase_itermemo150_evaldesc.md
#          -> results_hs_gemma27b_deepseekv4pro_dsbase_evaldesc/
#             probes/hs_gemma27b_deepseekv4pro_dsbase_evaldesc
#
# ACTIVATIONS. The shared cache dir (results_hs_gemma27b_devval/) is the one experiment18/19
# wrote. No cache key mentions the memo knobs, the eval-data description, view_limit,
# sessions_per_model, ensemble_size or the iteration count, so on a box that ran either of
# those the four EVAL blobs (~48 GB) and the 1908-row DEV blob (~21 GB) are already warm and
# neither is recomputed. The BASE blob is keyed on a hash of the base data file, so each of the
# two new 50-row bases gets its OWN key: arm 1 computes the gpt-oss one and arm 2 reuses it,
# arm 3 computes the deepseek one and arm 4 reuses it. Nothing existing is invalidated. The
# per-arm output and probe dirs are NOT shared.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hs_last_arms.sh > logs/run_hs_last_arms.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it —
# its built-in stage list already points at these four arms in this order:
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# The base data is this experiment's variable — check BOTH sets BEFORE the hours-long first
# arm rather than letting arm 3's train_initial_probe discover its file missing a day later.
GPTOSS_BASE="data/highstakes_gptoss_50.jsonl"
DS_BASE="data/highstakes_deepseekv4pro_50.jsonl"
for base_data in "$GPTOSS_BASE" "$DS_BASE"; do
    if [ ! -f "$base_data" ]; then
        echo "ERROR: base training data not found: $base_data" >&2
        echo "       It comes from the generator_experiment_1 branch:" >&2
        echo "       git show origin/generator_experiment_1:$base_data > $base_data" >&2
        exit 1
    fi
done
echo ">>> base training data (arms 1-2): $GPTOSS_BASE ($(wc -l < "$GPTOSS_BASE") rows, generated by openai/gpt-oss-120b)"
echo ">>> base training data (arms 3-4): $DS_BASE ($(wc -l < "$DS_BASE") rows, generated by deepseek/deepseek-v4-pro)"

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
# rotation and every retrain — 10 iterations x 2 error types x 4 arms of them here — so one
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

# NO PROBE_FUSED_ENSEMBLE HERE, unlike run_gemma27b_hu_harm_*base_arms.sh. That switch chooses
# between the fused and the sequential ensemble fit/score paths, and at probe.ensemble_size: 1
# neither is reached: retrain._resolve_ensemble_seeds carves n == 1 out, one ProbeFactory.build
# runs, and the pickle is a plain tuberlens probe with no EnsembleProbe to score. Setting it
# would be inert — and would imply this run has members it does not have.

SHARED_CACHE="results_hs_gemma27b_devval"   # shared, arm-independent activation cache (experiment18/19)

# No clobber guard on the per-arm output/probe dirs ON PURPOSE: --resume is on by default and
# is how a wiped container picks a run back up, which requires the existing probe-out-dir and
# its markers. To start genuinely fresh, move the per-arm dirs aside first.

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (the four eval blobs and the 1908-row dev blob are warm"
echo ">>>   already if this box ran experiment18/19; the two 50-row base blobs are new.)"

# --- run one arm ---------------------------------------------------------------------------
OUTAGE_EXIT_CODE=3   # cli.OUTAGE_EXIT_CODE — "OpenRouter is unusable"

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile, $4 = base training data
    echo ">>> $(date -Is)  START $1  -> $2   (base: $4, log: $3)"
    local rc=0
    # --iterations 10: ten red-team → retrain → eval cycles, i.e. NINE iteration boundaries
    # for the memo to cross. Identical in all four arms — the contrast is about what crosses a
    # boundary, so the number of boundaries must not vary.
    #
    # NOT passing --ensemble-size or --dev-data here on purpose. Both flags OVERRIDE the
    # configs (precedence is flag > config), and both are properties of the probe all four
    # arms share, so they live in the configs — probe.ensemble_size: 1 and validation.dev_data
    # — where the arms can be diffed against each other. Setting either here would silently
    # mask a config edit.
    # --test-size / --split-field are likewise absent: dev_data makes retrain.py ignore them.
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 10 \
        --base-training-data "$4" \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_sets/highstakes \
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

# Arms run in pair order (both gpt-oss arms, then both deepseek arms) so that each pair's
# shared base blob is computed by the first of the pair and reused immediately by the second,
# and so that a run cut short still leaves ONE complete, readable within-attacker contrast.
run_arm configs/gptoss120b_hs_gemma27b_gptossbase_itermemo150.md \
        probes/hs_gemma27b_gptoss120b_gptossbase_itermemo150 \
        logs/run_hs_gemma27b_gptoss120b_gptossbase_itermemo150.log \
        "$GPTOSS_BASE"

run_arm configs/gptoss120b_hs_gemma27b_gptossbase_itermemo150_evaldesc.md \
        probes/hs_gemma27b_gptoss120b_gptossbase_evaldesc \
        logs/run_hs_gemma27b_gptoss120b_gptossbase_evaldesc.log \
        "$GPTOSS_BASE"

run_arm configs/deepseekv4pro_hs_gemma27b_dsbase_itermemo150.md \
        probes/hs_gemma27b_deepseekv4pro_dsbase_itermemo150 \
        logs/run_hs_gemma27b_deepseekv4pro_dsbase_itermemo150.log \
        "$DS_BASE"

run_arm configs/deepseekv4pro_hs_gemma27b_dsbase_itermemo150_evaldesc.md \
        probes/hs_gemma27b_deepseekv4pro_dsbase_evaldesc \
        logs/run_hs_gemma27b_deepseekv4pro_dsbase_evaldesc.log \
        "$DS_BASE"

echo ">>> $(date -Is)  all four arms finished."
