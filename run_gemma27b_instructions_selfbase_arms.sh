#!/usr/bin/env bash
set -e

# SELF-GENERATED-BASE experiment on the INSTRUCTION-FOLLOWING concept, attacking a 10-MEMBER
# DEEP ENSEMBLE over a google/gemma-3-27b-it (L32) probe, for TEN iterations, with the probe
# fit validated against a HELD-OUT DEV SET (dev_samples/instructions).
#
# FOUR ARMS, TWO ATTACKERS, ONE BRANCH. This is experiment25 (gpt-oss base) and experiment26
# (deepseek base) re-run on the instruction-following concept, with both attackers' pairs kept
# together instead of one attacker per branch:
#
#   arm  attacker                            base data                       eval.data_description
#   ---  ----------------------------------  ------------------------------  ---------------------
#    1   nvidia/nemotron-3-ultra-550b-a55b   instructions_nemotron_50.jsonl  unset
#    2   nvidia/nemotron-3-ultra-550b-a55b   instructions_nemotron_50.jsonl  six data kinds
#    3   openai/gpt-oss-120b                 instructions_gptoss_50.jsonl    unset
#    4   openai/gpt-oss-120b                 instructions_gptoss_50.jsonl    six data kinds
#
#   arm 1 -> arm 2   what does telling the memo-writer which KINDS of conversation the probe is
#   arm 3 -> arm 4   scored on buy, on top of carrying a hand-off memo across the iteration
#                    boundary? Measured once per attacker, so the answer can be checked for
#                    replication instead of read off a single rotation.
#
# THE BASE DATA MOVES WITH THE ATTACKER, to preserve the self-generated-base property: the
# model that wrote the initial probe's training data is the one attacking it. Both 50-row sets
# are 25 assistant_follows_the_instruction / 25 assistant_does_not_follow_the_instruction, same
# schema, both from generator_experiment_1.
#
# CONSEQUENCE FOR READING THE RESULTS. Comparable: 1 vs 2, and 3 vs 4 — within a pair the two
# configs differ by exactly one key, probe.description is byte-identical, so the judge labels
# the same way and success rate, clone rate, red-team labels and eval CSVs are all comparable.
# NOT comparable as absolute numbers: 1 vs 3, or 2 vs 4 — a different base means a different
# initial probe, hence different absolute AUROCs and success rates throughout. What IS
# comparable across the attackers is the SHAPE of each pair's arm1->arm2 gap.
#
# ONE KNOB IS PINNED AGAINST ITS REPO DEFAULT, in all four arms: `judge.eval_scope_check:
# false`. The default is TRUE, and with an eval.data_description set it makes that description
# reach the judge's CLASSIFICATION prompt as a constraint — the judge then also rules on whether
# a candidate is the kind of conversation the probe is scored on, and out-of-scope candidates are
# refused as successes. Left on, arms 2 and 4 would differ from their siblings in the labelling
# function and in what counts as a success, not only in what the memos say, and neither pair
# would be a one-key contrast any more. Off, the description reaches the two SUMMARIZERS ONLY,
# which is the experiment25/26 design this run reproduces. Verified by rendering the judge's
# classification prompt for arms 1 and 2 and diffing it: byte-identical.
#
# ARM ORDER IS DELIBERATE: the two nemotron arms run first and the two gpt-oss arms second, so
# a box that dies half way still leaves ONE COMPLETE PAIR — i.e. one usable contrast — rather
# than two half-finished ones.
#
# THE EVAL AND DEV SETS ARE SIX SPLITS, NOT SEVEN. `oig_omission` is removed from BOTH
# eval_sets/instructions (1302 -> 1188 rows) and dev_samples/instructions (436 -> 404) on this
# branch, and from the eval-data description arms 2 and 4 carry. So:
#   - no comparison CSV here has an oig_omission column, and its mean over splits is a mean over
#     SIX — not comparable to experiment_instruction_cloud_4..7's mean over seven. Compare per
#     split, or re-average theirs over the same six.
#   - the DEV set is a different set, so the fits early-stop against different data and these
#     are different probes, not merely differently scored ones.
#   - the dev activation blob is keyed on a content hash of the dev JSONLs, so the six-split dev
#     set gets its own key and an existing 436-row blob can never be served in its place. The
#     per-split eval blobs are path-keyed and the removed one is simply never requested.
#
# THE SCHEDULE, identical in all four arms and carried from experiment25/26:
#
#     rounds: 5                 4 round-memo updates per error type per iteration
#     sessions_per_model: 3
#     max_turns: 5              batch size
#     concurrency: 3            >= sessions_per_model x len(models)
#     batch_target: 30          inert: a round can produce at most 3 x 5 = 15 attempts
#
# NOTE this is the experiment25/26 schedule, NOT experiment_instruction_cloud_4..7's
# (sessions_per_model: 10, five iterations). Ten iterations means NINE boundaries for the
# cross-iteration memo to cross, which is what the experiment is about; the narrower per-round
# fan-out is what pays for them. A consequence worth knowing before comparing yields against
# those runs: this schedule produces ~1500 attempts per arm against their ~2500.
#
# VOLUME per arm: 15 attempts/round x 5 rounds = 75 per error type per iteration, x2 error
# types = 150/iteration, x10 iterations = ~1500 attempts. Four arms => ~6000 total, each scored
# by a gemma-3-27b forward pass. With the eval and dev activations served from Kaggle that
# scoring is essentially the only thing loading the 27B model, so it dominates wall clock.
#
# ARMS AND WHERE THEY WRITE:
#
#   ARM 1  configs/nemotron_instructions_gemma27b_nmbase_itermemo150.md
#          -> results_instructions_gemma27b_nemotron_nmbase_itermemo150/
#             probes/instructions_gemma27b_nemotron_nmbase_itermemo150
#   ARM 2  configs/nemotron_instructions_gemma27b_nmbase_evaldesc.md
#          -> results_instructions_gemma27b_nemotron_nmbase_evaldesc/
#             probes/instructions_gemma27b_nemotron_nmbase_evaldesc
#   ARM 3  configs/gptoss120b_instructions_gemma27b_gobase_itermemo150.md
#          -> results_instructions_gemma27b_gptoss_gobase_itermemo150/
#             probes/instructions_gemma27b_gptoss_gobase_itermemo150
#   ARM 4  configs/gptoss120b_instructions_gemma27b_gobase_evaldesc.md
#          -> results_instructions_gemma27b_gptoss_gobase_evaldesc/
#             probes/instructions_gemma27b_gptoss_gobase_evaldesc
#
# ACTIVATIONS. The shared cache dir (results_instructions_gemma27b_shared/) is the one
# experiment_instruction_cloud_1/_3/_4/_5/_6/_7 wrote. No cache key mentions the memo knobs,
# the eval-data description, view_limit, sessions_per_model, ensemble_size or the iteration
# count, so on a box that ran any of those the EVAL blobs are already
# warm; on a clean box they are FETCHED FROM KAGGLE (see each config's `kaggle:` section), not
# recomputed. The BASE blob is keyed on a hash of the base data file, so each attacker's base
# gets its OWN key: arm 1 computes the nemotron one and arm 2 reuses it, arm 3 computes the
# gpt-oss one and arm 4 reuses it. Nothing existing is invalidated. The per-arm output and
# probe dirs are NOT shared.
#
# THE ARMS MUST RUN SEQUENTIALLY, which is why this script does not background them: two live
# writers into the shared activation cache can tear a blob.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_instructions_selfbase_arms.sh > logs/run_selfbase_arms.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it —
# its built-in stage list already points at these four arms in this order:
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

# The two base files are this experiment's variable — check them BEFORE the hours-long first
# arm rather than letting train_initial_probe discover one missing at arm 3.
NEMOTRON_BASE="data/instructions_nemotron_50.jsonl"
GPTOSS_BASE="data/instructions_gptoss_50.jsonl"
for base in "$NEMOTRON_BASE" "$GPTOSS_BASE"; do
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
# Defaults to 0 = SEQUENTIAL, as in every earlier instruction experiment, so this run's probes
# are fit on the same path those runs' were (one ProbeFactory.build per seed, one predict_proba
# per member). That costs wall-clock: the fused path stacks the 10 members and steps them under
# vmap (measured 3.8x on a comparable shape), and this run does 40 retrains.
#
# Export PROBE_FUSED_ENSEMBLE=1 before launching to take that speedup. It is safe WITHIN this
# experiment — one switch governs both the fit and the scoring path (ensemble.fusion_enabled),
# so all four arms move together and stay comparable to each other. What it costs is exact
# comparability of the probes themselves with experiment_instruction_cloud_4..7, which ran
# sequential: the fused path changes the floating-point reduction order, which is a 4th-decimal
# effect on AUROC (no prediction flipped when it was measured) but not bit-identity.
export PROBE_FUSED_ENSEMBLE="${PROBE_FUSED_ENSEMBLE:-0}"
if [ "$PROBE_FUSED_ENSEMBLE" = "0" ]; then
    echo ">>> ensemble: SEQUENTIAL fit and scoring (PROBE_FUSED_ENSEMBLE=0 — matches the earlier instruction runs, slower)"
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
echo ">>>   experiment_instruction_cloud_1/_3/_4/_5/_6/_7; otherwise fetched from Kaggle."
echo ">>>   The two 50-row base blobs are new: arm 1 computes one, arm 3 the other.)"

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

run_arm configs/nemotron_instructions_gemma27b_nmbase_itermemo150.md \
        probes/instructions_gemma27b_nemotron_nmbase_itermemo150 \
        "$NEMOTRON_BASE" \
        logs/run_instructions_gemma27b_nemotron_nmbase_itermemo150.log

run_arm configs/nemotron_instructions_gemma27b_nmbase_evaldesc.md \
        probes/instructions_gemma27b_nemotron_nmbase_evaldesc \
        "$NEMOTRON_BASE" \
        logs/run_instructions_gemma27b_nemotron_nmbase_evaldesc.log

run_arm configs/gptoss120b_instructions_gemma27b_gobase_itermemo150.md \
        probes/instructions_gemma27b_gptoss_gobase_itermemo150 \
        "$GPTOSS_BASE" \
        logs/run_instructions_gemma27b_gptoss_gobase_itermemo150.log

run_arm configs/gptoss120b_instructions_gemma27b_gobase_evaldesc.md \
        probes/instructions_gemma27b_gptoss_gobase_evaldesc \
        "$GPTOSS_BASE" \
        logs/run_instructions_gemma27b_gptoss_gobase_evaldesc.log

echo ">>> $(date -Is)  all four arms finished."
