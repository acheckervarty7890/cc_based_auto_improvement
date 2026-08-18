#!/usr/bin/env bash
set -e

# BATCH-SUBMISSION attacker ablation on the HUMAN-HARM concept, attacking a 10-MEMBER DEEP
# ENSEMBLE over a google/gemma-3-27b-it (L32) probe, for 5 iterations, with the probe fit
# validated against a HELD-OUT DEV SET.
#
# This is run_gemma27b_hu_harm_attacker_ablation_batch_ens10.sh (experiment16_cloud) with
# exactly ONE knob moved:
#
#   validation.dev_data: ../dev_samples/hu_ha   (was unset ⇒ a --test-size slice of the
#                                                training data) — in BOTH configs
#
# Same two arms, same attacker/judge/preprocessing models, same base data, same eval splits,
# same batch_submissions setup, same ensemble, same --iterations 5, same per-iteration
# schedule, same order. So this run's comparison CSVs line up row-for-row with
# experiment16's, and the validation set is the only thing that moved.
#
# WHAT THE DEV SET DOES. tuberlens' fit early-stops on a validation set. With the default
# --test-size slice that set is carved out of the training data, so ~20% of EVERY iteration's
# red-team successes lands in it: the yardstick changes shape at every retrain, and iteration
# N's best-epoch checkpoint is chosen against different data than iteration N-1's. Pointing
# validation at dev_samples/hu_ha (290 rows, 145/145 balanced, the same four sources as the
# eval splits and verified DISJOINT from them) holds the yardstick fixed for the whole run.
#
# The other half of the trade, and it is not separable within this run: with a dev set,
# NOTHING is held out of the base data or the red-team successes — they train in full, so
# each retrain also sees ~20% more training samples than experiment16's did. experiment16 is
# the control for both effects at once. --test-size / --split-field are IGNORED here
# (retrain.py forces test_size to 0.0); they are not passed below.
#
# WHAT THE ENSEMBLE DOES HERE (held FIXED, same as experiment16). Every training and every
# retrain fits 10 probes of the same architecture on the SAME activations — member i under
# the repo-pinned ENSEMBLE_SEEDS[i] — and averages their PROBABILITIES into one score, so the
# attacker has to fool the mean of a 10-member committee rather than a single seed's
# boundary. Nothing downstream changes shape: probe_iter{N}.pkl holds an EnsembleProbe that
# duck-types tuberlens' Probe, so the threshold, the judge, the red-team loop and the eval all
# see one averaged score, and `success` is still probe-label vs. judge-label.
#
# COST. Only the FIT repeats — the split, the message transforms and the activation extraction
# are shared and computed once per retrain, so 10 members cost 10 probe-head fits over
# activations already in memory, NOT 10 passes through the 27B extraction LLM. The extraction
# work, which is what dominates wall-clock, is unchanged. The dev set adds ONE extra
# extraction: 290 conversations, cached as a single blob keyed on the dev files' bytes +
# model/layer/transforms (no seed, no test_size — it is never split), so it is paid once by
# arm 1 and reused by every later retrain of both arms.
#
# Relative to run_gemma27b_hs_attacker_ablation_batch.sh (experiment9_cloud) this is the same
# ablation with the concept swapped. What the concept swap changes, and nothing else does:
#   base training data   data/hu_harm_llama70b_50.jsonl   (was data/hs_ls_200.jsonl)
#   eval splits          eval_sets/hu_ha/       (was eval_datasets/)
#   probe labels         harmful_to_human / not_harmful_to_human
#   preprocessing.assistant_centric: true — harm is a property of the ASSISTANT's reply, so the
#     contrastive generator is told to edit the assistant turns, and the judge's edge-case line
#     says the same. Both arms carry it identically, so the attacker model is still the only
#     variable between them.
#
# Under batch_submissions each session makes ONE API call, is asked for all `max_turns` (5)
# candidate conversations in that single reply, has every one of them scored, and ends — the
# attacker never sees a probe/judge verdict. With view_limit: 0 it is not shown past attempts
# either. Volume is 10 sessions × 5 conversations × 5 rounds ≈ 250 attempts per error type per
# ITERATION, over 5 iterations — identical to experiment16.
#
# Two arms, IDENTICAL in every knob except attacker.models, run sequentially and fully
# isolated. NEITHER arm uses contrastive label guidance (no preprocessing.concept_description
# / label_guidance) — that is held off in both, so the attacker model is the only variable:
#
#   ARM 1 (gpt-oss-120b):    configs/gptoss120b_hu_harm_gemma27b_batch_ens10_devval.md
#                            -> results_hu_harm_gemma27b_gptoss120b_batch_ens10_devval/
#                               probes/hu_harm_gemma27b_gptoss120b_batch_ens10_devval
#   ARM 2 (deepseek-v4-pro): configs/deepseekv4pro_hu_harm_gemma27b_batch_ens10_devval.md
#                            -> results_hu_harm_gemma27b_deepseekv4pro_batch_ens10_devval/
#                               probes/hu_harm_gemma27b_deepseekv4pro_batch_ens10_devval
#
# The judge (openai/gpt-5.1), the preprocessing model (openai/gpt-5.1), the probe
# (gemma-3-27b-it L32), the base data (data/hu_harm_llama70b_50.jsonl) and every scheduling knob are
# held fixed, so any delta in the comparison CSVs is attributable to the attacker.
#
# ACTIVATIONS. The shared cache dir (results_hu_harm_gemma27b_batch_ablation/) starts empty on
# a clean cloud box; arm 1 fills it and arm 2 hits it, because those blobs depend only on the
# probe model / layer / seed / base data / eval splits / transforms — NOT on the attacker.
# The redteam_acts_* per-conversation cache written into the same dir is content-keyed with a
# frozen LLM, so the two arms' distinct successes get distinct keys.
#
# It is the SAME dir name the experiment11/16 arms use, deliberately: no cache key mentions
# ensemble_size (the ensemble varies only the probe-head fit, never an activation) or the
# iteration count, so a box that already ran those arms reuses their eval blobs and every
# red-team conversation activation verbatim. Sharing is safe, not merely tolerated.
#
# The one thing the dev set does move here: the BASE blob's key includes test_size, which
# validation.dev_data forces to 0.0, so this run writes a NEW base blob (the whole 50-sample
# base set on the train side, an empty val side) alongside experiment16's rather than reusing
# it. Nothing is invalidated — the old key still addresses the old blob — and it is 50
# samples, minutes. The dev set's own 290-row blob is a third key in the same dir, computed
# once by arm 1.
#
# The per-arm output and probe dirs are NOT shared — those carry successes found against a
# different (ensembled, dev-validated) probe.
#
# The EVAL half is not computed at all: both configs carry a `kaggle:` section pointing at
# anku7890/{slug}-gemmaevalpt, so arm 1's first eval downloads the ~4.6 GB of precomputed
# gemma-3-27b activations for the four hu_ha splits (validated against the probe's model/layer
# and each split's row count) straight into eval_activations/ instead of running full splits
# through a 27B model. That needs credentials — see the KAGGLE_CONFIG_DIR check below — AND it
# needs the four datasets to exist. They are NOT the ones the high-stakes runs use; publish
# them once from the local blobs with:
#
#   .venv_claude/bin/python scripts/publish_kaggle_eval_activations.py \
#       --source-dir archive/results3/gemma27_activations --dry-run     # inspect, then drop --dry-run
#
# The BASE split and the DEV set are still computed locally by arm 1, as is every red-team
# conversation it has not seen before.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hu_harm_attacker_ablation_batch_ens10_devval.sh > logs/run_gemma27b_hu_harm_attacker_ablation_batch_ens10_devval.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it —
# its built-in stage list already points at these two arms in this order.

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

SHARED_CACHE="results_hu_harm_gemma27b_batch_ablation"   # shared, arm-independent activation cache

# No clobber guard on the per-arm output/probe dirs ON PURPOSE: --resume is on by default and
# is how a wiped container picks a run back up, which requires the existing probe-out-dir and
# its markers. (The SHARED cache dir is likewise meant to persist and grow across both arms
# and across re-runs.) To start genuinely fresh, move the per-arm dirs aside first.

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (starting empty — computed by arm 1, reused by arm 2;"
echo ">>>   also key-compatible with the experiment11 single-probe arms, if this box has them)"

# --- run one arm ---------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    # --iterations 5, same as experiment16: five red-team → retrain → eval cycles.
    # NOT passing --ensemble-size or --dev-data here on purpose. Both flags OVERRIDE the
    # configs (precedence is flag > config), and both knobs are properties of the probe the
    # two arms share, so they live in the configs — probe.ensemble_size: 10 and
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

run_arm configs/gptoss120b_hu_harm_gemma27b_batch_ens10_devval.md    probes/hu_harm_gemma27b_gptoss120b_batch_ens10_devval    logs/run_hu_harm_gemma27b_gptoss120b_batch_ens10_devval.log
run_arm configs/deepseekv4pro_hu_harm_gemma27b_batch_ens10_devval.md probes/hu_harm_gemma27b_deepseekv4pro_batch_ens10_devval logs/run_hu_harm_gemma27b_deepseekv4pro_batch_ens10_devval.log

echo ">>> $(date -Is)  both arms finished."
