#!/usr/bin/env bash
set -e

# CROSS-ITERATION MEMO ablation on the HUMAN-HARM concept, attacking a 10-MEMBER DEEP
# ENSEMBLE over a google/gemma-3-27b-it (L32) probe, for 5 iterations, with the probe fit
# validated against a HELD-OUT DEV SET.
#
# TWO arms, both driven by openai/gpt-oss-120b — the same attacker model as the CONTROL,
# which is NOT run here because it already has been:
#
#                                       cross_iteration_memos   view_limit   memo budget
#   [control, experiment17_cloud]                       false            0             —
#   ARM 1  (memo only)                                   true            0           150
#   ARM 2  (memo + past attempts)                        true            8           150
#
# The control is configs/gptoss120b_hu_harm_gemma27b_batch_ens10_devval.md on
# experiment17_cloud, already run to results_hu_harm_gemma27b_gptoss120b_batch_ens10_devval/.
# Both arms here are that config with the memo knobs (and, in ARM 2, view_limit) changed and
# nothing else, so their comparison CSVs read against its row-for-row. experiment17's second
# arm (deepseek-v4-pro) is not part of this experiment — the attacker model is held fixed, it
# is no longer the variable.
#
# So control -> ARM 1 isolates the cross-iteration memo, and ARM 1 -> ARM 2 isolates
# re-opening the past-attempts channel on top of it.
#
# The memo word budget of 150 (repo default 900) is not a further variable — it is what makes
# the memo small enough to be worth injecting: at judge.max_tokens: 1024 a 900-word ask is
# guillotined mid-sentence, and the memo is fed back as the next iteration's prior_memo, so
# the truncation compounds. Full arithmetic in the ARM 1 config header. Both arms carry 150.
#
# WHY THIS ABLATION, AND WHY UNDER batch_submissions. Under batch_submissions each session
# makes ONE API call, is asked for all `max_turns` (5) candidate conversations in that single
# reply, has every one scored, and ends — so the attacker never sees a probe/judge verdict.
# With view_limit: 0 on top (the control and ARM 1) it is not shown past attempts either. The
# rolling ROUND memo is the only thing left, and a fresh SummaryStore is built per
# `run_redteam` call, i.e. per (iteration, error_type) — so it RESETS at every iteration
# boundary. Net effect in the control: nothing at all crosses from iteration N to iteration
# N+1. Iteration 4's attacker opens as blind as iteration 0's and spends its batch on ground
# the probe has already been retrained on. That is exactly the gap cross_iteration_memos
# closes, and the control measures the cost of leaving it open.
#
# ARM 2 then asks whether ABSTRACTED, judge-written guidance (the memo) is worth more, less,
# or the same as RAW EXAMPLES of what has already been tried — and whether the two compose or
# interfere. It is also a clone-rate comparison, not only an AUROC one: shown eight recent
# attempts, a session can talk itself into producing a ninth variant of them, which is what
# near_dup_guard rejects at submit time and what scripts/clone_rate.py measures offline.
#
# ONE CAVEAT ON ARM 2, unavoidable within this design: view_limit also gates
# _render_near_dup_rejects, so ARM 2 additionally shows the guard's rejected openers as an
# "avoid these" block. Both channels are governed by the single knob by construction.
#
# WHAT THE DEV SET DOES (identical in both arms and the control). tuberlens' fit early-stops on a
# validation set. With the default --test-size slice that set is carved out of the training
# data, so ~20% of EVERY iteration's red-team successes lands in it: the yardstick changes
# shape at every retrain, and iteration N's best-epoch checkpoint is chosen against different
# data than iteration N-1's. Pointing validation at dev_samples/hu_ha (290 rows, 145/145
# balanced, the same four sources as the eval splits and verified DISJOINT from them) holds
# the yardstick fixed for the whole run. The other half of the trade: NOTHING is held out of
# the base data or the red-team successes — they train in full. --test-size / --split-field
# are IGNORED here (retrain.py forces test_size to 0.0); they are not passed below.
#
# WHAT THE ENSEMBLE DOES (identical in both arms and the control). Every training and every retrain fits
# 10 probes of the same architecture on the SAME activations — member i under the repo-pinned
# ENSEMBLE_SEEDS[i] — and averages their PROBABILITIES into one score, so the attacker has to
# fool the mean of a 10-member committee rather than a single seed's boundary. Nothing
# downstream changes shape: probe_iter{N}.pkl holds an EnsembleProbe that duck-types
# tuberlens' Probe, so the threshold, the judge, the red-team loop and the eval all see one
# averaged score, and `success` is still probe-label vs. judge-label.
#
# COST. Only the FIT repeats — the split, the message transforms and the activation
# extraction are shared and computed once per retrain, so 10 members cost 10 probe-head fits
# over activations already in memory, NOT 10 passes through the 27B extraction LLM. The dev
# set adds ONE extra extraction: 290 conversations, cached as a single blob keyed on the dev
# files' bytes + model/layer/transforms (no seed, no test_size — it is never split) — so on a
# box that ran the control it is already there, and on a fresh box the first arm pays for it
# once and every later retrain of both arms reuses it.
#
# WHAT THE CONCEPT FORCES, and nothing else does:
#   base training data   data/hu_harm_llama70b_50.jsonl
#   eval splits          eval_sets/hu_ha/
#   probe labels         harmful_to_human / not_harmful_to_human
#   preprocessing.assistant_centric: true — harm is a property of the ASSISTANT's reply, so the
#     contrastive generator is told to edit the assistant turns, and the judge's edge-case line
#     says the same. Both arms carry it identically.
#
# VOLUME, identical in both arms and the control: 10 sessions × 5 conversations × 5 rounds
# ≈ 250 attempts per error type per ITERATION, over 5 iterations.
#
# The two arms run sequentially and fully isolated:
#
#   ARM 1  configs/gptoss120b_hu_harm_gemma27b_batch_ens10_devval_itermemo150.md
#          -> results_hu_harm_gemma27b_gptoss120b_itermemo150/
#             probes/hu_harm_gemma27b_gptoss120b_itermemo150
#   ARM 2  configs/gptoss120b_hu_harm_gemma27b_batch_ens10_devval_itermemo150_view8.md
#          -> results_hu_harm_gemma27b_gptoss120b_itermemo150_view8/
#             probes/hu_harm_gemma27b_gptoss120b_itermemo150_view8
#
# The attacker model (openai/gpt-oss-120b), the judge (openai/gpt-5.1), the preprocessing
# model (openai/gpt-5.1), the probe (gemma-3-27b-it L32), the base data
# (data/hu_harm_llama70b_50.jsonl) and every scheduling knob are held fixed, so any delta in
# the comparison CSVs — between the two arms, or against the control's CSV — is attributable
# to the memo and view_limit knobs.
#
# ACTIVATIONS. The shared cache dir (results_hu_harm_gemma27b_batch_ablation/) is the SAME dir
# the control wrote, so on a box that ran experiment17 it is already warm: those blobs depend
# only on the probe model / layer / seed / base data / eval splits / transforms — NOT on the
# attacker or on any knob this experiment moves. On a clean box it starts empty, ARM 1 fills it
# and ARM 2 hits it. The redteam_acts_* per-conversation cache written into the same dir is
# content-keyed with a frozen LLM, so the two arms' distinct successes get distinct keys, and
# any conversation both arms produce is computed once.
#
# No cache key mentions ensemble_size (the ensemble varies only the probe-head fit, never an
# activation), the memo knobs, view_limit or the iteration count, so a box that already ran the
# experiment11/16/17 arms reuses their eval blobs, their dev-set blob and every red-team
# conversation activation verbatim. Sharing is safe, not merely tolerated. In particular the
# BASE blob's key includes test_size, which validation.dev_data forces to 0.0 — the same value
# experiment17 used — so this run reuses experiment17's base blob rather than writing a new one.
# On such a box the ONLY new extraction either arm pays for is its own novel red-team
# conversations.
#
# The per-arm output and probe dirs are NOT shared — those carry successes found under a
# different steering regime.
#
# The EVAL half is not computed at all: both configs carry a `kaggle:` section pointing at
# anku7890/{slug}-gemmaevalpt, so the first eval on a cold box downloads the ~4.6 GB of precomputed
# gemma-3-27b activations for the four hu_ha splits (validated against the probe's model/layer
# and each split's row count) straight into eval_activations/ instead of running full splits
# through a 27B model. That needs credentials — see the KAGGLE_CONFIG_DIR check below — AND it
# needs the four datasets to exist. They are NOT the ones the high-stakes runs use; publish
# them once from the local blobs with:
#
#   .venv_claude/bin/python scripts/publish_kaggle_eval_activations.py \
#       --source-dir archive/results3/gemma27_activations --dry-run     # inspect, then drop --dry-run
#
# The BASE split and the DEV set are still computed locally by the first arm to need them (on a
# cold box), as is every red-team conversation neither arm has seen before.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hu_harm_itermemo_ablation.sh > logs/run_gemma27b_hu_harm_itermemo_ablation.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it,
# with its stage list pointed at these two arms in this order.

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
# its markers. (The SHARED cache dir is likewise meant to persist and grow across both arms and
# across re-runs — and across the control's run before them.) To start genuinely fresh, move
# the per-arm dirs aside first.

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (warm already if this box ran the experiment17 control;"
echo ">>>   otherwise filled by arm 1 and reused by arm 2. Key-compatible with the"
echo ">>>   experiment11/16/17 arms either way.)"

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

run_arm configs/gptoss120b_hu_harm_gemma27b_batch_ens10_devval_itermemo150.md \
        probes/hu_harm_gemma27b_gptoss120b_itermemo150 \
        logs/run_hu_harm_gemma27b_gptoss120b_itermemo150.log

run_arm configs/gptoss120b_hu_harm_gemma27b_batch_ens10_devval_itermemo150_view8.md \
        probes/hu_harm_gemma27b_gptoss120b_itermemo150_view8 \
        logs/run_hu_harm_gemma27b_gptoss120b_itermemo150_view8.log

echo ">>> $(date -Is)  both arms finished."
