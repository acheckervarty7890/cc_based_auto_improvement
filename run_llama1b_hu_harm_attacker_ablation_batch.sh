#!/usr/bin/env bash
set -e

# BATCH-SUBMISSION attacker ablation on the HARMFUL_TO_HUMAN concept with a
# meta-llama/Llama-3.2-1B-Instruct (L8) probe, base data data/hu_harm_llama70b_50.jsonl.
#
# This is experiment9_cloud's batch setup (run_gemma27b_hs_attacker_ablation_batch.sh) moved
# to the harmful_to_human concept and the llama-1b probe. Both arms carry:
#
#   attacker.batch_submissions: true    (one blind API call per session)
#   attacker.view_limit:        0       (no past-attempts injection either)
#   attacker.capture_prompts:   false   (default, pinned)
#   attacker.cross_iteration_memos: false (default, pinned)
#   judge.hide_opposite_direction: true (default, pinned)
#
# Under batch_submissions each session makes ONE API call, is asked for all `max_turns` (5)
# candidate conversations in that single reply, has every one of them scored, and ends — the
# attacker never sees a probe/judge verdict, and with view_limit: 0 it is shown no past
# attempts either. Its only inputs are the system prompt (probe metadata + the judge's rolling
# round memo) and "submit all N now". Attempt volume is unchanged from every earlier hu_harm
# run: 10 sessions × 5 conversations × 5 rounds ≈ 250 attempts per error type per iteration.
#
# Two arms, IDENTICAL in every knob except attacker.models, run sequentially and fully
# isolated. NEITHER arm uses contrastive label guidance (no preprocessing.concept_description
# / label_guidance) — that is held off in both, so the attacker model is the only variable:
#
#   ARM 1 (gpt-oss-120b):    configs/gptoss120b_hu_harm_llama1b_batch.md
#                            -> results_hu_harm_llama70b50_gptoss120b_batch/
#                               probes/hu_harm_llama1b_gptoss120b_batch
#   ARM 2 (deepseek-v4-pro): configs/deepseekv4pro_hu_harm_llama1b_batch.md
#                            -> results_hu_harm_llama70b50_deepseekv4pro_batch/
#                               probes/hu_harm_llama1b_deepseekv4pro_batch
#
# The judge (openai/gpt-5.1), the preprocessing model (openai/gpt-5.1), the probe
# (Llama-3.2-1B-Instruct L8), the base data and every scheduling knob are held fixed, so any
# delta in the comparison CSVs is attributable to the attacker.
#
# ACTIVATIONS ARE COMPUTED FRESH — unlike the gemma-27b runs there is no `kaggle:` prefetch,
# because llama-1b eval activations are cheap to extract locally. The shared cache dir
# (results_hu_harm_llama70b50_batch_ablation/) starts empty on a clean cloud box; arm 1 fills
# both the base and eval halves and arm 2 hits them, because those blobs depend only on the
# probe model / layer / seed / base data / eval splits / transforms — NOT on the attacker. The
# redteam_acts_* per-conversation cache written into the same dir is content-keyed against a
# frozen LLM, so the two arms' distinct successes get distinct keys. Budget arm 1's wall-clock
# accordingly; arm 2 is cheaper.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_llama1b_hu_harm_attacker_ablation_batch.sh > logs/run_llama1b_hu_harm_attacker_ablation_batch.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it. It
# already defaults to these two arms, in this order, and hands itself off from arm 1 to arm 2
# when arm 1's comparison CSV lands:
#   git checkout -b failsafe/llama1b-hu-harm-attacker-ablation-batch
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

SHARED_CACHE="results_hu_harm_llama70b50_batch_ablation"   # shared, arm-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in \
    results_hu_harm_llama70b50_gptoss120b_batch results_hu_harm_llama70b50_deepseekv4pro_batch \
    probes/hu_harm_llama1b_gptoss120b_batch probes/hu_harm_llama1b_deepseekv4pro_batch ; do
    [ -e "$p" ] && { echo "ERROR: $p already exists — move it aside or bump the suffix." >&2; exit 1; }
done

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (starting empty — computed by arm 1, reused by arm 2)"

# --- run one arm ---------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 3 \
        --base-training-data data/hu_harm_llama70b_50.jsonl \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_dataset_hu_ha \
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

run_arm configs/gptoss120b_hu_harm_llama1b_batch.md    probes/hu_harm_llama1b_gptoss120b_batch    logs/run_hu_harm_llama1b_gptoss120b_batch.log
run_arm configs/deepseekv4pro_hu_harm_llama1b_batch.md probes/hu_harm_llama1b_deepseekv4pro_batch logs/run_hu_harm_llama1b_deepseekv4pro_batch.log

echo ">>> $(date -Is)  both arms finished."
