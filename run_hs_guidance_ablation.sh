#!/usr/bin/env bash
set -e

# deepseek-v4-pro attacker on the HIGH-STAKES concept (llama-1b probe) — contrastive
# concept-guidance ablation. Two arms, IDENTICAL in every knob except
# preprocessing.concept_description / preprocessing.label_guidance, run sequentially and
# fully isolated:
#
#   ARM A (noguidance): configs/deepseekv4pro_hs_llama1b_noguidance.md  no concept detail
#                       -> results_hs_llama1b_deepseekv4pro_noguidance/  probes/hs_llama1b_deepseekv4pro_noguidance
#   ARM B (guidance):   configs/deepseekv4pro_hs_llama1b_guidance.md    concept_description + label_guidance
#                       -> results_hs_llama1b_deepseekv4pro_guidance/    probes/hs_llama1b_deepseekv4pro_guidance
#
# The ablated knob affects ONLY the contrastive-generation prompt used on red-team
# successes before each retrain: arm A's generator sees just the two class-label strings,
# arm B's is additionally told what the high-stakes concept means and what each class looks
# like. Everything upstream (attacker deepseek/deepseek-v4-pro in prompt mode, judge
# openai/gpt-5.1, probe, base data, eval splits) is held fixed, so the arms diverge from
# iteration 0's RETRAIN onward — their red-team phases differ only by sampling noise.
#
# openai/gpt-5.1 is both the judge and the contrastive-pair generator (preprocessing.model)
# in both arms.
#
# ACTIVATIONS ARE COMPUTED FRESH, into a SHARED dir (results_hs_llama1b_deepseekv4pro/) that
# starts empty. Arm A computes the base + eval activation blobs; arm B hits them, because
# those blobs depend only on the probe model / layer / seed / base data / eval splits /
# transforms — NOT on the attacker or on the guidance knob. The redteam_acts_* per-conversation
# cache written into the same dir is content-keyed with a frozen LLM, so the two arms' distinct
# successes get distinct keys. Arm A therefore pays the full activation cost (~tens of minutes
# of GPU) and arm B mostly does not; budget wall-clock accordingly.
#
# Note the CONTRASTIVE cache is per-arm (it lives in --probe-out-dir) and, additionally, the
# guidance text is folded into its cache key — so arm B can never replay arm A's pairs.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, so contrastive pairs are regenerated per arm.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_hs_guidance_ablation.sh > logs/run_hs_guidance_ablation.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start the failsafe alongside it. It
# already defaults to these two arms, in this order, and hands itself off from arm A to arm B
# when arm A's comparison CSV lands:
#   git checkout -b failsafe/hs-guidance-ablation
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

SHARED_CACHE="results_hs_llama1b_deepseekv4pro"   # shared, guidance-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in \
    results_hs_llama1b_deepseekv4pro_noguidance results_hs_llama1b_deepseekv4pro_guidance \
    probes/hs_llama1b_deepseekv4pro_noguidance probes/hs_llama1b_deepseekv4pro_guidance ; do
    [ -e "$p" ] && { echo "ERROR: $p already exists — move it aside or bump the suffix." >&2; exit 1; }
done

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
echo ">>> activation cache: $SHARED_CACHE (starting empty — computed by arm A, reused by arm B)"

# --- run one arm ---------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile
    echo ">>> $(date -Is)  START $1  -> $2   (log: $3)"
    local rc=0
    .venv_claude/bin/python scripts/iterative_retrain.py "$1" \
        --iterations 3 \
        --base-training-data data/hs_ls_200.jsonl \
        --probe-out-dir "$2" \
        --eval --eval-dataset-dir eval_datasets \
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

run_arm configs/deepseekv4pro_hs_llama1b_noguidance.md probes/hs_llama1b_deepseekv4pro_noguidance logs/run_hs_llama1b_deepseekv4pro_noguidance.log
run_arm configs/deepseekv4pro_hs_llama1b_guidance.md   probes/hs_llama1b_deepseekv4pro_guidance   logs/run_hs_llama1b_deepseekv4pro_guidance.log

echo ">>> $(date -Is)  both arms finished."
