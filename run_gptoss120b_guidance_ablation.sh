#!/usr/bin/env bash
set -e

# gpt-oss-120b attacker on the HIGH-STAKES / llama-1b probe — contrastive label-guidance
# ablation. Two arms, IDENTICAL in every knob except preprocessing.concept_description +
# preprocessing.label_guidance, run sequentially and fully isolated:
#
#   ARM A (noguidance): configs/gptoss120b_hs_llama1b_noguidance.md   no concept detail — the
#                       contrastive generator sees only the raw label strings
#                       -> results_hs_llama1b_gptoss120b_noguidance/  probes/hs_llama1b_gptoss120b_noguidance
#   ARM B (guidance):   configs/gptoss120b_hs_llama1b_guidance.md     concept_description +
#                       per-label guidance injected into the generation prompt
#                       -> results_hs_llama1b_gptoss120b_guidance/    probes/hs_llama1b_gptoss120b_guidance
#
# The knob only touches PREPROCESSING (the contrastive-pair generator used before each
# retrain). The attacker (openai/gpt-oss-120b), the judge (openai/gpt-5.1-chat), the probe,
# the base data and every scheduling knob are held fixed, so any delta in the comparison CSVs
# is attributable to the quality of the generated contrastive pairs.
#
# ACTIVATIONS ARE COMPUTED FRESH. Meant for a clean cloud box, so there is NO symlink
# pre-seeding of the activation cache — the shared cache dir
# (results_hs_llama1b_gptoss120b_guidance_ablation/) starts empty. Arm A computes the base +
# eval activation blobs; arm B hits them, because those blobs depend only on the probe model /
# layer / seed / base data / eval splits / transforms — NOT on the attacker and NOT on the
# contrastive prompt. The redteam_acts_* per-conversation cache written into the same dir is
# content-keyed with a frozen LLM, so the two arms' distinct successes (and distinct
# guidance-generated pairs) get distinct keys. Arm A therefore pays the full activation cost
# and arm B mostly does not; budget wall-clock accordingly.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl. This is load-bearing here: guidance is folded
#     into the contrastive cache key (_guidance_fingerprint), so a shared cache could not serve
#     arm B arm A's pairs — but a shared cache would still blur the two arms' provenance.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gptoss120b_guidance_ablation.sh > logs/run_gptoss120b_guidance_ablation.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it,
# pointed at these two arms in this order.

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

SHARED_CACHE="results_hs_llama1b_gptoss120b_guidance_ablation"   # shared, arm-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in \
    results_hs_llama1b_gptoss120b_noguidance results_hs_llama1b_gptoss120b_guidance \
    probes/hs_llama1b_gptoss120b_noguidance probes/hs_llama1b_gptoss120b_guidance ; do
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

run_arm configs/gptoss120b_hs_llama1b_noguidance.md probes/hs_llama1b_gptoss120b_noguidance logs/run_hs_llama1b_gptoss120b_noguidance.log
run_arm configs/gptoss120b_hs_llama1b_guidance.md   probes/hs_llama1b_gptoss120b_guidance   logs/run_hs_llama1b_gptoss120b_guidance.log

echo ">>> $(date -Is)  both arms finished."
