#!/usr/bin/env bash
set -e

# Attacker ablation on the HIGH-STAKES concept with a google/gemma-3-27b-it (L32) probe.
# Two arms, IDENTICAL in every knob except attacker.models, run sequentially and fully
# isolated. NEITHER arm uses contrastive label guidance (no preprocessing.concept_description
# / label_guidance) — that is held off in both, so the attacker model is the only variable:
#
#   ARM 1 (gpt-oss-120b):   configs/gptoss120b_hs_gemma27b_noguidance.md
#                           -> results_hs_gemma27b_gptoss120b_noguidance/
#                              probes/hs_gemma27b_gptoss120b_noguidance
#   ARM 2 (deepseek-v4-pro): configs/deepseekv4pro_hs_gemma27b_noguidance.md
#                           -> results_hs_gemma27b_deepseekv4pro_noguidance/
#                              probes/hs_gemma27b_deepseekv4pro_noguidance
#
# The judge (openai/gpt-5.1), the preprocessing model (openai/gpt-5.1), the probe
# (gemma-3-27b-it L32), the base data (data/hs_ls_200.jsonl) and every scheduling knob are
# held fixed, so any delta in the comparison CSVs is attributable to the attacker.
#
# ACTIVATIONS ARE COMPUTED FRESH. Meant for a clean cloud box, so there is NO pre-seeding of
# the activation cache from archive/. The shared cache dir
# (results_hs_gemma27b_attacker_ablation/) starts empty: arm 1 computes the base + eval
# activation blobs, arm 2 hits them, because those blobs depend only on the probe model /
# layer / seed / base data / eval splits / transforms — NOT on the attacker. The
# redteam_acts_* per-conversation cache written into the same dir is content-keyed with a
# frozen LLM, so the two arms' distinct successes get distinct keys.
#
# Budget note: this is a 27B probe, not the 1B one the earlier hs ablations used. Arm 1 pays
# the full activation cost over eval_datasets (~1900 rows, full splits) plus the base split;
# arm 2 mostly does not. Expect arm 1 to be substantially the slower of the two.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_gemma27b_hs_attacker_ablation.sh > logs/run_gemma27b_hs_attacker_ablation.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it,
# pointed at these two arms in this order.

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

SHARED_CACHE="results_hs_gemma27b_attacker_ablation"   # shared, arm-independent activation cache

# Refuse to clobber per-arm outputs / probes. (The SHARED cache dir is intentionally NOT in
# this list — it is meant to persist and grow across both arms and across re-runs.)
for p in \
    results_hs_gemma27b_gptoss120b_noguidance results_hs_gemma27b_deepseekv4pro_noguidance \
    probes/hs_gemma27b_gptoss120b_noguidance probes/hs_gemma27b_deepseekv4pro_noguidance ; do
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

run_arm configs/gptoss120b_hs_gemma27b_noguidance.md    probes/hs_gemma27b_gptoss120b_noguidance    logs/run_hs_gemma27b_gptoss120b_noguidance.log
run_arm configs/deepseekv4pro_hs_gemma27b_noguidance.md probes/hs_gemma27b_deepseekv4pro_noguidance logs/run_hs_gemma27b_deepseekv4pro_noguidance.log

echo ">>> $(date -Is)  both arms finished."
