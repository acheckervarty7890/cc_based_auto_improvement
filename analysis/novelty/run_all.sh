#!/usr/bin/env bash
# The whole novelty study, in order, on one GPU. Phases are resumable: pool.py and
# novelty.py skip work that is already on disk, ablate.py appends to its JSONL.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=.venv_claude/bin/python

# Score on the SAME path the run did. run_gemma27b_instructions_xmemocat_arms.sh exports
# PROBE_FUSED_ENSEMBLE=0, so both arms fit and scored their 10 members sequentially; this
# analysis runs in a fresh process where the setting would default back ON, and the fused
# path is a different reduction order. Left unpinned, sanity.py reports MISMATCH on all 12
# published probes at ~1e-4 and every condition below is scored on a path the run never
# took. (cloud_3's study predates the fused path, which is why it did not need this.)
export PROBE_FUSED_ENSEMBLE=0

# instructions only on this box: experiment_instruction_cloud_3's highstakes arms need
# results_hs_gemma27b_*/ and probes/hs_*, neither of which exists here — this branch's
# shared cache holds the instruction concept alone. The cross-attacker comparison the
# study leans on is unaffected: it needs two arms sharing ONE activation cache, and
# gptoss/nemotron are exactly that.
for EXP in instructions; do
    echo ">>> $(date -Is)  pool $EXP"
    $PY analysis/novelty/pool.py --experiment "$EXP"
    echo ">>> $(date -Is)  novelty $EXP"
    $PY analysis/novelty/novelty.py --experiment "$EXP"
    echo ">>> $(date -Is)  regions $EXP"
    $PY analysis/novelty/regions.py --experiment "$EXP"
    $PY analysis/novelty/regions.py --experiment "$EXP" --method kmeans -k 6
done

# Ablations last and one experiment at a time: both want the whole card.
for EXP in instructions; do
    echo ">>> $(date -Is)  ablate $EXP"
    $PY analysis/novelty/ablate.py --experiment "$EXP"
done

$PY analysis/novelty/report.py
echo ">>> $(date -Is)  done"
