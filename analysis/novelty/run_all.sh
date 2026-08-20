#!/usr/bin/env bash
# The whole novelty study, in order, on one GPU. Phases are resumable: pool.py and
# novelty.py skip work that is already on disk, ablate.py appends to its JSONL.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_claude/bin/python

for EXP in instructions highstakes; do
    echo ">>> $(date -Is)  pool $EXP"
    $PY analysis/novelty/pool.py --experiment "$EXP"
    echo ">>> $(date -Is)  novelty $EXP"
    $PY analysis/novelty/novelty.py --experiment "$EXP"
    echo ">>> $(date -Is)  regions $EXP"
    $PY analysis/novelty/regions.py --experiment "$EXP"
    $PY analysis/novelty/regions.py --experiment "$EXP" --method kmeans -k 6
done

# Ablations last and one experiment at a time: both want the whole card.
for EXP in instructions highstakes; do
    echo ">>> $(date -Is)  ablate $EXP"
    $PY analysis/novelty/ablate.py --experiment "$EXP"
done

$PY analysis/novelty/report.py
echo ">>> $(date -Is)  done"
