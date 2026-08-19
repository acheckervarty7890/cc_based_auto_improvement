#!/usr/bin/env bash
# Append a status snapshot to ceiling_analysis/logs/PROGRESS.md, then commit and push it.
# Run under flock so it can never race the interactive commits happening in the same repo.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$(pwd)"
LOG=ceiling_analysis/logs/PROGRESS.md
mkdir -p ceiling_analysis/logs

{
  echo ""
  echo "## $(date -Is)"
  echo ""
  echo '```'
  echo "gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
  echo "acts on disk: $(du -sh ceiling_acts 2>/dev/null | cut -f1)"
  for c in highstakes hu_ha; do
    n=$(ls ceiling_acts/$c/redteam_cache/*/ 2>/dev/null | wc -l)
    echo "$c: $n red-team/base conversations extracted"
    for f in ceiling_analysis/results/ceiling_$c.jsonl ceiling_analysis/results/sweep_$c.jsonl; do
      [ -f "$f" ] && echo "$c: $(wc -l < "$f") rows in $(basename "$f")"
    done
  done
  for l in ceiling_analysis/logs/*.log; do
    [ -f "$l" ] || continue
    echo "--- $(basename "$l") ---"
    tail -c 400 "$l" | tr -d '\r' | tail -3
  done
  echo '```'
} >> "$LOG"

git add -A ceiling_analysis ceiling_acts 2>/dev/null
git add -A ceiling_analysis
if ! git diff --cached --quiet; then
  git commit -q -m "ceiling analysis: progress snapshot $(date -Is)"
fi
git push -q origin ceiling_analysis 2>&1 | tail -2
