#!/usr/bin/env bash
# Snapshot the ceiling analysis to the branch. Adds ONLY analysis/ceiling and the
# run log — never the untracked secrets/caches sitting in the worktree
# (hf_token.txt, kaggle/, acts_new/, probes/hs_*).
set -uo pipefail
cd "$(dirname "$0")/../.."
git add -f analysis/ceiling/*.py analysis/ceiling/*.sh \
          analysis/ceiling/results/*.jsonl analysis/ceiling/results/*.md \
          analysis/ceiling/results/*.log logs/ceiling_analysis.log 2>/dev/null
if git diff --cached --quiet; then
  echo "nothing new to commit"
  exit 0
fi
git commit -q -m "ceiling analysis: progress snapshot @ $(date -u +%FT%TZ)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push -q origin HEAD && echo "pushed $(git rev-parse --short HEAD)"
