#!/usr/bin/env bash
# Regenerate every report this phase produces, then commit and push them.
#
# Stages an EXPLICIT path list, never `git add -A`: the working tree holds live
# credentials (hf_token.txt, kaggle/kaggle.json, close_this.sh) and ~100 GB of
# activations, caches and probe pickles that must never reach the remote.
set -uo pipefail
cd /workspace/cc_based_auto_improvement || exit 1
PY=.venv_claude/bin/python

"$PY" scripts/concept_probes_report.py  >/dev/null 2>&1
"$PY" scripts/ceiling_report.py         >/dev/null 2>&1

PATHS=(
  scripts/concept_probes.py
  scripts/concept_probes_report.py
  scripts/cross_concept_ceiling.py
  scripts/ceiling_report.py
  scripts/publish_experiments_progress.sh
  reports/concept_probes_summary.md
)
for d in reports/llama8b_concept_probes reports/llama70b_concept_probes \
         reports/dsv4pro_concept_probes reports/cross_concept_ceiling; do
  [ -d "$d" ] && PATHS+=("$d")
done

git add -- "${PATHS[@]}" 2>/dev/null
if git diff --cached --quiet; then
  echo "[publish] nothing new to commit"
else
  n8=$(find results_llama8b   -name '*.pkl' 2>/dev/null | wc -l)
  n70=$(find results_llama70b -name '*.pkl' 2>/dev/null | wc -l)
  nds=$(find results_dsv4pro  -name '*.pkl' 2>/dev/null | wc -l)
  narm=$(tail -n +2 results_ceiling/cross_concept_ceiling.csv 2>/dev/null | cut -d, -f1 | sort -u | wc -l)
  git commit -q -m "chore(experiments): progress — llama8b ${n8}/12, llama70b ${n70}/12, dsv4pro ${nds}/12 probes, ${narm} ceiling arms

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
    && git push -q origin generalization_tests && echo "[publish] pushed $(git rev-parse --short HEAD)"
fi

echo "[health] $(date -u +%H:%M:%SZ) gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ') mem_avail=$(awk '/MemAvailable/{print int($2/1024)"MiB"}' /proc/meminfo) disk=$(df -h /workspace | awk 'NR==2{print $4}')"
pgrep -f "concept_probes.py|cross_concept_ceiling.py" >/dev/null \
  && echo "[health] run active" || echo "[health] no run process"
