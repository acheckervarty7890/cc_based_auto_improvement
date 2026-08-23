#!/usr/bin/env bash
# Probe-design study: what does a normalization step in front of LinearThenSoftmax do to
# the ceiling?
#
# One arm only (experiment22 arm 1, gpt-oss-120b). The ceiling never touches red-team data —
# it cross-validates inside the eval splits and early-stops on the reserved dev slice — so
# the arm choice affects nothing but which Concept entry supplies the eval/dev blobs, and
# running the second would re-measure the same computation. See ceiling_analysis/README.md.
#
# `--norm none` is not redundant with the existing ceiling_hu_ha_dd_gptoss120b.json: it runs
# the SAME numbers through the new subclass, so a mismatch would mean the plumbing moved.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export AGENTIC_REDTEAM_MAX_MEMORY="0=22GiB,cpu=45GiB" MAX_MEMORY="0=22GiB,cpu=45GiB"
export PROBE_FUSED_ENSEMBLE=0

PY=./.venv_claude/bin/python
LOGS=ceiling_analysis/logs
ARM=hu_ha_dd_gptoss120b
mkdir -p "$LOGS"

mark() { echo ">>> $(date -Is)  $1  $2" | tee -a "$LOGS/run_norm.log"; }

for NORM in none layernorm_noaffine layernorm rmsnorm standardize; do
  mark START "ceiling_norm_$NORM"
  if $PY ceiling_analysis/scripts/run_ceiling.py --concepts $ARM \
       --train-sizes 173 346 693 --add-dev-pool --norm "$NORM" \
       > "$LOGS/ceiling_${ARM}__norm-${NORM}.log" 2>&1
  then mark DONE "ceiling_norm_$NORM"
  else mark "FAILED(rc=$?)" "ceiling_norm_$NORM"; mark ABORT chain; exit 1; fi
done
mark ALLDONE chain
