#!/usr/bin/env bash
# The ceiling analysis for experiment22's two dataset-description arms.
#
# Differences from ceiling_analysis/scripts/run_all.sh, which targets the upstream
# `hu_ha` / `highstakes` concepts:
#   * no Kaggle fetch and no gemma download wait — prep_local_activations.py already built
#     ceiling_acts/hu_ha/ out of this box's own run caches;
#   * the concepts are the two experiment22 arms, which SHARE acts_name="hu_ha" and differ
#     only in their red-team set;
#   * verification runs against one of those arms rather than the absent "hu_ha".
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export AGENTIC_REDTEAM_MAX_MEMORY="0=22GiB,cpu=45GiB" MAX_MEMORY="0=22GiB,cpu=45GiB"
# Single probes here, so this cannot change a number — pinned anyway so the analysis and the
# run it explains sit on the same scoring path.
export PROBE_FUSED_ENSEMBLE=0

PY=./.venv_claude/bin/python
LOGS=ceiling_analysis/logs
mkdir -p "$LOGS" ceiling_analysis/results
A1=hu_ha_dd_gptoss120b
A2=hu_ha_dd_deepseekv4pro

mark() { echo ">>> $(date -Is)  $1  $2" | tee -a "$LOGS/run_all.log"; }
run() {
  local name="$1"; shift
  mark START "$name"
  if "$@" > "$LOGS/$name.log" 2>&1; then mark DONE "$name"
  else mark "FAILED(rc=$?)" "$name"; mark ABORT chain; exit 1; fi
}

# ---------------------------------------------------------------- 1. activation prep
run prep_$A1 $PY ceiling_analysis/scripts/prep_local_activations.py --concept $A1
run prep_$A2 $PY ceiling_analysis/scripts/prep_local_activations.py --concept $A2

# ---------------------------------------------------------------- 2. verification
# Worth running here rather than trusting the upstream numbers: this cache has MIXED
# provenance. Arm 2's blobs were written by the experiment run itself; arm 1's were
# extracted by this analysis after the box died. Both must equal a fresh single-row
# extraction, or the two arms are not on the same scale.
run verify_batch_padding    $PY ceiling_analysis/scripts/verify_batch_padding.py --concept $A2
run verify_extraction_noise $PY ceiling_analysis/scripts/verify_extraction_noise.py --concept $A2
run verify_fast_fit         $PY ceiling_analysis/scripts/verify_fast_fit.py --concept $A2

# ---------------------------------------------------------------- 3. ceiling + sweep
# The ceiling never touches red-team data (fit inside the eval splits, early-stopped on the
# reserved dev slice), so the two arms' ceilings are the same computation under the same
# seeds. Both are run anyway: make_report reads one JSON per concept, and agreement between
# them is a free determinism check.
run ceiling_$A1 $PY ceiling_analysis/scripts/run_ceiling.py --concepts $A1 \
    --train-sizes 173 346 693 --add-dev-pool
run sweep_$A1   $PY ceiling_analysis/scripts/run_sweep.py --concepts $A1

run ceiling_$A2 $PY ceiling_analysis/scripts/run_ceiling.py --concepts $A2 \
    --train-sizes 173 346 693 --add-dev-pool
run sweep_$A2   $PY ceiling_analysis/scripts/run_sweep.py --concepts $A2

# ---------------------------------------------------------------- 4. write-up
run make_report $PY ceiling_analysis/scripts/make_report.py --concepts $A1 $A2
mark ALLDONE chain
