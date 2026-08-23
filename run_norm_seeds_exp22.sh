#!/usr/bin/env bash
# How big is a difference worth believing? The normalization study's top-rung gaps are
# +0.0005 to +0.0014, and every number in it comes from ONE fit seed. This re-fits the
# same folds, the same training rows and the same validation slice under three more head
# seeds, for the unnormalized baseline and for the best variant, so the comparison can be
# read against its own run-to-run spread instead of against nothing.
#
# `--seed` is deliberately NOT varied: that would move the folds and the training-size
# subsample, i.e. change the data, which is a different question. `--fit-seed` moves only
# the head's init and the DataLoader's batch order.
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

mark() { echo ">>> $(date -Is)  $1  $2" | tee -a "$LOGS/run_norm_seeds.log"; }

for SEED in 7 1234 20260823; do
  for NORM in none layernorm; do
    NAME="${NORM}_fit${SEED}"
    mark START "$NAME"
    if $PY ceiling_analysis/scripts/run_ceiling.py --concepts $ARM \
         --train-sizes 173 346 693 --add-dev-pool --norm "$NORM" --fit-seed "$SEED" \
         > "$LOGS/ceiling_${ARM}__norm-${NORM}__fit${SEED}.log" 2>&1
    then mark DONE "$NAME"
    else mark "FAILED(rc=$?)" "$NAME"; mark ABORT chain; exit 1; fi
  done
done
mark ALLDONE chain
