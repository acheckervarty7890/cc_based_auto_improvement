#!/usr/bin/env bash
# The dev sweep under LayerNorm, and the replication the sweep never had.
#
# Two parts:
#
#  1. The full sweep (10 points x 3 arms x 3 dev draws) under --norm layernorm. The
#     unnormalized sweep already exists as results/sweep_hu_ha_dd_gptoss120b.jsonl, at the
#     same fit seed, so it is the comparison arm and is not re-run.
#
#  2. The N=0 point -- base + red-team only, no dev data -- under four head seeds for BOTH
#     architectures. That point is fit exactly once by run_sweep.py (with no dev rows drawn,
#     the draw seed is meaningless), and it is the point the write-ups quote as "red-team
#     only". The ceiling study showed a 0.0068 seed swing on a comparable fit, so a
#     single-fit comparison there would be unreadable. `--n-points 1` evaluates to points=[0]
#     and `--arms mixed` leaves exactly that one fit.
#     The `none/42` run is the control: it must reproduce the existing sweep's N=0 row.
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

mark() { echo ">>> $(date -Is)  $1  $2" | tee -a "$LOGS/run_norm_sweep.log"; }
run() {
  local name="$1"; shift
  mark START "$name"
  if "$@" > "$LOGS/$name.log" 2>&1; then mark DONE "$name"
  else mark "FAILED(rc=$?)" "$name"; mark ABORT chain; exit 1; fi
}

# ---------------------------------------------------------------- 1. full sweep
run sweep_${ARM}__norm-layernorm \
    $PY ceiling_analysis/scripts/run_sweep.py --concepts $ARM --norm layernorm

# ---------------------------------------------------------------- 2. N=0 replication
for SEED in 42 7 1234 20260823; do
  for NORM in none layernorm; do
    run "sweepN0_${ARM}__norm-${NORM}__fit${SEED}" \
        $PY ceiling_analysis/scripts/run_sweep.py --concepts $ARM \
        --norm "$NORM" --fit-seed "$SEED" --n-points 1 --arms mixed
  done
done
mark ALLDONE chain
