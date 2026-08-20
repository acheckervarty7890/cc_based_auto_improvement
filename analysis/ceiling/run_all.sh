#!/usr/bin/env bash
# Full ceiling analysis. One GPU, so strictly sequential.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=.venv_claude/bin/python
E=10

echo "=== [1/4] ceiling, gptoss (all four conditions) ==="
$PY analysis/ceiling/run_ceiling.py --arm gptoss --ensemble $E --folds 5 || exit 1

echo "=== [2/4] ceiling, nemotron (arm-dependent conditions only) ==="
# cv_eval and oracle never see red-team data, so they are identical to arm 1's.
$PY analysis/ceiling/run_ceiling.py --arm nemotron --ensemble $E --folds 5 \
    --conditions redteam_only,cv_eval_rt || exit 1

echo "=== [3/4] dev sweep, gptoss ==="
$PY analysis/ceiling/run_sweep.py --arm gptoss --ensemble $E --ft-lr default,1e-4 || exit 1

echo "=== [4/4] dev sweep, nemotron ==="
$PY analysis/ceiling/run_sweep.py --arm nemotron --ensemble $E --ft-lr default,1e-4 || exit 1

echo ">>> all done"
