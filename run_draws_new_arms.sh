#!/usr/bin/env bash
set -e
# Resampling grids for ARMS 9-12 (`attacker.show_eval_data_description: true`), matching
# what run_draws_per_arm.sh did for arms 1-8: 8 draws per arm at 90% and again at 80% of
# that arm's red-team successes, each refit from probe_iter0 and scored on the full eval
# splits. The spread across draws is the arm's own noise floor.
#
# ONE PROCESS PER ARM PER FRACTION — carried over from run_draws_per_arm.sh: a single
# process doing all the draws leaked ~2 GB per draw and would be OOM-killed partway.
# fit_redteam_draws.py skips any (arm, draw) already in the CSV, so this is also the
# restart path.
#
# Writes results_hs_draws/draws_comparison.csv (90%) and draws_comparison_f80.csv (80%),
# appending to the same files arms 1-8 already populated.
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
for frac in 0.9 0.8; do
    for arm in arm9 arm10 arm11 arm12; do
        echo ">>> $(date -Is)  $arm  fraction=$frac"
        .venv_claude/bin/python scripts/fit_redteam_draws.py --arms "$arm" --draws 8 --fraction "$frac"
        echo ">>> $(date -Is)  $arm  fraction=$frac done"
    done
done
echo ">>> $(date -Is)  all draws finished."
