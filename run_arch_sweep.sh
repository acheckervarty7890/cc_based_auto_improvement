#!/usr/bin/env bash
#
# run_arch_sweep.sh — drive scripts/arch_sweep.py SEED-MAJOR rather than arm-major.
#
# WHY. arch_sweep.py's own loop is `for arm in args.arm: run_arm(...)`, which finishes
# all 40 fits of one attacker arm before starting the other. At ~20 min per Adam fit that
# is ~13 h during which the second arm has no fits at all — so the cross-arm hard-core
# recovery, which is the sweep's primary readout, cannot be computed from a partial run.
# Invoking the script once per seed instead makes every completed seed a COMPLETE
# 8-architecture x 2-arm comparison, so the sweep is readable after the first seed and can
# be stopped after any seed rather than only at the end.
#
# The script resumes at (arm, architecture, seed, variant, legacy) granularity, so the
# repeated invocations skip everything already recorded and cost nothing.
#
# COST. One assembly + train/val build per (seed, arm) rather than per arm: ~2 min each,
# ~20 min added across 5 seeds, against a >12 h run. Worth it for the ordering.
#
# NOTE the fits are ~20 min each because every epoch moves ~10 GB over PCIe (47 batches of
# activations padded to 1024 tokens, plus a full validation pass). That is a property of
# tuberlens' ActivationDataset, which gathers each batch out of the 8.2 GB CPU tensor and
# then splits it into per-row tuples for the default collate to re-stack. Fixing it would
# change the batching the pipeline actually uses, so it is left alone.
#
# USAGE:
#   nohup bash run_arch_sweep.sh 42 43 44 45 46 >> logs/arch_sweep.log 2>&1 &
#
# To stop it, kill the python child by PID — a `pkill -f arch_sweep` also matches the
# shell wrapper of whatever command you typed it in, and will kill that too.
#
# DO NOT run `arch_sweep.py --summarize-only` (or anything else that loads activations)
# while this is going. Each new (seed, arm) starts with an assembly + train/val build that
# transiently peaks near 20 GB, and this box's cgroup limit sits just above that: a
# concurrent summarize was enough to get the sweep OOM-killed mid-run (exit 137). The
# damage is worse than one lost fit, because this loop treats a non-zero exit as "move to
# the next seed" — so the killed seed is left half-finished and loses the property that
# every completed seed is a full comparison. If a seed dies, re-run it explicitly:
# resume skips the fits it already has.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for seed in "$@"; do
    echo ">>> ===== seed $seed, both arms ===== $(date -Is)"
    .venv_claude/bin/python scripts/arch_sweep.py --seeds "$seed" \
        || echo ">>> seed $seed exited non-zero ($?) — continuing to the next seed"
done
echo ">>> all seeds done $(date -Is)"
