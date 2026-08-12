#!/usr/bin/env bash
# One-command cloud-box launch: start the failsafe committer, then the ablation.
#
# Order is deliberate. The failsafe goes first so that if the ablation dies in its first
# minute the box still has a committer running to capture whatever landed and to report
# it; starting the ablation first leaves a window where results exist and nothing is
# watching them.
#
# Both are nohup'd and disowned, so closing the ssh session does not kill them.
#
#   bash cloud_start_intact_pair.sh                       # commit + push, box stays up
#   CLOSE_SCRIPT=close_this.sh bash cloud_start_intact_pair.sh   # power off when done
#
# Check on it with:
#   tail -f intact_pair_nonlinear.log        # the ablation
#   tail -f /tmp/failsafe_intact_pair.out    # the committer
#   ls results/intact_pair_nonlinear/results/*.csv | wc -l   # jobs landed
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

WORK_DIR="${WORK_DIR:-results/intact_pair_nonlinear}"
CLOSE_SCRIPT="${CLOSE_SCRIPT:-}"
FAILSAFE_OUT="${FAILSAFE_OUT:-/tmp/failsafe_intact_pair.out}"

# Fail before launching anything if the box cannot push, rather than 13 hours in when the
# only thing left to lose is the results.
if ! git rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
    echo "ERROR: this branch has no upstream. Set one first:" >&2
    echo "  git push -u origin \$(git rev-parse --abbrev-ref HEAD)" >&2
    exit 2
fi
if [[ ! -f "$WORK_DIR/clusters.json" ]]; then
    echo "ERROR: $WORK_DIR/clusters.json is missing. It is committed to this branch —" >&2
    echo "       check out the right branch rather than letting the box recompute it." >&2
    exit 2
fi

FS_ARGS=(--work-dir "$WORK_DIR")
[[ -n "$CLOSE_SCRIPT" ]] && FS_ARGS+=(--close-script "$CLOSE_SCRIPT")

echo ">>> failsafe committer -> $FAILSAFE_OUT"
nohup bash failsafe_commit_arch_ablation.sh "${FS_ARGS[@]}" > "$FAILSAFE_OUT" 2>&1 &
disown
sleep 2
head -20 "$FAILSAFE_OUT"

echo
echo ">>> ablation -> intact_pair_nonlinear.log"
WORK_DIR="$REPO_ROOT/$WORK_DIR" nohup bash run_intact_pair_nonlinear.sh \
    > /tmp/intact_pair_launch.out 2>&1 &
disown
sleep 5
tail -20 /tmp/intact_pair_launch.out

echo
echo "Both running. 150 jobs at ~5.5 min each is roughly 14 hours."
