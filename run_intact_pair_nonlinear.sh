#!/usr/bin/env bash
# Cluster ablation on INTACT CONTRASTIVE PAIRS ONLY, non-linear architectures.
#
# Corpus: 736 rows = the 368 contrastive pairs that survive the 1024-token activation cap
# with BOTH sides intact. The 878-row dump is 439 matched pairs; the truncation filter
# removes 71 generated rows against 1 original, so filtering rows alone leaves 70 orphans
# and tilts the corpus to 372 neg / 434 pos. Restricting to intact pairs restores an exact
# 368/368 balance and means an arm removes whole pairs rather than half-pairs.
#
# Arms (10% of 736 = 74 rows each):
#   full_s{S}            no removal                             -> seed-noise floor
#   far_s{S}             the 74 farthest, taken from c2 then c5 -> fixed set, seed varies
#   near_s{S}            the 74 nearest (mirror of far)         -> fixed set, seed varies
#   neardraw_d{D}_s42    74 drawn at random from c3 u c4        -> fixed seed, draw varies
#   neardraw_d0_s{S}     draw 0 across seeds                    -> puts near on the seed axis
#
# --near-pool-mult 2.5 sizes the near pool at 2.5 x 74 = 185, and the pool is grown by
# whole clusters nearest-first, so it stops at c3 (89) + c4 (125) = 214 -- exactly the two
# nearest clusters and nothing else. Do not change it without re-checking which clusters
# the pool lands on; 3.0 pulls in c1 as well.
#
# Architectures: the four with a NON-LINEAR token aggregation, plus linear_then_mean as
# the linear control. linear_then_mean is the right control specifically because it is the
# same architecture as linear_then_softmax/max/rolling_max with only the pooling swapped,
# so the contrast isolates the non-linearity rather than confounding it with a different
# feature extractor. (pre_mean, linear_then_last, difference_of_means, lda and sklearn are
# also linear but differ in more than the aggregation.)
#
# NO MODEL IS LOADED. Every activation comes from the Kaggle-published caches.
#
#   bash run_intact_pair_nonlinear.sh              # fetch + run + analyze
#   STAGES="run analyze" bash run_intact_pair_nonlinear.sh   # caches already filled
#   DRY_RUN=1 bash run_intact_pair_nonlinear.sh    # print the matrix, run nothing
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

PY="${PY:-$REPO_ROOT/.venv_claude/bin/python}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/results/intact_pair_nonlinear}"
LOG="${LOG:-$REPO_ROOT/intact_pair_nonlinear.log}"

# Four non-linear aggregations + one linear control. Order matters only for scheduling:
# linear_then_softmax first so the arm it already has 5 seeds of on the previous branch
# lands early and is comparable soonest.
ARCHS="${ARCHS:-linear_then_softmax linear_then_max linear_then_rolling_max attention linear_then_mean}"
SEEDS="${SEEDS:-42 101 202 303 404}"
STAGES="${STAGES:-}"
DRY_RUN="${DRY_RUN:-0}"

# clusters.json is COMMITTED, so the default stage list deliberately skips pool+cluster:
# the partition every arm is defined against is frozen in git rather than recomputed on
# the box, where a different sklearn build could cut the dendrogram elsewhere and silently
# redefine "far". Pass STAGES="pool cluster run analyze" to rebuild it on purpose.
if [[ -z "$STAGES" ]]; then
    if [[ -f "$WORK_DIR/clusters.json" ]]; then
        STAGES="run analyze"
    else
        echo "WARN: $WORK_DIR/clusters.json missing — rebuilding the clustering on this box." >&2
        echo "      Expected it to come from git. Results will not be comparable to the" >&2
        echo "      committed partition unless the cut happens to land identically." >&2
        STAGES="pool cluster run analyze"
    fi
fi

ARGS=(
    --work-dir "$WORK_DIR"
    --stages $STAGES
    --exclude-truncated
    --intact-pairs-only
    --near-pool-mult 2.5
    --archs $ARCHS
    --seeds $SEEDS
)
# stage_fetch is additive to --stages, so ask for it explicitly rather than listing it.
[[ "${NO_FETCH:-0}" == "1" ]] || ARGS+=(--fetch-kaggle)
[[ "$DRY_RUN" == "1" ]] && ARGS+=(--dry-run)

echo "=== intact-pair non-linear ablation ==="
echo "  work dir : $WORK_DIR"
echo "  stages   : $STAGES$([[ "${NO_FETCH:-0}" == "1" ]] || echo " (+fetch)")"
echo "  archs    : $ARCHS"
echo "  seeds    : $SEEDS"
echo "  log      : $LOG"
echo

exec "$PY" -u scripts/arch_cluster_ablation.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
