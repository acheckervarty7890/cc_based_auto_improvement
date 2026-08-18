#!/usr/bin/env bash
# Extract the base training split's activations for the experiment11 gemma-3-27b arms on a
# box too small to hold the model in GPU + RAM.
#
# This box has 8 GB of VRAM and ~19 GB of free RAM against ~31 GB of truncated (layers 0..32)
# bf16 weights, so ~13 GB has to live on disk no matter how the budget is pinned. Both env
# knobs below are load-bearing:
#
#   AGENTIC_REDTEAM_MAX_MEMORY     pins accelerate's per-device budget instead of letting it
#                                  infer one from whatever is free at load time. Unpinned, it
#                                  reserves so much headroom that only 3 modules land on the
#                                  GPU; pinned, the GPU is filled first and the disk share
#                                  is as small as this box allows.
#   AGENTIC_REDTEAM_OFFLOAD_FOLDER mandatory here, not an optimisation: transformers REFUSES
#                                  to place weights on disk without one.
#
# The offload folder is on the NVMe (not /tmp-as-tmpfs, which would be RAM) and is reread on
# every forward pass, so its read bandwidth is the run's dominant cost.
set -euo pipefail

cd "$(dirname "$0")/.."

export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=7GiB,cpu=20GiB}"
export AGENTIC_REDTEAM_OFFLOAD_FOLDER="${AGENTIC_REDTEAM_OFFLOAD_FOLDER:-$PWD/results_hu_harm_gemma27b_batch_ablation/offload}"

echo "max_memory     : $AGENTIC_REDTEAM_MAX_MEMORY"
echo "offload folder : $AGENTIC_REDTEAM_OFFLOAD_FOLDER"
echo "started        : $(date -Is)"

exec .venv_claude/bin/python scripts/publish_kaggle_redteam_activations.py \
    extract --base-only "$@"
