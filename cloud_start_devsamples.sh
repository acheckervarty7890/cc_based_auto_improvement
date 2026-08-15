#!/usr/bin/env bash
# One-command cloud-box launch: start the failsafe committer, then the two experiments.
#
# Order is deliberate. The failsafe goes first so that if the run dies in its first minute
# the box still has a committer running to capture whatever landed and to report it;
# starting the run first leaves a window where results exist and nothing is watching them.
#
# Both are nohup'd and disowned, so closing the ssh session does not kill them.
#
#   export KAGGLE_CONFIG_DIR=/path/to/dir-holding-kaggle.json
#   bash cloud_start_devsamples.sh                                # commit + push, box stays up
#   CLOSE_SCRIPT=close_this.sh bash cloud_start_devsamples.sh     # power off when done
#   PARTS="cv" bash cloud_start_devsamples.sh                     # experiment (i) alone
#
# NOTE this branch carries no close_this.sh — the one on intact-pair-nonlinear embeds a
# live API key. Drop your own in if you want CLOSE_SCRIPT.
#
# Check on it with:
#   tail -f devsamples_kfold.log                       # the experiments
#   tail -f /tmp/failsafe_devsamples.out               # the committer
#   cat results/devsamples_kfold/kfold_cv/kfold_cv_summary.csv        # experiment (i)
#   ls results/devsamples_kfold/dev_samples/results/*.csv | wc -l     # experiment (ii), of 10
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

WORK_DIR="${WORK_DIR:-results/devsamples_kfold}"
CLOSE_SCRIPT="${CLOSE_SCRIPT:-}"
FAILSAFE_OUT="${FAILSAFE_OUT:-/tmp/failsafe_devsamples.out}"
PARTS="${PARTS:-cv dev}"
NO_FETCH="${NO_FETCH:-0}"

# Fail before launching anything if the box cannot push, rather than hours in when the
# only thing left to lose is the results.
if ! git rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
    echo "ERROR: this branch has no upstream. Set one first:" >&2
    echo "  git push -u origin \$(git rev-parse --abbrev-ref HEAD)" >&2
    exit 2
fi

# dev_samples/ is COMMITTED to this branch and is the whole input to experiment (ii).
# A box that checked out the wrong branch would otherwise get as far as the Kaggle
# restore before noticing.
if [[ ! -d dev_samples ]] || [[ -z "$(ls -A dev_samples/*.jsonl 2>/dev/null)" ]]; then
    echo "ERROR: dev_samples/*.jsonl is missing. It is committed to this branch —" >&2
    echo "       check out the right branch rather than letting the box run without it." >&2
    exit 2
fi

# Same for the two arms' postprocessed dumps: experiment (ii) trains from them, and they
# are the POST-shortening versions on this branch specifically.
for arm in deepseekv4pro gptoss120b; do
    dump="probes/hu_harm_gemma27b_${arm}_batch/redteam_postprocessed_iter3.jsonl"
    [[ -f "$dump" ]] || { echo "ERROR: missing $dump" >&2; exit 2; }
done

# Kaggle credentials: checked here rather than at first use because an unauthenticated
# KaggleApi.authenticate() ends in exit(1), not an exception, and the failure would land
# in a nohup'd log nobody is watching yet.
if [[ "$NO_FETCH" != "1" && -z "${KAGGLE_API_TOKEN:-}" ]]; then
    kj="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
    [[ -f "$kj" ]] || kj="$HOME/.config/kaggle/kaggle.json"
    if [[ ! -f "$kj" ]]; then
        echo "ERROR: no Kaggle credentials. Set KAGGLE_CONFIG_DIR to the DIRECTORY" >&2
        echo "       holding kaggle.json (not the file), or export KAGGLE_API_TOKEN." >&2
        exit 2
    fi
fi

# gemma-3-27b-it is gated, and the extraction is the only stage that loads it. Checked
# here as well as in run_devsamples_kfold.sh because THIS is the entry point a cloud box
# uses, and everything it starts is nohup'd — an error raised down there lands in a log
# nobody is watching yet.
if [[ " $PARTS " == *" dev "* && -z "${HF_TOKEN:-}${HUGGINGFACE_TOKEN:-}" ]]; then
    echo "ERROR: no HuggingFace token. Export HF_TOKEN (or HUGGINGFACE_TOKEN) from an" >&2
    echo "       account that has accepted the Gemma licence, or run PARTS=\"cv\"," >&2
    echo "       which scores cached activations and loads no model." >&2
    exit 2
fi

mkdir -p logs

FS_ARGS=(--work-dir "$WORK_DIR")
[[ -n "$CLOSE_SCRIPT" ]] && FS_ARGS+=(--close-script "$CLOSE_SCRIPT")

echo ">>> failsafe committer -> $FAILSAFE_OUT"
nohup bash failsafe_commit_devsamples.sh "${FS_ARGS[@]}" > "$FAILSAFE_OUT" 2>&1 &
disown
sleep 2
head -20 "$FAILSAFE_OUT"

echo
echo ">>> experiments -> devsamples_kfold.log"
WORK_DIR="$REPO_ROOT/$WORK_DIR" PARTS="$PARTS" NO_FETCH="$NO_FETCH" \
    nohup bash run_devsamples_kfold.sh > logs/devsamples_kfold_launch.out 2>&1 &
disown
sleep 5
tail -20 logs/devsamples_kfold_launch.out

echo
echo "Both running."
echo "  (i)  k-fold CV: 25 probe fits off cached activations — minutes."
echo "  (ii) dev samples: ~4.6 GB + ~10 GB of Kaggle downloads, then 120 gemma-3-27b"
echo "       forwards, then 10 retrain+eval jobs. Hours, dominated by the downloads"
echo "       and the extraction."
