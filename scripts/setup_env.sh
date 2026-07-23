#!/usr/bin/env bash
# Bootstrap the agentic_redteam environment from scratch on a fresh machine.
#
#   0. clone cc_based_auto_improvement and cd into it (skipped if already inside it)
#   1. create a Python virtualenv (.venv_claude)
#   2. clone tuberlens @ iterative_pipeline_2 and install it editable
#   3. install the pinned requirements.txt
#   4. install this repo (agentic_redteam) editable
#
# Usage:
#   bash scripts/setup_env.sh [--venv DIR] [--repo DIR] [--python BIN]
#                             [--branch NAME] [--no-editable-tuberlens]
#                             [--clone-dir DIR] [--repo-branch NAME] [--no-clone]
#
# On a bare machine, run it standalone — it clones the repo for you:
#   curl -fsSL https://raw.githubusercontent.com/acheckervarty7890/cc_based_auto_improvement/main/scripts/setup_env.sh | bash
# (private repo → fetch the script however you can, or just `git clone` first and
#  run `bash scripts/setup_env.sh` from inside the checkout.)
#
# Env overrides: VENV_DIR, REPO_ROOT, PYTHON_BIN, TUBERLENS_BRANCH,
#                REPO_URL, REPO_BRANCH, CLONE_DIR
#
# Notes for network/9p/NFS mounts: if the repo lives on a mount that can't host
# a venv (noexec, or slow), point --venv somewhere local, e.g.
#   bash scripts/setup_env.sh --venv "$HOME/.venvs/venv_claude"
set -euo pipefail

TUBERLENS_URL="${TUBERLENS_URL:-https://github.com/acheckervarty7890/tuberlens.git}"
TUBERLENS_BRANCH="${TUBERLENS_BRANCH:-iterative_pipeline_2}"
REPO_URL="${REPO_URL:-https://github.com/acheckervarty7890/cc_based_auto_improvement.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
CLONE_DIR="${CLONE_DIR:-}"
# When piped from curl, BASH_SOURCE is /dev/stdin or "bash" — then there is no
# surrounding checkout and step 0 has to clone one.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd || echo "")"
_script_repo="${_script_dir:+$(cd "$_script_dir/.." && pwd)}"
# An inherited REPO_ROOT (this project exports one) only wins if it really is a
# checkout; otherwise the directory above this script does.
if [ -n "${REPO_ROOT:-}" ] && [ ! -f "$REPO_ROOT/requirements.txt" ]; then
    REPO_ROOT=""
fi
REPO_ROOT="${REPO_ROOT:-$_script_repo}"
VENV_DIR="${VENV_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
TUBERLENS_EDITABLE=1
DO_CLONE=1

while [ $# -gt 0 ]; do
    case "$1" in
        --venv)   VENV_DIR="$2"; shift 2 ;;
        --repo)   REPO_ROOT="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --branch) TUBERLENS_BRANCH="$2"; shift 2 ;;
        --clone-dir)   CLONE_DIR="$2"; shift 2 ;;
        --repo-branch) REPO_BRANCH="$2"; shift 2 ;;
        --no-clone) DO_CLONE=0; shift ;;
        --no-editable-tuberlens) TUBERLENS_EDITABLE=0; shift ;;
        -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v git >/dev/null 2>&1 || { echo "ERROR: git not found on PATH." >&2; exit 1; }

# --------------------------------------------- 0) clone cc_based_auto_improvement ---
# Already inside a usable checkout? Then nothing to clone.
if [ -n "$REPO_ROOT" ] && [ -f "$REPO_ROOT/requirements.txt" ] && [ -f "$REPO_ROOT/pyproject.toml" ]; then
    :
elif [ "$DO_CLONE" -eq 1 ]; then
    CLONE_DIR="${CLONE_DIR:-$PWD/cc_based_auto_improvement}"
    if [ -d "$CLONE_DIR/.git" ]; then
        echo "==> repo checkout exists at $CLONE_DIR; fetching $REPO_BRANCH"
        git -C "$CLONE_DIR" fetch --quiet origin "$REPO_BRANCH"
        git -C "$CLONE_DIR" checkout "$REPO_BRANCH"
        git -C "$CLONE_DIR" merge --ff-only "origin/$REPO_BRANCH" || \
            echo "    (local commits present — skipping fast-forward)"
    else
        [ -e "$CLONE_DIR" ] && { echo "ERROR: $CLONE_DIR exists and is not a git checkout." >&2; exit 1; }
        echo "==> cloning $REPO_URL -> $CLONE_DIR"
        git clone --branch "$REPO_BRANCH" "$REPO_URL" "$CLONE_DIR" || {
            echo "ERROR: clone failed — private repo? configure an SSH key or PAT," >&2
            echo "       or set REPO_URL to the ssh form (git@github.com:...)." >&2
            exit 1; }
    fi
    REPO_ROOT="$CLONE_DIR"
    cd "$REPO_ROOT"
    echo "==> working inside $PWD"
else
    echo "ERROR: not inside a cc_based_auto_improvement checkout and --no-clone was given." >&2
    exit 1
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv_claude}"
mkdir -p "$(dirname "$VENV_DIR")"
VENV_DIR="$(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"

PY="$VENV_DIR/bin/python"
PIP="$PY -m pip"
TUBERLENS_DIR="$VENV_DIR/src/tuberlens"

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight ---
command -v git >/dev/null 2>&1 || die "git not found on PATH."
[ -f "$REPO_ROOT/requirements.txt" ] || die "no requirements.txt under $REPO_ROOT (wrong --repo?)."
[ -f "$REPO_ROOT/pyproject.toml" ]  || die "no pyproject.toml under $REPO_ROOT (wrong --repo?)."

if [ -z "$PYTHON_BIN" ]; then
    for cand in python3.12 python3.13 python3.11 python3; do
        if command -v "$cand" >/dev/null 2>&1; then PYTHON_BIN="$(command -v "$cand")"; break; fi
    done
fi
[ -n "$PYTHON_BIN" ] || die "no python3 interpreter found; pass --python /path/to/python3.12"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' \
    || die "$PYTHON_BIN is $("$PYTHON_BIN" -V 2>&1); need >= 3.10 (3.12 recommended)."
"$PYTHON_BIN" -c 'import venv' >/dev/null 2>&1 \
    || die "the 'venv' module is missing (on Debian/Ubuntu: apt install python3.12-venv)."

echo "==> repo:   $REPO_ROOT"
echo "==> venv:   $VENV_DIR"
echo "==> python: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"

# ------------------------------------------------------ 1) virtualenv ---------
if [ -x "$PY" ]; then
    echo "==> venv already present, reusing"
else
    echo "==> creating venv"
    "$PYTHON_BIN" -m venv "$VENV_DIR" || die "venv creation failed at $VENV_DIR (noexec mount? try --venv \$HOME/.venvs/venv_claude)"
    [ -x "$PY" ] || die "venv created but $PY is not executable (mount mounted noexec?)."
fi
# Confirm the interpreter actually runs from where it landed.
"$PY" -c 'import sys; print("    interpreter ok:", sys.executable)' \
    || die "$PY exists but will not execute."
$PIP install -U pip setuptools wheel

# ------------------------------------------------------ 2) tuberlens ----------
if [ -d "$TUBERLENS_DIR/.git" ]; then
    echo "==> tuberlens checkout exists; syncing to $TUBERLENS_BRANCH"
    git -C "$TUBERLENS_DIR" fetch --quiet origin "$TUBERLENS_BRANCH"
    git -C "$TUBERLENS_DIR" checkout "$TUBERLENS_BRANCH"
    git -C "$TUBERLENS_DIR" merge --ff-only "origin/$TUBERLENS_BRANCH"
else
    echo "==> cloning tuberlens@$TUBERLENS_BRANCH -> $TUBERLENS_DIR"
    rm -rf "$TUBERLENS_DIR"
    mkdir -p "$(dirname "$TUBERLENS_DIR")"
    git clone --branch "$TUBERLENS_BRANCH" "$TUBERLENS_URL" "$TUBERLENS_DIR" \
        || die "clone failed — private repo? configure an SSH key or PAT, or set TUBERLENS_URL to the ssh form."
fi
echo "    tuberlens at $(git -C "$TUBERLENS_DIR" rev-parse --short HEAD) on $(git -C "$TUBERLENS_DIR" rev-parse --abbrev-ref HEAD)"

echo "==> installing tuberlens"
if [ "$TUBERLENS_EDITABLE" -eq 1 ]; then
    $PIP install -e "$TUBERLENS_DIR"
else
    $PIP install "$TUBERLENS_DIR"
fi

# ------------------------------------------------------ 3) requirements -------
# After tuberlens, so the pins in requirements.txt win over whatever tuberlens'
# own metadata resolved to (notably torch/transformers).
echo "==> installing requirements.txt"
$PIP install -r "$REPO_ROOT/requirements.txt"

# ------------------------------------------------------ 4) this repo ----------
echo "==> installing agentic_redteam (editable)"
$PIP install -e "$REPO_ROOT"

# ------------------------------------------------------ smoke test ------------
echo "==> smoke test"
"$PY" - <<'EOF'
import importlib
for mod in ("tuberlens", "agentic_redteam", "torch", "anthropic", "openai", "yaml"):
    importlib.import_module(mod)
from agentic_redteam.attacker import run_redteam  # noqa: F401
import torch
print("    ok — torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
EOF

$PIP check || echo "    (pip check reported conflicts — review above)"

cat <<EOF

Done.
  repo:        $REPO_ROOT
  interpreter: $PY
  tuberlens:   $TUBERLENS_DIR

  cd $REPO_ROOT

Export the keys you need before running:
  export ANTHROPIC_API_KEY=...     # any section with provider: claude_sdk
  export OPENROUTER_API_KEY=...    # any section with provider: openrouter

Then e.g.:
  $PY scripts/run_redteam.py configs/example_config.md
EOF
