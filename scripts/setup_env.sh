#!/usr/bin/env bash
# Bootstrap the agentic_redteam environment from scratch on a fresh machine.
#
#   1. create a Python virtualenv (.venv_claude)
#   2. clone tuberlens @ iterative_pipeline_2 and install it editable
#   3. install the pinned requirements.txt
#   4. install this repo (agentic_redteam) editable
#
# Usage:
#   bash scripts/setup_env.sh [--venv DIR] [--repo DIR] [--python BIN]
#                             [--branch NAME] [--no-editable-tuberlens]
#
# Env overrides: VENV_DIR, REPO_ROOT, PYTHON_BIN, TUBERLENS_BRANCH
#
# Notes for network/9p/NFS mounts: if the repo lives on a mount that can't host
# a venv (noexec, or slow), point --venv somewhere local, e.g.
#   bash scripts/setup_env.sh --venv "$HOME/.venvs/venv_claude"
set -euo pipefail

TUBERLENS_URL="${TUBERLENS_URL:-https://github.com/acheckervarty7890/tuberlens.git}"
TUBERLENS_BRANCH="${TUBERLENS_BRANCH:-iterative_pipeline_2}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
TUBERLENS_EDITABLE=1

while [ $# -gt 0 ]; do
    case "$1" in
        --venv)   VENV_DIR="$2"; shift 2 ;;
        --repo)   REPO_ROOT="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --branch) TUBERLENS_BRANCH="$2"; shift 2 ;;
        --no-editable-tuberlens) TUBERLENS_EDITABLE=0; shift ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

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
  interpreter: $PY
  tuberlens:   $TUBERLENS_DIR

Export the keys you need before running:
  export ANTHROPIC_API_KEY=...     # any section with provider: claude_sdk
  export OPENROUTER_API_KEY=...    # any section with provider: openrouter

Then e.g.:
  $PY scripts/run_redteam.py configs/example_config.md
EOF
