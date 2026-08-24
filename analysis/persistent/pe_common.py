"""Shared plumbing for the persistent-error study.

The question is which **eval rows** every probe of experiment22 and experiment23 gets
wrong, so the unit of analysis is a (probe, eval row) cell: 45 probes x 866 rows. Three
things about the setup are load-bearing.

**The 45 probes span two branches.** experiment22's two arms sit in the working tree;
experiment23's three arms are committed on ``experiment23_cloud`` and are read out of git
into ``probes_exp23/`` by :func:`ensure_exp23_probes`. Nothing is checked out — a study
that spans both branches cannot ask the tree to be on one of them.

**Scoring runs with ``PROBE_FUSED_ENSEMBLE=0``.** That is how the runs themselves scored
(both run scripts set it), and it is what makes :mod:`score` able to assert its
probabilities reproduce each run's published comparison CSV. Fused scoring moves AUROC in
the 4th decimal and flips a handful of cells that sit within 1e-3 of 0.5 — harmless for a
mean, not harmless for a claim about *which rows* are always wrong.

**No LLM is loaded and no activation is recomputed.** Every eval row is scored off the
cached full-split blobs the runs themselves used, reached through the ceiling analysis's
``ca_common``/``ca_data`` sources.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STAGE = HERE / "probes_exp23"

CONCEPT = "hu_ha"
POS = "harmful_to_human"
NEG = "not_harmful_to_human"
EVAL_DIR = REPO / "eval_sets/hu_ha"

# The branch experiment23's three arms are committed on. Its probe pickles are the only
# input to this study that is not in the working tree.
EXP23_BRANCH = "experiment23_cloud"


@dataclass(frozen=True)
class Run:
    experiment: str
    arm: str          # short key used everywhere downstream
    label: str        # human-readable
    n_iters: int      # probe_iter0 .. probe_iter{n_iters}
    probe_dir: str    # repo-relative
    results_csv: str  # repo-relative published comparison CSV, for the reproduction check
    branch: str | None = None   # None -> working tree


RUNS: list[Run] = [
    Run("exp22", "gptoss120b_datadesc", "exp22 arm 1 - gpt-oss-120b", 5,
        "probes/hu_harm_gemma27b_gptoss120b_datadesc",
        "results_hu_harm_gemma27b_gptoss120b_datadesc/gptoss120b_datadesc_comparison.csv"),
    Run("exp22", "deepseekv4pro_datadesc", "exp22 arm 2 - deepseek-v4-pro", 5,
        "probes/hu_harm_gemma27b_deepseekv4pro_datadesc",
        "results_hu_harm_gemma27b_deepseekv4pro_datadesc/"
        "deepseekv4pro_datadesc_comparison.csv"),
    Run("exp23", "s3_control", "exp23 - control", 10,
        "probes/hu_harm_gemma27b_gptoss120b_s3_control",
        "results_hu_harm_gemma27b_gptoss120b_s3_control/"
        "gptoss120b_s3_control_comparison.csv", EXP23_BRANCH),
    Run("exp23", "s3_itermemo150", "exp23 - itermemo150", 10,
        "probes/hu_harm_gemma27b_gptoss120b_s3_itermemo150",
        "results_hu_harm_gemma27b_gptoss120b_s3_itermemo150/"
        "gptoss120b_s3_itermemo150_comparison.csv", EXP23_BRANCH),
    Run("exp23", "s3_evaldesc", "exp23 - evaldesc", 10,
        "probes/hu_harm_gemma27b_gptoss120b_s3_evaldesc",
        "results_hu_harm_gemma27b_gptoss120b_s3_evaldesc/"
        "gptoss120b_s3_evaldesc_comparison.csv", EXP23_BRANCH),
]

ARMS = [r.arm for r in RUNS]
ARM_LABEL = {r.arm: r.label for r in RUNS}


# ------------------------------------------------------------------ staging from git
def _git_show(ref: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"],
                          check=True, stdout=subprocess.PIPE).stdout


def ensure_exp23_probes() -> None:
    """Materialize the branch-resident probes under ``probes_exp23/``, once."""
    for run in RUNS:
        if run.branch is None:
            continue
        out = STAGE / run.arm
        out.mkdir(parents=True, exist_ok=True)
        for i in range(run.n_iters + 1):
            dst = out / f"probe_iter{i}.pkl"
            if dst.exists():
                continue
            dst.write_bytes(_git_show(run.branch, f"{run.probe_dir}/probe_iter{i}.pkl"))


def comparison_csv(run: Run) -> str:
    """The run's published per-iteration eval table, as text (branch or tree)."""
    if run.branch is None:
        return (REPO / run.results_csv).read_text(encoding="utf-8")
    return _git_show(run.branch, run.results_csv).decode("utf-8")


def probe_registry() -> list[tuple[str, str, int, Path]]:
    """``(experiment, arm, iteration, path)`` for all 45 probes, in a stable order."""
    ensure_exp23_probes()
    out = []
    for run in RUNS:
        base = (REPO / run.probe_dir) if run.branch is None else (STAGE / run.arm)
        for i in range(run.n_iters + 1):
            p = base / f"probe_iter{i}.pkl"
            if not p.exists():
                raise FileNotFoundError(p)
            out.append((run.experiment, run.arm, i, p))
    return out


def load_probe(path: Path):
    """Unpickle onto the CPU, then move every ensemble member to the fit device.

    Straight from ``ProbeJudge.load``: an ``EnsembleProbe`` has no ``_classifier`` of its
    own, so reconciling only the top-level object would leave all ten members on the CPU
    and the first forward would die on a device mismatch.
    """
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from agentic_redteam.ensemble import iter_probe_members
    from agentic_redteam.probe_judge import _cpu_unpickle
    from tuberlens.config import global_settings

    with Path(path).open("rb") as fh:
        probe = _cpu_unpickle(fh)
    for member in iter_probe_members(probe):
        clf = getattr(member, "_classifier", None)
        if clf is not None and getattr(clf, "model", None) is not None:
            clf.model.to(device=global_settings.DEVICE, dtype=global_settings.DTYPE)
    return probe


# ------------------------------------------------------------------------ eval rows
def eval_rows() -> list[dict]:
    """Every eval row, in ``sorted(*.jsonl)`` then file order.

    That is the order ``ca_common.eval_sources`` yields and the order the score matrix's
    columns are in, so a column index means the same row here.
    """
    out = []
    for path in sorted(EVAL_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.append({"split": path.stem, "messages": json.loads(r["inputs"]),
                            "label": r["labels"],
                            "explanation": r.get("harm_explanation", "")})
    return out


# ------------------------------------------------------------- error-set definitions
def errors_at_half(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``p > 0.5`` — the convention the runs' comparison CSVs report accuracy under."""
    return (P > 0.5) != y[None, :].astype(bool)


def errors_at_split_median(P: np.ndarray, y: np.ndarray, split: np.ndarray) -> np.ndarray:
    """Threshold-free: is the row on the wrong side of its split's median score?

    Needed because these probes are shifted — they call ~31% of eval rows positive against
    a 50% base rate — so ``p > 0.5`` conflates "the probe ranks this row wrongly" with
    "the probe's scores are low". Every eval split is exactly class balanced, so the
    per-probe, per-split median is the balanced-accuracy-optimal rank cut, and a row on the
    wrong side of it is misranked rather than merely mis-thresholded.
    """
    out = np.zeros(P.shape, dtype=bool)
    for s in np.unique(split):
        m = split == s
        med = np.median(P[:, m], axis=1, keepdims=True)
        out[:, m] = (P[:, m] > med) != y[None, m].astype(bool)
    return out


def load_scores(name: str = "scores.npz"):
    return np.load(RESULTS / name, allow_pickle=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
