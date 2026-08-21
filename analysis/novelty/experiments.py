"""The four red-team arms this analysis covers, and where each one's data lives.

Two concepts x two attacker models. Everything is read from the caches the original
runs wrote — no LLM is ever loaded, and a cache miss raises rather than silently
forwarding a 27B model (same contract as ``analysis/ceiling/harness.py``).

The parameters below are not guesses: the base/dev activation blob names are content
hashes of (data file, model, layer, seed, test_size, split_field, fraction, transforms)
and of (dev file names + bytes, model, layer, transforms) respectively, and every value
here was verified by reproducing the on-disk hash. ``SEED = 42`` in particular is the
CLI default, not the ``seed: 0`` a reader might assume from ``retrain_probe``'s
signature -- the blobs only reproduce at 42.

The red-team side is keyed the same way: ``_redteam_activation_cache_path`` hashes the
*transformed* messages, and all four arms hit 100% of their per-conversation blobs when
hashed with ``combine=True, convert=True``, which is what the eval configs use.

NOTE the two concepts differ in ways that matter downstream:

* instructions fits a **10-member ensemble** (``probe.ensemble_size: 10``), high-stakes
  fits a **single** probe. ``ensemble_size`` carries that.
* instructions' eval/dev activations are small enough to park on the GPU whole
  (1.9 GiB dev / 4.9 GiB eval); high-stakes' are **not** (19.6 GiB dev, 46 GiB eval,
  on a 24 GiB card and 62 GiB of host RAM). Anything touching high-stakes activations
  has to stream. ``heavy`` marks that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

MODEL = "google/gemma-3-27b-it"
LAYER = 32
COMBINE = True   # eval.combine_consecutive_messages
CONVERT = True   # eval.convert_tool_to_assistant
SEED = 42        # the runs' --seed (CLI default); verified against the blob hashes


@dataclass(frozen=True)
class Arm:
    """One attacker model's run of a concept."""

    name: str
    probe_dir: Path
    last_iteration: int
    comparison_csv: Path

    def redteam_jsonl(self, iteration: int | None = None) -> Path:
        it = self.last_iteration if iteration is None else iteration
        return self.probe_dir / f"redteam_postprocessed_iter{it}.jsonl"

    def probe_pkl(self, iteration: int | None = None) -> Path:
        it = self.last_iteration if iteration is None else iteration
        return self.probe_dir / f"probe_iter{it}.pkl"


@dataclass(frozen=True)
class Experiment:
    """One concept: shared activation cache, eval/dev/base data, and its arms."""

    key: str
    pos: str
    neg: str
    eval_dir: Path
    dev_dir: Path
    base_data: Path
    cache_dir: Path          # holds base_acts_*, dev_acts_*, redteam_acts_*/
    eval_acts_dir: Path
    ensemble_size: int
    heavy: bool              # activations too big for host RAM / GPU in one piece
    # Rows of the dev set used as the fit's VALIDATION set. None = all of them.
    # High-stakes' dev blob is 19.6 GiB: staged whole it fills the card and leaves the
    # training set to be copied across PCIe every epoch, which measured at >6 minutes
    # for a SINGLE member -- 20+ hours for the study. A stratified subsample small
    # enough to sit on the card *beside* the training set restores the all-resident fit
    # (the ~100x speedup retrain.py documents). Validation here only picks the best
    # epoch for a 5376-parameter linear head, which a few hundred balanced rows do
    # perfectly well. Every condition shares the same subsample, so conditions stay
    # comparable to each other -- which is what this analysis compares. It does mean the
    # high-stakes `full` is not the published probe; nothing here is compared to the
    # published number anyway (see README on row order).
    dev_fit_rows: int | None = None
    arms: dict[str, Arm] = field(default_factory=dict)

    @property
    def base_acts_dir(self) -> Path:
        return self.cache_dir

    def splits(self) -> list[str]:
        return sorted(p.stem for p in self.eval_dir.glob("*.jsonl"))


_INSTR_DIR = ROOT / "results_instructions_gemma27b_shared"
_HS_DIR = ROOT / "results_hs_gemma27b_devval"

INSTRUCTIONS = Experiment(
    key="instructions",
    pos="assistant_follows_the_instruction",
    neg="assistant_does_not_follow_the_instruction",
    eval_dir=ROOT / "eval_sets" / "instructions",
    dev_dir=ROOT / "dev_samples" / "instructions",
    base_data=ROOT / "data" / "instructions_llama70b_50.jsonl",
    cache_dir=_INSTR_DIR / "base_activations",
    eval_acts_dir=_INSTR_DIR / "eval_activations",
    ensemble_size=10,
    heavy=False,
    dev_fit_rows=None,
    arms={
        "gptoss": Arm(
            "gptoss",
            ROOT / "probes" / "instructions_gemma27b_xmemocat_gptoss",
            5,
            ROOT / "results_instructions_gemma27b_xmemocat_gptoss" / "gptoss120b_comparison.csv",
        ),
        "nemotron": Arm(
            "nemotron",
            ROOT / "probes" / "instructions_gemma27b_xmemocat_nemotron",
            5,
            ROOT / "results_instructions_gemma27b_xmemocat_nemotron" / "nemotron_comparison.csv",
        ),
    },
)

HIGHSTAKES = Experiment(
    key="highstakes",
    pos="high-stakes",
    neg="low-stakes",
    eval_dir=ROOT / "eval_sets" / "highstakes",
    dev_dir=ROOT / "dev_samples" / "highstakes",
    base_data=ROOT / "data" / "highstakes_llama70b_50.jsonl",
    cache_dir=_HS_DIR / "base_activations",
    eval_acts_dir=_HS_DIR / "eval_activations",
    ensemble_size=1,
    heavy=True,
    dev_fit_rows=400,
    arms={
        "gptoss120b": Arm(
            "gptoss120b",
            ROOT / "probes" / "hs_gemma27b_gptoss120b_devval",
            3,
            ROOT / "results_hs_gemma27b_gptoss120b_devval" / "gptoss120b_comparison.csv",
        ),
        "deepseekv4pro": Arm(
            "deepseekv4pro",
            ROOT / "probes" / "hs_gemma27b_deepseekv4pro_devval",
            3,
            ROOT / "results_hs_gemma27b_deepseekv4pro_devval" / "deepseekv4pro_comparison.csv",
        ),
    },
)

EXPERIMENTS = {"instructions": INSTRUCTIONS, "highstakes": HIGHSTAKES}

RESULTS = Path(__file__).resolve().parent / "results"


def get(key: str) -> Experiment:
    if key not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {key!r}; have {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[key]


def arm_of(exp_key: str, arm_key: str) -> tuple[Experiment, Arm]:
    exp = get(exp_key)
    if arm_key not in exp.arms:
        raise KeyError(f"{exp_key} has no arm {arm_key!r}; have {sorted(exp.arms)}")
    return exp, exp.arms[arm_key]


def all_arms() -> list[tuple[Experiment, Arm]]:
    return [(e, a) for e in EXPERIMENTS.values() for a in e.arms.values()]
