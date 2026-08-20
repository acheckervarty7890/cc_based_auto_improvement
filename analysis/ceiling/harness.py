"""Shared loaders + fit/score helpers for the ceiling analysis.

Everything here runs off the activation caches written by the ens10dev run — no LLM
is ever loaded. The three sources:

  eval  : results_instructions_gemma27b_shared/eval_activations/<split>-acts_full.pt
  dev   : results_instructions_gemma27b_shared/base_activations/dev_acts_*.pt (one blob)
  base  : results_instructions_gemma27b_shared/base_activations/base_acts_*_train.pt
  redteam: per-conversation blobs under base_activations/redteam_acts_<model>_L<layer>/

Metrics reproduce tuberlens' ``get_performances`` exactly: per-split AUROC /
accuracy / TPR@1%FPR, and the headline number is the MACRO MEAN over the 7 splits
(the "mean" row of the comparison CSVs), so anything computed here is directly
comparable to results_*/**_comparison.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MODEL = "google/gemma-3-27b-it"
LAYER = 32
POS = "assistant_follows_the_instruction"
NEG = "assistant_does_not_follow_the_instruction"
COMBINE = True          # eval.combine_consecutive_messages
CONVERT = True          # eval.convert_tool_to_assistant
SEED = 42               # the run's --seed

SHARED = ROOT / "results_instructions_gemma27b_shared"
EVAL_ACTS = SHARED / "eval_activations"
BASE_ACTS = SHARED / "base_activations"
EVAL_DIR = ROOT / "eval_sets" / "instructions"
DEV_DIR = ROOT / "dev_samples" / "instructions"
BASE_DATA = ROOT / "data" / "instructions_llama70b_50.jsonl"

ARMS = {
    "gptoss": ROOT / "probes" / "instructions_gemma27b_ens10dev_gptoss",
    "nemotron": ROOT / "probes" / "instructions_gemma27b_ens10dev_nemotron",
}

# The repo-pinned ensemble member seeds (agentic_redteam/ensemble.py).
from agentic_redteam.ensemble import ENSEMBLE_SEEDS  # noqa: E402


def _load_from(path: Path):
    from tuberlens.interfaces.dataset import LabelledDataset

    return LabelledDataset.load_from(
        path,
        pos_class_label=POS,
        neg_class_label=NEG,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )


def load_eval_splits() -> dict:
    """The 7 eval splits, each with its cached activations attached. 1302 rows total."""
    from tuberlens.model import LLMModel

    out = {}
    for f in sorted(EVAL_DIR.glob("*.jsonl")):
        ds = _load_from(f)
        blob = EVAL_ACTS / f"{f.stem}-acts_full.pt"
        if not blob.exists():
            raise FileNotFoundError(blob)
        act = LLMModel.load_activations(blob)
        if act.activations.shape[0] != len(ds):
            raise ValueError(f"{f.stem}: blob has {act.activations.shape[0]} rows, dataset {len(ds)}")
        out[f.stem] = ds.assign(
            activations=act.activations,
            attention_mask=act.attention_mask,
            input_ids=act.input_ids,
        )
    return out


def load_dev():
    """The 436-row held-out dev set with its single cached blob."""
    from agentic_redteam.retrain import _dev_activation_cache_path, _load_dev_dataset
    from tuberlens.model import LLMModel

    ds, files = _load_dev_dataset(DEV_DIR, POS, NEG, COMBINE, CONVERT, verbose=False)
    blob = _dev_activation_cache_path(BASE_ACTS, files, MODEL, LAYER, COMBINE, CONVERT)
    if not blob.exists():
        raise FileNotFoundError(blob)
    act = LLMModel.load_activations(blob)
    if act.activations.shape[0] != len(ds):
        raise ValueError(f"dev blob has {act.activations.shape[0]} rows, dataset {len(ds)}")
    return ds.assign(
        activations=act.activations,
        attention_mask=act.attention_mask,
        input_ids=act.input_ids,
    )


def load_base():
    """The 50-row base training split (test_size 0.0 → all of it is train)."""
    from agentic_redteam.retrain import (
        _base_activation_cache_paths,
        stable_fraction_subsample,
        stable_train_test_split,
    )
    from tuberlens.model import LLMModel

    ds = _load_from(BASE_DATA)
    ds = stable_fraction_subsample(ds, 1.0, SEED)
    train, _val = stable_train_test_split(ds, test_size=0.0, split_field=None, seed=SEED)
    train_cache, _ = _base_activation_cache_paths(
        BASE_ACTS, BASE_DATA, MODEL, LAYER, SEED, 0.0, None, COMBINE, CONVERT, 1.0
    )
    if not Path(train_cache).exists():
        raise FileNotFoundError(train_cache)
    act = LLMModel.load_activations(train_cache)
    return train.assign(
        activations=act.activations,
        attention_mask=act.attention_mask,
        input_ids=act.input_ids,
    )


def load_redteam(arm: str, iteration: int = 5):
    """The postprocessed red-team training data that trained probe_iter<iteration>.

    This is the exact filter+contrastive output the run concatenated with the base
    data, so `base ∪ this` reproduces that retrain's training set. The snapshot is
    rebuilt the way the run built it in memory (``_dicts_to_labelled_dataset`` +
    ``_apply_message_transforms``) rather than through ``load_from``, because
    ``_dump_labelled_dataset`` writes a ``label`` column holding tuberlens' canonical
    "positive"/"negative" values, not the probe's class-label strings. Rebuilding it
    this way is also what makes the per-conversation activation cache hit: its key is
    a hash of the transformed messages.

    Activations come from that cache; a miss would need the 27B model, so we refuse.
    """
    import json

    from agentic_redteam.retrain import (
        _apply_message_transforms,
        _concatenate_consuming,
        _redteam_activation_cache_path,
    )
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage
    from tuberlens.model import LLMModel

    path = ARMS[arm] / f"redteam_postprocessed_iter{iteration}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ds = LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]},
    )
    ds = _apply_message_transforms(ds, COMBINE, CONVERT)

    parts, misses = [], 0
    for i, msgs in enumerate(ds.inputs):
        p = _redteam_activation_cache_path(BASE_ACTS, msgs, MODEL, LAYER, COMBINE, CONVERT)
        if not p.exists():
            misses += 1
            continue
        act = LLMModel.load_activations(p)
        parts.append(
            ds[i : i + 1].assign(
                activations=act.activations,
                attention_mask=act.attention_mask,
                input_ids=act.input_ids,
            )
        )
    if misses:
        raise FileNotFoundError(
            f"{misses}/{len(ds)} red-team activation cache misses for {path}. "
            "Computing them would need gemma-3-27b; refusing."
        )
    return _concatenate_consuming(parts)


# ---------------------------------------------------------------- metrics


def tpr_at_fpr(y, p, fpr=0.01):
    """tuberlens' TPR@FPR, reproduced bug-for-bug so numbers stay comparable.

    It takes the ROC point whose FPR is *nearest* the target and reads its TPR. With
    few distinct scores the nearest point is often FPR=0 — and on a perfectly
    separating probe that point is (0, 0), so this returns 0.0 for a probe that in
    fact achieves TPR 1.0 at FPR 0. Use ``tpr_at_fpr_le`` for the conventional
    reading; both are reported.
    """
    from sklearn.metrics import roc_curve

    f, t, _ = roc_curve(y, p)
    return float(t[np.argmin(np.abs(f - fpr))])


def tpr_at_fpr_le(y, p, fpr=0.01):
    """The conventional definition: the best TPR reachable at FPR <= target."""
    from sklearn.metrics import roc_curve

    f, t, _ = roc_curve(y, p)
    ok = f <= fpr + 1e-12
    return float(t[ok].max()) if ok.any() else 0.0


def split_metrics(y, p, fpr=0.01):
    from sklearn.metrics import accuracy_score, roc_auc_score

    return {
        "auroc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, p > 0.5)),
        "tpr_at_fpr": tpr_at_fpr(y, p, fpr),
        "tpr_at_fpr_le": tpr_at_fpr_le(y, p, fpr),
    }


def macro(per_split: dict) -> dict:
    keys = ("auroc", "accuracy", "tpr_at_fpr", "tpr_at_fpr_le")
    return {k: float(np.mean([m[k] for m in per_split.values()])) for k in keys}


def labels_of(ds) -> np.ndarray:
    return np.array([lab.to_int() for lab in ds.labels])


# ---------------------------------------------------------------- fit / score


def probe_spec():
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType

    return ProbeSpec(name=ProbeType.linear_then_softmax)


def fit_member(train, val, seed: int, spec=None, verbose=False):
    """One ProbeFactory.build, seeded exactly the way retrain.py seeds a member."""
    from agentic_redteam.evaluation import seed_everything
    from tuberlens.probes.probe_factory import ProbeFactory

    seed_everything(seed)
    return ProbeFactory.build(
        probe_spec=spec or probe_spec(),
        train_dataset=train,
        model_name=MODEL,
        layer=LAYER,
        validation_dataset=val,
        use_store=False,
        pos_class_label=POS,
        neg_class_label=NEG,
        probe_description=None,
    )


def finetune_member(probe, train, val, seed: int, lr: float | None = None):
    """Continue training an already-fit probe on `train` (initialize_model=False).

    The classifier's weights are kept and only the optimizer/scheduler restart, so
    this is a genuine finetune rather than a fresh fit. `lr` overrides the AdamW
    learning rate for the finetune phase only.
    """
    import copy

    from agentic_redteam.evaluation import seed_everything

    probe = copy.deepcopy(probe)
    if lr is not None:
        args = dict(probe._classifier.training_args)
        args["optimizer_args"] = dict(args["optimizer_args"]) | {"lr": lr}
        args["final_lr"] = min(args["final_lr"], lr / 10)
        probe._classifier.training_args = args
        probe.hyper_params = args
    seed_everything(seed)
    probe.fit(train, val, initialize_model=False)
    return probe


def ensemble_of(members, seeds):
    from agentic_redteam.ensemble import EnsembleProbe

    if len(members) == 1:
        return members[0]
    return EnsembleProbe.from_members(members, list(seeds))


def score_splits(probe, eval_splits: dict) -> dict:
    """Per-split metrics, matching get_performances."""
    out = {}
    for name, ds in eval_splits.items():
        p = probe.predict_proba(ds)
        out[name] = split_metrics(labels_of(ds), np.asarray(p))
    return out


def proba_splits(probe, eval_splits: dict) -> dict:
    return {name: np.asarray(probe.predict_proba(ds)) for name, ds in eval_splits.items()}


# ---------------------------------------------------------------- gpu staging


def stage(*datasets, verbose=False):
    """Park these datasets' activations on the fit device (in place)."""
    from agentic_redteam.retrain import _to_device_for_fit

    _to_device_for_fit([d for d in datasets if d is not None], verbose=verbose)
    return datasets[0] if len(datasets) == 1 else datasets


def free():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_free_gib() -> float:
    from agentic_redteam.retrain import _allocatable_bytes

    return _allocatable_bytes(None) / 2**30 if torch.cuda.is_available() else 0.0


def concat(*datasets):
    from agentic_redteam.retrain import _concatenate_consuming

    return _concatenate_consuming([d for d in datasets if d is not None and len(d) > 0])


def take(dataset, idx):
    """Row subset, copying the pad fields (so the parent can be freed independently)."""
    return dataset[list(int(i) for i in idx)]


def silence_tqdm() -> None:
    """Replace tuberlens' tqdm with a passthrough.

    The fit and every scoring pass wrap their DataLoader in a bar; over a few
    hundred fits that is hundreds of megabytes of carriage returns in the log and
    nothing else. Patched at the module attribute, so nothing else's bars change.
    """
    import tuberlens.evaluation as _ev
    import tuberlens.probes.pytorch_classifiers as _pc

    class _NoBar:
        """Iterates the wrapped iterable and no-ops tqdm's display API."""

        def __init__(self, iterable=None, *a, **k):
            self._it = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self._it)

        def __len__(self):
            return len(self._it)

        def set_postfix(self, *a, **k):
            pass

        def set_description(self, *a, **k):
            pass

        def update(self, *a, **k):
            pass

        def close(self):
            pass

    _passthrough = _NoBar

    _pc.tqdm = _passthrough
    if hasattr(_ev, "tqdm"):
        _ev.tqdm = _passthrough


class Quiet:
    """Capture tuberlens' per-epoch prints during a fit instead of echoing them."""

    def __enter__(self):
        import contextlib, io

        self._buf = io.StringIO()
        self._ctx = contextlib.redirect_stdout(self._buf)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False

    @property
    def text(self) -> str:
        return self._buf.getvalue()

    def epochs_run(self) -> int:
        return self.text.count("Average loss:")
