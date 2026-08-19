"""Shared plumbing for the ceiling analysis.

Everything here runs on **precomputed** layer-32 activations: the eval and dev blobs come
from Kaggle, the red-team and base blobs from the repo's own content-keyed per-conversation
cache. gemma-3-27b is therefore loaded exactly once, by
``extract_redteam_activations.py`` — never by a fit or a score.

The three things worth knowing before changing anything:

* **The message transforms are part of the activation identity.** Every published blob was
  computed with ``combine_consecutive_messages=True`` and ``convert_tool_to_assistant=True``
  (the experiment 17/18/19 configs' ``eval:`` block), and the caches load *by path without
  validating their inputs*. So the loaders here pass the same two flags, always.
* **``LabelledDataset.concatenate`` mutates its inputs** — it pads each part's activation
  tensor in place to the common max before concatenating, and holds inputs and output at
  once. The pools here are reused across ~60 fits, so ``pool`` below allocates the output
  once and copies into it instead, leaving the parts untouched.
* **Padding is dead weight.** ``get_activations`` pads a call's rows to that call's max, so
  a split whose longest conversation is 859 tokens carries 859 columns for rows that are 79
  tokens long. ``trim`` slices the sequence dimension back to the longest *real* row in the
  pool, which is exactly lossless (the attention mask already zeroes the rest) and is what
  makes the high-stakes pools fit on the card.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ACTS_ROOT = REPO / "ceiling_acts"
RESULTS = REPO / "ceiling_analysis" / "results"

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
COMBINE = True
CONVERT = True
ARCH = "linear_then_softmax"
FIT_SEED = 42


@dataclass(frozen=True)
class Concept:
    name: str
    pos_class_label: str
    neg_class_label: str
    description: str
    eval_dir: Path
    dev_dir: Path
    redteam_jsonl: Path
    base_jsonl: Path
    eval_blob_suffix: str = "gemmaeval"
    dev_blob_suffix: str = "gemmadev"

    @property
    def eval_blob_dir(self) -> Path:
        return ACTS_ROOT / self.name / "eval"

    @property
    def dev_blob_dir(self) -> Path:
        return ACTS_ROOT / self.name / "dev"

    @property
    def redteam_cache_dir(self) -> Path:
        return ACTS_ROOT / self.name / "redteam_cache"


CONCEPTS: dict[str, Concept] = {
    "highstakes": Concept(
        name="highstakes",
        pos_class_label="high-stakes",
        neg_class_label="low-stakes",
        description=(
            "A linear probe on google/gemma-3-27b-it detecting whether the conversation "
            "is high-stakes."
        ),
        eval_dir=REPO / "eval_sets/highstakes",
        dev_dir=REPO / "dev_samples/highstakes",
        redteam_jsonl=REPO
        / "ceiling_analysis/data/redteam_highstakes_exp19_gptoss120b_iter5.jsonl",
        base_jsonl=REPO / "data/highstakes_llama70b_50.jsonl",
    ),
    "hu_ha": Concept(
        name="hu_ha",
        pos_class_label="harmful_to_human",
        neg_class_label="not_harmful_to_human",
        description=(
            "A linear probe on google/gemma-3-27b-it detecting whether the conversation "
            "is harmful to humans."
        ),
        eval_dir=REPO / "eval_sets/hu_ha",
        dev_dir=REPO / "dev_samples/hu_ha",
        redteam_jsonl=REPO
        / "ceiling_analysis/data/redteam_hu_ha_exp17_gptoss120b_iter5.jsonl",
        base_jsonl=REPO / "data/hu_harm_llama70b_50.jsonl",
    ),
}


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def _labelled_dataset():
    from tuberlens.interfaces.dataset import LabelledDataset

    return LabelledDataset


def load_jsonl_dataset(path: Path, concept: Concept, *, field_mapping=None):
    """Load one JSONL as a LabelledDataset with the blobs' own message transforms."""
    return _labelled_dataset().load_from(
        Path(path),
        field_mapping=field_mapping,
        pos_class_label=concept.pos_class_label,
        neg_class_label=concept.neg_class_label,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )


def attach_blob(dataset, blob_path: Path):
    """Attach a published activation blob to the dataset it was computed for."""
    blob = torch.load(blob_path, map_location="cpu", weights_only=False)
    if blob["model_name"] != MODEL_NAME or int(blob["layer"]) != LAYER:
        raise ValueError(
            f"{blob_path}: computed with {blob['model_name']} L{blob['layer']}, "
            f"expected {MODEL_NAME} L{LAYER}"
        )
    if blob["activations"].shape[0] != len(dataset):
        raise ValueError(
            f"{blob_path}: {blob['activations'].shape[0]} rows, dataset has {len(dataset)}"
        )
    return dataset.assign(
        activations=blob["activations"],
        attention_mask=blob["attention_mask"],
        input_ids=blob["input_ids"],
    )


def load_eval_splits(concept: Concept) -> dict[str, Any]:
    """{split stem: activated LabelledDataset} for every eval split of a concept."""
    out = {}
    for jsonl in sorted(concept.eval_dir.glob("*.jsonl")):
        ds = load_jsonl_dataset(jsonl, concept)
        blob = concept.eval_blob_dir / f"{jsonl.stem}-{concept.eval_blob_suffix}.pt"
        out[jsonl.stem] = attach_blob(ds, blob)
    return out


def load_dev_splits(concept: Concept) -> dict[str, Any]:
    """{split stem: activated LabelledDataset} for every dev split of a concept."""
    out = {}
    for jsonl in sorted(concept.dev_dir.glob("*.jsonl")):
        ds = load_jsonl_dataset(jsonl, concept)
        blob = concept.dev_blob_dir / f"{jsonl.stem}-{concept.dev_blob_suffix}.pt"
        out[jsonl.stem] = attach_blob(ds, blob)
    return out


def load_cached_conversations(dataset, cache_dir: Path):
    """Attach per-conversation cached activations to an in-memory dataset.

    Raises if any conversation is missing, since a partially activated pool would
    silently train on a subset. Run ``extract_redteam_activations.py`` first.
    """
    from tuberlens.model import LLMModel

    from agentic_redteam.retrain import (
        _apply_message_transforms,
        _redteam_activation_cache_path,
    )

    dataset = _apply_message_transforms(dataset, COMBINE, CONVERT)
    parts, missing = [], 0
    for messages in dataset.inputs:
        path = _redteam_activation_cache_path(
            cache_dir, messages, MODEL_NAME, LAYER, COMBINE, CONVERT
        )
        if not path.exists():
            missing += 1
            continue
        parts.append(LLMModel.load_activations(path))
    if missing:
        raise FileNotFoundError(
            f"{missing}/{len(dataset)} conversations have no cached activations under "
            f"{cache_dir} — run extract_redteam_activations.py"
        )
    max_len = max(p.activations.shape[1] for p in parts)
    n, dim = len(parts), parts[0].activations.shape[2]
    acts = torch.zeros((n, max_len, dim), dtype=parts[0].activations.dtype)
    mask = torch.zeros((n, max_len), dtype=parts[0].attention_mask.dtype)
    ids = torch.zeros((n, max_len), dtype=parts[0].input_ids.dtype)
    for i, p in enumerate(parts):
        w = p.activations.shape[1]
        acts[i, :w] = p.activations[0]
        mask[i, :w] = p.attention_mask[0]
        ids[i, :w] = p.input_ids[0]
    return dataset.assign(activations=acts, attention_mask=mask, input_ids=ids)


def load_redteam(concept: Concept):
    ds = load_jsonl_dataset(
        concept.redteam_jsonl, concept, field_mapping={"label": "labels"}
    )
    return load_cached_conversations(ds, concept.redteam_cache_dir)


def load_base(concept: Concept):
    ds = load_jsonl_dataset(concept.base_jsonl, concept)
    return load_cached_conversations(ds, concept.redteam_cache_dir)


# --------------------------------------------------------------------------------------
# dataset algebra
# --------------------------------------------------------------------------------------

_PAD_FIELDS = ("activations", "attention_mask", "input_ids")


def pool(datasets: Sequence[Any]):
    """Concatenate activated datasets, padding to the common length, without mutating them.

    ``LabelledDataset.concatenate`` pads each part *in place* and then allocates the result
    on top of the (already padded) parts. Both matter here: the parts are reused across
    every point of the sweep, and the high-stakes pool is tens of GB.
    """
    datasets = [d for d in datasets if d is not None and len(d) > 0]
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    cls = type(datasets[0])
    max_len = max(d.other_fields["activations"].shape[1] for d in datasets)
    total = sum(len(d) for d in datasets)
    out_fields: dict[str, Any] = {}
    for f in _PAD_FIELDS:
        ref = datasets[0].other_fields[f]
        shape = (total, max_len) + tuple(ref.shape[2:])
        buf = torch.zeros(shape, dtype=ref.dtype)
        at = 0
        for d in datasets:
            t = d.other_fields[f]
            buf[at : at + t.shape[0], : t.shape[1]] = t
            at += t.shape[0]
        out_fields[f] = buf
    keys = set(datasets[0].other_fields) - set(_PAD_FIELDS)
    for k in keys:
        if all(k in d.other_fields for d in datasets):
            out_fields[k] = [v for d in datasets for v in d.other_fields[k]]
    return cls(
        inputs=[i for d in datasets for i in d.inputs],
        ids=[i for d in datasets for i in d.ids],
        other_fields=out_fields,
    )


def trim(dataset):
    """Slice the sequence dimension down to the longest real row. Lossless."""
    if dataset is None or len(dataset) == 0:
        return dataset
    mask = dataset.other_fields["attention_mask"]
    real = int(mask.sum(-1).max().item())
    if real >= mask.shape[1]:
        return dataset
    return dataset.assign(
        activations=dataset.other_fields["activations"][:, :real].contiguous(),
        attention_mask=mask[:, :real].contiguous(),
        input_ids=dataset.other_fields["input_ids"][:, :real].contiguous(),
    )


def nbytes(dataset) -> int:
    if dataset is None:
        return 0
    return sum(
        dataset.other_fields[f].element_size() * dataset.other_fields[f].nelement()
        for f in _PAD_FIELDS
        if f in dataset.other_fields
    )


def to_device(datasets: Sequence[Any], device: str = "cuda", *, headroom: float = 0.12):
    """Park activation tensors on the GPU when they fit, else leave them on the host.

    Same values, same indices, same order — the only thing that changes is where
    ``ActivationDataset.__getitems__``'s ``.to(device)`` copies from. See the long note in
    ``retrain._to_device_for_fit`` for why this is worth ~100x on the fit.
    """
    if not torch.cuda.is_available() or device == "cpu":
        return [d for d in datasets], False
    need = sum(nbytes(d) for d in datasets if d is not None)
    free, total = torch.cuda.mem_get_info()
    if need > free * (1.0 - headroom):
        return [d for d in datasets], False
    out = []
    for d in datasets:
        if d is None:
            out.append(None)
            continue
        out.append(
            d.assign(
                **{
                    f: d.other_fields[f].to(device)
                    for f in _PAD_FIELDS
                    if f in d.other_fields
                }
            )
        )
    return out, True


def labels_array(dataset) -> np.ndarray:
    return np.array([lab.to_int() for lab in dataset.labels])


def stratified_indices(dataset, keys: Sequence[Any] | None = None):
    """Group row indices by (label, key) so draws keep the class/split composition."""
    y = labels_array(dataset)
    keys = keys if keys is not None else ["_"] * len(dataset)
    groups: dict[tuple, list[int]] = {}
    for i, (lab, k) in enumerate(zip(y, keys)):
        groups.setdefault((int(lab), k), []).append(i)
    return groups


def stratified_sample(dataset, n: int, rng: np.random.Generator, keys=None) -> list[int]:
    """`n` row indices drawn to preserve the (label, key) composition of `dataset`."""
    if n <= 0:
        return []
    if n >= len(dataset):
        return list(range(len(dataset)))
    groups = stratified_indices(dataset, keys)
    order = sorted(groups)
    shuffled = {k: list(rng.permutation(groups[k])) for k in order}
    quota = {k: len(shuffled[k]) * n / len(dataset) for k in order}
    take = {k: int(np.floor(q)) for k, q in quota.items()}
    short = n - sum(take.values())
    # hand the remainder to the largest fractional parts, deterministically
    for k in sorted(order, key=lambda k: (-(quota[k] - take[k]), k))[:short]:
        take[k] += 1
    return sorted(int(i) for k in order for i in shuffled[k][: take[k]])


# --------------------------------------------------------------------------------------
# fitting + scoring
# --------------------------------------------------------------------------------------


def _spec(hyperparams: dict | None = None):
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType

    return ProbeSpec(name=ProbeType(ARCH), hyperparams=hyperparams or {})


def fit_probe(train_ds, val_ds, concept: Concept, *, seed: int = FIT_SEED,
              hyperparams: dict | None = None, verbose: bool = False):
    """A single (never ensembled) probe, fit exactly the way `retrain` fits one."""
    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.evaluation import seed_everything

    seed_everything(seed)
    return ProbeFactory.build(
        probe_spec=_spec(hyperparams),
        train_dataset=train_ds,
        model_name=MODEL_NAME,
        layer=LAYER,
        validation_dataset=val_ds,
        use_store=False,
        pos_class_label=concept.pos_class_label,
        neg_class_label=concept.neg_class_label,
        probe_description=concept.description,
    )


def val_auroc(probe, val_ds) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels_array(val_ds), probe.predict_proba(val_ds)))


def finetune_probe(probe, train_ds, val_ds, *, seed: int = FIT_SEED,
                   verbose: bool = False):
    """Continue training an already-fit head on new data, keeping the better checkpoint.

    ``PytorchAdamClassifier.train(initialize_model=False)`` is tuberlens' own hook for
    this: it reuses ``self.model`` instead of constructing a fresh one, so the second
    stage starts from the first stage's weights. Its best-checkpoint tracking restarts at
    AUROC 0 though, so a stage that only ever hurts would still return its least-bad
    epoch. Since both stages early-stop against the *same* fixed validation set, the two
    checkpoints are directly comparable, and this keeps whichever is better — i.e. the
    best checkpoint over the union of both stages.
    """
    from agentic_redteam.evaluation import seed_everything

    before_state = copy.deepcopy(probe._classifier.model.state_dict())
    before_auroc = val_auroc(probe, val_ds)

    seed_everything(seed)
    from tuberlens.interfaces.activations import Activation

    probe._classifier.train(
        Activation.from_dataset(train_ds),
        train_ds.labels_torch(),
        validation_activations=Activation.from_dataset(val_ds),
        validation_y=val_ds.labels_torch(),
        initialize_model=False,
    )
    after_auroc = val_auroc(probe, val_ds)
    kept = "finetuned"
    if after_auroc < before_auroc:
        probe._classifier.model.load_state_dict(before_state)
        kept = "stage1"
    return probe, {
        "val_auroc_stage1": before_auroc,
        "val_auroc_finetuned": after_auroc,
        "checkpoint_kept": kept,
        "finetune_best_epoch": probe._classifier.best_epoch,
    }


def score_splits(probe, eval_splits: dict[str, Any], fpr: float = 0.01) -> dict[str, Any]:
    """tuberlens' own per-split metrics, plus the mean over splits."""
    from tuberlens.evaluation import calculate_metrics

    per_split = {}
    for name, ds in eval_splits.items():
        preds = probe.predict_proba(ds)
        per_split[name] = calculate_metrics(labels_array(ds), np.asarray(preds), fpr=fpr)
    mean = {
        k: float(np.mean([m[k] for m in per_split.values()]))
        for k in ("auroc", "accuracy", "tpr_at_fpr")
    }
    return {"per_split": per_split, "mean": mean}


def metrics_from_scores(y: np.ndarray, p: np.ndarray, fpr: float = 0.01) -> dict:
    from tuberlens.evaluation import calculate_metrics

    return calculate_metrics(np.asarray(y), np.asarray(p), fpr=fpr)


# --------------------------------------------------------------------------------------
# result logging
# --------------------------------------------------------------------------------------


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def done_keys(path: Path, key_fields: Sequence[str]) -> set[tuple]:
    """Keys already present in a results JSONL, so a re-run resumes instead of redoing."""
    out: set[tuple] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.add(tuple(row.get(f) for f in key_fields))
    return out


# --------------------------------------------------------------------------------------
# the dev validation / dev training-pool partition
# --------------------------------------------------------------------------------------

DEV_VAL_FRACTION = 0.25
DEV_PARTITION_SEED = 12345


def dev_partition(concept: Concept):
    """Split the dev set once, deterministically, into (validation, training pool).

    Every fit in this analysis — both sweep arms, every point, and the ceiling CV — early-
    stops against the *same* validation slice, and no fit ever trains on it. That is the
    whole reason the points are comparable: tuberlens keeps the best-validation-AUROC
    checkpoint (`pytorch_classifiers.py:358-405`), so a validation set that moved with `N`
    would mean each point's checkpoint was selected against different data — the exact
    failure the experiment 17/18/19 configs introduced `validation.dev_data` to avoid.

    The slice is stratified by (label, dev split) so its composition matches the dev set's,
    and it is a pure function of the dev files' contents plus `DEV_PARTITION_SEED`.

    Returns (val_dataset, train_pool_dataset, split_names_of_train_pool).
    """
    splits = load_dev_splits(concept)
    parts, keys = [], []
    for name, ds in splits.items():
        parts.append(ds)
        keys.extend([name] * len(ds))
    dev = trim(pool(parts))
    dev = dev.assign(dev_split=keys)
    rng = np.random.default_rng(DEV_PARTITION_SEED)
    n_val = int(round(DEV_VAL_FRACTION * len(dev)))
    val_idx = stratified_sample(dev, n_val, rng, keys=keys)
    val_set = set(val_idx)
    train_idx = [i for i in range(len(dev)) if i not in val_set]
    return dev[val_idx], dev[train_idx], [keys[i] for i in train_idx]


def sweep_points(n_pool: int, n_points: int = 10) -> list[int]:
    """`n_points` equidistant sample counts from 0 to `n_pool` inclusive."""
    return [int(round(x)) for x in np.linspace(0, n_pool, n_points)]
