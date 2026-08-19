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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ca_data as D  # noqa: E402

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

    @property
    def redteam_pool_blob(self) -> Path:
        return ACTS_ROOT / self.name / "redteam_base_pool.pt"


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


def _redteam_conv_sources(dataset, cache_dir: Path):
    """Per-conversation cache paths for an in-memory dataset, in row order."""
    from agentic_redteam.retrain import (
        _apply_message_transforms,
        _redteam_activation_cache_path,
    )

    dataset = _apply_message_transforms(dataset, COMBINE, CONVERT)
    paths = [
        _redteam_activation_cache_path(
            cache_dir, messages, MODEL_NAME, LAYER, COMBINE, CONVERT
        )
        for messages in dataset.inputs
    ]
    return dataset, paths


def eval_sources(concept: Concept) -> dict[str, D.BlobSource]:
    """{split stem: BlobSource} over the published eval blobs. Nothing is read yet."""
    out = {}
    for jsonl in sorted(concept.eval_dir.glob("*.jsonl")):
        ds = load_jsonl_dataset(jsonl, concept)
        blob = concept.eval_blob_dir / f"{jsonl.stem}-{concept.eval_blob_suffix}.pt"
        out[jsonl.stem] = D.BlobSource(jsonl.stem, blob, ds)
    return out


def dev_source(concept: Concept) -> D.MultiSource:
    """The whole dev set as one source, in `sorted(*.jsonl)` order.

    That order matters: it is the order `retrain._load_dev_dataset` concatenates the dev
    splits in, so a row index here means the same row it would mean to a real run.
    """
    parts, metas, keys = [], [], []
    for jsonl in sorted(concept.dev_dir.glob("*.jsonl")):
        ds = load_jsonl_dataset(jsonl, concept)
        blob = concept.dev_blob_dir / f"{jsonl.stem}-{concept.dev_blob_suffix}.pt"
        parts.append(D.BlobSource(jsonl.stem, blob, ds))
        metas.append(ds)
        keys.extend([jsonl.stem] * len(ds))
    cls = type(metas[0])
    merged = cls(
        inputs=[i for m in metas for i in m.inputs],
        ids=[i for m in metas for i in m.ids],
        other_fields={
            "labels": [v for m in metas for v in m.other_fields["labels"]],
            "dev_split": keys,
        },
    )
    return D.MultiSource("dev", parts, merged)


def redteam_source(concept: Concept) -> D.BlobSource:
    """base + red-team as one consolidated blob, built on first use from the row cache."""
    path = concept.redteam_pool_blob
    metas = []
    for jsonl, mapping in (
        (concept.base_jsonl, None),
        (concept.redteam_jsonl, {"label": "labels"}),
    ):
        ds = load_jsonl_dataset(jsonl, concept, field_mapping=mapping)
        ds, _ = _redteam_conv_sources(ds, concept.redteam_cache_dir)
        metas.append(ds)
    cls = type(metas[0])
    origin = ["base"] * len(metas[0]) + ["redteam"] * len(metas[1])
    merged = cls(
        inputs=[i for m in metas for i in m.inputs],
        ids=[i for m in metas for i in m.ids],
        other_fields={
            "labels": [v for m in metas for v in m.other_fields["labels"]],
            "origin": origin,
        },
    )
    if not path.exists():
        build_redteam_pool_blob(concept, merged)
    return D.BlobSource("redteam_base", path, merged)


def build_redteam_pool_blob(concept: Concept, merged) -> None:
    """Consolidate the per-conversation red-team/base cache into one mmap-able blob.

    The extraction writes one `.pt` per conversation (that is what makes it resumable and
    what lets a conversation first seen in iteration k be reused by every later retrain).
    Reading ~900 of those on every pool build is pure syscall overhead, so they are folded
    once into a single file, trimmed to the widest real row.
    """
    from tuberlens.model import LLMModel

    _, paths = _redteam_conv_sources(merged, concept.redteam_cache_dir)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(paths)} conversations have no cached activations under "
            f"{concept.redteam_cache_dir} — run extract_redteam_activations.py"
        )
    loaded = [LLMModel.load_activations(p) for p in paths]
    width = max(int(a.attention_mask.sum(-1).max().item()) for a in loaded)
    dim = loaded[0].activations.shape[2]
    acts = torch.zeros((len(loaded), width, dim), dtype=loaded[0].activations.dtype)
    mask = torch.zeros((len(loaded), width), dtype=loaded[0].attention_mask.dtype)
    ids = torch.zeros((len(loaded), width), dtype=loaded[0].input_ids.dtype)
    for i, a in enumerate(loaded):
        w = min(width, a.activations.shape[1])
        acts[i, :w] = a.activations[0, :w]
        mask[i, :w] = a.attention_mask[0, :w]
        ids[i, :w] = a.input_ids[0, :w]
    concept.redteam_pool_blob.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"activations": acts, "attention_mask": mask, "input_ids": ids,
         "model_name": MODEL_NAME, "layer": LAYER},
        concept.redteam_pool_blob,
    )


def nbytes(dataset) -> int:
    if dataset is None:
        return 0
    return sum(
        dataset.other_fields[f].element_size() * dataset.other_fields[f].nelement()
        for f in D.PAD_FIELDS
        if f in dataset.other_fields
    )


def to_device(datasets, device: str = "cuda", *, headroom: float = 0.15):
    """Park activation tensors on the GPU when they fit, else leave them on the host.

    Same values, same indices, same order — the only thing that changes is where
    `ActivationDataset.__getitems__`'s `.to(device)` copies from. See the long note in
    `retrain._to_device_for_fit` for why this is worth ~100x on a fit, and why it has to
    fall back rather than fail: the high-stakes training pool outgrows a 24 GB card at the
    top of the sweep, and a slow fit beats a dead one.
    """
    if not torch.cuda.is_available() or device == "cpu":
        return list(datasets), False
    need = sum(nbytes(d) for d in datasets if d is not None)
    free, _total = torch.cuda.mem_get_info()
    if need > free * (1.0 - headroom):
        return list(datasets), False
    out = []
    for d in datasets:
        if d is None:
            out.append(None)
            continue
        out.append(
            d.assign(
                **{f: d.other_fields[f].to(device) for f in D.PAD_FIELDS
                   if f in d.other_fields}
            )
        )
    return out, True


def labels_array(dataset) -> np.ndarray:
    return np.array([lab.to_int() for lab in dataset.labels])


def source_labels(source) -> np.ndarray:
    return np.array([lab.to_int() for lab in source.dataset.labels])


def stratified_indices(labels, keys=None):
    """Group row indices by (label, key) so draws keep the class/split composition."""
    keys = keys if keys is not None else ["_"] * len(labels)
    groups: dict[tuple, list[int]] = {}
    for i, (lab, k) in enumerate(zip(labels, keys)):
        groups.setdefault((int(lab), k), []).append(i)
    return groups


def stratified_sample(labels, n: int, rng, keys=None) -> list[int]:
    """`n` row indices drawn to preserve the (label, key) composition."""
    total = len(labels)
    if n <= 0:
        return []
    if n >= total:
        return list(range(total))
    groups = stratified_indices(labels, keys)
    order = sorted(groups)
    shuffled = {k: list(rng.permutation(groups[k])) for k in order}
    quota = {k: len(shuffled[k]) * n / total for k in order}
    take = {k: int(np.floor(q)) for k, q in quota.items()}
    short = n - sum(take.values())
    for k in sorted(order, key=lambda k: (-(quota[k] - take[k]), str(k)))[:short]:
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
    """Split the dev set once, deterministically, into (validation rows, training-pool rows).

    Every fit in this analysis — both sweep arms, every point, and the ceiling CV — early-
    stops against the *same* validation slice, and no fit ever trains on it. That is what
    makes the points comparable: tuberlens keeps the best-validation-AUROC checkpoint
    (`pytorch_classifiers.py:358-405`), so a validation set that moved with `N` would mean
    each point's checkpoint was selected against different data — the exact failure the
    experiment 17/18/19 configs introduced `validation.dev_data` to avoid.

    The slice is stratified by (label, dev split) so its composition matches the dev set's,
    and it is a pure function of the dev files plus `DEV_PARTITION_SEED`.

    Returns (source, val_idx, pool_idx).
    """
    src = dev_source(concept)
    labels = source_labels(src)
    keys = list(src.dataset.other_fields["dev_split"])
    rng = np.random.default_rng(DEV_PARTITION_SEED)
    n_val = int(round(DEV_VAL_FRACTION * len(src)))
    val_idx = stratified_sample(labels, n_val, rng, keys=keys)
    taken = set(val_idx)
    pool_idx = [i for i in range(len(src)) if i not in taken]
    return src, val_idx, pool_idx


def sweep_points(n_pool: int, n_points: int = 10) -> list[int]:
    """`n_points` equidistant sample counts from 0 to `n_pool` inclusive."""
    return [int(round(x)) for x in np.linspace(0, n_pool, n_points)]


def free_gpu() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# the ragged fit path (see ca_fit for why it is exact)
# --------------------------------------------------------------------------------------

USE_FAST_FIT = True


def hyperparams() -> dict:
    from tuberlens.interfaces.probes import ProbeType

    return dict(ProbeType(ARCH).default_hyperparams)


def fit_probe_fast(train_ds, val_ds, concept: Concept, *, seed: int = FIT_SEED,
                   verbose: bool = False):
    """Fit one probe over ragged, GPU-resident activations. Same arithmetic as `fit_probe`."""
    import ca_fit as F
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax

    from agentic_redteam.evaluation import seed_everything

    hp = hyperparams()
    train = F.RaggedActivations.from_dataset(train_ds)
    val = F.RaggedActivations.from_dataset(val_ds) if val_ds is not None else None
    seed_everything(seed)
    model, info = F.train_head(train, val, hp, arch=LinearThenSoftmax, verbose=verbose)
    probe = F.wrap_probe(
        model, hp, model_name=MODEL_NAME, layer=LAYER,
        pos_class_label=concept.pos_class_label,
        neg_class_label=concept.neg_class_label,
        description=concept.description, best_epoch=info["best_epoch"],
    )
    del train, val
    free_gpu()
    return probe


def finetune_probe_fast(probe, train_ds, val_ds, *, seed: int = FIT_SEED,
                        verbose: bool = False):
    import ca_fit as F

    from agentic_redteam.evaluation import seed_everything

    hp = hyperparams()
    train = F.RaggedActivations.from_dataset(train_ds)
    val = F.RaggedActivations.from_dataset(val_ds)
    seed_everything(seed)
    model, info = F.finetune_head(probe._classifier.model, train, val, hp, verbose=verbose)
    probe._classifier.model = model
    probe._classifier.best_epoch = info["finetune_best_epoch"]
    del train, val
    free_gpu()
    return probe, info


def fit(train_ds, val_ds, concept: Concept, *, seed: int = FIT_SEED, verbose: bool = False):
    if USE_FAST_FIT:
        return fit_probe_fast(train_ds, val_ds, concept, seed=seed, verbose=verbose)
    return fit_probe(train_ds, val_ds, concept, seed=seed, verbose=verbose)


def finetune(probe, train_ds, val_ds, *, seed: int = FIT_SEED, verbose: bool = False):
    if USE_FAST_FIT:
        return finetune_probe_fast(probe, train_ds, val_ds, seed=seed, verbose=verbose)
    return finetune_probe(probe, train_ds, val_ds, seed=seed, verbose=verbose)
