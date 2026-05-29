"""Convert successful red-team JSONL samples back into training data and retrain a probe."""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import torch

from agentic_redteam.persistence import AttemptRecord, JsonlStore

if TYPE_CHECKING:
    from agentic_redteam.config import PreprocessingConfig

# Default fresh probe architecture, mirroring tuberlens' collate_train_evaluate.py.
# Used when a retrain is asked for a fresh architecture without naming a specific one.
DEFAULT_FRESH_PROBE_ARCH = "linear_then_softmax"


def _cpu_unpickle(f: io.BufferedIOBase) -> Any:
    """Unpickle a torch-containing object, forcing all tensors to CPU."""
    _orig = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    try:
        return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = _orig


@dataclass
class RetrainResult:
    new_probe_path: Path
    n_redteam_samples: int
    n_training_samples_total: int


def _records_to_labelled_dataset(records: Iterable[AttemptRecord]):
    """Build a tuberlens LabelledDataset from successful red-team records.

    The label assigned to each sample is the **judge's predicted label** — the
    judge is the source of truth for the class. tuberlens uses canonical
    "positive" / "negative" enum values, so we map the judge's human-readable
    label back to the canonical form using the probe's pos/neg class labels.
    """
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

    inputs: list[list] = []
    ids: list[str] = []
    labels: list[str] = []

    for rec in records:
        if not rec.success:
            continue
        if rec.judge_label == rec.pos_class_label:
            canonical = "positive"
        elif rec.judge_label == rec.neg_class_label:
            canonical = "negative"
        else:
            # Older rows or unparseable judge output: fall back to the error_type-derived
            # truth label (negative for false_positive runs, positive for false_negative).
            canonical = "negative" if rec.error_type == "false_positive" else "positive"
        inputs.append(
            [TLMessage(role=m.role, content=m.content) for m in rec.sample.messages]
        )
        ids.append(f"redteam-{rec.run_id}-{rec.attacker_model}-{len(ids)}")
        labels.append(canonical)

    return LabelledDataset(
        inputs=inputs,
        ids=ids,
        other_fields={"labels": labels},
    )


def _success_human_label(rec: AttemptRecord, pos_label: str, neg_label: str) -> str:
    """The class label (human-readable) for a successful red-team record.

    Uses the judge's label when it matches one of the two classes; otherwise
    falls back to the error_type-implied truth class (negative for
    false_positive runs, positive for false_negative).
    """
    if rec.judge_label == pos_label:
        return pos_label
    if rec.judge_label == neg_label:
        return neg_label
    return neg_label if rec.error_type == "false_positive" else pos_label


def _successes_to_dicts(
    records: Iterable[AttemptRecord], pos_label: str, neg_label: str
) -> list[dict]:
    """Render successful records as ``{inputs, labels}`` dicts for preprocessing."""
    out: list[dict] = []
    for rec in records:
        if not rec.success:
            continue
        out.append(
            {
                "inputs": [
                    {"role": m.role, "content": m.content} for m in rec.sample.messages
                ],
                "labels": _success_human_label(rec, pos_label, neg_label),
            }
        )
    return out


def _dicts_to_labelled_dataset(records: Iterable[dict], pos_label: str, neg_label: str):
    """Build a tuberlens LabelledDataset from ``{inputs, labels}`` dicts.

    Maps the human-readable label back to tuberlens' canonical
    "positive"/"negative" enum values. Records whose label is neither class are
    skipped.
    """
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

    inputs: list[list] = []
    ids: list[str] = []
    labels: list[str] = []
    for i, rec in enumerate(records):
        label = rec.get("labels")
        if label == pos_label:
            canonical = "positive"
        elif label == neg_label:
            canonical = "negative"
        else:
            continue
        msgs = rec.get("inputs", [])
        inputs.append(
            [
                TLMessage(role=str(m.get("role", "user")), content=str(m.get("content", "")))
                for m in msgs
            ]
        )
        ids.append(f"redteam-{i}")
        labels.append(canonical)

    return LabelledDataset(inputs=inputs, ids=ids, other_fields={"labels": labels})


def _build_redteam_dataset(
    successes: list[AttemptRecord],
    pos_label: str,
    neg_label: str,
    preprocessing: "PreprocessingConfig | None",
    contrastive_cache_path: str | Path | None,
    verbose: bool,
):
    """Convert red-team successes into a LabelledDataset, optionally preprocessing.

    With no preprocessing config this is the plain success→dataset conversion.
    With one, it mirrors the collation step of tuberlens' pipeline applied to the
    "extra" data: drop confounders (``filter_dataset``) then add contrastive
    pairs (``generate_contrastive_dataset``).
    """
    if preprocessing is None or not successes:
        return _records_to_labelled_dataset(successes)

    from agentic_redteam.preprocessing import (
        filter_dataset,
        generate_contrastive_dataset,
    )

    dicts = _successes_to_dicts(successes, pos_label, neg_label)
    before = len(dicts)
    dicts = filter_dataset(
        dicts,
        pos_class_label=pos_label,
        filter_percentile=preprocessing.filter_percentile,
    )
    if verbose:
        print(f"filter_dataset: {before} → {len(dicts)} records after dropping confounders")
    dicts = generate_contrastive_dataset(
        dicts,
        pos_class_label=pos_label,
        neg_class_label=neg_label,
        provider=preprocessing.provider,
        model=preprocessing.model,
        max_concurrent=preprocessing.max_concurrent,
        max_tokens=preprocessing.max_tokens,
        cache_path=contrastive_cache_path,
    )
    return _dicts_to_labelled_dataset(dicts, pos_label, neg_label)


def retrain_probe(
    jsonl_path: str | Path | list[str | Path],
    base_probe_path: str | Path,
    base_training_data_path: str | Path | None,
    new_probe_path: str | Path,
    layer: int | None = None,
    probe_spec: "ProbeSpec | str | None" = None,
    preprocessing: "PreprocessingConfig | None" = None,
    contrastive_cache_path: str | Path | None = None,
    test_size: float = 0.2,
    split_field: str | None = None,
    verbose: bool = True,
) -> RetrainResult:
    """Train a fresh probe using `base_training_data_path` ∪ red-team successes from `jsonl_path`.

    The new probe inherits the architecture (`probe_spec`) and metadata
    (`pos_class_label`, `neg_class_label`, `description`, `model_name`, `layer`) from the
    base probe at `base_probe_path` — so retraining stays apples-to-apples with what the
    red-team agent attacked.

    Args:
        jsonl_path: Path (or list of paths) to red-team JSONL logs produced by run_redteam.
            When multiple error types produce separate files, pass all of them here.
        base_probe_path: Existing pickled probe; used to inherit architecture and metadata.
        base_training_data_path: JSONL/CSV consumed by tuberlens.LabelledDataset.load_from. If
            None, the new probe is trained on red-team successes alone.
        new_probe_path: Where to pickle the retrained probe.
        layer: Layer to probe. If None, reuse base_probe.layer.
        probe_spec: Architecture for the retrained probe. If None (default), inherit the
            base probe's architecture via `_infer_probe_spec` (apples-to-apples). Pass a
            `ProbeSpec` to train a specific fresh architecture, or a ProbeType name string
            (e.g. "linear_then_softmax") for a fresh probe of that type with default
            hyperparams.
        preprocessing: When provided, filter_dataset + generate_contrastive_dataset are
            applied to the red-team successes (mirroring tuberlens' collation step on the
            "extra" data) before concatenation with the base training data.
        contrastive_cache_path: Disk cache for generated contrastive pairs (per source
            conversation), so accumulating successes aren't re-generated every iteration.
        test_size: Fraction held out for validation via tuberlens.create_train_test_split.
        split_field: Optional field to keep grouped together when splitting (passed to
            create_train_test_split).
        verbose: Forwarded to tuberlens.train_probe.
    """
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.training import train_probe
    from tuberlens.utils import create_train_test_split

    if isinstance(jsonl_path, (str, Path)):
        jsonl_paths = [Path(jsonl_path)]
    else:
        jsonl_paths = [Path(p) for p in jsonl_path]

    base_probe_path = Path(base_probe_path)
    new_probe_path = Path(new_probe_path)
    new_probe_path.parent.mkdir(parents=True, exist_ok=True)

    with base_probe_path.open("rb") as f:
        base_probe = _cpu_unpickle(f)

    if base_probe.model_name is None or base_probe.layer is None:
        raise ValueError("Base probe is missing model_name or layer; cannot retrain.")

    pos_class_label = getattr(base_probe, "pos_class_label", "positive")
    neg_class_label = getattr(base_probe, "neg_class_label", "negative")
    probe_description = getattr(base_probe, "description", None)

    all_successes: list[AttemptRecord] = []
    for jp in jsonl_paths:
        if jp.exists():
            store = JsonlStore(path=jp)
            all_successes.extend(store.iter_successes())
    if verbose:
        print(f"Red-team successes loaded: {len(all_successes)}")

    redteam_dataset = _build_redteam_dataset(
        all_successes,
        pos_class_label,
        neg_class_label,
        preprocessing,
        contrastive_cache_path,
        verbose,
    )
    n_redteam = len(redteam_dataset)
    if verbose and preprocessing is not None:
        print(f"Red-team training samples after preprocessing: {n_redteam}")

    if base_training_data_path is not None:
        base_dataset = LabelledDataset.load_from(
            Path(base_training_data_path),
            pos_class_label=pos_class_label,
            neg_class_label=neg_class_label,
        )
        if n_redteam > 0:
            full_dataset = LabelledDataset.concatenate([base_dataset, redteam_dataset])
        else:
            full_dataset = base_dataset
    else:
        if n_redteam == 0:
            raise ValueError(
                "No red-team successes and no base_training_data_path provided — nothing to train on."
            )
        full_dataset = redteam_dataset

    if verbose:
        print(f"Total samples before split: {len(full_dataset)}")
        full_dataset.print_label_distribution()

    train_dataset, validation_dataset = create_train_test_split(
        full_dataset, test_size=test_size, split_field=split_field
    )
    if verbose:
        print(
            f"Train/validation split: {len(train_dataset)} train, "
            f"{len(validation_dataset)} validation"
        )

    layer_used = layer if layer is not None else int(base_probe.layer)

    # Resolve the architecture for the new probe. Default (None): inherit the base probe's
    # architecture so the retrain stays apples-to-apples. A ProbeType name string or an
    # explicit ProbeSpec requests a fresh architecture instead.
    if probe_spec is None:
        probe_spec = _infer_probe_spec(base_probe)
    else:
        probe_spec = _coerce_probe_spec(probe_spec)

    new_probe = train_probe(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        model_name=base_probe.model_name,
        layer=layer_used,
        probe_spec=probe_spec,
        verbose=verbose,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
    )

    with new_probe_path.open("wb") as f:
        pickle.dump(new_probe, f)
    if verbose:
        print(f"Saved retrained probe to {new_probe_path}")

    return RetrainResult(
        new_probe_path=new_probe_path,
        n_redteam_samples=n_redteam,
        n_training_samples_total=len(full_dataset),
    )


def train_initial_probe(
    base_training_data_path: str | Path,
    model_name: str,
    layer: int,
    new_probe_path: str | Path,
    pos_class_label: str,
    neg_class_label: str,
    probe_description: str | None = None,
    probe_spec: "ProbeSpec | str | None" = None,
    test_size: float = 0.2,
    split_field: str | None = None,
    verbose: bool = True,
) -> Path:
    """Train the first probe from base training data alone (no base probe to inherit from).

    Mirrors tuberlens' collate_train_evaluate.train_high_stakes_probe, but the concept
    (class labels, description, architecture) is supplied by the caller instead of
    hardcoded. `probe_spec` may be a `ProbeSpec`, a ProbeType name string, or None — None
    falls back to `DEFAULT_FRESH_PROBE_ARCH` ("linear_then_softmax").

    Args:
        base_training_data_path: JSONL/CSV consumed by tuberlens.LabelledDataset.load_from.
        model_name: tuberlens model name/key to probe.
        layer: Layer to probe.
        new_probe_path: Where to pickle the trained probe.
        pos_class_label / neg_class_label: Class labels; also used to load the dataset.
        probe_description: Optional human-readable probe description.
        probe_spec: Architecture (ProbeSpec | ProbeType name | None).
        test_size: Fraction held out for validation via tuberlens.create_train_test_split.
        split_field: Optional field to keep grouped together when splitting.
        verbose: Forwarded to tuberlens.train_probe.
    """
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.training import train_probe
    from tuberlens.utils import create_train_test_split

    new_probe_path = Path(new_probe_path)
    new_probe_path.parent.mkdir(parents=True, exist_ok=True)

    full_dataset = LabelledDataset.load_from(
        Path(base_training_data_path),
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
    )
    if verbose:
        print(f"Initial samples before split: {len(full_dataset)}")
        full_dataset.print_label_distribution()

    train_dataset, validation_dataset = create_train_test_split(
        full_dataset, test_size=test_size, split_field=split_field
    )
    if verbose:
        print(
            f"Train/validation split: {len(train_dataset)} train, "
            f"{len(validation_dataset)} validation"
        )

    spec = _coerce_probe_spec(probe_spec or DEFAULT_FRESH_PROBE_ARCH)

    probe = train_probe(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        model_name=model_name,
        layer=layer,
        probe_spec=spec,
        verbose=verbose,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
    )

    with new_probe_path.open("wb") as f:
        pickle.dump(probe, f)
    if verbose:
        print(f"Saved initial probe to {new_probe_path}")
    return new_probe_path


def _coerce_probe_spec(probe_spec):
    """ProbeSpec | str -> ProbeSpec. A string is treated as a ProbeType name with default hyperparams."""
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType

    if isinstance(probe_spec, str):
        return ProbeSpec(name=ProbeType(probe_spec), hyperparams={})
    return probe_spec


def _infer_probe_spec(base_probe):
    """Infer a ProbeSpec from a loaded probe object so we can train a fresh one of the same kind."""
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.probes.pytorch_modules import (
        AttnLite,
        LinearThenLast,
        LinearThenMax,
        LinearThenMean,
        LinearThenRollingMax,
        LinearThenSoftmax,
        MeanThenLinear,
    )

    classifier = getattr(base_probe, "_classifier", None)

    if classifier is None:
        # SklearnProbe carries hyper_params directly
        hyper = getattr(base_probe, "hyper_params", None) or {}
        return ProbeSpec(name=ProbeType.sklearn, hyperparams=hyper)

    arch = getattr(classifier, "probe_architecture", None)
    arch_to_type = {
        AttnLite: ProbeType.attention,
        MeanThenLinear: ProbeType.pre_mean,
        LinearThenMean: ProbeType.linear_then_mean,
        LinearThenMax: ProbeType.linear_then_max,
        LinearThenSoftmax: ProbeType.linear_then_softmax,
        LinearThenRollingMax: ProbeType.linear_then_rolling_max,
        LinearThenLast: ProbeType.linear_then_last,
    }
    if arch is not None and arch in arch_to_type:
        hyper = getattr(base_probe, "hyper_params", None) or {}
        return ProbeSpec(name=arch_to_type[arch], hyperparams=hyper)

    # Difference-of-means and LDA classifiers
    if hasattr(classifier, "use_lda"):
        hyper = getattr(base_probe, "hyper_params", None) or {}
        return ProbeSpec(
            name=ProbeType.lda if classifier.use_lda else ProbeType.difference_of_means,
            hyperparams=hyper,
        )

    raise ValueError(
        f"Could not infer ProbeSpec from base probe {type(base_probe).__name__}; "
        f"specify a ProbeSpec explicitly."
    )
