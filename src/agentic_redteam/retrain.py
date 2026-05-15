"""Convert successful red-team JSONL samples back into training data and retrain a probe."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agentic_redteam.persistence import AttemptRecord, Conversation, JsonlStore


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


def retrain_probe(
    jsonl_path: str | Path,
    base_probe_path: str | Path,
    base_training_data_path: str | Path | None,
    new_probe_path: str | Path,
    layer: int | None = None,
    validation_dataset_path: str | Path | None = None,
    verbose: bool = True,
) -> RetrainResult:
    """Train a fresh probe using `base_training_data_path` ∪ red-team successes from `jsonl_path`.

    The new probe inherits the architecture (`probe_spec`) and metadata
    (`pos_class_label`, `neg_class_label`, `description`, `model_name`, `layer`) from the
    base probe at `base_probe_path` — so retraining stays apples-to-apples with what the
    red-team agent attacked.

    Args:
        jsonl_path: Path to the red-team JSONL log produced by run_redteam.
        base_probe_path: Existing pickled probe; used to inherit architecture and metadata.
        base_training_data_path: JSONL/CSV consumed by tuberlens.LabelledDataset.load_from. If
            None, the new probe is trained on red-team successes alone.
        new_probe_path: Where to pickle the retrained probe.
        layer: Layer to probe. If None, reuse base_probe.layer.
        validation_dataset_path: Optional held-out dataset for training validation.
        verbose: Forwarded to tuberlens.train_probe.
    """
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.training import train_probe

    jsonl_path = Path(jsonl_path)
    base_probe_path = Path(base_probe_path)
    new_probe_path = Path(new_probe_path)
    new_probe_path.parent.mkdir(parents=True, exist_ok=True)

    with base_probe_path.open("rb") as f:
        base_probe = pickle.load(f)

    if base_probe.model_name is None or base_probe.layer is None:
        raise ValueError("Base probe is missing model_name or layer; cannot retrain.")

    pos_class_label = getattr(base_probe, "pos_class_label", "positive")
    neg_class_label = getattr(base_probe, "neg_class_label", "negative")
    probe_description = getattr(base_probe, "description", None)

    store = JsonlStore(path=jsonl_path)
    successes = list(store.iter_successes())
    redteam_dataset = _records_to_labelled_dataset(successes)
    n_redteam = len(redteam_dataset)
    if verbose:
        print(f"Red-team successes loaded: {n_redteam}")

    if base_training_data_path is not None:
        base_dataset = LabelledDataset.load_from(
            Path(base_training_data_path),
            pos_class_label=pos_class_label,
            neg_class_label=neg_class_label,
        )
        if n_redteam > 0:
            train_dataset = LabelledDataset.concatenate([base_dataset, redteam_dataset])
        else:
            train_dataset = base_dataset
    else:
        if n_redteam == 0:
            raise ValueError(
                "No red-team successes and no base_training_data_path provided — nothing to train on."
            )
        train_dataset = redteam_dataset

    if verbose:
        print(f"Total training samples: {len(train_dataset)}")
        train_dataset.print_label_distribution()

    validation_dataset = None
    if validation_dataset_path is not None:
        validation_dataset = LabelledDataset.load_from(
            Path(validation_dataset_path),
            pos_class_label=pos_class_label,
            neg_class_label=neg_class_label,
        )

    layer_used = layer if layer is not None else int(base_probe.layer)

    # Reuse the architecture of the base probe by inferring its ProbeType from the class.
    # If the base was a sklearn probe this still maps to ProbeType.sklearn; for pytorch
    # probes we need the class name → ProbeType mapping.
    probe_spec = _infer_probe_spec(base_probe)

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
        n_training_samples_total=len(train_dataset),
    )


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
