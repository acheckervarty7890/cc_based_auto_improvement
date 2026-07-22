"""Convert successful red-team JSONL samples back into training data and retrain a probe."""

from __future__ import annotations

import hashlib
import io
import json
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


def _split_unit_interval(messages, split_value, seed: int) -> float:
    """Map a sample to a deterministic value in [0, 1) for train/val assignment.

    The value depends only on the sample's own content (or, when a split_field is
    used, on `split_value`) plus `seed` — never on dataset size or ordering — so a
    given conversation always hashes to the same number across iterations.
    """
    if split_value is not None:
        basis = str(split_value)
    else:
        basis = json.dumps(
            [{"role": m.role, "content": m.content} for m in messages],
            sort_keys=True,
            ensure_ascii=False,
        )
    digest = hashlib.sha256(f"{seed}:{basis}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1_000_000) / 1_000_000


def stable_train_test_split(
    dataset, test_size: float = 0.2, split_field: str | None = None, seed: int = 0
):
    """Deterministic, content-addressed train/val split with no reshuffling.

    Drop-in replacement for tuberlens' ``create_train_test_split``, which draws a
    fresh split from the global numpy RNG every call. Here each sample's side is a
    pure function of its own content (or its ``split_field`` value) and ``seed``,
    independent of dataset size or order — so the base training samples land on the
    same side every iteration even as red-team successes accumulate, and the
    validation set stays comparable across iterations. Class balance is preserved
    in expectation because the threshold is uniform across all samples. When
    ``split_field`` is given, all samples sharing a field value go to the same side
    (the value, not the conversation, is hashed), matching the original's grouping.
    """
    split_values = (
        dataset.other_fields.get(split_field) if split_field is not None else None
    )
    if split_field is not None and split_values is None:
        raise ValueError(
            f"split_field {split_field!r} not found in dataset.other_fields"
        )

    train_indices: list[int] = []
    test_indices: list[int] = []
    for i, messages in enumerate(dataset.inputs):
        sv = split_values[i] if split_values is not None else None
        if _split_unit_interval(messages, sv, seed) < test_size:
            test_indices.append(i)
        else:
            train_indices.append(i)

    return dataset[train_indices], dataset[test_indices]


def stable_fraction_subsample(dataset, fraction: float, seed: int = 0):
    """Deterministically keep a random ``fraction`` of a dataset's samples.

    Content-addressed like ``stable_train_test_split``: each sample is kept iff a
    uniform hash of *its own content* (namespaced by ``seed``) falls below
    ``fraction``. So the selected subset is a pure function of the data + seed —
    independent of dataset size or order — which means the base training samples
    land identically every iteration (matching the repo's cache-correctness
    convention) and class balance is preserved in expectation. The hash is
    namespaced (``frac:{seed}``) so the draw is independent of the train/val split,
    which hashes on the bare ``seed``. ``fraction >= 1.0`` is a no-op.
    """
    if fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError(f"base_data_fraction must be in (0, 1]; got {fraction}")
    keep = [
        i
        for i, messages in enumerate(dataset.inputs)
        if _split_unit_interval(messages, None, seed=f"frac:{seed}") < fraction
    ]
    return dataset[keep]


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


def _apply_message_transforms(
    dataset,
    combine_consecutive_messages: bool,
    convert_tool_to_assistant: bool,
):
    """Apply tuberlens' message-loader transforms to an already-built dataset.

    ``LabelledDataset.load_from`` applies these to data read from disk, but the
    red-team successes are constructed in-memory from ``Message`` objects, so we
    apply the same transforms here for parity. Order matches ``load_from``:
    ``convert_tool_to_assistant`` first (so it doesn't create consecutive
    assistant turns), then ``combine_consecutive_messages``.
    """
    if not (combine_consecutive_messages or convert_tool_to_assistant):
        return dataset
    from tuberlens.interfaces.dataset import LabelledDataset

    new_inputs = []
    for dialogue in dataset.inputs:
        if convert_tool_to_assistant:
            dialogue = LabelledDataset._convert_tool_to_assistant(dialogue)
        if combine_consecutive_messages:
            dialogue = LabelledDataset._combine_consecutive_messages(dialogue)
        new_inputs.append(dialogue)
    return LabelledDataset(
        inputs=new_inputs, ids=dataset.ids, other_fields=dataset.other_fields
    )


def _base_activation_cache_paths(
    cache_dir: str | Path,
    base_data_path: str | Path,
    model_name: str,
    layer: int,
    seed: int,
    test_size: float,
    split_field: str | None,
    combine_consecutive_messages: bool,
    convert_tool_to_assistant: bool,
    base_data_fraction: float = 1.0,
) -> tuple[Path, Path]:
    """Return (train, val) cache paths for the *base* split's activations.

    The base train/val membership is a deterministic function of the base data
    file, ``seed``, ``test_size``, ``split_field`` and ``base_data_fraction`` (via
    stable_fraction_subsample + stable_train_test_split), and the activations
    themselves additionally depend on ``model_name``, ``layer`` and the
    message-loader transforms — so all of those are folded into the cache key.
    tuberlens' ``get_activations`` loads a saved blob *by path without validating
    the inputs*, so any change that would alter the base split or its activations
    (including subsampling a different fraction) must change this key, or stale
    activations would be silently reused.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    h.update(Path(base_data_path).read_bytes())
    h.update(
        f"|model={model_name}|layer={layer}|seed={seed}"
        f"|test_size={test_size}|split_field={split_field}"
        f"|fraction={base_data_fraction}"
        f"|combine={combine_consecutive_messages}|convert={convert_tool_to_assistant}".encode(
            "utf-8"
        )
    )
    key = h.hexdigest()[:16]
    safe_model = model_name.replace("/", "_")
    stem = f"base_acts_{safe_model}_L{layer}_{key}"
    return cache_dir / f"{stem}_train.pt", cache_dir / f"{stem}_val.pt"


def _redteam_activation_cache_path(
    cache_dir: str | Path,
    messages,
    model_name: str,
    layer: int,
    combine_consecutive_messages: bool,
    convert_tool_to_assistant: bool,
) -> Path:
    """Per-sample activation cache path for one red-team conversation.

    Unlike the base split — a fixed file cached as a single blob keyed on the whole
    file (`_base_activation_cache_paths`) — the red-team set *grows every iteration*,
    so a whole-set blob keyed on the set's contents would get a fresh key each time
    and never hit. Instead each conversation is cached individually, keyed on its own
    (already-transformed) message content plus ``model_name`` / ``layer`` and the
    message-loader transforms. A conversation first seen in iteration k is therefore
    computed once and reused by every later retrain. This is correct because the
    underlying LLM is frozen across iterations (only the probe head is retrained), so
    a conversation's layer activation is identical regardless of which iteration
    computes it. As with the base cache, load is *by path without validating inputs*,
    so everything that changes the activation is folded into the key.
    """
    basis = json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
    )
    h = hashlib.sha256(
        f"model={model_name}|layer={layer}"
        f"|combine={combine_consecutive_messages}|convert={convert_tool_to_assistant}"
        f"|{basis}".encode("utf-8")
    ).hexdigest()
    safe_model = model_name.replace("/", "_")
    return Path(cache_dir) / f"redteam_acts_{safe_model}_L{layer}" / f"{h[:32]}.pt"


def _activate_redteam_cached(
    dataset,
    cache_dir: str | Path | None,
    model_name: str,
    layer: int,
    combine_consecutive_messages: bool,
    convert_tool_to_assistant: bool,
    get_model,
    verbose: bool,
):
    """Activate red-team samples with a per-conversation disk cache.

    Mirrors ``_activate``'s load-or-compute-and-save logic but at per-conversation
    granularity (see ``_redteam_activation_cache_path`` for why a whole-set blob
    would never hit): samples accumulated in earlier iterations are loaded from disk
    and only newly-seen conversations are forwarded through the model (in one batched
    call). Per-row blobs use the same dict layout tuberlens' ``get_activations``
    writes, so ``LLMModel.load_activations`` reads them back; ``LabelledDataset``'s
    pad-aware ``concatenate`` re-pads the loaded + freshly-computed parts to a common
    length, so storing each row at its own width is safe. Returns a ``LabelledDataset``
    with activations assigned, or ``None`` for an empty/absent input. With
    ``cache_dir=None`` it degrades to a plain batched compute (matching the previous
    uncached behaviour).
    """
    if dataset is None or len(dataset) == 0:
        return None
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.model import LLMModel

    def _compute(ds):
        acts = get_model().get_activations(ds.inputs, layer=layer, show_progress=verbose)
        return acts, ds.assign(
            activations=acts.activations,
            attention_mask=acts.attention_mask,
            input_ids=acts.input_ids,
        )

    if cache_dir is None:
        return _compute(dataset)[1]

    paths = [
        _redteam_activation_cache_path(
            cache_dir,
            msgs,
            model_name,
            layer,
            combine_consecutive_messages,
            convert_tool_to_assistant,
        )
        for msgs in dataset.inputs
    ]
    cached_idx = [i for i, p in enumerate(paths) if p.exists()]
    uncached_idx = [i for i, p in enumerate(paths) if not p.exists()]

    parts = []
    for i in cached_idx:
        act = LLMModel.load_activations(paths[i])
        parts.append(
            dataset[i : i + 1].assign(
                activations=act.activations,
                attention_mask=act.attention_mask,
                input_ids=act.input_ids,
            )
        )
    if uncached_idx:
        acts, computed = _compute(dataset[uncached_idx])
        for j, i in enumerate(uncached_idx):
            p = paths[i]
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "activations": acts.activations[j : j + 1].clone(),
                    "attention_mask": acts.attention_mask[j : j + 1].clone(),
                    "input_ids": acts.input_ids[j : j + 1].clone(),
                    "layer": layer,
                    "model_name": model_name,
                },
                p,
            )
        parts.append(computed)

    if verbose:
        print(
            f"Red-team activations: {len(cached_idx)} loaded from cache, "
            f"{len(uncached_idx)} computed fresh"
        )
    return parts[0] if len(parts) == 1 else LabelledDataset.concatenate(parts)


def _train_with_cached_base_activations(
    *,
    base_train,
    base_val,
    redteam_train,
    redteam_val,
    model_name: str,
    layer: int,
    probe_spec,
    pos_class_label: str | None,
    neg_class_label: str | None,
    probe_description: str | None,
    base_train_cache: Path | None,
    base_val_cache: Path | None,
    redteam_cache_dir: str | Path | None = None,
    combine_consecutive_messages: bool = False,
    convert_tool_to_assistant: bool = False,
    seed: int = 0,
    verbose: bool,
):
    """Extract activations (base + red-team from disk caches) and fit the probe.

    Re-hosts the tail of tuberlens' ``train_probe`` (training.py): compute
    activations, assign them to the datasets, then ``ProbeFactory.build``. The one
    addition is splitting extraction by origin: the (fixed) base split is read from /
    written to a single-blob disk cache via tuberlens' ``get_activations`` blob cache,
    while the *growing* red-team set is cached **per conversation** (so samples seen in
    earlier iterations are reused and only newly-seen ones are forwarded — see
    ``_activate_redteam_cached``). With ``redteam_cache_dir=None`` the red-team set is
    recomputed each call (previous behaviour). Per-side parts are merged with
    ``LabelledDataset.concatenate``, which pads the activation tensors to a common
    length and concatenates them.

    The heavy ``LLMModel`` is loaded lazily and only if something actually needs
    computing — a full cache hit with no red-team samples needs no model at all.

    ``seed`` is re-applied via ``seed_everything`` immediately before
    ``ProbeFactory.build`` so the probe's random weight initialization (and any
    training stochasticity) is a pure function of ``seed`` and the training data —
    not of how far the global RNG has advanced through earlier red-teaming, eval
    subsampling, or prior retrains in the same process. Without this, two retrains
    on byte-identical data produce different probes (and different eval scores).
    """
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.model import LLMModel
    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.evaluation import seed_everything

    loaded: dict[str, Any] = {"model": None}

    def _get_model():
        if loaded["model"] is None:
            if verbose:
                print("Loading model for activation extraction ...")
            # offload_buffers=True: when device_map="auto" offloads layers to CPU/disk
            # (e.g. gemma-3-27b), buffers must offload too or accelerate warns about
            # insufficient GPU buffer space / risks OOM. Mirrors ProbeJudge._ensure_model.
            loaded["model"] = LLMModel.load(
                model_name, model_kwargs={"offload_buffers": True}
            )
        return loaded["model"]

    def _activate(dataset, cache_path: Path | None):
        if dataset is None or len(dataset) == 0:
            return None
        if cache_path is not None and Path(cache_path).exists():
            if verbose:
                print(f"Loaded cached base activations: {cache_path}")
            acts = LLMModel.load_activations(cache_path)
        else:
            acts = _get_model().get_activations(
                dataset.inputs,
                layer=layer,
                show_progress=verbose,
                save_path=str(cache_path) if cache_path is not None else None,
            )
        return dataset.assign(
            activations=acts.activations,
            attention_mask=acts.attention_mask,
            input_ids=acts.input_ids,
        )

    def _activate_redteam(dataset):
        return _activate_redteam_cached(
            dataset,
            redteam_cache_dir,
            model_name,
            layer,
            combine_consecutive_messages,
            convert_tool_to_assistant,
            _get_model,
            verbose,
        )

    base_train_a = _activate(base_train, base_train_cache)
    base_val_a = _activate(base_val, base_val_cache)
    redteam_train_a = _activate_redteam(redteam_train)
    redteam_val_a = _activate_redteam(redteam_val)

    def _combine(parts):
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else LabelledDataset.concatenate(parts)

    train_dataset = _combine([base_train_a, redteam_train_a])
    validation_dataset = _combine([base_val_a, redteam_val_a])
    if train_dataset is None:
        raise ValueError("No training data available to fit the probe.")

    if verbose:
        print(
            f"Train/validation: {len(train_dataset)} train, "
            f"{0 if validation_dataset is None else len(validation_dataset)} validation "
            f"(base {0 if base_train_a is None else len(base_train_a)}+"
            f"{0 if base_val_a is None else len(base_val_a)}; red-team "
            f"{0 if redteam_train_a is None else len(redteam_train_a)}+"
            f"{0 if redteam_val_a is None else len(redteam_val_a)})"
        )

    # Reseed right before the fit so the fresh probe's random weight init is
    # determined solely by `seed` + the (deterministic) training data, regardless of
    # how much the global RNG advanced earlier in the process. See docstring.
    seed_everything(seed)
    return ProbeFactory.build(
        probe_spec=probe_spec,
        train_dataset=train_dataset,
        model_name=model_name,
        layer=layer,
        validation_dataset=validation_dataset,
        use_store=False,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
    )


@dataclass
class RetrainResult:
    new_probe_path: Path
    n_redteam_samples: int
    n_training_samples_total: int


def _dump_labelled_dataset(dataset, out_path: str | Path) -> int:
    """Write a tuberlens LabelledDataset to JSONL — one `{id, inputs, label}` row per sample.

    Each `inputs` is the conversation as `[{role, content}, ...]`; `label` is the canonical
    "positive"/"negative" enum value the sample was trained with. Returns the row count.
    """
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = dataset.other_fields.get("labels", [])
    ids = dataset.ids
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, messages in enumerate(dataset.inputs):
            row = {
                "id": ids[i] if i < len(ids) else f"redteam-{i}",
                "inputs": [{"role": m.role, "content": m.content} for m in messages],
                "label": labels[i] if i < len(labels) else None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


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
        max_retries=preprocessing.max_generation_retries,
        cache_path=contrastive_cache_path,
        assistant_centric=preprocessing.assistant_centric,
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
    postprocessed_out_path: str | Path | None = None,
    min_judge_confidence: int = 0,
    test_size: float = 0.2,
    split_field: str | None = None,
    seed: int = 0,
    base_data_fraction: float = 1.0,
    base_activation_cache_dir: str | Path | None = None,
    combine_consecutive_messages: bool = False,
    convert_tool_to_assistant: bool = False,
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
        postprocessed_out_path: If given, write the postprocessed red-team samples (the
            filter + contrastive output that gets concatenated with the base data — base
            data itself excluded) to this JSONL as a per-iteration snapshot of exactly what
            red-team data trained this probe.
        min_judge_confidence: Drop red-team successes whose judge confidence is below this
            (keep `judge_confidence >= min_judge_confidence`). This is the *only* place
            judge confidence gates samples — view_past_attempts no longer filters on it.
            0 (default) keeps every success.
        test_size: Fraction held out for validation via stable_train_test_split.
        split_field: Optional field to keep grouped together when splitting (passed to
            stable_train_test_split).
        seed: Seed for the deterministic train/val split (stable_train_test_split).
        base_data_fraction: Fraction (0, 1] of the base training data to ingest,
            selected by a deterministic content-addressed random subsample
            (stable_fraction_subsample) applied *before* the train/val split.
            1.0 (default) uses all of it. The fraction is folded into the base
            activation cache key. Red-team successes are never subsampled.
        base_activation_cache_dir: If given, activations are cached here. The base
            training split is cached as a single blob (computed once for the whole run
            and reused every retrain). The growing red-team set is cached **per
            conversation** in a ``redteam_acts_*`` subdir, so a success first seen in an
            earlier iteration is forwarded once and reused by every later retrain; only
            newly-seen conversations are computed. None disables both caches.
        combine_consecutive_messages: Merge adjacent same-role messages in the training
            data (base loaded via load_from + red-team). Apply the same value used at eval
            time so the probe trains and is scored on the same message representation.
        convert_tool_to_assistant: Rewrite tool-role messages as assistant in the training
            data (applied before combine_consecutive_messages), matching the eval transform.
        verbose: Forwarded to the probe builder.
    """
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType

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
    n_before_conf = len(all_successes)
    if min_judge_confidence > 0:
        all_successes = [
            rec for rec in all_successes if rec.judge_confidence >= min_judge_confidence
        ]
    if verbose:
        if min_judge_confidence > 0:
            print(
                f"Red-team successes loaded: {n_before_conf} → {len(all_successes)} "
                f"after dropping judge_confidence < {min_judge_confidence}"
            )
        else:
            print(f"Red-team successes loaded: {len(all_successes)}")

    redteam_dataset = _build_redteam_dataset(
        all_successes,
        pos_class_label,
        neg_class_label,
        preprocessing,
        contrastive_cache_path,
        verbose,
    )
    redteam_dataset = _apply_message_transforms(
        redteam_dataset, combine_consecutive_messages, convert_tool_to_assistant
    )
    n_redteam = len(redteam_dataset)
    if verbose and preprocessing is not None:
        print(f"Red-team training samples after preprocessing: {n_redteam}")

    if postprocessed_out_path is not None:
        n_written = _dump_labelled_dataset(redteam_dataset, postprocessed_out_path)
        if verbose:
            print(f"Saved {n_written} postprocessed red-team samples to {postprocessed_out_path}")

    layer_used = layer if layer is not None else int(base_probe.layer)

    # Resolve the architecture for the new probe. Default (None): inherit the base probe's
    # architecture so the retrain stays apples-to-apples. A ProbeType name string or an
    # explicit ProbeSpec requests a fresh architecture instead.
    if probe_spec is None:
        probe_spec = _infer_probe_spec(base_probe)
    else:
        probe_spec = _coerce_probe_spec(probe_spec)

    # Split base and red-team *independently*, then combine per side. Because the
    # split is content-deterministic (stable_train_test_split), splitting the two
    # sources separately yields the same membership as splitting their
    # concatenation — but it lets us cache the fixed base activations on disk and
    # only ever recompute the small, growing red-team set.
    base_train = base_val = None
    base_train_cache = base_val_cache = None
    n_base = 0
    if base_training_data_path is not None:
        base_dataset = LabelledDataset.load_from(
            Path(base_training_data_path),
            pos_class_label=pos_class_label,
            neg_class_label=neg_class_label,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
        )
        n_loaded = len(base_dataset)
        base_dataset = stable_fraction_subsample(base_dataset, base_data_fraction, seed)
        n_base = len(base_dataset)
        if verbose and base_data_fraction < 1.0:
            print(
                f"Base data subsampled to fraction {base_data_fraction}: "
                f"{n_loaded} → {n_base}"
            )
        base_train, base_val = stable_train_test_split(
            base_dataset, test_size=test_size, split_field=split_field, seed=seed
        )
        if base_activation_cache_dir is not None:
            base_train_cache, base_val_cache = _base_activation_cache_paths(
                base_activation_cache_dir,
                base_training_data_path,
                base_probe.model_name,
                layer_used,
                seed,
                test_size,
                split_field,
                combine_consecutive_messages,
                convert_tool_to_assistant,
                base_data_fraction,
            )

    redteam_train = redteam_val = None
    if n_redteam > 0:
        redteam_train, redteam_val = stable_train_test_split(
            redteam_dataset, test_size=test_size, split_field=split_field, seed=seed
        )
    elif base_training_data_path is None:
        raise ValueError(
            "No red-team successes and no base_training_data_path provided — nothing to train on."
        )

    n_total = n_base + n_redteam
    if verbose:
        print(f"Total samples before split: {n_total} (base {n_base}, red-team {n_redteam})")

    new_probe = _train_with_cached_base_activations(
        base_train=base_train,
        base_val=base_val,
        redteam_train=redteam_train,
        redteam_val=redteam_val,
        model_name=base_probe.model_name,
        layer=layer_used,
        probe_spec=probe_spec,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
        base_train_cache=base_train_cache,
        base_val_cache=base_val_cache,
        redteam_cache_dir=base_activation_cache_dir,
        combine_consecutive_messages=combine_consecutive_messages,
        convert_tool_to_assistant=convert_tool_to_assistant,
        seed=seed,
        verbose=verbose,
    )

    with new_probe_path.open("wb") as f:
        pickle.dump(new_probe, f)
    if verbose:
        print(f"Saved retrained probe to {new_probe_path}")

    return RetrainResult(
        new_probe_path=new_probe_path,
        n_redteam_samples=n_redteam,
        n_training_samples_total=n_total,
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
    seed: int = 0,
    base_data_fraction: float = 1.0,
    base_activation_cache_dir: str | Path | None = None,
    combine_consecutive_messages: bool = False,
    convert_tool_to_assistant: bool = False,
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
        test_size: Fraction held out for validation via stable_train_test_split.
        split_field: Optional field to keep grouped together when splitting.
        seed: Seed for the deterministic train/val split (stable_train_test_split).
        base_data_fraction: Fraction (0, 1] of the base training data to ingest,
            selected by a deterministic content-addressed random subsample
            (stable_fraction_subsample) applied before the train/val split. 1.0
            (default) uses all of it; the fraction is folded into the activation
            cache key.
        base_activation_cache_dir: If given, the base split's activations are cached here.
            Using the same dir as the retrains means the initial training populates the
            cache that every subsequent retrain reuses (base activations computed once).
        combine_consecutive_messages: Merge adjacent same-role messages when loading the
            training data — apply the same value used at eval time so the probe trains and
            is scored on the same message representation.
        convert_tool_to_assistant: Rewrite tool-role messages as assistant (applied before
            combine_consecutive_messages), matching the eval transform.
        verbose: Forwarded to the probe builder.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    new_probe_path = Path(new_probe_path)
    new_probe_path.parent.mkdir(parents=True, exist_ok=True)

    base_dataset = LabelledDataset.load_from(
        Path(base_training_data_path),
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        combine_consecutive_messages=combine_consecutive_messages,
        convert_tool_to_assistant=convert_tool_to_assistant,
    )
    n_loaded = len(base_dataset)
    base_dataset = stable_fraction_subsample(base_dataset, base_data_fraction, seed)
    if verbose:
        if base_data_fraction < 1.0:
            print(
                f"Base data subsampled to fraction {base_data_fraction}: "
                f"{n_loaded} → {len(base_dataset)}"
            )
        print(f"Initial samples before split: {len(base_dataset)}")
        base_dataset.print_label_distribution()

    base_train, base_val = stable_train_test_split(
        base_dataset, test_size=test_size, split_field=split_field, seed=seed
    )

    spec = _coerce_probe_spec(probe_spec or DEFAULT_FRESH_PROBE_ARCH)

    base_train_cache = base_val_cache = None
    if base_activation_cache_dir is not None:
        base_train_cache, base_val_cache = _base_activation_cache_paths(
            base_activation_cache_dir,
            base_training_data_path,
            model_name,
            layer,
            seed,
            test_size,
            split_field,
            combine_consecutive_messages,
            convert_tool_to_assistant,
            base_data_fraction,
        )

    probe = _train_with_cached_base_activations(
        base_train=base_train,
        base_val=base_val,
        redteam_train=None,
        redteam_val=None,
        model_name=model_name,
        layer=layer,
        probe_spec=spec,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
        base_train_cache=base_train_cache,
        base_val_cache=base_val_cache,
        seed=seed,
        verbose=verbose,
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
