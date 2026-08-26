"""Convert successful red-team JSONL samples back into training data and retrain a probe."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import torch

from agentic_redteam.ensemble import (
    DETERMINISTIC_ARCHS,
    ENSEMBLE_SEEDS,
    MAX_ENSEMBLE_SIZE,
    EnsembleProbe,
    ensemble_size as probe_ensemble_size,
    fusion_enabled,
)
from agentic_redteam.persistence import AttemptRecord, JsonlStore
from agentic_redteam.token_budget import TokenBudget

if TYPE_CHECKING:
    from agentic_redteam.config import PreprocessingConfig

# Default fresh probe architecture, mirroring tuberlens' collate_train_evaluate.py.
# Used when a retrain is asked for a fresh architecture without naming a specific one.
DEFAULT_FRESH_PROBE_ARCH = "linear_then_softmax"


def _resolve_ensemble_seeds(seed: int, n: int) -> list[int]:
    """Training seeds for an ``n``-member deep ensemble.

    For ``n > 1`` these are the **repo-pinned** ``ENSEMBLE_SEEDS[:n]`` — the same
    numbers on every run, config and box — so an ensemble's members are fixed
    identities rather than an arithmetic walk off whatever ``--seed`` happens to
    be. That keeps ``--seed`` doing exactly one job (moving the *data*: the
    train/val split and the eval subsample) and this list doing the other (fixing
    each member's weight init and batch order), instead of entangling the two.

    ``n == 1`` is the carve-out: it returns ``[seed]``, because with a single
    probe the run's ``--seed`` *is* that probe's training seed and always has
    been. Routing the default single-probe path through the pinned list would
    silently retrain every existing non-ensemble config under a different seed.

    Whatever the seeds, only the **fit** is varied — never the split. Splitting
    per member would give each a different validation set (so early stopping would
    select against different data) and, worse, a different base-activation cache
    key, turning one cached extraction into ``n``.
    """
    if n < 1:
        raise ValueError(f"ensemble_size must be >= 1; got {n}")
    if n > MAX_ENSEMBLE_SIZE:
        raise ValueError(
            f"ensemble_size must be <= {MAX_ENSEMBLE_SIZE}; got {n}"
        )
    if n == 1:
        return [seed]
    return list(ENSEMBLE_SEEDS[:n])


def _warn_if_deterministic_arch(probe_spec, n: int) -> None:
    """Warn when the requested architecture can't actually be diversified by a seed.

    A deep ensemble only helps where the fit is stochastic. Difference-of-means and
    LDA are closed-form and SklearnProbe's logistic regression is solved by lbfgs
    under a fixed ``random_state``, so ``n`` seeds there produce ``n`` identical
    members and an average that equals a single probe — at ``n`` times the cost.
    """
    name = getattr(getattr(probe_spec, "name", None), "value", None)
    if name in DETERMINISTIC_ARCHS:
        print(
            f"WARNING: architecture {name!r} fits deterministically, so all {n} "
            "ensemble members will be identical and their averaged score will equal "
            "a single probe's. Use a stochastic architecture (e.g. "
            f"{DEFAULT_FRESH_PROBE_ARCH!r}) for the ensemble to do anything."
        )


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


def _dev_activation_cache_path(
    cache_dir: str | Path,
    dev_files: "list[Path]",
    model_name: str,
    layer: int,
    combine_consecutive_messages: bool,
    convert_tool_to_assistant: bool,
) -> Path:
    """Single-blob activation cache path for the held-out dev (validation) set.

    Cached like the base split rather than like the red-team set: the dev files are
    fixed for the whole run, so one blob keyed on their contents hits on every
    iteration. There is no ``seed`` / ``test_size`` / ``split_field`` in the key
    because the dev set is used *whole* — it is never split — so nothing about the
    train/val partition can change which rows it holds. As everywhere else here,
    load is *by path without validating inputs*, so everything that would change the
    activations is folded into the key.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for f in dev_files:
        h.update(f.name.encode("utf-8"))
        h.update(Path(f).read_bytes())
    h.update(
        f"|model={model_name}|layer={layer}"
        f"|combine={combine_consecutive_messages}|convert={convert_tool_to_assistant}".encode(
            "utf-8"
        )
    )
    key = h.hexdigest()[:16]
    safe_model = model_name.replace("/", "_")
    return cache_dir / f"dev_acts_{safe_model}_L{layer}_{key}.pt"


def _load_dev_dataset(
    dev_data_path: str | Path,
    pos_class_label: str,
    neg_class_label: str,
    combine_consecutive_messages: bool,
    convert_tool_to_assistant: bool,
    verbose: bool = True,
):
    """Load the held-out dev set that serves as the validation set.

    ``dev_data_path`` is either a single JSONL or a directory of them, in which case
    every ``*.jsonl`` is loaded (splits are auto-discovered the same way
    ``evaluate_probe`` discovers eval splits) and concatenated into one dataset —
    the probe fit wants a single validation set, and per-split provenance columns are
    irrelevant to it, so ``concatenate`` takes the column intersection.

    Every row must carry one of the probe's two class labels. tuberlens keeps an
    unmatched label as its raw string (``from_jsonl``) and only fails later, when
    ``.labels`` is first read — deep inside the fit, naming neither the file nor the
    labels it wanted. So the labels are read *here*, per split, and the failure is
    re-raised pointing at the offending file.

    Returns ``(dataset, files, sizes)`` — the files are what the activation cache is
    keyed on, and ``sizes`` is each file's row count in the same order, which is what
    lets a caller recover which split a concatenated row came from (the parts are
    concatenated in ``sorted(glob)`` order, each preserving its own row order).
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    path = Path(dev_data_path)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise ValueError(f"No *.jsonl dev splits found in {path}")
    elif path.exists():
        files = [path]
    else:
        raise ValueError(f"Dev data path does not exist: {path}")

    parts = []
    for f in files:
        part = LabelledDataset.load_from(
            f,
            pos_class_label=pos_class_label,
            neg_class_label=neg_class_label,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
        )
        try:
            part.labels
        except Exception as exc:
            raise ValueError(
                f"Dev split {f} has rows whose 'labels' is neither "
                f"{pos_class_label!r} nor {neg_class_label!r}: {exc}"
            ) from exc
        parts.append(part)
        if verbose:
            print(f"  dev split {f.name}: {len(part)} samples")

    sizes = [len(part) for part in parts]
    dataset = parts[0] if len(parts) == 1 else LabelledDataset.concatenate(parts)
    if len(dataset) == 0:
        raise ValueError(f"Dev data at {path} is empty.")
    return dataset, files, sizes


# How often _activate_redteam_cached reports progress while computing misses. Rows cost
# tens of seconds each on an offloaded gemma-sized model, so this is a line every few
# minutes, not a spinner.
_REDTEAM_PROGRESS_EVERY = 10


# Fields tuberlens' LabelledDataset.concatenate zero-pads along dim 1 before joining.
# These are the only large ones — everything else in `other_fields` is per-sample scalars.
_PAD_FIELDS = ("activations", "attention_mask", "input_ids")


def _concatenate_consuming(datasets):
    """Concatenate LabelledDatasets, freeing each part's activations as it is copied.

    Semantically identical to tuberlens' ``LabelledDataset.concatenate`` (columns are the
    intersection of the parts' ``other_fields``; ``activations`` / ``attention_mask`` /
    ``input_ids`` are zero-padded on dim 1 to the parts' common max length), but with a
    peak memory cost of ~1x the result instead of ~2x.

    ``concatenate`` pads *every* part first and then ``torch.cat``s them, so the padded
    inputs and the output are both resident at the moment of the cat. For a gemma-3-27b
    (hidden 5376, fp16, padded to 1024 tokens ⇒ 11 MB/sample) retrain over 966 samples
    that is ~19 GB of transient activations — on top of a 27B model whose CPU-offloaded
    shards are also resident. That is what SIGKILLed the 60 GB box in
    ``logs/run_hs_gemma27b_deepseekv4pro_noguidance.log`` (exit 137, no traceback).

    Here the output is allocated once with ``torch.empty`` and each part is copied into
    its slice and then dropped. ``torch.empty`` rather than ``torch.zeros`` is deliberate:
    a multi-GB CPU allocation is served by ``mmap``, so its pages only become resident as
    they are written, and every byte below is written exactly once — the real rows, then
    an explicit zero-fill of that part's pad region. So the output grows into memory at
    exactly the rate the parts are released, and peak stays at ~1x the result plus the
    single part in flight.

    **Consumes** ``datasets``: each part's pad fields are popped as they are copied, so
    callers must not read activations off the inputs afterwards (``len()``, ``inputs`` and
    ``ids`` stay valid — ``__len__`` is ``len(self.inputs)``). Anything this can't
    reproduce exactly (non-torch pad fields, mixed dtype/device/rank) falls back to
    ``LabelledDataset.concatenate``.
    """
    import numpy as np
    from tuberlens.interfaces.dataset import LabelledDataset

    datasets = [d for d in datasets if d is not None]
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]

    cols = set(datasets[0].other_fields).intersection(
        *[set(d.other_fields) for d in datasets]
    )
    pad_cols = [f for f in _PAD_FIELDS if f in cols]

    # Only handle the torch layout this repo's activation paths actually produce; defer
    # anything else to tuberlens rather than risk a subtly different result.
    for field in pad_cols:
        vals = [d.other_fields[field] for d in datasets]
        if not all(isinstance(v, torch.Tensor) for v in vals):
            return LabelledDataset.concatenate(datasets)
        if len({v.dtype for v in vals}) > 1 or len({v.device for v in vals}) > 1:
            return LabelledDataset.concatenate(datasets)
        if len({v.ndim for v in vals}) > 1 or vals[0].ndim < 2:
            return LabelledDataset.concatenate(datasets)
        if len({tuple(v.shape[2:]) for v in vals}) > 1:
            return LabelledDataset.concatenate(datasets)

    total = sum(len(d) for d in datasets)
    other_fields: dict[str, Any] = {}

    for field in pad_cols:
        max_len = max(d.other_fields[field].shape[1] for d in datasets)
        ref = datasets[0].other_fields[field]
        out = torch.empty(
            (total, max_len, *ref.shape[2:]), dtype=ref.dtype, device=ref.device
        )
        start = 0
        for d in datasets:
            arr = d.other_fields.pop(field)  # drop the part's reference ...
            n, seq = arr.shape[0], arr.shape[1]
            out[start : start + n, :seq] = arr
            if seq < max_len:
                out[start : start + n, seq:] = 0
            start += n
            del arr  # ... and free it before the next part is touched
        other_fields[field] = out

    for key in cols:
        if key in pad_cols:
            continue
        values = [d.other_fields[key] for d in datasets]
        if isinstance(values[0], np.ndarray):
            other_fields[key] = np.concatenate(values)
        elif isinstance(values[0], torch.Tensor):
            other_fields[key] = torch.cat(values)
        else:
            other_fields[key] = [item for v in values for item in v]

    return type(datasets[0])(
        inputs=[x for d in datasets for x in d.inputs],
        ids=[x for d in datasets for x in d.ids],
        other_fields=other_fields,
    )


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
    and only newly-seen conversations are forwarded through the model. Per-row blobs
    use the same dict layout tuberlens' ``get_activations``
    writes, so ``LLMModel.load_activations`` reads them back; ``_concatenate_consuming``
    re-pads the loaded + freshly-computed parts to a common length, so storing each row at
    its own width is safe. Returns a ``LabelledDataset``
    with activations assigned, or ``None`` for an empty/absent input. With
    ``cache_dir=None`` it degrades to a plain batched compute (matching the previous
    uncached behaviour).

    **Misses are computed in chunks and each row is written through immediately**,
    rather than one ``get_activations`` call over the whole miss set followed by a
    bulk save. Two reasons, both learned from a 770-sample gemma-3-27b retrain:

    - *Resumability.* The single-call form persists nothing until the last sample
      lands, so a crash at row 606 of 607 threw away ~25 hours of forwards. Now the
      next attempt reloads everything already computed.
    - *Width.* ``get_activations`` pads every row in a call to that call's max length,
      capped at 1024 (``tuberlens/model.py:433``). Over the whole miss set that is
      1024 for essentially every row; per chunk it is the chunk's own max, and at the
      default chunk size (tuberlens' ``BATCH_SIZE``, 1) it is each row's true length —
      roughly halving both the cache's disk footprint and resident RAM, since real
      conversations average ~535 tokens. ``_concatenate_consuming`` re-pads at merge,
      so the merged tensor is byte-identical either way.
    """
    if dataset is None or len(dataset) == 0:
        return None
    from tuberlens.model import LLMModel

    from agentic_redteam.model_loading import extraction_batch_size

    def _compute(ds, show_progress: bool):
        acts = get_model().get_activations(
            ds.inputs, layer=layer, show_progress=show_progress
        )
        return acts, ds.assign(
            activations=acts.activations,
            attention_mask=acts.attention_mask,
            input_ids=acts.input_ids,
        )

    if cache_dir is None:
        return _compute(dataset, verbose)[1]

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
        chunk_size = extraction_batch_size()
        if verbose:
            print(
                f"Red-team activations: {len(cached_idx)} loaded from cache, "
                f"{len(uncached_idx)} to compute (chunk size {chunk_size}) ..."
            )
        started = time.monotonic()
        for start in range(0, len(uncached_idx), chunk_size):
            chunk = uncached_idx[start : start + chunk_size]
            # show_progress=False: one tqdm bar per chunk would be hundreds of bars in
            # the log. The periodic line below reports the same thing far more cheaply.
            acts, computed = _compute(dataset[chunk], show_progress=False)
            for j, i in enumerate(chunk):
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
            # `computed` shares its tensors with `acts` (assign copies references, not
            # data), so the Activation must go too or _concatenate_consuming can't
            # actually free them.
            del acts
            done = start + len(chunk)
            if verbose and (done % _REDTEAM_PROGRESS_EVERY == 0 or done == len(uncached_idx)):
                elapsed = time.monotonic() - started
                rate = elapsed / done
                remaining = (len(uncached_idx) - done) * rate
                print(
                    f"  [red-team activations] {done}/{len(uncached_idx)} "
                    f"({rate:.1f}s/sample, ~{remaining / 60:.0f} min left)",
                    flush=True,
                )
    elif verbose:
        print(
            f"Red-team activations: {len(cached_idx)} loaded from cache, "
            f"0 computed fresh"
        )
    return _concatenate_consuming(parts)


def _fit_device():
    """The device tuberlens' probe fit will run on, or ``None`` if it is the CPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        from tuberlens.config import global_settings

        device = str(getattr(global_settings, "DEVICE", "cuda"))
    except Exception:  # pragma: no cover - tuberlens layout drift
        device = "cuda"
    return None if device.startswith("cpu") else device


def _allocatable_bytes(device) -> int:
    """How much CUDA memory a new allocation can actually get, in bytes.

    Not the same thing as ``mem_get_info``'s free figure, and the difference is the
    whole ballgame here. torch's caching allocator holds on to the segments it has
    already carved out; ``empty_cache()`` only returns segments that are *entirely*
    unused, and after a 27B model has been loaded, used and dropped, enough of them
    stay reserved that the driver still reports the card as full. Measured on this
    box right after releasing the extraction model::

        driver free    1.08 GiB      <- what mem_get_info reports
        torch reserved 21.62 GiB
        torch allocated 0.01 GiB     <- so 21.61 GiB of that reservation is FREE

    A 10.8 GiB allocation at that moment succeeds, served straight out of the
    reserved pool. Sizing the staging against the driver's figure alone therefore
    declines to stage anything on a card that has ~22 GiB going spare, and the fits
    silently fall back to host-resident activations — the exact failure this function
    exists to prevent, arrived at from the opposite direction.

    So: driver-free PLUS the reserved-but-unallocated pool. Both terms matter — the
    first is what can still be carved out, the second what has already been carved
    and handed back.
    """
    import torch

    free, _total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    return int(free) + max(0, int(reserved) - int(allocated))


_STAGING_ENV = "AGENTIC_REDTEAM_STAGE_ACTIVATIONS"
_STAGING_RESERVE_ENV = "AGENTIC_REDTEAM_FIT_STAGING_RESERVE_GIB"
_DEFAULT_STAGING_RESERVE_GIB = 2.0


def _staging_enabled() -> bool:
    return os.environ.get(_STAGING_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _staging_reserve_bytes() -> int:
    """Headroom kept free for the fit's own activations and gradients, in bytes.

    2 GiB by default, which is what a gemma-3-27b-shaped head fit wanted on a 24 GiB
    card. It is a property of the head and batch size rather than of the dataset, so
    it is a knob rather than a fraction of the card: raise it if a fit OOMs *after*
    staging reported success, lower it to squeeze a borderline set on.
    """
    raw = os.environ.get(_STAGING_RESERVE_ENV, "").strip()
    if raw:
        try:
            gib = float(raw)
        except ValueError:
            print(
                f"Ignoring {_STAGING_RESERVE_ENV}={raw!r}: expected a number of GiB. "
                f"Using {_DEFAULT_STAGING_RESERVE_GIB}."
            )
        else:
            if gib >= 0:
                return int(gib * 2**30)
            print(
                f"Ignoring {_STAGING_RESERVE_ENV}={raw!r}: must not be negative. "
                f"Using {_DEFAULT_STAGING_RESERVE_GIB}."
            )
    return int(_DEFAULT_STAGING_RESERVE_GIB * 2**30)


def _fit_dtype_for(name: str, tensor: "torch.Tensor"):
    """The dtype to stage `name` in — the fit's own, when that is not a size increase.

    Activations are extracted and cached in fp16 while the fit runs in tuberlens'
    `DTYPE` (bf16 on a GPU), and `ActivationDataset.__getitems__` closes with
    `.to(self.device).to(self.dtype)` — so every batch of every epoch of every member
    re-converts the same rows. Converting once here removes that per-batch cast for
    the ordinary and the fused trainer alike, and costs nothing: fp16 and bf16 are
    both 2 bytes, so the staged tensor is exactly the size it would have been.

    The conversion is elementwise, so it produces the values the per-batch cast
    produced, in the same order — the probes come out bit-identical, verified member
    for member. Guarded on itemsize because the rule must not silently double a
    tensor: a float32 fit dtype (a CPU-only box) leaves fp16 alone, and only the
    activations are cast at all — `attention_mask` and `input_ids` are integer
    columns whose dtype the dataset relies on.
    """
    if name != "activations":
        return tensor.dtype
    try:
        from tuberlens.config import global_settings
    except Exception:
        return tensor.dtype
    fit_dtype = getattr(global_settings, "DTYPE", None)
    if fit_dtype is None or not fit_dtype.is_floating_point:
        return tensor.dtype
    if fit_dtype.itemsize > tensor.dtype.itemsize:
        return tensor.dtype
    return fit_dtype


def _to_device_for_fit(
    datasets, *, verbose: bool = True, reserve_bytes: int | None = None
) -> bool:
    """Move merged activation tensors onto the fit device before ``ProbeFactory.build``.

    ``ActivationDataset.__getitems__`` ends every batch fetch in ``.to(self.device)``.
    With the tensors on the host that is a scattered CPU gather plus a host->device
    copy of the entire set, once per epoch, for every ensemble member — which is where
    a retrain's wall-clock actually goes. Pre-staging them here turns that ``.to`` into
    a no-op.

    Purely a data-placement change: the sampler, the batch order and the values are
    untouched, so the fitted probes are bit-identical.

    **Largest dataset first — the caller's order is not load-bearing.** Every dataset
    handed in is traversed once per epoch (the training set under forward+backward, the
    validation set under ``no_grad``), so the bytes a dataset moves per epoch is simply
    its own size, and when only some of them fit, the card is worth spending on the
    biggest. They are therefore sorted here rather than trusted in the order given.

    That ordering has to be computed, not assumed, because *which* set is the big one
    is a property of the configuration:

    ======================================  ==================  ==============
    configuration                           validation set      larger set
    ======================================  ==================  ==============
    ``--dev-data dev_samples/highstakes``   1908 rows           validation
    ``--dev-data dev_samples/hu_ha``        290 rows            training
    ``--dev-data dev_samples/instructions`` 436 rows            training
    default split (no ``--dev-data``)       ``test_size`` share training
    ======================================  ==================  ==============

    Pinning either end of that table strands the other. Measured on the high-stakes
    shape (666 train / 1908 dev, gemma-3-27b activations, 24 GiB card), staging the
    *smaller* of the two ran 4.3 epochs/min against 13.7 for the larger and ~4.0
    host-resident — i.e. picking wrong gives up nearly the whole gain.

    **All or nothing, per dataset.** tuberlens' ``Activation.__post_init__`` does
    ``activations *= attention_mask[:, :, None]``, so a dataset whose fields straddle
    two devices raises rather than merely running slowly.

    **Capacity-checked rather than left to OOM.** An unconditional ``.to(device)``
    relies on the copy raising ``OutOfMemoryError`` when the set does not fit. That
    holds on a normal Linux/CUDA box but *not* under WSL2, where the driver
    oversubscribes into host memory and pages over PCIe instead — there the fallback
    below never fires and the fit gets exactly the paging this exists to remove
    (measured: 26.4 GiB staged onto a 24 GiB card ran 5.3 epochs/min against ~4.0
    host-resident, where the data path predicts ~45x). So a dataset is moved only if it
    fits in ``_allocatable_bytes`` less ``reserve_bytes``.

    Returns True if anything was moved. On OOM — or any other failure — every tensor
    already moved is restored to its original device and the fit proceeds on the host
    exactly as before, because a slow retrain beats a dead one. That fallback is not
    hypothetical: the red-team set grows each iteration, so a long enough run will
    eventually present a set larger than the card.

    Two environment knobs, honoured on every call:

    ``AGENTIC_REDTEAM_STAGE_ACTIVATIONS=0``
        Skip staging entirely and fit host-resident, as before this existed.
    ``AGENTIC_REDTEAM_FIT_STAGING_RESERVE_GIB``
        Headroom left free for the fit itself (default 2) — see
        ``_staging_reserve_bytes``.

    ``reserve_bytes`` overrides the latter for a single call.
    """
    import torch

    if not _staging_enabled():
        if verbose:
            print(
                f"Activation staging disabled by {_STAGING_ENV}; the probe fits run "
                "host-resident."
            )
        return False

    device = _fit_device()
    if device is None:
        return False
    if reserve_bytes is None:
        reserve_bytes = _staging_reserve_bytes()

    try:
        target_type = torch.device(device).type
    except Exception:  # pragma: no cover - an unparseable DEVICE setting
        target_type = "cuda"

    # Size every dataset BEFORE moving any of it, so they can be ordered by what they
    # actually cost per epoch. A dataset already on the target device contributes no
    # pending tensors and drops out here.
    plan: list[tuple[int, dict, list[tuple[str, "torch.Tensor"]]]] = []
    for ds in datasets:
        if ds is None:
            continue
        fields = getattr(ds, "other_fields", None)
        if not isinstance(fields, dict):
            continue
        pending = [
            (name, fields[name])
            for name in _PAD_FIELDS
            if isinstance(fields.get(name), torch.Tensor)
            and fields[name].device.type != target_type
        ]
        if not pending:
            continue
        size = sum(tensor.numel() * tensor.element_size() for _n, tensor in pending)
        plan.append((size, fields, pending))
    plan.sort(key=lambda entry: entry[0], reverse=True)

    original: list[tuple[dict, str, "torch.Tensor"]] = []
    moved_bytes = 0
    skipped_bytes = 0
    try:
        for size, fields, pending in plan:
            # Re-read the budget each time: the previous dataset just consumed some.
            if size + reserve_bytes > _allocatable_bytes(device):
                skipped_bytes += size
                continue
            for name, tensor in pending:
                original.append((fields, name, tensor))
                fields[name] = tensor.to(device, _fit_dtype_for(name, tensor))
            moved_bytes += size
    except Exception as exc:  # torch.cuda.OutOfMemoryError is a RuntimeError subclass
        for fields, name, tensor in original:
            fields[name] = tensor
        torch.cuda.empty_cache()
        if verbose:
            print(
                f"Activations stay on the host for the fit ({type(exc).__name__}: {exc}); "
                "the retrain will be much slower but is unaffected otherwise."
            )
        return False

    if verbose and (moved_bytes or skipped_bytes):
        note = (
            f"Staged {moved_bytes / 2**30:.1f} GiB of activations on {device} for the "
            "probe fits (no host->device copy per epoch)"
        )
        if skipped_bytes:
            note += (
                f"; {skipped_bytes / 2**30:.1f} GiB left on the host — it does not fit in "
                f"the {_allocatable_bytes(device) / 2**30:.1f} GiB allocatable with a "
                f"{reserve_bytes / 2**30:.1f} GiB reserve ({_STAGING_RESERVE_ENV} to "
                "change), and staging it anyway risks paging over PCIe rather than "
                "failing cleanly"
            )
        print(note)
    return bool(original)


def _train_with_cached_base_activations(
    *,
    base_train,
    base_val,
    redteam_train,
    redteam_val,
    dev_val,
    model_name: str,
    layer: int,
    probe_spec,
    pos_class_label: str | None,
    neg_class_label: str | None,
    probe_description: str | None,
    base_train_cache: Path | None,
    base_val_cache: Path | None,
    dev_val_cache: Path | None = None,
    dev_train_idx: "list[int] | None" = None,
    dev_val_idx: "list[int] | None" = None,
    redteam_cache_dir: str | Path | None = None,
    combine_consecutive_messages: bool = False,
    convert_tool_to_assistant: bool = False,
    seed: int = 0,
    ensemble_seeds: list[int] | None = None,
    verbose: bool,
):
    """Extract activations (base + red-team from disk caches) and fit the probe.

    Re-hosts the tail of tuberlens' ``train_probe`` (training.py): compute
    activations, assign them to the datasets, then ``ProbeFactory.build``. The one
    addition is splitting extraction by origin: the (fixed) base split is read from /
    written to a single-blob disk cache via tuberlens' ``get_activations`` blob cache,
    while the *growing* red-team set is cached **per conversation** (so samples seen in
    earlier iterations are reused and only newly-seen ones are forwarded — see
    ``_activate_redteam_cached``). ``dev_val`` is a held-out dev set supplied whole as
    the validation set; like the base split it is fixed for the run, so it is cached as
    one blob (``_dev_activation_cache_path``). When it is given, the callers pass
    ``base_val``/``redteam_val`` as empty, so validation is the dev set *alone*.
    ``dev_train_idx`` / ``dev_val_idx`` optionally partition that dev set into a
    training slice and a validation slice — a row selection applied *after* extraction,
    so the blob (and its cache key) is untouched and the whole set is still fetched or
    computed exactly once. Both empty (the default) leaves the dev set undivided. With ``redteam_cache_dir=None`` the red-team set is
    recomputed each call (previous behaviour). Per-side parts are merged with
    ``_concatenate_consuming``, which pads the activation tensors to a common length and
    concatenates them at ~1x rather than ~2x the result's peak memory.

    The heavy ``LLMModel`` is loaded lazily and only if something actually needs
    computing — a full cache hit with no red-team samples needs no model at all — and is
    released as soon as the last activation is extracted, before the merge and fit. Both
    of those exist because this function's peak host-RAM cost (activations *plus* a
    partly CPU-offloaded gemma-sized model) is what OOM-kills long runs.

    ``seed`` is re-applied via ``seed_everything`` immediately before
    ``ProbeFactory.build`` so the probe's random weight initialization (and any
    training stochasticity) is a pure function of ``seed`` and the training data —
    not of how far the global RNG has advanced through earlier red-teaming, eval
    subsampling, or prior retrains in the same process. Without this, two retrains
    on byte-identical data produce different probes (and different eval scores).

    ``ensemble_seeds`` turns that single fit into a **score-averaging deep ensemble**:
    one fit per seed over the *same* pre-activated datasets, wrapped in an
    :class:`~agentic_redteam.ensemble.EnsembleProbe` whose score is the members' mean
    probability. Only the fit repeats — the split, the extraction and the merge are
    shared — so member ``k > 0`` costs a probe-head fit, not another pass through the
    extraction LLM. ``None`` or a single seed returns a plain probe, byte-identical to
    what the non-ensemble path produced. The seeds themselves come from
    ``_resolve_ensemble_seeds``, which draws them from the repo-pinned
    ``ENSEMBLE_SEEDS`` so a member's identity is the same on every run.

    The members are fit **together** (tuberlens' ``ProbeFactory.build_ensemble``,
    which steps them under ``vmap`` and so reads the activations once per epoch
    instead of once per member) or **sequentially** (one ``ProbeFactory.build`` per
    seed, the path this repo took before fusion existed).
    ``ensemble.fusion_enabled()`` — i.e. ``PROBE_FUSED_ENSEMBLE`` — picks between
    them, and picks the same way for scoring; see that function. Either way the
    members come back as ordinary independent probes, so the pickle and every
    consumer are identical and the choice costs only wall-clock.
    """
    from tuberlens.model import LLMModel
    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.model_loading import load_extraction_model, unhook_model

    loaded: dict[str, Any] = {"model": None}

    def _get_model():
        if loaded["model"] is None:
            if verbose:
                print("Loading model for activation extraction ...")
            # Loads only layers 0..layer (deeper ones are never executed) and carries
            # offload_buffers=True. Mirrors ProbeJudge._ensure_model — see
            # model_loading for why the truncation matters and why it's exact.
            loaded["model"] = load_extraction_model(model_name, layer, verbose=verbose)
        return loaded["model"]

    def _release_model():
        """Drop the extraction model, freeing its CPU-offloaded shards and GPU memory.

        Mirrors ``ProbeJudge.release``. No-op on a full cache hit, where no model was
        ever loaded.
        """
        if loaded["model"] is None:
            return
        if verbose:
            print("Releasing extraction model before probe fit ...")
        # Strip accelerate's hooks FIRST: without that the assignment below frees
        # nothing, the card stays full, and _to_device_for_fit then finds no room to
        # stage the activations. See model_loading.unhook_model for the measurements.
        unhook_model(loaded["model"])
        loaded["model"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose and torch.cuda.is_available():
            # Report what the staging decision will actually consult. Driver-free on
            # its own is misleading here: torch keeps its segments reserved, so this
            # routinely reads ~1 GiB on a card with ~22 GiB available. See
            # _allocatable_bytes.
            print(
                f"  {_allocatable_bytes(None) / 2**30:.1f} GiB allocatable after the release "
                f"({torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB of that free at the driver)"
            )

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
    dev_val_a = _activate(dev_val, dev_val_cache)

    n_by_origin = (
        0 if base_train_a is None else len(base_train_a),
        0 if base_val_a is None else len(base_val_a),
        0 if redteam_train_a is None else len(redteam_train_a),
        0 if redteam_val_a is None else len(redteam_val_a),
        0 if dev_val_a is None else len(dev_val_a),
    )

    # Every activation is extracted by this point, so release the LLM before the
    # concatenate + fit, which are the memory-hungry steps. This matters on host RAM, not
    # just GPU: `device_map="auto"` with an unpinned `max_memory` (the default — see
    # `model_loading` for the two ways to pin it) lets accelerate hand the "cpu"
    # device a budget equal to whatever RAM is free at load time, so a gemma-sized model
    # keeps multi-GB of CPU-offloaded shards resident for as long as it is referenced.
    # Holding those through the assembly of a ~10 GB activation set is what OOM-killed
    # `logs/run_hs_gemma27b_deepseekv4pro_noguidance.log`. Nothing below needs the model —
    # and `_get_model` would reload it if that ever changed.
    _release_model()

    # Split the dev set into a training slice and a validation slice AFTER extraction —
    # and after the model release, since the row selection transiently holds the dev
    # activations twice. The dev set is one cached blob keyed on the dev files' bytes
    # (assembled, under a `kaggle:` section, from published per-split blobs), so it is
    # always extracted whole and the partition is a row selection on the result: no
    # cache key moves, and the two slices cost exactly what the undivided set cost.
    # `dev_val_idx` is empty on the default path, which leaves this a no-op.
    dev_train_a = None
    if dev_val_a is not None and dev_val_idx:
        dev_train_a = dev_val_a[dev_train_idx] if dev_train_idx else None
        dev_val_a = dev_val_a[dev_val_idx]
        gc.collect()

    def _combine(parts):
        return _concatenate_consuming([p for p in parts if p is not None])

    n_dev_train = 0 if dev_train_a is None else len(dev_train_a)
    n_dev_val = 0 if dev_val_a is None else len(dev_val_a)
    train_dataset = _combine([base_train_a, redteam_train_a, dev_train_a])
    validation_dataset = _combine([base_val_a, redteam_val_a, dev_val_a])
    if train_dataset is None:
        raise ValueError("No training data available to fit the probe.")

    if verbose:
        # Counts captured before _combine, which consumes the per-origin parts.
        print(
            f"Train/validation: {len(train_dataset)} train, "
            f"{0 if validation_dataset is None else len(validation_dataset)} validation "
            f"(base {n_by_origin[0]}+{n_by_origin[1]}; red-team "
            f"{n_by_origin[2]}+{n_by_origin[3]}; dev {n_dev_train}+{n_dev_val})"
        )

    # Park the activations ON THE GPU for the duration of the fits. This is the
    # single largest speedup available to a retrain, and it is worth understanding
    # why before touching it.
    #
    # tuberlens' fit reads batches through `ActivationDataset.__getitems__`, which
    # ends in `.to(self.device)`. With the tensors on the host that means every
    # epoch performs a scattered CPU gather of 11 MB rows AND pushes the whole set
    # across PCIe — for a 892-train/290-val retrain, ~13 GB per epoch, every epoch,
    # for ~780 epochs across the 10 ensemble members. Measured on this box at real
    # shapes (1024 x 5376, fp16, batch 16), one epoch costs:
    #
    #     host-resident, gather + H2D per batch    18.35 ms/sample
    #     GPU-resident, same random gather          0.16 ms/sample   (113x)
    #
    # which is the difference between a 4.5-hour retrain and a ~4-minute one. The
    # head itself (a Linear(5376->1) plus a masked softmax) is arithmetically free;
    # essentially all of the fit's cost was moving data. Note the gather is NOT the
    # thing to fix — on-device, random gather costs only ~30% more than perfectly
    # sequential reads (0.16 vs 0.11 ms/sample), so the sampler and its shuffling
    # are left exactly as they are.
    #
    # Nothing about the arithmetic changes: same indices, same order, same values,
    # so `.to(self.device)` simply becomes a no-op and the probes come out
    # bit-identical (verified against probe_iter0.pkl, member for member).
    #
    # The model has already been released above, so the card is empty; we still
    # fall back to host residency on OOM, since a large enough red-team set will
    # eventually outgrow the GPU and a slow fit beats a dead one.
    # What costs wall-clock is the per-epoch host->device copy, so when only one of the
    # two fits, the card should hold whatever moves the most bytes per epoch. That is
    # simply the larger set, and WHICH set that is depends on the configuration — a
    # 1908-row dev set dwarfs the training data, a 290-row one is dwarfed by it — so
    # _to_device_for_fit sorts by size rather than trusting the order here. Measured on
    # a 666-train/1908-dev retrain (gemma-3-27b, 24 GiB card):
    #
    #   larger set staged (19.6 GiB), 6.9 GiB copied per epoch  13.7 epochs/min
    #   both staged (26.4 GiB, oversubscribed via WSL paging)    5.3 epochs/min
    #   smaller set staged (6.9 GiB), 19.6 GiB copied per epoch  4.3 epochs/min
    #   host-resident (upstream behaviour before staging)       ~4.0 epochs/min
    #
    # See _to_device_for_fit on the ordering, the capacity check (rather than relying on
    # OOM) and the AGENTIC_REDTEAM_STAGE_ACTIVATIONS / _FIT_STAGING_RESERVE_GIB knobs.
    _to_device_for_fit([train_dataset, validation_dataset], verbose=verbose)

    def _build(fit_seed: int):
        # Reseed right before each fit so the fresh probe's random weight init is
        # determined solely by `fit_seed` + the (deterministic) training data,
        # regardless of how much the global RNG advanced earlier in the process (or
        # in an earlier ensemble member). See docstring.
        seed_everything(fit_seed)
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

    if not ensemble_seeds or len(ensemble_seeds) == 1:
        return _build(ensemble_seeds[0] if ensemble_seeds else seed)

    _warn_if_deterministic_arch(probe_spec, len(ensemble_seeds))
    # The members share their activations, so fitting them one at a time re-reads
    # those activations once per member — and for a head that is a single linear
    # projection, reading them is nearly all of the wall-clock. tuberlens'
    # `build_ensemble` steps the members together under vmap where it can and
    # hands back ordinary independent probes, so `EnsembleProbe`, the pickle and
    # every consumer are unchanged. It falls back to exactly this loop whenever
    # fusion does not apply, and it reads the activations from wherever
    # `_to_device_for_fit` left them rather than staging a second copy.
    #
    # `getattr` rather than a direct call: a tuberlens checkout predating
    # `build_ensemble` must still train ensembles, just one member at a time.
    #
    # `fusion_enabled()` (PROBE_FUSED_ENSEMBLE) selects between the two, and when it
    # is off the members are fit by the loop below rather than by `build_ensemble`'s
    # own sequential fallback. The two are not quite the same fit: that fallback
    # seeds `torch` alone, whereas `_build` calls `seed_everything`, which also seeds
    # `random`/`numpy` and pins the cuDNN flags. Turning fusion off is something one
    # does to compare against the pre-fusion runs, so it has to land on exactly the
    # path those runs took — a "sequential" mode that is itself a third behaviour
    # would answer a question nobody asked.
    build_ensemble = getattr(ProbeFactory, "build_ensemble", None)
    if build_ensemble is not None and fusion_enabled():
        # Seed once for the whole fused fit, for the same reason `_build` reseeds
        # per member: what the members are must depend on their own seeds and the
        # data, not on how far the ambient RNG advanced through earlier red-teaming
        # or an earlier retrain in this process. `build_ensemble` then re-seeds per
        # member internally.
        seed_everything(ensemble_seeds[0])
        members = build_ensemble(
            probe_spec=probe_spec,
            train_dataset=train_dataset,
            model_name=model_name,
            layer=layer,
            seeds=list(ensemble_seeds),
            validation_dataset=validation_dataset,
            pos_class_label=pos_class_label,
            neg_class_label=neg_class_label,
            probe_description=probe_description,
            verbose=verbose,
        )
    else:
        members = []
        for i, fit_seed in enumerate(ensemble_seeds):
            if verbose:
                print(
                    f"\n--- Ensemble member {i + 1}/{len(ensemble_seeds)} "
                    f"(training seed {fit_seed}) ---"
                )
            members.append(_build(fit_seed))
    if verbose:
        print(
            f"Built a {len(members)}-member score-averaging ensemble "
            f"(seeds {ensemble_seeds})"
        )
    return EnsembleProbe.from_members(members, ensemble_seeds)


@dataclass
class RetrainResult:
    new_probe_path: Path
    n_redteam_samples: int
    n_training_samples_total: int
    # Members behind the probe just written: 1 for an ordinary probe, n when the
    # retrain built a score-averaging deep ensemble.
    ensemble_size: int = 1


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
    model_name: str = "",
    combine_consecutive_messages: bool = False,
    convert_tool_to_assistant: bool = False,
    eval_data_description: str = "",
):
    """Convert red-team successes into a LabelledDataset, optionally preprocessing.

    With no preprocessing config this is the plain success→dataset conversion.
    With one, it mirrors the collation step of tuberlens' pipeline applied to the
    "extra" data: drop confounders (``filter_dataset``) then add contrastive
    pairs (``generate_contrastive_dataset``).

    ``model_name`` and the two transform flags build the :class:`TokenBudget` the
    contrastive generator regenerates over-long pairs against — they describe the
    *probe's* tokenization, which is what activation extraction truncates at 1024
    tokens. With no model name the length safeguard is simply inert.
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
        concept_description=preprocessing.concept_description,
        label_guidance=preprocessing.label_guidance,
        eval_data_description=eval_data_description,
        token_budget=TokenBudget(
            model_name=model_name,
            max_tokens=preprocessing.max_sample_tokens,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
        ),
    )
    if getattr(preprocessing, "keep_only_generated", False):
        # Keep the generated partners, drop the finds they were written from. The
        # generator has already run over every find (and the cache is keyed by the
        # source conversation), so this costs nothing and a run with the knob off
        # reuses the same pairs.
        kept = [d for d in dicts if d.get("is_generated")]
        if verbose:
            print(
                f"  [contrastive] keep_only_generated: {len(dicts)} → {len(kept)} "
                "records (source red-team finds dropped, generated partners kept)"
            )
        dicts = kept
    return _dicts_to_labelled_dataset(dicts, pos_label, neg_label)


def _dev_row_splits(dev_files: "list[Path]", dev_sizes: "list[int]") -> list[str]:
    """Per-row split stem for the concatenated dev set.

    ``_load_dev_dataset`` concatenates the splits in ``sorted(glob)`` order, each
    keeping its own row order, so a row's origin is recoverable from the per-file row
    counts alone.
    """
    out: list[str] = []
    for f, n in zip(dev_files, dev_sizes):
        out.extend([f.stem] * n)
    return out


def _dev_pair_key(row_inputs) -> str:
    """Group key that puts the two halves of a contrastive dev pair together.

    Several dev splits are built as pairs — one user request answered once each way, so
    the two rows differ only in the assistant's turn and carry opposite labels. Keying
    on the USER turns alone therefore collects a pair into one group. A split that is
    not paired simply yields one row per group, which is the unpaired behaviour.

    Lending whole pairs matters because the halves carry opposite labels: drawing rows
    independently hands early rungs of a dose ladder a class skew that has nothing to do
    with the supervision being measured.
    """
    parts = [
        str(getattr(m, "content", ""))
        for m in row_inputs
        if str(getattr(m, "role", "")) == "user"
    ]
    return "\u0000".join(parts) if parts else repr(row_inputs)


def _dev_lending_groups(
    dev_val,
    dev_files: "list[Path]",
    dev_sizes: "list[int]",
    dev_train_split: str = "",
    dev_train_unit: str = "rows",
) -> list[list[int]]:
    """The draw units the lending reserve is taken from, in dev-row order.

    ``dev_train_split`` restricts the pool to one dev split by file stem — the rest of
    the dev set is then never lent and validates throughout. Without it every row is
    eligible. ``dev_train_unit="pairs"`` groups eligible rows by :func:`_dev_pair_key`
    so a draw takes both halves of a contrastive pair at once; ``"rows"`` (the default)
    makes each row its own group.
    """
    eligible = range(len(dev_val))
    if dev_train_split:
        origins = _dev_row_splits(dev_files, dev_sizes)
        eligible = [i for i in eligible if origins[i] == dev_train_split]
        if not eligible:
            raise ValueError(
                f"dev_train_split {dev_train_split!r} matches no dev split; "
                f"available: {', '.join(sorted({f.stem for f in dev_files}))}"
            )
    if dev_train_unit not in ("rows", "pairs"):
        raise ValueError(
            f"dev_train_unit must be 'rows' or 'pairs', got {dev_train_unit!r}"
        )
    if dev_train_unit == "rows":
        return [[i] for i in eligible]
    grouped: dict[str, list[int]] = {}
    for i in eligible:
        grouped.setdefault(_dev_pair_key(dev_val.inputs[i]), []).append(i)
    return list(grouped.values())


def _dev_lending_indices(
    n_dev: int,
    dev_train_n: int,
    dev_train_reserve: int,
    seed: int,
    groups: "list[list[int]] | None" = None,
    verbose: bool = False,
) -> tuple[list[int], list[int]]:
    """Partition a dev set into rows lent to training and rows kept for validation.

    ``dev_train_reserve`` **groups** are drawn from a ``seed``-keyed permutation and
    leave validation **for the whole run**; this call trains on the first
    ``dev_train_n`` of them. A group is one row by default (``groups=None``) and a
    contrastive pair under ``dev_train_unit="pairs"`` — see :func:`_dev_lending_groups`.
    Two properties make the resulting series of retrains comparable:

    * **Nested.** Iteration k's training rows are a prefix of iteration k+1's, so the
      series is a dose ladder on one sample rather than a fresh draw each time.
    * **Fixed validation.** The reserve, not the currently-trained count, is what leaves
      validation — so the validation set is byte-identical at every iteration and the
      best-epoch checkpoints stay comparable. That is the one thing a dev set exists to
      provide, and removing only the trained rows would give it away again.

    Both counts **saturate** at the number of available groups rather than raising: a
    restricted pool (one split, and paired) is easily smaller than a schedule asks for,
    and the honest behaviour there is a ladder that flattens once the pool is exhausted
    — not a run that dies at iteration 3. Saturation is reported when ``verbose``.

    No index appears in both lists, so nothing is ever trained and validated on.
    """
    if groups is None:
        groups = [[i] for i in range(n_dev)]
    if dev_train_n > dev_train_reserve:
        raise ValueError(
            f"dev_train_n ({dev_train_n}) exceeds dev_train_reserve "
            f"({dev_train_reserve}) — the reserve is the pool it draws from."
        )
    n_groups = len(groups)
    reserve = min(dev_train_reserve, n_groups)
    train_n = min(dev_train_n, reserve)
    if verbose and (reserve < dev_train_reserve or train_n < dev_train_n):
        print(
            f"  dev lending pool exhausted: asked for {dev_train_n}/{dev_train_reserve} "
            f"groups, only {n_groups} exist — using {train_n}/{reserve}"
        )
    order = list(range(n_groups))
    random.Random(f"devtrain:{seed}").shuffle(order)
    reserved = order[:reserve]
    train_idx = sorted(i for g in reserved[:train_n] for i in groups[g])
    reserved_rows = {i for g in reserved for i in groups[g]}
    val_idx = sorted(i for i in range(n_dev) if i not in reserved_rows)
    if not val_idx:
        raise ValueError(
            f"The lending reserve covers all {n_dev} dev rows, leaving no validation set."
        )
    return train_idx, val_idx


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
    dev_data_path: str | Path | None = None,
    dev_train_n: int = 0,
    dev_train_reserve: int = 0,
    dev_train_split: str = "",
    dev_train_unit: str = "rows",
    seed: int = 0,
    base_data_fraction: float = 1.0,
    ensemble_size: int | None = None,
    base_activation_cache_dir: str | Path | None = None,
    combine_consecutive_messages: bool = False,
    convert_tool_to_assistant: bool = False,
    eval_data_description: str = "",
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
        eval_data_description: Optional description of the eval data (the config's
            `eval.data_description`). Passed to the contrastive generator, which is then
            told to keep every generated pair inside that data's form — the pairs become
            training data for a probe scored on it. Folded into the contrastive cache
            key, so editing it regenerates pairs rather than reusing older ones. Inert
            without `preprocessing`.
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
            Ignored (forced to 0.0) when `dev_data_path` is given.
        split_field: Optional field to keep grouped together when splitting (passed to
            stable_train_test_split). Ignored when `dev_data_path` is given.
        dev_data_path: A held-out dev set (a JSONL, or a directory of `*.jsonl` splits)
            to use as the validation set. When given, validation is that dev set
            **alone**: nothing is held out of the base data or the red-team successes,
            which both go entirely into training. This is what makes the validation set
            — and therefore the best-epoch checkpoint the probe fit selects — identical
            across iterations, instead of drifting as red-team successes accumulate into
            the held-out slice. The dev set must be disjoint from the eval splits, or
            early stopping is selecting on the test set. None keeps the old behaviour
            (validation = the `test_size` slice of base + red-team).
        dev_train_n: Number of dev rows to move from validation into TRAINING for this
            retrain (0 = none, the default and the behaviour before this argument
            existed). The rows are the first `dev_train_n` of a `seed`-permuted dev set,
            so successive iterations train on nested prefixes rather than fresh draws.
        dev_train_reserve: Number of dev rows withheld from validation for the whole run
            — the pool `dev_train_n` draws its prefix from. Must be >= `dev_train_n`.
            Reserving the *whole* run's rows at every iteration is what keeps validation
            identical across iterations: were only the currently-trained rows removed,
            the validation set would shrink each retrain and its checkpoints would stop
            being comparable, which is the one thing `dev_data_path` exists to prevent.
            No row is ever both trained and validated on.
        dev_train_split: Restrict the lending pool to one dev split, by file stem (e.g.
            "oig_omission"). The other splits are then never lent and validate
            throughout. "" (default) makes every dev row eligible.
        dev_train_unit: "rows" (default) draws individual dev rows; "pairs" draws both
            halves of a contrastive dev pair at once (rows sharing their user turns).
            Pairs matter on a paired split: the two halves carry opposite labels, so
            drawing rows independently gives early rungs of a dose ladder a class skew
            unrelated to the supervision being measured. Both counts saturate at the
            pool's size rather than raising.
        seed: Seed for the deterministic train/val split (stable_train_test_split), and
            for the dev permutation the arguments above index into.
        base_data_fraction: Fraction (0, 1] of the base training data to ingest,
            selected by a deterministic content-addressed random subsample
            (stable_fraction_subsample) applied *before* the train/val split.
            1.0 (default) uses all of it. The fraction is folded into the base
            activation cache key. Red-team successes are never subsampled.
        ensemble_size: Number of independently-seeded probes to fit (1..
            MAX_ENSEMBLE_SIZE) over the same activations, averaged into one
            `EnsembleProbe` whose score is the members' mean probability. None
            (default) *inherits* the base probe's ensemble size, so a retrain stays
            apples-to-apples with what was attacked — mirroring how `probe_spec=None`
            inherits its architecture. 1 writes a plain single probe.
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
        model_name=str(base_probe.model_name),
        combine_consecutive_messages=combine_consecutive_messages,
        convert_tool_to_assistant=convert_tool_to_assistant,
        eval_data_description=eval_data_description,
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

    # A dev set replaces the held-out slice entirely: base and red-team both train in
    # full (test_size 0.0 makes stable_train_test_split put everything on the train
    # side), and the dev set is the whole validation set.
    dev_val = None
    dev_files: list[Path] = []
    if dev_data_path is not None:
        if verbose:
            print(f"Validation set: held-out dev data at {dev_data_path}")
        dev_val, dev_files, dev_sizes = _load_dev_dataset(
            dev_data_path,
            pos_class_label,
            neg_class_label,
            combine_consecutive_messages,
            convert_tool_to_assistant,
            verbose,
        )
        if verbose:
            print(
                f"Dev validation samples: {len(dev_val)} "
                "(base + red-team data all train, nothing held out)"
            )
        test_size = 0.0
        split_field = None

    # Optionally lend part of the dev set to TRAINING. The reserved pool leaves
    # validation for the whole run — see the `dev_train_reserve` docstring — and this
    # call trains on its first `dev_train_n` rows, so iteration k's rows are a prefix of
    # iteration k+1's. Indices, not a resliced dataset: the dev activations are one
    # cached blob keyed on the dev files' bytes, and splitting *after* extraction keeps
    # that blob (and the Kaggle-assembled one the CLI prefetches) valid untouched.
    dev_train_idx: list[int] = []
    dev_val_idx: list[int] = []
    if dev_val is not None and dev_train_reserve > 0:
        dev_train_idx, dev_val_idx = _dev_lending_indices(
            len(dev_val),
            dev_train_n,
            dev_train_reserve,
            seed,
            groups=_dev_lending_groups(
                dev_val, dev_files, dev_sizes, dev_train_split, dev_train_unit
            ),
            verbose=verbose,
        )
        if verbose:
            print(
                f"Dev rows lent to training: {len(dev_train_idx)}"
                + (
                    f" (from {dev_train_split}, by {dev_train_unit})"
                    if dev_train_split
                    else ""
                )
                + f"; validation is the remaining {len(dev_val_idx)} "
                "(fixed for every iteration of this run)"
            )

    layer_used = layer if layer is not None else int(base_probe.layer)

    # Resolve the architecture for the new probe. Default (None): inherit the base probe's
    # architecture so the retrain stays apples-to-apples. A ProbeType name string or an
    # explicit ProbeSpec requests a fresh architecture instead.
    if probe_spec is None:
        probe_spec = _infer_probe_spec(base_probe)
    else:
        probe_spec = _coerce_probe_spec(probe_spec)

    # Same inherit-by-default rule as the architecture: an unspecified ensemble size
    # reproduces the probe that was actually red-teamed, so the retrain compares
    # like with like across iterations.
    if ensemble_size is None:
        ensemble_size = probe_ensemble_size(base_probe)
    ensemble_seeds = _resolve_ensemble_seeds(seed, ensemble_size)
    if verbose and ensemble_size > 1:
        print(
            f"Training a {ensemble_size}-member score-averaging deep ensemble "
            f"(training seeds {ensemble_seeds})"
        )

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

    dev_val_cache = None
    if dev_val is not None and base_activation_cache_dir is not None:
        dev_val_cache = _dev_activation_cache_path(
            base_activation_cache_dir,
            dev_files,
            base_probe.model_name,
            layer_used,
            combine_consecutive_messages,
            convert_tool_to_assistant,
        )

    new_probe = _train_with_cached_base_activations(
        base_train=base_train,
        base_val=base_val,
        redteam_train=redteam_train,
        redteam_val=redteam_val,
        dev_val=dev_val,
        model_name=base_probe.model_name,
        layer=layer_used,
        probe_spec=probe_spec,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
        base_train_cache=base_train_cache,
        base_val_cache=base_val_cache,
        dev_val_cache=dev_val_cache,
        dev_train_idx=dev_train_idx,
        dev_val_idx=dev_val_idx,
        redteam_cache_dir=base_activation_cache_dir,
        combine_consecutive_messages=combine_consecutive_messages,
        convert_tool_to_assistant=convert_tool_to_assistant,
        seed=seed,
        ensemble_seeds=ensemble_seeds,
        verbose=verbose,
    )

    with new_probe_path.open("wb") as f:
        pickle.dump(new_probe, f)
    if verbose:
        kind = (
            f"{ensemble_size}-member ensemble probe" if ensemble_size > 1 else "probe"
        )
        print(f"Saved retrained {kind} to {new_probe_path}")

    return RetrainResult(
        new_probe_path=new_probe_path,
        n_redteam_samples=n_redteam,
        n_training_samples_total=n_total + len(dev_train_idx),
        ensemble_size=ensemble_size,
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
    dev_data_path: str | Path | None = None,
    seed: int = 0,
    base_data_fraction: float = 1.0,
    ensemble_size: int = 1,
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
            Ignored (forced to 0.0) when `dev_data_path` is given.
        split_field: Optional field to keep grouped together when splitting. Ignored
            when `dev_data_path` is given.
        dev_data_path: A held-out dev set (a JSONL, or a directory of `*.jsonl` splits)
            used as the validation set instead of a slice of the base data, which then
            trains in full. Pass the same value here as to `retrain_probe` so iteration 0
            and every retrain select their best epoch against the same validation set.
        seed: Seed for the deterministic train/val split (stable_train_test_split).
        base_data_fraction: Fraction (0, 1] of the base training data to ingest,
            selected by a deterministic content-addressed random subsample
            (stable_fraction_subsample) applied before the train/val split. 1.0
            (default) uses all of it; the fraction is folded into the activation
            cache key.
        ensemble_size: Number of independently-seeded probes to fit (1..
            MAX_ENSEMBLE_SIZE) over the same activations, averaged into one
            `EnsembleProbe` whose score is the members' mean probability. 1
            (default) writes a plain single probe. Unlike `retrain_probe` there is
            no base probe to inherit this from, so the caller states it.
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

    # See retrain_probe: a dev set is the whole validation set, so nothing is held out.
    dev_val = None
    dev_files: list[Path] = []
    if dev_data_path is not None:
        if verbose:
            print(f"Validation set: held-out dev data at {dev_data_path}")
        dev_val, dev_files, dev_sizes = _load_dev_dataset(
            dev_data_path,
            pos_class_label,
            neg_class_label,
            combine_consecutive_messages,
            convert_tool_to_assistant,
            verbose,
        )
        if verbose:
            print(
                f"Dev validation samples: {len(dev_val)} "
                "(base data all trains, nothing held out)"
            )
        test_size = 0.0
        split_field = None

    base_train, base_val = stable_train_test_split(
        base_dataset, test_size=test_size, split_field=split_field, seed=seed
    )

    spec = _coerce_probe_spec(probe_spec or DEFAULT_FRESH_PROBE_ARCH)

    ensemble_seeds = _resolve_ensemble_seeds(seed, ensemble_size)
    if verbose and ensemble_size > 1:
        print(
            f"Training a {ensemble_size}-member score-averaging deep ensemble "
            f"(training seeds {ensemble_seeds})"
        )

    base_train_cache = base_val_cache = dev_val_cache = None
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
        if dev_val is not None:
            dev_val_cache = _dev_activation_cache_path(
                base_activation_cache_dir,
                dev_files,
                model_name,
                layer,
                combine_consecutive_messages,
                convert_tool_to_assistant,
            )

    probe = _train_with_cached_base_activations(
        base_train=base_train,
        base_val=base_val,
        redteam_train=None,
        redteam_val=None,
        dev_val=dev_val,
        model_name=model_name,
        layer=layer,
        probe_spec=spec,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
        base_train_cache=base_train_cache,
        base_val_cache=base_val_cache,
        dev_val_cache=dev_val_cache,
        seed=seed,
        ensemble_seeds=ensemble_seeds,
        verbose=verbose,
    )

    with new_probe_path.open("wb") as f:
        pickle.dump(probe, f)
    if verbose:
        kind = (
            f"{ensemble_size}-member ensemble probe" if ensemble_size > 1 else "probe"
        )
        print(f"Saved initial {kind} to {new_probe_path}")
    return new_probe_path


def _coerce_probe_spec(probe_spec):
    """ProbeSpec | str -> ProbeSpec. A string is treated as a ProbeType name with default hyperparams."""
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType

    if isinstance(probe_spec, str):
        return ProbeSpec(name=ProbeType(probe_spec), hyperparams={})
    return probe_spec


def _infer_probe_spec(base_probe):
    """Infer a ProbeSpec from a loaded probe object so we can train a fresh one of the same kind.

    An :class:`EnsembleProbe` is inspected through its first member: every member
    is built from the same ``ProbeSpec``, so member 0 answers for all of them.
    """
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

    if isinstance(base_probe, EnsembleProbe):
        base_probe = base_probe.members[0]

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
