"""Shared loader for the frozen activation-extraction LLM.

Every phase that needs activations — red-team scoring (``ProbeJudge``) and the
retrain's extraction step — loads the same tuberlens ``LLMModel``. On a gemma-sized
probe that load, not the arithmetic, is the run's dominant wall-clock cost, for one
avoidable reason:

**Only layers ``0..probe.layer`` are ever executed, but all of them are placed.**
tuberlens' ``HookedModel.__enter__`` truncates the executed stack to
``original_layers[:layer + 1]`` (``tuberlens/model.py:144``) — *inside* the context
manager, long after ``from_pretrained`` has dispatched the whole model. For
``google/gemma-3-27b-it`` at layer 32 that means 29 of 62 layers, **24 GB of bf16
weights**, are downloaded, placed and CPU/disk-offloaded despite never running a
single forward. Since ``device_map="auto"`` fills the GPU in layer order and spills
the remainder, those dead layers are what push the *executed* tail off the GPU:

    ==============================  ==========
    google/gemma-3-27b-it, layer 32   bf16 size
    ==============================  ==========
    per layer                          0.83 GB
    embeddings (tied with lm_head)     2.82 GB
    layers 0-32   (executed)          30.1 GB
    layers 33-61  (never executed)    24.0 GB
    total placed                      54.1 GB
    ==============================  ==========

``load_extraction_model`` rebuilds the config with ``num_hidden_layers = layer + 1``
so only the executed prefix is instantiated and only its weights are read out of the
checkpoint shards. This is **exact, not an approximation**: the stack is causal, so
layer 32's output is a function of layers 0..32 alone. Activations are bit-identical
to the untruncated model, which is why no activation cache key mentions truncation
and why blobs computed either way stay interchangeable.

Set ``AGENTIC_REDTEAM_TRUNCATE_LAYERS=0`` to disable (e.g. to A/B the timing, or if a
future architecture mis-handles the rebuilt config).

``AGENTIC_REDTEAM_MAX_MEMORY`` optionally pins accelerate's per-device budget, e.g.
``"0=21GiB,cpu=45GiB"``. Unset (the default) keeps tuberlens' ``max_memory=None``,
under which accelerate infers the budget from whatever is *free at load time* and
will silently fall back to **disk** offload when the box is tight — the state the
gemma-3-27b runs were in, and worth ~40 s/sample.

Truncating the config makes the model's module tree a strict subset of the
checkpoint's, and transformers' **disk**-offload bookkeeping does not tolerate that —
see ``_install_truncated_load_shims``, which is why a box tight enough to need
truncation is exactly the box on which it used to crash.
"""

from __future__ import annotations

import collections
import os
from typing import Any

_TRUNCATE_ENV = "AGENTIC_REDTEAM_TRUNCATE_LAYERS"
_MAX_MEMORY_ENV = "AGENTIC_REDTEAM_MAX_MEMORY"


def _truncation_enabled() -> bool:
    return os.environ.get(_TRUNCATE_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _parse_max_memory(raw: str) -> dict[int | str, str]:
    """Parse ``"0=21GiB,cpu=45GiB"`` into accelerate's ``max_memory`` mapping.

    GPU indices must be ints (accelerate keys devices by ordinal); ``cpu`` / ``disk``
    stay strings.
    """
    budget: dict[int | str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        device, _, size = item.partition("=")
        device, size = device.strip(), size.strip()
        if not size:
            raise ValueError(
                f"{_MAX_MEMORY_ENV}: expected 'device=size' entries, got {item!r}"
            )
        budget[int(device) if device.isdigit() else device] = size
    return budget


def _truncated_config(model_name: str, layer: int):
    """Return a config holding only layers ``0..layer``, or ``None`` to leave it alone.

    ``None`` means "nothing to do" — truncation disabled, the model already has no
    more layers than we execute, the config doesn't expose a layer count we recognise,
    or the config couldn't be read at all. Falling back to the full model is always
    correct, just slower, so nothing here is allowed to fail a load that would
    otherwise have worked.
    """
    if not _truncation_enabled():
        print(
            f"[model_loading] {_TRUNCATE_ENV} disables layer truncation — placing ALL "
            f"layers of {model_name} even though the probe only reads layer {layer}."
        )
        return None

    from transformers import AutoConfig
    from tuberlens.config import global_settings

    try:
        # cache_dir mirrors LLMModel.load's, so the config resolves from the same
        # place (and the same offline cache) as the weights.
        config = AutoConfig.from_pretrained(
            model_name, cache_dir=global_settings.CACHE_DIR
        )
    except Exception as exc:  # noqa: BLE001 — never block the load on this
        print(f"[model_loading] could not read config for {model_name}, loading all "
              f"layers: {exc}")
        return None

    # Multimodal checkpoints (gemma-3-*-it) nest the decoder under `text_config`;
    # text-only ones expose num_hidden_layers on the top-level config.
    text_config = getattr(config, "text_config", config)
    n_layers = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(n_layers, int):
        print(
            f"[model_loading] {type(config).__name__} for {model_name} exposes no int "
            f"num_hidden_layers (got {n_layers!r}) — loading all layers."
        )
        return None
    if n_layers <= layer + 1:
        # Benign: the probe reads at or past the last layer, so there is nothing to
        # drop. Not worth a line in the log.
        return None
    text_config.num_hidden_layers = layer + 1
    return config


_SHIMS_INSTALLED = False


def _install_truncated_load_shims() -> None:
    """Teach transformers' disk-offload bookkeeping about the layers we dropped.

    ``from_pretrained`` builds its offload index from the **checkpoint's** key list,
    not the model's: ``_get_key_renaming_mapping`` maps *every* serialized key, so
    layers our truncated config never instantiates are still in the ``weight_map``.
    Two helpers then look those names up in the ``device_map``, which only knows about
    modules that exist:

    * ``get_disk_only_shard_files`` walks a name's prefixes until one is a key of the
      map. For a dropped layer nothing ever matches, the walk bottoms out at ``""``,
      and it indexes with that — ``KeyError: ''``.
    * ``expand_device_map`` silently omits those names, so the ``disk_offload_index``
      comprehension built immediately after raises ``KeyError`` on them in turn.

    Neither is reached unless ``"disk" in device_map.values()``. That is the whole
    shape of the bug: truncation is free on a box roomy enough to hold layers
    ``0..layer`` across GPU+CPU, and crashes the load on the tight box it exists to
    help. Observed on the dev-sample extraction — ``google/gemma-3-27b-it`` at layer
    32 (30 GB of executed weights) against an 8 GB GPU and 15 GB of RAM.

    Both replacements are behaviour-preserving when there is no mismatch, so the
    (idempotent, process-wide) patch is safe to leave installed.
    """
    global _SHIMS_INSTALLED
    if _SHIMS_INSTALLED:
        return

    try:
        from transformers import modeling_utils
    except Exception as exc:  # noqa: BLE001 — never block a load on the shim
        print(f"[model_loading] could not patch transformers for truncation: {exc}")
        return

    inner_expand = getattr(modeling_utils, "expand_device_map", None)
    if inner_expand is None or not hasattr(modeling_utils, "get_disk_only_shard_files"):
        # A transformers that has restructured these away has also restructured the
        # bug away; the load either works or fails loudly on its own terms.
        print(
            "[model_loading] transformers exposes no expand_device_map / "
            "get_disk_only_shard_files — skipping the truncated-checkpoint shims."
        )
        _SHIMS_INSTALLED = True
        return

    def _device_of(device_map, weight_name):
        """The device holding ``weight_name``, or ``None`` if the model has no such
        module — i.e. it belongs to a layer the truncated config dropped."""
        while weight_name and weight_name not in device_map:
            weight_name = weight_name.rpartition(".")[0]
        return device_map.get(weight_name)

    class _DroppedAreMeta(dict):
        """``device_map`` expanded to parameters, tolerant of dropped ones.

        A miss means a checkpoint weight with no module in the truncated model.
        Reporting it as ``"meta"`` (never ``"disk"``) keeps it out of the caller's
        disk-offload index — which is right: there is nothing to offload it *to*, and
        nothing will ever ask for it. Iteration is unaffected, so the second caller
        (``caching_allocator_warmup``, which passes the model's own key list and
        therefore never misses) behaves exactly as before.
        """

        def __missing__(self, key: str) -> str:
            return "meta"

    def expand_device_map(device_map, param_names):
        return _DroppedAreMeta(inner_expand(device_map, param_names))

    def get_disk_only_shard_files(device_map, weight_map):
        """Shards from which nothing needs loading — now including shards that hold
        only dropped layers, which upstream would open and then discard key by key."""
        files_content = collections.defaultdict(list)
        for weight_name, filename in weight_map.items():
            files_content[filename].append(_device_of(device_map, weight_name))
        return [
            filename
            for filename, devices in files_content.items()
            # {"disk"} as upstream, plus None for dropped layers: a shard whose kept
            # weights are all disk-offloaded is read straight from the safetensors
            # file at forward time either way, and one with no kept weights at all
            # has nothing to contribute.
            if devices and set(devices) <= {"disk", None}
        ]

    modeling_utils.expand_device_map = expand_device_map
    modeling_utils.get_disk_only_shard_files = get_disk_only_shard_files
    _SHIMS_INSTALLED = True


def load_extraction_model(model_name: str, layer: int, *, verbose: bool = False):
    """Load the tuberlens ``LLMModel`` used to extract layer-``layer`` activations.

    Carries ``offload_buffers=True`` (when ``device_map="auto"`` offloads layers,
    buffers must offload too or accelerate warns about insufficient GPU buffer space
    and risks OOM), plus the layer truncation and optional ``max_memory`` pin
    documented at module level.

    **The one-line summary of what was placed is printed unconditionally**, not gated
    on ``verbose``. ``ProbeJudge._ensure_model`` — the red-team path, and the one whose
    forwards dominate a rotation — calls this with ``verbose=False``, so gating it hid
    the only evidence that truncation had silently not fired. A whole run does a
    handful of loads, so this costs a handful of lines.
    """
    from tuberlens.model import LLMModel

    model_kwargs: dict[str, Any] = {"offload_buffers": True}

    config = _truncated_config(model_name, layer)
    if config is not None:
        model_kwargs["config"] = config
        text_config = getattr(config, "text_config", config)
        placed = f"truncated to {text_config.num_hidden_layers} layers"
        # Only truncation can put checkpoint keys outside the model's module tree,
        # so the shims go in only when it actually fired.
        _install_truncated_load_shims()
    else:
        placed = "ALL layers (not truncated)"

    raw_budget = os.environ.get(_MAX_MEMORY_ENV, "").strip()
    if raw_budget:
        model_kwargs["max_memory"] = _parse_max_memory(raw_budget)
        budget = f"max_memory={model_kwargs['max_memory']}"
    else:
        budget = f"max_memory unpinned (set {_MAX_MEMORY_ENV} to pin it)"

    print(
        f"[model_loading] loading {model_name}, {placed} "
        f"(probe reads layer {layer}); {budget}"
    )

    return LLMModel.load(model_name, model_kwargs=model_kwargs)


def extraction_batch_size() -> int:
    """tuberlens' configured extraction batch size (``BATCH_SIZE``, default 1).

    Read through ``global_settings`` rather than hardcoded so the ``BATCH_SIZE`` env
    var reaches the red-team chunking in ``retrain`` as well as ``get_activations``.
    """
    from tuberlens.config import global_settings

    return max(1, int(global_settings.BATCH_SIZE))
