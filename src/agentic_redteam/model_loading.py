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
``"0=21GiB,cpu=45GiB"``. Fully unpinned, accelerate infers the budget from whatever is
*free at load time* and will silently fall back to **disk** offload when the box is
tight — the state the gemma-3-27b runs were in, and worth ~40 s/sample.

tuberlens now carries its **own** pin (``MAX_MEMORY`` / per-model ``MODEL_MAX_MEMORY``,
resolved by ``global_settings.get_max_memory``) plus an ``OFFLOAD_BUFFERS`` toggle, so
the budget is no longer something only this repo can set. Precedence here is
``AGENTIC_REDTEAM_MAX_MEMORY`` > tuberlens' settings > unpinned, and whichever won is
named in the printed load line — a log that reported "unpinned" while tuberlens had
quietly pinned it would defeat the point of printing it at all. The resolved budget is
passed explicitly to ``LLMModel.load`` so the line and the load can't disagree.

Everything here degrades to the old behaviour on a tuberlens checkout predating those
settings (``get_max_memory`` / ``OFFLOAD_BUFFERS`` are looked up defensively), so this
module works against both.
"""

from __future__ import annotations

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

    Delegates to tuberlens' ``parse_max_memory`` when it exists so the two entry points
    to the same budget can't drift apart in what they accept; the body below is the
    fallback for checkouts predating it, and is deliberately equivalent.
    """
    try:
        from tuberlens.config import parse_max_memory
    except ImportError:
        pass
    else:
        return parse_max_memory(raw) or {}

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


def _resolve_max_memory(model_name: str) -> tuple[dict[int | str, Any] | None, str]:
    """Resolve accelerate's per-device budget for ``model_name``, and say who set it.

    Precedence: this repo's ``AGENTIC_REDTEAM_MAX_MEMORY`` (which predates tuberlens'
    own knob and stays authoritative, so existing run scripts keep behaving), then
    tuberlens' ``MAX_MEMORY`` / ``MODEL_MAX_MEMORY`` settings, then unpinned.

    Returns ``(None, "")`` when nothing pinned it. The second element is only ever used
    for the log line.
    """
    raw = os.environ.get(_MAX_MEMORY_ENV, "").strip()
    if raw:
        return _parse_max_memory(raw), _MAX_MEMORY_ENV

    from tuberlens.config import global_settings

    # Absent on tuberlens checkouts predating the memory-pinning settings, where the
    # only budget was MODEL_MAX_MEMORY[DEFAULT_MODEL] and nothing here populated it.
    resolve = getattr(global_settings, "get_max_memory", None)
    if resolve is not None:
        budget = resolve(model_name)
        if budget:
            return budget, "tuberlens MAX_MEMORY/MODEL_MAX_MEMORY"
    return None, ""


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


def load_extraction_model(model_name: str, layer: int, *, verbose: bool = False):
    """Load the tuberlens ``LLMModel`` used to extract layer-``layer`` activations.

    Carries ``offload_buffers`` (when ``device_map="auto"`` offloads layers, buffers
    must offload too or accelerate warns about insufficient GPU buffer space and risks
    OOM) — taken from tuberlens' ``OFFLOAD_BUFFERS`` setting, which now owns that knob,
    defaulting to True as before on checkouts that don't have it. Plus the layer
    truncation and the ``max_memory`` pin documented at module level.

    **The one-line summary of what was placed is printed unconditionally**, not gated
    on ``verbose``. ``ProbeJudge._ensure_model`` — the red-team path, and the one whose
    forwards dominate a rotation — calls this with ``verbose=False``, so gating it hid
    the only evidence that truncation had silently not fired. A whole run does a
    handful of loads, so this costs a handful of lines.
    """
    from tuberlens.config import global_settings
    from tuberlens.model import LLMModel

    model_kwargs: dict[str, Any] = {
        "offload_buffers": getattr(global_settings, "OFFLOAD_BUFFERS", True)
    }

    config = _truncated_config(model_name, layer)
    if config is not None:
        model_kwargs["config"] = config
        text_config = getattr(config, "text_config", config)
        placed = f"truncated to {text_config.num_hidden_layers} layers"
    else:
        placed = "ALL layers (not truncated)"

    resolved, source = _resolve_max_memory(model_name)
    if resolved:
        # Passed explicitly even when tuberlens would have resolved the same thing, so
        # the line below reports the budget the load actually used.
        model_kwargs["max_memory"] = resolved
        budget = f"max_memory={resolved} (from {source})"
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
