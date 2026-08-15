#!/usr/bin/env python
"""Check that layer truncation survives a **disk**-offloaded load.

WHY THIS EXISTS SEPARATELY FROM A REAL RUN
    ``model_loading`` truncates the extraction LLM's config to layers ``0..probe.layer``,
    which makes the model's module tree a strict subset of the checkpoint's. transformers'
    disk-offload bookkeeping is built from the *checkpoint's* key list and does not
    tolerate that: the load dies with ``KeyError: ''`` inside
    ``get_disk_only_shard_files`` (see ``_install_truncated_load_shims``). The branch runs
    only when ``"disk" in device_map.values()``, so the failure needs a box too small to
    hold the executed prefix across GPU+CPU — for the probe this repo uses that means
    30 GB of gemma-3-27b weights, a 54 GB download and an hour before you learn anything.

    So it is reproduced here at a scale that runs in seconds: a randomly initialised
    ~1 M-parameter Llama, deliberately saved in many tiny shards (several of which end up
    holding only dropped layers), loaded with an explicit ``device_map`` that puts the
    last kept layer on ``"disk"``. No network, no HF token, no GPU.

WHAT IS ASSERTED
    1. the patched load succeeds;
    2. the kept layers' activations are **bit-identical** to the untruncated model's —
       the property every activation cache in this repo relies on, since no cache key
       mentions truncation.

    It also reports whether the *unpatched* load still reproduces the KeyError. That is
    informational, not a failure: a future transformers may fix it upstream, at which
    point the shims become no-ops rather than wrong.

Usage:  .venv_claude/bin/python scripts/check_truncated_disk_offload.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

KEEP = 4    # the "probe layer" is KEEP - 1
TOTAL = 8   # layers in the checkpoint
IDS = torch.tensor([[1, 5, 9, 13, 2]])


def _build(root: Path) -> Path:
    """A tiny checkpoint sharded finely enough that some shards are all dropped layers."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=TOTAL, num_attention_heads=4, num_key_value_heads=4,
        )
    )
    ckpt = root / "ckpt"
    model.save_pretrained(ckpt, max_shard_size="60KB", safe_serialization=True)
    assert (ckpt / "model.safetensors.index.json").exists(), (
        "need a SHARDED checkpoint — get_disk_only_shard_files is only reached for one"
    )
    return ckpt


def _device_map() -> dict[str, str]:
    """Force the disk branch deterministically, with no free-memory guesswork."""
    dm = {"model.embed_tokens": "cpu", "model.rotary_emb": "cpu",
          "model.norm": "cpu", "lm_head": "cpu"}
    for i in range(KEEP):
        dm[f"model.layers.{i}"] = "cpu"
    dm[f"model.layers.{KEEP - 1}"] = "disk"
    return dm


def _load_truncated(ckpt: Path, offload: Path):
    config = AutoConfig.from_pretrained(ckpt)
    config.num_hidden_layers = KEEP
    return AutoModelForCausalLM.from_pretrained(
        ckpt, config=config, device_map=_device_map(), offload_folder=str(offload),
        offload_buffers=True, torch_dtype=torch.float32,
    )


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="trunc-disk-"))
    try:
        ckpt = _build(root)
        offload = root / "offload"
        offload.mkdir()

        try:
            _load_truncated(ckpt, offload)
            print("unpatched : loaded (transformers no longer has the bug)")
        except KeyError as exc:
            print(f"unpatched : reproduced KeyError({exc})")

        from agentic_redteam.model_loading import _install_truncated_load_shims

        _install_truncated_load_shims()
        truncated = _load_truncated(ckpt, offload)
        print(f"patched   : loaded {truncated.config.num_hidden_layers} of {TOTAL} layers")

        full = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32)
        with torch.no_grad():
            got = truncated(IDS, output_hidden_states=True).hidden_states
            want = full(IDS, output_hidden_states=True).hidden_states
            # The truncated model's LAST hidden state has been through the final norm,
            # while the full model's hidden_states[KEEP] is layer KEEP-1's raw output.
            # Compare like with like or this "fails" on a perfectly correct load.
            want_last = full.model.norm(want[KEEP])

        ok = all(torch.equal(got[i], want[i]) for i in range(KEEP))
        ok_last = torch.equal(got[KEEP], want_last)
        print(f"layers 0..{KEEP - 1} bit-identical to untruncated : {ok}")
        print(f"layer {KEEP - 1} output (post-norm) bit-identical  : {ok_last}")
        if ok and ok_last:
            print("PASS")
            return 0
        print("FAIL")
        return 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
