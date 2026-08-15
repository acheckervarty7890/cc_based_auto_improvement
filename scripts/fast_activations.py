"""Make the vintage sweep's refits fast by fixing *where the activations live* — and
nothing else.

Why this exists
---------------
Profiling a refit shows the training loop is not training-bound. Per optimizer step the
probe does one 5376->1 matvec per token, which measures ~13 ms of forward+backward; the
same step spends ~170 ms gathering a 121 MB batch out of the host-resident activation
tensor and pushing it over PCIe as *pageable* memory. So ~93% of a fit is data plumbing.

The waste is structural: ``get_activations`` pads every conversation to the longest one in
the set (704 tokens for gptoss120b, 952 for nemotron) while the mean red-team conversation
is ~80 tokens. Nine tenths of every byte moved is padding.

What this does NOT do, and why
------------------------------
The obvious fix — trim each batch to its own real width — is **rejected**. Padded positions
carry ``z = 0`` and softmax weight ``p = 0``, so they contribute exact zeros to the pooled
logit and *look* free; but a shorter reduction regroups the real terms, and in bf16 that
moves the result. Measured over 40 random batches: 0 of 40 were bit-identical, worst output
delta 9.8e-4 and worst **gradient** delta 7.8e-3. A gradient perturbation that size,
compounded over 200 epochs, gives a different probe. The sweep's whole claim is that only
set membership varies between vintages, and it is validated against the original run's
committed CSV (v0 reproduces `iter0` to 4 decimals, v3 reproduces `iter3`), so a change that
perturbs the math is not available at any speed.

What this does instead
----------------------
Pure transfer mechanics, which are bit-exact by construction — the same values, the same
dtype, the same order of operations, only a different route to the GPU:

1. **Residency.** If the fit's activation tensors fit in free VRAM with headroom, they are
   moved to the GPU once. ``ActivationDataset.__getitems__``'s ``self.activations[indices]``
   then gathers on-device at ~270 GB/s and the ``.to(device)`` is a no-op. This removes the
   host gather *and* the PCIe transfer.
2. **Pinned staging.** When they do not fit (nemotron's v3 set is ~7.9 GB against an 8.2 GB
   card), the per-batch gather goes into a reusable pinned buffer and the copy is issued
   ``non_blocking``. Pinned H2D measured ~12x faster than pageable on this box.

Both paths keep the DataLoader's shuffle order untouched, so the sampler consumes the same
RNG stream and the batches are the same batches.

Enable with ``AGENTIC_FAST_ACTS=1``. ``scripts/verify_fast_activations.py`` re-runs an
already-recorded fit with it on and refuses to endorse it unless the AUROC matches the
recorded value exactly.
"""

from __future__ import annotations

import os

import torch

# Leave this much VRAM for the model, optimizer, autograd graph and the batch workspace.
_VRAM_HEADROOM_BYTES = 900 * 1024 * 1024

_patched = False


def enabled() -> bool:
    return os.environ.get("AGENTIC_FAST_ACTS", "") not in ("", "0", "false", "False")


def _free_vram() -> int:
    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def maybe_resident(dataset, label: str = "") -> bool:
    """Move one dataset's activation fields to the GPU if they fit. Returns whether they did.

    Bit-exactness: this is a device copy of the identical bytes. The subsequent gather and
    ``.to(dtype)`` cast run on the GPU either way — ``__getitems__`` already casts *after*
    ``.to(self.device)`` — so the arithmetic the probe sees is unchanged.
    """
    fields = ("activations", "attention_mask", "input_ids")
    tensors = [dataset.other_fields.get(f) for f in fields]
    if any(t is None for t in tensors) or any(t.is_cuda for t in tensors):
        return all(t is not None and t.is_cuda for t in tensors)

    need = sum(t.numel() * t.element_size() for t in tensors)
    free = _free_vram()
    if need + _VRAM_HEADROOM_BYTES > free:
        print(
            f"    [fast-acts] {label}: {need / 1e9:.2f} GB does not fit in {free / 1e9:.2f} GB "
            f"free VRAM — staying on host, pinned path",
            flush=True,
        )
        return False

    # Two sweep processes share this card, so `free` can be stale by the time the copy
    # runs — the sibling may have claimed the same bytes between the check and the move.
    # Losing that race must cost the fit its residency, not the fit.
    moved = {}
    try:
        for f in fields:
            moved[f] = dataset.other_fields[f].cuda(non_blocking=False)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        del moved
        torch.cuda.empty_cache()
        print(
            f"    [fast-acts] {label}: lost the VRAM race for {need / 1e9:.2f} GB — "
            f"staying on host, pinned path",
            flush=True,
        )
        return False

    dataset.other_fields.update(moved)
    print(
        f"    [fast-acts] {label}: {need / 1e9:.2f} GB resident on GPU "
        f"({free / 1e9:.2f} GB was free)",
        flush=True,
    )
    return True


def patch_getitems() -> None:
    """Route the host-resident batch gather through a reusable pinned staging buffer.

    ``ActivationDataset.__getitems__`` gathers the batch, moves it to the device and then
    splits it into per-sample tuples that the default collate immediately re-stacks. The
    split/re-stack is cheap; the pageable host->device copy is not. Gathering with
    ``torch.index_select(..., out=<pinned>)`` lets the copy be a pinned DMA instead.

    The buffer is keyed on (shape, dtype) and reused, so a fit allocates it once. Falls
    back to the original path for any tensor that is already on the GPU (the resident case)
    or whose layout the buffer cannot serve.
    """
    global _patched
    if _patched:
        return

    from tuberlens.interfaces.activations import ActivationDataset

    original = ActivationDataset.__getitems__

    def _staged(self, source: torch.Tensor, indices_t: torch.Tensor, cache_key: str):
        want = (indices_t.numel(),) + tuple(source.shape[1:])
        buf = getattr(self, cache_key, None)
        if buf is None or tuple(buf.shape) != want or buf.dtype != source.dtype:
            buf = torch.empty(want, dtype=source.dtype, pin_memory=True)
            setattr(self, cache_key, buf)
        torch.index_select(source, 0, indices_t, out=buf)
        return buf.to(self.device, non_blocking=True)

    def fast_getitems(self, indices: list[int]):
        acts_src = self.activations
        if acts_src.is_cuda:
            return original(self, indices)  # resident: the original path is already ideal

        idx = torch.as_tensor(indices, dtype=torch.long)
        try:
            batch_acts = _staged(self, acts_src, idx, "_pin_acts").to(self.dtype)
            batch_mask = _staged(self, self.attention_mask, idx, "_pin_mask")
            batch_ids = _staged(self, self.input_ids, idx, "_pin_ids").to(self.dtype)
        except RuntimeError:
            # Pinned allocation can fail on a host under memory pressure; a slow batch is
            # always better than a dead fit.
            return original(self, indices)

        batch_y = self.y[indices].to(self.device).to(self.dtype)
        torch.cuda.synchronize()  # the non_blocking copies must land before the buffer is reused
        return [
            (batch_acts[i], batch_mask[i], batch_ids[i], batch_y[i])
            for i in range(len(indices))
        ]

    ActivationDataset.__getitems__ = fast_getitems
    _patched = True
    print("    [fast-acts] pinned staging enabled for host-resident batches", flush=True)
