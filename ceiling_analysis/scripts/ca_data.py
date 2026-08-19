"""Memory-bounded access to the activation blobs.

The high-stakes side does not fit in RAM: `anthropic_hh_balanced` alone is ~33 GB of
layer-32 activations, the four eval splits together ~48 GB, and the dev set another ~21 GB.
A 62 GB box has to hold a training pool, a validation set and an eval pass at once, so
nothing here ever holds a whole blob as a live tensor.

Two ideas do all the work:

* **Blobs are memory-mapped and rows are materialized on demand.** `torch.load(mmap=True)`
  hands back a tensor backed by the file, so reading `idx` reads those rows and the pages
  behind them stay reclaimable. Scoring streams in chunks and never materializes a split.
* **A pool is allocated once and filled from the mmaps.** Building `base ∪ red-team ∪ N dev`
  by materializing each part and concatenating would hold parts and result at once — ~40 GB
  at the top of the high-stakes sweep. `build_pool` allocates the result and copies rows
  straight in, so the peak is the result alone.

Widths are the other half of the cost. `get_activations` pads every row of a call to that
call's longest row (capped at 1024), so a split whose longest conversation is 859 tokens
carries 859 columns for rows that are 79 tokens long. Every materialization trims the
sequence dimension to the longest *real* row it contains — lossless, since the attention
mask already zeroes the rest, and often a 2-5x saving.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

PAD_FIELDS = ("activations", "attention_mask", "input_ids")
SLAB = 64


@lru_cache(maxsize=64)
def _mmap(path: str) -> dict:
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


class Source:
    """Row-addressable activations plus the metadata dataset they belong to."""

    name: str
    dataset: object

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def dtypes(self) -> tuple:
        raise NotImplementedError

    def lengths(self) -> np.ndarray:
        raise NotImplementedError

    def slabs(self, idx: Sequence[int]) -> Iterable[tuple]:
        """Yield (acts, mask, ids) slabs of at most SLAB rows, in the order of `idx`."""
        raise NotImplementedError

    def width(self, idx: Sequence[int]) -> int:
        lens = self.lengths()
        return int(lens[list(idx)].max()) if len(idx) else 0

    def take(self, idx: Sequence[int]):
        idx = [int(i) for i in idx]
        return build_pool([(self, idx)])

    def chunks(self, size: int = SLAB) -> Iterable[tuple[list[int], object]]:
        for start in range(0, len(self), size):
            idx = list(range(start, min(start + size, len(self))))
            yield idx, self.take(idx)


class BlobSource(Source):
    """One published `.pt` blob (an eval or dev split, or a consolidated pool)."""

    def __init__(self, name: str, blob_path: Path, dataset):
        self.name = name
        self.blob_path = Path(blob_path)
        self.dataset = dataset
        self._len_cache = None
        blob = self.blob
        if blob["activations"].shape[0] != len(dataset):
            raise ValueError(
                f"{blob_path}: {blob['activations'].shape[0]} rows vs {len(dataset)} "
                f"in {name}"
            )

    @property
    def blob(self) -> dict:
        return _mmap(str(self.blob_path))

    @property
    def dim(self) -> int:
        return int(self.blob["activations"].shape[2])

    @property
    def dtypes(self) -> tuple:
        b = self.blob
        return (b["activations"].dtype, b["attention_mask"].dtype, b["input_ids"].dtype)

    def lengths(self) -> np.ndarray:
        # Summing a memory-mapped mask touches the whole mask file, so cache per instance.
        if getattr(self, "_len_cache", None) is None:
            self._len_cache = self.blob["attention_mask"].sum(-1).numpy()
        return self._len_cache

    def slabs(self, idx):
        """Yield slabs trimmed to each slab's own longest real row.

        Reading `activations[sl]` would fault in the full 1024-token width of every row —
        for `anthropic_hh_balanced`, whose mean conversation is 226 tokens, that is ~4.5x
        the bytes the rows actually hold. Slicing per row means only the pages behind the
        real tokens are ever touched, which is most of the I/O of an eval pass.
        """
        b = self.blob
        lens = self.lengths()
        idx = [int(i) for i in idx]
        for start in range(0, len(idx), SLAB):
            sl = idx[start : start + SLAB]
            w = int(max(lens[i] for i in sl))
            yield (
                torch.stack([b["activations"][i, :w] for i in sl]),
                torch.stack([b["attention_mask"][i, :w] for i in sl]),
                torch.stack([b["input_ids"][i, :w] for i in sl]),
            )


class MultiSource(Source):
    """Several blobs addressed as one dataset, in the order they were given.

    The dev set of a concept is four published per-split blobs; treating them as one
    source keeps the 25%/75% validation/training partition a single stratified draw over
    the whole dev set rather than four independent ones.
    """

    def __init__(self, name: str, parts: Sequence[Source], dataset):
        self.name = name
        self.parts = list(parts)
        self.dataset = dataset
        self.offsets = np.cumsum([0] + [len(p) for p in self.parts])
        if self.offsets[-1] != len(dataset):
            raise ValueError(f"{name}: {self.offsets[-1]} rows vs {len(dataset)} metadata")

    @property
    def dim(self) -> int:
        return self.parts[0].dim

    @property
    def dtypes(self) -> tuple:
        return self.parts[0].dtypes

    def lengths(self) -> np.ndarray:
        return np.concatenate([p.lengths() for p in self.parts])

    def slabs(self, idx):
        idx = [int(i) for i in idx]
        for start in range(0, len(idx), SLAB):
            sl = idx[start : start + SLAB]
            acts, mask, ids = [], [], []
            for g in sl:
                p = int(np.searchsorted(self.offsets, g, side="right") - 1)
                local = g - int(self.offsets[p])
                a, m, i = next(self.parts[p].slabs([local]))
                acts.append(a)
                mask.append(m)
                ids.append(i)
            w = max(a.shape[1] for a in acts)
            yield (
                _pad_cat(acts, w),
                _pad_cat(mask, w),
                _pad_cat(ids, w),
            )


def _pad_cat(tensors, width):
    out = torch.zeros((sum(t.shape[0] for t in tensors), width) + tuple(tensors[0].shape[2:]),
                      dtype=tensors[0].dtype)
    at = 0
    for t in tensors:
        out[at : at + t.shape[0], : t.shape[1]] = t
        at += t.shape[0]
    return out


def build_pool(parts: Sequence[tuple[Source, Sequence[int]]]):
    """`(source, row indices)` -> one activated LabelledDataset, allocated once and trimmed."""
    parts = [(src, [int(i) for i in idx]) for src, idx in parts if src is not None and len(idx)]
    if not parts:
        return None
    width = max(src.width(idx) for src, idx in parts)
    total = sum(len(idx) for _, idx in parts)
    a_dtype, m_dtype, i_dtype = parts[0][0].dtypes
    dim = parts[0][0].dim
    acts = torch.zeros((total, width, dim), dtype=a_dtype)
    mask = torch.zeros((total, width), dtype=m_dtype)
    ids = torch.zeros((total, width), dtype=i_dtype)
    metas = []
    at = 0
    for src, idx in parts:
        for a, m, i in src.slabs(idx):
            n, w = a.shape[0], min(width, a.shape[1])
            acts[at : at + n, :w] = a[:, :w]
            mask[at : at + n, :w] = m[:, :w]
            ids[at : at + n, :w] = i[:, :w]
            at += n
        metas.append(src.dataset[idx])
    cls = type(metas[0])
    keys = set(metas[0].other_fields)
    for m in metas[1:]:
        keys &= set(m.other_fields)
    other = {k: [v for m in metas for v in m.other_fields[k]] for k in keys
             if k not in PAD_FIELDS}
    other.update(activations=acts, attention_mask=mask, input_ids=ids)
    return cls(
        inputs=[i for m in metas for i in m.inputs],
        ids=[i for m in metas for i in m.ids],
        other_fields=other,
    )


def write_consolidated(path: Path, sources: Sequence[Source], model_name: str, layer: int):
    """Write several sources into one blob file, trimmed to the widest real row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = build_pool([(s, list(range(len(s)))) for s in sources])
    torch.save(
        {
            "activations": ds.other_fields["activations"],
            "attention_mask": ds.other_fields["attention_mask"],
            "input_ids": ds.other_fields["input_ids"],
            "model_name": model_name,
            "layer": layer,
        },
        path,
    )
    return ds
