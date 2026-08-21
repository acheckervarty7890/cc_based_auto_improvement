"""Dataset + activation loaders that work for BOTH concepts.

``analysis/ceiling/harness.py`` does this for the instruction-following run, but its
paths and class labels are module constants. Everything here takes an
:class:`~analysis.novelty.experiments.Experiment` instead, because the high-stakes run
has to be loaded too — and high-stakes cannot be loaded the same way. Its eval blobs
total 46 GiB and its dev blob is 19.6 GiB against 62 GiB of host RAM, so the pooling
path never materialises a whole split: :func:`pooled_vectors` reads each blob through
``mmap`` and reduces it in row chunks.

The loaders that *do* materialise (``load_eval_split`` / ``load_dev`` / ``load_base`` /
``load_redteam``) are for the fits, which need per-token activations. Call them one
split at a time on the heavy experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments import COMBINE, CONVERT, LAYER, MODEL, SEED, Arm, Experiment


# ------------------------------------------------------------------ datasets


def load_jsonl_dataset(path: Path, exp: Experiment):
    from tuberlens.interfaces.dataset import LabelledDataset

    return LabelledDataset.load_from(
        path,
        pos_class_label=exp.pos,
        neg_class_label=exp.neg,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )


def eval_blob(exp: Experiment, split: str) -> Path:
    return exp.eval_acts_dir / f"{split}-acts_full.pt"


def dev_blob(exp: Experiment) -> Path:
    from agentic_redteam.retrain import _dev_activation_cache_path, _load_dev_dataset

    _ds, files = _load_dev_dataset(exp.dev_dir, exp.pos, exp.neg, COMBINE, CONVERT, verbose=False)
    return _dev_activation_cache_path(exp.cache_dir, files, MODEL, LAYER, COMBINE, CONVERT)


def base_blob(exp: Experiment) -> Path:
    from agentic_redteam.retrain import _base_activation_cache_paths

    train, _val = _base_activation_cache_paths(
        exp.cache_dir, exp.base_data, MODEL, LAYER, SEED, 0.0, None, COMBINE, CONVERT, 1.0
    )
    return Path(train)


def _attach(ds, blob: Path, what: str):
    from tuberlens.model import LLMModel

    if not blob.exists():
        raise FileNotFoundError(f"{what}: missing activation blob {blob}")
    act = LLMModel.load_activations(blob)
    if act.activations.shape[0] != len(ds):
        raise ValueError(
            f"{what}: blob has {act.activations.shape[0]} rows, dataset has {len(ds)}"
        )
    return ds.assign(
        activations=act.activations,
        attention_mask=act.attention_mask,
        input_ids=act.input_ids,
    )


def load_eval_split(exp: Experiment, split: str):
    ds = load_jsonl_dataset(exp.eval_dir / f"{split}.jsonl", exp)
    return _attach(ds, eval_blob(exp, split), f"eval/{split}")


def load_eval_splits(exp: Experiment, splits: list[str] | None = None) -> dict:
    """All eval splits at once. Refuses on the heavy experiment -- 46 GiB."""
    if exp.heavy and splits is None:
        raise MemoryError(
            f"{exp.key} eval activations are ~46 GiB; load one split at a time "
            "(load_eval_split) or use pooled vectors."
        )
    return {s: load_eval_split(exp, s) for s in (splits or exp.splits())}


def load_dev(exp: Experiment):
    from agentic_redteam.retrain import _load_dev_dataset

    ds, _files = _load_dev_dataset(exp.dev_dir, exp.pos, exp.neg, COMBINE, CONVERT, verbose=False)
    return _attach(ds, dev_blob(exp), "dev")


def load_base(exp: Experiment):
    """The 50-row base training split (test_size 0.0 -> all of it trains)."""
    from agentic_redteam.retrain import stable_fraction_subsample, stable_train_test_split

    ds = load_jsonl_dataset(exp.base_data, exp)
    ds = stable_fraction_subsample(ds, 1.0, SEED)
    train, _val = stable_train_test_split(ds, test_size=0.0, split_field=None, seed=SEED)
    return _attach(train, base_blob(exp), "base")


# ------------------------------------------------------------------ red team


def redteam_dataset(exp: Experiment, arm: Arm, iteration: int | None = None):
    """Rebuild the postprocessed red-team snapshot exactly as the run held it.

    Not via ``load_from``: ``_dump_labelled_dataset`` writes tuberlens' canonical
    "positive"/"negative" in the ``label`` column rather than the probe's class-label
    strings, and the per-conversation activation cache is keyed on the *transformed*
    messages -- so the snapshot has to be rebuilt the way ``retrain_probe`` built it
    in memory or the cache would miss.
    """
    from agentic_redteam.retrain import _apply_message_transforms
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

    path = arm.redteam_jsonl(iteration)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ds = LabelledDataset(
        inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
        ids=[r["id"] for r in rows],
        other_fields={"labels": [r["label"] for r in rows]},
    )
    return _apply_message_transforms(ds, COMBINE, CONVERT)


def redteam_paths(exp: Experiment, ds) -> list[Path]:
    from agentic_redteam.retrain import _redteam_activation_cache_path

    return [
        _redteam_activation_cache_path(exp.cache_dir, msgs, MODEL, LAYER, COMBINE, CONVERT)
        for msgs in ds.inputs
    ]


def load_redteam(exp: Experiment, arm: Arm, iteration: int | None = None, keep: np.ndarray | None = None):
    """Red-team snapshot with cached activations attached.

    ``keep`` is an optional boolean mask or index array over the snapshot's rows --
    this is the ablation hook. Rows are emitted in snapshot (file) order; see
    ``README.md`` on why that is a deliberate, self-consistent choice rather than a
    reproduction of the run's own row order.
    """
    from agentic_redteam.retrain import _concatenate_consuming
    from tuberlens.model import LLMModel

    ds = redteam_dataset(exp, arm, iteration)
    paths = redteam_paths(exp, ds)
    idx = _resolve_keep(keep, len(ds))
    parts, misses = [], []
    for i in idx:
        p = paths[i]
        if not p.exists():
            misses.append(i)
            continue
        act = LLMModel.load_activations(p)
        parts.append(
            ds[i : i + 1].assign(
                activations=act.activations,
                attention_mask=act.attention_mask,
                input_ids=act.input_ids,
            )
        )
    if misses:
        raise FileNotFoundError(
            f"{len(misses)}/{len(idx)} red-team activation cache misses for "
            f"{arm.redteam_jsonl(iteration)}. Computing them needs gemma-3-27b; refusing."
        )
    if not parts:
        return None
    return _concatenate_consuming(parts)


def _resolve_keep(keep, n: int) -> list[int]:
    if keep is None:
        return list(range(n))
    keep = np.asarray(keep)
    if keep.dtype == bool:
        if len(keep) != n:
            raise ValueError(f"keep mask has {len(keep)} entries, snapshot has {n} rows")
        return [int(i) for i in np.flatnonzero(keep)]
    return [int(i) for i in keep]


def redteam_provenance(exp: Experiment, arm: Arm) -> dict:
    """Per-row iteration of first appearance, for the final snapshot.

    The postprocessed snapshots accumulate, so a row in ``iter_k`` that is absent from
    ``iter_{k-1}`` was introduced by iteration k. Rows are identified by their
    activation-cache hash (the transformed-message digest), which is exactly the
    identity the cache uses. Returns ``{"hash": [...], "first_iter": [...]}`` aligned
    to the final snapshot's row order.
    """
    seen: dict[str, int] = {}
    for it in range(1, arm.last_iteration + 1):
        p = arm.redteam_jsonl(it)
        if not p.exists():
            continue
        ds = redteam_dataset(exp, arm, it)
        for path in redteam_paths(exp, ds):
            seen.setdefault(path.stem, it)
    final = redteam_dataset(exp, arm)
    hashes = [p.stem for p in redteam_paths(exp, final)]
    return {"hash": hashes, "first_iter": [seen.get(h, arm.last_iteration) for h in hashes]}


# ------------------------------------------------------------------ pooling


def _masked_mean_from_blob(blob: Path, chunk: int = 32) -> np.ndarray:
    """Mask-weighted mean over real tokens -> (rows, hidden) float32.

    Read through ``mmap`` in row chunks so a 30 GiB blob costs a chunk of resident
    memory, not 30 GiB. Same reduction ``analysis/redteam_space_ens3_fast/pooled.npz``
    used, so vectors from the two are comparable.
    """
    obj = torch.load(blob, map_location="cpu", mmap=True, weights_only=False)
    acts, mask = obj["activations"], obj["attention_mask"]
    n = acts.shape[0]
    out = np.empty((n, acts.shape[-1]), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        a = acts[s:e].to(torch.float32)
        m = mask[s:e].to(torch.float32).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        out[s:e] = ((a * m).sum(dim=1) / denom).numpy()
        del a, m
    del obj, acts, mask
    return out


def pool_blob(blob: Path, chunk: int = 32) -> np.ndarray:
    return _masked_mean_from_blob(blob, chunk)


def pool_rows(paths: list[Path]) -> np.ndarray:
    """Pool a list of single-row blobs (the red-team per-conversation cache)."""
    vecs = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        vecs.append(_masked_mean_from_blob(p, chunk=1)[0])
    return np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)


def labels_of(ds) -> np.ndarray:
    return np.array([lab.to_int() for lab in ds.labels])


def jsonl_labels(exp: Experiment, path: Path) -> np.ndarray:
    """Labels of a split JSONL as ints, in file order, without touching activations."""
    ds = load_jsonl_dataset(path, exp)
    return labels_of(ds)
