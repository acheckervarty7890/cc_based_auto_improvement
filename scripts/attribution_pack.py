"""Build the GPU-resident packed activation sets one arm's refit sweep needs.

Row order matters and is not arbitrary: the reference retrain concatenates
``[base_train, redteam_train]`` (and likewise for val), and the trainer's shuffle is
a permutation of row *indices*, so reproducing the reference's trajectory requires
reproducing that order exactly.
"""

from __future__ import annotations

import numpy as np
import torch

import attribution_lib as A
from attribution_fasttrain import Packed, pack


def _blob_rows(path, n_expected=None):
    blob = torch.load(path, weights_only=False)
    acts, mask = blob["activations"], blob["attention_mask"]
    if n_expected is not None and acts.shape[0] != n_expected:
        raise SystemExit(f"{path}: {acts.shape[0]} rows, expected {n_expected}")
    for i in range(acts.shape[0]):
        yield acts[i], mask[i]


def _redteam_rows(dataset, indices):
    for i in indices:
        blob = torch.load(A.redteam_blob_path(dataset.inputs[i]), weights_only=False)
        yield blob["activations"][0], blob["attention_mask"][0]


def build(arm: str, iteration: int = 3, device: str = "cuda", verbose: bool = True):
    """Return ``(train, val, eval_packed, meta)`` for one arm.

    ``meta`` carries what a drop-set sweep needs to address rows: the red-team row
    indices in train order, the pair list, and each pair's train-side rows expressed
    as offsets into the packed training set.
    """
    probe = A.load_probe(A.ARMS[arm] / f"probe_iter{iteration}.pkl")
    redteam = A.load_redteam_dataset(arm, iteration)
    pairs, stats = A.build_pairs(arm, redteam)
    rt_is_val = np.array([A.is_val(m) for m in redteam.inputs], dtype=bool)
    rt_train_idx = np.flatnonzero(~rt_is_val)
    rt_val_idx = np.flatnonzero(rt_is_val)

    base = A.load_base_dataset(probe)
    base_train, base_val = A.split_sides(base)
    btr_blob, bval_blob = A.base_blob_paths()

    rt_y = np.array(
        [1.0 if l == "positive" else 0.0 for l in redteam.other_fields["labels"]],
        dtype=np.float32,
    )
    base_train_y = base_train.labels_torch().float().cpu().numpy()
    base_val_y = base_val.labels_torch().float().cpu().numpy()

    def chain(*iters):
        for it in iters:
            yield from it

    train = pack(
        chain(_blob_rows(btr_blob, len(base_train)), _redteam_rows(redteam, rt_train_idx)),
        np.concatenate([base_train_y, rt_y[rt_train_idx]]),
        device=device,
    )
    val = pack(
        chain(_blob_rows(bval_blob, len(base_val)), _redteam_rows(redteam, rt_val_idx)),
        np.concatenate([base_val_y, rt_y[rt_val_idx]]),
        device=device,
    )

    # Eval packs stay in host RAM: train+val already occupy most of an 8 GB card and
    # eval is touched once per fit, not once per epoch.
    from tuberlens.interfaces.dataset import LabelledDataset

    eval_packed = {}
    for split in A.EVAL_SPLITS:
        ds = LabelledDataset.load_from(
            A.EVAL_DATASET_DIR / f"{split}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        y = ds.labels_torch().float().cpu().numpy()
        pk = pack(
            _blob_rows(A.EVAL_ACTIVATIONS_DIR / f"{split}-acts_full.pt", len(ds)),
            y,
            device="cpu",
        )
        eval_packed[split] = (pk, y)

    # Map red-team dataset indices onto packed row positions. Both packs are laid out
    # base-first, matching the reference's [base, redteam] concatenation order.
    n_base_train, n_base_val = len(base_train), len(base_val)
    row_of_rt = {int(idx): n_base_train + k for k, idx in enumerate(rt_train_idx)}
    vrow_of_rt = {int(idx): n_base_val + k for k, idx in enumerate(rt_val_idx)}
    for p in pairs:
        p.packed_train_rows = [row_of_rt[i] for i in p.train_rows]  # type: ignore[attr-defined]
        p.packed_val_rows = [vrow_of_rt[i] for i in p.val_rows]  # type: ignore[attr-defined]

    meta = {
        "probe": probe,
        "pairs": pairs,
        "stats": stats,
        "n_base_train": n_base_train,
        "n_train": train.n,
        "n_val": val.n,
        "rt_train_idx": rt_train_idx,
    }
    if verbose:
        print(f"packed {arm}: train {train.n} rows / {train.gb:.2f} GB, "
              f"val {val.n} rows / {val.gb:.2f} GB, "
              f"eval {sum(pk.n for pk, _ in eval_packed.values())} rows "
              f"/ {sum(pk.gb for pk, _ in eval_packed.values()):.2f} GB (host)")
    return train, val, eval_packed, meta
