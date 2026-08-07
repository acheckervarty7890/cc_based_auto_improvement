"""Stage 1: turn the on-disk activations into per-sample logits + Jacobians.

For one arm this reads the base train/val blobs, the per-conversation red-team blobs
and the four eval split blobs, and writes a single ``.npz`` holding, for every row,
its fp32 sequence logit ``s`` and its parameter Jacobian ``J = ds/dw`` at the
reference probe's weights. Everything downstream (influence, counterfactual AUROC)
is linear algebra on that file — no activations, no model, seconds per query.

The Jacobian is the exact derivative through the softmax pooling, not the
frozen-pooling approximation; see ``attribution_lib.sequence_logits_and_jacobians``.

Usage:
    .venv_claude/bin/python scripts/attribution_features.py --arm gptoss120b
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import attribution_lib as A


def _load_blob(path: Path, mmap: bool = False):
    blob = torch.load(path, weights_only=False, mmap=mmap)
    return blob["activations"], blob["attention_mask"]


def _rows_from_per_conversation_blobs(dataset, w, b, T, device, verbose=True):
    """Logits + Jacobians for a red-team dataset, one cached blob per conversation."""
    n = len(dataset)
    s_out = np.empty(n, dtype=np.float32)
    j_out = np.empty((n, w.numel()), dtype=np.float32)
    t0 = time.time()
    for i, messages in enumerate(dataset.inputs):
        acts, mask = _load_blob(A.redteam_blob_path(messages))
        s, j = A.sequence_logits_and_jacobians(
            acts, mask, w, b, T, device=device, batch_size=1
        )
        s_out[i], j_out[i] = s[0], j[0]
        if verbose and (i + 1) % 200 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"    {i + 1}/{n} rows  ({rate * 1000:.0f} ms/row)", flush=True)
    return s_out, j_out


def _eval_rows(probe, split, w, b, T, device):
    """Logits + Jacobians + labels for one eval split, from its cached full-split blob."""
    from tuberlens.interfaces.dataset import LabelledDataset

    dataset = LabelledDataset.load_from(
        A.EVAL_DATASET_DIR / f"{split}.jsonl",
        pos_class_label=probe.pos_class_label,
        neg_class_label=probe.neg_class_label,
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    y = dataset.labels_torch().float().cpu().numpy()
    # mmap: eval_balanced_refusal alone is 3.7 GB and only ever read in batches.
    acts, mask = _load_blob(A.EVAL_ACTIVATIONS_DIR / f"{split}-acts_full.pt", mmap=True)
    if acts.shape[0] != len(dataset):
        raise SystemExit(
            f"{split}: blob has {acts.shape[0]} rows, split has {len(dataset)}"
        )
    s, j = A.sequence_logits_and_jacobians(acts, mask, w, b, T, device=device, batch_size=4)
    return s, j, y, dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(A.ARMS), required=True)
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=A.REPO / "results_hu_harm_gemma27b_batch_ablation/attribution",
    )
    args = ap.parse_args()

    probe_path = A.ARMS[args.arm] / f"probe_iter{args.iteration}.pkl"
    probe = A.load_probe(probe_path)
    w, b, T = A.probe_params(probe)
    print(f"probe {probe_path.name}: {probe.model_name} L{probe.layer} "
          f"temperature={T} dim={w.numel()}")

    # --- training side -------------------------------------------------------
    redteam = A.load_redteam_dataset(args.arm, args.iteration)
    pairs, stats = A.build_pairs(args.arm, redteam)
    print(f"red-team: {stats['n_rows']} rows, {stats['n_pairs']} pairs "
          f"({stats['n_straddling']} straddle the train/val split, "
          f"{stats['n_fully_val']} sit entirely in val)")

    print("  red-team activations ...", flush=True)
    s_rt, j_rt = _rows_from_per_conversation_blobs(redteam, w, b, T, args.device)
    y_rt = np.array(
        [1.0 if lbl == "positive" else 0.0 for lbl in redteam.other_fields["labels"]],
        dtype=np.float32,
    )
    val_rt = np.array([A.is_val(m) for m in redteam.inputs], dtype=bool)

    base = A.load_base_dataset(probe)
    base_train, base_val = A.split_sides(base)
    base_train_blob, base_val_blob = A.base_blob_paths()
    print("  base activations ...", flush=True)
    s_base, j_base, y_base, val_base = [], [], [], []
    for ds, blob_path, is_val_side in (
        (base_train, base_train_blob, False),
        (base_val, base_val_blob, True),
    ):
        acts, mask = _load_blob(blob_path)
        if acts.shape[0] != len(ds):
            raise SystemExit(f"{blob_path.name}: {acts.shape[0]} rows vs {len(ds)}")
        s, j = A.sequence_logits_and_jacobians(acts, mask, w, b, T, args.device, 4)
        s_base.append(s)
        j_base.append(j)
        y_base.append(ds.labels_torch().float().cpu().numpy())
        val_base.append(np.full(len(ds), is_val_side, dtype=bool))

    # --- eval side -----------------------------------------------------------
    payload: dict[str, np.ndarray] = {}
    print("  eval activations ...", flush=True)
    for split in A.EVAL_SPLITS:
        s, j, y, dataset = _eval_rows(probe, split, w, b, T, args.device)
        payload[f"eval_{split}_s"] = s
        payload[f"eval_{split}_J"] = j
        payload[f"eval_{split}_y"] = y
        both = A.auroc_both(y, s)
        print(f"    {split:22s} n={len(y):4d} "
              f"AUROC pipeline={both['pipeline']:.5f} rank={both['rank']:.5f}")

    # --- assemble ------------------------------------------------------------
    payload.update(
        {
            "rt_s": s_rt,
            "rt_J": j_rt,
            "rt_y": y_rt,
            "rt_is_val": val_rt,
            "base_s": np.concatenate(s_base),
            "base_J": np.concatenate(j_base),
            "base_y": np.concatenate(y_base),
            "base_is_val": np.concatenate(val_base),
            "pair_of_row": np.array(
                [
                    next(
                        p.pair_id
                        for p in pairs
                        if idx in (p.source_idx, p.generated_idx)
                    )
                    for idx in range(len(redteam))
                ],
                dtype=np.int32,
            ),
            "pair_source_idx": np.array(
                [-1 if p.source_idx is None else p.source_idx for p in pairs],
                dtype=np.int32,
            ),
            "pair_generated_idx": np.array(
                [-1 if p.generated_idx is None else p.generated_idx for p in pairs],
                dtype=np.int32,
            ),
            "w": w.detach().cpu().numpy(),
            "b": np.float32(b),
            "T": np.float32(T),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.arm}_iter{args.iteration}_features.npz"
    np.savez_compressed(out, **payload)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
