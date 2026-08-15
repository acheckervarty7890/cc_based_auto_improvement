"""Real probe refits on cached activations, with an arbitrary set of rows dropped.

This is the ground-truth arm of the attribution: it goes through the *actual*
``ProbeFactory.build`` path the pipeline uses, so a measured AUROC delta is the
delta the pipeline would have reported. What it skips is only the parts that cannot
change — preprocessing (the postprocessed set is already dumped) and activation
extraction (every blob is on disk).

The activations are assembled once and held, so the per-refit cost is the fit plus
the eval scoring; a caller sweeping hundreds of drop-sets pays the assembly once.

Usage:
    .venv_claude/bin/python scripts/attribution_refit.py --arm gptoss120b --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# tuberlens' trainer wraps every epoch and every scoring pass in tqdm, which writes to
# stderr — hundreds of refits would bury the log. Set before torch/tqdm are imported.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

import attribution_lib as A


@dataclass
class Assembled:
    """Everything a refit needs, held in host RAM across many refits."""

    probe: object
    probe_spec: object
    base_train: object
    base_val: object
    redteam: object          # full red-team set, activations attached, unsplit
    rt_is_val: np.ndarray
    eval_sets: dict          # split -> (dataset_with_activations, y)


def _attach(dataset, blob_path: Path):
    blob = torch.load(blob_path, weights_only=False)
    return dataset.assign(
        activations=blob["activations"],
        attention_mask=blob["attention_mask"],
        input_ids=blob["input_ids"],
    )


def _attach_redteam(dataset):
    """Attach per-conversation cached activations, padded to a common length.

    Blobs are stored at each conversation's own token length (see
    ``retrain._activate_redteam_cached``), so they have to be padded before they can
    be stacked — exactly what ``_concatenate_consuming`` does at retrain time.
    """
    acts, masks, ids = [], [], []
    for messages in dataset.inputs:
        blob = torch.load(A.redteam_blob_path(messages), weights_only=False)
        acts.append(blob["activations"])
        masks.append(blob["attention_mask"])
        ids.append(blob["input_ids"])
    width = max(a.shape[1] for a in acts)
    n, dim = len(acts), acts[0].shape[2]
    out_a = torch.zeros((n, width, dim), dtype=acts[0].dtype)
    out_m = torch.zeros((n, width), dtype=masks[0].dtype)
    out_i = torch.zeros((n, width), dtype=ids[0].dtype)
    for k, (a, m, i) in enumerate(zip(acts, masks, ids)):
        out_a[k, : a.shape[1]] = a[0]
        out_m[k, : m.shape[1]] = m[0]
        out_i[k, : i.shape[1]] = i[0]
    return dataset.assign(activations=out_a, attention_mask=out_m, input_ids=out_i)


def assemble(arm: str, iteration: int = 3, verbose: bool = True) -> Assembled:
    from tuberlens.interfaces.dataset import LabelledDataset

    from agentic_redteam.retrain import _infer_probe_spec

    probe = A.load_probe(A.ARMS[arm] / f"probe_iter{iteration}.pkl")
    base = A.load_base_dataset(probe)
    base_train, base_val = A.split_sides(base)
    btr_blob, bval_blob = A.base_blob_paths()
    base_train = _attach(base_train, btr_blob)
    base_val = _attach(base_val, bval_blob)

    redteam = A.load_redteam_dataset(arm, iteration)
    t0 = time.time()
    redteam = _attach_redteam(redteam)
    rt_is_val = np.array([A.is_val(m) for m in redteam.inputs], dtype=bool)
    if verbose:
        gb = redteam.other_fields["activations"].nbytes / 1e9
        print(f"  red-team activations: {len(redteam)} rows, {gb:.1f} GB "
              f"({time.time() - t0:.0f}s)")

    eval_sets = {}
    for split in A.EVAL_SPLITS:
        ds = LabelledDataset.load_from(
            A.EVAL_DATASET_DIR / f"{split}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        y = ds.labels_torch().float().cpu().numpy()
        ds = _attach(ds, A.EVAL_ACTIVATIONS_DIR / f"{split}-acts_full.pt")
        eval_sets[split] = (ds, y)

    return Assembled(
        probe=probe,
        probe_spec=_infer_probe_spec(probe),
        base_train=base_train,
        base_val=base_val,
        redteam=redteam,
        rt_is_val=rt_is_val,
        eval_sets=eval_sets,
    )


def refit(asm: Assembled, drop_rows: set[int] | None = None, seed: int = A.SEED,
          quiet: bool = True, include_base: bool = True):
    """Train one probe with ``drop_rows`` (red-team row indices) removed.

    ``include_base=False`` trains on the red-team rows **alone**, dropping the base
    training data from both sides of the split. Every caller in this directory wants the
    default: the base data is what the pipeline actually trains on, and holding it fixed
    is what makes a red-team drop-set the only thing that varies. The exception is
    ``heldout_v3_vs_v2_eval_tags.py --no-base``, which asks what a red-team vintage
    teaches *by itself* — there the base data is the confound, since it is identical
    across the conditions being contrasted and so is shared credit neither can be
    separated from.

    Note the val side loses the base rows too. Early stopping then selects against a
    purely red-team validation set, which is a different (and much smaller) selection
    signal — 142 rows rather than 150 on the largest condition here, 31 rather than 39 on
    the smallest. Fits are correspondingly noisier, and that is a property of the
    question, not a bug to tune away.
    """
    import contextlib
    import io

    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.evaluation import seed_everything
    from agentic_redteam.retrain import _concatenate_consuming

    drop = drop_rows or set()
    keep_train = [
        i for i in range(len(asm.redteam)) if not asm.rt_is_val[i] and i not in drop
    ]
    keep_val = [
        i for i in range(len(asm.redteam)) if asm.rt_is_val[i] and i not in drop
    ]
    # _concatenate_consuming pops the pad fields out of its inputs, so every part must
    # be a fresh view — asm.redteam is reused across hundreds of refits. (With one part
    # it returns that part unchanged, and an indexed view is already fresh, so the
    # include_base=False path leaves asm.redteam intact too.)
    train_parts = [asm.base_train[:]] if include_base else []
    val_parts = [asm.base_val[:]] if include_base else []
    train = _concatenate_consuming(train_parts + [asm.redteam[keep_train]])
    val = _concatenate_consuming(val_parts + [asm.redteam[keep_val]])

    seed_everything(seed)
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        probe = ProbeFactory.build(
            probe_spec=asm.probe_spec,
            train_dataset=train,
            model_name=asm.probe.model_name,
            layer=asm.probe.layer,
            validation_dataset=val,
            use_store=False,
            pos_class_label=asm.probe.pos_class_label,
            neg_class_label=asm.probe.neg_class_label,
            probe_description=asm.probe.description,
        )
    return probe, len(train), len(val)


def score(asm: Assembled, probe) -> dict[str, dict[str, float]]:
    """Per-split AUROC of one probe, on both the pipeline and rank scales."""
    import contextlib
    import io

    from tuberlens.interfaces.activations import Activation

    out = {}
    for split, (ds, y) in asm.eval_sets.items():
        with contextlib.redirect_stdout(io.StringIO()):
            s = probe._classifier.logits(Activation.from_dataset(ds))
        s = s.float().cpu().numpy()
        out[split] = A.auroc_both(y, s)
    for scale in ("pipeline", "rank"):
        out.setdefault("mean", {})[scale] = float(
            np.mean([out[s][scale] for s in A.EVAL_SPLITS])
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(A.ARMS), required=True)
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=1, help="refit the full set with N seeds")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"assembling {args.arm} iter{args.iteration} ...", flush=True)
    t0 = time.time()
    asm = assemble(args.arm, args.iteration)
    print(f"  assembled in {time.time() - t0:.0f}s")

    rows = []
    for k in range(args.seeds):
        seed = A.SEED + k
        t1 = time.time()
        probe, n_tr, n_val = refit(asm, seed=seed)
        fit_s = time.time() - t1
        t2 = time.time()
        res = score(asm, probe)
        print(f"seed {seed}: fit {fit_s:5.1f}s  score {time.time() - t2:4.1f}s  "
              f"best_epoch={probe._classifier.best_epoch}  train={n_tr} val={n_val}")
        print("   " + "  ".join(
            f"{s.replace('eval_', ''):14s}={res[s]['pipeline']:.5f}" for s in A.EVAL_SPLITS
        ))
        rows.append({"seed": seed, "fit_seconds": fit_s, "auroc": res,
                     "best_epoch": probe._classifier.best_epoch})

    if args.seeds > 1:
        print("\nper-split spread across seeds (pipeline scale):")
        for split in A.EVAL_SPLITS + ["mean"]:
            v = np.array([r["auroc"][split]["pipeline"] for r in rows])
            print(f"  {split:22s} mean={v.mean():.5f} sd={v.std(ddof=1):.5f} "
                  f"min={v.min():.5f} max={v.max():.5f} range={v.max() - v.min():.5f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
