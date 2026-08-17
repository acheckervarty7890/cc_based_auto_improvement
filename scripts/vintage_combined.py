"""Fit the HIGH-STAKES probe on the base data plus **both** attackers' red-team sets.

Companion to ``attribution_vintage.py``. That sweep fits each arm separately —
``deepseekv4pro`` and ``gptoss120b`` each get their own curve over vintages v0..v3. This
one pools them: at each vintage the training set is the base data plus the *union* of
both arms' iteration-3 pairs of that vintage. v3 is therefore base + every iteration-3
pair either attacker produced (1278 rows against 716 / 562 for the single arms), which
is the "combine v1, v2 and v3 for both attackers" set.

Everything that is not the row set is held at the cumulative sweep's values: the same
per-conversation cached activations, the same content-deterministic train/val split, the
same ``ProbeSpec`` read off ``probe_iter3.pkl`` (verified identical between the two arms
at load time), the same ten seeds, the same four ``eval_datasets/`` splits, the same
``--drop-overlong pair``. So the combined row is directly comparable to the two single-arm
rows in ``vintage/SUMMARY.md``.

Vintages are **cumulative**, as in that sweep: v_k is every iteration-3 pair whose source
success already existed at iteration k, so v3 ⊇ v2 ⊇ v1 and "v1 + v2 + v3" *is* v3. The
intermediate points are fitted anyway because they cost little and say whether pooling
helps at every set size or only at the end.

Why the assembly is disk-backed
-------------------------------
``attribution_vintage.build_side`` allocates the padded ``(n, width, 5376)`` fp16 tensor
in anonymous RAM. Pooled v3 is 1216 train rows at width 926 (12.1 GB) plus 262 val rows at
width 934 (2.6 GB), and both must be resident at once because the trainer validates every
epoch. That is 14.7 GB on a 15 GB box — the single-arm sweep's largest fit was 7.5 GB, which
is why it never had to care. :func:`build_side_mm` writes the same tensors into ``np.memmap``
files instead, so the pages are file-backed and evictable: RSS stays at whatever the kernel
chooses to cache rather than at the tensor size, and the shortfall costs page-cache misses
instead of the OOM killer. The tensors handed to ``ProbeFactory.build`` are ordinary
``torch`` tensors either way — only their storage differs — so the fit path is unchanged.

The v0 fits are a check on that claim, not filler: v0 has no red-team rows, so its training
set is byte-for-byte the one the single-arm sweep fitted, and ``--check-v0`` asserts this
script reproduces that sweep's recorded v0 AUROC to within 1e-9.

Usage:
    .venv_claude/bin/python scripts/vintage_combined.py --seeds 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

# tuberlens' trainer wraps every epoch in tqdm on stderr; a sweep would bury the log.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_vintage as V

ARM_ORDER = ["deepseekv4pro", "gptoss120b"]
COMBINED = "combined"
EMBED_DIM = V.EMBED_DIM

_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.bfloat16: None,  # torch.from_numpy has no bfloat16; never hit for these blobs
    torch.int64: np.int64,
    torch.int32: np.int32,
}


# --- the pooled row set -------------------------------------------------------------


def combined_vintages(iteration: int, drop_mode: str):
    """``(dataset, {vintage: row indices}, report)`` for the two arms pooled.

    The per-arm membership is ``attribution_vintage.vintages`` verbatim — this only
    concatenates the two arms' iteration-``iteration`` datasets and re-indexes. Both arms'
    rows are keyed by conversation content, and the red-team activation cache is
    content-keyed too (``retrain._redteam_activation_cache_path``), so a conversation both
    attackers happened to write would be one row with one blob. Measured: the two arms
    share **no** conversation at iteration 3, and the assert below keeps that from becoming
    a silent double-count if it ever changes.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    inputs, ids, labels = [], [], []
    keep: dict[int, list[int]] = {k: [] for k in range(iteration + 1)}
    report = {"membership": "cumulative", "drop_overlong": drop_mode, "arms": {}}
    seen: dict[str, str] = {}

    for arm in ARM_ORDER:
        exclude, drop_report = V.dropped_rows(arm, iteration, drop_mode)
        arm_keep, arm_report = V.vintages(arm, iteration, exclude, "cumulative")
        arm_report["overlong_drop"] = drop_report
        report["arms"][arm] = arm_report

        ds = A.load_redteam_dataset(arm, iteration)
        offset = len(inputs)
        for i, msgs in enumerate(ds.inputs):
            key = A.canon(msgs)
            assert key not in seen, (
                f"conversation shared by {seen[key]} and {arm} — pooling would "
                "double-count it; dedupe before fitting"
            )
            seen[key] = arm
        inputs.extend(ds.inputs)
        ids.extend(f"{arm}:{i}" for i in ds.ids)
        labels.extend(ds.other_fields["labels"])

        for k, rows in arm_keep.items():
            keep[k].extend(offset + i for i in rows)

    report["vintages"] = {
        k: {
            "n_rows": len(rows),
            "n_rows_by_arm": {
                arm: report["arms"][arm]["vintages"][k]["n_rows"] if k else 0
                for arm in ARM_ORDER
            },
        }
        for k, rows in keep.items()
    }
    dataset = LabelledDataset(
        inputs=inputs, ids=ids, other_fields={"labels": labels}
    )
    return dataset, keep, report


# --- disk-backed assembly -----------------------------------------------------------


class _Memmapped:
    """The three padded tensors of one side, backed by files rather than by RAM.

    Kept as one object so the ``np.memmap`` handles outlive the ``torch`` views — dropping
    them would unmap the storage the tensors point at.
    """

    def __init__(self, store: Path, tag: str):
        self.store = store
        self.tag = tag
        self._arrays: dict[str, np.memmap] = {}
        store.mkdir(parents=True, exist_ok=True)

    def empty(self, name: str, shape: tuple[int, ...], dtype: torch.dtype):
        np_dtype = _NP_DTYPE.get(dtype)
        if np_dtype is None:
            raise TypeError(f"no numpy equivalent for {dtype}")
        path = self.store / f"{self.tag}_{name}.mm"
        arr = np.memmap(path, dtype=np_dtype, mode="w+", shape=shape)
        self._arrays[name] = arr
        return torch.from_numpy(arr)

    def release(self) -> None:
        for arr in self._arrays.values():
            arr.flush()
        self._arrays.clear()


def build_side_mm(base_ds, base_blob_path: Path, redteam_ds, rows: list[int],
                  store: Path, tag: str):
    """``attribution_vintage.build_side`` with the result tensors on disk.

    Byte-for-byte the same content and the same fill discipline — every element written
    exactly once, real rows then an explicit zero of each row's pad region — only the
    allocation differs. ``np.memmap`` mode ``w+`` is sparse until touched, exactly like the
    ``torch.empty`` it replaces, so the write pattern still matters.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    base_blob = V._blob(base_blob_path)
    rt_paths = [A.redteam_blob_path(redteam_ds.inputs[i]) for i in rows]
    rt_widths = [A.blob_width(p) for p in rt_paths]

    n = len(base_ds) + len(rows)
    width = max([int(base_blob["activations"].shape[1])] + rt_widths)

    mm = _Memmapped(store, tag)
    acts = mm.empty("acts", (n, width, EMBED_DIM), base_blob["activations"].dtype)
    mask = mm.empty("mask", (n, width), base_blob["attention_mask"].dtype)
    ids_t = mm.empty("ids", (n, width), base_blob["input_ids"].dtype)

    def place(k: int, a, m, i):
        w = a.shape[0]
        acts[k, :w] = a
        mask[k, :w] = m
        ids_t[k, :w] = i
        if w < width:
            acts[k, w:] = 0
            mask[k, w:] = 0
            ids_t[k, w:] = 0

    for k in range(len(base_ds)):
        place(
            k,
            base_blob["activations"][k],
            base_blob["attention_mask"][k],
            base_blob["input_ids"][k],
        )
    del base_blob
    gc.collect()

    for k, path in enumerate(rt_paths, start=len(base_ds)):
        b = V._blob(path)
        place(k, b["activations"][0], b["attention_mask"][0], b["input_ids"][0])
        del b

    ds = LabelledDataset(
        inputs=list(base_ds.inputs) + [redteam_ds.inputs[i] for i in rows],
        ids=list(base_ds.ids) + [redteam_ds.ids[i] for i in rows],
        other_fields={
            "labels": list(base_ds.other_fields["labels"])
            + [redteam_ds.other_fields["labels"][i] for i in rows],
            "activations": acts,
            "attention_mask": mask,
            "input_ids": ids_t,
        },
    )
    return ds, mm


# --- v0 cross-check -----------------------------------------------------------------


def check_v0(out_dir: Path, reference: Path, tol: float = 1e-9) -> None:
    """Assert this script's v0 fits match the single-arm sweep's, seed by seed.

    v0 holds no red-team rows, so its training set does not depend on which arm — or how
    many — supplied them. The two sweeps must therefore agree exactly, and if they do, the
    disk-backed assembly and the pooled indexing are not perturbing the fit path. A
    mismatch here invalidates every other number this script prints.
    """
    ours = {
        (r["vintage"], r["seed"]): r
        for r in _read_progress(out_dir / "vintage_progress.jsonl")
        if r["vintage"] == 0
    }
    theirs = {}
    for r in _read_progress(reference / "vintage_progress.jsonl"):
        if r["vintage"] == 0:
            theirs.setdefault((0, r["seed"]), r)

    shared = sorted(set(ours) & set(theirs))
    if not shared:
        print("v0 check: no overlapping seeds recorded on both sides — skipped")
        return

    worst, worst_at = 0.0, None
    for key in shared:
        for sp in list(A.EVAL_SPLITS) + ["mean"]:
            d = abs(ours[key]["auroc"][sp]["pipeline"] - theirs[key]["auroc"][sp]["pipeline"])
            if d > worst:
                worst, worst_at = d, (key, sp)
    print(
        f"v0 check against {reference.name}: {len(shared)} seed(s), "
        f"max |delta| = {worst:.3e} at {worst_at}"
    )
    if worst > tol:
        raise SystemExit(
            "combined v0 does not reproduce the single-arm sweep's v0 — the pooled "
            "assembly is perturbing the fit path; aborting"
        )


def _read_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a row truncated by a hard kill; the fit simply re-runs
    return out


# --- main ---------------------------------------------------------------------------


def run(iteration: int, seeds: list[int], eval_dir: Path, out_dir: Path,
        drop_mode: str, resume: bool, only_vintages: list[int] | None,
        order: list[int], scratch: Path) -> None:
    from agentic_redteam.retrain import _infer_probe_spec

    redteam, keep, report = combined_vintages(iteration, drop_mode)

    print(f"\n=== {COMBINED} ({' + '.join(ARM_ORDER)}) ===", flush=True)
    for arm in ARM_ORDER:
        print(f"  {arm}: over-long drop = "
              f"{report['arms'][arm]['overlong_drop']}", flush=True)
    for k in sorted(keep):
        by_arm = report["vintages"][k]["n_rows_by_arm"]
        parts = " + ".join(f"{arm} {by_arm[arm]}" for arm in ARM_ORDER)
        print(f"  vintage {k}: {len(keep[k]):4d} rows ({parts})", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "combined_vintage_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    progress_path = out_dir / "vintage_progress.jsonl"
    done = {
        (r["vintage"], r["seed"]) for r in _read_progress(progress_path)
    } if resume else set()

    # The two arms' probe_iter3 carry the same ProbeSpec, labels, model and layer (they
    # came from one config), so either is the reference. Asserted rather than assumed:
    # pooling under two different architectures would be meaningless.
    probes = [A.load_probe(A.ARMS[arm] / f"probe_iter{iteration}.pkl") for arm in ARM_ORDER]
    specs = [_infer_probe_spec(p) for p in probes]
    assert specs[0] == specs[1], f"arms disagree on probe spec: {specs}"
    for attr in ("model_name", "layer", "pos_class_label", "neg_class_label", "description"):
        vals = {getattr(p, attr) for p in probes}
        assert len(vals) == 1, f"arms disagree on {attr}: {vals}"
    probe, probe_spec = probes[0], specs[0]

    base = A.load_base_dataset(probe)
    base_train, base_val = A.split_sides(base)
    btr_blob, bval_blob = A.base_blob_paths()
    rt_is_val = np.array([A.is_val(m) for m in redteam.inputs], dtype=bool)

    wanted = [k for k in order if k in keep]
    if only_vintages is not None:
        wanted = [k for k in wanted if k in only_vintages]

    for k in wanted:
        todo = [s for s in seeds if (k, s) not in done]
        if not todo:
            print(f"  vintage {k}: all {len(seeds)} seed(s) already recorded", flush=True)
            continue

        rows = keep[k]
        need_fit = [s for s in todo
                    if not V.fit_path(out_dir, COMBINED, k, s).exists()]

        # --- phase A: train, checkpointing each classifier as it lands ---------------
        if need_fit:
            tr_rows = [i for i in rows if not rt_is_val[i]]
            val_rows = [i for i in rows if rt_is_val[i]]

            t0 = time.time()
            train, train_mm = build_side_mm(
                base_train, btr_blob, redteam, tr_rows, scratch, f"v{k}_train"
            )
            val, val_mm = build_side_mm(
                base_val, bval_blob, redteam, val_rows, scratch, f"v{k}_val"
            )
            gb = (
                train.other_fields["activations"].nbytes
                + val.other_fields["activations"].nbytes
            ) / 1e9
            print(
                f"  vintage {k}: assembled train={len(train)} val={len(val)} "
                f"width={train.other_fields['activations'].shape[1]} "
                f"({gb:.1f} GB on disk, {time.time() - t0:.0f}s); "
                f"{len(need_fit)} fit(s) to run",
                flush=True,
            )

            for seed in need_fit:
                t1 = time.time()
                fitted = V.fit(train, val, probe, probe_spec, seed)
                fit_s = time.time() - t1
                V._save_fitted(
                    V.fit_path(out_dir, COMBINED, k, seed),
                    fitted,
                    {
                        "n_train": len(train),
                        "n_val": len(val),
                        "n_redteam_rows": len(rows),
                        "fit_seconds": fit_s,
                    },
                )
                print(
                    f"    seed {seed} vintage {k}: fit {fit_s:6.1f}s "
                    f"best_epoch={fitted._classifier.best_epoch}",
                    flush=True,
                )
                del fitted
                gc.collect()
                torch.cuda.empty_cache()

            # Freed BEFORE scoring so the 11.3 GB anthropic blob has page cache to live
            # in — that is what makes scoring ten seeds cost about as much as one. Here
            # it also releases the assembly's own page-cache footprint, which is the
            # larger of the two claims on it.
            del train, val
            train_mm.release()
            val_mm.release()
            _clear_scratch(scratch)
            gc.collect()
            torch.cuda.empty_cache()

        # --- phase B: score every un-recorded seed in one pass over the eval blobs ---
        loaded, meta = {}, {}
        for seed in todo:
            p, ckpt = V._load_fitted(probe, V.fit_path(out_dir, COMBINED, k, seed))
            loaded[seed] = p
            meta[seed] = ckpt
        t2 = time.time()
        scored = V.score_many(loaded, probe, eval_dir)
        print(f"  vintage {k}: scored {len(loaded)} seed(s) in {time.time() - t2:.0f}s",
              flush=True)
        for seed in todo:
            res = scored[seed]
            print(
                f"    seed {seed} v{k}: "
                + "  ".join(f"{sp:9s}={res[sp]['pipeline']:.4f}" for sp in A.EVAL_SPLITS)
                + f"  MEAN={res['mean']['pipeline']:.4f}",
                flush=True,
            )
            V._append_progress(
                progress_path,
                {
                    "arm": COMBINED,
                    "vintage": k,
                    "seed": seed,
                    "membership": "cumulative",
                    "drop_overlong": drop_mode,
                    "n_redteam_rows": len(rows),
                    "n_train": meta[seed]["n_train"],
                    "n_val": meta[seed]["n_val"],
                    "best_epoch": (
                        None
                        if meta[seed].get("best_epoch") is None
                        else int(meta[seed]["best_epoch"])
                    ),
                    "fit_seconds": meta[seed]["fit_seconds"],
                    "auroc": res,
                },
            )
        del loaded
        gc.collect()
        torch.cuda.empty_cache()


def _clear_scratch(scratch: Path) -> None:
    """Drop the assembly files — they are ~15 GB and specific to one vintage."""
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seed", type=int, default=A.SEED, help="first seed")
    ap.add_argument("--seeds", type=int, default=10, help="how many consecutive seeds")
    ap.add_argument("--vintages", nargs="*", type=int, default=None)
    ap.add_argument(
        "--vintage-order", default="0,3,1,2",
        help="order vintages are fitted in. v0 first because it is the cross-check "
             "against the single-arm sweep and costs seconds; then v3, the set the "
             "question is about, so a sweep cut short still has the headline.",
    )
    ap.add_argument("--drop-overlong", choices=("none", "row", "pair"), default="pair")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval-dir", type=Path, default=A.EVAL_ACTIVATIONS_DIR)
    ap.add_argument(
        "--out-dir", type=Path,
        default=A.REPO / "results_hs_gemma27b_batch_ablation/vintage_combined",
    )
    ap.add_argument(
        "--reference-dir", type=Path,
        default=A.REPO / "results_hs_gemma27b_batch_ablation/vintage",
        help="the single-arm sweep, for the v0 cross-check and the summary's context rows",
    )
    ap.add_argument(
        "--scratch-dir", type=Path, default=None,
        help="where the disk-backed assembly lives (default <out-dir>/_assembly). Needs "
             "~15 GB free at the largest vintage; deleted after each vintage's fits.",
    )
    ap.add_argument("--check-v0", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.seeds)]
    order = [int(x) for x in args.vintage_order.split(",") if x.strip() != ""]
    scratch = args.scratch_dir or (args.out_dir / "_assembly")

    try:
        run(args.iteration, seeds, args.eval_dir, args.out_dir, args.drop_overlong,
            args.resume, args.vintages, order, scratch)
    finally:
        _clear_scratch(scratch)

    if args.check_v0:
        check_v0(args.out_dir, args.reference_dir)

    # Report from the SIDECAR, not from this process's return values: on a resumed run
    # the rows computed before the restart are on disk and nowhere else.
    rows = [
        r for r in _read_progress(args.out_dir / "vintage_progress.jsonl")
        if args.vintages is None or r["vintage"] in args.vintages
    ]
    V.summarize(rows, args.out_dir)


if __name__ == "__main__":
    main()
