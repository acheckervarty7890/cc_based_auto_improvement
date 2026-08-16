"""Score every probe of the vintage / red-team-only sweeps on the **other two concepts'**
eval sets — high-stakes (`eval_datasets/`) and harmful_to_human (`eval_dataset_hu_ha/`).

All three concepts' runs extract from the same frozen model at the same layer
(`google/gemma-3-27b-it` L32) under the same two loader transforms, so a conversation's
activation is the same tensor whichever concept's run computed it. That is what makes this
possible without a forward pass: `scripts/fetch_crossconcept_eval.py` pulls the published
blobs, and only the probe *head* differs.

What the number means
---------------------
These probes were trained to separate `assistant_follows_the_instruction` from
`assistant_does_not_follow_the_instruction`. Scoring them on a high-stakes or harm split
asks a transfer question — does the direction they learned also order *those* labels? —
so:

- AUROC is reported against the **target concept's own positive class** (`high-stakes`,
  `harmful_to_human`). 0.5 is chance; **below** 0.5 is real signal pointing the other way
  (the probe's "follows the instruction" end lines up with high-stakes / harmful), and is
  not a worse result than 0.5, it is a stronger one.
- Nothing here is thresholded. `probe.threshold` was calibrated for the training concept,
  so accuracy on a different label set would only measure how the two concepts' base rates
  happen to line up.

Two stages, both resumable
--------------------------
`fit`   — refit the 90 probes the reports name (base-only ×10, plus v2+base / v3+base /
          v2-alone / v3only-alone × 2 arms × 10 seeds) and pickle each one. Every refit is
          asserted to reproduce the sweep's probe for that key by re-scoring one
          `eval_instructions` split against the recorded AUROC, bit for bit — the same
          check `vintage_holdout_success.py` runs. A probe pickle is ~13 KB.
`score` — for each target split, load its (multi-GB) blob **once** and score every saved
          probe against it. Split-outer, probe-inner is the whole reason this is affordable:
          the reverse order would re-read 25 GB of blobs 90 times.

Usage:
    .venv_claude/bin/python scripts/crossconcept_eval.py --stage fit
    .venv_claude/bin/python scripts/crossconcept_eval.py --stage score
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A  # noqa: E402
import attribution_refit as R  # noqa: E402
import attribution_vintage as V  # noqa: E402
import fetch_crossconcept_eval as FC  # noqa: E402
import redteam_only_fits as RO  # noqa: E402

OUT_DIR = A.REPO / "results_instructions_gemma27b_vintage"
# The shared probe store, also written by redteam_only_fits.py under the same naming —
# see the note on PROBE_DIR there for why a fitted probe is kept rather than dropped.
PROBE_DIR = RO.PROBE_DIR
FIT_LOG = OUT_DIR / "crossconcept_fits.jsonl"
SCORE_LOG = OUT_DIR / "crossconcept_progress.jsonl"

# The split used to prove a refit reproduces the sweep's probe: the smallest of the seven
# instruction splits (114 rows), so the check costs seconds.
VERIFY_SPLIT = "oig_omission"

# condition -> (which sidecar holds the reference AUROC, how the training set is built)
#   "vintage" conditions include the base training data; "nobase" ones do not.
CONDITIONS = {
    "base_only": ("vintage", 0),
    "v2_base": ("vintage", 2),
    "v3_base": ("vintage", 3),
    "v2_alone": ("nobase", "v2"),
    "v3only_alone": ("nobase", "v3only"),
}
# base_only trains on the base data and no red-team rows at all, so it is the same probe
# for both arms (the vintage sweep's two v0 blocks match to the last digit). Fitted once
# under this label rather than twice.
SHARED_ARM = "shared"


def probe_path(arm: str, cond: str, seed: int) -> Path:
    return RO.probe_path(PROBE_DIR, arm, cond, seed)


def _sidecar_rows(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _recorded_auroc(cond: str, arm: str, seed: int) -> dict | None:
    """The sweep's recorded AUROC for this key, or None if it never fitted it."""
    kind, key = CONDITIONS[cond]
    if kind == "vintage":
        for r in _sidecar_rows(OUT_DIR / "vintage_progress.jsonl"):
            if (r["arm"], r["vintage"], r["seed"]) == (arm, key, seed):
                return r["auroc"][VERIFY_SPLIT]
    else:
        for r in _sidecar_rows(OUT_DIR / "redteam_only_progress.jsonl"):
            if (r["arm"], r["condition"], r["seed"]) == (arm, key, seed):
                return r["auroc"][VERIFY_SPLIT]
    return None


def _load_split(concept: str, split: str):
    """The split's dataset (labels only) and its positive-class vector."""
    from tuberlens.interfaces.dataset import LabelledDataset

    c = FC.CONCEPTS[concept]
    ds = LabelledDataset.load_from(
        c["dir"] / f"{split}.jsonl",
        pos_class_label=c["pos"],
        neg_class_label=c["neg"],
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    return ds, ds.labels_torch().float().cpu().numpy()


def _logits(probe, dataset) -> np.ndarray:
    from tuberlens.interfaces.activations import Activation

    with contextlib.redirect_stdout(io.StringIO()):
        s = probe._classifier.logits(Activation.from_dataset(dataset))
    return s.float().cpu().numpy()


# --- stage 1: refit and save ------------------------------------------------------


def _verify(probe, asm, arm: str, cond: str, seed: int, verify_ds) -> str:
    """Assert this refit is the probe the sweep reported for ``(arm, cond, seed)``."""
    want = _recorded_auroc(cond, "gptoss120b" if arm == SHARED_ARM else arm, seed)
    if want is None:
        return "no recorded fit to compare against"
    ds, y = verify_ds
    got = A.auroc_both(y, _logits(probe, ds))
    for scale in ("pipeline", "rank"):
        if got[scale] != want[scale]:
            raise SystemExit(
                f"refit does NOT reproduce {arm} {cond} seed {seed}: {VERIFY_SPLIT} "
                f"{scale} AUROC {got[scale]!r} vs recorded {want[scale]!r}. Cross-concept "
                f"numbers from a different probe would not answer the question."
            )
    return f"reproduces sweep probe ({want['pipeline']:.6f})"


def _verify_dataset(asm):
    """`oig_omission` with its activations attached — built once, reused by every check."""
    from tuberlens.interfaces.dataset import LabelledDataset

    ds = LabelledDataset.load_from(
        A.EVAL_DATASET_DIR / f"{VERIFY_SPLIT}.jsonl",
        pos_class_label=asm.probe.pos_class_label,
        neg_class_label=asm.probe.neg_class_label,
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    y = ds.labels_torch().float().cpu().numpy()
    blob = torch.load(A.EVAL_ACTIVATIONS_DIR / f"{VERIFY_SPLIT}-acts_full.pt",
                      weights_only=False, mmap=True)
    ds = ds.assign(activations=blob["activations"], attention_mask=blob["attention_mask"],
                   input_ids=blob["input_ids"])
    return ds, y


def fit_stage(arms: list[str], conds: list[str], seeds: list[int], iteration: int,
              drop_long: str) -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    # (label, arm whose assembly is used) — base_only is fitted once, under SHARED_ARM.
    plan: list[tuple[str, str, str, int]] = []
    for cond in conds:
        if cond == "base_only":
            for s in seeds:
                plan.append((SHARED_ARM, "gptoss120b", cond, s))
        else:
            for arm in arms:
                for s in seeds:
                    plan.append((arm, arm, cond, s))
    todo = [p for p in plan if not probe_path(p[0], p[2], p[3]).exists()]
    print(f"{len(plan)} probe(s) planned, {len(plan) - len(todo)} already on disk",
          flush=True)
    if not todo:
        return

    by_asm = defaultdict(list)
    for label_arm, asm_arm, cond, seed in todo:
        by_asm[asm_arm].append((label_arm, cond, seed))

    for asm_arm, jobs in by_asm.items():
        print(f"\n=== assembling {asm_arm} ===", flush=True)
        keep, _ = V.vintages(asm_arm, iteration, drop_long)
        asm = V.assemble_train_only(asm_arm, iteration)
        n_rows = len(asm.redteam)
        verify_ds = _verify_dataset(asm)

        for label_arm, cond, seed in jobs:
            kind, key = CONDITIONS[cond]
            t0 = time.time()
            if kind == "vintage":
                drop = set(range(n_rows)) - set(keep[key])
                probe, n_tr, n_val = R.refit(asm, drop_rows=drop, seed=seed)
            else:
                rows = RO.condition_rows(keep, key, iteration)
                probe, n_tr, n_val = RO.refit_redteam_only(asm, rows, seed)
            fit_s = time.time() - t0
            status = _verify(probe, asm, label_arm, cond, seed, verify_ds)
            path = probe_path(label_arm, cond, seed)
            with path.open("wb") as fh:
                pickle.dump(probe, fh)
            print(f"  {label_arm:11s} {cond:13s} seed {seed}: fit {fit_s:5.1f}s "
                  f"train={n_tr} val={n_val} — {status}", flush=True)
            with FIT_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"arm": label_arm, "condition": cond, "seed": seed,
                                     "n_train": n_tr, "n_val": n_val,
                                     "fit_seconds": fit_s, "verify": status,
                                     "path": str(path.relative_to(A.REPO))}) + "\n")
            del probe
            gc.collect()
            torch.cuda.empty_cache()

        del asm, verify_ds
        gc.collect()


# --- stage 2: score every saved probe on every target split -----------------------


def _done_score_keys() -> set[tuple[str, str, int, str]]:
    return {(r["arm"], r["condition"], int(r["seed"]), r["split"])
            for r in _sidecar_rows(SCORE_LOG)}


def score_stage(concepts: list[str]) -> None:
    probes = sorted(PROBE_DIR.glob("*.pkl"))
    if not probes:
        raise SystemExit("no probes on disk — run --stage fit first")
    done = _done_score_keys()
    print(f"{len(probes)} probe(s) on disk, {len(done)} (probe, split) pairs already scored",
          flush=True)

    for concept in concepts:
        c = FC.CONCEPTS[concept]
        for split in FC.splits_of(concept):
            blob_path = Path(c["cache"]) / f"{split}-acts_full.pt"
            if not blob_path.exists():
                raise SystemExit(f"missing {blob_path} — run fetch_crossconcept_eval.py")
            ds, y = _load_split(concept, split)
            # Load the blob ONCE for all probes. mmap keeps it in page cache rather than
            # anonymous memory, which matters for the 11 GB anthropic split.
            t0 = time.time()
            blob = torch.load(blob_path, weights_only=False, mmap=True)
            ds = ds.assign(activations=blob["activations"],
                           attention_mask=blob["attention_mask"],
                           input_ids=blob["input_ids"])
            print(f"\n=== {concept}/{split}: {len(ds)} rows, "
                  f"{int(y.sum())} positive ({c['pos']}) — blob in {time.time() - t0:.0f}s",
                  flush=True)

            for path in probes:
                arm, cond, seed_s = path.stem.split("__")
                seed = int(seed_s.removeprefix("seed"))
                if (arm, cond, seed, split) in done:
                    continue
                t1 = time.time()
                with path.open("rb") as fh:
                    probe = pickle.load(fh)
                res = A.auroc_both(y, _logits(probe, ds))
                row = {"arm": arm, "condition": cond, "seed": seed, "concept": concept,
                       "split": split, "n_rows": len(ds), "n_pos": int(y.sum()),
                       "auroc_pipeline": res["pipeline"], "auroc_rank": res["rank"],
                       "score_seconds": time.time() - t1}
                with SCORE_LOG.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                print(f"  {arm:11s} {cond:13s} seed {seed}: "
                      f"AUROC {res['pipeline']:.4f} (rank {res['rank']:.4f})  "
                      f"{time.time() - t1:.0f}s", flush=True)
                del probe
                gc.collect()

            del ds, blob
            gc.collect()
            torch.cuda.empty_cache()


# --- reporting --------------------------------------------------------------------


def summarize() -> None:
    rows = _sidecar_rows(SCORE_LOG)
    if not rows:
        print("nothing scored yet")
        return
    by = defaultdict(list)
    for r in rows:
        by[(r["concept"], r["arm"], r["condition"], r["split"])].append(r)

    order = list(CONDITIONS)
    for concept in sorted({r["concept"] for r in rows}):
        splits = sorted({r["split"] for r in rows if r["concept"] == concept})
        print(f"\n=== {concept} — AUROC vs {FC.CONCEPTS[concept]['pos']} "
              f"(mean +/- sd over seeds, pipeline scale) ===")
        head = f"{'arm':11s} {'condition':13s} " + " ".join(f"{s[:20]:>20s}" for s in splits) \
               + f"{'MEAN':>20s}"
        print(head)
        print("-" * len(head))
        for arm in sorted({r["arm"] for r in rows}):
            for cond in order:
                cells, means = [], []
                for sp in splits:
                    rs = by.get((concept, arm, cond, sp))
                    if not rs:
                        cells.append(f"{'-':>20s}")
                        continue
                    v = np.array([x["auroc_pipeline"] for x in rs])
                    cells.append(f"{v.mean():.4f}+/-{v.std(ddof=1):.4f}"
                                 if len(v) > 1 else f"{v.mean():.4f}")
                    means.append(v)
                if not means:
                    continue
                per_seed_mean = np.mean(np.stack(means), axis=0)
                cells.append(f"{per_seed_mean.mean():.4f}+/-{per_seed_mean.std(ddof=1):.4f}")
                print(f"{arm:11s} {cond:13s} " + " ".join(f"{c:>20s}" for c in cells))

    csv = OUT_DIR / "crossconcept_auroc.csv"
    with csv.open("w", encoding="utf-8") as fh:
        fh.write("concept,arm,condition,seed,split,n_rows,auroc_pipeline,auroc_rank\n")
        for r in sorted(rows, key=lambda x: (x["concept"], x["arm"],
                                             order.index(x["condition"]), x["split"],
                                             x["seed"])):
            fh.write(f"{r['concept']},{r['arm']},{r['condition']},{r['seed']},"
                     f"{r['split']},{r['n_rows']},{r['auroc_pipeline']},{r['auroc_rank']}\n")
    print(f"\nwrote {csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("fit", "score", "summarize"), required=True)
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
                    default=list(CONDITIONS))
    ap.add_argument("--concept", nargs="+", choices=sorted(FC.CONCEPTS),
                    default=sorted(FC.CONCEPTS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seed", type=int, default=A.SEED)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--drop-long", choices=("pair", "row", "none"), default="pair")
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.seeds)]
    if args.stage == "fit":
        fit_stage(args.arm, args.conditions, seeds, args.iteration, args.drop_long)
    elif args.stage == "score":
        score_stage(args.concept)
        summarize()
    else:
        summarize()


if __name__ == "__main__":
    main()
