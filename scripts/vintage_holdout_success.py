"""How often do the v2 probes still fail on the red-team samples v2 never saw?

The vintage sweep (``scripts/attribution_vintage.py``) answered *how much eval AUROC*
each iteration's red-team data bought. This answers the complementary question at the
level of individual conversations: take the **v2** probes — trained on the iteration-3
content of every pair whose source success already existed at iteration 2 — and put the
**v3-only** rows (the pairs that first appeared at iteration 3) in front of them. Those
rows are genuinely held out of every v2 fit, on both the train and the val side, so a
misclassification is exactly what the red-teamer was hunting for: a sample the probe of
the day gets wrong.

"Success rate among the seeds" is therefore, per held-out row, the fraction of the
``--seeds`` independently-initialised v2 probes that misclassify it. It separates two
things a single fit cannot: samples that beat *the architecture on this data* (10/10)
from samples that beat *one draw* (1/10 — the kind a single-seed red-team run reports as
a find and a rerun would not reproduce).

Thresholding. ``ProbeJudge.evaluate`` calls a row positive at ``predict_proba >= 0.5``
(``probe.threshold``, the config default both arms use), i.e. at ``logit >= 0``, and that
is the rule applied here. The logits come from ``probe._classifier.logits``, the same
forward path ``score_streaming`` uses for the eval splits — not the fp32 hand-rolled one
in ``attribution_lib``, which differs by ~1 ULP of bf16.

The v2 probes are **re-fit, not reloaded** (the sweep recorded AUROC and dropped the
probes). Refits are deterministic — same assembly, same ``seed_everything(seed)``, same
bit-identical fast path — so each one reproduces the sweep's probe for that seed. That is
not assumed: every fit re-scores one eval split and asserts it matches the AUROC recorded
in ``vintage_progress.jsonl`` to the last bit, and the run aborts if it does not.

Usage:
    AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/vintage_holdout_success.py \
        --arm gptoss120b --seeds 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A  # noqa: E402
import attribution_refit as R  # noqa: E402
import attribution_vintage as V  # noqa: E402

# The eval split used to prove a refit reproduces the sweep's probe: the smallest of the
# seven (114 rows), so the check costs seconds rather than the full seven-blob pass.
VERIFY_SPLIT = "oig_omission"


# --- scoring ----------------------------------------------------------------------


def logits_for(probe, dataset) -> np.ndarray:
    """Sequence logits for one dataset, through the probe's own batched forward."""
    import contextlib
    import io

    from tuberlens.interfaces.activations import Activation

    with contextlib.redirect_stdout(io.StringIO()):
        s = probe._classifier.logits(Activation.from_dataset(dataset))
    return s.float().cpu().numpy()


def verify_against_sweep(arm: str, seed: int, vintage: int, probe, asm: R.Assembled,
                         eval_dir: Path, progress: Path) -> str:
    """Assert this refit is the sweep's probe for ``(arm, vintage, seed)``.

    Compares one split's AUROC on both scales against the recorded value. Returns a short
    status string; raises if the sweep has the fit and the numbers differ.
    """
    import contextlib
    import io

    from tuberlens.interfaces.activations import Activation
    from tuberlens.interfaces.dataset import LabelledDataset

    recorded = None
    if progress.exists():
        for line in progress.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r["arm"], r["vintage"], r["seed"]) == (arm, vintage, seed):
                recorded = r
    if recorded is None:
        return "no recorded fit to compare against"

    ds = LabelledDataset.load_from(
        A.EVAL_DATASET_DIR / f"{VERIFY_SPLIT}.jsonl",
        pos_class_label=asm.probe.pos_class_label,
        neg_class_label=asm.probe.neg_class_label,
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    y = ds.labels_torch().float().cpu().numpy()
    blob = torch.load(eval_dir / f"{VERIFY_SPLIT}-acts_full.pt", weights_only=False, mmap=True)
    ds = ds.assign(
        activations=blob["activations"],
        attention_mask=blob["attention_mask"],
        input_ids=blob["input_ids"],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        s = probe._classifier.logits(Activation.from_dataset(ds))
    got = A.auroc_both(y, s.float().cpu().numpy())
    del ds, blob, s
    gc.collect()

    want = recorded["auroc"][VERIFY_SPLIT]
    for scale in ("pipeline", "rank"):
        if got[scale] != want[scale]:
            raise SystemExit(
                f"refit does NOT reproduce the sweep's {arm} v{vintage} seed {seed} probe: "
                f"{VERIFY_SPLIT} {scale} AUROC {got[scale]!r} vs recorded {want[scale]!r}. "
                f"Held-out success rates from a different probe would not answer the question."
            )
    return f"reproduces sweep probe ({VERIFY_SPLIT} AUROC {want['pipeline']:.6f})"


# --- one arm ----------------------------------------------------------------------


def _done_seeds(path: Path, arm: str) -> set[int]:
    if not path.exists():
        return set()
    out = set()
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["arm"] == arm:
            out.add(int(r["seed"]))
    return out


def run_arm(arm: str, seeds: list[int], *, iteration: int, vintage: int, drop_long: str,
            eval_dir: Path, out_dir: Path, sidecar: Path, sweep_progress: Path,
            resume: bool = True) -> None:
    keep, report = V.vintages(arm, iteration, drop_long)
    trained = sorted(keep[vintage])
    full = sorted(keep[iteration])
    heldout = sorted(set(full) - set(trained))

    print(f"\n=== {arm} ===", flush=True)
    print(
        f"  v{vintage}: {len(trained)} rows trained on; v{iteration}: {len(full)} rows; "
        f"held out of v{vintage} but in v{iteration}: {len(heldout)} rows",
        flush=True,
    )
    if not heldout:
        print("  nothing held out — nothing to measure", flush=True)
        return

    todo = [s for s in seeds if not (resume and s in _done_seeds(sidecar, arm))]
    if not todo:
        print(f"  all {len(seeds)} seeds already recorded", flush=True)
    else:
        print(f"  {len(todo)} fit(s) to run", flush=True)
        asm = V.assemble_train_only(arm, iteration)
        n_rows = len(asm.redteam)
        drop = set(range(n_rows)) - set(trained)
        y = asm.redteam.labels_torch().float().cpu().numpy()

        for seed in todo:
            t0 = time.time()
            probe, n_tr, n_val = R.refit(asm, drop_rows=drop, seed=seed)
            fit_s = time.time() - t0
            status = verify_against_sweep(arm, seed, vintage, probe, asm, eval_dir,
                                          sweep_progress)
            # Every red-team row is scored, not just the held-out ones: the rows v2 DID
            # train on cost nothing extra here and are the only honest reference for what
            # a "high" held-out failure rate means. Scored unsliced — a slice would copy
            # the whole (multi-GB) activation tensor for no benefit.
            s = logits_for(probe, asm.redteam)
            wrong = (s >= 0.0).astype(int) != y.astype(int)
            ho = np.array(heldout)
            print(
                f"  seed {seed}: fit {fit_s:5.0f}s  train={n_tr} val={n_val}  "
                f"held-out attack success {wrong[ho].mean():.3f} "
                f"({int(wrong[ho].sum())}/{len(ho)})  — {status}",
                flush=True,
            )
            row = {
                "arm": arm,
                "seed": seed,
                "vintage": vintage,
                "iteration": iteration,
                "drop_long": drop_long,
                "n_train": n_tr,
                "n_val": n_val,
                "fit_seconds": fit_s,
                "verify": status,
                "logits": [float(x) for x in s],
                "wrong": [int(x) for x in wrong],
            }
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            with sidecar.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            del probe
            gc.collect()
            torch.cuda.empty_cache()

        del asm
        gc.collect()

    # The metadata is cheap to rebuild and must exist even on a fully-resumed run.
    asm_meta = _metadata_only(arm, iteration)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{arm}_holdout_membership.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "vintage_trained": vintage,
                "iteration": iteration,
                "n_trained_rows": len(trained),
                "n_full_rows": len(full),
                "n_heldout_rows": len(heldout),
                "heldout_rows": heldout,
                "vintage_report": report,
                "rows": asm_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _metadata_only(arm: str, iteration: int) -> list[dict]:
    """Row provenance without touching the activations (no blobs, no GPU)."""
    import numpy as _np

    ds = A.load_redteam_dataset(arm, iteration)
    gen2src = A.generated_to_source(arm)
    labels = ds.other_fields["labels"]
    ids = list(ds.ids)
    rt_is_val = _np.array([A.is_val(m) for m in ds.inputs], dtype=bool)
    out = []
    for i, messages in enumerate(ds.inputs):
        key = A.canon(messages)
        out.append(
            {
                "row": i,
                "id": ids[i],
                "label": labels[i],
                "kind": "generated" if key in gen2src else "success",
                "pair": A.sha16(gen2src.get(key, key)),
                "split_side": "val" if bool(rt_is_val[i]) else "train",
            }
        )
    return out


# --- aggregation -------------------------------------------------------------------


def summarize(arms: list[str], *, iteration: int, vintage: int, drop_long: str,
              out_dir: Path, sidecar: Path) -> None:
    fits = defaultdict(list)
    if not sidecar.exists():
        print(f"no fits recorded yet ({sidecar} does not exist)")
        return
    for line in sidecar.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["arm"] in arms and r["vintage"] == vintage and r["iteration"] == iteration:
            fits[r["arm"]].append(r)

    per_row_csv = out_dir / "holdout_success_rows.csv"
    summary_csv = out_dir / "holdout_success_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    frows = per_row_csv.open("w", encoding="utf-8")
    frows.write(
        "arm,row,id,label,kind,split_side,pair,n_seeds,n_success,success_rate,"
        "mean_logit,min_logit,max_logit\n"
    )
    fsum = summary_csv.open("w", encoding="utf-8")
    fsum.write("arm,group,subgroup,n_rows,n_seeds,success_rate,sd_across_seeds,"
               "n_rows_always,n_rows_never\n")

    for arm in arms:
        rs = sorted(fits.get(arm, []), key=lambda r: r["seed"])
        if not rs:
            print(f"{arm}: no fits recorded", flush=True)
            continue
        keep, _ = V.vintages(arm, iteration, drop_long)
        trained = set(keep[vintage])
        full = sorted(keep[iteration])
        heldout = [i for i in full if i not in trained]
        meta = _metadata_only(arm, iteration)

        W = np.array([r["wrong"] for r in rs], dtype=float)     # (seeds, rows)
        L = np.array([r["logits"] for r in rs], dtype=float)
        seeds = [r["seed"] for r in rs]

        print(f"\n=== {arm} — {len(seeds)} v{vintage} probes (seeds {seeds[0]}..{seeds[-1]}) ===")
        print(f"held-out rows (in v{iteration}, not in v{vintage}): {len(heldout)}")

        def block(name: str, rows: list[int]) -> None:
            if not rows:
                return
            sub = W[:, rows]
            per_seed = sub.mean(axis=1)
            per_row = sub.mean(axis=0)
            fsum.write(
                f"{arm},{name.split('|')[0]},{name.split('|')[1] if '|' in name else ''},"
                f"{len(rows)},{len(seeds)},{per_row.mean()},"
                f"{per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0},"
                f"{int((per_row == 1.0).sum())},{int((per_row == 0.0).sum())}\n"
            )
            print(
                f"  {name:38s} n={len(rows):4d}  success {per_row.mean():.3f}  "
                f"(per-seed sd {per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0:.3f})  "
                f"always {int((per_row == 1.0).sum()):3d}  never {int((per_row == 0.0).sum()):3d}"
            )

        by_meta = {m["row"]: m for m in meta}
        block("held-out (v3 only)|all", heldout)
        for kind in ("success", "generated"):
            block(f"held-out (v3 only)|{kind}",
                  [i for i in heldout if by_meta[i]["kind"] == kind])
        for label in sorted({by_meta[i]["label"] for i in heldout}):
            block(f"held-out (v3 only)|label={label}",
                  [i for i in heldout if by_meta[i]["label"] == label])
        for side in ("train", "val"):
            block(f"held-out (v3 only)|side={side}",
                  [i for i in heldout if by_meta[i]["split_side"] == side])
        tr = sorted(i for i in trained if by_meta[i]["split_side"] == "train")
        va = sorted(i for i in trained if by_meta[i]["split_side"] == "val")
        block(f"v{vintage} rows|fit on (train side)", tr)
        block(f"v{vintage} rows|early-stopping (val side)", va)

        counts = Counter(int(c) for c in W[:, heldout].sum(axis=0))
        print("  per-row success count across seeds:")
        for k in range(len(seeds) + 1):
            n = counts.get(k, 0)
            bar = "#" * int(round(40 * n / max(1, len(heldout))))
            print(f"    {k:2d}/{len(seeds)} seeds: {n:4d} rows  {bar}")

        for i in full:
            m = by_meta[i]
            if i in trained:
                continue
            col = W[:, i]
            frows.write(
                f"{arm},{i},{m['id']},{m['label']},{m['kind']},{m['split_side']},{m['pair']},"
                f"{len(seeds)},{int(col.sum())},{col.mean()},"
                f"{L[:, i].mean()},{L[:, i].min()},{L[:, i].max()}\n"
            )

    frows.close()
    fsum.close()
    print(f"\nwrote {per_row_csv}\nwrote {summary_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--vintage", type=int, default=2, help="which vintage's probes to fit")
    ap.add_argument("--seed", type=int, default=A.SEED)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--drop-long", choices=("pair", "row", "none"), default="pair")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--eval-dir", type=Path, default=A.EVAL_ACTIVATIONS_DIR)
    ap.add_argument("--out-dir", type=Path,
                    default=A.REPO / "results_instructions_gemma27b_vintage")
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.seeds)]
    sidecar = args.out_dir / "holdout_progress.jsonl"
    sweep_progress = args.out_dir / "vintage_progress.jsonl"

    if not args.summarize_only:
        for arm in args.arm:
            run_arm(arm, seeds, iteration=args.iteration, vintage=args.vintage,
                    drop_long=args.drop_long, eval_dir=args.eval_dir,
                    out_dir=args.out_dir, sidecar=sidecar,
                    sweep_progress=sweep_progress, resume=args.resume)

    summarize(args.arm, iteration=args.iteration, vintage=args.vintage,
              drop_long=args.drop_long, out_dir=args.out_dir, sidecar=sidecar)


if __name__ == "__main__":
    main()
