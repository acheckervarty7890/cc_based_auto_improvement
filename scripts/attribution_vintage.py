"""Retrain each arm's HIGH-STAKES probe on the iteration-1, -2 and -3 red-team *vintages*,
and score the eval AUROC of each — all from activations already on disk.

Port of ``experiment11_cloud``'s ``scripts/attribution_vintage.py`` to experiment9's
high-stakes arms, with two deliberate changes (see below): rows whose activations
overran the extraction window are dropped, and the assembly is rebuilt to fit a 15 GB
box.

The question this answers: **how much of the final probe's eval AUROC was already
bought by the red-team data that existed at iteration 1?** The committed
``*_comparison.csv`` cannot answer it, because ``probe_iter1`` was trained by a
different retrain than ``probe_iter3`` — different filter percentile draw and different
contrastive pairs. This script holds everything else fixed and varies only *which*
red-team conversations are in the training set, so the points differ in their data and
nothing else.

Why the unit of selection is the pair, not the row
--------------------------------------------------
``preprocessing`` turns each red-team success into two training rows — the success and
an LLM-written opposite-class counterpart — and ``retrain`` never sees one without the
other. Selecting rows individually would unbalance the labels and orphan half-pairs, so
membership is decided per **source success** and both of its rows follow. That is also
what makes the vintages comparable: every one of them is exactly 50/50 by construction.

Why membership is re-derived rather than read off the iteration-N dumps
----------------------------------------------------------------------
``redteam_postprocessed_iter{N}.jsonl`` is the verbatim training set of iteration N, but
its rows are **not** a subset of iteration 3's: ``filter_dataset`` refits its
bag-of-words classifier on the growing success set each cycle, so a different
top-percentile is dropped. A vintage here is therefore *"every iteration-3 pair whose
originating success was already present in iteration N"*, carrying **iteration 3's**
content — which is the only content whose activations exist. The vintages are
consequently **not strictly nested**, which is faithful to what the runs actually
trained on rather than an idealised add-only curve.

Dropping the over-long pairs (``--drop-overlong``)
--------------------------------------------------
``get_activations`` truncates at 1024 tokens (``tuberlens/model.py:433``), while the
contrastive generation prompt only asks for "similar structure and length" — so the
generator routinely writes a counterpart far longer than its source, and the tail that
gets cut is exactly the part carrying the opposite-class label. Measured here
(``vintage_length_report.py``): **every** over-cap row in both arms is a generated row —
14 of 295 pairs for gptoss120b, 6 of 364 for deepseekv4pro; not one attacker-written
source overran. Those rows train the probe on a conversation whose ending it cannot see.

``--drop-overlong pair`` (the default) removes the affected pair *whole*. Dropping the
generated row alone would leave an unpaired success and break the exact 50/50 balance
that makes the vintages comparable — the same reasoning that makes the pair the unit of
selection above. ``row`` drops only the offending row, ``none`` reproduces the
human-harm sweep's behaviour.

Over-length is read off the cached blob's width, not a tokenizer: ``_activate_redteam_cached``
writes each conversation at its own length, so the stored width is
``min(true_tokens, 1024)`` under the tokenization the extraction actually performed.

Memory, and the fit/score split
-------------------------------
Every fit needs the whole training side resident as one rectangular ``(n, width, 5376)``
fp16 tensor — ~11 MB per conversation at the full window. The reference implementation
holds three copies of that (the full red-team set, the selected subset, and the
concatenated result), which is why its docstring records a 31 GB box going into swap.
This box has 15 GB, so :func:`build_side` fills the result tensor **once**, in place,
straight from the per-conversation blobs, and the loop runs **vintage-outer / seed-inner**
so a vintage's tensors are built once and reused by all its seeds. The reference's
seed-outer order (which owns a complete curve at every prefix) is not affordable here;
``--vintage-order`` front-loads the two endpoints of the curve instead.

Scoring is then **batched over seeds**, which is where most of the wall-clock saving is.
The four high-stakes eval blobs total ~20 GB and anthropic alone is 11.3 GB, so scoring
one probe at a time — as the reference does — re-reads 20 GB per fit, 80 times. Instead
a vintage's training tensors are freed, then each eval blob is opened **once** and every
seed's probe is scored against it while it is warm in page cache. The probes are 5376
floats each, so holding ten costs nothing.

Splitting fit from score would make a crash cost up to ten fits rather than one, so each
fitted classifier's state dict is written to ``fits/`` the moment it is trained (21 KB).
A resumed run reloads those and goes straight to scoring. The normal path reloads them
too rather than passing the live object through, so the resume path is the exercised one
— :func:`_load_fitted` is verified against the live object by ``--self-check``.

Usage:
    .venv_claude/bin/python scripts/attribution_vintage.py --seeds 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# tuberlens' trainer wraps every epoch in tqdm on stderr; a sweep would bury the log.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A

EMBED_DIM = 5376


# --- over-long rows ---------------------------------------------------------------


def overlong_rows(arm: str, iteration: int) -> tuple[set[int], dict]:
    """Row indices whose cached activations hit the extraction window, and a report."""
    ds = A.load_redteam_dataset(arm, iteration)
    gen2src = A.generated_to_source(arm)
    widths = [A.blob_width(A.redteam_blob_path(m)) for m in ds.inputs]
    at_cap = {i for i, w in enumerate(widths) if w >= A.ACTIVATION_MAX_TOKENS}
    generated = {i for i, m in enumerate(ds.inputs) if A.canon(m) in gen2src}
    return at_cap, {
        "n_at_cap": len(at_cap),
        "n_at_cap_generated": len(at_cap & generated),
        "n_at_cap_source": len(at_cap - generated),
        "max_width": max(widths),
    }


def dropped_rows(arm: str, iteration: int, mode: str) -> tuple[set[int], dict]:
    """Rows excluded from every vintage, under ``none`` / ``row`` / ``pair``."""
    if mode == "none":
        return set(), {"mode": mode, "n_dropped": 0}
    at_cap, report = overlong_rows(arm, iteration)
    report["mode"] = mode
    if mode == "row":
        report["n_dropped"] = len(at_cap)
        return at_cap, report

    pairs, _ = A.build_pairs(arm, A.load_redteam_dataset(arm, iteration))
    drop: set[int] = set()
    n_pairs = 0
    for p in pairs:
        members = {i for i in (p.source_idx, p.generated_idx) if i is not None}
        if members & at_cap:
            drop |= members
            n_pairs += 1
    report["n_pairs_dropped"] = n_pairs
    report["n_dropped"] = len(drop)
    return drop, report


# --- vintage membership -----------------------------------------------------------


def _source_keys(arm: str, iteration: int, gen2src: dict[str, str]) -> list[str]:
    """The source-success key of every row of ``redteam_postprocessed_iter{N}.jsonl``.

    A row is either a success (its own key) or a generated counterpart (its source's).
    """
    from tuberlens.interfaces.dataset import Message

    path = A.ARMS[arm] / f"redteam_postprocessed_iter{iteration}.jsonl"
    keys = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            keys.append(
                A.canon(
                    [Message(role=m["role"], content=m["content"]) for m in row["inputs"]]
                )
            )
    return [gen2src.get(k, k) for k in keys]


def vintages(arm: str, iteration: int, exclude: set[int]) -> tuple[dict[int, list[int]], dict]:
    """``{vintage: iteration-3 row indices}`` plus a provenance report.

    ``vintage k`` is every row of the iteration-``iteration`` dump whose source success
    was already present in the iteration-``k`` dump, minus ``exclude``. Vintage
    ``iteration`` is therefore the whole (kept) set, and vintage 0 the empty list.
    """
    gen2src = A.generated_to_source(arm)
    final = _source_keys(arm, iteration, gen2src)
    report = {"arm": arm, "n_rows_final": len(final), "n_excluded": len(exclude),
              "vintages": {}}

    keep: dict[int, list[int]] = {0: []}
    for k in range(1, iteration + 1):
        seen = set(_source_keys(arm, k, gen2src))
        keep[k] = [
            i for i, src in enumerate(final) if src in seen and i not in exclude
        ]
        report["vintages"][k] = {
            "n_rows": len(keep[k]),
            "n_pairs_in_dump": len(seen),
            "n_pairs_kept": len({final[i] for i in keep[k]}),
            "n_pairs_dropped": len(seen - set(final)),
        }
    report["n_not_nested_1_in_2"] = len(
        {final[i] for i in keep[1]} - {final[i] for i in keep[2]}
    )
    return keep, report


# --- memory-lean assembly ---------------------------------------------------------


def _blob(path: Path):
    return torch.load(path, weights_only=False, mmap=True)


def build_side(base_ds, base_blob_path: Path, redteam_ds, rows: list[int]):
    """One side (train or val) as a single dataset with padded activations attached.

    The result tensor is allocated once and every source row is copied straight into
    its slice from the row's own blob, so peak host RAM is ~1x the result plus the one
    blob in flight — never the 3x the reference's
    ``asm.redteam[keep] -> _concatenate_consuming`` path costs.

    ``torch.empty`` over ``torch.zeros`` is load-bearing for the same reason it is in
    ``retrain._concatenate_consuming``: the allocation stays lazily faulted, so every
    byte must be written exactly once — the real rows, then an explicit zero-fill of
    each row's pad region.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    base_blob = _blob(base_blob_path)
    rt_paths = [A.redteam_blob_path(redteam_ds.inputs[i]) for i in rows]
    rt_widths = [A.blob_width(p) for p in rt_paths]

    n = len(base_ds) + len(rows)
    width = max([int(base_blob["activations"].shape[1])] + rt_widths)

    acts = torch.empty((n, width, EMBED_DIM), dtype=base_blob["activations"].dtype)
    mask = torch.empty((n, width), dtype=base_blob["attention_mask"].dtype)
    ids_t = torch.empty((n, width), dtype=base_blob["input_ids"].dtype)

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
        b = _blob(path)
        place(k, b["activations"][0], b["attention_mask"][0], b["input_ids"][0])
        del b

    return LabelledDataset(
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


def fit(train, val, probe, probe_spec, seed: int):
    """One real ``ProbeFactory.build`` on the assembled sides."""
    import contextlib
    import io

    from tuberlens.probes.probe_factory import ProbeFactory

    from agentic_redteam.evaluation import seed_everything

    seed_everything(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return ProbeFactory.build(
            probe_spec=probe_spec,
            train_dataset=train,
            model_name=probe.model_name,
            layer=probe.layer,
            validation_dataset=val,
            use_store=False,
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
            probe_description=probe.description,
        )


# --- fitted-probe checkpoints -----------------------------------------------------


def fit_path(out_dir: Path, arm: str, vintage: int, seed: int) -> Path:
    return out_dir / "fits" / f"{arm}_v{vintage}_s{seed}.pt"


def _save_fitted(path: Path, fitted, meta: dict) -> None:
    """Persist one trained classifier (21 KB) so a crash costs one fit, not ten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "state_dict": {
                k: v.cpu() for k, v in fitted._classifier.model.state_dict().items()
            },
            "best_epoch": fitted._classifier.best_epoch,
            **meta,
        },
        tmp,
    )
    tmp.replace(path)


def _load_fitted(ref_probe, path: Path):
    """Rebuild a scoreable probe from a checkpoint.

    A deep copy of the reference probe carries the identical architecture,
    ``training_args`` and metadata — only the weights differ — so loading the state
    dict into it reproduces the fitted probe's forward exactly. Used on the normal path
    as well as on resume, so the resume path is the one that gets exercised.
    """
    import copy

    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    probe = copy.deepcopy(ref_probe)
    probe._classifier.model.load_state_dict(ckpt["state_dict"])
    probe._classifier.model = probe._classifier.model.to(
        probe._classifier.device
    ).to(probe._classifier.dtype)
    probe._classifier.best_epoch = ckpt.get("best_epoch")
    return probe, ckpt


# --- eval scoring, one blob at a time, all seeds at once ---------------------------


def score_many(probes: dict[int, object], ref_probe, eval_dir: Path) -> dict[int, dict]:
    """Per-split AUROC for several probes, opening each eval blob exactly once.

    The four high-stakes blobs total ~20 GB (anthropic alone is 11.3 GB), so they are
    never all resident: ``mmap=True`` keeps each in page cache rather than anonymous
    memory, and each is dropped before the next split is opened. Scoring every seed
    against a blob while it is open is what turns 80 passes over 20 GB into 8.
    """
    import contextlib
    import io

    from tuberlens.interfaces.activations import Activation
    from tuberlens.interfaces.dataset import LabelledDataset

    out: dict[int, dict] = {seed: {} for seed in probes}
    for split in A.EVAL_SPLITS:
        ds = LabelledDataset.load_from(
            A.EVAL_DATASET_DIR / f"{split}.jsonl",
            pos_class_label=ref_probe.pos_class_label,
            neg_class_label=ref_probe.neg_class_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        y = ds.labels_torch().float().cpu().numpy()
        blob = _blob(eval_dir / f"{split}-acts_full.pt")
        ds = ds.assign(
            activations=blob["activations"],
            attention_mask=blob["attention_mask"],
            input_ids=blob["input_ids"],
        )
        acts = Activation.from_dataset(ds)
        for seed, probe in probes.items():
            with contextlib.redirect_stdout(io.StringIO()):
                s = probe._classifier.logits(acts)
            out[seed][split] = A.auroc_both(y, s.float().cpu().numpy())
            del s
        del ds, blob, acts
        gc.collect()

    for seed in out:
        for scale in ("pipeline", "rank"):
            out[seed].setdefault("mean", {})[scale] = float(
                np.mean([out[seed][sp][scale] for sp in A.EVAL_SPLITS])
            )
    return out


# --- progress sidecar -------------------------------------------------------------


def _append_progress(path: Path, row: dict) -> None:
    """Append one finished (vintage, seed) fit, fsync'd.

    A full sweep is hours, so the box dying must cost one fit rather than the run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _done_keys(path: Path) -> set[tuple[str, int, int]]:
    """``(arm, vintage, seed)`` triples already recorded in the progress sidecar."""
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a row truncated by a hard kill; the fit simply re-runs
            out.add((r["arm"], int(r["vintage"]), int(r["seed"])))
    return out


# --- main -------------------------------------------------------------------------


def run_arm(arm: str, iteration: int, seeds: list[int], eval_dir: Path, out_dir: Path,
            progress_path: Path, drop_mode: str, resume: bool = True,
            only_vintages: list[int] | None = None,
            vintage_order: list[int] | None = None) -> None:
    from agentic_redteam.retrain import _infer_probe_spec

    exclude, drop_report = dropped_rows(arm, iteration, drop_mode)
    keep, report = vintages(arm, iteration, exclude)
    report["overlong_drop"] = drop_report

    print(f"\n=== {arm} ===", flush=True)
    print(f"  over-long drop ({drop_mode}): {drop_report}", flush=True)
    for k in sorted(report["vintages"]):
        v = report["vintages"][k]
        print(
            f"  vintage {k}: {v['n_rows']:4d} rows of iter{iteration} "
            f"({v['n_pairs_kept']} of {v['n_pairs_in_dump']} iter{k} pairs survive; "
            f"{v['n_pairs_dropped']} dropped before iter{iteration})",
            flush=True,
        )
    print(f"  pairs in vintage 1 but not vintage 2: {report['n_not_nested_1_in_2']}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{arm}_vintage_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    done = _done_keys(progress_path) if resume else set()
    wanted = [k for k in (vintage_order or sorted(keep)) if k in keep]
    if only_vintages is not None:
        wanted = [k for k in wanted if k in only_vintages]

    probe = A.load_probe(A.ARMS[arm] / f"probe_iter{iteration}.pkl")
    probe_spec = _infer_probe_spec(probe)
    base = A.load_base_dataset(probe)
    base_train, base_val = A.split_sides(base)
    btr_blob, bval_blob = A.base_blob_paths()
    redteam = A.load_redteam_dataset(arm, iteration)
    rt_is_val = np.array([A.is_val(m) for m in redteam.inputs], dtype=bool)

    for k in wanted:
        todo = [s for s in seeds if (arm, k, s) not in done]
        if not todo:
            print(f"  vintage {k}: all {len(seeds)} seed(s) already recorded", flush=True)
            continue

        rows = keep[k]
        need_fit = [s for s in todo if not fit_path(out_dir, arm, k, s).exists()]

        # --- phase A: train, checkpointing each classifier as it lands ---------------
        if need_fit:
            tr_rows = [i for i in rows if not rt_is_val[i]]
            val_rows = [i for i in rows if rt_is_val[i]]

            t0 = time.time()
            train = build_side(base_train, btr_blob, redteam, tr_rows)
            val = build_side(base_val, bval_blob, redteam, val_rows)
            gb = (
                train.other_fields["activations"].nbytes
                + val.other_fields["activations"].nbytes
            ) / 1e9
            print(
                f"  vintage {k}: assembled train={len(train)} val={len(val)} "
                f"width={train.other_fields['activations'].shape[1]} "
                f"({gb:.1f} GB, {time.time() - t0:.0f}s); {len(need_fit)} fit(s) to run",
                flush=True,
            )

            for seed in need_fit:
                t1 = time.time()
                fitted = fit(train, val, probe, probe_spec, seed)
                fit_s = time.time() - t1
                _save_fitted(
                    fit_path(out_dir, arm, k, seed),
                    fitted,
                    {
                        "n_train": len(train),
                        "n_val": len(val),
                        "n_redteam_rows": len(rows),
                        "fit_seconds": fit_s,
                    },
                )
                print(
                    f"    seed {seed} vintage {k}: fit {fit_s:5.1f}s "
                    f"best_epoch={fitted._classifier.best_epoch}",
                    flush=True,
                )
                del fitted
                gc.collect()
                torch.cuda.empty_cache()

            # Freed BEFORE scoring so the 11.3 GB anthropic blob has page cache to live
            # in — that is what makes scoring ten seeds cost about as much as one.
            del train, val
            gc.collect()
            torch.cuda.empty_cache()

        # --- phase B: score every un-recorded seed in one pass over the eval blobs ---
        loaded = {}
        meta = {}
        for seed in todo:
            p, ckpt = _load_fitted(probe, fit_path(out_dir, arm, k, seed))
            loaded[seed] = p
            meta[seed] = ckpt
        t2 = time.time()
        scored = score_many(loaded, probe, eval_dir)
        print(
            f"  vintage {k}: scored {len(loaded)} seed(s) in {time.time() - t2:.0f}s",
            flush=True,
        )
        for seed in todo:
            res = scored[seed]
            print(
                f"    seed {seed} v{k}: "
                + "  ".join(f"{sp:9s}={res[sp]['pipeline']:.4f}" for sp in A.EVAL_SPLITS)
                + f"  MEAN={res['mean']['pipeline']:.4f}",
                flush=True,
            )
            _append_progress(
                progress_path,
                {
                    "arm": arm,
                    "vintage": k,
                    "seed": seed,
                    "drop_overlong": drop_mode,
                    "n_redteam_rows": len(rows),
                    "n_train": meta[seed]["n_train"],
                    "n_val": meta[seed]["n_val"],
                    # None when validation AUROC never improved on its initial value —
                    # which happens on the tiny base-only vintage, where a seed can
                    # leave the probe at chance for all 200 epochs. Not an error, and it
                    # must not abort a sweep whose other fits are fine.
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


def summarize(all_rows: list[dict], out_dir: Path) -> None:
    """Aggregate the sidecar into a mean +/- sd table and a per-fit CSV.

    Reported unpaired (mean and sd over seeds per vintage), because these are separate
    ``ProbeFactory`` fits with independent initialisations — there is no shared shuffle
    stream to difference against. That is exactly why the sd column matters more than
    the mean: it is the quantity that makes or breaks a single-seed reading.
    """
    from collections import defaultdict

    by = defaultdict(list)
    for r in all_rows:
        by[(r["arm"], r["vintage"], r["n_redteam_rows"])].append(r)

    print("\n\n=== eval AUROC by red-team vintage — mean +/- sd over seeds (pipeline) ===")
    header = (f"{'arm':14s} {'vint':>4s} {'rows':>5s} {'n':>3s} "
              + " ".join(f"{sp:>19s}" for sp in A.EVAL_SPLITS)
              + f"{'MEAN':>19s}")
    print(header)
    print("-" * len(header))
    for (arm, vint, rows_n), rs in sorted(by.items()):
        cells = []
        for sp in list(A.EVAL_SPLITS) + ["mean"]:
            v = np.array([r["auroc"][sp]["pipeline"] for r in rs])
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            cells.append(f"{v.mean():.4f}+/-{sd:.4f}")
        print(f"{arm:14s} {vint:>4d} {rows_n:>5d} {len(rs):>3d} "
              + " ".join(f"{c:>19s}" for c in cells))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "vintage_auroc.csv"
    with csv.open("w", encoding="utf-8") as fh:
        fh.write("arm,vintage,seed,n_redteam_rows,dataset,auroc_pipeline,auroc_rank\n")
        for r in sorted(all_rows, key=lambda x: (x["arm"], x["vintage"], x["seed"])):
            for sp in list(A.EVAL_SPLITS) + ["mean"]:
                fh.write(
                    f"{r['arm']},{r['vintage']},{r['seed']},{r['n_redteam_rows']},{sp},"
                    f"{r['auroc'][sp]['pipeline']},{r['auroc'][sp]['rank']}\n"
                )

    stats_path = out_dir / "vintage_summary.csv"
    with stats_path.open("w", encoding="utf-8") as fh:
        fh.write("arm,vintage,n_redteam_rows,n_seeds,dataset,mean,sd,min,max\n")
        for (arm, vint, rows_n), rs in sorted(by.items()):
            for sp in list(A.EVAL_SPLITS) + ["mean"]:
                v = np.array([r["auroc"][sp]["pipeline"] for r in rs])
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                fh.write(f"{arm},{vint},{rows_n},{len(rs)},{sp},"
                         f"{v.mean()},{sd},{v.min()},{v.max()}\n")
    print(f"\nwrote {csv}\nwrote {stats_path}")


def self_check(arm: str, iteration: int, eval_dir: Path, out_dir: Path) -> None:
    """Verify ``_load_fitted`` reproduces a live fitted probe's logits exactly.

    The sweep scores reloaded checkpoints rather than the objects it just trained, so
    this equivalence is load-bearing: if the reload lost anything, every number in the
    sweep would be silently wrong. Uses vintage 0 (base data only, 160 train rows) and
    the smallest eval split, so it costs seconds.
    """
    import contextlib
    import io

    from tuberlens.interfaces.activations import Activation
    from tuberlens.interfaces.dataset import LabelledDataset

    from agentic_redteam.retrain import _infer_probe_spec

    probe = A.load_probe(A.ARMS[arm] / f"probe_iter{iteration}.pkl")
    base = A.load_base_dataset(probe)
    base_train, base_val = A.split_sides(base)
    btr_blob, bval_blob = A.base_blob_paths()
    redteam = A.load_redteam_dataset(arm, iteration)

    train = build_side(base_train, btr_blob, redteam, [])
    val = build_side(base_val, bval_blob, redteam, [])
    live = fit(train, val, probe, _infer_probe_spec(probe), A.SEED)

    path = out_dir / "fits" / "_selfcheck.pt"
    _save_fitted(path, live, {"n_train": len(train), "n_val": len(val),
                              "n_redteam_rows": 0, "fit_seconds": 0.0})
    reloaded, _ = _load_fitted(probe, path)

    split = "mts"
    ds = LabelledDataset.load_from(
        A.EVAL_DATASET_DIR / f"{split}.jsonl",
        pos_class_label=probe.pos_class_label,
        neg_class_label=probe.neg_class_label,
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    blob = _blob(eval_dir / f"{split}-acts_full.pt")
    ds = ds.assign(activations=blob["activations"],
                   attention_mask=blob["attention_mask"],
                   input_ids=blob["input_ids"])
    acts = Activation.from_dataset(ds)
    with contextlib.redirect_stdout(io.StringIO()):
        a = live._classifier.logits(acts).float().cpu()
        b = reloaded._classifier.logits(acts).float().cpu()
    path.unlink(missing_ok=True)

    same = bool(torch.equal(a, b))
    print(f"self-check on {split}: max|delta| = {float((a - b).abs().max()):.3e}  "
          f"bitwise identical = {same}")
    if not same:
        raise SystemExit("_load_fitted does not reproduce the live probe — aborting")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seed", type=int, default=A.SEED, help="first seed")
    ap.add_argument("--seeds", type=int, default=10, help="how many consecutive seeds")
    ap.add_argument("--vintages", nargs="*", type=int, default=None,
                    help="which vintages to fit (default: all, including 0)")
    ap.add_argument(
        "--vintage-order", default="0,3,1,2",
        help="order vintages are fitted in. Seed-outer (the human-harm sweep's order) "
             "would rebuild the padded tensors for every fit, which this box cannot "
             "afford — so the endpoints of the curve are front-loaded instead, and a "
             "sweep cut short still has the floor (v0) and the full set (v3).",
    )
    ap.add_argument(
        "--drop-overlong", choices=("none", "row", "pair"), default="pair",
        help="exclude rows whose cached activations hit the 1024-token extraction "
             "window. 'pair' (default) removes the affected pair whole.",
    )
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="skip (vintage, seed) pairs already in the progress sidecar")
    ap.add_argument("--self-check", action="store_true",
                    help="verify _load_fitted reproduces a live fitted probe, then exit")
    ap.add_argument("--eval-dir", type=Path, default=A.EVAL_ACTIVATIONS_DIR)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=A.REPO / "results_hs_gemma27b_batch_ablation/vintage",
    )
    args = ap.parse_args()

    if args.self_check:
        self_check(args.arm[0], args.iteration, args.eval_dir, args.out_dir)
        return

    seeds = [args.seed + i for i in range(args.seeds)]
    order = [int(x) for x in args.vintage_order.split(",") if x.strip() != ""]
    progress = args.out_dir / "vintage_progress.jsonl"

    for arm in args.arm:
        run_arm(arm, args.iteration, seeds, args.eval_dir, args.out_dir, progress,
                drop_mode=args.drop_overlong, resume=args.resume,
                only_vintages=args.vintages, vintage_order=order)

    # Report from the SIDECAR, not from this process's return values: on a resumed run
    # the rows computed before the restart are on disk and nowhere else, and a summary
    # covering only this process's fits would silently understate the sweep.
    all_rows = [
        r for r in (json.loads(l) for l in progress.open(encoding="utf-8"))
        if r["arm"] in args.arm
        and (args.vintages is None or r["vintage"] in args.vintages)
    ]
    summarize(all_rows, args.out_dir)


if __name__ == "__main__":
    main()
