#!/usr/bin/env python
"""Train one probe per SUBSET of the four generators, and score it on the real eval splits.

The four attacker models have each written ~50 balanced two-turn conversations per
concept (see ``scripts/concept_probes.py``). That script asked what each generator is
worth on its own. This one asks what they are worth *pooled*: every non-empty subset of

    llama8b  ·  llama70b  ·  dsv4pro  ·  nemotron550b

is concatenated into one training set, so the matrix runs from 4 single-generator cuts
(~50 rows) through 6 pairs (~100), 4 triples (~150) to the one all-four pool (~200).

Only **single** probes are fitted — no ensembles — crossed with the two validation
sources:

  val_mode  dev     the concept's held-out ``dev_samples/`` set is the whole validation
                    set, and every training row trains
            split   a 0.2 content-deterministic slice of the training data is held out
                    instead, so ~80% of it trains

15 combos x 2 validation modes x 3 concepts = 90 probes.

NO MODEL IS EVER LOADED. ``scripts/concept_probes.py`` already extracted each
generator's conversations into a master activation blob, and the split is
content-deterministic — a conversation lands on the same side of the train/val line in
every combo it appears in — so each combo's train/val cache blob is assembled by
addressing rows in those masters *by conversation content* and padding them to a common
sequence length. ``train_initial_probe`` then finds its cache populated and loads no LLM.

The eval is likewise batched rather than per-probe: ``evaluate_probe`` reloads a split's
activations from disk for every probe it scores, which for high-stakes is 46 GB read 30
times. ``eval_concept`` instead reads each split once and scores all of that concept's
probes against it, computing the metrics with tuberlens' own ``calculate_metrics`` so
the numbers are identical to ``get_performances``.

Phases (``--phase``): prepare | train | eval | all. Each is idempotent — an existing
blob, probe or eval row is left alone unless ``--force``.

    .venv_claude/bin/python scripts/combo_probes.py
    .venv_claude/bin/python scripts/combo_probes.py --concept hu_ha --phase train
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported for its constants and its two load-bearing helpers: the module-level env
# pinning (HF token, accelerate memory budget) and ``capped_spec``, whose gradient
# accumulation cap is what keeps a ~40-row fit from returning an untrained probe.
from concept_probes import (  # noqa: E402
    CACHE_DIR,
    COMBINE,
    CONCEPTS,
    CONVERT,
    LAYER,
    MODEL_NAME,
    SEED,
    TEST_SIZE,
    VAL_MODES,
    _free_gpu,
    _key,
    _load_dataset,
    capped_spec,
    data_path,
    labels_for,
    master_acts,
)

RESULTS = REPO_ROOT / "results_combos"

# Canonical generator order. Every combo is written, named and concatenated in this
# order, so a combo's training file is a pure function of its member set.
ORDER = ["llama8b", "llama70b", "dsv4pro", "nemotron550b"]
SHORT = {"llama8b": "l8b", "llama70b": "l70b", "dsv4pro": "dsv", "nemotron550b": "nem"}


def all_combos() -> list[tuple[str, ...]]:
    """The 15 non-empty subsets, smallest first, each in ORDER order."""
    out: list[tuple[str, ...]] = []
    for k in range(1, len(ORDER) + 1):
        out.extend(itertools.combinations(ORDER, k))
    return out


def combo_name(combo: tuple[str, ...]) -> str:
    return "+".join(SHORT[g] for g in combo)


def combo_data_path(concept: str, combo: tuple[str, ...]) -> Path:
    return RESULTS / "data" / f"{concept}__{combo_name(combo)}.jsonl"


def probe_path(concept: str, combo: tuple[str, ...], val_mode: str) -> Path:
    return RESULTS / concept / "probes" / f"{combo_name(combo)}__{val_mode}.pkl"


# --------------------------------------------------------------------------------------
# training files
# --------------------------------------------------------------------------------------
def build_data_files(concepts: list[str], force: bool = False) -> None:
    """Concatenate each subset's per-generator JSONLs into one training file.

    Byte-deterministic (fixed generator order, blank lines dropped), because the
    activation cache is keyed on a hash of this file's bytes — a reordering would
    silently orphan every blob prepared for it.
    """
    (RESULTS / "data").mkdir(parents=True, exist_ok=True)
    for concept in concepts:
        for combo in all_combos():
            out = combo_data_path(concept, combo)
            if out.exists() and not force:
                continue
            rows: list[str] = []
            for g in combo:
                rows += [l for l in data_path(g, concept).read_text().splitlines() if l.strip()]
            out.write_text("\n".join(rows) + "\n")
        n = len(list((RESULTS / "data").glob(f"{concept}__*.jsonl")))
        print(f"  {concept}: {n} combo training files")


# --------------------------------------------------------------------------------------
# activation cache
# --------------------------------------------------------------------------------------
def _load_masters(generators: list[str]) -> dict:
    """{generator: master blob}. ~215 MB each, so all four fit comfortably in RAM."""
    import torch

    out = {}
    for g in generators:
        p = master_acts(g)
        if not p.is_file():
            raise SystemExit(
                f"missing master activations {p} — run scripts/concept_probes.py "
                f"--generator {g} --phase prepare first"
            )
        out[g] = torch.load(p, map_location="cpu", weights_only=False)
    return out


def _content_index(concept: str) -> dict:
    """{conversation content key: (generator, row in that generator's master)}.

    The masters hold each generator's conversations in ``CONCEPTS`` order across all
    three concepts, so the row for a given conversation is found by walking the same
    order the extractor used. Asserted collision-free: two byte-identical conversations
    would make the address ambiguous (they are in fact all distinct, checked across
    generators, but a future data cut could break that).
    """
    index: dict = {}
    for g in ORDER:
        at = 0
        for c in CONCEPTS:
            rows = _load_dataset(g, c).inputs
            if c == concept:
                for i, messages in enumerate(rows):
                    k = _key(messages)
                    if k in index:
                        raise RuntimeError(
                            f"conversation appears twice ({index[k]} and ({g}, {at+i})); "
                            "content addressing is ambiguous"
                        )
                    index[k] = (g, at + i)
            at += len(rows)
    return index


def _save_slice(masters: dict, rows: list[tuple[str, int]], out: Path) -> None:
    """Write one cache blob gathering rows from several generators' masters.

    Each master is padded to its own generator's longest conversation, so a mixed slice
    is zero-padded up to the widest of them. Padding is invisible downstream —
    ``Activation.__post_init__`` masks the activations and ``_concatenate_consuming``
    re-pads at merge — so a blob's width is a storage detail, not part of its identity.
    """
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    seq = max(masters[g]["activations"].shape[1] for g, _i in rows)
    hid = masters[rows[0][0]]["activations"].shape[2]
    blob = {
        "activations": torch.zeros(
            (len(rows), seq, hid), dtype=masters[rows[0][0]]["activations"].dtype
        ),
        "attention_mask": torch.zeros(
            (len(rows), seq), dtype=masters[rows[0][0]]["attention_mask"].dtype
        ),
        "input_ids": torch.zeros(
            (len(rows), seq), dtype=masters[rows[0][0]]["input_ids"].dtype
        ),
        "layer": LAYER,
        "model_name": MODEL_NAME,
    }
    for j, (g, i) in enumerate(rows):
        s = masters[g]["activations"].shape[1]
        for k in ("activations", "attention_mask", "input_ids"):
            blob[k][j, :s] = masters[g][k][i]
    tmp = out.with_name(f".{out.name}.partial")
    torch.save(blob, tmp)
    tmp.replace(out)


def prefill_caches(concepts: list[str], force: bool = False) -> None:
    from agentic_redteam.retrain import (
        _base_activation_cache_paths,
        stable_fraction_subsample,
        stable_train_test_split,
    )

    masters = None
    for concept in concepts:
        index = _content_index(concept)
        for combo in all_combos():
            path = combo_data_path(concept, combo)
            ds = stable_fraction_subsample(
                _load_combo_dataset(concept, combo), 1.0, SEED
            )
            for val_mode in VAL_MODES:
                test_size = 0.0 if val_mode == "dev" else TEST_SIZE
                train_cache, val_cache = _base_activation_cache_paths(
                    CACHE_DIR, path, MODEL_NAME, LAYER, SEED, test_size, None,
                    COMBINE, CONVERT, 1.0,
                )
                train, val = stable_train_test_split(
                    ds, test_size=test_size, split_field=None, seed=SEED
                )
                for side, part, p in (("train", train, train_cache), ("val", val, val_cache)):
                    if len(part) == 0:
                        continue
                    if p.exists() and not force:
                        continue
                    if masters is None:
                        masters = _load_masters(ORDER)
                    rows = [index[_key(m)] for m in part.inputs]
                    _save_slice(masters, rows, p)
                    print(f"  {concept:<12} {combo_name(combo):<20} {val_mode:<5} "
                          f"{side:<5} {len(rows):>3} rows -> {p.name}", flush=True)
        print(f"  {concept}: caches ready")


def _load_combo_dataset(concept: str, combo: tuple[str, ...]):
    from tuberlens.interfaces.dataset import LabelledDataset

    pos, neg = labels_for(concept)
    return LabelledDataset.load_from(
        combo_data_path(concept, combo),
        pos_class_label=pos,
        neg_class_label=neg,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )


# --------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------
def train_concept(concept: str, force: bool = False) -> None:
    from agentic_redteam.retrain import stable_fraction_subsample, stable_train_test_split, train_initial_probe

    pos, neg = labels_for(concept)
    dev_dir = REPO_ROOT / "dev_samples" / concept
    for combo in all_combos():
        path = combo_data_path(concept, combo)
        ds = stable_fraction_subsample(_load_combo_dataset(concept, combo), 1.0, SEED)
        for val_mode in VAL_MODES:
            out = probe_path(concept, combo, val_mode)
            if out.exists() and not force:
                print(f"[skip] {concept}/{out.name}")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            train, _val = stable_train_test_split(
                ds, test_size=(0.0 if val_mode == "dev" else TEST_SIZE),
                split_field=None, seed=SEED,
            )
            spec = capped_spec(len(train))
            print(f"\n=== {concept} | {combo_name(combo)} | val={val_mode} ===  "
                  f"({len(train)} train rows, "
                  f"accum={spec.hyperparams['gradient_accumulation_steps']})", flush=True)
            t0 = time.time()
            train_initial_probe(
                base_training_data_path=path,
                model_name=MODEL_NAME,
                layer=LAYER,
                new_probe_path=out,
                pos_class_label=pos,
                neg_class_label=neg,
                probe_description=(
                    f"{concept} probe trained on the pooled {', '.join(combo)} synthetic "
                    f"cuts ({len(train)} rows), single probe, validation={val_mode}."
                ),
                probe_spec=spec,
                test_size=TEST_SIZE,
                dev_data_path=dev_dir if val_mode == "dev" else None,
                seed=SEED,
                ensemble_size=1,
                base_activation_cache_dir=CACHE_DIR,
                combine_consecutive_messages=COMBINE,
                convert_tool_to_assistant=CONVERT,
                verbose=True,
            )
            print(f"    fit in {time.time()-t0:.0f}s -> {out}", flush=True)
            _free_gpu()


# --------------------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------------------
def eval_concept(concept: str, force: bool = False) -> None:
    """Score every fitted probe for this concept, reading each eval split ONCE.

    ``evaluate_probe`` is per-probe and reloads the split blobs each time; at 30 probes
    and 46 GB of high-stakes activations that is 1.4 TB of reads for 1.4 TB of nothing.
    The metrics come from tuberlens' own ``calculate_metrics``, on full splits with no
    subsampling, so a row here is what ``get_performances`` would have produced.
    """
    import pickle

    import numpy as np
    import pandas as pd
    from tuberlens.evaluation import calculate_metrics
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.model import LLMModel

    from agentic_redteam.evaluation import seed_everything

    out_csv = RESULTS / concept / "eval_results.csv"
    done: set = set()
    prev = None
    if out_csv.exists() and not force:
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["combo"], prev["val_mode"]))

    wanted = []
    for combo in all_combos():
        for val_mode in VAL_MODES:
            if (combo_name(combo), val_mode) in done:
                continue
            p = probe_path(concept, combo, val_mode)
            if not p.exists():
                print(f"[missing probe] {p.name}")
                continue
            wanted.append((combo, val_mode, p))
    if not wanted:
        print(f"  {concept}: nothing to evaluate")
        return

    probes = []
    for combo, val_mode, p in wanted:
        with p.open("rb") as f:
            probes.append((combo, val_mode, pickle.load(f)))
    print(f"  {concept}: scoring {len(probes)} probes", flush=True)

    pos, neg = labels_for(concept)
    eval_dir = REPO_ROOT / "eval_sets" / concept
    acts_dir = REPO_ROOT / "activations" / "eval" / concept
    seed_everything(SEED)

    rows = []
    for split_file in sorted(eval_dir.glob("*.jsonl")):
        name = split_file.stem
        dataset = LabelledDataset.load_from(
            split_file,
            pos_class_label=pos,
            neg_class_label=neg,
            combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT,
        )
        blob = acts_dir / f"{name}-acts_full.pt"
        if not blob.is_file():
            raise SystemExit(f"missing eval activations {blob}")
        acts = LLMModel.load_activations(blob)
        if len(acts.activations) != len(dataset):
            raise SystemExit(
                f"{name}: blob has {len(acts.activations)} rows, split has {len(dataset)}"
            )
        dataset = dataset.assign(
            activations=acts.activations,
            attention_mask=acts.attention_mask,
            input_ids=acts.input_ids,
        )
        y = np.array([label.to_int() for label in dataset.labels])
        t0 = time.time()
        for combo, val_mode, probe in probes:
            preds = probe.predict_proba(dataset)
            rows.append(
                calculate_metrics(y, preds, fpr=0.01)
                | {
                    "concept": concept,
                    "combo": combo_name(combo),
                    "n_generators": len(combo),
                    "generators": "+".join(combo),
                    "val_mode": val_mode,
                    "dataset": name,
                }
            )
        print(f"    {name:<28} {len(dataset):>5} rows x {len(probes)} probes "
              f"in {time.time()-t0:.0f}s", flush=True)
        del dataset, acts
        _free_gpu()

    df = pd.DataFrame(rows)
    # Mirror get_performances' trailing per-probe mean row.
    means = (
        df.groupby(["concept", "combo", "n_generators", "generators", "val_mode"],
                   as_index=False)[["auroc", "accuracy", "tpr_at_fpr", "fpr"]].mean()
    )
    means["dataset"] = "mean"
    df = pd.concat([df, means], ignore_index=True)
    cols = ["concept", "combo", "n_generators", "generators", "val_mode", "dataset",
            "auroc", "accuracy", "tpr_at_fpr", "fpr"]
    df = df[cols]
    if prev is not None:
        df = pd.concat([prev, df], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", action="append", choices=sorted(CONCEPTS),
                    help="Concept to run (repeatable; default: all three)")
    ap.add_argument("--phase", default="all", choices=["prepare", "train", "eval", "all"])
    ap.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    args = ap.parse_args()
    concepts = args.concept or list(CONCEPTS)

    if args.phase in ("prepare", "all"):
        print("######## prepare ########")
        build_data_files(concepts, force=args.force)
        prefill_caches(concepts, force=args.force)
    for concept in concepts:
        if args.phase in ("train", "all"):
            print(f"\n######## train / {concept} ########")
            train_concept(concept, force=args.force)
        if args.phase in ("eval", "all"):
            print(f"\n######## eval / {concept} ########")
            eval_concept(concept, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
