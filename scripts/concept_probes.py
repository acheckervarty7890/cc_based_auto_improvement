#!/usr/bin/env python
"""Train probes on a generator's synthetic cut of each concept, and score them.

Three attacker-model "generators" have each written ~50 balanced two-turn conversations
per concept, exhibiting that concept's positive and negative class:

    llama8b    data/<concept>_llama8b.jsonl        (meta-llama Llama-3.1-8B)
    llama70b   data/<concept>_llama70b_50.jsonl    (meta-llama Llama-3.3-70B)
    dsv4pro    data/<concept>_dsv4pro.jsonl        (deepseek v4 pro)

Every probe is trained on one such cut and scored on that concept's real eval splits at
full size, so this is a transfer question: how much of a concept does a 50-row synthetic
cut teach a probe, and does the generator matter?

Four arms per (generator, concept), the cross of

  config    single      one probe
            seq_ens10   10-member score-averaging ensemble, fit SEQUENTIALLY
                        (PROBE_FUSED_ENSEMBLE=0 — one ProbeFactory.build per seed)
  val_mode  dev         the concept's held-out dev set is the whole validation set;
                        all ~50 training rows train
            split       a 0.2 content-deterministic slice of the training data is held
                        out instead, leaving ~40 rows to train on

WHY THIS SCRIPT RATHER THAN THE CLI: ``train_initial_probe`` keys its base-activation
cache on the data file plus ``test_size``, so each (generator, concept, val_mode) would
extract its own copy of the same conversations — eighteen gemma-3-27b loads for 449
distinct rows. The `prepare` phase instead extracts every requested generator's rows in
ONE model load and writes each cache blob by slicing that master, addressing rows by
conversation content rather than by position. After `prepare`, the fits load no model.

The activation cache dir is shared across generators and with ``results_generalization``
deliberately: base blobs are keyed on the training file's own hash (so they cannot
collide) and the three dev blobs — 24 GB, already assembled — are keyed on the dev
files' bytes, so they are the same object for every run.

Phases (``--phase``): prepare | train | eval | all. Each is idempotent — an existing
blob, probe or eval row is left alone unless ``--force``.

    .venv_claude/bin/python scripts/concept_probes.py --generator llama70b
    .venv_claude/bin/python scripts/concept_probes.py --generator llama70b \
        --generator dsv4pro --phase prepare      # both, in one model load
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Sequential ensemble fits: one ProbeFactory.build per seed, the path this repo took
# before fusion existed. Set before tuberlens is imported so global_settings picks it
# up; also set on the settings object in main(), since that is what is actually read.
os.environ.setdefault("PROBE_FUSED_ENSEMBLE", "0")

# Pin accelerate's per-device budget for the one extraction this script triggers.
# Unpinned, accelerate derives the "cpu" budget from whatever RAM is free at load time
# and can fall back to DISK offload, which costs 48-264 s/sample against ~2.4 here.
# 24 GiB card, 108 GiB host: leave ~3 GiB of card for the activations being gathered.
os.environ.setdefault("AGENTIC_REDTEAM_MAX_MEMORY", "0=21GiB,cpu=60GiB")


def _load_hf_token() -> None:
    """tuberlens' hf_login() reads HF_TOKEN from the environment and raises without it."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"):
        return
    tf = REPO_ROOT / "hf_token.txt"
    if not tf.is_file():
        return
    for line in tf.read_text().splitlines():
        line = line.strip()
        if line.startswith("hf_"):
            os.environ["HF_TOKEN"] = line
        elif "=" in line:
            k, v = line.split("=", 1)
            if k.strip().upper() in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                os.environ["HF_TOKEN"] = v.strip().strip("'\"")


_load_hf_token()

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
# Both transforms are no-ops on this data (two-turn user/assistant, no tool turns), but
# they are folded into every activation cache key, so they must match the value the
# concept configs use — see configs/gemma27_config_hu_harm.md.
COMBINE = True
CONVERT = True
SEED = 42
TEST_SIZE = 0.2
DEFAULT_ARCH = "linear_then_softmax"

# Shared with the generalization experiment on purpose — see the module docstring.
CACHE_DIR = REPO_ROOT / "results_generalization" / "base_activation_cache"

# concept -> (file stem prefix, pos_class_label, neg_class_label)
CONCEPTS = {
    "hu_ha": ("hu_harm", "harmful_to_human", "not_harmful_to_human"),
    "highstakes": ("highstakes", "high-stakes", "low-stakes"),
    "instructions": (
        "instructions",
        "assistant_follows_the_instruction",
        "assistant_does_not_follow_the_instruction",
    ),
}

# generator -> training-file suffix
GENERATORS = {
    "llama8b": "llama8b",
    "llama70b": "llama70b_50",
    "dsv4pro": "dsv4pro",
}

CONFIGS = {"single": 1, "seq_ens10": 10}
VAL_MODES = ["dev", "split"]


def results_dir(generator: str) -> Path:
    return REPO_ROOT / f"results_{generator}"


def master_acts(generator: str) -> Path:
    return results_dir(generator) / f"{generator}_master_acts.pt"


def data_path(generator: str, concept: str) -> Path:
    prefix = CONCEPTS[concept][0]
    return REPO_ROOT / "data" / f"{prefix}_{GENERATORS[generator]}.jsonl"


def labels_for(concept: str) -> tuple[str, str]:
    return CONCEPTS[concept][1], CONCEPTS[concept][2]


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------------------
def _key(messages) -> tuple:
    """Content key for one conversation, used to address master-blob rows by identity."""
    return tuple((m.role, m.content) for m in messages)


def _load_dataset(generator: str, concept: str):
    from tuberlens.interfaces.dataset import LabelledDataset

    pos, neg = labels_for(concept)
    return LabelledDataset.load_from(
        data_path(generator, concept),
        pos_class_label=pos,
        neg_class_label=neg,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )


def _generator_inputs(generator: str) -> list:
    """One generator's conversations across all concepts, in CONCEPTS order."""
    inputs = []
    for concept in CONCEPTS:
        inputs.extend(_load_dataset(generator, concept).inputs)
    return inputs


def extract_masters(generators: list[str], force: bool = False) -> None:
    """Extract every requested generator's conversations in ONE model load.

    A generator whose master blob already exists is skipped; if that leaves nothing to
    do, no model is loaded at all. The rows of all pending generators go through a
    single ``get_activations`` call and are then sliced back apart, so N generators cost
    one load rather than N.
    """
    import torch

    pending = [g for g in generators if force or not master_acts(g).exists()]
    for g in generators:
        if g not in pending:
            print(f"master activations already at {master_acts(g)}")
    if not pending:
        return

    from agentic_redteam.model_loading import load_extraction_model, unhook_model

    counts, inputs = [], []
    for g in pending:
        rows = _generator_inputs(g)
        counts.append((g, len(rows)))
        inputs.extend(rows)
    print(f"extracting activations for {len(inputs)} conversations across "
          f"{', '.join(g for g, _n in counts)} ({MODEL_NAME} L{LAYER}) ...")

    model = load_extraction_model(MODEL_NAME, LAYER, verbose=True)
    try:
        t0 = time.time()
        acts = model.get_activations(inputs, layer=LAYER, show_progress=True)
        print(f"  {tuple(acts.activations.shape)} in {time.time()-t0:.0f}s")
    finally:
        unhook_model(model)
        del model
        _free_gpu()

    at = 0
    for g, n in counts:
        out = master_acts(g)
        out.parent.mkdir(parents=True, exist_ok=True)
        sl = slice(at, at + n)
        torch.save(
            {
                "activations": acts.activations[sl].clone(),
                "attention_mask": acts.attention_mask[sl].clone(),
                "input_ids": acts.input_ids[sl].clone(),
                "layer": LAYER,
                "model_name": MODEL_NAME,
            },
            out,
        )
        print(f"  {g}: {n} rows -> {out} ({out.stat().st_size / 1e9:.2f} GB)")
        at += n


def _master_index(generator: str) -> dict:
    """{content key: row index} over the generator's master blob, asserted collision-free."""
    index = {}
    for i, messages in enumerate(_generator_inputs(generator)):
        k = _key(messages)
        if k in index:
            raise RuntimeError(
                f"two {generator} conversations are byte-identical after transforms, so "
                "a content-addressed slice would be ambiguous; fall back to positional "
                "mapping if this ever fires"
            )
        index[k] = i
    return index


def _save_slice(master: dict, rows: list[int], out: Path) -> None:
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    idx = torch.tensor(rows, dtype=torch.long)
    torch.save(
        {
            "activations": master["activations"][idx].clone(),
            "attention_mask": master["attention_mask"][idx].clone(),
            "input_ids": master["input_ids"][idx].clone(),
            "layer": LAYER,
            "model_name": MODEL_NAME,
        },
        out,
    )


def prefill_base_caches(generator: str, concept: str, force: bool = False) -> None:
    """Write the train/val activation blob each validation mode will look for.

    Reproduces exactly what ``train_initial_probe`` does to derive them — load, subsample
    at fraction 1.0, ``stable_train_test_split`` — so the blobs land at the paths its
    cache-key function computes and its ``_activate`` loads them instead of the model.
    """
    import torch

    from agentic_redteam.retrain import (
        _base_activation_cache_paths,
        stable_fraction_subsample,
        stable_train_test_split,
    )

    path = data_path(generator, concept)
    index = _master_index(generator)
    master = None

    for val_mode in VAL_MODES:
        test_size = 0.0 if val_mode == "dev" else TEST_SIZE
        train_cache, val_cache = _base_activation_cache_paths(
            CACHE_DIR, path, MODEL_NAME, LAYER, SEED, test_size, None,
            COMBINE, CONVERT, 1.0,
        )
        ds = stable_fraction_subsample(_load_dataset(generator, concept), 1.0, SEED)
        train, val = stable_train_test_split(ds, test_size=test_size,
                                             split_field=None, seed=SEED)
        for side, part, p in (("train", train, train_cache), ("val", val, val_cache)):
            if len(part) == 0:
                continue
            if p.exists() and not force:
                print(f"  {val_mode:<5} {side:<5} cached  -> {p.name}")
                continue
            if master is None:
                master = torch.load(master_acts(generator), map_location="cpu",
                                    weights_only=False)
            rows = [index[_key(m)] for m in part.inputs]
            _save_slice(master, rows, p)
            print(f"  {val_mode:<5} {side:<5} {len(rows):>3} rows -> {p.name}")


def prefill_dev_cache(concept: str, force: bool = False) -> None:
    """Assemble the concept's dev blob from the per-split Kaggle downloads.

    ``_load_dev_dataset`` concatenates ``sorted(glob("*.jsonl"))`` in that order, so the
    blob is the splits' rows in the same order, zero-padded to the longest split's
    sequence length. Normally a no-op — the first run wrote these into the shared cache
    dir, and the key (dev files' bytes + model|layer|transforms) is generator-independent.
    """
    import torch

    from agentic_redteam.retrain import _dev_activation_cache_path

    dev_dir = REPO_ROOT / "dev_samples" / concept
    dev_files = sorted(dev_dir.glob("*.jsonl"))
    target = _dev_activation_cache_path(CACHE_DIR, dev_files, MODEL_NAME, LAYER,
                                        COMBINE, CONVERT)
    if target.exists() and not force:
        print(f"  dev blob already at {target.name} "
              f"({target.stat().st_size / 1e9:.2f} GB)")
        return

    blob_dir = REPO_ROOT / "activations" / "dev" / concept
    parts = []
    for f in dev_files:
        blob = blob_dir / f"{f.stem}-acts_full.pt"
        if not blob.is_file():
            raise SystemExit(f"missing downloaded dev blob {blob}")
        n = sum(1 for l in f.open() if l.strip())
        parts.append((f.stem, torch.load(blob, map_location="cpu", mmap=True,
                                         weights_only=False), n))

    total = sum(n for _s, _h, n in parts)
    seq = max(h["activations"].shape[1] for _s, h, _n in parts)
    hid = parts[0][1]["activations"].shape[2]
    out = {
        "activations": torch.zeros((total, seq, hid), dtype=parts[0][1]["activations"].dtype),
        "attention_mask": torch.zeros((total, seq), dtype=parts[0][1]["attention_mask"].dtype),
        "input_ids": torch.zeros((total, seq), dtype=parts[0][1]["input_ids"].dtype),
        "layer": LAYER,
        "model_name": MODEL_NAME,
    }
    at = 0
    for split, head, n in parts:
        if head["activations"].shape[0] != n:
            raise SystemExit(f"{split}: blob has {head['activations'].shape[0]} rows, "
                             f"split has {n}")
        s = head["activations"].shape[1]
        for k in ("activations", "attention_mask", "input_ids"):
            out[k][at:at + n, :s] = head[k]
        at += n
    tmp = target.with_name(f".{target.name}.partial")
    torch.save(out, tmp)
    tmp.replace(target)
    print(f"  dev blob {total} rows x {seq} -> {target.name} "
          f"({target.stat().st_size / 1e9:.2f} GB)")


# --------------------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------------------
def probe_path(generator: str, concept: str, config: str, val_mode: str) -> Path:
    return results_dir(generator) / concept / "probes" / f"{config}__{val_mode}.pkl"


def _n_train(generator: str, concept: str, val_mode: str) -> int:
    """Training rows the fit will actually see, without touching activations."""
    from agentic_redteam.retrain import stable_fraction_subsample, stable_train_test_split

    ds = stable_fraction_subsample(_load_dataset(generator, concept), 1.0, SEED)
    train, _val = stable_train_test_split(
        ds, test_size=(0.0 if val_mode == "dev" else TEST_SIZE),
        split_field=None, seed=SEED,
    )
    return len(train)


def capped_spec(n_train: int):
    """``linear_then_softmax`` defaults, with gradient accumulation capped at batches/epoch.

    The default spec is ``batch_size: 16, gradient_accumulation_steps: 4``, and the
    trainer steps only on ``(batch_idx + 1) % accumulation == 0`` with **no end-of-epoch
    flush** (``pytorch_classifiers.py:299-327``). So a training set yielding fewer than
    4 batches never calls ``optimizer.step()`` at all: the fit runs its full epoch
    budget, loss sits at ln 2, validation AUROC is exactly 0.5, and the probe comes back
    at its initialization. That is not hypothetical here — every ``split`` arm is ~40
    rows = 3 batches and would hit it exactly. ``experiment16_cloud`` ran that shape
    unguarded on ``hu_harm_llama70b_50`` and scored 0.336 at iteration 0, below chance
    on three of four splits, against 0.846 for the otherwise-identical dev-validated
    ``experiment17_cloud``.

    Capping the accumulation at the number of batches is a no-op for any arm that
    already has 4 or more, so those probes are bit-identical to an uncapped fit.
    """
    import math

    from tuberlens.interfaces.probes import ProbeSpec

    from agentic_redteam.retrain import _coerce_probe_spec

    base = _coerce_probe_spec(DEFAULT_ARCH)
    hp = dict(base.hyperparams)
    n_batches = max(1, math.ceil(n_train / int(hp.get("batch_size", 16))))
    hp["gradient_accumulation_steps"] = min(
        int(hp.get("gradient_accumulation_steps", 1)), n_batches
    )
    return ProbeSpec(name=base.name, hyperparams=hp)


def train_concept(generator: str, concept: str, force: bool = False) -> None:
    from agentic_redteam.retrain import train_initial_probe

    pos, neg = labels_for(concept)
    path = data_path(generator, concept)
    dev_dir = REPO_ROOT / "dev_samples" / concept
    for config, size in CONFIGS.items():
        for val_mode in VAL_MODES:
            out = probe_path(generator, concept, config, val_mode)
            if out.exists() and not force:
                print(f"[skip] {out.name}")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            n_train = _n_train(generator, concept, val_mode)
            spec = capped_spec(n_train)
            print(f"\n=== {generator} | {concept} | {config} | val={val_mode} ==="
                  f"  ({n_train} train rows, "
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
                    f"{concept} probe trained on the {generator} synthetic cut "
                    f"({path.name}), {config}, validation={val_mode}."
                ),
                probe_spec=spec,
                test_size=TEST_SIZE,
                dev_data_path=dev_dir if val_mode == "dev" else None,
                seed=SEED,
                ensemble_size=size,
                base_activation_cache_dir=CACHE_DIR,
                combine_consecutive_messages=COMBINE,
                convert_tool_to_assistant=CONVERT,
                verbose=True,
            )
            print(f"    fit in {time.time()-t0:.0f}s -> {out}", flush=True)
            _free_gpu()


def eval_concept(generator: str, concept: str, force: bool = False) -> None:
    import pandas as pd

    from agentic_redteam.evaluation import evaluate_probe

    out_csv = results_dir(generator) / concept / "eval_results.csv"
    done = set()
    frames = []
    if out_csv.exists() and not force:
        prev = pd.read_csv(out_csv)
        frames.append(prev)
        done = set(zip(prev["config"], prev["val_mode"]))

    for config in CONFIGS:
        for val_mode in VAL_MODES:
            if (config, val_mode) in done:
                continue
            p = probe_path(generator, concept, config, val_mode)
            if not p.exists():
                print(f"[missing probe] {p.name}")
                continue
            print(f"--- eval {generator} | {concept} | {config} | val={val_mode}", flush=True)
            df = evaluate_probe(
                p,
                REPO_ROOT / "eval_sets" / concept,
                REPO_ROOT / "activations" / "eval" / concept,
                max_samples=None,
                seed=SEED,
                combine_consecutive_messages=COMBINE,
                convert_tool_to_assistant=CONVERT,
            )
            df = df.copy()
            df.insert(0, "val_mode", val_mode)
            df.insert(0, "config", config)
            df.insert(0, "concept", concept)
            df.insert(0, "generator", generator)
            frames.append(df)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
            _free_gpu()

    if frames:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", action="append", choices=sorted(GENERATORS),
                    help="Generator to run (repeatable; default: llama8b)")
    ap.add_argument("--concept", action="append", choices=sorted(CONCEPTS),
                    help="Concept to run (repeatable; default: all three)")
    ap.add_argument("--phase", default="all", choices=["prepare", "train", "eval", "all"])
    ap.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    args = ap.parse_args()
    generators = args.generator or ["llama8b"]
    concepts = args.concept or list(CONCEPTS)

    from tuberlens.config import global_settings

    # The sequential-ensemble arm. Read on every call by ensemble.fusion_enabled() and
    # internally by ProbeFactory.build_ensemble, so setting it here governs both.
    global_settings.PROBE_FUSED_ENSEMBLE = False
    print(f"PROBE_FUSED_ENSEMBLE = {global_settings.PROBE_FUSED_ENSEMBLE} (sequential fits)")

    # One model load for every generator that still needs one.
    if args.phase in ("prepare", "all"):
        extract_masters(generators, force=args.force)

    for generator in generators:
        for concept in concepts:
            print(f"\n######## {generator} / {concept} ########")
            if args.phase in ("prepare", "all"):
                prefill_base_caches(generator, concept, force=args.force)
                prefill_dev_cache(concept, force=args.force)
            if args.phase in ("train", "all"):
                train_concept(generator, concept, force=args.force)
            if args.phase in ("eval", "all"):
                eval_concept(generator, concept, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
