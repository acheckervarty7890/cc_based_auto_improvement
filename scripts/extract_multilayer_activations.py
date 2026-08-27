#!/usr/bin/env python
"""Extract activations at SEVERAL layers in ONE forward pass, for a layer sweep.

``scripts/extract_eval_activations.py`` computes one split at one layer, through
tuberlens' ``get_activations``, which wraps ``HookedModel(model, [layer])``. Running it
once per layer would re-run the whole forward stack once per layer — and on
``google/gemma-3-27b-it`` the forward is the entire cost.

``HookedModel`` already accepts a LIST of layers and registers one hook per layer, so a
single pass truncated to ``max(layers)`` yields every layer at once. Layers 16/24/32/40/48/56
therefore cost exactly what layer 56 alone costs, not six times it. That is the whole
point of this script.

Two properties are preserved so the blobs are drop-in for everything downstream:

* **The output format is byte-compatible with ``get_activations``' own** —
  ``{activations, attention_mask, input_ids, layer, model_name}``, fp16 activations,
  padded/truncated to ``min(1024, split max length)`` by the same rule
  (``tuberlens/model.py:466``). So ``LLMModel.load_activations`` reads them, and a blob
  written here is interchangeable with one written by ``get_activations``.
* **The model is loaded through ``model_loading.load_extraction_model``**, so only layers
  ``0..max(layers)`` are placed. At layer 56 that is 57 of 62 layers — still a big load,
  but 5 layers (4.2 GB of bf16) lighter than the full model, and the truncation is exact
  (causal stack).

THE TRANSFORMS ARE THE FOOTGUN, exactly as in ``extract_eval_activations.py``:
``combine_consecutive_messages`` / ``convert_tool_to_assistant`` change how a conversation
is tokenized but not how many rows a split has, so blobs computed under the wrong setting
pass every row-count check while encoding different token sequences. They are required
arguments here rather than defaulted.

Batch size defaults to 1, which is what every blob in this repo was computed at and the
only setting that involves no padding inside a batch. ``--batch-size N > 1`` forces
``padding_side="right"`` first: with left padding the real tokens sit at the right of each
row, and the final pad-to-split-max would then leave pads on BOTH sides — a layout nothing
downstream expects. Right padding is safe (the stack is causal and the pads are masked),
but verify it before trusting it: ``--verify-batching`` computes the first few rows both
ways and reports the max absolute difference.

Resumable: a (split, layer) whose blob already exists and validates is skipped, so an
interrupted run restarts at the split it died on. Layers still missing for a split force
that split's forward pass to be re-run (the pass is per split, not per layer).

Typical use::

    HF_HOME=$PWD/hf_cache AGENTIC_REDTEAM_MAX_MEMORY="0=22GiB,cpu=45GiB" \\
    .venv_claude/bin/python scripts/extract_multilayer_activations.py \\
        --layers 16 24 32 40 48 56 \\
        --data-dir eval_sets/instructions --data-dir dev_samples/instructions \\
        --out-root results_instructions_gemma27b_layersweep/activations \\
        --probe probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl \\
        --combine-consecutive-messages --convert-tool-to-assistant
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MAX_ACTIVATION_TOKENS = 1024
CACHE_STEM = "acts_full.pt"


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract activations at several layers in one forward pass.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--data-dir", type=Path, action="append", default=[],
                    help="Dir of split JSONLs. Repeatable (e.g. eval dir + dev dir); each "
                         "dir's PARENT name becomes a subdir under --out-root/L<layer>/ "
                         "(eval_sets/instructions -> eval_sets). Optional: a run may "
                         "supply only --extra-jsonl.")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True,
                    help="Probe pickle giving model_name (and the class labels used to load "
                         "the splits). Its own layer is irrelevant here.")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--extra-jsonl", action="append", default=[], metavar="NAME=PATH",
                    help="Repeatable. A single JSONL that is not a split dir — the base "
                         "training data, or a dumped red-team set. Written to "
                         "<out-root>/L<layer>/extra/<NAME>-acts_full.pt. Two schemas are "
                         "accepted, detected per file: the standard LabelledDataset form "
                         "({inputs: json-string, labels: class-label string}) and "
                         "retrain._dump_labelled_dataset's form ({id, inputs: list of "
                         "{role,content}, label: 'positive'/'negative'}). Row order is the "
                         "file's, which is what the consumers rebuild from.")
    ap.add_argument("--combine-consecutive-messages", action=argparse.BooleanOptionalAction,
                    required=True)
    ap.add_argument("--convert-tool-to-assistant", action=argparse.BooleanOptionalAction,
                    required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: only the first N rows of each split (writes to a *_limitN "
                         "out-root so it can never be mistaken for a real blob).")
    ap.add_argument("--verify-batching", type=int, default=0, metavar="N",
                    help="Debug: compute the first N rows of the first split at batch 1 and "
                         "at --batch-size, report the max abs difference, and exit.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.data_dir and not args.extra_jsonl:
        ap.error("nothing to extract: pass --data-dir and/or --extra-jsonl")
    return args


def _blob_path(out_root: Path, layer: int, group: str, split: str) -> Path:
    return out_root / f"L{layer}" / group / f"{split}-{CACHE_STEM}"


def _blob_ok(path: Path, *, model_name: str, layer: int, n_rows: int) -> bool:
    """Cheap header check: right model, right layer, right row count. mmap => instant."""
    if not path.exists():
        return False
    import torch
    try:
        blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        return False
    return (
        blob.get("model_name") == model_name
        and int(blob.get("layer", -1)) == layer
        and blob["activations"].shape[0] == n_rows
    )


def _load_extra_jsonl(path: Path, pos: str, neg: str, combine: bool, convert: bool):
    """``inputs`` of one loose JSONL, in file order, under the run's message transforms.

    Two schemas turn up in this repo and they are not interchangeable:

    * the standard ``LabelledDataset`` form — ``inputs`` a JSON-encoded string, ``labels``
      one of the probe's class-label strings (``data/instructions_llama70b_50.jsonl``);
    * ``retrain._dump_labelled_dataset``'s form — ``inputs`` already a list of
      ``{role, content}``, ``label`` (singular) the canonical ``"positive"`` /
      ``"negative"`` (``redteam_postprocessed_iter*.jsonl``).

    Only the ``inputs`` are needed here — activations do not depend on the label — but the
    two are detected apart anyway so a file in the second form is not silently handed to
    ``load_from``, which would read its ``labels`` column as absent.

    Row order is the file's in both branches, which is what makes the blob index-alignable
    with the ``LabelledDataset`` a consumer rebuilds from the same file.
    """
    import json

    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

    with path.open() as f:
        first = json.loads(f.readline())
    if isinstance(first.get("inputs"), list):
        rows = [json.loads(line) for line in path.open()]
        ds = LabelledDataset(
            inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]]
                    for r in rows],
            ids=[r.get("id", str(i)) for i, r in enumerate(rows)],
            other_fields={"labels": [r.get("label", r.get("labels")) for r in rows]})
        # load_from applies the transforms on the way in; this branch has to do it here.
        ds = _apply_transforms(ds, combine, convert)
    else:
        ds = LabelledDataset.load_from(
            path, pos_class_label=pos, neg_class_label=neg,
            combine_consecutive_messages=combine, convert_tool_to_assistant=convert)
    return list(ds.inputs)


def _apply_transforms(ds, combine: bool, convert: bool):
    """The same transforms ``load_from`` applies, for the branch that bypasses it.

    Delegates to ``retrain._apply_message_transforms``, which is what the retrain path
    uses on in-memory red-team records — so a red-team conversation is tokenized here
    exactly as it is when it becomes training data.
    """
    from agentic_redteam.retrain import _apply_message_transforms

    return _apply_message_transforms(ds, combine, convert)


def _pad_or_truncate(tensors, max_len):
    """Byte-for-byte the rule in ``LLMModel.get_activations`` (tuberlens/model.py:468)."""
    import torch
    out = []
    for t in tensors:
        if t.shape[1] < max_len:
            pad = torch.zeros(t.shape[0], max_len - t.shape[1], *t.shape[2:],
                              device=t.device, dtype=t.dtype)
            out.append(torch.cat([t, pad], dim=1))
        else:
            out.append(t[:, :max_len])
    return out


def _extract_split(llm, hooked, layers, inputs, *, batch_size: int, label: str):
    """One pass over ``inputs``; returns {layer: (acts, mask, ids)} padded like tuberlens."""
    import torch

    per_layer = {layer: [] for layer in layers}
    masks, ids = [], []
    t0 = time.time()
    n = len(inputs)
    for i in range(0, n, batch_size):
        batch = llm.tokenize(inputs[i : i + batch_size])
        acts = hooked.get_acts(batch)                     # (n_layers, b, s, e) on CPU
        for j, layer in enumerate(layers):
            per_layer[layer].append(acts[j].half().cpu())
        masks.append(batch["attention_mask"].cpu())
        ids.append(batch["input_ids"].cpu())
        del acts, batch
        done = min(i + batch_size, n)
        if done % max(batch_size, 10) < batch_size or done == n:
            el = time.time() - t0
            print(f"    [{label}] {done}/{n} rows  {el/done:.2f}s/row  "
                  f"eta {(n-done)*el/done/60:.1f} min", flush=True)

    max_len = min(MAX_ACTIVATION_TOKENS, max(a.shape[1] for a in per_layer[layers[0]]))
    masks = torch.cat(_pad_or_truncate(masks, max_len), dim=0)
    ids = torch.cat(_pad_or_truncate(ids, max_len), dim=0)
    out = {}
    for layer in layers:
        out[layer] = (torch.cat(_pad_or_truncate(per_layer.pop(layer), max_len), dim=0),
                      masks, ids)
    return out


def main(argv=None) -> int:
    args = _parse_args(argv)
    import torch
    from agentic_redteam.model_loading import load_extraction_model, unhook_model
    from agentic_redteam.retrain import _cpu_unpickle
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.model import HookedModel

    layers = sorted(set(args.layers))
    out_root = args.out_root
    if args.limit:
        out_root = out_root.parent / f"{out_root.name}_limit{args.limit}"

    with args.probe.open("rb") as f:
        probe = _cpu_unpickle(f)
    model_name = probe.model_name
    pos, neg = probe.pos_class_label, probe.neg_class_label
    print(f"[extract] model={model_name} layers={layers} batch_size={args.batch_size}")
    print(f"[extract] transforms: combine={args.combine_consecutive_messages} "
          f"convert={args.convert_tool_to_assistant}")

    # --- what needs computing ---------------------------------------------------------
    work = []   # (group, split, jsonl_path, inputs, missing_layers)
    for spec in args.extra_jsonl:
        name, _, raw = spec.partition("=")
        if not raw:
            raise SystemExit(f"--extra-jsonl expects NAME=PATH, got {spec!r}")
        inputs = _load_extra_jsonl(Path(raw), pos, neg,
                                   args.combine_consecutive_messages,
                                   args.convert_tool_to_assistant)
        if args.limit:
            inputs = inputs[: args.limit]
        missing = [
            l for l in layers
            if args.force or not _blob_ok(_blob_path(out_root, l, "extra", name),
                                          model_name=model_name, layer=l,
                                          n_rows=len(inputs))
        ]
        work.append(("extra", name, Path(raw), inputs, missing))
    for data_dir in args.data_dir:
        # eval_sets/instructions -> "eval_sets"; dev_samples/instructions -> "dev_samples"
        group = data_dir.parent.name or data_dir.name
        for jsonl in sorted(data_dir.glob("*.jsonl")):
            split = jsonl.stem
            if args.splits and split not in args.splits:
                continue
            ds = LabelledDataset.load_from(
                jsonl, pos_class_label=pos, neg_class_label=neg,
                combine_consecutive_messages=args.combine_consecutive_messages,
                convert_tool_to_assistant=args.convert_tool_to_assistant)
            inputs = list(ds.inputs)[: args.limit] if args.limit else list(ds.inputs)
            missing = [
                l for l in layers
                if args.force or not _blob_ok(_blob_path(out_root, l, group, split),
                                              model_name=model_name, layer=l,
                                              n_rows=len(inputs))
            ]
            work.append((group, split, jsonl, inputs, missing))

    total_rows = sum(len(inp) for *_r, inp, miss in work if miss)
    print(f"[extract] {len(work)} splits; {sum(1 for *_ , m in work if m)} need work; "
          f"{total_rows} rows to forward")
    for group, split, _, inputs, missing in work:
        state = f"layers {missing}" if missing else "cached"
        print(f"    {group}/{split}: {len(inputs)} rows -> {state}")
    if args.dry_run:
        return 0
    if not total_rows and not args.verify_batching:
        print("[extract] nothing to do")
        return 0

    # --- load once --------------------------------------------------------------------
    llm = load_extraction_model(model_name, max(layers), verbose=True)
    if args.batch_size > 1:
        llm.tokenizer.padding_side = "right"
        print("[extract] padding_side forced to 'right' for batched extraction")

    try:
        with torch.no_grad(), HookedModel(llm.model, layers) as hooked:
            if args.verify_batching:
                n = args.verify_batching
                inputs = work[0][3][:n]
                a1 = _extract_split(llm, hooked, layers, inputs, batch_size=1, label="b1")
                llm.tokenizer.padding_side = "right"
                ab = _extract_split(llm, hooked, layers, inputs,
                                    batch_size=args.batch_size, label=f"b{args.batch_size}")
                for l in layers:
                    x, y = a1[l][0].float(), ab[l][0].float()
                    m = a1[l][1].bool()
                    d = (x - y).abs()[m].max().item() if m.any() else float("nan")
                    print(f"  layer {l}: max|b1-b{args.batch_size}| over real tokens = {d:.5f} "
                          f"(scale {x[m].abs().mean().item():.3f})")
                return 0

            for group, split, jsonl, inputs, missing in work:
                if not missing:
                    continue
                print(f"[extract] {group}/{split}: {len(inputs)} rows, layers {missing}",
                      flush=True)
                res = _extract_split(llm, hooked, layers, inputs,
                                     batch_size=args.batch_size, label=f"{group}/{split}")
                for layer in missing:
                    acts, mask, ids = res[layer]
                    path = _blob_path(out_root, layer, group, split)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".pt.tmp")
                    torch.save({"activations": acts, "attention_mask": mask,
                                "input_ids": ids, "layer": layer, "model_name": model_name},
                               tmp)
                    tmp.rename(path)
                    print(f"    wrote {path}  {tuple(acts.shape)}  "
                          f"{path.stat().st_size/1e9:.2f} GB", flush=True)
                del res
                gc.collect()
    finally:
        unhook_model(llm)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    print("[extract] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
