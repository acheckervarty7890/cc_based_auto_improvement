#!/usr/bin/env python
"""Extract the red-team activations both experiment11 gemma-3-27b arms retrained on,
and publish them to Kaggle.

This is the red-team counterpart of ``scripts/publish_kaggle_eval_activations.py``. That
script publishes *eval-split* blobs; this one publishes the **per-conversation** blobs the
retrain path reads through ``retrain._redteam_activation_cache_path`` — the cache that
turns a retrain of a 27B-probe run from days of forward passes into a disk read.

Why it exists
-------------
The two arms of ``run_gemma27b_hu_harm_attacker_ablation_batch.sh`` ran on a cloud box and
wrote their activation cache into ``results_hu_harm_gemma27b_batch_ablation/``. That
directory was never synced back — only the probes, the postprocessed red-team snapshots and
the results JSONLs survived. So the ~1.8k activations those three retrains per arm were
built on exist nowhere, and any re-derivation (retrain sweeps, probe-architecture ablations,
regression analysis) has to recompute them through gemma-3-27b from scratch.

What is reconstructed, and why it is exact
------------------------------------------
``probes/<arm>/redteam_postprocessed_iter{1,2,3}.jsonl`` is dumped by ``retrain_probe``
*after* ``_apply_message_transforms`` and *before* the split, so it holds the red-team
conversations in exactly the form ``_activate_redteam_cached`` hashed and forwarded — it is
a verbatim record of what trained each probe. The union over the three iterations is
therefore the exact set of conversations that arm's cache held when the run finished.

Keys are computed by importing ``retrain._redteam_activation_cache_path`` rather than
re-deriving the hash here, so the blobs this writes cannot drift from the ones a retrain
looks for. ``model_name`` / ``layer`` come off the arm's own ``probe_iter0.pkl`` (probe
metadata is the source of truth); the two message transforms come from the config's
``eval:`` section and are additionally *checked* against the dump — re-applying them must be
a no-op, and if it is not, the config does not describe the run that wrote the dump and every
key would be wrong.

Three stages, run independently (extract is the expensive one and is resumable):

    extract   compute the missing blobs into a cache dir, one file per conversation
    publish   zip one (arm, iteration)'s blobs + a manifest and upload as a Kaggle dataset
    restore   pull an (arm, iteration) dataset back into a cache dir, verified against this repo

Iterations are addressed separately, and each dataset is SELF-CONTAINED
-----------------------------------------------------------------------
``extract`` works over the **union** — a conversation carried across iterations is one
content-addressed blob, computed once — but ``publish`` emits one dataset per
``(arm, iteration)`` holding *every* blob that iteration trained on, not a delta.

That is not redundancy for its own sake: **the iterations do not nest.** ``filter_dataset``
refits its bag-of-words classifier on the growing success set each cycle, so the
top-percentile it drops shifts, and 16-84 conversations present in one iteration are absent
from the next (measured across both arms). So "iterations 1..k unioned" is *not* what probe
k was trained on, and only an exact per-iteration membership reproduces it. Self-contained
datasets make each one independently correct — restoring iteration 3 needs iteration 3 and
nothing else — at the cost of storing the shared blobs more than once.

Because the blobs are content-addressed and restore writes them into the flat cache layout
``redteam_acts_<model>_L<layer>/<key>.pt``, restoring several iterations into one cache dir
is safe: a key shared by two iterations is the same bytes from either archive.

Cost warning
------------
Extraction is ~1.8k forward passes through ``google/gemma-3-27b-it`` truncated to layers
0..32. On a box that can hold the truncated model this is hours; on one that disk-offloads
it is days. Blobs are written through per chunk, so ``extract`` is interruptible and picks
up exactly where it stopped — and ``--dry-run`` tells you the miss count before you commit.

Typical use::

    # what would be computed, and where it would go
    .venv_claude/bin/python scripts/publish_kaggle_redteam_activations.py extract --dry-run

    # compute (long; nohup it)
    .venv_claude/bin/python scripts/publish_kaggle_redteam_activations.py extract

    # inspect the upload plan (all arms x all iterations), then drop --dry-run
    .venv_claude/bin/python scripts/publish_kaggle_redteam_activations.py publish --dry-run

    # on another box, prime a retrain's cache with exactly what probe_iter3 trained on
    .venv_claude/bin/python scripts/publish_kaggle_redteam_activations.py restore \
        --arm gptoss120b --iterations 3 \
        --cache-dir results_hu_harm_gemma27b_batch_ablation/base_activations

Needs ``KAGGLE_CONFIG_DIR`` (the DIRECTORY holding kaggle.json) or ``KAGGLE_API_TOKEN`` for
``publish`` / ``restore``; ``extract`` needs neither.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_redteam.kaggle_activations import (  # noqa: E402
    KaggleActivationError,
    _authenticate,
    _blob_header,
    _slugify,
)
from agentic_redteam.retrain import _redteam_activation_cache_path  # noqa: E402

# The manifest that travels inside each uploaded archive. Bumped only if its shape changes.
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

# Progress cadence while computing misses — matches retrain._REDTEAM_PROGRESS_EVERY, i.e. a
# line every few minutes on an offloaded 27B, not a spinner.
PROGRESS_EVERY = 10


@dataclass(frozen=True)
class Arm:
    """One arm of the experiment11 gemma-3-27b batch ablation."""

    name: str
    config: str  # relative to REPO_ROOT
    probe_dir: str  # relative to REPO_ROOT

    @property
    def config_path(self) -> Path:
        return REPO_ROOT / self.config

    @property
    def probe_path(self) -> Path:
        return REPO_ROOT / self.probe_dir


# Both arms of run_gemma27b_hu_harm_attacker_ablation_batch.sh, in the order it ran them.
ARMS: dict[str, Arm] = {
    "gptoss120b": Arm(
        name="gptoss120b",
        config="configs/gptoss120b_hu_harm_gemma27b_batch.md",
        probe_dir="probes/hu_harm_gemma27b_gptoss120b_batch",
    ),
    "deepseekv4pro": Arm(
        name="deepseekv4pro",
        config="configs/deepseekv4pro_hu_harm_gemma27b_batch.md",
        probe_dir="probes/hu_harm_gemma27b_deepseekv4pro_batch",
    ),
}

# The shared cache dir the two arms wrote to on the cloud box (configs'
# output.base_activation_cache_dir). Both arms share it: the per-conversation keys are
# content-addressed against a frozen LLM, so their distinct successes get distinct names.
DEFAULT_CACHE_DIR = REPO_ROOT / "results_hu_harm_gemma27b_batch_ablation" / "base_activations"

DEFAULT_OWNER = "anku7890"
# {arm} is the slugified arm name, {iteration} the 1-based retrain index. Kaggle slugs cap
# at 50 characters; the longest these render to is 36.
DEFAULT_SLUG = "hu-harm-gemma27b-{arm}-iter{iteration}"
DEFAULT_ARCHIVE = "{arm}-iter{iteration}-redteam-acts.zip"


# --------------------------------------------------------------------------------------
# Resolving an arm: what model, what layer, what transforms, which conversations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Conversation:
    """One red-team conversation as it was fed to the extraction model."""

    key: str  # the cache blob's stem — first 32 hex of the content hash
    path: Path  # where its blob lives under the cache dir
    messages: list  # tuberlens Message objects
    label: str  # canonical "positive" / "negative" it trained with
    ids: list[str]  # the dump's own id(s) for it
    iterations: list[int]  # which redteam_postprocessed_iter{N} files it appeared in


@dataclass(frozen=True)
class ArmPlan:
    """One arm's conversation set — either its whole union, or one iteration's slice."""

    arm: Arm
    model_name: str
    layer: int
    combine_consecutive_messages: bool
    convert_tool_to_assistant: bool
    pos_class_label: str
    neg_class_label: str
    conversations: list[Conversation]
    iteration: int | None = None  # None = the union across every iteration

    @property
    def blob_subdir(self) -> str:
        """Directory name the retrain path writes these blobs under, relative to the cache dir.

        Deliberately carries no iteration: the cache a retrain reads is flat and
        content-addressed, so a conversation shared by two iterations is one file. Only the
        *archives* are sliced per iteration.
        """
        return f"redteam_acts_{self.model_name.replace('/', '_')}_L{self.layer}"

    @property
    def iterations(self) -> list[int]:
        """Every iteration this plan's conversations appear in, ascending."""
        return sorted({it for c in self.conversations for it in c.iterations})

    def for_iteration(self, iteration: int) -> "ArmPlan":
        """The slice of this plan that iteration ``iteration`` was retrained on.

        A full membership, not a delta — see the module docstring on why the iterations
        cannot be reconstructed by unioning the ones before them.
        """
        return replace(
            self,
            iteration=iteration,
            conversations=[c for c in self.conversations if iteration in c.iterations],
        )

    def missing(self) -> list[Conversation]:
        return [c for c in self.conversations if not c.path.exists()]

    def present(self) -> list[Conversation]:
        return [c for c in self.conversations if c.path.exists()]


def _probe_metadata(arm: Arm) -> tuple[str, int, str, str]:
    """Read ``(model_name, layer, pos_class_label, neg_class_label)`` off the arm's probe.

    The probe pickle, not the config, is the source of truth for these (see the conventions
    in CLAUDE.md). ``probe_iter0.pkl`` is the initial probe; every later one in the run
    shares its model and layer, so any of them would do. Unpickled onto the CPU — nothing
    here touches a GPU.
    """
    from agentic_redteam.probe_judge import _cpu_unpickle

    candidates = sorted(arm.probe_path.glob("probe_iter*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"{arm.name}: no probe_iter*.pkl in {arm.probe_path} — cannot read model/layer."
        )
    with candidates[0].open("rb") as fh:
        probe = _cpu_unpickle(fh)
    if probe.model_name is None or probe.layer is None:
        raise ValueError(f"{arm.name}: {candidates[0]} carries no model_name/layer.")
    return (
        str(probe.model_name),
        int(probe.layer),
        getattr(probe, "pos_class_label", "positive") or "positive",
        getattr(probe, "neg_class_label", "negative") or "negative",
    )


def _check_transforms_are_noop(arm: Arm, dialogues, combine: bool, convert: bool) -> None:
    """Fail loudly if re-applying the transforms would change the dumped conversations.

    ``retrain_probe`` dumps the postprocessed snapshot *after* ``_apply_message_transforms``,
    so applying them again must be a no-op. If it is not, the transform flags we are folding
    into the cache key are not the ones the run used, and every key here would be wrong —
    which the cache cannot detect later, because it loads by path without validating inputs.
    """
    if not (combine or convert):
        return
    from tuberlens.interfaces.dataset import LabelledDataset

    for i, dialogue in enumerate(dialogues):
        again = dialogue
        if convert:
            again = LabelledDataset._convert_tool_to_assistant(again)
        if combine:
            again = LabelledDataset._combine_consecutive_messages(again)
        if [(m.role, m.content) for m in again] != [(m.role, m.content) for m in dialogue]:
            raise ValueError(
                f"{arm.name}: conversation {i} of the postprocessed dump changes under "
                f"combine={combine}/convert={convert}, so the dump was NOT written with "
                "those transforms. Refusing to compute activations under keys the retrain "
                "would never look up — check the config's eval: section."
            )


def build_plan(
    arm: Arm,
    cache_dir: Path,
    *,
    combine_override: bool | None = None,
    convert_override: bool | None = None,
    limit: int | None = None,
) -> ArmPlan:
    """Resolve one arm into the exact conversation set its retrains activated."""
    from tuberlens.interfaces.dataset import Message as TLMessage

    from agentic_redteam.config import load_config

    model_name, layer, pos_label, neg_label = _probe_metadata(arm)

    config = load_config(arm.config_path)
    combine = (
        combine_override
        if combine_override is not None
        else bool(config.eval.combine_consecutive_messages)
    )
    convert = (
        convert_override
        if convert_override is not None
        else bool(config.eval.convert_tool_to_assistant)
    )

    dumps = sorted(
        arm.probe_path.glob("redteam_postprocessed_iter*.jsonl"),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0),
    )
    if not dumps:
        raise FileNotFoundError(
            f"{arm.name}: no redteam_postprocessed_iter*.jsonl in {arm.probe_path} — "
            "there is no record of what this run retrained on."
        )

    # Union across iterations, first-seen order. A conversation carried forward from an
    # earlier iteration is one cache entry, not three, which is exactly how the retrain
    # path treated it (a hit in iteration k+1 for a row first computed in iteration k).
    by_key: dict[str, Conversation] = {}
    for dump in dumps:
        it = int("".join(ch for ch in dump.stem if ch.isdigit()) or 0)
        with dump.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = [
                    TLMessage(role=m["role"], content=m["content"]) for m in row["inputs"]
                ]
                path = _redteam_activation_cache_path(
                    cache_dir, messages, model_name, layer, combine, convert
                )
                key = path.stem
                prior = by_key.get(key)
                if prior is None:
                    by_key[key] = Conversation(
                        key=key,
                        path=path,
                        messages=messages,
                        label=row.get("label"),
                        ids=[row.get("id")],
                        iterations=[it],
                    )
                else:
                    if it not in prior.iterations:
                        prior.iterations.append(it)
                    if row.get("id") not in prior.ids:
                        prior.ids.append(row.get("id"))

    conversations = list(by_key.values())
    _check_transforms_are_noop(arm, [c.messages for c in conversations], combine, convert)
    if limit is not None:
        conversations = conversations[:limit]

    return ArmPlan(
        arm=arm,
        model_name=model_name,
        layer=layer,
        combine_consecutive_messages=combine,
        convert_tool_to_assistant=convert,
        pos_class_label=pos_label,
        neg_class_label=neg_label,
        conversations=conversations,
    )


def _selected_arms(names: list[str] | None) -> list[Arm]:
    if not names or "all" in names:
        return list(ARMS.values())
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")
    return [ARMS[n] for n in names]


# --------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------


def _save_blob(path: Path, acts, j: int, model_name: str, layer: int) -> None:
    """Write one conversation's activations in the layout ``LLMModel.load_activations`` reads.

    Byte-for-byte the dict ``retrain._activate_redteam_cached`` saves, including the
    ``model_name`` / ``layer`` provenance fields tuberlens itself drops on load but which
    ``_validate_blob`` checks before a downloaded blob is trusted.
    """
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.partial")
    torch.save(
        {
            "activations": acts.activations[j : j + 1].clone(),
            "attention_mask": acts.attention_mask[j : j + 1].clone(),
            "input_ids": acts.input_ids[j : j + 1].clone(),
            "layer": layer,
            "model_name": model_name,
        },
        tmp,
    )
    # Rename last: a blob only becomes visible to the cache once it is complete, so a kill
    # mid-write can never leave a truncated file that a later run would happily load.
    os.replace(tmp, path)


def cmd_extract(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir).resolve()
    plans = [
        build_plan(
            arm,
            cache_dir,
            combine_override=args.combine_consecutive_messages,
            convert_override=args.convert_tool_to_assistant,
            limit=args.limit,
        )
        for arm in _selected_arms(args.arm)
    ]

    print(f"cache dir: {cache_dir}")
    total_missing: list[tuple[ArmPlan, Conversation]] = []
    for plan in plans:
        missing = plan.missing()
        print(
            f"  {plan.arm.name:<14} {len(plan.conversations):>5} conversations  "
            f"{len(plan.present()):>5} cached  {len(missing):>5} to compute   "
            f"[{plan.model_name} L{plan.layer}, combine={plan.combine_consecutive_messages}, "
            f"convert={plan.convert_tool_to_assistant}]"
        )
        # The union is what gets computed; the per-iteration counts are what gets published,
        # and they sum to more than the union because the iterations overlap.
        per_iter = "  ".join(
            f"iter{it}={len(plan.for_iteration(it).conversations)}" for it in plan.iterations
        )
        print(f"  {'':<14} publishes as: {per_iter}")
        total_missing.extend((plan, c) for c in missing)

    # Two arms can legitimately share a conversation (nothing forbids it); computing it
    # twice would be wasted forwards writing the same file.
    seen: set[str] = set()
    work: list[tuple[ArmPlan, Conversation]] = []
    for plan, conv in total_missing:
        if conv.key in seen:
            continue
        seen.add(conv.key)
        work.append((plan, conv))

    print(f"\n{len(work)} unique conversation(s) to compute.")
    if not work:
        print("Nothing to do — every blob is already on disk.")
        return 0
    if args.dry_run:
        print("--dry-run: nothing computed. Re-run without it to extract.")
        return 0

    from agentic_redteam.model_loading import extraction_batch_size, load_extraction_model

    # All conversations in a run share one model/layer, but the loop is written per plan so
    # a future arm on a different probe still works: the model is reloaded when it changes.
    chunk_size = extraction_batch_size()
    print(f"chunk size {chunk_size} (tuberlens BATCH_SIZE)\n", flush=True)

    model = None
    loaded_for: tuple[str, int] | None = None
    started = time.monotonic()
    done = 0
    try:
        for start in range(0, len(work), chunk_size):
            chunk = work[start : start + chunk_size]
            # Never mix models within a chunk.
            spec = (chunk[0][0].model_name, chunk[0][0].layer)
            chunk = [item for item in chunk if (item[0].model_name, item[0].layer) == spec]
            if loaded_for != spec:
                if model is not None:
                    del model
                    gc.collect()
                model = load_extraction_model(spec[0], spec[1])
                loaded_for = spec
            acts = model.get_activations(
                [c.messages for _, c in chunk], layer=spec[1], show_progress=False
            )
            for j, (_plan, conv) in enumerate(chunk):
                _save_blob(conv.path, acts, j, spec[0], spec[1])
            del acts
            done += len(chunk)
            if done % PROGRESS_EVERY == 0 or done == len(work):
                elapsed = time.monotonic() - started
                rate = elapsed / done
                print(
                    f"  [red-team activations] {done}/{len(work)} "
                    f"({rate:.1f}s/sample, ~{(len(work) - done) * rate / 60:.0f} min left)",
                    flush=True,
                )
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup must never mask the real error
            pass

    still_missing = sum(len(p.missing()) for p in plans)
    print(f"\nExtracted {done} conversation(s). Still missing: {still_missing}.")
    return 0 if still_missing == 0 else 1


# --------------------------------------------------------------------------------------
# publish
# --------------------------------------------------------------------------------------


def _validate_row_blob(plan: ArmPlan, conv: Conversation) -> None:
    """Raise unless this blob really is one row of this arm's probe's activations.

    Same contract as ``kaggle_activations._validate_blob``, at per-conversation
    granularity: a blob is checked against the probe it claims to belong to *before* it is
    published, so a mismatch surfaces here rather than hours into a remote retrain.
    """
    try:
        data = _blob_header(conv.path)
    except Exception as e:  # noqa: BLE001 — reported as a validation failure
        raise KaggleActivationError(f"{plan.arm.name}/{conv.key}: unreadable: {e}") from e

    missing = {"activations", "attention_mask", "input_ids"} - set(data)
    if missing:
        raise KaggleActivationError(
            f"{plan.arm.name}/{conv.key}: missing tensor field(s) {sorted(missing)}."
        )
    problems = []
    if data.get("model_name") not in (None, plan.model_name):
        problems.append(f"model_name={data['model_name']!r} (expected {plan.model_name!r})")
    if data.get("layer") is not None and int(data["layer"]) != plan.layer:
        problems.append(f"layer={data['layer']} (expected {plan.layer})")
    rows = int(data["activations"].shape[0])
    if rows != 1:
        problems.append(f"{rows} rows (a per-conversation blob must hold exactly 1)")
    if problems:
        raise KaggleActivationError(f"{plan.arm.name}/{conv.key}: " + "; ".join(problems))


def _manifest(plan: ArmPlan) -> dict:
    """Provenance for the archive: enough to tell what these blobs are without this repo."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "arm": plan.arm.name,
        "iteration": plan.iteration,
        "config": plan.arm.config,
        "probe_dir": plan.arm.probe_dir,
        "model_name": plan.model_name,
        "layer": plan.layer,
        "combine_consecutive_messages": plan.combine_consecutive_messages,
        "convert_tool_to_assistant": plan.convert_tool_to_assistant,
        "pos_class_label": plan.pos_class_label,
        "neg_class_label": plan.neg_class_label,
        "blob_subdir": plan.blob_subdir,
        "n_conversations": len(plan.conversations),
        "self_contained": True,
        "source": (
            f"probes/{plan.arm.probe_dir.split('/')[-1]}/redteam_postprocessed_iter"
            f"{plan.iteration if plan.iteration is not None else '*'}.jsonl — the FULL set of "
            "conversations that retrain was trained on (not a delta against earlier "
            "iterations, which do not nest), keyed by "
            "agentic_redteam.retrain._redteam_activation_cache_path"
        ),
        "conversations": [
            {
                "key": c.key,
                "file": f"{plan.blob_subdir}/{c.key}.pt",
                "label": c.label,
                "ids": c.ids,
                "iterations": sorted(c.iterations),
                "n_messages": len(c.messages),
            }
            for c in plan.conversations
        ],
    }


def _build_archive(plan: ArmPlan, dest: Path) -> Path:
    """Zip an arm's blobs + manifest into ``dest``.

    ``ZIP_STORED``: the payload is fp16 activation tensors, which do not compress, so
    deflating would cost CPU-minutes per GB for ~nothing. The archive's internal layout is
    the cache's own (``redteam_acts_<model>_L<layer>/<key>.pt``), so restoring is an
    extract-in-place into any ``base_activation_cache_dir``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(_manifest(plan), indent=2))
        for i, conv in enumerate(plan.conversations, 1):
            zf.write(conv.path, arcname=f"{plan.blob_subdir}/{conv.key}.pt")
            if i % 200 == 0 or i == len(plan.conversations):
                print(f"    packed {i}/{len(plan.conversations)}", flush=True)
    return dest


def _title_for(arm_name: str, iteration: int) -> str:
    """Kaggle titles must be 6-50 characters."""
    return f"{arm_name} gemma27b L32 redteam acts iter{iteration}"[:50]


def _resolve_iterations(plan: ArmPlan, requested: list[int] | None) -> list[int]:
    """The iterations to act on for this arm, validated against what the dumps contain."""
    available = plan.iterations
    if not requested:
        return available
    unknown = [it for it in requested if it not in available]
    if unknown:
        raise SystemExit(
            f"{plan.arm.name}: no redteam_postprocessed_iter{unknown} dump(s); "
            f"available iterations are {available}."
        )
    return sorted(requested)


def cmd_publish(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir).resolve()
    plans = [
        build_plan(
            arm,
            cache_dir,
            combine_override=args.combine_consecutive_messages,
            convert_override=args.convert_tool_to_assistant,
        )
        for arm in _selected_arms(args.arm)
    ]

    # Fan each arm out into one self-contained plan per requested iteration.
    slices = [
        plan.for_iteration(it)
        for plan in plans
        for it in _resolve_iterations(plan, args.iterations)
    ]

    # --- validate everything BEFORE uploading anything --------------------------------
    # Blobs are validated once per unique key, not once per slice: the same conversation
    # appears in up to three slices and re-reading its header each time buys nothing.
    problems: list[str] = []
    validated: set[str] = set()
    for plan in slices:
        missing = plan.missing()
        if missing:
            problems.append(
                f"{plan.arm.name} iter{plan.iteration}: {len(missing)} of "
                f"{len(plan.conversations)} blob(s) not computed yet — run `extract` first."
            )
            continue
        for conv in plan.conversations:
            if conv.key in validated:
                continue
            validated.add(conv.key)
            try:
                _validate_row_blob(plan, conv)
            except KaggleActivationError as e:
                problems.append(str(e))
    if problems:
        for p in problems[:20]:
            print(f"INVALID  {p}", file=sys.stderr)
        if len(problems) > 20:
            print(f"... and {len(problems) - 20} more", file=sys.stderr)
        print(f"\n{len(problems)} problem(s) — nothing was uploaded.", file=sys.stderr)
        return 1

    print(f"{'PUBLIC' if args.public else 'private'} dataset(s) owned by {args.owner}:\n")
    plan_rows = []
    for plan in slices:
        slug = _render(args.dataset_slug, plan)
        archive_name = _render(args.archive_name, plan)
        size = sum(c.path.stat().st_size for c in plan.conversations)
        plan_rows.append((plan, slug, archive_name, size))
        label = f"{plan.arm.name} iter{plan.iteration}"
        print(f"  {label:<22} {len(plan.conversations):>5} conversations  {size / 1e9:>6.2f} GB")
        print(f"  {'':<22} -> kaggle.com/datasets/{args.owner}/{slug}  ({archive_name})")
    # Self-contained slices overlap, so the upload total exceeds the on-disk union.
    unique_bytes = sum(
        c.path.stat().st_size
        for c in {c.key: c for plan in slices for c in plan.conversations}.values()
    )
    print(
        f"\n  {len(plan_rows)} dataset(s), {sum(r[3] for r in plan_rows) / 1e9:.2f} GB uploaded "
        f"({unique_bytes / 1e9:.2f} GB of unique blobs — the iterations overlap)\n"
    )
    if args.dry_run:
        print("--dry-run: nothing uploaded. Re-run without it to publish.")
        return 0

    api = _authenticate()
    failures = 0
    for plan, slug, archive_name, _size in plan_rows:
        # Stage inside the cache dir so the archive lands on the same filesystem as the
        # blobs it is built from (and is cleaned up even if the upload raises).
        label = f"{plan.arm.name} iter{plan.iteration}"
        with tempfile.TemporaryDirectory(
            dir=str(cache_dir), prefix=f".publish_{plan.arm.name}_{plan.iteration}_"
        ) as tmp:
            staging = Path(tmp)
            print(f">>> packing {label} ...", flush=True)
            _build_archive(plan, staging / archive_name)
            (staging / "dataset-metadata.json").write_text(
                json.dumps(
                    {
                        "title": _title_for(plan.arm.name, plan.iteration),
                        "id": f"{args.owner}/{slug}",
                        "licenses": [{"name": "CC0-1.0"}],
                    },
                    indent=2,
                )
            )
            print(f">>> uploading {label} -> {args.owner}/{slug} ...", flush=True)
            try:
                if args.new_version:
                    api.dataset_create_version(
                        str(staging),
                        version_notes=(
                            f"{label} red-team activations "
                            f"({plan.model_name} L{plan.layer}, "
                            f"{len(plan.conversations)} conversations)"
                        ),
                        dir_mode="skip",
                        convert_to_csv=False,
                    )
                else:
                    api.dataset_create_new(
                        str(staging),
                        public=args.public,
                        dir_mode="skip",
                        convert_to_csv=False,
                    )
            except Exception as e:  # noqa: BLE001 — report and continue to the next slice
                failures += 1
                print(f"    FAILED {label}: {e}", file=sys.stderr)
                continue
            print(f"    done: kaggle.com/datasets/{args.owner}/{slug}")

    if failures:
        print(f"\n{failures} of {len(plan_rows)} upload(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(plan_rows)} dataset(s) published.")
    return 0


# --------------------------------------------------------------------------------------
# restore
# --------------------------------------------------------------------------------------


def _locate_archive(staging: Path, archive_name: str) -> Path:
    """Find our archive in whatever Kaggle actually served.

    ``dataset_download_file`` names its output from the download URL, not from the
    requested name, and Kaggle may wrap a large file in an extra zip — so the archive has
    to be discovered rather than assumed.
    """
    direct = staging / archive_name
    if direct.is_file():
        return direct
    zips = sorted(p for p in staging.rglob("*.zip") if p.is_file())
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            if MANIFEST_NAME in names:
                return z
            # An outer wrapper: unwrap once and look again.
            if len(names) == 1 and names[0].endswith(".zip"):
                zf.extractall(staging / "_unwrapped")
                inner = sorted((staging / "_unwrapped").rglob("*.zip"))
                if inner:
                    return inner[0]
    landed = sorted(p.name for p in staging.rglob("*") if p.is_file())
    raise KaggleActivationError(
        f"no archive containing {MANIFEST_NAME} in the download (got {landed or 'nothing'})."
    )


def cmd_restore(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    plans = [
        build_plan(
            arm,
            cache_dir,
            combine_override=args.combine_consecutive_messages,
            convert_override=args.convert_tool_to_assistant,
        )
        for arm in _selected_arms(args.arm)
    ]

    slices = [
        plan.for_iteration(it)
        for plan in plans
        for it in _resolve_iterations(plan, args.iterations)
    ]

    api = _authenticate()
    failures = 0
    for plan in slices:
        slug = _render(args.dataset_slug, plan)
        archive_name = _render(args.archive_name, plan)
        handle = f"{args.owner}/{slug}"
        expected = {c.key: c for c in plan.conversations}
        label = f"{plan.arm.name} iter{plan.iteration}"
        print(f">>> {label}: downloading {handle} ({archive_name}) ...", flush=True)
        with tempfile.TemporaryDirectory(
            dir=str(cache_dir), prefix=f".restore_{plan.arm.name}_{plan.iteration}_"
        ) as tmp:
            staging = Path(tmp)
            try:
                api.dataset_download_file(handle, archive_name, path=str(staging), quiet=False)
                archive = _locate_archive(staging, archive_name)
            except Exception as e:  # noqa: BLE001 — report and continue to the next slice
                failures += 1
                print(f"    FAILED {label}: {e}", file=sys.stderr)
                continue

            with zipfile.ZipFile(archive) as zf:
                names = [n for n in zf.namelist() if n.endswith(".pt")]
                got = {Path(n).stem for n in names}
                # The archive is validated against THIS repo's postprocessed dumps, not
                # against its own manifest: the keys are a pure function of the
                # conversations plus model/layer/transforms, so a mismatch means the blobs
                # are not the ones this arm's retrains used — exactly the silent staleness
                # the path-keyed caches cannot catch on their own.
                unexpected = got - set(expected)
                absent = set(expected) - got
                if unexpected or absent:
                    failures += 1
                    print(
                        f"    FAILED {label}: archive does not match "
                        f"redteam_postprocessed_iter{plan.iteration}.jsonl "
                        f"({len(absent)} expected key(s) absent, {len(unexpected)} "
                        "unexpected).",
                        file=sys.stderr,
                    )
                    continue
                written = 0
                for name in names:
                    dest = expected[Path(name).stem].path
                    if dest.exists() and not args.force:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp_dest = dest.with_suffix(".pt.partial")
                    with zf.open(name) as src, tmp_dest.open("wb") as out:
                        while True:
                            block = src.read(1 << 20)
                            if not block:
                                break
                            out.write(block)
                    os.replace(tmp_dest, dest)
                    written += 1
            print(
                f"    restored {written} new blob(s) "
                f"({len(expected) - written} already present) into {cache_dir}"
            )

    if failures:
        print(f"\n{failures} of {len(plans)} restore(s) failed.", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------------------


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--arm",
        nargs="*",
        default=None,
        help=f"Arms to act on (default: all). Known: {', '.join(sorted(ARMS))}",
    )
    ap.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Activation cache dir (a retrain's --base-activation-cache-dir). "
        f"Default: {DEFAULT_CACHE_DIR}",
    )
    ap.add_argument(
        "--combine-consecutive-messages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the config's eval.combine_consecutive_messages (folded into the "
        "cache key). Default: read from the config.",
    )
    ap.add_argument(
        "--convert-tool-to-assistant",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the config's eval.convert_tool_to_assistant (folded into the cache key).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print the plan; change nothing")


def _render(template: str, plan: ArmPlan) -> str:
    """Fill a slug / archive-name template for one (arm, iteration) slice."""
    return template.format(arm=_slugify(plan.arm.name), iteration=plan.iteration)


def _add_kaggle(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--owner", default=DEFAULT_OWNER, help="Kaggle username owning the datasets")
    ap.add_argument(
        "--iterations",
        nargs="*",
        type=int,
        default=None,
        help="Retrain iterations to act on (default: every one with a postprocessed dump). "
        "Each is a SELF-CONTAINED dataset — the iterations do not nest, so restoring "
        "iteration N needs only iteration N.",
    )
    ap.add_argument(
        "--dataset-slug",
        default=DEFAULT_SLUG,
        help="Slug template; {arm} is the slugified arm name, {iteration} the retrain index. "
        "Default: " + DEFAULT_SLUG,
    )
    ap.add_argument(
        "--archive-name",
        default=DEFAULT_ARCHIVE,
        help="Name of the archive inside the dataset. Default: " + DEFAULT_ARCHIVE,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="stage", required=True)

    p_ex = sub.add_parser("extract", help="Compute the missing per-conversation blobs")
    _add_common(p_ex)
    p_ex.add_argument(
        "--limit", type=int, default=None, help="Only the first N conversations per arm (smoke test)"
    )
    p_ex.set_defaults(func=cmd_extract)

    p_pub = sub.add_parser("publish", help="Zip an arm's blobs and upload to Kaggle")
    _add_common(p_pub)
    _add_kaggle(p_pub)
    p_pub.add_argument("--public", action="store_true", help="Publish public (default: private)")
    p_pub.add_argument(
        "--new-version",
        action="store_true",
        help="Push a new version of an existing dataset instead of creating one",
    )
    p_pub.set_defaults(func=cmd_publish)

    p_res = sub.add_parser("restore", help="Download an arm's dataset back into a cache dir")
    _add_common(p_res)
    _add_kaggle(p_res)
    p_res.add_argument("--force", action="store_true", help="Overwrite blobs already on disk")
    p_res.set_defaults(func=cmd_restore)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except KaggleActivationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
