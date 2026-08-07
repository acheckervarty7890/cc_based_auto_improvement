#!/usr/bin/env python
"""Extract the activations both experiment9 HIGH-STAKES gemma-3-27b arms retrained on, and
publish them to Kaggle — base training split included, uploaded incrementally so an unstable
box never loses completed work.

This is the red-team counterpart of ``scripts/publish_kaggle_eval_activations.py``. That
script publishes *eval-split* blobs; this one publishes the two caches a **retrain** reads:
the base training split's whole-split blobs (``retrain._base_activation_cache_paths``) and
the per-conversation red-team blobs (``retrain._redteam_activation_cache_path``).

Why it exists
-------------
The two arms of ``run_gemma27b_hs_attacker_ablation_batch.sh`` ran on a cloud box and wrote
their activation cache into ``results_hs_gemma27b_batch_ablation/``. That directory was never
synced back — only the probes, the postprocessed red-team snapshots and the results JSONLs
survived. So neither the base blobs nor the red-team activations exist anywhere, and any
re-derivation has to push them through gemma-3-27b again.

What is reconstructed, and why it is exact
------------------------------------------
*Base split.* Rebuilt the way ``retrain_probe`` builds it — ``LabelledDataset.load_from``
with the config's transforms, then ``stable_fraction_subsample``, then
``stable_train_test_split`` — and written to the path ``_base_activation_cache_paths``
derives. Both functions are imported, not reimplemented, so the key cannot drift; the CLI
defaults here (seed 42, test_size 0.2, no split field, fraction 1.0) are the ones
``run_gemma27b_hs_attacker_ablation_batch.sh`` left at their defaults.

*Red-team set.* ``probes/<arm>/redteam_postprocessed_iter{1,2,3}.jsonl`` is dumped by
``retrain_probe`` *after* ``_apply_message_transforms`` and *before* the split, so it holds
the conversations in exactly the form ``_activate_redteam_cached`` hashed and forwarded — a
verbatim record of what trained each probe.

``model_name`` / ``layer`` come off each arm's own ``probe_iter0.pkl`` (probe metadata is the
source of truth); the two message transforms come from the config's ``eval:`` section and are
additionally *checked* against the dump — re-applying them must be a no-op, and if it is not,
the config does not describe the run that wrote the dump and every key would be wrong.

Not clashing with the human-harm publication
--------------------------------------------
``experiment11_cloud`` carries a sibling script (``publish_kaggle_redteam_activations.py``)
that publishes the **harmful_to_human** gemma-3-27b arms. Both target the same Kaggle account
and the same probe model/layer, so the two are kept apart deliberately:

*Kaggle slugs.* One owner's dataset slugs are a flat namespace, so this is the only surface
where a collision would actually destroy data. Everything here lives under
``hs-gemma27b-*``; the human-harm run lives under ``hu-harm-gemma27b-*``. That separation is
**enforced, not just conventional** — ``_check_slug_namespace`` refuses to publish a unit
whose slug leaves this namespace (or enters a known foreign one), so a mistyped
``--dataset-slug`` cannot silently push high-stakes blobs over a human-harm dataset. Pass
``--allow-foreign-slug`` if you genuinely mean to.

*Script name.* ``publish_kaggle_hs_redteam_activations.py``, not the sibling's name, so the
two survive a merge of the branches side by side.

*Cache dir.* ``results_hs_gemma27b_batch_ablation/`` vs the human-harm run's
``results_hu_harm_gemma27b_batch_ablation/``.

*Blob file names need no separation, and get none.* The base blobs are keyed on the base data
file's bytes, so ``hs_ls_200.jsonl`` and ``hu_harm_llama70b_50.jsonl`` produce different names
in any directory. The red-team blobs live under ``redteam_acts_google_gemma-3-27b-it_L32/``
for **both** concepts — that subdir is derived from the model and layer alone — but each blob
is content-addressed on its own conversation, so two concepts mixed into one cache dir simply
do not collide, and a conversation genuinely common to both is the same bytes either way.

Units, and why uploads are interleaved with extraction
------------------------------------------------------
Work is divided into **units**, each of which becomes one Kaggle dataset:

    base                      the base split's train + val blobs (shared by both arms)
    <arm> iter<N>             every blob that arm's iteration-N retrain was trained on

``sync`` walks the units in order and, for each, extracts what is missing and uploads it
**before starting the next**. On a container that can vanish mid-run that is the difference
between losing one unit's forwards and losing all of them. On restart ``sync`` asks Kaggle
which units already exist, skips those, and — unless ``--no-restore-published`` — pulls their
blobs back into the local cache first, so a fresh box does not recompute the conversations
iteration 2 shares with iteration 1.

``extract`` / ``publish`` / ``restore`` remain available as separate stages for when the box
is stable and the interleaving is not worth the round trips.

Each iteration dataset is SELF-CONTAINED
----------------------------------------
``publish`` emits every blob that iteration trained on, not a delta against earlier ones.
That is not redundancy for its own sake: **the iterations do not nest.** ``filter_dataset``
refits its bag-of-words classifier on the growing success set each cycle, so the
top-percentile it drops shifts, and conversations present in one iteration are absent from the
next. "Iterations 1..k unioned" is therefore *not* what probe k was trained on, and only an
exact per-iteration membership reproduces it. Self-contained datasets make each one
independently correct — restoring iteration 3 needs iteration 3 and nothing else — at the cost
of storing shared blobs more than once.

Because red-team blobs are content-addressed and restore writes them into the flat cache
layout ``redteam_acts_<model>_L<layer>/<key>.pt``, restoring several iterations into one cache
dir is safe: a key shared by two iterations is the same bytes from either archive.

Cost
----
Every blob is a forward through ``google/gemma-3-27b-it`` truncated to layers 0..32. On a box
that can hold the truncated model this is hours; on one that disk-offloads it is days. Every
blob is written through as soon as it is computed, so any stage is interruptible and picks up
where it stopped — and ``--dry-run`` reports the miss count per unit before you commit.

Typical use::

    # what would be computed and uploaded, unit by unit
    .venv_claude/bin/python scripts/publish_kaggle_hs_redteam_activations.py sync --dry-run

    # the unstable-box path: extract + upload each unit as it completes, resumable
    nohup .venv_claude/bin/python scripts/publish_kaggle_hs_redteam_activations.py sync \
        > logs/sync_hs_redteam_acts.log 2>&1 &

    # on another box, prime a retrain's cache with the base split + probe_iter3's data
    .venv_claude/bin/python scripts/publish_kaggle_hs_redteam_activations.py restore \
        --arm gptoss120b --iterations 3 \
        --cache-dir results_hs_gemma27b_batch_ablation/base_activations

Needs ``KAGGLE_CONFIG_DIR`` (the DIRECTORY holding kaggle.json) or ``KAGGLE_API_TOKEN`` for
``sync`` / ``publish`` / ``restore``; ``extract`` needs neither.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_redteam.kaggle_activations import (  # noqa: E402
    KaggleActivationError,
    _authenticate,
    _blob_header,
)
from agentic_redteam.retrain import (  # noqa: E402
    _base_activation_cache_paths,
    _redteam_activation_cache_path,
    stable_fraction_subsample,
    stable_train_test_split,
)

# The manifest that travels inside each uploaded archive. Bumped only if its shape changes.
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 2

# Progress cadence while computing misses — matches retrain._REDTEAM_PROGRESS_EVERY, i.e. a
# line every few minutes on an offloaded 27B, not a spinner.
PROGRESS_EVERY = 10


def _slugify(text: str) -> str:
    """Lowercase, non-alphanumeric runs collapsed to one hyphen — Kaggle's slug alphabet.

    Defined here rather than imported: this branch's ``kaggle_activations`` predates the
    helper, and the alternative (editing library code on an experiment branch to add a
    three-line string function) would be a worse trade.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class Arm:
    """One arm of the experiment9 high-stakes gemma-3-27b batch ablation."""

    name: str
    config: str  # relative to REPO_ROOT
    probe_dir: str  # relative to REPO_ROOT
    base_training_data: str  # relative to REPO_ROOT — what run_arm passed as --base-training-data

    @property
    def config_path(self) -> Path:
        return REPO_ROOT / self.config

    @property
    def probe_path(self) -> Path:
        return REPO_ROOT / self.probe_dir

    @property
    def base_data_path(self) -> Path:
        return REPO_ROOT / self.base_training_data


# Both arms of run_gemma27b_hs_attacker_ablation_batch.sh, in the order it ran them.
ARMS: dict[str, Arm] = {
    "gptoss120b": Arm(
        name="gptoss120b",
        config="configs/gptoss120b_hs_gemma27b_batch.md",
        probe_dir="probes/hs_gemma27b_gptoss120b_batch",
        base_training_data="data/hs_ls_200.jsonl",
    ),
    "deepseekv4pro": Arm(
        name="deepseekv4pro",
        config="configs/deepseekv4pro_hs_gemma27b_batch.md",
        probe_dir="probes/hs_gemma27b_deepseekv4pro_batch",
        base_training_data="data/hs_ls_200.jsonl",
    ),
}

# The shared cache dir the two arms wrote to on the cloud box (configs'
# output.base_activation_cache_dir). Both arms share it: the base key folds in only the base
# data + probe + split knobs (none of which differ between arms), and the per-conversation
# red-team keys are content-addressed against a frozen LLM.
DEFAULT_CACHE_DIR = REPO_ROOT / "results_hs_gemma27b_batch_ablation" / "base_activations"

DEFAULT_OWNER = "anku7890"

# --- the Kaggle namespace, and the guard that keeps it separate -----------------------------
# Every dataset this script owns lives under this prefix. See "Not clashing with the
# human-harm publication" at module level for why this is enforced rather than assumed.
SLUG_NAMESPACE = "hs-gemma27b-"
# Prefixes owned by OTHER runs on this account. Landing in one of these is never an accident
# worth completing silently; the error names the run it would have overwritten.
FOREIGN_SLUG_PREFIXES: dict[str, str] = {
    "hu-harm-gemma27b-": "the experiment11_cloud harmful_to_human gemma-3-27b ablation",
}

# {arm} is the slugified arm name, {iteration} the 1-based retrain index. Kaggle slugs cap at
# 50 characters; the longest these render to is 31.
DEFAULT_SLUG = SLUG_NAMESPACE + "{arm}-iter{iteration}"
DEFAULT_ARCHIVE = "hs-{arm}-iter{iteration}-redteam-acts.zip"
# The base split is one unit shared by both arms, so its templates take no {arm}. {key} is
# the base cache key, available for runs whose arms disagree on the base data.
DEFAULT_BASE_SLUG = SLUG_NAMESPACE + "base"
DEFAULT_BASE_ARCHIVE = "hs-base-acts.zip"

# Defaults iterative_retrain.py applies when the run script passes no override — which is
# what run_gemma27b_hs_attacker_ablation_batch.sh did. All four are in the base key.
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_BASE_FRACTION = 1.0


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
    """One arm resolved against the cache dir: its probe, its transforms, its conversations."""

    arm: Arm
    model_name: str
    layer: int
    combine_consecutive_messages: bool
    convert_tool_to_assistant: bool
    pos_class_label: str
    neg_class_label: str
    conversations: list[Conversation]

    @property
    def blob_subdir(self) -> str:
        """Directory the retrain path writes red-team blobs under, relative to the cache dir.

        Deliberately carries no iteration: the cache a retrain reads is flat and
        content-addressed, so a conversation shared by two iterations is one file. Only the
        *archives* are sliced per iteration. It carries no concept either — it is derived
        from the model and layer alone, which is why the human-harm run's blobs share this
        subdir name and why that is harmless (see the module docstring).
        """
        return f"redteam_acts_{self.model_name.replace('/', '_')}_L{self.layer}"

    @property
    def iterations(self) -> list[int]:
        """Every iteration this arm's conversations appear in, ascending."""
        return sorted({it for c in self.conversations for it in c.iterations})

    def conversations_for(self, iteration: int) -> list[Conversation]:
        """The FULL set iteration ``iteration`` retrained on — not a delta.

        See the module docstring on why the iterations cannot be reconstructed by unioning
        the ones before them.
        """
        return [c for c in self.conversations if iteration in c.iterations]


def _probe_metadata(arm: Arm) -> tuple[str, int, str, str]:
    """Read ``(model_name, layer, pos_class_label, neg_class_label)`` off the arm's probe.

    The probe pickle, not the config, is the source of truth for these (see the conventions
    in CLAUDE.md). Every probe in a run shares its model and layer, so the earliest one will
    do. Unpickled onto the CPU — nothing here touches a GPU.
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
    # earlier iteration is one cache entry, not three, which is exactly how the retrain path
    # treated it (a hit in iteration k+1 for a row first computed in iteration k).
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
# Units — one publishable, independently-resumable chunk of work
# --------------------------------------------------------------------------------------


@dataclass
class Unit:
    """A chunk of activations that is extracted, then uploaded, as one Kaggle dataset.

    Two flavours, distinguished by ``kind``. Both expose the same surface so ``sync`` can
    walk them uniformly: what files they own, which are missing, how to compute them, and
    what to write in the manifest.
    """

    kind: str  # "base" | "iteration"
    label: str  # human-readable, used in every log line
    slug: str
    archive_name: str
    title: str
    files: list[tuple[str, Path]]  # (name inside the archive, path in the cache dir)
    manifest: dict
    # iteration units only
    plan: ArmPlan | None = None
    iteration: int | None = None
    conversations: list[Conversation] = field(default_factory=list)
    # base units only
    base_spec: dict | None = None

    def missing(self) -> list[Path]:
        return [p for _, p in self.files if not p.exists()]

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for _, p in self.files if p.exists())


def _title_clip(text: str) -> str:
    """Kaggle titles must be 6-50 characters."""
    return text[:50]


def _base_unit(
    plans: list[ArmPlan],
    cache_dir: Path,
    args: argparse.Namespace,
) -> list[Unit]:
    """The base training split as one unit per distinct cache key.

    Normally that is exactly one unit: both arms train off the same base file with the same
    probe and split knobs, so ``_base_activation_cache_paths`` gives them the same key —
    which is why the two arms shared a cache dir on the cloud box in the first place.
    """
    units: dict[str, Unit] = {}
    for plan in plans:
        train_path, val_path = _base_activation_cache_paths(
            cache_dir,
            plan.arm.base_data_path,
            plan.model_name,
            plan.layer,
            args.seed,
            args.test_size,
            args.split_field,
            plan.combine_consecutive_messages,
            plan.convert_tool_to_assistant,
            args.base_data_fraction,
        )
        # stem is base_acts_<model>_L<layer>_<key>_train — the key is the last segment.
        key = train_path.stem.removesuffix("_train").rsplit("_", 1)[-1]
        if key in units:
            units[key].manifest["arms"].append(plan.arm.name)
            continue
        spec = {
            "base_training_data": plan.arm.base_training_data,
            "base_data_sha256": hashlib.sha256(
                plan.arm.base_data_path.read_bytes()
            ).hexdigest(),
            "model_name": plan.model_name,
            "layer": plan.layer,
            "seed": args.seed,
            "test_size": args.test_size,
            "split_field": args.split_field,
            "base_data_fraction": args.base_data_fraction,
            "combine_consecutive_messages": plan.combine_consecutive_messages,
            "convert_tool_to_assistant": plan.convert_tool_to_assistant,
            "pos_class_label": plan.pos_class_label,
            "neg_class_label": plan.neg_class_label,
        }
        units[key] = Unit(
            kind="base",
            label="base split",
            slug=_template(args, "base_slug", DEFAULT_BASE_SLUG).format(key=key),
            archive_name=_template(args, "base_archive", DEFAULT_BASE_ARCHIVE).format(
                key=key
            ),
            title=_title_clip("gemma27b L32 high-stakes base training acts"),
            # Archive names are the bare cache filenames, so restore drops them straight
            # into the cache dir root — where retrain._base_activation_cache_paths looks.
            files=[(train_path.name, train_path), (val_path.name, val_path)],
            manifest={
                "manifest_version": MANIFEST_VERSION,
                "kind": "base",
                "concept": f"{plan.pos_class_label}/{plan.neg_class_label}",
                "cache_key": key,
                "arms": [plan.arm.name],
                "files": [train_path.name, val_path.name],
                "source": (
                    "the base training split, rebuilt exactly as retrain_probe builds it "
                    "(load_from -> stable_fraction_subsample -> stable_train_test_split) "
                    "and keyed by retrain._base_activation_cache_paths"
                ),
                **spec,
            },
            base_spec=spec,
        )
    return list(units.values())


def _iteration_units(
    plans: list[ArmPlan], args: argparse.Namespace
) -> list[Unit]:
    """One self-contained unit per (arm, iteration)."""
    units: list[Unit] = []
    for plan in plans:
        for it in _resolve_iterations(plan, args.iterations):
            convs = plan.conversations_for(it)
            slug = _template(args, "dataset_slug", DEFAULT_SLUG).format(
                arm=_slugify(plan.arm.name), iteration=it
            )
            units.append(
                Unit(
                    kind="iteration",
                    label=f"{plan.arm.name} iter{it}",
                    slug=slug,
                    archive_name=_template(
                        args, "archive_name", DEFAULT_ARCHIVE
                    ).format(arm=_slugify(plan.arm.name), iteration=it),
                    title=_title_clip(
                        f"{plan.arm.name} gemma27b L32 hs redteam acts iter{it}"
                    ),
                    files=[(f"{plan.blob_subdir}/{c.key}.pt", c.path) for c in convs],
                    manifest=_iteration_manifest(plan, it, convs),
                    plan=plan,
                    iteration=it,
                    conversations=convs,
                )
            )
    return units


def _iteration_manifest(plan: ArmPlan, iteration: int, convs: list[Conversation]) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "kind": "iteration",
        "concept": f"{plan.pos_class_label}/{plan.neg_class_label}",
        "arm": plan.arm.name,
        "iteration": iteration,
        "config": plan.arm.config,
        "probe_dir": plan.arm.probe_dir,
        "model_name": plan.model_name,
        "layer": plan.layer,
        "combine_consecutive_messages": plan.combine_consecutive_messages,
        "convert_tool_to_assistant": plan.convert_tool_to_assistant,
        "pos_class_label": plan.pos_class_label,
        "neg_class_label": plan.neg_class_label,
        "blob_subdir": plan.blob_subdir,
        "n_conversations": len(convs),
        "self_contained": True,
        "source": (
            f"{plan.arm.probe_dir}/redteam_postprocessed_iter{iteration}.jsonl — the FULL "
            "set of conversations that retrain was trained on (not a delta against earlier "
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
            for c in convs
        ],
    }


def _template(args: argparse.Namespace, name: str, default: str) -> str:
    """A naming template, falling back to its default when the stage has no Kaggle flags.

    ``extract`` builds the same units as ``sync`` but never uploads, so it does not carry
    ``--dataset-slug`` and friends — yet units are named at construction time. Reading
    through here keeps one unit-construction path for every stage.
    """
    return getattr(args, name, None) or default


def _check_slug_namespace(units: list[Unit], args: argparse.Namespace) -> None:
    """Refuse to touch a Kaggle dataset outside this run's slug namespace.

    An owner's dataset slugs are one flat namespace, and ``dataset_create_version`` on the
    wrong slug replaces somebody else's blobs with these. The templates make that impossible
    by default; this makes it impossible after a ``--dataset-slug`` / ``--base-slug``
    override too, which is the only way it could realistically happen.
    """
    if getattr(args, "allow_foreign_slug", False):
        return
    problems: list[str] = []
    for unit in units:
        for prefix, owner in FOREIGN_SLUG_PREFIXES.items():
            if unit.slug.startswith(prefix):
                problems.append(
                    f"{unit.label}: slug {unit.slug!r} belongs to {owner}."
                )
                break
        else:
            if not unit.slug.startswith(SLUG_NAMESPACE):
                problems.append(
                    f"{unit.label}: slug {unit.slug!r} is outside this run's namespace "
                    f"{SLUG_NAMESPACE!r}."
                )
    if problems:
        raise SystemExit(
            "refusing to publish outside the high-stakes slug namespace:\n  "
            + "\n  ".join(problems)
            + "\npass --allow-foreign-slug if that is really what you want."
        )


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


def build_units(args: argparse.Namespace, cache_dir: Path) -> list[Unit]:
    """Every unit this invocation covers, in the order ``sync`` should process them.

    Base first: it is the smallest unit, and no retrain is a full cache hit without it — so
    it is the cheapest thing to get safely uploaded first.
    """
    plans = [
        build_plan(
            arm,
            cache_dir,
            combine_override=args.combine_consecutive_messages,
            convert_override=args.convert_tool_to_assistant,
            limit=getattr(args, "limit", None),
        )
        for arm in _selected_arms(args.arm)
    ]
    units: list[Unit] = []
    if getattr(args, "base", True):
        units.extend(_base_unit(plans, cache_dir, args))
    if not getattr(args, "base_only", False):
        units.extend(_iteration_units(plans, args))
    return units


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------


class ModelHost:
    """Lazily loads the extraction model and keeps it across units.

    Loading gemma-3-27b costs minutes, and ``sync`` touches it once per unit, so it is held
    for the whole run and released at the end (mirroring ``ProbeJudge.release`` — see "Free
    GPU memory between heavy phases" in CLAUDE.md). A unit that is fully cached never
    triggers a load at all.
    """

    def __init__(self) -> None:
        self._model = None
        self._spec: tuple[str, int] | None = None

    def get(self, model_name: str, layer: int):
        from agentic_redteam.model_loading import load_extraction_model

        if self._spec != (model_name, layer):
            self.release()
            self._model = load_extraction_model(model_name, layer)
            self._spec = (model_name, layer)
        return self._model

    def release(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._spec = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup must never mask the real error
            pass


def _save_conversation_blob(path: Path, acts, j: int, model_name: str, layer: int) -> None:
    """Write one conversation's activations in the layout ``LLMModel.load_activations`` reads.

    Byte-for-byte the dict ``retrain._activate_redteam_cached`` saves, including the
    ``model_name`` / ``layer`` provenance fields tuberlens itself drops on load but which
    the validation below checks before a blob is published or trusted.
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


def _extract_base(unit: Unit, host: ModelHost, verbose: bool = True) -> int:
    """Compute the base split's train/val blobs, rebuilding the split the way retrain does.

    Written through a ``.partial`` and renamed for the same reason the per-conversation
    blobs are: ``get_activations(save_path=...)`` writes in place, and a container killed
    mid-save would otherwise leave a truncated file that the cache loads without validating.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    spec = unit.base_spec
    assert spec is not None
    missing = [(name, p) for name, p in unit.files if not p.exists()]
    if not missing:
        return 0

    dataset = LabelledDataset.load_from(
        REPO_ROOT / spec["base_training_data"],
        pos_class_label=spec["pos_class_label"],
        neg_class_label=spec["neg_class_label"],
        combine_consecutive_messages=spec["combine_consecutive_messages"],
        convert_tool_to_assistant=spec["convert_tool_to_assistant"],
    )
    dataset = stable_fraction_subsample(dataset, spec["base_data_fraction"], spec["seed"])
    train, val = stable_train_test_split(
        dataset,
        test_size=spec["test_size"],
        split_field=spec["split_field"],
        seed=spec["seed"],
    )
    sides = {unit.files[0][1]: train, unit.files[1][1]: val}

    model = host.get(spec["model_name"], spec["layer"])
    done = 0
    for _name, path in missing:
        side = sides[path]
        if side is None or len(side) == 0:
            raise ValueError(f"base split produced an empty side for {path.name}")
        if verbose:
            print(f"  [base] computing {path.name} ({len(side)} samples) ...", flush=True)
        tmp = path.with_suffix(".pt.partial")
        model.get_activations(
            side.inputs, layer=spec["layer"], show_progress=verbose, save_path=str(tmp)
        )
        os.replace(tmp, path)
        done += len(side)
    return done


def _extract_conversations(
    convs: list[Conversation],
    plan: ArmPlan,
    host: ModelHost,
    verbose: bool = True,
) -> int:
    """Compute whatever of ``convs`` is not already on disk, writing each blob through."""
    from agentic_redteam.model_loading import extraction_batch_size

    todo = [c for c in convs if not c.path.exists()]
    if not todo:
        return 0
    chunk_size = extraction_batch_size()
    model = host.get(plan.model_name, plan.layer)
    started = time.monotonic()
    done = 0
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start : start + chunk_size]
        acts = model.get_activations(
            [c.messages for c in chunk], layer=plan.layer, show_progress=False
        )
        for j, conv in enumerate(chunk):
            _save_conversation_blob(conv.path, acts, j, plan.model_name, plan.layer)
        del acts
        done += len(chunk)
        if verbose and (done % PROGRESS_EVERY == 0 or done == len(todo)):
            elapsed = time.monotonic() - started
            rate = elapsed / done
            print(
                f"  [red-team activations] {done}/{len(todo)} "
                f"({rate:.1f}s/sample, ~{(len(todo) - done) * rate / 60:.0f} min left)",
                flush=True,
            )
    return done


def _extract_unit(unit: Unit, host: ModelHost, verbose: bool = True) -> int:
    if unit.kind == "base":
        return _extract_base(unit, host, verbose)
    assert unit.plan is not None
    return _extract_conversations(unit.conversations, unit.plan, host, verbose)


def cmd_extract(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir).resolve()
    units = build_units(args, cache_dir)

    print(f"cache dir: {cache_dir}\n")
    for unit in units:
        print(
            f"  {unit.label:<22} {len(unit.files):>5} file(s)  "
            f"{len(unit.files) - len(unit.missing()):>5} cached  "
            f"{len(unit.missing()):>5} to compute"
        )
    # Units overlap (iterations share conversations; both arms share the base), so the work
    # is the union of their missing files, not the sum.
    todo = {p for unit in units for p in unit.missing()}
    print(f"\n{len(todo)} unique file(s) to compute.")
    if not todo:
        print("Nothing to do — every blob is already on disk.")
        return 0
    if args.dry_run:
        print("--dry-run: nothing computed. Re-run without it to extract.")
        return 0

    host = ModelHost()
    try:
        for unit in units:
            if not unit.missing():
                continue
            print(f"\n>>> extracting {unit.label} ...", flush=True)
            _extract_unit(unit, host)
    finally:
        host.release()

    still = {p for unit in units for p in unit.missing()}
    print(f"\nDone. Still missing: {len(still)}.")
    return 0 if not still else 1


# --------------------------------------------------------------------------------------
# Validation + archive
# --------------------------------------------------------------------------------------


def _validate_blob(unit: Unit, path: Path, expected_rows: int | None) -> None:
    """Raise unless the blob really is activations of this unit's probe.

    Same contract as ``kaggle_activations._validate_blob``: a blob is checked against the
    probe it claims to belong to *before* it is published, so a mismatch surfaces here
    rather than hours into a remote retrain. ``expected_rows`` is 1 for a per-conversation
    blob and ``None`` for the base sides (whose row count depends on the split and is
    checked implicitly by the key).
    """
    model_name = unit.manifest["model_name"]
    layer = unit.manifest["layer"]
    try:
        data = _blob_header(path)
    except Exception as e:  # noqa: BLE001 — reported as a validation failure
        raise KaggleActivationError(f"{unit.label}/{path.name}: unreadable: {e}") from e

    missing = {"activations", "attention_mask", "input_ids"} - set(data)
    if missing:
        raise KaggleActivationError(
            f"{unit.label}/{path.name}: missing tensor field(s) {sorted(missing)}."
        )
    problems = []
    if data.get("model_name") not in (None, model_name):
        problems.append(f"model_name={data['model_name']!r} (expected {model_name!r})")
    if data.get("layer") is not None and int(data["layer"]) != int(layer):
        problems.append(f"layer={data['layer']} (expected {layer})")
    rows = int(data["activations"].shape[0])
    if expected_rows is not None and rows != expected_rows:
        problems.append(f"{rows} rows (expected {expected_rows})")
    if rows == 0:
        problems.append("0 rows")
    if problems:
        raise KaggleActivationError(f"{unit.label}/{path.name}: " + "; ".join(problems))


def _validate_unit(unit: Unit, already: set[Path] | None = None) -> list[str]:
    """Validate every file of a unit; returns problem strings (empty = clean)."""
    problems: list[str] = []
    if unit.missing():
        return [
            f"{unit.label}: {len(unit.missing())} of {len(unit.files)} file(s) not computed "
            "yet — run `extract` (or `sync`) first."
        ]
    expected_rows = 1 if unit.kind == "iteration" else None
    for _name, path in unit.files:
        if already is not None:
            if path in already:
                continue
            already.add(path)
        try:
            _validate_blob(unit, path, expected_rows)
        except KaggleActivationError as e:
            problems.append(str(e))
    return problems


def _build_archive(unit: Unit, dest: Path, verbose: bool = True) -> Path:
    """Zip a unit's blobs + manifest into ``dest``.

    ``ZIP_STORED``: the payload is fp16 activation tensors, which do not compress, so
    deflating would cost CPU-minutes per GB for ~nothing. The archive's internal layout is
    the cache's own, so restoring is an extract-in-place into any cache dir.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(unit.manifest, indent=2))
        for i, (arcname, path) in enumerate(unit.files, 1):
            zf.write(path, arcname=arcname)
            if verbose and (i % 200 == 0 or i == len(unit.files)):
                print(f"    packed {i}/{len(unit.files)}", flush=True)
    return dest


# --------------------------------------------------------------------------------------
# Kaggle I/O
# --------------------------------------------------------------------------------------


def _remote_exists(api, handle: str) -> bool:
    """Whether a dataset already exists — the resume signal on a box that lost its disk.

    Any failure to answer is reported as "does not exist": the caller then tries to create
    it, and a create against an existing dataset fails loudly with a message naming the
    conflict, which is a far better outcome than silently skipping a unit that was never
    actually uploaded.
    """
    try:
        api.dataset_status(handle)
        return True
    except Exception:  # noqa: BLE001 — 404s arrive in several shapes across kaggle versions
        return False


def _already_exists_error(exc: Exception) -> bool:
    """Whether a create failure means "someone already uploaded this"."""
    text = str(exc).lower()
    return "already exists" in text or "already in use" in text or "409" in text


def _upload_unit(api, unit: Unit, args: argparse.Namespace, cache_dir: Path) -> None:
    """Pack and upload one unit. Raises on failure."""
    handle = f"{args.owner}/{unit.slug}"
    # Stage inside the cache dir so the archive lands on the same filesystem as the blobs it
    # is built from, and is cleaned up even if the upload raises.
    with tempfile.TemporaryDirectory(
        dir=str(cache_dir), prefix=f".publish_{_slugify(unit.label)}_"
    ) as tmp:
        staging = Path(tmp)
        print(f">>> packing {unit.label} ...", flush=True)
        _build_archive(unit, staging / unit.archive_name)
        (staging / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": unit.title,
                    "id": handle,
                    "licenses": [{"name": "CC0-1.0"}],
                },
                indent=2,
            )
        )
        print(f">>> uploading {unit.label} -> {handle} ...", flush=True)
        notes = f"{unit.label} ({len(unit.files)} file(s))"
        if args.new_version:
            api.dataset_create_version(
                str(staging), version_notes=notes, dir_mode="skip", convert_to_csv=False
            )
        else:
            api.dataset_create_new(
                str(staging), public=args.public, dir_mode="skip", convert_to_csv=False
            )
    print(f"    done: kaggle.com/datasets/{handle}")


def _locate_archive(staging: Path, archive_name: str) -> Path:
    """Find our archive in whatever Kaggle actually served.

    ``dataset_download_file`` names its output from the download URL, not from the requested
    name, and Kaggle may wrap a large file in an extra zip — so the archive has to be
    discovered rather than assumed.
    """
    direct = staging / archive_name
    if direct.is_file():
        return direct
    for z in sorted(p for p in staging.rglob("*.zip") if p.is_file()):
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            if MANIFEST_NAME in names:
                return z
            if len(names) == 1 and names[0].endswith(".zip"):
                zf.extractall(staging / "_unwrapped")
                inner = sorted((staging / "_unwrapped").rglob("*.zip"))
                if inner:
                    return inner[0]
    landed = sorted(p.name for p in staging.rglob("*") if p.is_file())
    raise KaggleActivationError(
        f"no archive containing {MANIFEST_NAME} in the download (got {landed or 'nothing'})."
    )


def _download_unit(api, unit: Unit, args: argparse.Namespace, cache_dir: Path) -> int:
    """Pull one unit's archive into the cache dir. Returns the number of blobs written.

    The archive is validated against **this repo's** expectations — the file names a unit
    derives from the probes and dumps — not against the archive's own manifest. A mismatch
    means the blobs are not the ones that unit's retrain used, which is exactly the silent
    staleness the path-keyed caches cannot catch on their own. It is also what catches a
    cross-concept restore: a human-harm archive pulled into this cache dir has none of the
    keys this unit expects, so it is rejected rather than half-applied.
    """
    handle = f"{args.owner}/{unit.slug}"
    expected = {arcname: path for arcname, path in unit.files}
    with tempfile.TemporaryDirectory(
        dir=str(cache_dir), prefix=f".restore_{_slugify(unit.label)}_"
    ) as tmp:
        staging = Path(tmp)
        try:
            api.dataset_download_file(
                handle, unit.archive_name, path=str(staging), quiet=False
            )
        except Exception as e:  # noqa: BLE001 — retried as a whole-dataset pull below
            # Kaggle EXPANDS an uploaded .zip server-side: the dataset's remote files are the
            # archive's members at their arcnames, and the archive name we uploaded under is
            # not addressable at all (404). Pulling the whole dataset gives one zip back in
            # exactly that layout — manifest included — which _locate_archive then accepts.
            print(
                f"    {unit.archive_name} is not a remote file ({str(e)[:80]}); "
                "pulling the whole dataset instead.",
                flush=True,
            )
            api.dataset_download_files(handle, path=str(staging), quiet=False, unzip=False)
        archive = _locate_archive(staging, unit.archive_name)
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if n != MANIFEST_NAME]
            unexpected = set(names) - set(expected)
            absent = set(expected) - set(names)
            if unexpected or absent:
                raise KaggleActivationError(
                    f"{unit.label}: archive does not match this repo "
                    f"({len(absent)} expected file(s) absent, {len(unexpected)} unexpected)."
                )
            written = 0
            for name in names:
                dest = expected[name]
                if dest.exists() and not getattr(args, "force", False):
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
    return written


# --------------------------------------------------------------------------------------
# publish / restore / sync
# --------------------------------------------------------------------------------------


def _print_plan(units: list[Unit], args: argparse.Namespace) -> None:
    print(f"{'PUBLIC' if args.public else 'private'} dataset(s) owned by {args.owner}:\n")
    for unit in units:
        print(
            f"  {unit.label:<22} {len(unit.files):>5} file(s)  "
            f"{unit.size_bytes() / 1e9:>6.2f} GB"
            + ("" if not unit.missing() else f"   ({len(unit.missing())} NOT YET COMPUTED)")
        )
        print(
            f"  {'':<22} -> kaggle.com/datasets/{args.owner}/{unit.slug}  "
            f"({unit.archive_name})"
        )
    unique = {p for unit in units for _, p in unit.files}
    unique_bytes = sum(p.stat().st_size for p in unique if p.exists())
    print(
        f"\n  {len(units)} dataset(s), {sum(u.size_bytes() for u in units) / 1e9:.2f} GB "
        f"uploaded ({unique_bytes / 1e9:.2f} GB unique — units overlap)\n"
    )


def cmd_publish(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir).resolve()
    units = build_units(args, cache_dir)
    _check_slug_namespace(units, args)

    # Validate everything BEFORE uploading anything. Blobs are checked once per unique path,
    # not once per unit: iterations share conversations, and re-reading a header buys nothing.
    problems: list[str] = []
    already: set[Path] = set()
    for unit in units:
        problems.extend(_validate_unit(unit, already))
    if problems:
        for p in problems[:20]:
            print(f"INVALID  {p}", file=sys.stderr)
        if len(problems) > 20:
            print(f"... and {len(problems) - 20} more", file=sys.stderr)
        print(f"\n{len(problems)} problem(s) — nothing was uploaded.", file=sys.stderr)
        return 1

    _print_plan(units, args)
    if args.dry_run:
        print("--dry-run: nothing uploaded. Re-run without it to publish.")
        return 0

    api = _authenticate()
    failures = 0
    for unit in units:
        try:
            _upload_unit(api, unit, args, cache_dir)
        except Exception as e:  # noqa: BLE001 — report and continue to the next unit
            if _already_exists_error(e):
                print(f"    exists already, skipped: {args.owner}/{unit.slug}")
                continue
            failures += 1
            print(f"    FAILED {unit.label}: {e}", file=sys.stderr)
    if failures:
        print(f"\n{failures} of {len(units)} upload(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(units)} dataset(s) published.")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    units = build_units(args, cache_dir)
    _check_slug_namespace(units, args)

    api = _authenticate()
    failures = 0
    for unit in units:
        if not unit.missing() and not args.force:
            print(f">>> {unit.label}: already complete on disk, skipped.")
            continue
        print(f">>> {unit.label}: downloading {args.owner}/{unit.slug} ...", flush=True)
        try:
            written = _download_unit(api, unit, args, cache_dir)
        except Exception as e:  # noqa: BLE001 — report and continue to the next unit
            failures += 1
            print(f"    FAILED {unit.label}: {e}", file=sys.stderr)
            continue
        print(
            f"    restored {written} file(s) "
            f"({len(unit.files) - written} already present) into {cache_dir}"
        )
    if failures:
        print(f"\n{failures} of {len(units)} restore(s) failed.", file=sys.stderr)
        return 1
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Extract and upload unit by unit, so a box that dies loses at most one unit's work.

    The loop is deliberately "publish before starting the next unit" rather than
    "extract everything, then publish": on a container that can vanish mid-run, the cost of
    an interruption is bounded by the current unit instead of the whole job.

    Resume comes from Kaggle, not from local state, because the failure mode being defended
    against takes the disk with it. A unit whose dataset already exists is skipped, and its
    blobs are pulled back down first (unless ``--no-restore-published``) so later units get
    cache hits — iteration 2 shares most of its conversations with iteration 1, which is
    hours of forwards on a 27B that a download makes free.
    """
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    units = build_units(args, cache_dir)
    _check_slug_namespace(units, args)

    print(f"cache dir: {cache_dir}")
    _print_plan(units, args)
    print("Order: base first (smallest, and no retrain is a full cache hit without it),")
    print("then each (arm, iteration). Each unit is uploaded before the next one starts.\n")
    if args.dry_run:
        print("--dry-run: nothing computed or uploaded.")
        return 0

    api = _authenticate()
    host = ModelHost()
    failures = 0
    try:
        for i, unit in enumerate(units, 1):
            head = f"[{i}/{len(units)}] {unit.label}"
            if _remote_exists(api, f"{args.owner}/{unit.slug}"):
                print(f"\n>>> {head}: already published, skipping upload.", flush=True)
                if args.restore_published and unit.missing():
                    try:
                        written = _download_unit(api, unit, args, cache_dir)
                        print(f"    pulled {written} blob(s) back into the cache.")
                    except Exception as e:  # noqa: BLE001 — a failed pull only costs forwards
                        print(
                            f"    could not restore {unit.label} ({e}); later units may "
                            "recompute its blobs.",
                            file=sys.stderr,
                        )
                continue

            print(f"\n>>> {head}: {len(unit.missing())} of {len(unit.files)} file(s) to "
                  f"compute.", flush=True)
            try:
                _extract_unit(unit, host)
            except Exception as e:  # noqa: BLE001 — one unit's failure must not sink the rest
                failures += 1
                print(f"    FAILED extracting {unit.label}: {e}", file=sys.stderr)
                continue

            problems = _validate_unit(unit)
            if problems:
                failures += 1
                for p in problems[:5]:
                    print(f"    INVALID {p}", file=sys.stderr)
                continue

            # Release the model before a multi-GB pack+upload: the upload is I/O-bound and
            # can take many minutes, and on a tight box those are minutes a 27B's offloaded
            # shards need not stay resident for.
            if args.release_between_units:
                host.release()
            try:
                _upload_unit(api, unit, args, cache_dir)
            except Exception as e:  # noqa: BLE001
                if _already_exists_error(e):
                    print(f"    exists already, skipped: {args.owner}/{unit.slug}")
                    continue
                failures += 1
                print(f"    FAILED uploading {unit.label}: {e}", file=sys.stderr)
                print(
                    "    its blobs are on disk, so a re-run resumes at this unit.",
                    file=sys.stderr,
                )
    finally:
        host.release()

    if failures:
        print(f"\n{failures} unit(s) failed — re-run to resume.", file=sys.stderr)
        return 1
    print(f"\nAll {len(units)} unit(s) extracted and published.")
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
        "--iterations",
        nargs="*",
        type=int,
        default=None,
        help="Retrain iterations to act on (default: every one with a postprocessed dump). "
        "Each is a SELF-CONTAINED dataset — the iterations do not nest, so restoring "
        "iteration N needs only iteration N.",
    )
    ap.add_argument(
        "--base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the base training split as a unit (default: yes).",
    )
    ap.add_argument(
        "--base-only",
        action="store_true",
        help="Act on the base unit alone, skipping every iteration unit. Use to retry the "
        "base upload without re-packing iteration archives that are already published.",
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
        help="Override the config's eval.combine_consecutive_messages (in both cache keys). "
        "Default: read from the config.",
    )
    ap.add_argument(
        "--convert-tool-to-assistant",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the config's eval.convert_tool_to_assistant (in both cache keys).",
    )
    # These four are in the BASE cache key only. The defaults are what
    # run_gemma27b_hs_attacker_ablation_batch.sh left iterative_retrain.py at.
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Split seed (base key)")
    ap.add_argument(
        "--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Val fraction (base key)"
    )
    ap.add_argument(
        "--split-field", default=None, help="Group-by field for the split (base key)"
    )
    ap.add_argument(
        "--base-data-fraction",
        type=float,
        default=DEFAULT_BASE_FRACTION,
        help="Fraction of the base data ingested (base key)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print the plan; change nothing")


def _add_kaggle(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--owner", default=DEFAULT_OWNER, help="Kaggle username owning the datasets")
    ap.add_argument(
        "--dataset-slug",
        default=DEFAULT_SLUG,
        help="Iteration slug template; {arm} and {iteration}. Default: " + DEFAULT_SLUG,
    )
    ap.add_argument(
        "--archive-name",
        default=DEFAULT_ARCHIVE,
        help="Iteration archive name inside the dataset. Default: " + DEFAULT_ARCHIVE,
    )
    ap.add_argument(
        "--base-slug",
        default=DEFAULT_BASE_SLUG,
        help="Base-unit slug template; {key} is the base cache key. Default: "
        + DEFAULT_BASE_SLUG,
    )
    ap.add_argument(
        "--base-archive",
        default=DEFAULT_BASE_ARCHIVE,
        help="Base-unit archive name. Default: " + DEFAULT_BASE_ARCHIVE,
    )
    ap.add_argument(
        "--allow-foreign-slug",
        action="store_true",
        help=f"Permit slugs outside {SLUG_NAMESPACE!r}. Off by default so an overridden "
        "template cannot push high-stakes blobs over another run's datasets "
        f"(reserved: {', '.join(sorted(FOREIGN_SLUG_PREFIXES))}).",
    )
    ap.add_argument("--public", action="store_true", help="Publish public (default: private)")
    ap.add_argument(
        "--new-version",
        action="store_true",
        help="Push a new version of an existing dataset instead of creating one",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="stage", required=True)

    p_sync = sub.add_parser(
        "sync", help="Extract AND upload unit by unit, resumable (use on an unstable box)"
    )
    _add_common(p_sync)
    _add_kaggle(p_sync)
    p_sync.add_argument(
        "--restore-published",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On resume, pull an already-published unit's blobs back into the cache so "
        "later overlapping units don't recompute them (default: yes).",
    )
    p_sync.add_argument(
        "--release-between-units",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the extraction model before each pack+upload (default: yes). Costs a "
        "reload per unit; pass --no-release-between-units on a box with RAM to spare.",
    )
    p_sync.add_argument("--force", action="store_true", help="Overwrite blobs already on disk")
    p_sync.set_defaults(func=cmd_sync)

    p_ex = sub.add_parser("extract", help="Compute missing blobs only; upload nothing")
    _add_common(p_ex)
    p_ex.add_argument(
        "--limit", type=int, default=None, help="Only the first N conversations per arm (smoke test)"
    )
    p_ex.set_defaults(func=cmd_extract)

    p_pub = sub.add_parser("publish", help="Upload already-extracted units to Kaggle")
    _add_common(p_pub)
    _add_kaggle(p_pub)
    p_pub.set_defaults(func=cmd_publish)

    p_res = sub.add_parser("restore", help="Download units back into a cache dir")
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
