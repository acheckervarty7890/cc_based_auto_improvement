"""Shared machinery for attributing eval AUROC to individual red-team conversations.

HIGH-STAKES port of ``experiment11_cloud``'s ``scripts/attribution_lib.py``. The logic is
that file's; only the constants below name experiment9's concept — base data
``data/hs_ls_200.jsonl``, the four ``eval_datasets/`` splits, the
``results_hs_gemma27b_batch_ablation/`` cache and the two ``hs_gemma27b_*_batch`` arms.
The probe (gemma-3-27b-it L32), the transforms, the seed and the val fraction are the
same, because ``run_gemma27b_hs_attacker_ablation_batch.sh`` left them at the same
values ``run_gemma27b_hu_harm_attacker_ablation_batch.sh`` did.

Everything here runs off activations that are already on disk — the four eval split
blobs, the base train/val blobs, and the per-conversation red-team blobs. No forward
pass through gemma-3-27b is ever needed, so a full attribution costs minutes rather
than the days the extraction itself took.

Three things in here are load-bearing and were each established by measurement on the
human-harm run, not assumption:

**The unit of attribution is the pair, not the row.** ``preprocessing`` turns each
red-team success into two training rows: the success itself and an LLM-generated
opposite-class counterpart. Dropping one without the other is a different (and
label-unbalancing) intervention, so ``build_pairs`` rejoins them by re-deriving
``preprocessing._cache_key`` against the arm's ``contrastive_cache.jsonl``.

**Scoring must go through the probe's own forward path, or match it in fp32
deliberately.** The probe is stored in bf16, whose ULP at these logit magnitudes is
~0.1, and a hand-rolled matmul differing by 1-2 ULP moved a human-harm split's AUROC
by ~0.006. Callers must not mix the two scales within one comparison.

**The pipeline's AUROC is computed on saturated probabilities.** ``get_performances``
scores ``predict_proba``, i.e. ``sigmoid(s)`` in bf16, where anything above logit ~5.5
rounds to exactly 1.0. Those rows are mutually tied and sklearn awards 0.5 per tied
pair, so the published AUROC is *not* the rank AUROC of the logits. Hence
:func:`auroc_pipeline` (replicates the CSV) and :func:`auroc_rank` (rank-faithful), and
every report carries both.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# --- the two arms of run_gemma27b_hs_attacker_ablation_batch.sh -------------------
BASE_TRAINING_DATA = REPO / "data/hs_ls_200.jsonl"
EVAL_DATASET_DIR = REPO / "eval_datasets"
SHARED_CACHE = REPO / "results_hs_gemma27b_batch_ablation/base_activations"


def _resolve_eval_activations_dir() -> Path:
    """Where the four ``<split>-acts_full.pt`` eval blobs live.

    The blobs are multi-GB and never committed, so a fresh checkout has none until
    ``attribution_fetch_eval.py`` pulls them from Kaggle into the dir both arms'
    configs name. ``AGENTIC_ATTRIB_EVAL_DIR`` overrides.
    """
    override = os.environ.get("AGENTIC_ATTRIB_EVAL_DIR")
    if override:
        return Path(override)
    return REPO / "results_hs_gemma27b_batch_ablation/eval_activations"


EVAL_ACTIVATIONS_DIR = _resolve_eval_activations_dir()

# Both arms' configs: eval.combine_consecutive_messages / convert_tool_to_assistant are
# both true, the CLI defaults gave seed 42 and a 0.2 val split.
COMBINE_CONSECUTIVE_MESSAGES = True
CONVERT_TOOL_TO_ASSISTANT = True
SEED = 42
TEST_SIZE = 0.2

PROBE_MODEL = "google/gemma-3-27b-it"
PROBE_LAYER = 32

# The activation window ``get_activations`` truncates at (tuberlens/model.py:433). A
# conversation longer than this loses its tail, which for a generated contrastive row
# is often the part carrying the label — see ``token_length``.
ACTIVATION_MAX_TOKENS = 1024

ARMS = {
    "gptoss120b": REPO / "probes/hs_gemma27b_gptoss120b_batch",
    "deepseekv4pro": REPO / "probes/hs_gemma27b_deepseekv4pro_batch",
}
EVAL_SPLITS = ["anthropic", "mt", "mts", "toolace"]


def canon(messages) -> str:
    """Canonical text of a conversation — the join key used everywhere here.

    Matches the basis ``retrain._redteam_activation_cache_path`` and
    ``retrain._split_unit_interval`` hash, so a row's cache blob, its train/val side
    and its pair membership are all keyed off the same string.
    """
    return json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
    )


def apply_transforms(dialogue):
    """The config's two loader transforms, in ``LabelledDataset.load_from`` order."""
    from tuberlens.interfaces.dataset import LabelledDataset

    if CONVERT_TOOL_TO_ASSISTANT:
        dialogue = LabelledDataset._convert_tool_to_assistant(dialogue)
    if COMBINE_CONSECUTIVE_MESSAGES:
        dialogue = LabelledDataset._combine_consecutive_messages(dialogue)
    return dialogue


def load_probe(path: Path):
    import pickle

    with Path(path).open("rb") as f:
        return pickle.load(f)


# --- dataset reconstruction -------------------------------------------------------


def load_redteam_dataset(arm: str, iteration: int = 3):
    """The exact red-team half of ``probe_iter{iteration}``'s training set.

    ``redteam_postprocessed_iter{N}.jsonl`` is dumped by ``retrain_probe`` *after*
    preprocessing and message transforms and *before* the train/val split, so it is
    the postprocessed set verbatim — no filtering or contrastive generation has to be
    replayed (and no LLM calls are made).
    """
    from tuberlens.interfaces.dataset import LabelledDataset, Message

    path = ARMS[arm] / f"redteam_postprocessed_iter{iteration}.jsonl"
    inputs, ids, labels = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            inputs.append(
                [Message(role=m["role"], content=m["content"]) for m in row["inputs"]]
            )
            ids.append(row["id"])
            labels.append(row["label"])
    return LabelledDataset(inputs=inputs, ids=ids, other_fields={"labels": labels})


def load_base_dataset(probe):
    from tuberlens.interfaces.dataset import LabelledDataset

    return LabelledDataset.load_from(
        BASE_TRAINING_DATA,
        pos_class_label=probe.pos_class_label,
        neg_class_label=probe.neg_class_label,
        combine_consecutive_messages=COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=CONVERT_TOOL_TO_ASSISTANT,
    )


def split_sides(dataset):
    """``(train, val)`` under the run's content-deterministic split."""
    from agentic_redteam.retrain import stable_train_test_split

    return stable_train_test_split(
        dataset, test_size=TEST_SIZE, split_field=None, seed=SEED
    )


def is_val(messages) -> bool:
    """Which side of the split one conversation lands on, without materialising it."""
    from agentic_redteam.retrain import _split_unit_interval

    return _split_unit_interval(messages, None, SEED) < TEST_SIZE


# --- pairing ----------------------------------------------------------------------


def generated_to_source(arm: str) -> dict[str, str]:
    """Map each generated row's canonical text to the success it was written from.

    The contrastive cache is the authoritative link: ``generate_contrastive_dataset``
    returns sources and generated rows in different orders, so position cannot be
    trusted. Both sides get the config's message transforms applied here, so the keys
    match the dumps.
    """
    from tuberlens.interfaces.dataset import Message

    def as_key(raw):
        return canon(
            apply_transforms(
                [Message(role=m["role"], content=m["content"]) for m in raw]
            )
        )

    out: dict[str, str] = {}
    with (ARMS[arm] / "contrastive_cache.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line).get("record", {})
            if not rec.get("inputs") or not rec.get("original_messages"):
                continue
            out[as_key(rec["inputs"])] = as_key(rec["original_messages"])
    return out


@dataclass
class Pair:
    """One red-team success and its generated opposite-class counterpart."""

    pair_id: int
    source_key: str
    source_idx: int | None = None
    generated_idx: int | None = None
    source_label: str = ""
    generated_label: str = ""
    train_rows: list[int] = field(default_factory=list)
    val_rows: list[int] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.train_rows) + len(self.val_rows)


def build_pairs(arm: str, redteam_dataset) -> tuple[list[Pair], dict]:
    """Rejoin every generated row to the success it was written from."""
    gen2src = generated_to_source(arm)
    labels = redteam_dataset.other_fields["labels"]
    keys = [canon(m) for m in redteam_dataset.inputs]

    pairs: dict[str, Pair] = {}
    stats = {"n_rows": len(keys), "n_generated": 0, "n_source": 0, "n_orphan": 0}

    def get(source_key: str) -> Pair:
        if source_key not in pairs:
            pairs[source_key] = Pair(pair_id=len(pairs), source_key=source_key)
        return pairs[source_key]

    for idx, key in enumerate(keys):
        if key in gen2src:
            pair = get(gen2src[key])
            pair.generated_idx = idx
            pair.generated_label = labels[idx]
            stats["n_generated"] += 1
        else:
            pair = get(key)
            pair.source_idx = idx
            pair.source_label = labels[idx]
            stats["n_source"] += 1

    for pair in pairs.values():
        for idx in (pair.source_idx, pair.generated_idx):
            if idx is None:
                continue
            side = pair.val_rows if is_val(redteam_dataset.inputs[idx]) else pair.train_rows
            side.append(idx)
        if pair.source_idx is None or pair.generated_idx is None:
            stats["n_orphan"] += 1

    ordered = sorted(pairs.values(), key=lambda p: p.pair_id)
    stats["n_pairs"] = len(ordered)
    stats["n_straddling"] = sum(1 for p in ordered if p.train_rows and p.val_rows)
    stats["n_fully_val"] = sum(1 for p in ordered if not p.train_rows)
    return ordered, stats


# --- activations ------------------------------------------------------------------


def redteam_blob_path(messages) -> Path:
    from agentic_redteam.retrain import _redteam_activation_cache_path

    return _redteam_activation_cache_path(
        SHARED_CACHE,
        messages,
        PROBE_MODEL,
        PROBE_LAYER,
        COMBINE_CONSECUTIVE_MESSAGES,
        CONVERT_TOOL_TO_ASSISTANT,
    )


def base_blob_paths() -> tuple[Path, Path]:
    from agentic_redteam.retrain import _base_activation_cache_paths

    return _base_activation_cache_paths(
        SHARED_CACHE,
        BASE_TRAINING_DATA,
        PROBE_MODEL,
        PROBE_LAYER,
        SEED,
        TEST_SIZE,
        None,
        COMBINE_CONSECUTIVE_MESSAGES,
        CONVERT_TOOL_TO_ASSISTANT,
        1.0,
    )


def blob_width(path: Path) -> int:
    """Token length of a cached conversation, read from the blob header alone.

    ``_activate_redteam_cached`` writes each conversation at its **own** length (chunk
    size 1), and ``get_activations`` truncates at ``ACTIVATION_MAX_TOKENS``. So the
    stored width is ``min(true_tokens, 1024)`` and a width of exactly 1024 means the
    conversation reached — and almost certainly overran — the activation window.

    This is the tokenization the extraction actually used, so it needs no tokenizer and
    cannot disagree with the model. ``mmap=True`` reads the header without paging the
    tensor in.
    """
    blob = torch.load(path, weights_only=False, mmap=True)
    return int(blob["activations"].shape[1])


# --- metrics ----------------------------------------------------------------------


def auroc_pipeline(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC exactly as the pipeline reports it: bf16 ``sigmoid`` then sklearn."""
    from sklearn.metrics import roc_auc_score

    p = torch.sigmoid(torch.from_numpy(np.asarray(s, dtype=np.float32))).to(
        torch.bfloat16
    )
    return float(roc_auc_score(y, p.float().numpy()))


def auroc_rank(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-faithful AUROC on the fp32 logits — no saturation, no induced ties."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def auroc_both(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    return {"pipeline": auroc_pipeline(y, s), "rank": auroc_rank(y, s)}


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
