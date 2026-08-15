"""Shared machinery for attributing eval AUROC to individual red-team conversations.

Ported from ``experiment11_cloud`` (the ``harmful_to_human`` gemma-3-27b batch ablation)
onto **this branch's instruction-following experiment**: the two arms of
``run_gemma27b_instructions_attackers.sh`` (attacker ``openai/gpt-oss-120b`` vs
``nvidia/nemotron-3-ultra-550b-a55b``), a ``google/gemma-3-27b-it`` L32 probe, and the
seven class-balanced ``eval_instructions`` splits. Only the constants below and the
token-length filter at the end are new; the mechanics are the originals.

The question this supports: *how much of the final probe's eval AUROC was already bought
by the red-team data that existed at iteration 1?*

Everything here runs off activations that are already on disk — the seven eval split
blobs, the base train/val blobs, and the per-conversation red-team blobs. On this branch
none of those were ever synced back from the cloud box, so they are pulled from Kaggle:
``scripts/publish_kaggle_redteam_activations.py restore --experiment instructions`` for
the base + red-team caches and ``scripts/attribution_fetch_eval.py`` for the eval splits.
No forward pass through gemma-3-27b is ever needed afterwards.

Three things in here are load-bearing and were each established by measurement, not
assumption:

**The unit of attribution is the pair, not the row.** ``preprocessing`` turns each
red-team success into two training rows: the success itself and an LLM-generated
opposite-class counterpart. Dropping one without the other is a different (and
label-unbalancing) intervention, so ``build_pairs`` rejoins them by re-deriving
``preprocessing._cache_key`` against the arm's ``contrastive_cache.jsonl``.

**Scoring must go through the probe's own forward path, or match it in fp32
deliberately.** The probe is stored in bf16, whose ULP at these logit magnitudes is
~0.1. A hand-rolled matmul that differs by 1-2 ULP moves a split's AUROC by ~0.006.
``sequence_logits`` therefore computes in fp32 (self-consistent, and the more accurate
evaluation of the same weights) and callers must not mix the two scales within one
comparison.

**The pipeline's AUROC is computed on saturated probabilities.** ``get_performances``
scores ``predict_proba``, i.e. ``sigmoid(s)`` in bf16, where anything above logit ~5.5
rounds to exactly 1.0. Those rows are mutually tied and sklearn awards 0.5 per tied
pair, so the published AUROC is *not* the rank AUROC of the logits. Any perturbation
that only reshuffles the saturated block is invisible to the published metric. Hence
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

# --- the two arms of run_gemma27b_instructions_attackers.sh ------------------------
# Both arm configs are byte-identical apart from attacker.models (the run script asserts
# it at launch), so every constant below is read off either one.
BASE_TRAINING_DATA = REPO / "data/instructions_llama70b_50.jsonl"
EVAL_DATASET_DIR = REPO / "eval_instructions"
SHARED_CACHE = REPO / "results_instructions_gemma27b_shared/base_activations"


def _resolve_eval_activations_dir() -> Path:
    """Where the seven ``<split>-acts_full.pt`` eval blobs live.

    The configs name ``results_instructions_gemma27b_shared/eval_activations``, and that
    is where ``attribution_fetch_eval.py`` lands the Kaggle copies. Activation blobs are
    multi-GB and never committed, so a fresh checkout has none until something downloads
    them. ``AGENTIC_ATTRIB_EVAL_DIR`` overrides.
    """
    override = os.environ.get("AGENTIC_ATTRIB_EVAL_DIR")
    if override:
        return Path(override)
    return REPO / "results_instructions_gemma27b_shared/eval_activations"


EVAL_ACTIVATIONS_DIR = _resolve_eval_activations_dir()

# Both arms' configs: `eval:` sets both transforms true, and the run script left seed and
# test_size at the CLI defaults.
COMBINE_CONSECUTIVE_MESSAGES = True
CONVERT_TOOL_TO_ASSISTANT = True
SEED = 42
TEST_SIZE = 0.2

# tuberlens' get_activations pads *or truncates* to this width (tuberlens/model.py), so a
# conversation longer than it was scored — and trained on — from its opening only.
MAX_ACTIVATION_TOKENS = 1024

ARMS = {
    "gptoss120b": REPO / "probes/instructions_gemma27b_gptoss",
    "nemotron": REPO / "probes/instructions_gemma27b_nemotron",
}
# Every *.jsonl in eval_instructions/, which is what evaluate_probe(splits=None)
# auto-discovers. Each is exactly class-balanced; 1302 rows in total.
EVAL_SPLITS = [
    "anthropic_harmless_refusal",
    "bbq_substitution",
    "hc_context_drift",
    "hc_contradiction",
    "mm_substitution",
    "oig_context_drift",
    "oig_omission",
]


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


def probe_params(probe) -> tuple[torch.Tensor, float, float]:
    """``(w, b, temperature)`` in fp32 — see the module docstring on precision."""
    linear = probe._classifier.model.linear
    w = linear.weight.detach().float().flatten()
    b = float(linear.bias.detach().float().item())
    return w, b, float(probe.hyper_params["temperature"])


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


@dataclass
class Pair:
    """One red-team success and its generated opposite-class counterpart."""

    pair_id: int
    source_key: str
    source_idx: int | None = None
    generated_idx: int | None = None
    source_label: str = ""
    generated_label: str = ""
    # Row indices into the *training* side only; the val-side members are recorded
    # separately because influence is blind to them (they act via early stopping).
    train_rows: list[int] = field(default_factory=list)
    val_rows: list[int] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.train_rows) + len(self.val_rows)


def generated_to_source(arm: str) -> dict[str, str]:
    """Map each generated row's canonical text to the success it was written from.

    ``generate_contrastive_dataset`` returns ``sources + generated`` with the two halves
    in different orders, so position cannot be trusted. The contrastive cache stores each
    generated record together with its ``original_messages``, which is the authoritative
    link — and it survives the message transforms because both sides get the same ones
    applied here.
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
            (pair.val_rows if is_val(redteam_dataset.inputs[idx]) else pair.train_rows).append(idx)
        if pair.source_idx is None or pair.generated_idx is None:
            stats["n_orphan"] += 1

    ordered = sorted(pairs.values(), key=lambda p: p.pair_id)
    stats["n_pairs"] = len(ordered)
    stats["n_straddling"] = sum(1 for p in ordered if p.train_rows and p.val_rows)
    stats["n_fully_val"] = sum(1 for p in ordered if not p.train_rows)
    return ordered, stats


# --- the over-1024-token filter ---------------------------------------------------
#
# ``attacker.max_sample_tokens`` stops the *attacker* from submitting a conversation the
# probe cannot read whole, but ``preprocessing.max_sample_tokens`` — the same guard on the
# contrastive generator — landed after these runs, so their generated counterparts were
# never length-checked. Anything past MAX_ACTIVATION_TOKENS is truncated by
# ``get_activations``: the stored blob is the conversation's first 1024 tokens and the tail
# was never seen, by the probe at score time or by the trainer at fit time. This experiment
# drops those rows rather than training on the amputation.


def token_lengths(messages_list, model_name: str) -> list[int | None]:
    """Token width each conversation reaches under ``tokenize_inputs``.

    Delegates to ``token_budget.count_tokens``, which bakes in both of that function's
    traps (the no-op ``<bos>`` strip ⇒ never subtract 1; the chat template's own special
    tokens ⇒ ``add_special_tokens=False``). Counting by hand would be off by one exactly
    at the cap, which is the only place it matters. ``None`` for a conversation the
    tokenizer or chat template rejects — the caller must treat that as "keep", matching
    the fail-open rule the submit-time guard follows.
    """
    from agentic_redteam.token_budget import count_tokens

    return [
        count_tokens(
            model_name,
            list(messages),
            combine_consecutive_messages=COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=CONVERT_TOOL_TO_ASSISTANT,
        )
        for messages in messages_list
    ]


def over_long_rows(redteam_dataset, model_name: str,
                   max_tokens: int = MAX_ACTIVATION_TOKENS) -> tuple[set[int], dict]:
    """Row indices whose conversation exceeds ``max_tokens``, plus a length report.

    The dump's rows are already transformed, so re-applying the transforms inside
    ``count_tokens`` is a no-op — the same property ``publish_kaggle_redteam_activations``
    asserts before it trusts a dump's cache keys.
    """
    lengths = token_lengths(redteam_dataset.inputs, model_name)
    over = {i for i, n in enumerate(lengths) if n is not None and n > max_tokens}
    known = [n for n in lengths if n is not None]
    report = {
        "max_tokens": max_tokens,
        "n_rows": len(lengths),
        "n_uncountable": sum(1 for n in lengths if n is None),
        "n_over": len(over),
        "mean_tokens": float(np.mean(known)) if known else 0.0,
        "p95_tokens": float(np.percentile(known, 95)) if known else 0.0,
        "max_observed": int(max(known)) if known else 0,
    }
    return over, report


def truncated_blob_rows(redteam_dataset, model_name: str) -> tuple[set[int], dict]:
    """Rows whose *cached activation* holds fewer real tokens than the conversation has.

    The token count says which conversations are too long in principle; this says which
    ones were actually stored amputated, and the two are not the same set — because
    ``get_activations`` truncates a *batch*, not a row. It pads the batch to its longest
    member (left-padded for gemma), then keeps position ``[:1024]``. So a conversation
    comfortably under the cap that merely shared a chunk with an over-long one loses its
    tail too: measured here, nemotron's 952-token row 649 is stored with 761 real tokens
    because a 1215-token conversation from an earlier iteration was extracted alongside it.

    That row is the same defect the length filter is aimed at — trained on an amputation —
    so it is detected from the blobs rather than inferred from the lengths. Rows whose blob
    is absent or unreadable are reported and left alone, matching the fail-open rule.
    """
    lengths = token_lengths(redteam_dataset.inputs, model_name)
    truncated, missing = set(), 0
    for i, messages in enumerate(redteam_dataset.inputs):
        path = redteam_blob_path(messages)
        if lengths[i] is None or not path.exists():
            missing += 1
            continue
        try:
            blob = torch.load(path, weights_only=False, mmap=True)
        except Exception:  # noqa: BLE001 — an unreadable blob is a keep, not a drop
            missing += 1
            continue
        if int(blob["attention_mask"].sum()) < lengths[i]:
            truncated.add(i)
    return truncated, {"n_truncated_blobs": len(truncated), "n_unchecked": missing}


# --- activations ------------------------------------------------------------------


def redteam_blob_path(messages) -> Path:
    from agentic_redteam.retrain import _redteam_activation_cache_path

    return _redteam_activation_cache_path(
        SHARED_CACHE,
        messages,
        "google/gemma-3-27b-it",
        32,
        COMBINE_CONSECUTIVE_MESSAGES,
        CONVERT_TOOL_TO_ASSISTANT,
    )


def base_blob_paths() -> tuple[Path, Path]:
    from agentic_redteam.retrain import _base_activation_cache_paths

    return _base_activation_cache_paths(
        SHARED_CACHE,
        BASE_TRAINING_DATA,
        "google/gemma-3-27b-it",
        32,
        SEED,
        TEST_SIZE,
        None,
        COMBINE_CONSECUTIVE_MESSAGES,
        CONVERT_TOOL_TO_ASSISTANT,
        1.0,
    )


# --- forward pass, logits and Jacobians -------------------------------------------


def _forward(h: torch.Tensor, mask: torch.Tensor, w: torch.Tensor, b: float, T: float):
    """``LinearThenSoftmax`` in fp32, plus the per-token pooling weights."""
    z = h @ w + b
    z = z.masked_fill(~mask, 0.0)
    p = torch.softmax(z.masked_fill(~mask, float("-inf")) / T, dim=-1)
    s = (z * p).sum(-1)
    return s, z, p


def sequence_logits_and_jacobians(
    acts: torch.Tensor,
    mask: torch.Tensor,
    w: torch.Tensor,
    b: float,
    T: float,
    device: str = "cuda",
    batch_size: int = 8,
):
    """``(s, J)`` where ``J[i] = ds_i/dw`` — the exact derivative through the pooling."""
    n = acts.shape[0]
    out_s = torch.empty(n, dtype=torch.float32)
    out_j = torch.empty((n, w.numel()), dtype=torch.float32)
    w = w.to(device)
    for i in range(0, n, batch_size):
        h = acts[i : i + batch_size].to(device, torch.float32)
        mk = mask[i : i + batch_size].to(device).bool()
        h = h * mk[:, :, None]
        s, z, p = _forward(h, mk, w, b, T)
        coeff = p * (1.0 + (z - s[:, None]) / T)
        j = torch.einsum("bt,bte->be", coeff, h)
        out_s[i : i + batch_size] = s.cpu()
        out_j[i : i + batch_size] = j.float().cpu()
    return out_s.numpy(), out_j.numpy()


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
