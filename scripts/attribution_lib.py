"""Shared machinery for attributing eval AUROC to individual red-team conversations.

The question this supports: *which red-team conversations in a probe's training set
had no effect on the eval AUROC of any split, or made one worse?*

Everything here runs off activations that are already on disk — the four hu_ha eval
split blobs, the base train/val blobs, and the per-conversation red-team blobs. No
forward pass through gemma-3-27b is ever needed, so a full attribution costs minutes
rather than the days the extraction itself took.

Three things in here are load-bearing and were each established by measurement, not
assumption:

**The unit of attribution is the pair, not the row.** ``preprocessing`` turns each
red-team success into two training rows: the success itself and an LLM-generated
opposite-class counterpart. Dropping one without the other is a different (and
label-unbalancing) intervention, so ``build_pairs`` rejoins them by re-deriving
``preprocessing._cache_key`` against the arm's ``contrastive_cache.jsonl``.

**Scoring must go through the probe's own forward path, or match it in fp32
deliberately.** The probe is stored in bf16, whose ULP at these logit magnitudes
(|s| up to ~25) is ~0.1. A hand-rolled matmul that differs by 1-2 ULP moves
``eval_ant_hh``'s AUROC by ~0.006 — 26 of its 4489 class pairs sit within 2 ULP of
each other. ``sequence_logits`` therefore computes in fp32 (self-consistent, and the
more accurate evaluation of the same weights) and callers must not mix the two
scales within one comparison.

**The pipeline's AUROC is computed on saturated probabilities.** ``get_performances``
scores ``predict_proba``, i.e. ``sigmoid(s)`` in bf16, where anything above logit
~5.5 rounds to exactly 1.0 — 63/136, 34/134, 81/400 and 71/196 rows across the four
splits. Those rows are mutually tied and sklearn awards 0.5 per tied pair, so the
published AUROC is *not* the rank AUROC of the logits (they differ by 0.0057 on
ant_hh). Any perturbation that only reshuffles the saturated block is invisible to
the published metric. Hence :func:`auroc_pipeline` (replicates the CSV) and
:func:`auroc_rank` (rank-faithful), and every report carries both.
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

# --- the two arms of run_gemma27b_hu_harm_attacker_ablation_batch.sh --------------
BASE_TRAINING_DATA = REPO / "data/hu_harm_llama70b_50.jsonl"
EVAL_DATASET_DIR = REPO / "eval_dataset_hu_ha"
SHARED_CACHE = REPO / "results_hu_harm_gemma27b_batch_ablation/base_activations"


def _resolve_eval_activations_dir() -> Path:
    """Where the four ``<split>-acts_full.pt`` eval blobs live.

    Two locations are in play and neither is wrong: this box happens to hold them
    under ``archive/`` (left over from the run that computed them), while a box that
    fetched them from Kaggle puts them in the shared cache dir the configs name. The
    activation blobs are multi-GB and therefore never committed, so a fresh checkout
    has neither until something downloads them — hence "first that exists, else the
    shared dir" rather than a hard path. ``AGENTIC_ATTRIB_EVAL_DIR`` overrides.
    """
    override = os.environ.get("AGENTIC_ATTRIB_EVAL_DIR")
    if override:
        return Path(override)
    shared = REPO / "results_hu_harm_gemma27b_batch_ablation/eval_activations"
    legacy = REPO / "archive/results3/gemma27_activations"
    for candidate in (shared, legacy):
        if any(candidate.glob("*-acts_full.pt")):
            return candidate
    return shared


EVAL_ACTIVATIONS_DIR = _resolve_eval_activations_dir()

# Both arms' configs: no `eval:` overrides beyond these, seed 42, 0.2 val split.
COMBINE_CONSECUTIVE_MESSAGES = True
CONVERT_TOOL_TO_ASSISTANT = True
SEED = 42
TEST_SIZE = 0.2

ARMS = {
    "gptoss120b": REPO / "probes/hu_harm_gemma27b_gptoss120b_batch",
    "deepseekv4pro": REPO / "probes/hu_harm_gemma27b_deepseekv4pro_batch",
}
EVAL_SPLITS = [
    "eval_ai_dilemmas",
    "eval_ant_hh",
    "eval_balanced_refusal",
    "eval_daily_dilemmas",
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


def build_pairs(arm: str, redteam_dataset) -> tuple[list[Pair], dict]:
    """Rejoin every generated row to the success it was written from.

    ``generate_contrastive_dataset`` returns ``sources + generated`` with the two
    halves in different orders, so position cannot be trusted. The contrastive cache
    stores each generated record together with its ``original_messages``, which is the
    authoritative link — and it survives the message transforms because both sides get
    the same ones applied here.
    """
    cache_path = ARMS[arm] / "contrastive_cache.jsonl"
    from tuberlens.interfaces.dataset import Message

    def as_messages(raw):
        return apply_transforms(
            [Message(role=m["role"], content=m["content"]) for m in raw]
        )

    generated_to_source: dict[str, str] = {}
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line).get("record", {})
            if not rec.get("inputs") or not rec.get("original_messages"):
                continue
            generated_to_source[canon(as_messages(rec["inputs"]))] = canon(
                as_messages(rec["original_messages"])
            )

    labels = redteam_dataset.other_fields["labels"]
    keys = [canon(m) for m in redteam_dataset.inputs]

    pairs: dict[str, Pair] = {}
    stats = {"n_rows": len(keys), "n_generated": 0, "n_source": 0, "n_orphan": 0}

    def get(source_key: str) -> Pair:
        if source_key not in pairs:
            pairs[source_key] = Pair(pair_id=len(pairs), source_key=source_key)
        return pairs[source_key]

    for idx, key in enumerate(keys):
        if key in generated_to_source:
            pair = get(generated_to_source[key])
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
    stats["n_straddling"] = sum(
        1 for p in ordered if p.train_rows and p.val_rows
    )
    stats["n_fully_val"] = sum(1 for p in ordered if not p.train_rows)
    return ordered, stats


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
    """``LinearThenSoftmax`` in fp32, plus the per-token pooling weights.

    Mirrors ``LinearThenAgg.forward`` + ``LinearThenSoftmax.agg``: linear per token,
    zero the pad positions, softmax the *logits* at temperature ``T``, return the
    weighted sum. Pad positions get ``p = 0`` so they drop out of both the value and
    the derivative.
    """
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
    """``(s, J)`` where ``J[i] = ds_i/dw`` — the exact derivative through the pooling.

    Analytic rather than autograd, because the closed form is short and avoids
    building a graph over ~10^6 token activations:

        ds/dz_u = p_u * (1 + (z_u - s) / T)        (softmax Jacobian folded in)
        ds/dw   = sum_u p_u * (1 + (z_u - s) / T) * h_u

    Note ``ds/db == 1`` identically (the softmax is shift-invariant and the pooling
    weights sum to one), so the bias column is not returned — callers append a column
    of ones when they need the full 5377-dim parameter Jacobian.

    Freezing the pooling weights instead and using ``sum_u p_u h_u`` would reproduce
    the logit exactly at ``w`` but get the *derivative* wrong: it drops the term where
    moving ``w`` moves the attention itself. That term is what the ``(z_u - s)/T``
    factor carries.
    """
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
    """AUROC exactly as the pipeline reports it: bf16 ``sigmoid`` then sklearn.

    ``get_performances`` scores ``predict_proba``, and in bf16 every logit above ~5.5
    rounds to exactly 1.0. Those rows tie, sklearn credits 0.5 per tied pair, and the
    published number is depressed accordingly. Replicating the saturation is what
    makes a computed delta comparable to the committed comparison CSVs.
    """
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
