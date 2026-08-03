"""Evaluate a probe on local eval datasets via tuberlens.get_performances."""

from __future__ import annotations

import pickle
import random
from pathlib import Path

DEFAULT_EVAL_MAX_SAMPLES = 100
DEFAULT_SEED = 42


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility (ported from tuberlens' pipeline script)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[assignment]
    torch.backends.cudnn.benchmark = False  # type: ignore[assignment]


def evaluate_probe(
    probe_path: str | Path,
    eval_dataset_dir: str | Path,
    activations_cache_dir: str | Path,
    splits: list[str] | None = None,
    max_samples: int | None = DEFAULT_EVAL_MAX_SAMPLES,
    seed: int = DEFAULT_SEED,
    combine_consecutive_messages: bool = False,
    convert_tool_to_assistant: bool = False,
    kaggle_source=None,
):
    """Score one probe on the local eval splits, returning a per-split DataFrame.

    ``splits`` are split *names*; each is read from ``<eval_dataset_dir>/<name>.jsonl``.
    When ``splits is None`` (the default) the splits are auto-discovered: every
    ``*.jsonl`` file in ``eval_dataset_dir`` is used, keyed by its filename stem.
    This keeps the evaluator dataset-agnostic — drop new eval JSONLs into the dir and
    they are picked up without code changes.

    Each split is read using the probe's own
    pos/neg class labels, then subsampled to a balanced ``max_samples`` subset
    (``max_samples // 2`` per class) via tuberlens' ``subsample_balanced_subset`` —
    pass ``max_samples=None`` to evaluate the full split.

    ``seed_everything(seed)`` is called before subsampling so the subset is identical
    across every call (e.g. base + each retrained probe). This also keeps the
    path-keyed Llama activation cache valid: ``get_performances`` reloads activations
    purely by file path without checking they match the inputs, so a deterministic
    subsample is what makes that cache safe to reuse across probes. The seed and
    sample count are baked into the cache filename so a different subsample config
    can't silently reuse stale activations.

    ``combine_consecutive_messages`` / ``convert_tool_to_assistant`` are tuberlens
    ``LabelledDataset`` loader transforms applied to the eval splits only (mirroring
    the eval step of tuberlens' ``collate_train_evaluate.py``): merge adjacent
    same-role messages, and rewrite ``tool`` messages as ``assistant`` messages
    (the latter applied first). Training/retraining data is unaffected.

    ``kaggle_source`` (a ``KaggleActivationSource``) pre-populates the activation
    cache from precomputed blobs published on Kaggle, before ``get_performances``
    runs. Each downloaded blob is validated against the probe's ``model_name`` /
    ``layer`` and the split's row count, then written to the exact path
    ``get_performances`` would have computed — so the eval becomes a pure cache hit
    and never loads the LLM. Only valid for full splits (``max_samples=None``): the
    published blobs cover whole splits, and a subsampled run needs different rows.
    """
    from tuberlens.evaluation import get_performances
    from tuberlens.interfaces.dataset import LabelledDataset, subsample_balanced_subset

    probe_path = Path(probe_path)
    eval_dataset_dir = Path(eval_dataset_dir)
    activations_cache_dir = Path(activations_cache_dir)
    if splits is None:
        splits = sorted(p.stem for p in eval_dataset_dir.glob("*.jsonl"))
        if not splits:
            raise ValueError(
                f"No eval split JSONLs (*.jsonl) found in {eval_dataset_dir}"
            )

    with probe_path.open("rb") as f:
        probe = pickle.load(f)

    seed_everything(seed)

    eval_datasets = {}
    for name in splits:
        dataset = LabelledDataset.load_from(
            eval_dataset_dir / f"{name}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
        )
        if max_samples is not None:
            dataset = subsample_balanced_subset(dataset, n_per_class=max_samples // 2)
        eval_datasets[name] = dataset

    activations_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = (
        f"acts_n{max_samples}_seed{seed}.pt" if max_samples is not None else "acts_full.pt"
    )

    if kaggle_source is not None:
        from agentic_redteam.kaggle_activations import prefetch_eval_activations

        if max_samples is not None:
            raise ValueError(
                "kaggle_source requires full splits (eval_max_samples: 0 / "
                f"max_samples=None), but max_samples={max_samples}. The published "
                "blobs cover whole splits; a subsampled run needs different rows."
            )
        prefetch_eval_activations(
            activations_cache_dir,
            eval_datasets,
            kaggle_source,
            model_name=probe.model_name,
            layer=int(probe.layer),
            cache_stem=cache_stem,
        )

    activations_save_path = activations_cache_dir / cache_stem
    _assign_cached_activations(eval_datasets, activations_save_path)

    return get_performances(
        probe,
        eval_datasets,
        activations_save_path=str(activations_save_path),
    )


def _assign_cached_activations(eval_datasets: dict, activations_save_path: Path) -> None:
    """Attach already-cached split activations in place, so the LLM is never loaded.

    ``get_performances`` loads the extraction model the moment it meets a split with
    no ``activations`` field (``tuberlens/evaluation.py:75-77``) — *before*
    ``get_activations`` gets as far as checking ``save_path.exists()``. So an eval
    whose splits are all cached (the normal case once ``kaggle:`` has prefetched them)
    still paid a multi-minute gemma-3-27b load it made no use of. Assigning the blobs
    ourselves means every split arrives pre-activated and that branch is never taken.

    Splits whose blob is missing, unreadable, or the wrong row count are left alone
    for ``get_performances`` to compute as before — this is a fast path, never a
    correctness gate.
    """
    from tuberlens.model import LLMModel

    stem = activations_save_path.stem
    for name, dataset in list(eval_datasets.items()):
        path = activations_save_path.with_stem(f"{name}-{stem}")
        if not path.exists():
            continue
        try:
            acts = LLMModel.load_activations(path)
        except Exception as exc:  # noqa: BLE001 — fall back to recomputing this split
            print(f"[eval] ignoring unreadable activation cache {path}: {exc}")
            continue
        if len(acts.activations) != len(dataset):
            print(
                f"[eval] ignoring activation cache {path}: {len(acts.activations)} rows "
                f"but split '{name}' has {len(dataset)}"
            )
            continue
        eval_datasets[name] = dataset.assign(
            activations=acts.activations,
            attention_mask=acts.attention_mask,
            input_ids=acts.input_ids,
        )
