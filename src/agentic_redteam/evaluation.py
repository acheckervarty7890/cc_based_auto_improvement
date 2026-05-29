"""Evaluate a probe on local eval datasets via tuberlens.get_performances."""

from __future__ import annotations

import pickle
import random
from pathlib import Path

DEFAULT_EVAL_SPLITS = ["anthropic", "mts"]
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
):
    """Score one probe on the named local eval splits, returning a per-split DataFrame.

    Each split is read from ``<eval_dataset_dir>/<name>.jsonl`` using the probe's own
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
    """
    from tuberlens.evaluation import get_performances
    from tuberlens.interfaces.dataset import LabelledDataset, subsample_balanced_subset

    probe_path = Path(probe_path)
    eval_dataset_dir = Path(eval_dataset_dir)
    activations_cache_dir = Path(activations_cache_dir)
    splits = splits or DEFAULT_EVAL_SPLITS

    with probe_path.open("rb") as f:
        probe = pickle.load(f)

    seed_everything(seed)

    eval_datasets = {}
    for name in splits:
        dataset = LabelledDataset.load_from(
            eval_dataset_dir / f"{name}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
        )
        if max_samples is not None:
            dataset = subsample_balanced_subset(dataset, n_per_class=max_samples // 2)
        eval_datasets[name] = dataset

    activations_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = (
        f"acts_n{max_samples}_seed{seed}.pt" if max_samples is not None else "acts_full.pt"
    )
    return get_performances(
        probe,
        eval_datasets,
        activations_save_path=str(activations_cache_dir / cache_stem),
    )
