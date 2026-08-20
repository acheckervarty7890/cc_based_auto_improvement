"""Deterministic, split- and class-stratified partitions used by both experiments."""
from __future__ import annotations

import numpy as np

# The dev splits, in sorted(glob("*.jsonl")) order — the order _load_dev_dataset
# concatenates them in, so these row counts give each dev row its split identity.
DEV_SPLIT_SIZES = [
    ("anthropic_harmless_refusal", 68),
    ("bbq_substitution", 68),
    ("hc_context_drift", 66),
    ("hc_contradiction", 68),
    ("mm_substitution", 68),
    ("oig_context_drift", 66),
    ("oig_omission", 32),
]


def dev_split_of_row() -> list[str]:
    out = []
    for name, n in DEV_SPLIT_SIZES:
        out.extend([name] * n)
    return out


def stratified_partition(strata: list, n_parts: int, seed: int) -> np.ndarray:
    """Assign each row to one of `n_parts`, round-robin within each stratum.

    Deterministic given `seed`. Every stratum is spread as evenly as it can be, so
    each part carries ~1/n_parts of every (split, label) cell.
    """
    rng = np.random.default_rng(seed)
    assign = np.empty(len(strata), dtype=int)
    keys = sorted(set(map(str, strata)))
    for k in keys:
        idx = np.array([i for i, s in enumerate(strata) if str(s) == k])
        rng.shuffle(idx)
        assign[idx] = np.arange(len(idx)) % n_parts
    return assign


def stratified_order(strata: list, seed: int) -> np.ndarray:
    """A shuffle whose every prefix is as class/split-balanced as it can be.

    Rows are shuffled within each stratum and then interleaved round-robin across
    strata, so `order[:N]` is a balanced N-sample draw for every N — which is what
    makes the sweep's nested subsets comparable to one another.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(set(map(str, strata)))
    buckets = []
    for k in keys:
        idx = np.array([i for i, s in enumerate(strata) if str(s) == k])
        rng.shuffle(idx)
        buckets.append(list(idx))
    out = []
    while any(buckets):
        for b in buckets:
            if b:
                out.append(b.pop(0))
    return np.array(out)
