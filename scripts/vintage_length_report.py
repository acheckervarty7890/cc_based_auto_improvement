"""Token-length census of each arm's iteration-3 red-team training rows.

Answers two questions the vintage sweep needs settled before it runs:

1. **Which rows overran the activation window?** ``get_activations`` truncates at
   ``ACTIVATION_MAX_TOKENS`` (1024) and ``_activate_redteam_cached`` writes each
   conversation at its own length, so a blob of width exactly 1024 is a conversation
   that reached the cap — its tail is gone. For a *generated* contrastive row the tail
   is disproportionately the part carrying the opposite-class label, which is why the
   human-harm run shortened those pairs rather than leaving them in.

   Reading the width off the blob header uses the tokenization the extraction actually
   performed, so it cannot disagree with the model and needs no tokenizer.

2. **How wide will the padded training tensor be?** Every fit materialises one
   rectangular ``(n, width, 5376)`` fp16 tensor, and ``width`` is the max over the rows
   kept. At 11 MB per row per 1024 tokens that number decides whether the sweep fits in
   host RAM.

Usage:
    .venv_claude/bin/python scripts/vintage_length_report.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import attribution_lib as A


def census(arm: str, iteration: int) -> dict:
    ds = A.load_redteam_dataset(arm, iteration)
    pairs, stats = A.build_pairs(arm, ds)

    widths = np.array([A.blob_width(A.redteam_blob_path(m)) for m in ds.inputs])
    gen2src = A.generated_to_source(arm)
    is_generated = np.array([A.canon(m) in gen2src for m in ds.inputs])
    at_cap = widths >= A.ACTIVATION_MAX_TOKENS

    n_pairs_gen_capped = sum(
        1 for p in pairs if p.generated_idx is not None and at_cap[p.generated_idx]
    )
    n_pairs_any_capped = sum(
        1
        for p in pairs
        if any(at_cap[i] for i in (p.source_idx, p.generated_idx) if i is not None)
    )
    rows_in_gen_capped_pairs = sorted(
        i
        for p in pairs
        if p.generated_idx is not None and at_cap[p.generated_idx]
        for i in (p.source_idx, p.generated_idx)
        if i is not None
    )

    return {
        "arm": arm,
        "iteration": iteration,
        "n_rows": len(ds),
        "pair_stats": stats,
        "widths": {
            "mean": float(widths.mean()),
            "median": float(np.median(widths)),
            "p90": float(np.percentile(widths, 90)),
            "max": int(widths.max()),
        },
        "n_at_cap_total": int(at_cap.sum()),
        "n_at_cap_generated": int((at_cap & is_generated).sum()),
        "n_at_cap_source": int((at_cap & ~is_generated).sum()),
        "n_pairs": len(pairs),
        "n_pairs_generated_at_cap": n_pairs_gen_capped,
        "n_pairs_any_at_cap": n_pairs_any_capped,
        "n_rows_dropped_by_pair_rule": len(rows_in_gen_capped_pairs),
        "label_counts": dict(Counter(ds.other_fields["labels"])),
        "width_hist": {
            str(b): int(c)
            for b, c in zip(
                ["<256", "256-511", "512-767", "768-1023", "1024"],
                [
                    int((widths < 256).sum()),
                    int(((widths >= 256) & (widths < 512)).sum()),
                    int(((widths >= 512) & (widths < 768)).sum()),
                    int(((widths >= 768) & (widths < 1024)).sum()),
                    int((widths >= 1024).sum()),
                ],
            )
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = []
    for arm in args.arm:
        rep = census(arm, args.iteration)
        out.append(rep)
        w = rep["widths"]
        print(f"\n=== {arm} iter{args.iteration} ===")
        print(f"  rows {rep['n_rows']}  pairs {rep['n_pairs']}  labels {rep['label_counts']}")
        print(f"  pair stats: {rep['pair_stats']}")
        print(
            f"  token width: mean {w['mean']:.0f}  median {w['median']:.0f}  "
            f"p90 {w['p90']:.0f}  max {w['max']}"
        )
        print(f"  histogram: {rep['width_hist']}")
        print(
            f"  AT CAP (>=1024, truncated): {rep['n_at_cap_total']} rows "
            f"({rep['n_at_cap_generated']} generated, {rep['n_at_cap_source']} source)"
        )
        print(
            f"  pairs whose GENERATED half is at cap: {rep['n_pairs_generated_at_cap']} "
            f"-> dropping those pairs removes {rep['n_rows_dropped_by_pair_rule']} rows"
        )
        print(f"  pairs with EITHER half at cap: {rep['n_pairs_any_at_cap']}")
        gb = rep["n_rows"] * w["max"] * 5376 * 2 / 1e9
        print(f"  padded tensor if nothing dropped: {gb:.1f} GB")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
