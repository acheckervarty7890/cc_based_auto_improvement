#!/usr/bin/env python
"""Retrain base ∪ a JSONL of samples, then score on the eval split.

Uses ``retrain_probe(samples=...)`` rather than folding the samples into the base training
file, which matters for cost: the base split is read from its whole-set blob and each
sample goes through the **per-conversation** cache, so only genuinely new conversations are
forwarded through the 27B model. Stuffing them into the base file instead gives that blob a
new content hash and re-extracts everything, base rows included.

Architecture and ensemble size are inherited from ``--base-probe``, so the result is
comparable to that probe's own ledger without restating its settings here.

    set -a; . ./.env; set +a
    .venv_claude/bin/python scripts/retrain_with_samples.py \
        --samples data/arm4_shortened.jsonl --label arm4_shortened \
        --base-probe probes/gen_gemma27b_oig_omission_nemotron_topics_v1/probe_iter0.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agentic_redteam.cli import _free_gpu  # noqa: E402
from agentic_redteam.evaluation import evaluate_probe  # noqa: E402
from agentic_redteam.retrain import retrain_probe  # noqa: E402

POS = "assistant_follows_the_instruction"
NEG = "assistant_does_not_follow_the_instruction"
BASE = REPO / "data/instructions_llama70b_50.jsonl"
DEV = REPO / "dev_samples/oig_omission"
EVAL = REPO / "eval_sets/oig_omission"
CACHE = REPO / "cache_oig_omission"
OUT = REPO / "results_shortened"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-probe", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # The on-disk LabelledDataset schema stores `inputs` as a JSON-encoded STRING, but
    # samples_to_dataset takes in-memory dicts and expects an already-parsed message list.
    rows = []
    for line in args.samples.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        inp = r["inputs"]
        rows.append({"inputs": json.loads(inp) if isinstance(inp, str) else inp,
                     "labels": r["labels"]})
    labels = {r["labels"] for r in rows}
    if not labels <= {POS, NEG}:
        raise SystemExit(f"unexpected labels: {labels - {POS, NEG}}")
    n_pos = sum(1 for r in rows if r["labels"] == POS)
    chars = sorted(sum(len(m["content"]) for m in r["inputs"]) for r in rows)
    print(f"{len(rows)} samples ({n_pos} pos / {len(rows) - n_pos} neg) from {args.samples.name}"
          f" · median {chars[len(chars)//2]} chars")

    probe_path = REPO / "probes/shortened" / f"{args.label}.pkl"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result = retrain_probe(
        samples=rows,
        base_probe_path=args.base_probe,
        base_training_data_path=BASE,
        new_probe_path=probe_path,
        dev_data_path=DEV,
        seed=args.seed,
        base_activation_cache_dir=CACHE / "base_activations",
        combine_consecutive_messages=True,
        convert_tool_to_assistant=True,
        verbose=True,
    )
    print(f"\nfit {(time.monotonic() - t0) / 60:.1f} min · {result.n_training_samples_total} "
          f"training rows · ensemble {result.ensemble_size} · dev AUROC {result.dev_auroc}")
    _free_gpu()

    df = evaluate_probe(probe_path, EVAL, CACHE / "eval_activations", max_samples=None,
                        seed=args.seed, combine_consecutive_messages=True,
                        convert_tool_to_assistant=True)
    _free_gpu()
    print(f"\n--- {args.label} ---\n{df.to_string(index=False)}")
    auroc = float(df[df["dataset"] == "oig_omission"]["auroc"].iloc[0])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.label}.json").write_text(json.dumps({
        "label": args.label, "samples": str(args.samples), "n_samples": len(rows),
        "n_training_total": result.n_training_samples_total,
        "ensemble_size": result.ensemble_size, "dev_auroc": result.dev_auroc,
        "eval_auroc": auroc, "probe": str(probe_path)}, indent=2))
    print(f"\neval AUROC {auroc:.4f}   "
          f"(base probe 0.7979 · base+32 real dev rows 0.8717 · ceiling 0.9254)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
