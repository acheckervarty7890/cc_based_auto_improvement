#!/usr/bin/env python
"""Train on base data plus a chosen set of batches, and score it on the eval split.

The loop accepts a batch only if it raises dev AUROC **on its own**, against the current
probe. That is a per-batch test, so a batch that is individually harmful can still be
useful in combination — nothing in the loop ever tries the union of rejected batches. This
script does exactly that: pick batches by run and index, concatenate their samples with the
base training set, fit one probe under the run's own settings, and score it on the eval
split the run was measured against.

Everything is pinned to the arm's config so the number is comparable to its ledger: layer
32, ``linear_then_softmax``, a 5-member ensemble, ``combine``/``convert`` on, seed 42, and
the 32-row dev split as the fit's sole validation set (so all 80 training rows train).

    set -a; . ./.env; set +a
    .venv_claude/bin/python scripts/train_on_rejected_batches.py \
        --run results_oig_omission_nemotron_topics_v1 --label arm4_all
    # or pick explicitly:  --batch 0:0 --batch 0:2
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
from agentic_redteam.retrain import train_initial_probe  # noqa: E402

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
ARCH = "linear_then_softmax"
POS = "assistant_follows_the_instruction"
NEG = "assistant_does_not_follow_the_instruction"
SEED = 42
COMBINE = CONVERT = True

BASE = REPO / "data/instructions_llama70b_50.jsonl"
DEV = REPO / "dev_samples/oig_omission"
EVAL = REPO / "eval_sets/oig_omission"
CACHE = REPO / "cache_oig_omission"
OUT = REPO / "results_rejected_union"
DESC = (
    "A conversation where the user asks for several distinct things at once and the label "
    "is set by whether the assistant's reply addresses every part."
)


def _newest_batches(run_dir: Path) -> dict:
    newest = {}
    for line in (run_dir / "batches.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        newest[(r["iteration"], r["batch_index"])] = r
    return newest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="a results_* directory")
    ap.add_argument("--batch", action="append", default=[],
                    help="iteration:batch_index; repeatable. Default: every batch in the run.")
    ap.add_argument("--label", required=True, help="names the probe and the output json")
    ap.add_argument("--ensemble-size", type=int, default=5)
    args = ap.parse_args()

    newest = _newest_batches(args.run)
    if args.batch:
        keys = []
        for spec in args.batch:
            it, bi = spec.split(":")
            k = (int(it), int(bi))
            if k not in newest:
                raise SystemExit(f"{args.run}: no batch {spec}")
            keys.append(k)
    else:
        keys = sorted(newest)

    rows, chosen = [], []
    for k in keys:
        r = newest[k]
        chosen.append((k, r["delta"], r["accepted"], len(r["samples"])))
        for s in r["samples"]:
            rows.append({"inputs": json.dumps(s["messages"], ensure_ascii=False),
                         "labels": s["label"]})

    print(f"run {args.run.name}")
    for (it, bi), delta, acc, n in chosen:
        print(f"  i{it} b{bi + 1}: Δ{delta:+.4f} ({round(delta * 256):+d} pairs) "
              f"{'accepted' if acc else 'REJECTED'} · {n} samples")
    labels = {r["labels"] for r in rows}
    if not labels <= {POS, NEG}:
        raise SystemExit(f"unexpected labels: {labels - {POS, NEG}}")
    n_pos = sum(1 for r in rows if r["labels"] == POS)
    print(f"  -> {len(rows)} samples ({n_pos} pos / {len(rows) - n_pos} neg)")

    base_rows = [l for l in BASE.read_text().splitlines() if l.strip()]
    train_path = REPO / "data" / f"instructions_base50_plus_{args.label}.jsonl"
    train_path.write_text("\n".join(base_rows + [json.dumps(r, ensure_ascii=False) for r in rows]) + "\n")
    print(f"training data: {len(base_rows)} base + {len(rows)} generated = "
          f"{len(base_rows) + len(rows)} rows -> {train_path.name}")

    probe_path = REPO / "probes/rejected_union" / f"{args.label}.pkl"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result = train_initial_probe(
        base_training_data_path=train_path, model_name=MODEL_NAME, layer=LAYER,
        new_probe_path=probe_path, pos_class_label=POS, neg_class_label=NEG,
        probe_description=DESC, probe_spec=ARCH, dev_data_path=DEV, seed=SEED,
        ensemble_size=args.ensemble_size,
        base_activation_cache_dir=CACHE / "base_activations",
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT, verbose=True,
    )
    print(f"\nfit {(time.monotonic() - t0) / 60:.1f} min · dev AUROC {result.dev_auroc}")
    _free_gpu()

    df = evaluate_probe(probe_path, EVAL, CACHE / "eval_activations", max_samples=None,
                        seed=SEED, combine_consecutive_messages=COMBINE,
                        convert_tool_to_assistant=CONVERT)
    _free_gpu()
    print(f"\n--- {args.label} ---\n{df.to_string(index=False)}")

    OUT.mkdir(parents=True, exist_ok=True)
    auroc = float(df[df["dataset"] == "oig_omission"]["auroc"].iloc[0])
    (OUT / f"{args.label}.json").write_text(json.dumps({
        "label": args.label, "run": str(args.run),
        "batches": [{"iteration": k[0], "batch": k[1], "delta": d, "accepted": a, "n": n}
                    for k, d, a, n in chosen],
        "n_base": len(base_rows), "n_generated": len(rows),
        "dev_auroc": result.dev_auroc, "eval_auroc": auroc,
        "probe": str(probe_path),
    }, indent=2))
    print(f"\neval AUROC {auroc:.4f}   (base probe 0.7979 · base+32 real dev rows 0.8717 · ceiling 0.9254)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
