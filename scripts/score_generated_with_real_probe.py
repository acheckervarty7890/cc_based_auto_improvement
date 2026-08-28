#!/usr/bin/env python
"""Train on base + the real eval split, then score a run's GENERATED conversations.

Every experiment so far has asked whether generated data helps a probe on real data. This
reverses the direction: fit a probe on data we trust — the 50-row base set plus all 114
rows of the real ``oig_omission`` eval split — and then use it to score the generated
conversations, treating *their* labels as the thing under test.

What the number means. If the generated rows carry the concept the real data carries, a
probe that has learned it well should separate them, and AUROC on them will be high. Near
0.5 means the generator's labels are uninformative to a probe that understands the real
concept; **below** 0.5 means they are anti-correlated — the samples labelled "follows" look
more like real omissions than the ones labelled "does not follow", which is a labelling
failure rather than a distribution-shift one.

Note what is and is not held out here. The eval split is *training* data in this script and
the generated rows are the test set, so nothing leaks: the probe never sees a generated row
before scoring it. Early stopping uses the 32-row dev split, which is disjoint from both.

    set -a; . ./.env; set +a
    .venv_claude/bin/python scripts/score_generated_with_real_probe.py \
        --run results_oig_omission_nemotron_topics_v1 --label arm4
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
from agentic_redteam.retrain import load_probe, train_initial_probe  # noqa: E402

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
ARCH = "linear_then_softmax"
POS = "assistant_follows_the_instruction"
NEG = "assistant_does_not_follow_the_instruction"
SEED = 42
COMBINE = CONVERT = True

BASE = REPO / "data/instructions_llama70b_50.jsonl"
EVAL_JSONL = REPO / "eval_sets/oig_omission/oig_omission.jsonl"
DEV = REPO / "dev_samples/oig_omission"
CACHE = REPO / "cache_oig_omission"
OUT = REPO / "results_generated_scored"
DESC = (
    "A conversation where the user asks for several distinct things at once and the label "
    "is set by whether the assistant's reply addresses every part."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batch", action="append", default=[], help="iteration:batch_index; repeatable")
    ap.add_argument("--ensemble-size", type=int, default=5)
    args = ap.parse_args()

    # ---- the generated rows become a one-split eval directory ---------------------
    newest = {}
    for line in (args.run / "batches.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line); newest[(r["iteration"], r["batch_index"])] = r
    keys = ([tuple(int(x) for x in b.split(":")) for b in args.batch] if args.batch
            else sorted(newest))
    gen = [{"inputs": json.dumps(s["messages"], ensure_ascii=False), "labels": s["label"]}
           for k in keys for s in newest[k]["samples"]]
    n_pos = sum(1 for r in gen if r["labels"] == POS)
    print(f"generated test set: {len(gen)} rows ({n_pos} pos / {len(gen) - n_pos} neg) "
          f"from {args.run.name}, batches {[f'i{i}b{b+1}' for i, b in keys]}")

    test_dir = REPO / "eval_sets" / f"generated_{args.label}"
    test_dir.mkdir(parents=True, exist_ok=True)
    split = test_dir / f"generated_{args.label}.jsonl"
    split.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gen))

    # ---- train on base + the real eval split ---------------------------------------
    base_rows = [l for l in BASE.read_text().splitlines() if l.strip()]
    eval_rows = [l for l in EVAL_JSONL.read_text().splitlines() if l.strip()]
    train_path = REPO / "data" / f"instructions_base50_plus_eval114_{args.label}.jsonl"
    train_path.write_text("\n".join(base_rows + eval_rows) + "\n")
    print(f"training data: {len(base_rows)} base + {len(eval_rows)} real eval = "
          f"{len(base_rows) + len(eval_rows)} rows")

    probe_path = REPO / "probes/generated_scored" / f"base_plus_eval_{args.label}.pkl"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result = train_initial_probe(
        base_training_data_path=train_path, model_name=MODEL_NAME, layer=LAYER,
        new_probe_path=probe_path, pos_class_label=POS, neg_class_label=NEG,
        probe_description=DESC, probe_spec=ARCH, dev_data_path=DEV, seed=SEED,
        ensemble_size=args.ensemble_size,
        base_activation_cache_dir=CACHE / "base_activations",
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT, verbose=True)
    print(f"\nfit {(time.monotonic() - t0) / 60:.1f} min · dev AUROC {result.dev_auroc}")
    _free_gpu()

    # ---- score the generated rows ---------------------------------------------------
    df = evaluate_probe(probe_path, test_dir, CACHE / f"gen_{args.label}_activations",
                        max_samples=None, seed=SEED, combine_consecutive_messages=COMBINE,
                        convert_tool_to_assistant=CONVERT)
    _free_gpu()
    print(f"\n--- probe(base + real eval) scored on {len(gen)} GENERATED rows ---")
    print(df.to_string(index=False))

    # per-class mean score, to separate "uninformative" from "inverted"
    from tuberlens.interfaces.dataset import LabelledDataset
    import numpy as np
    ds = LabelledDataset.load_from(split.resolve(), pos_class_label=POS, neg_class_label=NEG,
                                   combine_consecutive_messages=COMBINE,
                                   convert_tool_to_assistant=CONVERT)
    from agentic_redteam.evaluation import _assign_cached_activations
    _assign_cached_activations({split.stem: ds}, CACHE / f"gen_{args.label}_activations" / "acts_full.pt")
    probe = load_probe(probe_path)
    p = np.asarray(probe.predict_proba(ds), dtype=float).reshape(-1)
    y = ds.labels_numpy()
    print(f"\nmean probe score  labelled-follows {p[y == 1].mean():.4f} | "
          f"labelled-omits {p[y == 0].mean():.4f}  (higher = more 'follows')")

    auroc = float(df[df["dataset"] == split.stem]["auroc"].iloc[0])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.label}.json").write_text(json.dumps({
        "label": args.label, "run": str(args.run), "n_generated": len(gen),
        "n_train": len(base_rows) + len(eval_rows), "dev_auroc": result.dev_auroc,
        "auroc_on_generated": auroc,
        "mean_score_follows": float(p[y == 1].mean()),
        "mean_score_omits": float(p[y == 0].mean()),
        "probe": str(probe_path)}, indent=2))
    verdict = ("agrees with the generated labels" if auroc > 0.65 else
               "cannot tell the generated classes apart" if auroc >= 0.45 else
               "ANTI-correlated with the generated labels")
    print(f"\nAUROC on the generated rows: {auroc:.4f} — the probe {verdict}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
