#!/usr/bin/env python
"""Control arm for generator_experiment_1: unsteered generation, same generator, same budget.

The three loop arms accepted 180 / 50 / 80 generated samples into training. Nothing in
that experiment separated "the judge's steering and the probe's arbitration help" from
"more Llama-3.3-70B data helps". This trains the other half of that comparison: the SAME
generator writing 150 samples per concept with **no memo, no direction and no probe
verdict**, concatenated with the same 50-row base set, fit exactly the way
``probe_iter0.pkl`` was and scored on the same full eval splits.

Everything that could otherwise differ is pinned to the run configs: layer 32,
``linear_then_softmax``, a single probe (no ensemble), ``combine_consecutive_messages``
and ``convert_tool_to_assistant`` both on, seed 42, and the dev directory as the fit's
sole validation set — so ``test_size`` is forced to 0.0 inside ``retrain`` and all 200
rows train.

Needs the Kaggle blobs already in place (``scripts/fetch_all_kaggle_activations.py``):
the dev blob is the fit's validation set and the eval blobs make scoring a pure cache
hit. Only the 200 training rows are extracted through gemma-3-27b, once per concept,
into the same content-keyed cache every later run reuses.

    set -a; . ./.env; set +a
    .venv_claude/bin/python scripts/run_control_arm.py [--concept highstakes]
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
SEED = 42
COMBINE = True
CONVERT = True

CONCEPTS = {
    "highstakes": dict(
        base=REPO / "data/highstakes_llama70b_50.jsonl",
        control=REPO / "data/highstakes_llama70b_150_control.jsonl",
        eval_dir=REPO / "eval_sets/highstakes",
        dev_dir=REPO / "dev_samples/highstakes",
        pos="high-stakes",
        neg="low-stakes",
        desc="A linear probe on google/gemma-3-27b-it detecting whether the conversation is high-stakes.",
        cache=REPO / "cache_gen_gemma27b_highstakes",
    ),
    "hu_harm": dict(
        base=REPO / "data/hu_harm_llama70b_50.jsonl",
        control=REPO / "data/hu_harm_llama70b_150_control.jsonl",
        eval_dir=REPO / "eval_sets/hu_ha",
        dev_dir=REPO / "dev_samples/hu_ha",
        pos="harmful_to_human",
        neg="not_harmful_to_human",
        desc="A linear probe on google/gemma-3-27b-it detecting whether the conversation is harmful to humans.",
        cache=REPO / "cache_gen_gemma27b_hu_harm",
    ),
    "instructions": dict(
        base=REPO / "data/instructions_llama70b_50.jsonl",
        control=REPO / "data/instructions_llama70b_150_control.jsonl",
        eval_dir=REPO / "eval_sets/instructions",
        dev_dir=REPO / "dev_samples/instructions",
        pos="assistant_follows_the_instruction",
        neg="assistant_does_not_follow_the_instruction",
        desc=(
            "A linear probe on google/gemma-3-27b-it that reads a conversation between a user "
            "and an assistant and detects whether the assistant's response follows the user's "
            "instruction."
        ),
        cache=REPO / "cache_gen_gemma27b_instructions",
    ),
}

OUT_DIR = REPO / "results_control_arm"
PROBE_DIR = REPO / "probes/control_arm"


def _combine_training_data(cfg: dict, name: str) -> tuple[Path, int, int]:
    """Write base ∪ control to one JSONL and return (path, n_base, n_control).

    Rows are passed through verbatim — both files already carry the ``{inputs, labels}``
    shape ``LabelledDataset.load_from`` reads, and the control generator emits the same
    schema as the base generator by construction (it is the same script family).
    """
    base_rows = [l for l in cfg["base"].read_text().splitlines() if l.strip()]
    ctrl_rows = [l for l in cfg["control"].read_text().splitlines() if l.strip()]
    labels = {cfg["pos"], cfg["neg"]}
    for src, rows in (("base", base_rows), ("control", ctrl_rows)):
        bad = [i for i, r in enumerate(rows) if json.loads(r).get("labels") not in labels]
        if bad:
            raise SystemExit(f"{name}/{src}: {len(bad)} row(s) carry a label outside {labels}")
    out = REPO / "data" / f"{name}_base50_plus_control150.jsonl"
    out.write_text("\n".join(base_rows + ctrl_rows) + "\n")
    return out, len(base_rows), len(ctrl_rows)


def run(name: str, cfg: dict) -> dict:
    print(f"\n{'=' * 70}\n### CONTROL ARM — {name}\n{'=' * 70}", flush=True)
    train_path, n_base, n_ctrl = _combine_training_data(cfg, name)
    print(f"training data: {n_base} base + {n_ctrl} control = {n_base + n_ctrl} rows -> {train_path.name}",
          flush=True)

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    probe_path = PROBE_DIR / f"control_{name}.pkl"

    t0 = time.monotonic()
    result = train_initial_probe(
        base_training_data_path=train_path,
        model_name=MODEL_NAME,
        layer=LAYER,
        new_probe_path=probe_path,
        pos_class_label=cfg["pos"],
        neg_class_label=cfg["neg"],
        probe_description=cfg["desc"],
        probe_spec=ARCH,
        dev_data_path=cfg["dev_dir"],
        seed=SEED,
        ensemble_size=1,
        base_activation_cache_dir=cfg["cache"] / "base_activations",
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
        verbose=True,
    )
    fit_min = (time.monotonic() - t0) / 60
    print(f"\nfit done in {fit_min:.1f} min — dev AUROC {result.dev_auroc}", flush=True)
    _free_gpu()

    df = evaluate_probe(
        probe_path,
        cfg["eval_dir"],
        cfg["cache"] / "eval_activations",
        max_samples=None,          # full splits — matches eval_max_samples: 0
        seed=SEED,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )
    _free_gpu()
    print(f"\n--- {name} control eval ---\n{df.to_string(index=False)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"control_{name}_eval.csv", index=False)
    rec = {
        "concept": name,
        "n_base": n_base,
        "n_control": n_ctrl,
        "n_training": n_base + n_ctrl,
        "probe": str(probe_path),
        "dev_auroc": result.dev_auroc,
        "eval": {r["dataset"]: float(r["auroc"]) for _, r in df.iterrows()},
        "fit_minutes": round(fit_min, 2),
    }
    (OUT_DIR / f"control_{name}.json").write_text(json.dumps(rec, indent=2))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", choices=sorted(CONCEPTS), action="append")
    args = ap.parse_args()

    records, failures = [], []
    for name in (args.concept or list(CONCEPTS)):
        try:
            records.append(run(name, CONCEPTS[name]))
        except Exception as e:  # noqa: BLE001 — one dead concept must not lose the others
            import traceback
            traceback.print_exc()
            failures.append(f"{name}: {type(e).__name__}: {e}")

    if records:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "control_summary.json").write_text(json.dumps(records, indent=2))
        print("\n\n===== CONTROL ARM SUMMARY =====")
        for r in records:
            print(f"{r['concept']:14s} n={r['n_training']:4d}  "
                  f"dev {r['dev_auroc'].get('mean', float('nan')):.4f}  "
                  f"eval {r['eval'].get('mean', float('nan')):.4f}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(" ", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
