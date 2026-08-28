#!/usr/bin/env python
"""Unsteered nemotron arm for the instruction-following concept.

The generator-identity twin of ``scripts/run_control_arm.py``: the SAME unsteered
one-shot generation script (``scripts/generate_instructions_dataset.py``), the same
50-row base set, the same probe, dev set, transforms and full eval splits — only the
generator model is ``nvidia/nemotron-3-ultra-550b-a55b`` instead of
``meta-llama/Llama-3.3-70B-Instruct``.

Two arms are fit so the numbers answer two different questions:

* ``nemotron200``  base 50 + all 200 nemotron rows (250 training rows) — the run the
  task asked for.
* ``nemotron150``  base 50 + the first 75 rows per label (200 training rows) — budget-
  matched to ``results_control_arm/control_instructions.json`` (base 50 + 150 llama
  rows), so a delta there reads as "this generator vs that generator" and nothing else.

Everything else is pinned to the control arm's settings: layer 32,
``linear_then_softmax``, a single probe (no ensemble), ``combine_consecutive_messages``
and ``convert_tool_to_assistant`` both on, seed 42, and ``dev_samples/instructions`` as
the fit's sole validation set.

    set -a; . ./.env; set +a
    .venv_claude/bin/python scripts/run_nemotron_instructions_arm.py
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

POS = "assistant_follows_the_instruction"
NEG = "assistant_does_not_follow_the_instruction"
DESC = (
    "A linear probe on google/gemma-3-27b-it that reads a conversation between a user "
    "and an assistant and detects whether the assistant's response follows the user's "
    "instruction."
)

BASE = REPO / "data/instructions_llama70b_50.jsonl"
GEN = REPO / "data/instructions_nemotron_200.jsonl"
EVAL_DIR = REPO / "eval_sets/instructions"
DEV_DIR = REPO / "dev_samples/instructions"
CACHE = REPO / "cache_gen_gemma27b_instructions"
OUT_DIR = REPO / "results_nemotron_arm"
PROBE_DIR = REPO / "probes/nemotron_arm"

# arm name -> generated rows per label (None = all)
ARMS = {"nemotron200": None, "nemotron150": 75}


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _build_training_file(arm: str, per_label: int | None) -> tuple[Path, int, int]:
    """Write base ∪ (a per-label prefix of) the generated rows to one JSONL."""
    base_rows = _read(BASE)
    gen_rows = _read(GEN)
    labels = {POS, NEG}
    for src, rows in (("base", base_rows), ("nemotron", gen_rows)):
        bad = [i for i, r in enumerate(rows) if r.get("labels") not in labels]
        if bad:
            raise SystemExit(f"{src}: {len(bad)} row(s) carry a label outside {labels}")

    if per_label is not None:
        kept, seen = [], {POS: 0, NEG: 0}
        for r in gen_rows:  # file order is pos-block then neg-block; count per label
            if seen[r["labels"]] < per_label:
                seen[r["labels"]] += 1
                kept.append(r)
        gen_rows = kept

    out = REPO / "data" / f"instructions_base50_plus_{arm}.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in base_rows + gen_rows) + "\n"
    )
    return out, len(base_rows), len(gen_rows)


def run(arm: str, per_label: int | None) -> dict:
    print(f"\n{'=' * 70}\n### NEMOTRON ARM — {arm}\n{'=' * 70}", flush=True)
    train_path, n_base, n_gen = _build_training_file(arm, per_label)
    print(
        f"training data: {n_base} base + {n_gen} nemotron = {n_base + n_gen} rows "
        f"-> {train_path.name}",
        flush=True,
    )

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    probe_path = PROBE_DIR / f"{arm}.pkl"

    t0 = time.monotonic()
    result = train_initial_probe(
        base_training_data_path=train_path,
        model_name=MODEL_NAME,
        layer=LAYER,
        new_probe_path=probe_path,
        pos_class_label=POS,
        neg_class_label=NEG,
        probe_description=DESC,
        probe_spec=ARCH,
        dev_data_path=DEV_DIR,
        seed=SEED,
        ensemble_size=1,
        base_activation_cache_dir=CACHE / "base_activations",
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
        verbose=True,
    )
    fit_min = (time.monotonic() - t0) / 60
    print(f"\nfit done in {fit_min:.1f} min — dev AUROC {result.dev_auroc}", flush=True)
    _free_gpu()

    df = evaluate_probe(
        probe_path,
        EVAL_DIR,
        CACHE / "eval_activations",
        max_samples=None,  # full splits — matches eval_max_samples: 0
        seed=SEED,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
    )
    _free_gpu()
    print(f"\n--- {arm} eval ---\n{df.to_string(index=False)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{arm}_eval.csv", index=False)
    rec = {
        "arm": arm,
        "generator": "nvidia/nemotron-3-ultra-550b-a55b",
        "n_base": n_base,
        "n_generated": n_gen,
        "n_training": n_base + n_gen,
        "probe": str(probe_path),
        "dev_auroc": result.dev_auroc,
        "eval": {r["dataset"]: float(r["auroc"]) for _, r in df.iterrows()},
        "fit_minutes": round(fit_min, 2),
    }
    (OUT_DIR / f"{arm}.json").write_text(json.dumps(rec, indent=2))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), action="append")
    args = ap.parse_args()

    records, failures = [], []
    for arm in args.arm or list(ARMS):
        try:
            records.append(run(arm, ARMS[arm]))
        except Exception as e:  # noqa: BLE001 — one dead arm must not lose the other
            import traceback

            traceback.print_exc()
            failures.append(f"{arm}: {type(e).__name__}: {e}")

    if records:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "nemotron_summary.json").write_text(json.dumps(records, indent=2))
        print("\n\n===== NEMOTRON ARM SUMMARY =====")
        for r in records:
            print(
                f"{r['arm']:14s} n={r['n_training']:4d}  "
                f"dev {r['dev_auroc'].get('mean', float('nan')):.4f}  "
                f"eval {r['eval'].get('mean', float('nan')):.4f}"
            )
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(" ", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
