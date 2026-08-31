#!/usr/bin/env python
"""Fit `base ∪ <jsonl>` for one concept and score dev + eval.

Generalizes `scripts/fit_base_plus.py` (which is pinned to the instructions concept and
its nemotron arm) over the concepts in `CONCEPTS`. Every knob per concept is copied from
that concept's `configs/gen_gemma27b_<concept>.md`, so a fit here is apples-to-apples
with that arm's `probe_iter*.pkl`:

    probe      google/gemma-3-27b-it, layer 32, linear_then_softmax, single (no ensemble)
    dev        the concept's dev_samples/ dir, used whole
    eval       the concept's eval_sets/ dir, FULL splits (no subsampling)
    transforms combine_consecutive_messages = convert_tool_to_assistant = True
    seed       42

`--base-only` skips the fit and scores the concept's `probe_iter0.pkl` — the probe
trained on the 50 base rows alone — as the reference point every generated set is
measured against.

Measured, hu_harm:

    base only (50 rows, probe_iter0)          dev 0.87589   eval 0.85232
    base ∪ hu_harm_gptoss_600 (650 rows)      dev 0.89724   eval 0.87323
    base ∪ hu_harm_deepseekv4pro_600 (650)    dev 0.93052   eval 0.90852

Both sets were written by the same script, same prompt, same 300/300 split — only the
generator differs — so the ~0.035 eval gap between them is the generator's. Read the
per-split numbers before reading the mean: nearly all of both gains sit on
ai_dilemmas and balanced_refusal, and BOTH sets leave eval_ant_hh flat-to-worse
(0.737 base → 0.721 gptoss, 0.729 deepseek). ant_hh is the one eval split that is not
class-paired, and the script's one-shot pair makes the harmful class casual/dismissive
and the safe class careful — a surface cue the paired splits cannot reward but ant_hh
does not supply either.

Measured, highstakes:

    base only (50 rows, probe_iter0)             dev 0.89324   eval 0.89974
    base ∪ highstakes_gptoss_600 (650 rows)      dev 0.93685   eval 0.92776
    base ∪ highstakes_deepseekv4pro_600 (650)    dev 0.93629   eval 0.93276

This concept starts far higher than hu_harm (eval 0.89974 vs 0.85232), so there is much
less headroom for a generated set to claim — and the two generators land much closer
together here (0.005 apart on eval) than on hu_harm (0.035). Note the dev means are
effectively tied (0.93685 vs 0.93629) while eval separates them, so on this concept dev
does not rank the two generators.

toolace_balanced is the split that resists, and it resists BOTH sets — 0.85566 base →
0.84417 gptoss → 0.83051 deepseek, i.e. it gets monotonically worse as the other three
improve, and deepseek (the better set overall) hurts it most. The same shape as hu_harm's
eval_ant_hh, one split down while the rest move.

Cost note: a highstakes fit is ~20x a hu_harm one on the SAME 650 training rows, and it
is not the fit that is bigger. The 19.6 GiB dev set takes the whole card, so
`_to_device_for_fit` leaves the 1.1 GiB training set host-resident and every epoch pays a
scattered CPU gather + H2D on it: 4.5 ms/sample against hu_harm's 0.19 ms/sample, where
both sets are staged (1.5 GiB total). That matches the table under `retrain.py`
(0.16 ms/sample GPU-resident, 18.35 ms/sample host-resident at 11 MB rows; the highstakes
training rows are ~1.8 MB, and 18.35 × 1.8/11 ≈ 3.0). `_to_device_for_fit` sorts by size
and is right to — 19.6 GiB moves more bytes than 1.1 GiB — but here the big set is the one
read once per epoch under no_grad while the small one carries forward+backward.

Measured, instructions — the concept where this DOES NOT WORK:

    base only (50 rows, probe_iter0)             dev 0.75728   eval 0.77787
    base ∪ instructions_gptoss_600 (650 rows)    dev 0.60864   eval 0.66488
    base ∪ instructions_nemotron_600 (650)       dev 0.69101   eval 0.76176

Both generated sets make the probe WORSE, gpt-oss by 0.113 eval and nemotron by 0.016.
The labels are not the problem — spot-checked, the negatives are genuine violations. It
is a distribution mismatch: the generated rows are short synthetic format-compliance
tasks ("list three colors, comma-separated"), where not-following means a few extra
words or wrong spacing, while the eval splits test refusal, context drift across turns,
contradiction of a provided source, omission and answer substitution. The probe learns
"terse exact-format reply vs chatty reply", which anti-correlates with the eval concept
— bbq_substitution lands at 0.355/0.371 dev, BELOW chance, for both generators.

So a generated set helping is a property of the concept, not of the method: the same
script and the same two-turn shape gained 0.021-0.056 on hu_harm and 0.028-0.033 on
highstakes. Where the generator can only reach a different region of the input space
than the eval splits occupy, more data is worse than none.

The eval and dev activations come from Kaggle (`prefetch_*`, the `kaggle:` block of the
concept's config), so neither is ever extracted locally. Only the generated samples and
the 50 base rows go through the 27B model, once each, into the shared per-sample cache.

Examples:
    ${REPO_ROOT}/.venv_claude/bin/python scripts/fit_base_plus_concept.py \\
        --concept hu_harm --base-only
    ${REPO_ROOT}/.venv_claude/bin/python scripts/fit_base_plus_concept.py \\
        --concept highstakes data/highstakes_gptoss_600.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
SEED, COMBINE, CONVERT = 42, True, True

# The kaggle: block every gemma-3-27b config in this repo carries, verbatim. {slug} in a
# dataset slug is mandatory (Kaggle rejects underscores, and every split stem has one);
# {split} names the file inside the dataset, which is unrestricted.
KAGGLE_OWNER = "anku7890"
EVAL_SLUG, EVAL_FILE = "{slug}-gemmaevalpt", "{split}-gemmaeval.pt"
DEV_SLUG, DEV_FILE = "{slug}-gemmadevpt", "{split}-gemmadev.pt"


@dataclass(frozen=True)
class Concept:
    """Everything that differs between concepts. Paths are relative to the repo root."""

    name: str
    probe_dir: Path
    base_data: Path
    dev_data: Path
    eval_dir: Path
    cache_dir: Path
    pos_label: str
    neg_label: str

    @property
    def base_probe(self) -> Path:
        return self.probe_dir / "probe_iter0.pkl"

    @property
    def base_cache(self) -> Path:
        return self.cache_dir / "base_activations"

    @property
    def eval_cache(self) -> Path:
        return self.cache_dir / "eval_activations"


CONCEPTS = {
    c.name: c
    for c in [
        Concept(
            name="hu_harm",
            probe_dir=REPO / "probes/gen_gemma27b_hu_harm",
            base_data=REPO / "data/hu_harm_llama70b_50.jsonl",
            dev_data=REPO / "dev_samples/hu_ha",
            eval_dir=REPO / "eval_sets/hu_ha",
            cache_dir=REPO / "cache_gen_gemma27b_hu_harm",
            pos_label="harmful_to_human",
            neg_label="not_harmful_to_human",
        ),
        Concept(
            name="highstakes",
            probe_dir=REPO / "probes/gen_gemma27b_highstakes",
            base_data=REPO / "data/highstakes_llama70b_50.jsonl",
            dev_data=REPO / "dev_samples/highstakes",
            eval_dir=REPO / "eval_sets/highstakes",
            cache_dir=REPO / "cache_gen_gemma27b_highstakes",
            pos_label="high-stakes",
            neg_label="low-stakes",
        ),
        Concept(
            name="instructions",
            probe_dir=REPO / "probes/gen_gemma27b_instructions",
            base_data=REPO / "data/instructions_llama70b_50.jsonl",
            dev_data=REPO / "dev_samples/instructions",
            eval_dir=REPO / "eval_sets/instructions",
            cache_dir=REPO / "cache_gen_gemma27b_instructions",
            pos_label="assistant_follows_the_instruction",
            neg_label="assistant_does_not_follow_the_instruction",
        ),
    ]
}


def eval_source():
    from agentic_redteam.kaggle_activations import KaggleActivationSource

    return KaggleActivationSource(KAGGLE_OWNER, EVAL_SLUG, EVAL_FILE)


def prefetch_dev(concept: Concept) -> None:
    """Assemble the dev blob from Kaggle into the exact path the fit looks for."""
    from agentic_redteam.kaggle_activations import (
        KaggleActivationSource,
        prefetch_dev_activations,
    )
    from agentic_redteam.retrain import _dev_activation_cache_path

    dev_files = sorted(concept.dev_data.glob("*.jsonl"))
    if not dev_files:
        raise SystemExit(f"{concept.dev_data} holds no *.jsonl splits")
    concept.base_cache.mkdir(parents=True, exist_ok=True)
    prefetch_dev_activations(
        _dev_activation_cache_path(
            concept.base_cache, dev_files, MODEL_NAME, LAYER, COMBINE, CONVERT
        ),
        dev_files,
        KaggleActivationSource(KAGGLE_OWNER, DEV_SLUG, DEV_FILE),
        model_name=MODEL_NAME,
        layer=LAYER,
        verbose=True,
    )


def load_rows(path: Path, concept: Concept) -> list[dict]:
    """Read a `{inputs, labels}` JSONL into the shape ``retrain_probe`` wants.

    On disk `inputs` is a JSON-encoded string (the tuberlens LabelledDataset schema every
    file under data/ and eval_sets/ uses). The in-memory `samples` path of
    ``retrain_probe`` wants it already parsed — ``_dicts_to_labelled_dataset`` calls
    ``m.get("role")`` on each message — so decode it here. Rows that already carry a list
    (a ``_dump_labelled_dataset`` snapshot, e.g. accepted_iter*.jsonl) pass through.
    """
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "inputs": json.loads(r["inputs"]) if isinstance(r["inputs"], str) else r["inputs"],
            "labels": r["labels"],
        })
    unknown = sum(1 for r in rows if r["labels"] not in (concept.pos_label, concept.neg_label))
    if unknown:
        raise SystemExit(f"{path}: {unknown} rows carry a label that is neither class")
    return rows


def report(name: str, n_rows: int | None, dev: dict[str, float] | None, df) -> float:
    ev = float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])
    print(f"\n=== {name}" + (f"  ({n_rows} training rows)" if n_rows else "") + " ===")
    if dev is not None:
        print(f"  dev  mean {dev['mean']:.5f}")
        for k, v in sorted(dev.items()):
            if k != "mean":
                print(f"      {k:<28} {v:.5f}")
    print(f"  eval mean {ev:.5f}")
    for _, r in df[df["dataset"] != "mean"].iterrows():
        print(f"      {r['dataset']:<28} {r['auroc']:.5f}")
    return ev


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    ap.add_argument("samples", type=Path, nargs="?", help="JSONL of {inputs, labels} rows")
    ap.add_argument("--base-only", action="store_true",
                    help="score probe_iter0 (base 50 rows) — the reference point")
    ap.add_argument("--no-base", action="store_true", help="fit the samples alone")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-prefetch", action="store_true",
                    help="assume the eval/dev activation caches are already populated")
    args = ap.parse_args()

    concept = CONCEPTS[args.concept]
    if not args.base_only and args.samples is None:
        ap.error("give a samples JSONL, or --base-only")

    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import (
        retrain_probe,
        score_probe_on_dev,
        warm_sample_activation_cache,
    )

    if not args.skip_prefetch:
        prefetch_dev(concept)

    if args.base_only:
        dev = score_probe_on_dev(
            concept.base_probe, concept.dev_data, concept.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        df = evaluate_probe(
            concept.base_probe, concept.eval_dir, concept.eval_cache, max_samples=None,
            seed=SEED, combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT, kaggle_source=eval_source(),
        )
        report(f"{concept.name}: base only ({concept.base_probe.name})", 50, dev, df)
        return

    rows = load_rows(args.samples, concept)
    npos = sum(1 for r in rows if r["labels"] == concept.pos_label)
    prefix = "no base ∪ " if args.no_base else "base ∪ "
    print(f"{prefix}{len(rows)} ({args.samples.name}): "
          f"{npos} {concept.pos_label} / {len(rows) - npos} {concept.neg_label}")

    out = args.out or concept.probe_dir / f"baseplus_{args.samples.stem}.pkl"
    warm_sample_activation_cache(
        rows, base_probe_path=concept.base_probe,
        base_activation_cache_dir=concept.base_cache,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
        verbose=True,
    )
    res = retrain_probe(
        samples=rows, base_probe_path=concept.base_probe,
        base_training_data_path=None if args.no_base else concept.base_data,
        new_probe_path=out, dev_data_path=concept.dev_data, seed=SEED,
        base_data_fraction=1.0, base_activation_cache_dir=concept.base_cache,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
        verbose=True,
    )
    df = evaluate_probe(
        out, concept.eval_dir, concept.eval_cache, max_samples=None, seed=SEED,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
        kaggle_source=eval_source(),
    )
    report(f"{concept.name}: {prefix}{args.samples.name}",
           res.n_training_samples_total, res.dev_auroc, df)


if __name__ == "__main__":
    main()
