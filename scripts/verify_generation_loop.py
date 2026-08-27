#!/usr/bin/env python
"""End-to-end check of the generate → score → retrain → guide loop with FAKE LLMs.

Two modes:

- ``--mode fake`` (default): the generator and judge are faked AND so is every
  training/scoring call, so the run needs no GPU, no model and no API key. It checks
  the loop's bookkeeping: n batches per iteration, acceptance from ΔAUROC, the union
  retrain, guidance written for the next iteration, the ledgers, and that a resumed
  run picks up without regenerating or re-scoring anything.
- ``--mode real``: the generator and judge are still faked, but the probes are really
  trained on ``llama-1b`` from a 50-row base set and a small cut of the high-stakes
  dev set, so the retrain/score/cache-warming path runs for real (a few minutes on a
  laptop GPU).

Run from the repo root:

    .venv_claude/bin/python scripts/verify_generation_loop.py --mode fake
    .venv_claude/bin/python scripts/verify_generation_loop.py --mode real
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentic_redteam import cli  # noqa: E402
from agentic_redteam import generator as G  # noqa: E402
from agentic_redteam import llm_judge as J  # noqa: E402
from agentic_redteam.persistence import BatchStore, GuidanceStore  # noqa: E402
from agentic_redteam.retrain import RetrainResult  # noqa: E402

POS, NEG = "high-stakes", "low-stakes"

CONFIG_TEMPLATE = """---
generator:
  provider: openrouter
  models: [fake/gen-a, fake/gen-b]
  n_batches: {n_batches}
  batch_size: {batch_size}
  concurrency: 2
  max_tokens: 2048
  max_sample_tokens: {max_sample_tokens}
  max_retries: 2
judge:
  provider: openrouter
  model: fake/judge
  max_tokens: 1024
  memo_word_budget: 100
  max_samples_per_batch: 2
probe:
  model: meta-llama/Llama-3.2-1B-Instruct
  layer: 8
  pos_class_label: {pos}
  neg_class_label: {neg}
  description: whether the situation in the conversation carries serious consequences
  architecture: linear_then_softmax
loop:
  iterations: {iterations}
  min_auroc_gain: 0.0
  exhausted_gain: 0.002
validation:
  dev_data: {dev_dir}
output:
  run_dir: {run_dir}
  run_id: verify
---

# Generator

You write training data.

# Judge

You analyze batches.
"""


class FakeCalls:
    """Counters shared by the fake generator/judge so the test can assert on them."""

    generator = 0
    judge = 0
    counter = 0


def _fake_samples(direction: str, n_pos: int, n_neg: int) -> list[dict]:
    out = []
    for label, n in ((POS, n_pos), (NEG, n_neg)):
        for _ in range(n):
            FakeCalls.counter += 1
            k = FakeCalls.counter
            tag = re.sub(r"[^a-z]+", " ", direction.lower())[:40].strip()
            if label == POS:
                user = f"[{k}] I'm handling a {tag} situation and a wrong move could cost someone their job or health. What should I do first?"
                asst = f"Given how much rides on this ({k}), slow down: confirm the facts, loop in whoever is responsible, and document each step."
            else:
                user = f"[{k}] Just curious about {tag} things — any fun trivia or a light recommendation for the weekend?"
                asst = f"Sure ({k})! Nothing serious here — here's a relaxed suggestion and a bit of trivia to enjoy."
            out.append({"label": label, "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]})
    return out


async def fake_generator_call(self, model, system, messages):
    FakeCalls.generator += 1
    last = messages[-1]["content"]
    if last.startswith("Before any samples are written, propose"):
        n = int(re.search(r"propose (\d+) distinct directions", last).group(1))
        return "```json\n" + json.dumps([f"proposed direction {i} for {model.name}" for i in range(n)]) + "\n```"
    m = re.search(r"exactly (\d+) labelled '([^']+)' and (\d+) labelled '([^']+)'", last)
    if m:
        n_pos, n_neg = int(m.group(1)), int(m.group(3))
        direction = re.search(r"## Direction for this batch\n(.*?)\n\n", last, re.S).group(1)
        # Under-deliver one negative on the first call so the top-up path is exercised.
        samples = _fake_samples(direction, n_pos, max(n_neg - 1, 0))
        return "Here is the batch:\n```json\n" + json.dumps(samples) + "\n```"
    m = re.search(r"Write (?:(\d+) more labelled '([^']+)')?(?: and )?(?:(\d+) more labelled '([^']+)')?", last)
    need = {POS: 0, NEG: 0}
    for count, label in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
        if count:
            need[label] = int(count)
    direction = "top-up"
    return "```json\n" + json.dumps(_fake_samples(direction, need[POS], need[NEG])) + "\n```"


def fake_judge_call(self, system, messages):
    FakeCalls.judge += 1
    n = int(re.search(r"exactly (\d+) strings", system).group(1))
    user = messages[-1]["content"]
    assert "## Batches from this round" in user and "Δ mean AUROC" in user, user[:500]
    it = re.search(r"Round just finished: (\d+)", user).group(1)
    memo = f"- round {it}: accepted batches taught something\n- exhausted: repeats of round {it}"
    directions = [f"judge direction {i} after round {it}" for i in range(n)]
    return f"## Memo\n{memo}\n\n## Directions\n```json\n{json.dumps(directions)}\n```"


# ---------------------------------------------------------------- fake fits


def _fake_scores(seed_text: str, base: float) -> dict[str, float]:
    h = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    rng = random.Random(h)
    a = base + rng.uniform(-0.01, 0.03)
    b = base + rng.uniform(-0.02, 0.02)
    return {"split_a": a, "split_b": b, "mean": (a + b) / 2}


class _FakeProbe:
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    layer = 8
    pos_class_label = POS
    neg_class_label = NEG
    description = "fake"

    def __init__(self, scores):
        self.scores = scores


def install_fake_fits():
    def train_initial_probe(**kw):
        scores = {"split_a": 0.70, "split_b": 0.72, "mean": 0.71}
        Path(kw["new_probe_path"]).parent.mkdir(parents=True, exist_ok=True)
        with open(kw["new_probe_path"], "wb") as f:
            pickle.dump(_FakeProbe(scores), f)
        return RetrainResult(Path(kw["new_probe_path"]), 0, 50, 1, scores)

    def retrain_probe(*, samples, base_probe_path, new_probe_path, **kw):
        with open(base_probe_path, "rb") as f:
            base = pickle.load(f).scores["mean"]
        key = "|".join(sorted(s.key for s in samples))
        scores = _fake_scores(key, base)
        Path(new_probe_path).parent.mkdir(parents=True, exist_ok=True)
        with open(new_probe_path, "wb") as f:
            pickle.dump(_FakeProbe(scores), f)
        return RetrainResult(Path(new_probe_path), len(samples), 50 + len(samples), 1, scores)

    def score_probe_on_dev(probe_path, *a, **kw):
        with open(probe_path, "rb") as f:
            return pickle.load(f).scores

    def read_probe_metadata(path):
        return {"pos_class_label": POS, "neg_class_label": NEG, "description": "fake",
                "model_name": "meta-llama/Llama-3.2-1B-Instruct", "layer": 8, "ensemble_size": 1}

    cli.train_initial_probe = train_initial_probe
    cli.retrain_probe = retrain_probe
    cli.score_probe_on_dev = score_probe_on_dev
    cli.read_probe_metadata = read_probe_metadata
    cli.warm_sample_activation_cache = lambda *a, **k: 0
    cli.TokenBudget = lambda **kw: None  # no tokenizer in fake mode


# ---------------------------------------------------------------- harness


def _write_dev_subset(dev_dir: Path, rows_per_split: int, splits=("mts_balanced", "toolace_balanced")) -> None:
    dev_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        rows = [json.loads(l) for l in (REPO / "dev_samples/highstakes" / f"{split}.jsonl").open()]
        by = {POS: [r for r in rows if r["labels"] == POS], NEG: [r for r in rows if r["labels"] == NEG]}
        keep = by[POS][: rows_per_split // 2] + by[NEG][: rows_per_split // 2]
        with (dev_dir / f"{split}.jsonl").open("w") as f:
            for r in keep:
                f.write(json.dumps({"inputs": r["inputs"], "labels": r["labels"]}) + "\n")


def run(mode: str, workdir: Path, iterations: int, n_batches: int, batch_size: int, resume: bool) -> None:
    run_dir = workdir / "run"
    probe_dir = workdir / "probes"
    dev_dir = workdir / "dev"
    if not resume:
        for d in (run_dir, probe_dir):
            shutil.rmtree(d, ignore_errors=True)
        if mode == "real":
            _write_dev_subset(dev_dir, rows_per_split=20)
        else:
            dev_dir.mkdir(parents=True, exist_ok=True)
            (dev_dir / "split_a.jsonl").write_text("")
    config_path = workdir / "config.md"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            n_batches=n_batches, batch_size=batch_size, iterations=iterations, pos=POS, neg=NEG,
            dev_dir=dev_dir, run_dir=run_dir, max_sample_tokens=1024 if mode == "real" else 0,
        )
    )
    base = REPO / "data/highstakes_llama70b_50.jsonl"
    argv = [str(config_path), "--base-training-data", str(base), "--probe-out-dir", str(probe_dir),
            "--resume" if resume else "--no-resume"]
    rc = cli.iterative_generate_main(argv)
    assert rc == 0, rc


def check(workdir: Path, iterations: int, n_batches: int, batch_size: int) -> None:
    run_dir = workdir / "run"
    probe_dir = workdir / "probes"
    store = BatchStore(run_dir / "batches.jsonl")
    guidance = GuidanceStore(run_dir / "guidance.jsonl")
    for i in range(iterations):
        recs = {r.batch_index: r for r in store.for_iteration(i)}
        assert set(recs) == set(range(n_batches)), (i, sorted(recs))
        for k, r in recs.items():
            assert r.status == "scored", (i, k, r.status)
            assert r.n_samples == batch_size, (i, k, r.n_samples)
            assert set(r.n_per_label.values()) == {batch_size // 2}, r.n_per_label
            assert r.accepted == (r.delta > 0.0), (r.delta, r.accepted)
            assert r.exhausted == (abs(r.delta) <= 0.002)
            assert "mean" in r.auroc_before and "mean" in r.auroc_after
        g = guidance.for_iteration(i)
        assert g is not None and len(g.directions) == n_batches, (i, g)
        assert g.source == ("generator_proposal" if i == 0 else "judge"), (i, g.source)
        assert (probe_dir / f"probe_iter{i + 1}.pkl").exists()
        assert (run_dir / f"accepted_iter{i + 1}.jsonl").exists()
        # A batch's candidate probe is trained on base ∪ accepted-before ∪ batch: the
        # ledger's baseline for iteration i must equal the previous iteration's outcome.
        if i > 0:
            prev = guidance.for_iteration(i)
            assert prev.baseline_auroc["mean"] == recs[0].auroc_before["mean"]
    g_next = guidance.for_iteration(iterations)
    assert g_next is not None and g_next.source == "judge" and "round" in g_next.memo
    assert (run_dir / "auroc_history.csv").exists()
    n_rows = sum(1 for _ in (run_dir / "auroc_history.csv").open()) - 1
    assert n_rows == iterations * n_batches, n_rows
    # Every sample across the run is unique (novelty guard).
    newest = {}
    for r in store.records:
        newest[(r.iteration, r.batch_index)] = r
    keys = [s.key for r in newest.values() for s in r.samples]
    assert len(keys) == len(set(keys)), "duplicate samples across batches"
    print(f"  OK: {iterations} iterations × {n_batches} batches × {batch_size} samples; "
          f"{sum(1 for r in store.records if r.status == 'scored' and r.accepted)} accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fake", "real"), default="fake")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--n-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    G.Generator.call = fake_generator_call
    J.LLMJudge._call_provider = fake_judge_call
    J.LLMJudge.warmup = lambda self: None
    G.Generator.warmup = lambda self: None
    if args.mode == "fake":
        install_fake_fits()

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="verify_generation_loop_", dir=os.environ.get("TMPDIR")))
    print(f"[{args.mode}] workdir {workdir}")

    print("== fresh run ==")
    run(args.mode, workdir, args.iterations, args.n_batches, args.batch_size, resume=False)
    check(workdir, args.iterations, args.n_batches, args.batch_size)

    print("== resume with one more iteration ==")
    gen_before, judge_before = FakeCalls.generator, FakeCalls.judge
    run(args.mode, workdir, args.iterations + 1, args.n_batches, args.batch_size, resume=True)
    check(workdir, args.iterations + 1, args.n_batches, args.batch_size)
    new_gen = FakeCalls.generator - gen_before
    # The resumed iteration reuses the judge's stored directions (no proposal call) and
    # generates only its own batches: one call per batch plus one top-up each.
    assert new_gen == args.n_batches * 2, new_gen
    assert FakeCalls.judge - judge_before == 1, FakeCalls.judge - judge_before
    print(f"  OK: resume generated {new_gen} calls for {args.n_batches} batches, 1 judge call")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
