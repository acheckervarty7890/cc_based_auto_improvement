#!/usr/bin/env python
"""Blend the ACCEPTED directions into new ones, and generate a batch under each.

The loop accepted 8 of 61 batches, and §1 of AUGMENTATION_FINDINGS.md found no
sample-level property separating them. `generate_like_accepted.py` asked whether the
accepted SAMPLES could be extended (few-shot from real pairs). This asks a different
question: do the accepted DIRECTIONS carry value that recombines? Each new direction is
a fusion of 2-3 randomly drawn accepted directions, written by the same generator model
under a dedicated fusion prompt; nothing about the eval splits, the dev set or any
measured delta enters it.

Two phases, both through `Generator` so the OpenRouter client, retry and circuit-breaker
paths are the loop's:

  1. FUSE   — `--n-directions` calls, each handed k randomly picked accepted directions,
              asked for one coherent direction in the same register. Picks are seeded
              (`--seed`), so the mixture is reproducible.
  2. GENERATE — `Generator.generate_batch` under each fused direction, `batch_size`
              samples, half per class, with the loop's own length/label/class-cap guards
              and `max_retries` in-context top-ups. `memo=""`: these directions belong to
              no iteration, so there is no rolling memo to inherit.

Novelty is seeded with every conversation the run ever generated plus the base training
data — what the loop's own BatchStore held.

Writes `{inputs, labels, family}` rows (family = `mixNN`) plus a sidecar JSON recording
each fused direction and which accepted families it came from.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"
CONFIG = REPO / "configs/gen_gemma27b_instructions_nemotron.md"
PROBE = REPO / "probes/gen_gemma27b_instructions_nemotron/probe_iter13.pkl"
BASE_DATA = REPO / "data/instructions_llama70b_50.jsonl"

ACCEPTED_KEYS = [(0, 4), (1, 1), (2, 0), (4, 3), (5, 4), (7, 2), (9, 1), (11, 3)]

FUSION_SYSTEM = """You write "directions" for batches of training data.

A direction is a 3-5 sentence brief for one batch of labelled conversations used to train a
text classifier. It names the kind of situation the conversations show, the structure they
take, and — explicitly — what makes a sample positive and what makes it negative.

You will be given several existing directions. Write ONE new direction that genuinely
COMBINES them: a single coherent task family in which the elements of every source
direction are present at once, not a list of alternatives and not a restatement of one of
them. Keep the register and shape of the sources ("<short name>: <the setup>. Positives
<...>. Negatives <...>."), and keep the positive/negative distinction as sharp as theirs.

Reply with the direction text only — no preamble, no quotes, no numbering."""


def latest_batches() -> dict[tuple[int, int], dict]:
    latest: dict[tuple[int, int], dict] = {}
    with (RUN_DIR / "batches.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            latest[(rec["iteration"], rec["batch_index"])] = rec
    return latest


SHOT_HEADER = """## Style examples already in the training set

Below are real samples from each of the source families this direction was built from —
{counts}. Write NEW conversations that match their style, length, register and structure:
these examples define what a sample of this family looks like. Do not reuse their subject
matter, and do not narrow the direction to only what they show.

{body}
"""


def pick_shots(rec: dict, n: int, pos_label: str) -> list[dict]:
    """``n`` samples from one accepted batch, alternating class so both sides show."""
    shots, used = [], set()
    for want_pos in (True, False, True, False, True, False):
        if len(shots) >= n:
            break
        for i, sm in enumerate(rec["samples"]):
            if i in used:
                continue
            if (sm["label"] == pos_label) == want_pos:
                shots.append(sm)
                used.add(i)
                break
    # A family that holds only one class (it2b0) simply contributes what it has.
    for i, sm in enumerate(rec["samples"]):
        if len(shots) >= n:
            break
        if i not in used:
            shots.append(sm)
            used.add(i)
    return shots[:n]


def direction_with_shots(d: dict, latest: dict, keys: dict, n_per_source: int, pos_label: str) -> str:
    """The fused direction plus ``n_per_source`` real samples from EACH source family.

    The examples ride inside the direction slot of ``_batch_request`` so the rest of the
    generation path — guards, class caps, top-up asks — stays exactly the loop's.
    """
    blocks, counts = [], []
    for fam in d["sources"]:
        rec = latest[keys[fam]]
        shots = pick_shots(rec, n_per_source, pos_label)
        counts.append(f"{len(shots)} from {fam}")
        body = json.dumps([{"messages": sm["messages"], "label": sm["label"]} for sm in shots],
                          indent=2, ensure_ascii=False)
        blocks.append(f"### From {fam}\n{body}")
    return d["direction"] + "\n\n" + SHOT_HEADER.format(
        counts=", ".join(counts), body="\n\n".join(blocks)
    )


def fusion_request(sources: list[tuple[str, str]]) -> str:
    body = "\n\n".join(f"SOURCE DIRECTION {i + 1}\n{d}" for i, (_f, d) in enumerate(sources))
    return (
        f"{body}\n\nWrite one new direction that combines all {len(sources)} of the above "
        "into a single task family. Every source's setup and its positive/negative "
        "distinction must be recognisably present in the result."
    )


async def main_async(args) -> None:
    from agentic_redteam.config import load_config
    from agentic_redteam.generator import Generator, ProbeMeta
    from agentic_redteam.persistence import BatchStore, Conversation
    from agentic_redteam.retrain import read_probe_metadata
    from agentic_redteam.token_budget import TokenBudget

    config = load_config(CONFIG)
    info = read_probe_metadata(PROBE)
    probe = ProbeMeta(
        pos_class_label=info["pos_class_label"],
        neg_class_label=info["neg_class_label"],
        description=info["description"],
        model_name=info["model_name"],
    )
    print(f"classes: {probe.pos_class_label} / {probe.neg_class_label}")
    print(f"generator: {config.generator.model_names}, batch_size {config.generator.batch_size}, "
          f"max_retries {config.generator.max_retries}, max_tokens {config.generator.max_tokens}")

    budget = TokenBudget(
        model_name=probe.model_name,
        max_tokens=config.generator.max_sample_tokens,
        combine_consecutive_messages=config.eval.combine_consecutive_messages,
        convert_tool_to_assistant=config.eval.convert_tool_to_assistant,
    )
    generator = Generator(config=config.generator, probe=probe, token_budget=budget)
    generator.warmup()

    latest = latest_batches()
    accepted = [(f"it{it}b{bk}", " ".join(latest[(it, bk)]["direction"].split()))
                for it, bk in ACCEPTED_KEYS]

    # ---- phase 1: fuse (or reload a previous fusion) ----
    if args.use_saved_directions:
        directions = json.loads(args.directions_out.read_text())
        print(f"reusing {len(directions)} saved direction(s) from {args.directions_out}")
        await _generate(args, config, generator, probe, latest, directions)
        return

    rng = random.Random(args.seed)
    picks = []
    for _ in range(args.n_directions):
        k = rng.choice(args.mix_sizes)
        picks.append(rng.sample(accepted, k))

    model = config.generator.models[0]
    sem = asyncio.Semaphore(args.concurrency)

    async def fuse(idx: int, sources: list[tuple[str, str]]) -> tuple[int, str]:
        async with sem:
            names = "+".join(f for f, _d in sources)
            print(f"  [mix{idx:02d}] fusing {names} ...", flush=True)
            text = await generator.call(
                model, FUSION_SYSTEM,
                [{"role": "user", "content": fusion_request(sources)}],
            )
        return idx, " ".join((text or "").split())

    fused = dict(await asyncio.gather(*(fuse(i, s) for i, s in enumerate(picks))))
    directions = []
    for i, sources in enumerate(picks):
        text = fused.get(i, "")
        if not text:
            print(f"  [mix{i:02d}] FUSION FAILED — skipped")
            continue
        directions.append({
            "family": f"mix{i:02d}",
            "sources": [f for f, _d in sources],
            "direction": text,
        })
    args.directions_out.parent.mkdir(parents=True, exist_ok=True)
    args.directions_out.write_text(json.dumps(directions, indent=2, ensure_ascii=False))
    print(f"\nfused {len(directions)} direction(s) -> {args.directions_out}\n")
    for d in directions:
        print(f"--- {d['family']}  ({'+'.join(d['sources'])})\n{d['direction']}\n")

    if args.directions_only:
        return

    await _generate(args, config, generator, probe, latest, directions)


async def _generate(args, config, generator, probe, latest, directions) -> None:
    import asyncio
    import json
    from agentic_redteam.persistence import BatchStore, Conversation

    if args.only:
        keep = set(args.only.split(","))
        directions = [d for d in directions if d["family"] in keep]
        print(f"restricted to {[d['family'] for d in directions]}")
    sem = asyncio.Semaphore(args.concurrency)

    # ---- phase 2: generate ----
    store = BatchStore(args.batches_out)
    store.forget_loaded()
    for rec in latest.values():
        for s in rec["samples"]:
            store.seen_keys.add(Conversation.from_messages(s["messages"]).to_canonical_text())
    n_run = len(store.seen_keys)
    with BASE_DATA.open() as fh:
        for line in fh:
            msgs = json.loads(line)["inputs"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            store.seen_keys.add(Conversation.from_messages(msgs).to_canonical_text())
    if args.append and args.out.exists():
        n_before = len(store.seen_keys)
        with args.out.open() as fh:
            for line in fh:
                store.seen_keys.add(
                    Conversation.from_messages(json.loads(line)["inputs"]).to_canonical_text()
                )
        print(f"novelty also seeded with {len(store.seen_keys) - n_before} rows already in {args.out.name}")
    print(f"novelty guard: {len(store.seen_keys)} conversations")

    out_rows: list[dict] = []

    keys = {f"it{it}b{bk}": (it, bk) for it, bk in ACCEPTED_KEYS}

    async def one(idx: int, d: dict) -> None:
        fam = d["family"]
        direction = d["direction"]
        if args.shots_per_source > 0:
            direction = direction_with_shots(
                d, latest, keys, args.shots_per_source, probe.pos_class_label
            )
        async with sem:
            print(f"  [{fam}] generating {config.generator.batch_size}"
                  + (f" with {args.shots_per_source}x{len(d['sources'])} shots" if args.shots_per_source else "")
                  + " ...", flush=True)
            gen = await generator.generate_batch(
                batch_index=idx,
                n_batches=len(directions),
                direction=direction,
                memo="",
                store=store,
                iteration=0,
            )
        npos = gen.count(probe.pos_class_label)
        print(f"  [{fam}] kept {len(gen.samples)} ({npos} pos / {len(gen.samples) - npos} neg) "
              f"in {gen.n_calls} call(s); dropped long={gen.n_dropped_too_long} "
              f"dup={gen.n_dropped_duplicate} bad={gen.n_dropped_bad_label}"
              + (f"; ERROR {gen.error}" if gen.error else ""), flush=True)
        for s in gen.samples:
            row = s.to_training_row()
            row["family"] = fam
            out_rows.append(row)

    if args.dump_prompt:
        from agentic_redteam.generator import _batch_request
        d = next(x for x in directions if x["family"] == args.dump_prompt)
        direction = direction_with_shots(d, latest, keys, args.shots_per_source, probe.pos_class_label) \
            if args.shots_per_source > 0 else d["direction"]
        idx = [x["family"] for x in directions].index(args.dump_prompt)
        print(_batch_request(probe, direction, config.generator.batch_size, idx, len(directions)))
        return

    await asyncio.gather(*(one(i, d) for i, d in enumerate(directions)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a" if args.append else "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    npos = sum(1 for r in out_rows if r["labels"] == probe.pos_class_label)
    print(f"\n{'appended' if args.append else 'wrote'} {len(out_rows)} rows ({npos} pos / {len(out_rows) - npos} neg) to {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-directions", type=int, default=10)
    ap.add_argument("--mix-sizes", type=int, nargs="+", default=[2, 3],
                    help="how many accepted directions each mixture draws from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--directions-only", action="store_true")
    ap.add_argument("--use-saved-directions", action="store_true",
                    help="skip the fusion phase and reload --directions-out")
    ap.add_argument("--only", default=None, help="comma-separated families to (re)generate")
    ap.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    ap.add_argument("--shots-per-source", type=int, default=0,
                    help="real samples shown from EACH source family (0 = no few-shot)")
    ap.add_argument("--dump-prompt", default=None, help="print the request for this family and exit")
    ap.add_argument("--out", type=Path, default=REPO / "data/instructions_mixed_directions.jsonl")
    ap.add_argument("--directions-out", type=Path, default=RUN_DIR / "mixed_directions.json")
    ap.add_argument("--batches-out", type=Path, default=RUN_DIR / "mixed_direction_batches.jsonl")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
