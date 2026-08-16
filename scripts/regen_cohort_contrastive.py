"""Regenerate contrastive counterparts for the vintage-3-minus-vintage-2 cohort.

The generation prompt changed (``GENERATION_PROMPT_VERSION`` v1 → v2): it used to ask
for a "similar-looking conversation ... that belongs to the <target> class", and now asks
for the **minimal edit** to the original that flips the label. v1's phrasing let the
generator write a fresh unambiguous exemplar of the target class rather than a near
neighbour of the source — measured on these arms, ``probe_iter2`` scores the v1
counterparts at ~0.99+ toward their own label, so a pair's whole contrast sat on the
attacker-written side.

This script re-runs generation for the cohort the transfer analysis is about — the 157
attacker-written successes that are in vintage 3 and not vintage 2 — under the new
prompt, and reports whether the counterparts actually came out closer to their sources.
Nothing is trained or scored here; it produces pairs and a comparison.

What it feeds the LLM, and why that is the raw conversation
-----------------------------------------------------------
``retrain._build_redteam_dataset`` runs ``generate_contrastive_dataset`` on the
**untransformed** success messages and applies ``convert_tool_to_assistant`` /
``combine_consecutive_messages`` only afterwards. So the contrastive cache is keyed on
raw messages, while ``redteam_postprocessed_iter3.jsonl`` — which is how a vintage is
defined — holds transformed ones. This script therefore selects the cohort from the dump
and then maps each row back to its raw messages in the attempt log, so the pairs it
writes are keyed exactly as a real retrain would key them and its cache is a drop-in.

It calls ``generate_contrastive_dataset`` itself rather than reimplementing the loop, so
the prompt, retry policy, well-formedness rejection, circuit-breaker reporting and cache
format are the production ones — what this measures is what a retrain would produce.

Safety of the output
--------------------
The new pairs go to a **separate** cache file, not the arm's
``contrastive_cache.jsonl``. Merging them into the arm's cache would change what the next
retrain of that arm trains on; ``--write-arm-cache`` does it explicitly if that is wanted.
(The v2 key differs from v1's, so even then nothing already cached is overwritten — the
old pairs simply stop being reachable under the new prompt version.)

Usage:
    # look at the exact prompts first, no API calls, no spend
    .venv_claude/bin/python scripts/regen_cohort_contrastive.py --dry-run --limit 3

    # a small real sample
    .venv_claude/bin/python scripts/regen_cohort_contrastive.py --limit 10

    # the whole cohort (157 pairs across both arms)
    .venv_claude/bin/python scripts/regen_cohort_contrastive.py
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
from attribution_vintage import dropped_rows, vintages

ARM_CONFIGS = {
    "gptoss120b": A.REPO / "configs/gptoss120b_hs_gemma27b_batch.md",
    "deepseekv4pro": A.REPO / "configs/deepseekv4pro_hs_gemma27b_batch.md",
}
# Attempt logs, per arm; both error types write their own file.
REDTEAM_LOGS = {
    "gptoss120b": A.REPO / "results_hs_gemma27b_gptoss120b_batch/gptoss120b_probing",
    "deepseekv4pro": A.REPO / "results_hs_gemma27b_deepseekv4pro_batch/deepseekv4pro_probing",
}


# --- selecting the cohort ----------------------------------------------------------


def cohort_source_keys(arm: str, iteration: int, drop_mode: str) -> tuple[set[str], dict]:
    """Canonical (transformed) text of every attacker-written source in v3 \\ v2.

    Sources only: the generated halves are the thing being *replaced*, so feeding them
    back in would ask the LLM to write a counterpart to a counterpart.
    """
    ds = A.load_redteam_dataset(arm, iteration)
    gen2src = A.generated_to_source(arm)
    exclude, _ = dropped_rows(arm, iteration, drop_mode)
    keep, _ = vintages(arm, iteration, exclude)
    new_rows = set(keep[iteration]) - set(keep[2])

    keys, n_gen = set(), 0
    for i in sorted(new_rows):
        key = A.canon(ds.inputs[i])
        if key in gen2src:
            n_gen += 1
            continue
        keys.add(key)
    return keys, {"n_rows": len(new_rows), "n_sources": len(keys), "n_generated": n_gen}


def cohort_records(arm: str, keys: set[str], pos_label: str, neg_label: str):
    """The attempt-log records for those sources, as ``{inputs, labels}`` dicts.

    Matching is by canonical text *after* the config's message transforms, because that
    is the form the dump (and therefore the vintage definition) is in; the dict that
    comes back carries the **raw** messages, because that is what generation is keyed on.
    """
    from agentic_redteam.persistence import JsonlStore
    from agentic_redteam.retrain import _successes_to_dicts

    found, seen = [], set()
    for suffix in ("_fp", "_fn"):
        path = REDTEAM_LOGS[arm].with_name(REDTEAM_LOGS[arm].name + suffix + ".jsonl")
        if not path.exists():
            continue
        for rec in JsonlStore(path=path).iter_all():
            if not rec.success:
                continue
            key = A.canon(A.apply_transforms(list(rec.sample.messages)))
            if key in keys and key not in seen:
                seen.add(key)
                found.append(rec)
    missing = keys - seen
    return _successes_to_dicts(found, pos_label, neg_label), missing


# --- similarity --------------------------------------------------------------------


def cache_path_for(arm: str, args) -> Path:
    """Where this arm's newly generated pairs are written.

    One definition, used by both the generation call and ``--write-arm-cache`` — they
    named the file independently before, so a future prompt-version bump would have left
    the merge step copying a path that no longer existed.
    """
    from agentic_redteam.preprocessing import GENERATION_PROMPT_VERSION

    if args.cache_path:
        return args.cache_path
    return args.out_dir / f"{arm}_contrastive_{GENERATION_PROMPT_VERSION}.jsonl"


def transcript(messages) -> str:
    from agentic_redteam.preprocessing import _render_transcript

    return _render_transcript(
        [{"role": m["role"], "content": m["content"]} if isinstance(m, dict)
         else {"role": m.role, "content": m.content} for m in messages]
    )


def similarity(a, b) -> float:
    """difflib ratio between two rendered transcripts.

    ``autojunk=False`` for the same reason ``persistence._is_near`` sets it: the
    heuristic derives its junk set from the second argument, so with it on the measure
    is not symmetric and badly under-reports on texts of this length.
    """
    return difflib.SequenceMatcher(None, transcript(a), transcript(b),
                                   autojunk=False).ratio()


# --- main --------------------------------------------------------------------------


def run_arm(arm: str, args) -> list[dict]:
    from agentic_redteam.config import load_config
    from agentic_redteam.preprocessing import (
        GENERATION_PROMPT_VERSION,
        _generation_system_prompt,
        _render_transcript,
        _short_label,
        generate_contrastive_dataset,
    )

    cfg = load_config(ARM_CONFIGS[arm])
    pre = cfg.preprocessing
    if pre is None:
        raise SystemExit(f"{ARM_CONFIGS[arm]} has no preprocessing section")

    # Probe metadata is the source of truth for the class labels (unpickling the probe
    # does not load the extraction LLM).
    probe = A.load_probe(A.ARMS[arm] / f"probe_iter{args.iteration}.pkl")
    pos, neg = probe.pos_class_label, probe.neg_class_label

    keys, stats = cohort_source_keys(arm, args.iteration, args.drop_overlong)
    records, missing = cohort_records(arm, keys, pos, neg)

    print(f"\n=== {arm} ===", flush=True)
    print(f"  cohort v3 \\ v2: {stats['n_rows']} rows "
          f"= {stats['n_sources']} source + {stats['n_generated']} generated", flush=True)
    print(f"  matched to attempt log: {len(records)} of {stats['n_sources']}"
          + (f"   [UNMATCHED {len(missing)}]" if missing else ""), flush=True)
    if missing:
        print("  unmatched sources are skipped — they cannot be keyed as a retrain "
              "would key them", flush=True)

    if args.limit and len(records) > args.limit:
        if args.sample_seed is not None:
            random.Random(args.sample_seed).shuffle(records)
        records = records[: args.limit]
        print(f"  --limit: {len(records)} record(s) this run", flush=True)

    if not records:
        return []

    # The old (v1) counterpart of each source, for the before/after comparison.
    ds = A.load_redteam_dataset(arm, args.iteration)
    gen2src = A.generated_to_source(arm)
    old_by_src: dict[str, list] = {}
    for i, msgs in enumerate(ds.inputs):
        key = A.canon(msgs)
        if key in gen2src:
            old_by_src[gen2src[key]] = msgs

    if args.dry_run:
        for r in records[: args.limit or 3]:
            cur = r["labels"]
            tgt = neg if cur == pos else pos
            print("\n" + "#" * 78)
            print(f"# SYSTEM  ({cur} -> {tgt})   prompt {GENERATION_PROMPT_VERSION}")
            print("#" * 78)
            print(_generation_system_prompt(
                cur, tgt,
                assistant_centric=pre.assistant_centric,
                concept_description=pre.concept_description,
                label_guidance=pre.label_guidance))
            print("\n" + "#" * 78)
            print("# USER")
            print("#" * 78)
            print(f'Original "{_short_label(cur)}" conversation:\n\n'
                  f"{_render_transcript(r['inputs'])}\n\n"
                  f'Now produce the "{_short_label(tgt)}" version as instructed.')
        print(f"\n[dry run] {len(records)} record(s) would be sent to "
              f"{pre.provider}:{pre.model}. No API calls made.", flush=True)
        return []

    cache_path = cache_path_for(arm, args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  generating {len(records)} pair(s) via {pre.provider}:{pre.model} "
          f"(prompt {GENERATION_PROMPT_VERSION}, cache {cache_path.name})", flush=True)

    t0 = time.time()
    out = generate_contrastive_dataset(
        records,
        pos_class_label=pos,
        neg_class_label=neg,
        provider=pre.provider,
        model=pre.model,
        max_concurrent=args.max_concurrent or pre.max_concurrent,
        max_tokens=pre.max_tokens,
        max_retries=pre.max_generation_retries,
        cache_path=cache_path,
        assistant_centric=pre.assistant_centric,
        concept_description=pre.concept_description,
        label_guidance=pre.label_guidance,
    )
    print(f"  done in {time.time() - t0:.0f}s — {len(out)} record(s) back "
          f"({len(records)} in)", flush=True)

    rows = []
    for rec in out:
        if not rec.get("is_generated"):
            continue
        src = rec["original_messages"]
        # Join to the v1 counterpart through the TRANSFORMED source key, since the dump
        # (and so old_by_src) is transformed while generation input is raw.
        from tuberlens.interfaces.dataset import Message
        skey = A.canon(A.apply_transforms(
            [Message(role=m["role"], content=m["content"]) for m in src]))
        old = old_by_src.get(skey)
        rows.append({
            "arm": arm,
            "source_key": A.sha16(skey),
            "source_label": rec["original_label"],
            "target_label": rec["labels"],
            "sim_new": similarity(src, rec["inputs"]),
            "sim_old": (None if old is None else similarity(src, old)),
            "len_src": len(transcript(src)),
            "len_new": len(transcript(rec["inputs"])),
            "len_old": (None if old is None else len(transcript(old))),
            "n_turns_src": len(src),
            "n_turns_new": len(rec["inputs"]),
            "n_turns_old": (None if old is None else len(old)),
            "explanation": rec.get("generation_explanation", ""),
            "source_messages": [{"role": m["role"], "content": m["content"]} for m in src],
            "new_messages": [{"role": m["role"], "content": m["content"]}
                             for m in rec["inputs"]],
            "old_messages": (None if old is None else
                             [{"role": m.role, "content": m.content} for m in old]),
        })
    return rows


def report(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        print("\nnothing generated.")
        return

    print("\n\n=== how close is the counterpart to its source? (difflib ratio) ===")
    print("   v1 asked for a similar-looking conversation; v2 asks for a minimal edit.")
    print("   Higher = the counterpart is a nearer neighbour of the attack it opposes.\n")
    hdr = (f"{'arm':14s} {'pairs':>5s} {'sim v1':>8s} {'sim v2':>8s} {'delta':>8s} "
           f"{'turns kept':>11s} {'len ratio v1':>13s} {'len ratio v2':>13s}")
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        old = [r["sim_old"] for r in rs if r["sim_old"] is not None]
        new = [r["sim_new"] for r in rs]
        kept = np.mean([r["n_turns_new"] == r["n_turns_src"] for r in rs])
        lr_old = np.mean([r["len_old"] / max(1, r["len_src"])
                          for r in rs if r["len_old"] is not None])
        lr_new = np.mean([r["len_new"] / max(1, r["len_src"]) for r in rs])
        print(f"{arm:14s} {len(rs):>5d} "
              f"{(np.mean(old) if old else float('nan')):>8.3f} {np.mean(new):>8.3f} "
              f"{(np.mean(new) - np.mean(old) if old else float('nan')):>+8.3f} "
              f"{kept:>11.2f} {lr_old:>13.2f} {lr_new:>13.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / "cohort_contrastive_v2.jsonl"
    jl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                  encoding="utf-8")
    csv = out_dir / "cohort_contrastive_v2.csv"
    with csv.open("w", encoding="utf-8") as fh:
        fh.write("arm,source_key,source_label,target_label,sim_old,sim_new,"
                 "len_src,len_old,len_new,n_turns_src,n_turns_old,n_turns_new\n")
        for r in sorted(rows, key=lambda x: (x["arm"], x["source_key"])):
            fh.write(",".join(str(r[k]) for k in
                              ("arm", "source_key", "source_label", "target_label",
                               "sim_old", "sim_new", "len_src", "len_old", "len_new",
                               "n_turns_src", "n_turns_old", "n_turns_new")) + "\n")
    print(f"\nwrote {jl}\nwrote {csv}")
    print("\nNothing was written to any arm's contrastive_cache.jsonl. The new pairs are "
          "in the per-arm cache next to those files; pass --write-arm-cache to merge.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--drop-overlong", choices=("none", "row", "pair"), default="pair",
                    help="must match the sweep, so the cohort is the one measured there")
    ap.add_argument("--limit", type=int, default=0,
                    help="generate at most N pairs per arm (0 = all). Start small.")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="with --limit, take a random sample instead of the first N")
    ap.add_argument("--max-concurrent", type=int, default=0,
                    help="override the config's preprocessing.max_concurrent")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact prompts and exit without calling the LLM")
    ap.add_argument("--write-arm-cache", action="store_true",
                    help="ALSO append the new pairs into the arm's production "
                         "contrastive_cache.jsonl, changing what its next retrain trains on")
    ap.add_argument("--cache-path", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=A.REPO / "results_hs_gemma27b_batch_ablation/contrastive_v2")
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set — the arms' preprocessing "
                         "provider is openrouter. Set it, or use --dry-run.")

    rows: list[dict] = []
    for arm in args.arm:
        rows += run_arm(arm, args)
    report(rows, args.out_dir)

    if args.write_arm_cache and rows:
        # Appended, never rewritten: the file is an append-only key→record log and the
        # v2 keys cannot collide with the v1 ones already in it.
        for arm in args.arm:
            src = cache_path_for(arm, args)
            dst = A.ARMS[arm] / "contrastive_cache.jsonl"
            if not src.exists():
                continue
            with dst.open("a", encoding="utf-8") as out, src.open(encoding="utf-8") as fh:
                n = 0
                for line in fh:
                    out.write(line)
                    n += 1
            print(f"appended {n} pair(s) to {dst}")


if __name__ == "__main__":
    main()
