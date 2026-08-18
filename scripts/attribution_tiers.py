#!/usr/bin/env python
"""Stage 4: turn the LOO cube's per-pair verdicts into a tier file the viewer reads.

``attribution_verify.flag_sets`` partitions an arm's red-team pairs into three
classes off the *flagging* half of the LOO cube's seeds:

    harmful   t > +2 on at least one eval split  (removing it raised that AUROC)
    useful    t < -2 on at least one eval split  (removing it lowered that AUROC)
    inert     |t| < 2 on all four splits

This writes them as ``{tier, tier_description, pair_id, in_iters, note, original,
contrastive}`` rows — the schema ``build_redteam_training_viewer.py --tiers`` expects,
plus a per-pair ``note`` carrying the t-statistics that put the pair in its tier. So
the conversations can be read next to the number that flagged them.

Three things worth knowing before reading the tabs:

**A pair can be in two tiers.** ``harmful``/``useful`` are per-split "any" tests, so a
pair that helps one split and hurts another is emitted in both tabs. Two deepseek pairs
are (115 and 275); their notes say so.

**"inert" means "not resolvable at 25 seeds", not "no effect".** Most inert pairs sit
just under the bar, and dropping the inert set *as a set* is the largest measured
effect in the study. The tier is a statement about statistical power, not about the
conversation.

**None of the three tiers survives multiplicity correction per pair.** BH at q=0.10
over pairs x splits rejects nothing in either arm. The set-level verification is what
carries the finding; these tabs are for reading what the selection picked, not for
concluding that a given conversation is bad.

Pairs are addressed by the cube's own ``pair_source_idx`` / ``pair_generated_idx``
rather than by re-deriving the pairing, so a tier row cannot drift from the row the
deltas were measured on.

Usage:
    .venv_claude/bin/python scripts/attribution_tiers.py --arm deepseekv4pro
    .venv_claude/bin/python scripts/attribution_tiers.py --arm deepseekv4pro --dump-tier harmful
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import attribution_lib as A
import attribution_verify as V

TIERS = {
    "harmful (t>+2)": "removing the pair raised at least one split's AUROC by more than "
                      "2 standard errors over the flagging seeds",
    "useful (t<-2)": "removing the pair lowered at least one split's AUROC by more than "
                     "2 standard errors over the flagging seeds",
    "inert (|t|<2)": "no split moved by 2 standard errors either way — undetectable at "
                     "this seed count, which is not the same as having no effect",
}


def canon_inputs(inputs) -> str:
    return json.dumps([[m.get("role", ""), m.get("content", "")] for m in inputs],
                      ensure_ascii=False)


def load_dump(probe_dir: Path, iteration: int) -> list[dict]:
    path = probe_dir / f"redteam_postprocessed_iter{iteration}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_seen(probe_dir: Path, iteration: int) -> dict[str, list[int]]:
    """{canonical conversation: [iterations whose dump contains it]}.

    The dumps are near-cumulative but not nested, so membership is checked per
    iteration rather than inferred from the earliest hit.
    """
    out: dict[str, list[int]] = {}
    for it in range(1, iteration + 1):
        path = probe_dir / f"redteam_postprocessed_iter{it}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                out.setdefault(canon_inputs(json.loads(line)["inputs"]), []).append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(A.ARMS), required=True)
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--flag-seeds", type=int, default=None,
                    help="LOO seeds used to classify (default: half the cube, matching "
                         "attribution_verify)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <probe-dir>/attribution_tiers.jsonl")
    ap.add_argument("--dump-tier", default=None,
                    help="also print every conversation in this tier to stdout")
    ap.add_argument(
        "--loo-dir",
        type=Path,
        default=A.REPO / "results_hu_harm_gemma27b_batch_ablation/attribution",
    )
    args = ap.parse_args()

    probe_dir = A.ARMS[args.arm]
    loo_path = args.loo_dir / f"{args.arm}_iter{args.iteration}_loo.npz"
    if not loo_path.exists():
        raise SystemExit(f"no LOO result at {loo_path} — run attribution_loo.py first")

    cube = np.load(loo_path, allow_pickle=True)
    n_flag = args.flag_seeds or int(cube["deltas"].shape[2]) // 2
    info = V.flag_sets(loo_path, n_flag_seeds=n_flag)
    splits, mean, se = info["splits"], info["mean"], info["se"]
    t = mean / np.maximum(se, 1e-12)

    dump = load_dump(probe_dir, args.iteration)
    membership = first_seen(probe_dir, args.iteration)
    src_idx = cube["pair_source_idx"]
    gen_idx = cube["pair_generated_idx"]
    labels = cube["source_label"]

    sets = info["sets"]
    members = {
        "harmful (t>+2)": sets["drop_harmful_2se"],
        "useful (t<-2)": info["useful_2se"],
        "inert (|t|<2)": sets["drop_inert"],
    }

    out_path = args.out or probe_dir / "attribution_tiers.jsonl"
    rows = []
    for tier, mask in members.items():
        for pid in np.flatnonzero(mask):
            s, g = int(src_idx[pid]), int(gen_idx[pid])
            if s < 0 or g < 0:
                continue  # an unpaired half has no card to draw
            if str(dump[s]["label"]) != str(labels[pid]):
                raise SystemExit(
                    f"pair {pid}: dump row {s} is {dump[s]['label']!r} but the cube "
                    f"recorded {labels[pid]!r} — the dump and the cube disagree"
                )
            up = [f"{splits[j]} t={t[pid, j]:+.1f} (Δ{mean[pid, j]:+.4f})"
                  for j in range(len(splits)) if t[pid, j] > 2]
            down = [f"{splits[j]} t={t[pid, j]:+.1f} (Δ{mean[pid, j]:+.4f})"
                    for j in range(len(splits)) if t[pid, j] < -2]
            note = f"pair {pid} · {n_flag} flagging seeds · "
            note += ("removal HELPS " + "; ".join(up) if up else "no split helped at 2 SE")
            note += " · "
            note += ("removal HURTS " + "; ".join(down) if down else "no split hurt at 2 SE")
            rows.append({
                "tier": tier,
                "tier_description": TIERS[tier],
                "pair_id": int(pid),
                "in_iters": membership.get(canon_inputs(dump[s]["inputs"]), []),
                "note": note,
                "original": {"inputs": dump[s]["inputs"], "label": dump[s]["label"]},
                "contrastive": {"inputs": dump[g]["inputs"], "label": dump[g]["label"]},
            })

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{args.arm}: {info['n_pairs']} pairs classified on LOO seeds "
          f"{A.SEED}..{A.SEED + n_flag - 1}")
    for tier, mask in members.items():
        print(f"  {tier:18s} {int(mask.sum()):4d} pairs")
    print(f"  (in two tiers: {int((members['harmful (t>+2)'] & members['useful (t<-2)']).sum())})")
    print(f"-> {out_path.relative_to(A.REPO)}  ({len(rows)} rows)")

    if args.dump_tier:
        match = [k for k in members if args.dump_tier.lower() in k.lower()]
        if not match:
            raise SystemExit(f"--dump-tier {args.dump_tier!r} matched none of {list(members)}")
        for row in rows:
            if row["tier"] != match[0]:
                continue
            print("\n" + "=" * 100)
            print(f"{row['note']}   | in iters {row['in_iters']}")
            for side in ("original", "contrastive"):
                print("-" * 100)
                print(f"[{side.upper()}]  label={row[side]['label']}")
                for m in row[side]["inputs"]:
                    print(f"  <{m['role']}> {m['content']}")


if __name__ == "__main__":
    main()
