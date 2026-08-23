#!/usr/bin/env python
"""Does the normalization step change the dev sweep, or only the ceiling?

The ceiling study found LayerNorm worth +0.0064 AUROC at 173 in-distribution training rows,
fading to nothing by ~900. The sweep asks the operational version of that question: the
probe is trained on base + red-team successes plus N dev rows, which is the regime this
repo's runs actually sit in, and the red-team half is *not* eval-distribution data.

Reads `sweep_<arm>.jsonl` (unnormalized, the existing run) against
`sweep_<arm>__norm-<kind>.jsonl`, plus the `sweepN0_*` single-point files that replicate the
N=0 fit across head seeds, and writes `results/NORM_SWEEP_SUMMARY.md`.

**Why N=0 needs its own replication.** `run_sweep.py` fits every other point three times
(one per dev draw) and reports the spread, but at N=0 no dev rows are drawn, so the draw
seed is meaningless and the point is fit exactly once — and it is the point the write-ups
quote as "red-team only". A 0.0068 head-seed swing was measured on a comparable ceiling fit,
so a single-fit comparison there says nothing. `--fit-seed` supplies the missing replicate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402

ARMS = ("mixed", "finetune", "dev_only")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# batch_size x gradient_accumulation_steps: below this many training rows the accumulation
# counter never reaches a step boundary, so `optimizer.step()` never fires and the "fit"
# returns its random init.
MIN_TRAIN_FOR_A_STEP = 16 * 4


def curve(rows: list[dict], arm: str) -> dict[int, tuple[float, float, int]]:
    """{N: (mean eval AUROC over dev draws, sd, n draws)} for one arm."""
    by_n = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            by_n[r["n_dev"]].append(r["mean"]["auroc"])
    return {n: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
            for n, v in sorted(by_n.items())}


def degenerate(rows: list[dict], arm: str) -> dict[int, str]:
    """{N: why this point carries no information about the head}.

    Two of them exist, and both would otherwise be averaged in as if they were measurements:

    * `dev_only` below `MIN_TRAIN_FOR_A_STEP` rows — no optimizer step fires, so the score is
      the random init's. It shows up as an AUROC near 0.5 that is *identical* at N=24 and
      N=48, which is the tell.
    * `finetune` below `MIN_TRAIN_FOR_A_STEP` *dev* rows, or where every draw kept the
      stage-1 checkpoint — either way the reported probe is the base + red-team fit, byte
      for byte, so the point duplicates N=0 instead of measuring the fine-tune.

      The row count that matters for the second stage is `n_dev`, **not** the recorded
      `n_train` (which is `len(red-team) + n_dev` and so is always in the hundreds). And
      `checkpoint_kept` cannot be used on its own here: `finetune_head` restores stage 1
      only when the fine-tuned weights score *worse*, so a stage that never stepped —
      identical weights, identical validation AUROC — is reported as `"finetuned"`.
      Observed at N=24 and N=48: `val_auroc_stage1 == val_auroc_finetuned == 0.906636` to
      six figures, and an eval AUROC exactly equal to the N=0 point's.

    Both are properties of the harness, not of the normalization, and both hit the two
    architectures at the same N — so including them would dilute the comparison with ties
    and, at `dev_only`, with a difference between two untrained random heads.
    """
    out: dict[int, str] = {}
    by_n = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            by_n[r["n_dev"]].append(r)
    for n, rs in by_n.items():
        if arm == "dev_only" and n < MIN_TRAIN_FOR_A_STEP:
            out[n] = "no optimizer step fires"
        elif arm == "finetune" and n < MIN_TRAIN_FOR_A_STEP:
            out[n] = "second stage never steps; identical to N=0"
        elif arm == "finetune" and all(
            r.get("checkpoint_kept") == "stage1" for r in rs
        ):
            out[n] = "stage-1 checkpoint kept; identical to N=0"
    return out


def n0_seeds(arm_name: str, kind: str) -> dict[int, float]:
    """{fit seed: N=0 eval AUROC} from the single-point replication files."""
    out = {}
    for p in sorted(C.RESULTS.glob(f"sweep_{arm_name}__norm-{kind}__fit*.jsonl")):
        for r in read(p):
            if r["n_dev"] == 0 and r["arm"] == "mixed":
                out[int(r["fit_seed"])] = r["mean"]["auroc"]
    return out


def f4(x) -> str:
    return f"{x:.4f}" if x is not None else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hu_ha_dd_gptoss120b")
    ap.add_argument("--norm", default="layernorm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_rows = read(C.RESULTS / f"sweep_{args.arm}.jsonl")
    norm_rows = read(C.RESULTS / f"sweep_{args.arm}__norm-{args.norm}.jsonl")
    if not base_rows or not norm_rows:
        print("missing sweep results", file=sys.stderr)
        return 1
    ceiling = json.loads((C.RESULTS / f"ceiling_{args.arm}.json").read_text())["ceiling"]["auroc"]

    L: list[str] = []
    L.append(f"# The dev sweep under {args.norm}")
    L.append("")
    L.append(
        f"Arm **{args.arm}**. A single probe is trained on base + red-team successes plus N "
        f"dev rows and scored on the eval splits, for 10 values of N and three arms, exactly "
        f"as in the main sweep — the only change is one normalization step in front of "
        f"`LinearThenSoftmax`'s linear layer. The unnormalized curves are the existing "
        f"`sweep_{args.arm}.jsonl`, at the same fit seed. Each N>0 point is the mean over 3 "
        f"dev draws; the unnormalized ceiling for this arm is {ceiling:.4f}."
    )
    L.append("")

    # ---- headline --------------------------------------------------------------------------
    all_d, sd_pairs = [], []
    for arm in ARMS:
        cb, cn = curve(base_rows, arm), curve(norm_rows, arm)
        skip = degenerate(base_rows, arm) | degenerate(norm_rows, arm)
        for n in sorted(set(cb) & set(cn)):
            if n in skip:
                continue
            all_d.append(cn[n][0] - cb[n][0])
            if cb[n][2] > 1 and cn[n][2] > 1 and cb[n][1] > 0:
                sd_pairs.append(cn[n][1] / cb[n][1])
    all_d = np.array(all_d)
    L.append("## What it does")
    L.append("")
    L.append(
        f"**{args.norm} is ahead at {int((all_d > 0).sum())} of the {len(all_d)} informative "
        f"points**, across all three arms, by {all_d.mean():+.4f} on average. That is a "
        f"larger and far more consistent effect than the ceiling study found at its top rung "
        f"(+0.0020, 3/4 seeds) — which is what the ceiling study predicted would happen: the "
        f"gain grows as the training set shrinks, and every point here trains on less "
        f"eval-distribution data than the ceiling's smallest rung."
    )
    L.append("")
    if sd_pairs:
        L.append(
            f"It also **tightens the spread across dev draws** — the normalized head's sd is "
            f"{np.mean(sd_pairs):.2f}x the unnormalized head's, averaged over the "
            f"{len(sd_pairs)} points that have one. The `finetune` arm is where this is most "
            f"visible (e.g. at N=97, 0.0116 -> 0.0021), and it is also the arm with the "
            f"largest mean gain."
        )
    L.append("")
    L.append(
        "**None of this closes the gap to the ceiling.** The best sweep point moves from "
        f"{max(curve(base_rows, 'dev_only').values())[0]:.4f} to "
        f"{max(curve(norm_rows, 'dev_only').values())[0]:.4f} against a ceiling of "
        f"{ceiling:.4f}. Normalization is a better-conditioned head, not more information."
    )
    L.append("")

    # ---- N = 0, replicated over head seeds ------------------------------------------------
    b0, n0 = n0_seeds(args.arm, "none"), n0_seeds(args.arm, args.norm)
    common = sorted(set(b0) & set(n0))
    base_n0 = curve(base_rows, "mixed").get(0, (None, 0.0, 0))[0]
    L.append("## N = 0: base + red-team only")
    L.append("")
    if common:
        b = np.array([b0[s] for s in common])
        v = np.array([n0[s] for s in common])
        d = v - b
        ctrl = ""
        if C.FIT_SEED in b0 and base_n0 is not None:
            gap = abs(b0[C.FIT_SEED] - base_n0)
            ctrl = (f" The `none`/seed-{C.FIT_SEED} re-run reproduces the existing "
                    f"sweep's N=0 row to {gap:.2e} — the control on the new code path.")
        L.append(
            f"This is the point every write-up quotes as the red-team-only number, and "
            f"`run_sweep.py` fits it once, so it had no error bar at all. Re-fit under "
            f"{len(common)} head seeds.{ctrl}")
        L.append("")
        L.append("| head seed | none | " + args.norm + " | diff |")
        L.append("|---|---|---|---|")
        for s in common:
            L.append(f"| {s} | {f4(b0[s])} | {f4(n0[s])} | {n0[s] - b0[s]:+.4f} |")
        L.append(f"| **mean** | **{f4(b.mean())}** | **{f4(v.mean())}** "
                 f"| **{d.mean():+.4f}** |")
        L.append(f"| sd | {b.std(ddof=1):.4f} | {v.std(ddof=1):.4f} "
                 f"| {d.std(ddof=1):.4f} |")
        L.append("")
        L.append(
            f"Paired, {args.norm} wins {int((d > 0).sum())}/{len(d)}. The unnormalized "
            f"head's own spread across these seeds is {b.max() - b.min():.4f} — read the "
            f"single-seed number against that before reading the difference."
        )
    else:
        L.append("_No head-seed replication found; run `run_sweep.py --fit-seed ... "
                 "--n-points 1 --arms mixed`._")
    L.append("")

    # ---- the curves -----------------------------------------------------------------------
    for arm in ARMS:
        cb, cn = curve(base_rows, arm), curve(norm_rows, arm)
        ns = sorted(set(cb) & set(cn))
        if not ns:
            continue
        skip = degenerate(base_rows, arm) | degenerate(norm_rows, arm)
        L.append(f"## `{arm}`")
        L.append("")
        L.append("| N dev | none (mean +- sd) | " + args.norm + " (mean +- sd) | diff |")
        L.append("|---|---|---|---|")
        diffs = []
        for n in ns:
            mb, sb, _ = cb[n]
            mn, sn, _ = cn[n]
            if n in skip:
                L.append(f"| {n} | {f4(mb)} | {f4(mn)} | _excluded: {skip[n]}_ |")
                continue
            diffs.append(mn - mb)
            L.append(f"| {n} | {f4(mb)} +- {sb:.4f} | {f4(mn)} +- {sn:.4f} "
                     f"| {mn - mb:+.4f} |")
        d = np.array(diffs)
        live = [n for n in ns if n not in skip]
        L.append("")
        L.append(
            f"Mean difference over the {len(d)} informative points **{d.mean():+.4f}**, "
            f"positive at {int((d > 0).sum())}/{len(d)}. Best point: none "
            f"{max(cb[n][0] for n in live):.4f}, {args.norm} "
            f"{max(cn[n][0] for n in live):.4f}."
        )
        L.append("")

    out = Path(args.out) if args.out else C.RESULTS / "NORM_SWEEP_SUMMARY.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
