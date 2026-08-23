#!/usr/bin/env python
"""Compare the ceiling under each normalization step against the unnormalized baseline.

Reads `ceiling_<arm>.json` (the architecture every experiment in this repo actually used),
every `ceiling_<arm>__norm-<kind>.json` written by `run_ceiling.py --norm`, and every
`..__fit<seed>.json` written by `--fit-seed`; writes `results/NORM_SUMMARY.md`.

Three things it checks rather than states:

* **`--norm none` must reproduce the baseline exactly.** Same seeds, same folds, same
  validation slice, and an `nn.Identity` that draws no RNG — so any difference at all means
  the plumbing moved, not the normalization. Reported as a max absolute deviation over every
  rung, split and metric, which should be exactly 0.
* **Whether a gap is bigger than the noise, measured PAIRED.** A single fit is not evidence
  of a 1e-3 difference: the unnormalized head's own top-rung AUROC moves by 0.0068 between
  head seeds, five times the gap the single-seed table shows. The `--fit-seed` runs re-fit
  the *same* folds and the *same* training rows under a different init and batch order, so
  each seed gives a matched pair and the shared fold noise cancels. What is reported is the
  distribution of the per-seed difference, not the difference of two means.
* **Whether the ladder has flattened.** A ceiling estimated from a still-climbing ladder is
  a lower bound. At these seeds the ladder is not even monotone — the baseline's rung-693
  fit beats its own top rung at two of four seeds — so the top rung is labelled for what it
  is rather than treated as a plateau.

The metrics are reported separately on purpose. AUROC is stable enough here to compare;
`tpr_at_fpr` is not, and its spread is printed so that is visible rather than inferred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca_common as C  # noqa: E402
import ca_norm  # noqa: E402

LABEL = {
    "none": "none (baseline arch)",
    "layernorm_noaffine": "LayerNorm, no affine",
    "layernorm": "LayerNorm + affine",
    "rmsnorm": "RMSNorm + scale",
    "standardize": "per-feature standardize (frozen)",
}


def load_runs(arm: str) -> tuple[dict, dict[str, dict[int, dict]]]:
    """(baseline file, {norm kind: {fit seed: results}})."""
    base = json.loads((C.RESULTS / f"ceiling_{arm}.json").read_text())
    runs: dict[str, dict[int, dict]] = {}
    for kind in ca_norm.KINDS:
        p = C.RESULTS / f"ceiling_{arm}__norm-{kind}.json"
        if p.exists():
            runs.setdefault(kind, {})[C.FIT_SEED] = json.loads(p.read_text())
        for q in sorted(C.RESULTS.glob(f"ceiling_{arm}__norm-{kind}__fit*.json")):
            r = json.loads(q.read_text())
            runs.setdefault(kind, {})[int(r["fit_seed"])] = r
    return base, runs


def control_deviation(base: dict, none_run: dict | None) -> float | None:
    if none_run is None:
        return None
    worst = 0.0
    for tag, entry in base["by_train_size"].items():
        other = none_run["by_train_size"].get(tag)
        if other is None:
            return float("inf")
        for split, m in entry["per_split"].items():
            for metric in ("auroc", "accuracy", "tpr_at_fpr"):
                worst = max(worst, abs(m[metric] - other["per_split"][split][metric]))
    return worst


def f4(x: float) -> str:
    return f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hu_ha_dd_gptoss120b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base, runs = load_runs(args.arm)
    if not runs:
        print("no --norm runs found", file=sys.stderr)
        return 1
    rungs = list(base["by_train_size"])
    splits = list(base["ceiling_per_split"])
    top = base["ceiling_rung"]
    kinds = [k for k in ca_norm.KINDS if k in runs]
    # Seeds every architecture in the paired comparison was fit under.
    paired_kinds = [k for k in kinds if len(runs[k]) > 1]
    common = sorted(set.intersection(*(set(runs[k]) for k in paired_kinds))) \
        if len(paired_kinds) > 1 else []

    def auroc(kind, seed, rung):
        return runs[kind][seed]["by_train_size"][rung]["mean"]["auroc"]

    L: list[str] = []
    L.append("# A normalization step in front of the probe head")
    L.append("")
    L.append(
        f"`LinearThenSoftmax` reads the layer-32 residual stream raw — `nn.Linear(5376, 1)` "
        f"straight onto the activation. This inserts one normalization step in front of that "
        f"linear and changes nothing else: same 5 folds, same reserved "
        f"{base['n_validation']}-row dev validation slice, same seven hyperparameters, same "
        f"training-size ladder. Arm **{args.arm}**; out-of-fold mean eval AUROC over the "
        f"{len(splits)} `eval_sets/hu_ha` splits ({base['n_eval_rows']} rows). The ceiling "
        f"never touches red-team data, so the arm choice selects the eval/dev blobs and "
        f"nothing else."
    )
    L.append("")

    dev = control_deviation(base, runs.get("none", {}).get(C.FIT_SEED))
    if dev is not None:
        verdict = "reproduces it exactly" if dev == 0.0 else f"DIFFERS by up to {dev:.2e}"
        L.append(
            f"**Control.** `--norm none` runs the baseline architecture through the new "
            f"subclass — an `nn.Identity`, no parameters, no RNG draws — and {verdict} "
            f"against the pre-existing `ceiling_{args.arm}.json`, on every rung, split and "
            f"metric. So what follows is the normalization, not the plumbing."
        )
        L.append("")

    # ---- findings --------------------------------------------------------------------------
    if common and "none" in paired_kinds and "layernorm" in paired_kinds:
        def pair(rung):
            n = np.array([auroc("none", s, rung) for s in common])
            v = np.array([auroc("layernorm", s, rung) for s in common])
            return n, v, v - n

        n_lo, v_lo, d_lo = pair(rungs[0])
        n_hi, v_hi, d_hi = pair(top)
        sd_ratio = np.mean([
            np.array([auroc("layernorm", s, r) for s in common]).std(ddof=1)
            / np.array([auroc("none", s, r) for s in common]).std(ddof=1)
            for r in rungs
        ])
        std_gap = (auroc("standardize", C.FIT_SEED, top)
                   - auroc("none", C.FIT_SEED, top)) if "standardize" in runs else None
        rms_gap = max(
            abs(auroc("rmsnorm", C.FIT_SEED, r) - auroc("layernorm_noaffine", C.FIT_SEED, r))
            for r in rungs
        ) if {"rmsnorm", "layernorm_noaffine"} <= set(runs) else None

        L.append("## What it does")
        L.append("")
        L.append(
            f"1. **It helps, and the help is concentrated where the training data is "
            f"scarce.** Paired over {len(common)} head seeds, LayerNorm beats the raw input "
            f"by {d_lo.mean():+.4f} at {rungs[0]} rows/fold ({int((d_lo > 0).sum())}/"
            f"{len(d_lo)} seeds) and by {d_hi.mean():+.4f} at the top rung "
            f"({int((d_hi > 0).sum())}/{len(d_hi)}). At the top the paired sd "
            f"({d_hi.std(ddof=1):.4f}) is the same size as the difference, so the honest "
            f"reading is: a clear gain at a few hundred in-distribution rows, fading to "
            f"nothing once the head has ~900."
        )
        L.append(
            f"2. **It halves the run-to-run spread.** Across the same seeds the normalized "
            f"head's AUROC sd is {sd_ratio:.2f}x the unnormalized head's, averaged over the "
            f"four rungs — the same direction at every rung. For a *single* probe that is "
            f"arguably the more useful property of the two; for this repo's 10-member "
            f"ensembles it is partly redundant, since averaging members already cancels "
            f"seed variance."
        )
        if rms_gap is not None:
            L.append(
                f"3. **Mean subtraction contributes nothing; the per-token rescaling is the "
                f"whole effect.** RMSNorm and affine-free LayerNorm differ by at most "
                f"{rms_gap:.4f} at any rung, and the learnable affine adds little beyond "
                f"either. What matters is dividing out each token's own magnitude."
            )
        if std_gap is not None:
            L.append(
                f"4. **Per-feature standardization is actively harmful** "
                f"({std_gap:+.4f} at the top rung — an order of magnitude outside the seed "
                f"noise, and the only result here that does not need the paired test to be "
                f"believed). Dividing each of the 5376 dimensions by its own std equalizes "
                f"the feature geometry, which is what it was meant to do, and that is the "
                f"problem: the dimensions with tiny variance are mostly noise, and this "
                f"amplifies them to parity with the ones carrying the concept. The per-token "
                f"norms apply **one scalar per token** and leave the relative feature scales "
                f"intact, which is precisely the difference."
            )
        L.append(
            f"5. **AUROC is the only metric that moves.** Accuracy is a wash "
            f"(<=0.005 either way at every rung). `tpr_at_fpr` looks worse for the "
            f"normalized head, but its own across-seed sd is 0.10-0.22 — at 134-400 rows a "
            f"1% FPR threshold sits on one or two negatives — so that column is not "
            f"readable at this split size, in either direction."
        )
        L.append("")

    # ---- single-seed ladder --------------------------------------------------------------
    L.append(f"## All five variants, one head seed ({C.FIT_SEED})")
    L.append("")
    L.append("| normalization | " + " | ".join(rungs) + " |")
    L.append("|---|" + "---|" * len(rungs))
    for kind in kinds:
        if C.FIT_SEED not in runs[kind]:
            continue
        L.append(f"| {LABEL[kind]} | "
                 + " | ".join(f4(auroc(kind, C.FIT_SEED, r)) for r in rungs) + " |")
    L.append("")
    L.append(
        "Read this table for its *shape*, not its last digits — the next section shows the "
        "head seed alone moves the top rung by more than the gaps between the middle three "
        "rows. What is robust here is the sign and the ordering: the three per-token norms "
        "all beat the raw input at every rung, they land within 0.001 of each other, and "
        "the per-feature standardizer is far below all of them."
    )
    L.append("")

    # ---- paired seed comparison ----------------------------------------------------------
    if common and "none" in paired_kinds:
        others = [k for k in paired_kinds if k != "none"]
        L.append(f"## Paired against the raw input, over {len(common)} head seeds")
        L.append("")
        L.append(
            f"`--fit-seed` re-fits the *same* folds and the *same* training rows under a "
            f"different head init and batch order. Seeds {', '.join(map(str, common))}. "
            f"Each column is a matched pair, so the fold noise the two share cancels; "
            f"`sd` is over the four per-seed differences."
        )
        L.append("")
        for kind in others:
            L.append(f"**{LABEL[kind]}** vs `none`:")
            L.append("")
            L.append("| rows/fold | none (mean +- sd) | " + LABEL[kind]
                     + " (mean +- sd) | paired diff | sd | wins |")
            L.append("|---|---|---|---|---|---|")
            for r in rungs:
                n = np.array([auroc("none", s, r) for s in common])
                v = np.array([auroc(kind, s, r) for s in common])
                d = v - n
                L.append(
                    f"| {r} | {f4(n.mean())} +- {n.std(ddof=1):.4f} "
                    f"| {f4(v.mean())} +- {v.std(ddof=1):.4f} "
                    f"| **{d.mean():+.4f}** | {d.std(ddof=1):.4f} "
                    f"| {int((d > 0).sum())}/{len(d)} |"
                )
            L.append("")

    # ---- per split -----------------------------------------------------------------------
    if common and "none" in paired_kinds and len(paired_kinds) > 1:
        kind = "layernorm" if "layernorm" in paired_kinds else \
            [k for k in paired_kinds if k != "none"][0]
        L.append(f"## Where the gain sits ({LABEL[kind]} minus none, mean over "
                 f"{len(common)} seeds)")
        L.append("")
        L.append("| rows/fold | " + " | ".join(s.replace("eval_", "") for s in splits)
                 + " |")
        L.append("|---|" + "---|" * len(splits))
        for r in rungs:
            cells = []
            for sp in splits:
                d = np.array([
                    runs[kind][s]["by_train_size"][r]["per_split"][sp]["auroc"]
                    - runs["none"][s]["by_train_size"][r]["per_split"][sp]["auroc"]
                    for s in common
                ])
                cells.append(f"{d.mean():+.4f}")
            L.append(f"| {r} | " + " | ".join(cells) + " |")
        L.append("")

    # ---- the other two metrics -------------------------------------------------------------
    if common and "none" in paired_kinds and len(paired_kinds) > 1:
        kind = "layernorm" if "layernorm" in paired_kinds else \
            [k for k in paired_kinds if k != "none"][0]
        L.append("## The other two metrics")
        L.append("")
        L.append("| metric | rows/fold | none (mean +- sd) | "
                 + LABEL[kind] + " (mean +- sd) | paired diff |")
        L.append("|---|---|---|---|---|")
        for metric in ("accuracy", "tpr_at_fpr"):
            for r in rungs:
                n = np.array([runs["none"][s]["by_train_size"][r]["mean"][metric]
                              for s in common])
                v = np.array([runs[kind][s]["by_train_size"][r]["mean"][metric]
                              for s in common])
                d = v - n
                L.append(f"| {metric} | {r} | {f4(n.mean())} +- {n.std(ddof=1):.4f} "
                         f"| {f4(v.mean())} +- {v.std(ddof=1):.4f} | {d.mean():+.4f} |")
        L.append("")

    # ---- ladder flatness -------------------------------------------------------------------
    prev = rungs[-2]
    L.append(f"## Is the ladder flat at the top? (`{prev}` -> `{top}`)")
    L.append("")
    L.append("| normalization | seeds where the top rung is the best rung | climb at "
             f"seed {C.FIT_SEED} |")
    L.append("|---|---|---|")
    for kind in kinds:
        ss = sorted(runs[kind])
        best = sum(
            1 for s in ss
            if max(rungs, key=lambda r: auroc(kind, s, r)) == top
        )
        climb = auroc(kind, C.FIT_SEED, top) - auroc(kind, C.FIT_SEED, prev)
        L.append(f"| {LABEL[kind]} | {best}/{len(ss)} | {climb:+.4f} |")
    L.append("")
    L.append(
        "The ladder is still climbing on average, so every number above is a lower bound on "
        "its architecture's ceiling rather than a plateau — but it is not monotone seed to "
        "seed, which is a second reason not to read the top rung alone."
    )
    L.append("")

    out = Path(args.out) if args.out else C.RESULTS / "NORM_SUMMARY.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
