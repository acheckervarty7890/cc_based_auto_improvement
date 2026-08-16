"""Re-run the counterpart misclassification count on the v2 (minimal-edit) pairs.

The transfer analysis measured, for the vintage-3-minus-vintage-2 cohort, how often each
row is misclassified by the ten v2-vintage probes and by the shipped ``probe_iter2``.
Under the **v1** generation prompt the two halves of a pair came out wildly asymmetric:

    attacker source      0.856 / 0.901   (v2 seeds)      1.000 / 1.000   (probe_iter2)
    generated counterpart 0.000 / 0.008                  0.000 / 0.011

i.e. the counterpart was free — the probe already classified it with ~0.99+ confidence,
so all of a pair's contrast sat on the attacker-written side.

The v1 prompt asked for a "similar-looking conversation ... that belongs to the target
class", and the generator duly wrote a fresh unambiguous exemplar: measured over this
cohort, the v1 counterparts share only 0.37–0.43 of their source's text and run 2.0–2.3x
its length. The v2 prompt asks for the **minimal edit** that flips the label, and the
same measurement gives 0.90 similarity at 1.06–1.08x length.

This script asks what that did to the number. Same probes, same prediction rule
(``sigmoid(logit) >= 0.5``, the config's ``probe.threshold``), same cohort — only the
counterpart conversations changed. Every row is out-of-sample for all ten probes: they
were trained on vintage 2, and these are rewrites of vintage-3-only attacks.

A caveat that belongs on the result. Sitting near a conversation that fools the probe is
exactly what a minimal edit does, so a *higher* counterpart misclassification rate is the
expected outcome, not a surprise — and it is not automatically an improvement. A
counterpart the probe gets wrong is a training row asserting the opposite label close to
the decision boundary, which is more informative than a free one but also more able to
do damage if the rewrite failed to genuinely flip the class. The rate below says how
often the probe disagrees with the label; it cannot say whether the label is right. That
needs the judge, which is a separate run.

Usage:
    .venv_claude/bin/python scripts/score_contrastive_v2.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
from attribution_vintage import _load_fitted, fit_path
from vintage_new_sample_success import THRESHOLD, _rows_dataset, probs_for

DEFAULT_IN = A.REPO / "results_hs_gemma27b_batch_ablation/contrastive_v2/cohort_contrastive_v2.jsonl"


class _Rows:
    """Minimal stand-in for a ``LabelledDataset`` for the scoring helpers.

    ``probs_for`` / ``_rows_dataset`` only ever read ``.inputs``, ``.ids`` and
    ``other_fields["labels"]``, so reusing them on a plain list of conversations needs
    nothing more than this — and reusing them is the point: the chunk-by-width padding
    and the blob lookup are then provably the same code that produced the v1 numbers.
    """

    def __init__(self, inputs, ids, labels):
        self.inputs = inputs
        self.ids = ids
        self.other_fields = {"labels": labels}

    def __len__(self):
        return len(self.inputs)


def build(rows: list[dict], which: str, arm: str):
    """Conversations of one side for one arm, transformed and blob-backed."""
    from tuberlens.interfaces.dataset import Message

    key = {"new": "new_messages", "old": "old_messages",
           "source": "source_messages"}[which]
    label_key = "source_label" if which == "source" else "target_label"

    inputs, ids, labels, keep = [], [], [], []
    for r in rows:
        if r["arm"] != arm or r.get(key) is None:
            continue
        msgs = A.apply_transforms(
            [Message(role=m["role"], content=m["content"]) for m in r[key]]
        )
        if not A.redteam_blob_path(msgs).exists():
            continue  # not extracted yet; reported by the caller
        inputs.append(msgs)
        ids.append(r["source_key"])
        labels.append(r[label_key])
        keep.append(r)
    return _Rows(inputs, ids, labels), keep


def analyse(arm: str, rows: list[dict], seeds: list[int], iteration: int,
            fits_dir: Path, chunk: int) -> list[dict]:
    ref = A.load_probe(A.ARMS[arm] / f"probe_iter{iteration}.pkl")
    probes: dict = {}
    for s in seeds:
        p = fit_path(fits_dir, arm, 2, s)
        if not p.exists():
            raise SystemExit(f"missing v2 checkpoint {p} — run attribution_vintage.py first")
        probes[f"v2_s{s}"], _ = _load_fitted(ref, p)
    probes["pipeline_iter2"] = A.load_probe(A.ARMS[arm] / "probe_iter2.pkl")
    seed_keys = [f"v2_s{s}" for s in seeds]

    mine = [r for r in rows if r["arm"] == arm]
    out: list[dict] = []
    for which in ("source", "old", "new"):
        ds, kept = build(mine, which, arm)
        missing = len(mine) - len(ds)
        print(f"  {which:7s}: {len(ds)} scored"
              + (f"   [{missing} without activations — skipped]" if missing else ""),
              flush=True)
        if not len(ds):
            continue
        probs = probs_for(probes, ds, list(range(len(ds))), chunk)
        for j, r in enumerate(kept):
            pos = ds.other_fields["labels"][j] == ref.pos_class_label
            wrong = {k: bool((probs[k][j] >= THRESHOLD) != pos) for k in probs}
            nw = sum(wrong[k] for k in seed_keys)
            out.append({
                "arm": arm,
                "side": which,
                "source_key": r["source_key"],
                "label": ds.other_fields["labels"][j],
                "n_seeds": len(seed_keys),
                "n_misclassified": nw,
                "rate": nw / len(seed_keys),
                "mean_prob": float(np.mean([probs[k][j] for k in seed_keys])),
                "pipeline_iter2_prob": float(probs["pipeline_iter2"][j]),
                "pipeline_iter2_wrong": wrong["pipeline_iter2"],
                "sim_to_source": r["sim_new"] if which == "new" else (
                    r["sim_old"] if which == "old" else 1.0),
                "per_seed": {str(s): wrong[f"v2_s{s}"] for s in seeds},
            })
        del probs
        gc.collect()
    return out


def report(recs: list[dict], seeds: list[int], out_dir: Path) -> None:
    by = defaultdict(list)
    for r in recs:
        by[(r["arm"], r["side"])].append(r)

    LABEL = {"source": "attacker source", "old": "counterpart v1", "new": "counterpart v2"}
    print("\n\n=== misclassification rate, by side of the pair ===")
    print("   Same ten v2-vintage probes and the same >= 0.5 rule as the transfer table;")
    print("   only the counterpart conversations differ between v1 and v2.\n")
    hdr = (f"{'arm':14s} {'side':17s} {'n':>4s} {'RATE':>7s} {'always':>7s} {'never':>6s} "
           f"{'iter2':>7s} {'mean p(own)':>12s} {'sim to src':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted({r["arm"] for r in recs}):
        for side in ("source", "old", "new"):
            rs = by.get((arm, side))
            if not rs:
                continue
            trials = sum(r["n_seeds"] for r in rs)
            rate = sum(r["n_misclassified"] for r in rs) / trials
            # probability mass on the row's OWN label, so >0.5 is a correct call
            own = np.mean([r["mean_prob"] if r["label"] == "high-stakes"
                           else 1 - r["mean_prob"] for r in rs])
            print(f"{arm:14s} {LABEL[side]:17s} {len(rs):>4d} {rate:>7.3f} "
                  f"{sum(1 for r in rs if r['n_misclassified'] == r['n_seeds']):>7d} "
                  f"{sum(1 for r in rs if r['n_misclassified'] == 0):>6d} "
                  f"{np.mean([r['pipeline_iter2_wrong'] for r in rs]):>7.3f} "
                  f"{own:>12.3f} {np.mean([r['sim_to_source'] for r in rs]):>11.3f}")

    n_s = len(seeds)
    print("\n\n=== distribution over seeds, counterpart v2 ===")
    h = f"{'arm':14s} " + " ".join(f"{k}/{n_s}".rjust(6) for k in range(n_s + 1))
    print(h)
    print("-" * len(h))
    for arm in sorted({r["arm"] for r in recs}):
        rs = by.get((arm, "new"), [])
        print(f"{arm:14s} " + " ".join(
            f"{sum(1 for r in rs if r['n_misclassified'] == k):>6d}" for k in range(n_s + 1)))

    print("\n\n=== both halves wrong at once? (the pair no longer contrasts) ===")
    print("   A pair whose source AND counterpart are both misclassified by the same probe")
    print("   asserts two opposite labels on two nearly identical texts, and the probe")
    print("   disagrees with both. Counted per (pair, seed).\n")
    hdr2 = f"{'arm':14s} {'pairs':>6s} {'v1 both-wrong':>14s} {'v2 both-wrong':>14s}"
    print(hdr2)
    print("-" * len(hdr2))
    for arm in sorted({r["arm"] for r in recs}):
        idx = {(r["side"], r["source_key"]): r for r in recs if r["arm"] == arm}
        keys = {k for (s, k) in idx if s == "source"}
        cells = []
        for side in ("old", "new"):
            tot = both = 0
            for k in keys:
                src, cp = idx.get(("source", k)), idx.get((side, k))
                if not src or not cp:
                    continue
                for s in seeds:
                    tot += 1
                    if src["per_seed"][str(s)] and cp["per_seed"][str(s)]:
                        both += 1
            cells.append(both / tot if tot else float("nan"))
        print(f"{arm:14s} {len(keys):>6d} {cells[0]:>14.3f} {cells[1]:>14.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "contrastive_v2_success.csv"
    with csv.open("w", encoding="utf-8") as fh:
        fh.write("arm,side,source_key,label,n_seeds,n_misclassified,rate,mean_prob,"
                 "pipeline_iter2_prob,pipeline_iter2_wrong,sim_to_source\n")
        for r in sorted(recs, key=lambda x: (x["arm"], x["side"], x["source_key"])):
            fh.write(f"{r['arm']},{r['side']},{r['source_key']},{r['label']},{r['n_seeds']},"
                     f"{r['n_misclassified']},{r['rate']},{r['mean_prob']},"
                     f"{r['pipeline_iter2_prob']},{int(r['pipeline_iter2_wrong'])},"
                     f"{r['sim_to_source']}\n")
    jl = out_dir / "contrastive_v2_success.jsonl"
    jl.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    print(f"\nwrote {csv}\nwrote {jl}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--in-path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seed", type=int, default=A.SEED)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--fits-dir", type=Path,
                    default=A.REPO / "results_hs_gemma27b_batch_ablation/vintage")
    ap.add_argument("--out-dir", type=Path,
                    default=A.REPO / "results_hs_gemma27b_batch_ablation/contrastive_v2")
    args = ap.parse_args()

    with args.in_path.open(encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    seeds = [args.seed + i for i in range(args.seeds)]

    recs: list[dict] = []
    for arm in args.arm:
        print(f"\n=== {arm} ===", flush=True)
        recs += analyse(arm, rows, seeds, args.iteration, args.fits_dir, args.chunk)
    report(recs, seeds, args.out_dir)


if __name__ == "__main__":
    main()
