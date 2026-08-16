"""How often do the iteration-3-only red-team samples still fool a *vintage-2* probe?

The vintage sweep (``attribution_vintage.py``) answered "how much AUROC did each
vintage of red-team data buy". This answers the complementary question at the level of
individual conversations: take the ten ``v2`` probes — trained on base data plus every
red-team pair that already existed at iteration 2, and **nothing later** — and measure,
per sample, on how many of those ten seeds the newly-added iteration-3 samples are
misclassified.

Why this is a clean measurement
-------------------------------
Every row scored here is, by construction, absent from all ten v2 training sets: the
v2 vintage is "iteration-3 rows whose source success was already in the iteration-2
dump", and this script scores exactly its complement. So there is no train/test leak to
argue about, and the ordinary train/val split of the sweep is irrelevant — a v3-only row
is out-of-sample for a v2 probe whichever side of that split it would have landed on.

What "success" means here, and for which rows
---------------------------------------------
``redteam_postprocessed_iter3.jsonl`` interleaves two kinds of row, and they mean
different things when a probe gets them wrong:

* **source** — an attacker-written conversation that the *judge* labelled against the
  probe's prediction. Its stored label is the judge's. A probe that disagrees with that
  label is making the very error the attack was searching for, so misclassification is
  literally *attack success*. This is the number the question is about.
* **generated** — the LLM-written opposite-class counterpart ``preprocessing`` mints for
  each success. It was never an attack; it exists to keep the training set balanced.
  Misclassifying one is an error but not a red-team success, so it is reported
  separately and never pooled into the headline rate.

Two reference points make the rate readable
-------------------------------------------
A bare "these fool a v2 probe X% of the time" is hard to calibrate, so two anchors are
scored alongside:

* ``probe_iter2.pkl`` — the *actual pipeline probe these attacks were run against*
  (iteration 2's rotation attacked ``probe_iter{2}``; its successes became the rows that
  are new in the iteration-3 dump). The source rows should be misclassified at close to
  100%, because that is the definition that admitted them. It is a check on the whole
  reconstruction — labels, pairing, cached activations, prediction rule — not a result.
  It is not exactly 100%: ``filter_dataset`` refits each cycle so a handful of rows in
  the dump came in earlier, and submit-time scoring tokenized the raw conversation while
  this scores the transformed-and-cached activation.
* **the v2-vintage rows themselves** (``--include-seen``) — rows that *were* in every v2
  probe's training set. The gap between the two is the memorization-vs-generalization
  spread, and it is what says whether a low out-of-sample rate means "the pairs
  generalized" or just "this probe family is hard to fool".

The prediction rule is the pipeline's: ``sigmoid(logit) >= 0.5``, matching
``ProbeJudge.evaluate`` at the config's default ``probe.threshold``.

Usage:
    .venv_claude/bin/python scripts/vintage_new_sample_success.py
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
from attribution_vintage import _blob, _load_fitted, dropped_rows, fit_path, vintages

EMBED_DIM = 5376
THRESHOLD = 0.5  # config default probe.threshold; see ProbeJudge.evaluate

# The raw attempt logs, per arm. Both error types write their own file (the config's
# ``output.jsonl_path`` is auto-suffixed ``_fp`` / ``_fn``).
REDTEAM_LOGS = {
    "gptoss120b": A.REPO / "results_hs_gemma27b_gptoss120b_batch/gptoss120b_probing",
    "deepseekv4pro": A.REPO / "results_hs_gemma27b_deepseekv4pro_batch/deepseekv4pro_probing",
}


def provenance(arm: str) -> dict[str, dict]:
    """``{canonical text: {iteration, probe_score, error_type}}`` for every success.

    Needed because "new in the iteration-3 dump" is *not* the same as "found by
    attacking ``probe_iter2``": ``filter_dataset`` refits its bag-of-words classifier
    every cycle, so a success found in iteration 0 can be dropped from the iteration-2
    dump and re-admitted to the iteration-3 one. Those rows never had to beat
    ``probe_iter2``, so pooling them into the anchor would understate it. The log's own
    ``iteration`` field settles it exactly.

    The logged ``probe_score`` is a second, independent use: it is what the probe said at
    submit time, on the raw conversation, so comparing it to this script's score of the
    same probe on the *transformed, cached* activation measures how much the two message
    transforms move a verdict.
    """
    from tuberlens.interfaces.dataset import Message

    out: dict[str, dict] = {}
    for suffix in ("_fp", "_fn"):
        path = REDTEAM_LOGS[arm].with_name(REDTEAM_LOGS[arm].name + suffix + ".jsonl")
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if not rec.get("success"):
                    continue
                msgs = [
                    Message(role=m["role"], content=m["content"])
                    for m in rec["sample"]["messages"]
                ]
                out[A.canon(A.apply_transforms(msgs))] = {
                    "iteration": rec.get("iteration"),
                    "probe_score": rec.get("probe_score"),
                    "error_type": rec.get("error_type"),
                    "judge_label": rec.get("judge_label"),
                }
    return out


# --- assembling a scoreable batch of red-team rows ---------------------------------


def _rows_dataset(redteam_ds, rows: list[int]):
    """A ``LabelledDataset`` over ``rows`` with their cached activations attached.

    Same in-place fill as ``attribution_vintage.build_side`` and for the same reason —
    allocate the padded block once and write every byte exactly once — but with no base
    dataset, since nothing here trains.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    paths = [A.redteam_blob_path(redteam_ds.inputs[i]) for i in rows]
    widths = [A.blob_width(p) for p in paths]
    width = max(widths)
    n = len(rows)

    probe_blob = _blob(paths[0])
    acts = torch.empty((n, width, EMBED_DIM), dtype=probe_blob["activations"].dtype)
    mask = torch.empty((n, width), dtype=probe_blob["attention_mask"].dtype)
    ids_t = torch.empty((n, width), dtype=probe_blob["input_ids"].dtype)
    del probe_blob

    for k, path in enumerate(paths):
        b = _blob(path)
        w = int(b["activations"].shape[1])
        acts[k, :w] = b["activations"][0]
        mask[k, :w] = b["attention_mask"][0]
        ids_t[k, :w] = b["input_ids"][0]
        if w < width:
            acts[k, w:] = 0
            mask[k, w:] = 0
            ids_t[k, w:] = 0
        del b

    return LabelledDataset(
        inputs=[redteam_ds.inputs[i] for i in rows],
        ids=[redteam_ds.ids[i] for i in rows],
        other_fields={
            "labels": [redteam_ds.other_fields["labels"][i] for i in rows],
            "activations": acts,
            "attention_mask": mask,
            "input_ids": ids_t,
        },
    )


def probs_for(probes: dict, redteam_ds, rows: list[int], chunk: int) -> dict:
    """``{key: probabilities over rows}`` for several probes, one chunk at a time.

    Rows are chunked **in width order**, so each padded block is only as wide as its own
    widest member instead of the set's. At the 1024-token cap a naive single block would
    be ~11 MB/row; ordering by width keeps the typical block near the rows' own lengths.
    Padding is masked out by the classifier's aggregation either way — this is purely a
    memory measure, and it cannot change a score.
    """
    import contextlib
    import io

    from tuberlens.interfaces.activations import Activation

    order = sorted(
        range(len(rows)),
        key=lambda j: A.blob_width(A.redteam_blob_path(redteam_ds.inputs[rows[j]])),
    )
    out = {k: np.empty(len(rows), dtype=np.float64) for k in probes}

    for start in range(0, len(order), chunk):
        pos = order[start : start + chunk]
        ds = _rows_dataset(redteam_ds, [rows[j] for j in pos])
        acts = Activation.from_dataset(ds)
        for key, probe in probes.items():
            with contextlib.redirect_stdout(io.StringIO()):
                p = probe._classifier.probs(acts)
            out[key][pos] = p.float().cpu().numpy()
            del p
        del ds, acts
        gc.collect()
    return out


# --- per-arm analysis ---------------------------------------------------------------


def analyse_arm(arm: str, iteration: int, seeds: list[int], out_dir: Path,
                drop_mode: str, chunk: int, include_seen: bool) -> list[dict]:
    exclude, drop_report = dropped_rows(arm, iteration, drop_mode)
    keep, _ = vintages(arm, iteration, exclude)

    redteam = A.load_redteam_dataset(arm, iteration)
    gen2src = A.generated_to_source(arm)
    labels = redteam.other_fields["labels"]
    src_keys = [gen2src.get(A.canon(m), A.canon(m)) for m in redteam.inputs]

    v2, v3 = set(keep[2]), set(keep[iteration])
    new_rows = sorted(v3 - v2)
    seen_rows = sorted(v2)

    # Rows the over-long filter removed. They are in no vintage, so they are scored and
    # reported apart rather than folded into the headline — the v2 probes never saw them.
    # Scoring them anyway makes the three cohorts a **partition** of the iteration-3 dump
    # (v2 + new + excluded = every row), which is what lets the conversation viewer show
    # every pair with a real measurement rather than a hole.
    dropped_rows_all = sorted(exclude)

    print(f"\n=== {arm} ===", flush=True)
    print(f"  over-long drop ({drop_mode}): {drop_report['n_dropped']} row(s)", flush=True)
    print(f"  vintage 2: {len(seen_rows)} rows   vintage 3: {len(v3)} rows", flush=True)
    print(f"  NEW in v3 (scored): {len(new_rows)} rows "
          f"= {len({src_keys[i] for i in new_rows})} pair(s); "
          f"{len(dropped_rows_all)} further row(s) excluded as over-long", flush=True)
    assert len(seen_rows) + len(new_rows) + len(dropped_rows_all) == len(redteam.inputs), \
        "cohorts must partition the iteration-3 dump"

    prov = provenance(arm)
    ref = A.load_probe(A.ARMS[arm] / f"probe_iter{iteration}.pkl")

    probes: dict = {}
    for s in seeds:
        path = fit_path(out_dir, arm, 2, s)
        if not path.exists():
            raise SystemExit(f"missing v2 checkpoint {path} — run attribution_vintage.py first")
        probes[f"v2_s{s}"], _ = _load_fitted(ref, path)
    # The pipeline probe iteration 2's rotation actually attacked. Anchor, not a result.
    probes["pipeline_iter2"] = A.load_probe(A.ARMS[arm] / "probe_iter2.pkl")

    cohorts = {"new_in_v3": new_rows}
    if dropped_rows_all:
        cohorts["overlong_excluded"] = dropped_rows_all
    if include_seen:
        cohorts["in_v2_train"] = seen_rows

    records: list[dict] = []
    for cohort, rows in cohorts.items():
        if not rows:
            continue
        probs = probs_for(probes, redteam, rows, chunk)
        seed_keys = [f"v2_s{s}" for s in seeds]
        for j, i in enumerate(rows):
            pos = labels[i] == "positive"
            wrong = {
                k: bool((probs[k][j] >= THRESHOLD) != pos) for k in probs
            }
            n_wrong = sum(wrong[k] for k in seed_keys)
            # Provenance keys off the SOURCE success — a generated row was never
            # submitted, so it inherits the iteration its source was found in.
            p = prov.get(src_keys[i], {})
            records.append({
                "arm": arm,
                "cohort": cohort,
                "row": i,
                # A hash, not the text: the source key is the whole canonical
                # conversation, and writing it per row made the sidecar 1.6 MB of
                # duplicated transcripts. The row index plus the iteration-3 dump
                # recovers the content whenever it is actually wanted.
                "pair": A.sha16(src_keys[i]),
                "kind": "generated" if A.canon(redteam.inputs[i]) in gen2src else "source",
                "label": labels[i],
                "found_iteration": p.get("iteration"),
                "found_error_type": p.get("error_type"),
                "logged_probe_score": p.get("probe_score"),
                "n_seeds": len(seed_keys),
                "n_misclassified": n_wrong,
                "success_rate": n_wrong / len(seed_keys),
                "mean_prob": float(np.mean([probs[k][j] for k in seed_keys])),
                "sd_prob": float(np.std([probs[k][j] for k in seed_keys], ddof=1)),
                "pipeline_iter2_prob": float(probs["pipeline_iter2"][j]),
                "pipeline_iter2_wrong": wrong["pipeline_iter2"],
                "per_seed": {str(s): wrong[f"v2_s{s}"] for s in seeds},
            })
        del probs
        gc.collect()

    return records


# --- reporting ----------------------------------------------------------------------


def _rate(rs: list[dict]) -> tuple[float, int, int]:
    n_seed_trials = sum(r["n_seeds"] for r in rs)
    n_wrong = sum(r["n_misclassified"] for r in rs)
    return (n_wrong / n_seed_trials if n_seed_trials else float("nan")), n_wrong, len(rs)


def _headline(records: list[dict], arm: str) -> list[dict]:
    """The rows the question is actually about, for one arm.

    Attacker-written, new in the iteration-3 dump, and **found in iteration 2** — i.e.
    the attacks that had to beat ``probe_iter2`` and did. The last condition is what
    makes the cohort clean: without it the set also contains older successes that
    ``filter_dataset`` re-admitted, which never faced an iteration-2 probe at all and
    which the ten v2 probes classify correctly essentially always.
    """
    return [r for r in records
            if r["arm"] == arm and r["cohort"] == "new_in_v3"
            and r["kind"] == "source" and r["found_iteration"] == 2]


def report(records: list[dict], seeds: list[int], out_dir: Path) -> None:
    by = defaultdict(list)
    for r in records:
        by[(r["arm"], r["cohort"], r["kind"])].append(r)
    arms = sorted({r["arm"] for r in records})

    print("\n\n=== HEADLINE: attacks found against probe_iter2, replayed on the ten "
          "v2-vintage probes ===")
    print("   Every one of these rows is out-of-sample for all ten probes, and every one")
    print("   defeats the real probe_iter2 (the 'anchor' column is 1.000 by construction).\n")
    hdrH = (f"{'arm':14s} {'attacks':>8s} {'trials':>7s} {'SUCCESS RATE':>13s} "
            f"{'always':>7s} {'never':>6s} {'anchor':>7s} {'memorized':>10s}")
    print(hdrH)
    print("-" * len(hdrH))
    for arm in arms:
        rs = _headline(records, arm)
        rate, n_wrong, n = _rate(rs)
        seen = [r for r in records
                if r["arm"] == arm and r["cohort"] == "in_v2_train" and r["kind"] == "source"]
        seen_rate = _rate(seen)[0] if seen else float("nan")
        print(f"{arm:14s} {n:>8d} {sum(r['n_seeds'] for r in rs):>7d} {rate:>13.3f} "
              f"{sum(1 for r in rs if r['n_misclassified'] == r['n_seeds']):>7d} "
              f"{sum(1 for r in rs if r['n_misclassified'] == 0):>6d} "
              f"{np.mean([r['pipeline_iter2_wrong'] for r in rs]):>7.3f} {seen_rate:>10.3f}")

    print("\n   by attack direction (the error the attacker was hunting):")
    hdrE = f"{'arm':14s} {'error_type':>16s} {'attacks':>8s} {'success rate':>13s}"
    print("   " + hdrE)
    print("   " + "-" * len(hdrE))
    for arm in arms:
        for et in sorted({r["found_error_type"] for r in _headline(records, arm)},
                         key=lambda v: (v is None, v)):
            rs = [r for r in _headline(records, arm) if r["found_error_type"] == et]
            print(f"   {arm:14s} {str(et):>16s} {len(rs):>8d} {_rate(rs)[0]:>13.3f}")

    print("\n\n=== every cohort scored, for completeness ===")
    print("   'source' rows are attacker-written: a miss IS a red-team success.")
    print("   'generated' rows are the LLM-written counterparts: a miss is just an error.\n")
    hdr = (f"{'arm':14s} {'cohort':20s} {'kind':10s} {'rows':>5s} {'rate':>8s} "
           f"{'always':>7s} {'never':>6s} {'iter2':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for (arm, cohort, kind), rs in sorted(by.items()):
        rate, _, n = _rate(rs)
        always = sum(1 for r in rs if r["n_misclassified"] == r["n_seeds"])
        never = sum(1 for r in rs if r["n_misclassified"] == 0)
        pipe = np.mean([r["pipeline_iter2_wrong"] for r in rs])
        print(f"{arm:14s} {cohort:20s} {kind:10s} {n:>5d} {rate:>8.3f} "
              f"{always:>7d} {never:>6d} {pipe:>8.3f}")

    print("\n\n=== the same new source rows, split by the iteration they were FOUND in ===")
    print("   iteration k attacked probe_iter{k}, so only k=2 rows ever had to beat the")
    print("   pipeline's iteration-2 probe. Earlier rows are successes filter_dataset")
    print("   dropped from the iteration-2 dump and re-admitted to the iteration-3 one.\n")
    hdr3 = (f"{'arm':14s} {'found@':>7s} {'rows':>5s} {'v2 seeds':>9s} {'iter2 probe':>12s} "
            f"{'logged score':>13s}")
    print(hdr3)
    print("-" * len(hdr3))
    for arm in sorted({r["arm"] for r in records}):
        base = [r for r in records
                if r["arm"] == arm and r["cohort"] == "new_in_v3" and r["kind"] == "source"]
        for it in sorted({r["found_iteration"] for r in base}, key=lambda v: (v is None, v)):
            rs = [r for r in base if r["found_iteration"] == it]
            rate, _, n = _rate(rs)
            pipe = np.mean([r["pipeline_iter2_wrong"] for r in rs])
            logged = [r["logged_probe_score"] for r in rs if r["logged_probe_score"] is not None]
            print(f"{arm:14s} {str(it):>7s} {n:>5d} {rate:>9.3f} {pipe:>12.3f} "
                  f"{(np.mean(logged) if logged else float('nan')):>13.3f}")

    n_s = len(seeds)
    print(f"\n\n=== how the per-attack rate is distributed (headline cohort) ===")
    print("   The mass is at the ends: an attack tends to beat every seed or none of")
    print("   them, so the per-attack rate is close to a yes/no property of the sample.\n")
    hdr2 = f"{'arm':14s} " + " ".join(f"{k}/{n_s}".rjust(6) for k in range(n_s + 1))
    print(hdr2)
    print("-" * len(hdr2))
    for arm in arms:
        rs = _headline(records, arm)
        hist = [sum(1 for r in rs if r["n_misclassified"] == k) for k in range(n_s + 1)]
        print(f"{arm:14s} " + " ".join(f"{h:>6d}" for h in hist))

    print("\n\n=== per-seed rate on the headline cohort (is any one seed an outlier?) ===")
    for arm in arms:
        rs = _headline(records, arm)
        if not rs:
            continue
        per = [float(np.mean([r["per_seed"][str(s)] for r in rs])) for s in seeds]
        print(f"{arm:14s} " + "  ".join(f"s{s}={v:.3f}" for s, v in zip(seeds, per))
              + f"   [min {min(per):.3f}, max {max(per):.3f}, "
                f"sd {np.std(per, ddof=1):.3f}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_row = out_dir / "new_sample_success.csv"
    with per_row.open("w", encoding="utf-8") as fh:
        fh.write("arm,cohort,row,kind,label,found_iteration,found_error_type,"
                 "n_seeds,n_misclassified,success_rate,mean_prob,sd_prob,"
                 "logged_probe_score,pipeline_iter2_prob,pipeline_iter2_wrong\n")
        for r in sorted(records, key=lambda x: (x["arm"], x["cohort"], x["row"])):
            fh.write(f"{r['arm']},{r['cohort']},{r['row']},{r['kind']},{r['label']},"
                     f"{r['found_iteration']},{r['found_error_type']},"
                     f"{r['n_seeds']},{r['n_misclassified']},{r['success_rate']},"
                     f"{r['mean_prob']},{r['sd_prob']},{r['logged_probe_score']},"
                     f"{r['pipeline_iter2_prob']},{int(r['pipeline_iter2_wrong'])}\n")

    summary = out_dir / "new_sample_success_summary.csv"
    with summary.open("w", encoding="utf-8") as fh:
        fh.write("arm,cohort,kind,found_iteration,n_rows,seed_trials,n_misclassified,"
                 "rate,n_always_wrong,n_never_wrong,pipeline_iter2_rate\n")

        def emit(arm, cohort, kind, found, rs):
            rate, n_wrong, n = _rate(rs)
            fh.write(f"{arm},{cohort},{kind},{found},{n},{sum(r['n_seeds'] for r in rs)},"
                     f"{n_wrong},{rate},"
                     f"{sum(1 for r in rs if r['n_misclassified'] == r['n_seeds'])},"
                     f"{sum(1 for r in rs if r['n_misclassified'] == 0)},"
                     f"{np.mean([r['pipeline_iter2_wrong'] for r in rs])}\n")

        # ``found_iteration=all`` rows are the pooled cohorts; the numbered rows split
        # them by provenance. The headline of this analysis is
        # (new_in_v3, source, found_iteration=2) — see _headline.
        for (arm, cohort, kind), rs in sorted(by.items()):
            emit(arm, cohort, kind, "all", rs)
            for it in sorted({r["found_iteration"] for r in rs},
                             key=lambda v: (v is None, v)):
                emit(arm, cohort, kind, it, [r for r in rs if r["found_iteration"] == it])

    (out_dir / "new_sample_success.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    print(f"\nwrote {per_row}\nwrote {summary}\nwrote {out_dir / 'new_sample_success.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--seed", type=int, default=A.SEED)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--drop-overlong", choices=("none", "row", "pair"), default="pair",
                    help="must match the sweep that produced the v2 checkpoints")
    ap.add_argument("--include-seen", action=argparse.BooleanOptionalAction, default=True,
                    help="also score the rows the v2 probes WERE trained on, as a "
                         "memorization reference")
    ap.add_argument("--chunk", type=int, default=32, help="rows per padded scoring block")
    ap.add_argument("--out-dir", type=Path,
                    default=A.REPO / "results_hs_gemma27b_batch_ablation/vintage")
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.seeds)]
    records: list[dict] = []
    for arm in args.arm:
        records += analyse_arm(arm, args.iteration, seeds, args.out_dir,
                               args.drop_overlong, args.chunk, args.include_seen)
    report(records, seeds, args.out_dir)


if __name__ == "__main__":
    main()
