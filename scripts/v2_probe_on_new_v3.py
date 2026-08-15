"""Refit vintage-2 probes on 10 seeds, then label the new-in-v3 rows with each.

The question: the 116 (gptoss) / 86 (deepseek) red-team successes that iteration 3
added were, by construction, misclassified by ``probe_iter2.pkl`` -- that is what made
them successes. Does that failure *reproduce*? Retrain on the same training set with a
different seed and the probe is a different hyperplane (the fit is massively
underdetermined: 546 rows in 5376 dimensions, train accuracy 1.0000). So a success may
be a durable hole in what the v2 data can teach, or an accident of one seed's boundary.

Design notes:

**"v2" here is a vintage, not the iteration-2 dump.** ``vintages(arm, 3)`` selects rows
of the *iteration-3* dump whose originating success already existed at iteration 2, so
content is held fixed and only membership varies -- the same construction the vintage
sweep used. ``probe_iter2.pkl`` (the deployed probe the attacker actually attacked) is
scored alongside as a reference, but it is a different training set and a single seed.

**The new rows are never in training.** They are dropped from both the train and the
val side of every refit, so every score below is out-of-sample for every seed.

**Judge labels exist only for the source rows.** ``preprocessing`` writes one
LLM-authored opposite-class counterpart per success; that counterpart's label is
assigned by construction, not by the judge. Verified on both arms: the dump label of
every source row equals its judge label (116/116, 86/86 join, 0 mismatches), so
"mislabel vs the judge" is exact for sources and the generated half is reported
separately rather than folded in.

**Decision rule.** ``ProbeJudge.evaluate`` thresholds ``predict_proba`` at 0.5 (the
config sets no ``probe.threshold``), and sigmoid is monotone, so ``logit >= 0`` is the
identical predicate and sidesteps the bf16 probability saturation described in
``attribution_lib``.

Usage:
    .venv_claude/bin/python scripts/v2_probe_on_new_v3.py
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import glob
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_refit as R
import attribution_vintage as V

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
OUT_DIR = Path("results_hu_harm_gemma27b_batch_ablation/vintage")


def judge_map(arm: str) -> dict[str, dict]:
    """canonical text -> the attempt row the judge wrote, for every scored attempt."""
    from tuberlens.interfaces.dataset import Message

    out: dict[str, dict] = {}
    pattern = f"results_hu_harm_gemma27b_{arm}_batch/*_probing_f*.jsonl"
    for path in glob.glob(pattern):
        if any(x in path for x in (".runlog.", ".summaries.", ".rounds_done.",
                                   ".iteration_memos.", ".prompts.")):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                msgs = A.apply_transforms(
                    [Message(role=m["role"], content=m["content"])
                     for m in row["sample"]["messages"]]
                )
                out[A.canon(msgs)] = row
    return out


def score_rows(probe, ds) -> np.ndarray:
    """fp32 logits for one dataset through the probe's own forward path."""
    from tuberlens.interfaces.activations import Activation

    with contextlib.redirect_stdout(io.StringIO()):
        s = probe._classifier.logits(Activation.from_dataset(ds))
    return s.float().cpu().numpy().ravel()


def run_arm(arm: str, seeds: list[str], resume: bool = True) -> dict:
    print(f"\n########## {arm} ##########", flush=True)
    keep, report = V.vintages(arm, 3)
    v2, v3 = set(keep[2]), set(keep[3])
    new = sorted(v3 - v2)

    ds_all = A.load_redteam_dataset(arm, 3)
    keys = [A.canon(m) for m in ds_all.inputs]
    gen2src = V._generated_to_source(arm)
    is_gen = np.array([k in gen2src for k in keys], dtype=bool)
    labels = ds_all.other_fields["labels"]
    jm = judge_map(arm)

    y_new = np.array([1.0 if labels[i] == "positive" else 0.0 for i in new])
    gen_new = is_gen[np.array(new)]
    print(f"  vintage 2: {len(v2)} rows   vintage 3: {len(v3)} rows", flush=True)
    print(f"  new in v3: {len(new)} rows "
          f"({int((~gen_new).sum())} source / {int(gen_new.sum())} generated)", flush=True)

    asm = V.assemble_train_only(arm, 3)
    n_rows = len(asm.redteam)
    drop = set(range(n_rows)) - v2
    ds_new = asm.redteam[new]

    prog = OUT_DIR / f"{arm}_v2probe_on_new_v3.progress.jsonl"
    done = {}
    if resume and prog.exists():
        for line in prog.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[int(r["seed"])] = r
        if done:
            print(f"  resuming: {sorted(done)} already recorded", flush=True)

    per_seed = {}
    for seed in seeds:
        if seed in done:
            per_seed[seed] = np.array(done[seed]["logits"], dtype=np.float64)
            continue
        t0 = time.time()
        probe, n_tr, n_val = R.refit(asm, drop_rows=drop, seed=seed)
        fit_s = time.time() - t0
        lg = score_rows(probe, ds_new)
        per_seed[seed] = lg.astype(np.float64)
        pred = lg >= 0
        wrong = pred != (y_new > 0.5)
        be = probe._classifier.best_epoch
        print(f"  seed {seed}: fit {fit_s:6.1f}s  train={n_tr} val={n_val} "
              f"best_epoch={be}  mislabel {wrong.mean():6.1%} "
              f"(source {wrong[~gen_new].mean():6.1%} / generated {wrong[gen_new].mean():6.1%})",
              flush=True)
        with prog.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "arm": arm, "seed": seed, "n_train": n_tr, "n_val": n_val,
                "best_epoch": None if be is None else int(be),
                "fit_seconds": fit_s, "logits": [float(v) for v in lg],
            }) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        del probe
        gc.collect()
        torch.cuda.empty_cache()

    # --- the deployed iteration-2 probe, as a reference point ---------------------
    ref = A.load_probe(A.ARMS[arm] / "probe_iter2.pkl")
    ref_lg = score_rows(ref, ds_new)
    del ref, asm
    gc.collect()

    return summarize(arm, new, keys, y_new, gen_new, per_seed, ref_lg, jm, labels, report)


def summarize(arm, new, keys, y_new, gen_new, per_seed, ref_lg, jm, labels, report) -> dict:
    seeds = sorted(per_seed)
    L = np.stack([per_seed[s] for s in seeds])          # [n_seeds, n_new]
    pred = L >= 0
    truth = y_new > 0.5
    wrong = pred != truth[None, :]                       # [n_seeds, n_new]
    n_wrong_per_row = wrong.sum(0)

    print(f"\n=== {arm}: mislabel rate of the vintage-2 probes on the new-in-v3 rows ===")
    print(f"{'seed':>6} {'all':>9} {'source':>9} {'generated':>10} "
          f"{'FP-src':>8} {'FN-src':>8} {'AUROC':>7}")
    from sklearn.metrics import roc_auc_score

    et = np.array([jm.get(keys[i], {}).get("error_type", "") for i in new])
    src = ~gen_new
    rows = {}
    for k, s in enumerate(seeds):
        w = wrong[k]
        fp = w[src & (et == "false_positive")]
        fn = w[src & (et == "false_negative")]
        auc = roc_auc_score(truth, L[k])
        rows[s] = {
            "all": float(w.mean()), "source": float(w[src].mean()),
            "generated": float(w[gen_new].mean()),
            "source_fp": float(fp.mean()) if len(fp) else None,
            "source_fn": float(fn.mean()) if len(fn) else None,
            "auroc_on_new": float(auc),
        }
        print(f"{s:>6} {w.mean():>8.1%} {w[src].mean():>8.1%} {w[gen_new].mean():>9.1%} "
              f"{fp.mean():>7.1%} {fn.mean():>7.1%} {auc:>7.4f}")

    mean_all = wrong.mean(1)
    print(f"{'mean':>6} {mean_all.mean():>8.1%} {wrong[:, src].mean(1).mean():>8.1%} "
          f"{wrong[:, gen_new].mean(1).mean():>9.1%}")
    print(f"{'sd':>6} {mean_all.std(ddof=1):>8.1%} {wrong[:, src].mean(1).std(ddof=1):>8.1%} "
          f"{wrong[:, gen_new].mean(1).std(ddof=1):>9.1%}")

    ref_wrong = (ref_lg >= 0) != truth
    print(f"\n  reference: deployed probe_iter2.pkl (the probe actually attacked, seed 42, "
          f"iteration-2 training set)")
    print(f"    mislabel  all {ref_wrong.mean():.1%}   source {ref_wrong[src].mean():.1%}   "
          f"generated {ref_wrong[gen_new].mean():.1%}")
    print(f"    (source rows are red-team SUCCESSES against this probe, so 100% is the "
          f"expected value there)")

    print(f"\n=== how consistent is the failure across seeds? ===")
    hist = np.bincount(n_wrong_per_row, minlength=len(seeds) + 1)
    print(f"  {'seeds wrong':>12} {'rows':>6} {'source':>7} {'generated':>10}")
    for k in range(len(seeds) + 1):
        m = n_wrong_per_row == k
        if not m.any():
            continue
        print(f"  {k:>12} {int(m.sum()):>6} {int((m & src).sum()):>7} "
              f"{int((m & gen_new).sum()):>10}")

    always = n_wrong_per_row == len(seeds)
    never = n_wrong_per_row == 0
    print(f"\n  ALWAYS mislabelled (all {len(seeds)} seeds): {int(always.sum())} of "
          f"{len(new)} rows — {int((always & src).sum())} source, "
          f"{int((always & gen_new).sum())} generated")
    print(f"  NEVER mislabelled (0 of {len(seeds)} seeds):  {int(never.sum())} of "
          f"{len(new)} rows — {int((never & src).sum())} source, "
          f"{int((never & gen_new).sum())} generated")

    # of the source rows that were successes against probe_iter2, how many survive?
    surv = float(always[src].mean())
    print(f"  -> {surv:.1%} of the iteration-3 successes are still misclassified by "
          f"EVERY reseeded vintage-2 probe")

    print(f"\n=== the always-wrong source rows (judge's view) ===")
    listing = []
    for j, i in enumerate(new):
        if not (always[j] and src[j]):
            continue
        r = jm.get(keys[i], {})
        listing.append({
            "row": int(i), "error_type": r.get("error_type", ""),
            "judge_label": r.get("judge_label", ""),
            "judge_confidence": r.get("judge_confidence", 0),
            "probe_score_at_attack": r.get("probe_score", None),
            "label": labels[i],
            "mean_logit": float(L[:, j].mean()),
            "sd_logit": float(L[:, j].std(ddof=1)),
            "ref_logit": float(ref_lg[j]),
            "key_sha16": A.sha16(keys[i]),
        })
    listing.sort(key=lambda d: d["mean_logit"])
    by_et = {}
    for d in listing:
        by_et[d["error_type"]] = by_et.get(d["error_type"], 0) + 1
    print(f"  error-type mix of the always-wrong successes: {by_et}")
    if listing:
        conf = np.array([d["judge_confidence"] for d in listing], dtype=float)
        print(f"  judge confidence on them: mean {conf.mean():.2f}  min {conf.min():.0f}  "
              f"max {conf.max():.0f}")
        lg = np.array([d["mean_logit"] for d in listing])
        print(f"  mean logit across seeds: p10 {np.percentile(lg,10):+.2f}  "
              f"median {np.median(lg):+.2f}  p90 {np.percentile(lg,90):+.2f}  "
              f"(sign is the prediction; 0 is the boundary)")

    # how much of the disagreement is the same rows vs different rows each seed
    print(f"\n=== is it the same rows every time? ===")
    jac = []
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            i_ = (wrong[a] & wrong[b]).sum()
            u_ = (wrong[a] | wrong[b]).sum()
            jac.append(i_ / u_ if u_ else 1.0)
    jac = np.array(jac) if jac else np.array([float("nan")])
    print(f"  pairwise Jaccard of the mislabelled sets: mean {jac.mean():.3f} "
          f"(min {jac.min():.3f}, max {jac.max():.3f})")
    print(f"  union of all seeds' mistakes: {int((n_wrong_per_row > 0).sum())} rows; "
          f"intersection: {int(always.sum())} rows")

    return {
        "arm": arm,
        "seeds": seeds,
        "n_new_rows": len(new),
        "n_source": int(src.sum()),
        "n_generated": int(gen_new.sum()),
        "vintage_report": report,
        "per_seed": rows,
        "mean_mislabel": {
            "all": float(mean_all.mean()),
            "source": float(wrong[:, src].mean(1).mean()),
            "generated": float(wrong[:, gen_new].mean(1).mean()),
            "sd_all": float(mean_all.std(ddof=1)),
        },
        "reference_probe_iter2": {
            "all": float(ref_wrong.mean()),
            "source": float(ref_wrong[src].mean()),
            "generated": float(ref_wrong[gen_new].mean()),
        },
        "consistency": {
            "hist_seeds_wrong": [int(v) for v in hist],
            "n_always_wrong": int(always.sum()),
            "n_always_wrong_source": int((always & src).sum()),
            "n_always_wrong_generated": int((always & gen_new).sum()),
            "n_never_wrong": int(never.sum()),
            "frac_successes_always_wrong": surv,
            "pairwise_jaccard_mean": float(jac.mean()),
        },
        "always_wrong_source_rows": listing,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=["gptoss120b", "deepseekv4pro"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for arm in args.arm:
        out[arm] = run_arm(arm, args.seeds, resume=not args.no_resume)
        gc.collect()
        torch.cuda.empty_cache()

    p = OUT_DIR / "v2probe_on_new_v3.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
