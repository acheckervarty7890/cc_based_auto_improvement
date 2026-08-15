"""Refit probes on the *new-in-v3* red-team rows vs on the *v2* rows, 5 seeds each,
then tag every eval row each probe gets right and wrong — and measure the overlap.

The question: iteration 3's red-team successes are, by construction, conversations the
iteration-2 probe got wrong. ``v2_probe_on_new_v3.py`` showed 42% of them stay wrong
under every reseeded vintage-2 probe, i.e. they are a durable hole rather than one
seed's boundary. This asks the complementary question on the *eval* side: **does
training on that hole buy a different probe, or the same one?** If the eval rows the
two conditions get wrong are the same rows, the new data is redundant with what v2
already taught; if they are disjoint, the two red-team vintages are teaching genuinely
different decision surfaces.

Two training conditions, both on top of the same base training data (there is no
red-team-free arm here — that is vintage 0 of ``attribution_vintage.py``):

- **``v2``** — base + every iteration-3 row whose source success already existed at
  iteration 2. 546 rows (gptoss120b) / 706 (deepseekv4pro).
- **``v3new``** — base + exactly the rows ``v2`` excludes, i.e. the successes iteration
  3 added. 232 rows (gptoss120b) / 172 (deepseekv4pro).

The two red-team sets are disjoint by construction and their union is the full
iteration-3 set, so this is a partition, not a nesting.

Three things to keep in view when reading the output:

**The conditions differ in size, not only in content.** ``v2`` carries ~3x the rows.
Any difference in error *rate* is therefore confounded with training-set size; the
overlap *structure* (which rows, not how many) is the part that is not. Reported both
ways, and the within-condition seed spread is printed alongside so a between-condition
difference can be read against the noise floor it has to clear.

**"Vintage" is membership, not content** — same construction as
``attribution_vintage.py``: rows are selected out of the iteration-3 dump, so a pair
whose generated half was later rewritten enters in its rewritten form, and both
conditions carry identical content for the rows they share (they share none, being a
partition — but they share the *base* data).

**The decision rule is ``logit >= 0``.** ``ProbeJudge.evaluate`` thresholds
``predict_proba`` at 0.5 and the configs set no ``probe.threshold``; sigmoid is
monotone, so this is the identical predicate computed in fp32, sidestepping the bf16
probability saturation described in ``attribution_lib``. Note AUROC is reported on the
pipeline scale as well, for comparability with the committed CSVs.

Every eval row is out-of-sample for both conditions — the eval splits are not in any
training set — so no held-out bookkeeping is needed beyond the red-team partition.

Usage:
    .venv_claude/bin/python scripts/heldout_v3_vs_v2_eval_tags.py
    .venv_claude/bin/python scripts/heldout_v3_vs_v2_eval_tags.py --summarize-only
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# tuberlens' trainer wraps every epoch in tqdm on stderr; 20 fits would bury the log.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_refit as R
import attribution_vintage as V

SEEDS = [42, 43, 44, 45, 46]
CONDITIONS = ["v2", "v3new"]
OUT_DIR = Path("results_hu_harm_gemma27b_batch_ablation/heldout_v3_vs_v2")
PROGRESS = OUT_DIR / "eval_tags_progress.jsonl"


# --- eval side --------------------------------------------------------------------


def load_eval_truth(probe) -> dict[str, dict]:
    """``split -> {y, sha16, n}`` — labels and a join key per eval row, loaded once.

    The key is the sha16 of the canonical text *after* the config's message transforms,
    so a tag can be rejoined to the conversation (and to any other analysis in this
    directory) without carrying the text itself in every sidecar row.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    out = {}
    for split in A.EVAL_SPLITS:
        ds = LabelledDataset.load_from(
            A.EVAL_DATASET_DIR / f"{split}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        y = ds.labels_torch().float().cpu().numpy()
        out[split] = {
            "y": y,
            "sha16": [A.sha16(A.canon(m)) for m in ds.inputs],
            "n": len(y),
        }
    return out


def eval_logits(probe, eval_dir: Path) -> dict[str, list[float]]:
    """fp32 logits per eval split, one blob in memory at a time.

    Mirrors ``attribution_vintage.score_streaming``'s memory discipline (mmap, dropped
    before the next split opens) — the four blobs are 4.5 GB and this box has 31 GB with
    the padded red-team activations already resident.
    """
    from tuberlens.interfaces.activations import Activation
    from tuberlens.interfaces.dataset import LabelledDataset

    out = {}
    for split in A.EVAL_SPLITS:
        ds = LabelledDataset.load_from(
            A.EVAL_DATASET_DIR / f"{split}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        blob = torch.load(eval_dir / f"{split}-acts_full.pt", weights_only=False, mmap=True)
        ds = ds.assign(
            activations=blob["activations"],
            attention_mask=blob["attention_mask"],
            input_ids=blob["input_ids"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            s = probe._classifier.logits(Activation.from_dataset(ds))
        out[split] = [float(v) for v in s.float().cpu().numpy().ravel()]
        del ds, blob, s
        gc.collect()
    return out


# --- progress ---------------------------------------------------------------------


def append_progress(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # truncated by a hard kill; that fit simply re-runs
    return rows


# --- the sweep --------------------------------------------------------------------


def run_arm(arm: str, seeds: list[int], eval_dir: Path, resume: bool = True) -> None:
    keep, report = V.vintages(arm, 3)
    v2 = set(keep[2])
    v3 = set(keep[3])
    new = v3 - v2
    all_rows = set(range(report["n_rows_final"]))
    assert v3 == all_rows, "vintage 3 should be the whole iteration-3 dump"

    print(f"\n=== {arm} ===", flush=True)
    print(f"  iteration-3 dump: {len(all_rows)} rows", flush=True)
    print(f"  condition v2   : {len(v2)} red-team rows", flush=True)
    print(f"  condition v3new: {len(new)} red-team rows (disjoint from v2)", flush=True)

    done = {(r["arm"], r["condition"], r["seed"]) for r in read_progress(PROGRESS)} if resume else set()
    todo = [
        (s, c) for s in seeds for c in CONDITIONS if (arm, c, s) not in done
    ]
    if not todo:
        print("  every (condition, seed) already recorded — nothing to do", flush=True)
        return
    print(f"  {len(todo)} fit(s) to run", flush=True)

    asm = V.assemble_train_only(arm, 3)
    truth = load_eval_truth(asm.probe)
    row_sets = {"v2": v2, "v3new": new}

    # Seeds outer, conditions inner: after seed s lands, the sweep holds a COMPLETE
    # paired comparison at s seeds. The other order would leave one whole condition
    # unmeasured until the end, which is the wrong thing to own if the box dies.
    for seed, cond in todo:
        drop = all_rows - row_sets[cond]
        t0 = time.time()
        probe, n_tr, n_val = R.refit(asm, drop_rows=drop, seed=seed)
        fit_s = time.time() - t0
        t1 = time.time()
        logits = eval_logits(probe, eval_dir)
        score_s = time.time() - t1

        accs, aurocs = {}, {}
        for split in A.EVAL_SPLITS:
            s = np.array(logits[split])
            y = truth[split]["y"]
            accs[split] = float(((s >= 0) == (y > 0.5)).mean())
            aurocs[split] = A.auroc_both(y, s)
        print(
            f"  seed {seed} {cond:>5}: fit {fit_s:5.1f}s score {score_s:4.1f}s "
            f"train={n_tr} val={n_val} best_epoch={probe._classifier.best_epoch}",
            flush=True,
        )
        print(
            "     "
            + "  ".join(
                f"{sp.replace('eval_', ''):14s} acc={accs[sp]:.3f} auc={aurocs[sp]['pipeline']:.4f}"
                for sp in A.EVAL_SPLITS
            ),
            flush=True,
        )
        append_progress(
            PROGRESS,
            {
                "arm": arm,
                "condition": cond,
                "seed": seed,
                "n_redteam_rows": len(row_sets[cond]),
                "n_train": n_tr,
                "n_val": n_val,
                "best_epoch": (
                    None if probe._classifier.best_epoch is None
                    else int(probe._classifier.best_epoch)
                ),
                "fit_seconds": fit_s,
                "accuracy": accs,
                "auroc": aurocs,
                "logits": logits,
            },
        )
        del probe
        gc.collect()
        torch.cuda.empty_cache()

    del asm
    gc.collect()
    torch.cuda.empty_cache()


# --- analysis ---------------------------------------------------------------------


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard of two boolean row masks; 1.0 when both are empty."""
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def analyse(arm: str, rows: list[dict], truth: dict) -> dict:
    """Correct/incorrect tags per condition, and the overlap between them."""
    by = defaultdict(dict)
    for r in rows:
        if r["arm"] == arm:
            by[r["condition"]][r["seed"]] = r
    if not all(by.get(c) for c in CONDITIONS):
        return {}

    seeds = sorted(set(by["v2"]) & set(by["v3new"]))
    splits = A.EVAL_SPLITS
    # Concatenate the four splits into one row axis, keeping the offsets so per-split
    # numbers stay recoverable. Everything below is over this flat axis.
    offsets, y_all, key_all, split_all = {}, [], [], []
    at = 0
    for sp in splits:
        offsets[sp] = at
        y_all.append(truth[sp]["y"])
        key_all += truth[sp]["sha16"]
        split_all += [sp] * truth[sp]["n"]
        at += truth[sp]["n"]
    y = np.concatenate(y_all)
    split_all = np.array(split_all)
    n = len(y)

    def logit_matrix(cond: str) -> np.ndarray:
        return np.stack(
            [np.concatenate([np.array(by[cond][s]["logits"][sp]) for sp in splits])
             for s in seeds]
        )

    L = {c: logit_matrix(c) for c in CONDITIONS}
    correct = {c: ((L[c] >= 0) == (y > 0.5)[None, :]) for c in CONDITIONS}
    n_correct = {c: correct[c].sum(0) for c in CONDITIONS}          # 0..len(seeds)
    # Consensus tag: a row is "correct" for a condition when a strict majority of its
    # seeds get it right. Unanimity is reported separately rather than used as the tag,
    # because with 5 seeds it would push most of the mass into an "inconsistent" bucket
    # and hide the structure the overlap question is about.
    maj = {c: n_correct[c] * 2 > len(seeds) for c in CONDITIONS}

    out: dict = {
        "arm": arm,
        "seeds": seeds,
        "n_eval_rows": int(n),
        "n_redteam_rows": {c: by[c][seeds[0]]["n_redteam_rows"] for c in CONDITIONS},
        "per_split_n": {sp: int(truth[sp]["n"]) for sp in splits},
    }

    # --- accuracy / AUROC, mean +/- sd over seeds ---------------------------------
    perf = {}
    for c in CONDITIONS:
        perf[c] = {}
        for sp in list(splits) + ["ALL"]:
            if sp == "ALL":
                acc = correct[c].mean(1)
            else:
                m = split_all == sp
                acc = correct[c][:, m].mean(1)
            auc = np.array([
                by[c][s]["auroc"][sp]["pipeline"] for s in seeds
            ]) if sp != "ALL" else np.array([
                np.mean([by[c][s]["auroc"][x]["pipeline"] for x in splits]) for s in seeds
            ])
            perf[c][sp] = {
                "acc_mean": float(acc.mean()), "acc_sd": float(acc.std(ddof=1)),
                "auroc_mean": float(auc.mean()), "auroc_sd": float(auc.std(ddof=1)),
            }
    out["performance"] = perf

    # --- the overlap question -----------------------------------------------------
    a, b = maj["v2"], maj["v3new"]
    tbl = {
        "correct_both": int((a & b).sum()),
        "correct_v2_only": int((a & ~b).sum()),
        "correct_v3new_only": int((~a & b).sum()),
        "wrong_both": int((~a & ~b).sum()),
    }
    err_a, err_b = ~a, ~b
    # Under independence, |A n B| would be n * P(A) * P(B). The ratio of observed to
    # that is the plain statement of "are these the same rows?" — 1.0 means the two
    # conditions fail independently, higher means shared failures.
    exp_both_wrong = float(err_a.mean() * err_b.mean() * n)
    out["overlap_majority"] = {
        **tbl,
        "jaccard_correct": _jaccard(a, b),
        "jaccard_error": _jaccard(err_a, err_b),
        "n_error_v2": int(err_a.sum()),
        "n_error_v3new": int(err_b.sum()),
        "expected_wrong_both_if_independent": exp_both_wrong,
        "lift_over_independence": (
            float(tbl["wrong_both"] / exp_both_wrong) if exp_both_wrong else None
        ),
        "agreement_rate": float((a == b).mean()),
    }

    # --- within- vs between-condition seed stability ------------------------------
    # The load-bearing comparison. If between-condition Jaccard is indistinguishable
    # from within-condition Jaccard, the training-set difference moved the error set no
    # further than reseeding the same training set does, and "different data" has bought
    # nothing measurable on the eval axis.
    def pairwise(ca: str, cb: str, mask: np.ndarray | None = None) -> dict:
        vals_e, vals_c = [], []
        for i, sa in enumerate(seeds):
            for j, sb in enumerate(seeds):
                if ca == cb and j <= i:
                    continue
                ea = ~correct[ca][i]
                eb = ~correct[cb][j]
                cca, ccb = correct[ca][i], correct[cb][j]
                if mask is not None:
                    ea, eb, cca, ccb = ea[mask], eb[mask], cca[mask], ccb[mask]
                vals_e.append(_jaccard(ea, eb))
                vals_c.append(_jaccard(cca, ccb))
        return {
            "jaccard_error_mean": float(np.mean(vals_e)),
            "jaccard_error_sd": float(np.std(vals_e, ddof=1)) if len(vals_e) > 1 else 0.0,
            "jaccard_correct_mean": float(np.mean(vals_c)),
            "n_pairs": len(vals_e),
        }

    out["seed_stability"] = {
        "within_v2": pairwise("v2", "v2"),
        "within_v3new": pairwise("v3new", "v3new"),
        "between": pairwise("v2", "v3new"),
    }
    out["seed_stability_per_split"] = {
        sp: {
            "within_v2": pairwise("v2", "v2", split_all == sp),
            "within_v3new": pairwise("v3new", "v3new", split_all == sp),
            "between": pairwise("v2", "v3new", split_all == sp),
        }
        for sp in splits
    }

    # --- per-split contingency ----------------------------------------------------
    out["overlap_per_split"] = {}
    for sp in splits:
        m = split_all == sp
        aa, bb = a[m], b[m]
        out["overlap_per_split"][sp] = {
            "n": int(m.sum()),
            "correct_both": int((aa & bb).sum()),
            "correct_v2_only": int((aa & ~bb).sum()),
            "correct_v3new_only": int((~aa & bb).sum()),
            "wrong_both": int((~aa & ~bb).sum()),
            "jaccard_error": _jaccard(~aa, ~bb),
        }

    # --- unanimity ----------------------------------------------------------------
    k = len(seeds)
    out["unanimity"] = {
        c: {
            "always_correct": int((n_correct[c] == k).sum()),
            "always_wrong": int((n_correct[c] == 0).sum()),
            "mixed": int(((n_correct[c] > 0) & (n_correct[c] < k)).sum()),
            "hist_n_correct": [int(v) for v in np.bincount(n_correct[c], minlength=k + 1)],
        }
        for c in CONDITIONS
    }
    hard_a = n_correct["v2"] == 0
    hard_b = n_correct["v3new"] == 0
    easy_a = n_correct["v2"] == k
    easy_b = n_correct["v3new"] == k
    out["unanimity"]["cross"] = {
        "always_wrong_both": int((hard_a & hard_b).sum()),
        "always_wrong_v2_only": int((hard_a & ~hard_b).sum()),
        "always_wrong_v3new_only": int((~hard_a & hard_b).sum()),
        "always_correct_both": int((easy_a & easy_b).sum()),
        "jaccard_always_wrong": _jaccard(hard_a, hard_b),
        "jaccard_always_correct": _jaccard(easy_a, easy_b),
    }

    # --- per-row tags, for downstream joins ---------------------------------------
    out["_tags"] = [
        {
            "split": split_all[i],
            "idx_in_split": int(i - offsets[split_all[i]]),
            "sha16": key_all[i],
            "label": "positive" if y[i] > 0.5 else "negative",
            "v2_n_correct": int(n_correct["v2"][i]),
            "v3new_n_correct": int(n_correct["v3new"][i]),
            "v2_tag": "correct" if maj["v2"][i] else "incorrect",
            "v3new_tag": "correct" if maj["v3new"][i] else "incorrect",
            "v2_mean_logit": float(L["v2"][:, i].mean()),
            "v3new_mean_logit": float(L["v3new"][:, i].mean()),
        }
        for i in range(n)
    ]
    return out


def print_report(res: dict) -> None:
    arm = res["arm"]
    k = len(res["seeds"])
    print(f"\n\n########## {arm} — {k} seeds ##########")
    print(f"  red-team rows: v2={res['n_redteam_rows']['v2']}  "
          f"v3new={res['n_redteam_rows']['v3new']}   eval rows: {res['n_eval_rows']}")

    print("\n--- accuracy (logit>=0) and AUROC (pipeline scale), mean +/- sd over seeds ---")
    print(f"{'split':22s} {'v2 acc':>16s} {'v3new acc':>16s} {'v2 auroc':>16s} {'v3new auroc':>16s}")
    for sp in list(A.EVAL_SPLITS) + ["ALL"]:
        p2, p3 = res["performance"]["v2"][sp], res["performance"]["v3new"][sp]
        print(f"{sp:22s} "
              f"{p2['acc_mean']:.3f}+/-{p2['acc_sd']:.3f} "
              f"  {p3['acc_mean']:.3f}+/-{p3['acc_sd']:.3f} "
              f"  {p2['auroc_mean']:.4f}+/-{p2['auroc_sd']:.4f} "
              f"  {p3['auroc_mean']:.4f}+/-{p3['auroc_sd']:.4f}")

    o = res["overlap_majority"]
    n = res["n_eval_rows"]
    print(f"\n--- overlap of the majority-vote tags ({n} eval rows) ---")
    print(f"{'':>22s} {'v3new correct':>15s} {'v3new incorrect':>17s}")
    print(f"{'v2 correct':>22s} {o['correct_both']:>15d} {o['correct_v2_only']:>17d}")
    print(f"{'v2 incorrect':>22s} {o['correct_v3new_only']:>15d} {o['wrong_both']:>17d}")
    print(f"  agreement on the tag: {o['agreement_rate']:.1%}")
    print(f"  errors: v2 {o['n_error_v2']}, v3new {o['n_error_v3new']}, "
          f"both {o['wrong_both']} "
          f"(Jaccard {o['jaccard_error']:.3f}; "
          f"{o['expected_wrong_both_if_independent']:.1f} expected if independent, "
          f"lift {o['lift_over_independence']:.2f}x)")
    print(f"  correct-set Jaccard: {o['jaccard_correct']:.3f}")

    s = res["seed_stability"]
    print(f"\n--- is that overlap bigger than seed noise? (pairwise Jaccard of error sets) ---")
    for name in ("within_v2", "within_v3new", "between"):
        d = s[name]
        print(f"  {name:14s} error {d['jaccard_error_mean']:.3f}+/-{d['jaccard_error_sd']:.3f}"
              f"   correct {d['jaccard_correct_mean']:.3f}   ({d['n_pairs']} pairs)")

    print("\n--- per split (majority tags) ---")
    print(f"{'split':22s} {'n':>5s} {'both ok':>8s} {'v2 only':>8s} {'v3new only':>11s} "
          f"{'both wrong':>11s} {'errJacc':>8s}")
    for sp, d in res["overlap_per_split"].items():
        print(f"{sp:22s} {d['n']:>5d} {d['correct_both']:>8d} {d['correct_v2_only']:>8d} "
              f"{d['correct_v3new_only']:>11d} {d['wrong_both']:>11d} {d['jaccard_error']:>8.3f}")

    u = res["unanimity"]
    print(f"\n--- unanimity across the {k} seeds ---")
    for c in CONDITIONS:
        print(f"  {c:6s} always correct {u[c]['always_correct']:>4d}   "
              f"always wrong {u[c]['always_wrong']:>4d}   mixed {u[c]['mixed']:>4d}   "
              f"hist(n_correct 0..{k}) {u[c]['hist_n_correct']}")
    x = u["cross"]
    print(f"  always-wrong under BOTH conditions: {x['always_wrong_both']} "
          f"(v2 only {x['always_wrong_v2_only']}, v3new only {x['always_wrong_v3new_only']}; "
          f"Jaccard {x['jaccard_always_wrong']:.3f})")
    print(f"  always-correct under BOTH: {x['always_correct_both']} "
          f"(Jaccard {x['jaccard_always_correct']:.3f})")


def summarize(arms: list[str]) -> None:
    rows = read_progress(PROGRESS)
    if not rows:
        print("no progress rows yet")
        return
    probe = A.load_probe(A.ARMS[arms[0]] / "probe_iter3.pkl")
    truth = load_eval_truth(probe)
    del probe
    gc.collect()

    out = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        res = analyse(arm, rows, truth)
        if not res:
            print(f"\n{arm}: not enough fits recorded yet — skipping")
            continue
        print_report(res)
        tags = res.pop("_tags")
        path = OUT_DIR / f"{arm}_eval_tags.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for t in tags:
                fh.write(json.dumps(t) + "\n")
        print(f"\nwrote {path}")
        out[arm] = res

    if out:
        p = OUT_DIR / "heldout_v3_vs_v2.json"
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--eval-dir", type=Path, default=A.EVAL_ACTIVATIONS_DIR)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for arm in args.arm:
            run_arm(arm, args.seeds, args.eval_dir, resume=not args.no_resume)
    summarize(list(args.arm))


if __name__ == "__main__":
    main()
