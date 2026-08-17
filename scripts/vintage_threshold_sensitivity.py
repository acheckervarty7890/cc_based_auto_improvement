"""Can moving the decision threshold raise the red-team success rate on a vintage increment?

The setting is ``v2_probe_on_new_v3.py``'s, parameterised by vintage: take the ten
**vintage-k** probes — base training data plus every iteration-3 red-team pair whose
source success already existed at iteration ``k``, and nothing later — and score the rows
vintage ``k+1`` *adds*. Every such row is out-of-sample for all ten probes by
construction. That script asked how often they are misclassified at the deployed
operating point; this one asks how that rate moves as the operating point moves.

``--vintage 2`` is the v2-probes-on-new-v3 question; ``--vintage 1`` is the same question
one cycle earlier, and the two are directly comparable because membership is always taken
over the *iteration-3* dump, so only which rows are in training differs — not their
content, their activations, or any hyperparameter.

**The probes here are the saved ones**, ``results_hu_harm_vintage_cross_eval/probes/
{arm}_v{k}_s{seed}.pkl`` (written by ``vintage_probes_cross_eval.py fit``), not fresh
refits. Same construction and same ten seeds as the sweep, so the numbers are comparable
to ``vintage/SUMMARY.md``; at ``k=2`` the committed
``{arm}_v2probe_on_new_v3.progress.jsonl`` logits from the *other* set of refits are
loaded and reported as an independent cross-check on row ordering and scale, never merged
in. No such sidecar exists for other k, so the check is silently skipped there.

What "success" means, and why the direction matters
--------------------------------------------------
A **source** row is attacker-written and its stored label is the judge's, so the probe
disagreeing with that label *is* the error the attack was hunting. The direction is
fixed by the label: an ``error_type=false_positive`` hunt yields rows the judge called
negative (success ⟺ the probe says positive ⟺ ``logit >= tau``), and a
``false_negative`` hunt yields rows the judge called positive (success ⟺
``logit < tau``). So the two directions move **in opposite directions** under any
threshold change, and a single global tau cannot improve both. That is the whole shape
of the answer, and it is why every curve here is reported per direction as well as
pooled — a pooled maximum reached by shoving tau to an extreme is not an attack getting
better, it is the probe being told to answer one class for everything.

**generated** rows (the LLM-written opposite-class counterpart ``preprocessing`` mints
per success) were never attacks; they are scored and reported separately, never pooled
into the headline.

Threshold convention
--------------------
``ProbeJudge.evaluate`` thresholds ``predict_proba`` at ``probe.threshold`` (0.5, unset
in both arms' configs) and sigmoid is monotone, so the sweep runs over the **logit**
threshold ``tau`` and reports the equivalent probability threshold
``t = sigmoid(tau)``. Working in logit space sidesteps the bf16 probability saturation
described in ``attribution_lib`` — above logit ~5.5 every ``predict_proba`` rounds to
exactly 1.0, so a probability grid cannot separate rows that a logit grid can.
``tau = 0`` is the deployed point in every table.

The eval-side cost is measured, not assumed
-------------------------------------------
Raising the success rate by moving tau is only interesting if the probe is still worth
deploying at that tau, so the same ten probes are scored on the four ``eval_dataset_hu_ha``
splits (866 rows, each split exactly 50/50) over the same grid. Three operating points
are then compared head to head: the deployed ``tau = 0``, the tau that maximises pooled
red-team success, and the tau that maximises eval accuracy. All from cached activations —
no gemma-3-27b forward runs anywhere.

Usage::

    .venv_claude/bin/python scripts/vintage_threshold_sensitivity.py --vintage 2
    .venv_claude/bin/python scripts/vintage_threshold_sensitivity.py --vintage 1
    .venv_claude/bin/python scripts/vintage_threshold_sensitivity.py --arm gptoss120b
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
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A  # noqa: E402
import attribution_refit as R  # noqa: E402
import attribution_vintage as V  # noqa: E402
from vintage_probes_cross_eval import multi_probe_logits  # noqa: E402

SEEDS = list(range(42, 52))
# Which vintage's probes are on trial, and hence which increment they are scored on:
# ``--vintage k`` loads the ten ``{arm}_v{k}_s*.pkl`` fits and scores the rows vintage
# ``k+1`` adds. Membership always comes from ``V.vintages(arm, ITERATION)``, i.e. rows of
# the iteration-3 dump — so content is iteration 3's for every k and only *which* rows
# are in training varies. That is what makes k=1 and k=2 comparable to each other.
DEFAULT_VINTAGE = 2
ITERATION = 3
PROBE_DIR = A.REPO / "results_hu_harm_vintage_cross_eval/probes"
OUT_DIR = A.REPO / "results_hu_harm_gemma27b_batch_ablation/threshold_sensitivity"
EMBED_DIM = 5376

# Logit-space grid. Wide enough to reach both saturated ends (|logit| runs to ~25 on
# these probes) and fine enough that the deployed point and its neighbourhood are
# resolved; 0.0 is inserted exactly so the deployed rate is read off the grid itself.
TAU_GRID = np.unique(np.concatenate([
    np.arange(-30.0, 30.0 + 1e-9, 0.25),
    np.array([0.0]),
]))


# --- red-team rows -----------------------------------------------------------------


def judge_map(arm: str) -> dict[str, dict]:
    """canonical text -> the attempt row the judge wrote, for every scored attempt.

    Lifted from ``v2_probe_on_new_v3.py`` so the provenance join is the same one, and
    for the same reason: the dump carries the label but not which error type the attack
    was hunting, nor which iteration found it.
    """
    from tuberlens.interfaces.dataset import Message

    out: dict[str, dict] = {}
    pattern = str(A.REPO / f"results_hu_harm_gemma27b_{arm}_batch/*_probing_f*.jsonl")
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


def _subset(ds, rows: list[int]):
    """``ds`` restricted to ``rows``, activations not yet attached."""
    from tuberlens.interfaces.dataset import LabelledDataset

    return LabelledDataset(
        inputs=[ds.inputs[i] for i in rows],
        ids=[ds.ids[i] for i in rows],
        other_fields={"labels": [ds.other_fields["labels"][i] for i in rows]},
    )


def _blob_width(messages) -> int:
    blob = torch.load(A.redteam_blob_path(messages), weights_only=False, mmap=True)
    return int(blob["activations"].shape[1])


def logits_for(probes: dict, ds, chunk: int) -> dict[str, np.ndarray]:
    """``{probe key: fp32 logits over ds}``, scored in width-sorted chunks.

    Each chunk is padded only to its own widest member, so the resident block stays near
    the rows' true lengths instead of the set's maximum (11 MB/row at the 1024-token
    cap). Padding is masked out inside the classifier either way — this cannot change a
    score, only the peak memory, which on a 15 GB box is the binding constraint.
    """
    from tuberlens.interfaces.activations import Activation

    order = sorted(range(len(ds.inputs)), key=lambda j: _blob_width(ds.inputs[j]))
    out = {k: np.empty(len(order), dtype=np.float64) for k in probes}
    for start in range(0, len(order), chunk):
        pos = order[start:start + chunk]
        part = R._attach_redteam(_subset(ds, pos))
        acts = Activation.from_dataset(part)
        for key, probe in probes.items():
            with contextlib.redirect_stdout(io.StringIO()):
                s = probe._classifier.logits(acts)
            out[key][pos] = s.float().cpu().numpy().ravel()
            del s
        del part, acts
        gc.collect()
    return out


# --- eval rows ---------------------------------------------------------------------


def eval_logits(probes: dict, device: str, batch_size: int) -> dict:
    """Per-split ``(y, logits[n_rows, n_probes])`` for the four hu_ha splits.

    One fused pass per split over all heads (``multi_probe_logits``) rather than one
    ``_classifier.logits`` call per probe: the per-probe path re-reads the whole
    activation tensor, which at 20 heads x 4 splits is hours instead of minutes.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    keys = list(probes)
    params = [A.probe_params(probes[k]) for k in keys]
    W = torch.stack([p[0] for p in params], dim=1).float()
    B = torch.tensor([p[1] for p in params], dtype=torch.float32)
    T = torch.tensor([p[2] for p in params], dtype=torch.float32)

    ref = probes[keys[0]]
    out = {}
    for split in A.EVAL_SPLITS:
        ds = LabelledDataset.load_from(
            A.EVAL_DATASET_DIR / f"{split}.jsonl",
            pos_class_label=ref.pos_class_label,
            neg_class_label=ref.neg_class_label,
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        y = ds.labels_torch().float().cpu().numpy()
        blob = torch.load(A.EVAL_ACTIVATIONS_DIR / f"{split}-acts_full.pt",
                          weights_only=False, mmap=True)
        if len(blob["activations"]) != len(ds):
            raise SystemExit(f"{split}: blob has {len(blob['activations'])} rows, "
                             f"split has {len(ds)}")
        s = multi_probe_logits(blob["activations"], blob["attention_mask"],
                              W, B, T, device, batch_size).numpy()
        out[split] = {"y": y, "logits": s, "keys": keys}
        print(f"  eval {split:24s} {len(ds):>4d} rows x {len(keys)} probes "
              f"(pos rate {y.mean():.2f})", flush=True)
        del ds, blob
        gc.collect()
    return out


# --- curves ------------------------------------------------------------------------


def success_curve(L: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Misclassification rate over ``TAU_GRID``, pooled over seeds and rows.

    ``L`` is ``[n_seeds, n_rows]`` and ``truth`` is the boolean "label is positive".
    Prediction is ``logit >= tau``, so success (the probe disagreeing with the judge) is
    ``(L >= tau) != truth``. Returns one rate per grid point; an empty row set gives
    NaN rather than 0, so a missing direction can't read as a perfect defence.
    """
    if L.shape[1] == 0:
        return np.full(len(TAU_GRID), np.nan)
    pred = L[None, :, :] >= TAU_GRID[:, None, None]      # [n_tau, n_seeds, n_rows]
    return (pred != truth[None, None, :]).mean(axis=(1, 2))


def eval_accuracy_curve(ev: dict, keys: list[str]) -> dict[str, np.ndarray]:
    """Per-split and pooled eval accuracy over ``TAU_GRID``, mean over probes.

    Every hu_ha split is exactly 50/50, so plain accuracy on a split already *is*
    balanced accuracy; pooling the four splits is a row-weighted mean of them, and the
    pooled positive rate stays 0.5 because each split is balanced.
    """
    curves, num, den = {}, np.zeros(len(TAU_GRID)), 0
    for split, d in ev.items():
        s, y = d["logits"], d["y"] > 0.5                  # s: [n_rows, n_probes]
        pred = s[None, :, :] >= TAU_GRID[:, None, None]
        acc = (pred == y[None, :, None]).mean(axis=1)     # [n_tau, n_probes]
        curves[split] = acc.mean(axis=1)
        num += curves[split] * len(y)
        den += len(y)
    curves["ALL"] = num / den
    return curves


def _at(curve: np.ndarray, tau: float) -> float:
    return float(curve[int(np.argmin(np.abs(TAU_GRID - tau)))])


# --- per-arm ------------------------------------------------------------------------


def analyse_arm(arm: str, seeds: list[int], chunk: int, device: str,
                batch_size: int, vintage: int) -> dict:
    nxt = vintage + 1
    print(f"\n########## {arm}  (v{vintage} probes on the rows v{nxt} adds) ##########",
          flush=True)
    keep, _ = V.vintages(arm, ITERATION)
    prev, cur = set(keep[vintage]), set(keep[nxt])
    # Vintages are NOT strictly nested — the iteration-2 filter dropped a handful of
    # iteration-1 pairs that iteration 3 took back — so the increment is the set
    # difference, and the reverse difference is reported rather than assumed empty.
    new = sorted(cur - prev)
    n_lost = len(prev - cur)

    ds_all = A.load_redteam_dataset(arm, ITERATION)
    keys = [A.canon(m) for m in ds_all.inputs]
    gen2src = V._generated_to_source(arm)
    labels = ds_all.other_fields["labels"]
    jm = judge_map(arm)

    is_gen = np.array([keys[i] in gen2src for i in new], dtype=bool)
    truth = np.array([labels[i] == "positive" for i in new], dtype=bool)
    err_type = np.array([jm.get(keys[i], {}).get("error_type", "") for i in new])
    found_it = np.array([jm.get(keys[i], {}).get("iteration", -99) for i in new])
    src = ~is_gen

    print(f"  vintage {vintage}: {len(prev)} rows   vintage {nxt}: {len(cur)} rows "
          f"({n_lost} in v{vintage} but not v{nxt})", flush=True)
    print(f"  new in v{nxt}: {len(new)} rows "
          f"({int(src.sum())} source / {int(is_gen.sum())} generated)", flush=True)

    # Direction is fixed by the judge's label for a source row; the attempt log's
    # error_type must agree, and a mismatch would mean the join is wrong, not that the
    # row is interesting. Checked rather than assumed.
    for i in np.flatnonzero(src):
        et, want = err_type[i], ("false_negative" if truth[i] else "false_positive")
        assert et in ("", want), f"row {new[i]}: error_type {et!r} vs label {labels[new[i]]!r}"

    probes = {}
    for s in seeds:
        p = PROBE_DIR / f"{arm}_v{vintage}_s{s}.pkl"
        if not p.exists():
            raise SystemExit(f"missing saved probe {p}")
        probes[f"v2_s{s}"] = A.load_probe(p)
    # Rows new in dump k+1 were found by iteration k's rotation, which attacked
    # probe_iter{k} — so this is the deployed probe they had to beat, and its
    # mislabel rate on them is a definitional check, not a result.
    probes["pipeline_anchor"] = A.load_probe(A.ARMS[arm] / f"probe_iter{vintage}.pkl")

    print(f"  scoring {len(new)} rows with {len(probes)} probes ...", flush=True)
    lg = logits_for(probes, _subset(ds_all, new), chunk)
    seed_keys = [f"v2_s{s}" for s in seeds]
    L = np.stack([lg[k] for k in seed_keys])              # [n_seeds, n_new]
    ref = lg["pipeline_anchor"]

    del probes["pipeline_anchor"]
    ev = eval_logits(probes, device, batch_size)
    eval_curves = eval_accuracy_curve(ev, seed_keys)
    del probes
    gc.collect()

    # cohorts of interest, all within "new in v3"
    fp_dir = src & ~truth                                  # false-positive hunts
    fn_dir = src & truth                                   # false-negative hunts
    cohorts = {
        "source_all": src,
        "source_fp_hunt": fp_dir,
        "source_fn_hunt": fn_dir,
        "source_found_at_vintage": src & (found_it == vintage),
        "generated": is_gen,
    }
    curves = {name: success_curve(L[:, m], truth[m]) for name, m in cohorts.items()}

    # Two derived curves, because the pooled rate is not the quantity an attacker faces:
    #  - ``attacker_best_direction`` — the attacker *chooses* which error to hunt, so the
    #    reachable rate at a given tau is the better of the two directions, not their mix.
    #  - ``direction_balanced`` — mean of the two directional rates, i.e. the pooled rate
    #    this cohort would have had at a 50/50 FP/FN mix. The measured mix is not 50/50
    #    (it is whatever the rotation happened to find), so the pooled curve above is
    #    partly a fact about the cohort rather than about the probe.
    curves["attacker_best_direction"] = np.maximum(curves["source_fp_hunt"],
                                                  curves["source_fn_hunt"])
    curves["direction_balanced"] = 0.5 * (curves["source_fp_hunt"]
                                          + curves["source_fn_hunt"])

    # the operating points
    pooled = curves["source_all"]
    tau_star = float(TAU_GRID[int(np.nanargmax(pooled))])
    tau_bal = float(TAU_GRID[int(np.nanargmax(curves["direction_balanced"]))])
    tau_eval = float(TAU_GRID[int(np.nanargmax(eval_curves["ALL"]))])

    # per-seed eval-optimal tau, for the "each probe centred on its own scale" reading
    per_seed_tau = []
    for j in range(len(seed_keys)):
        num = np.zeros(len(TAU_GRID))
        for d in ev.values():
            s, y = d["logits"][:, j], d["y"] > 0.5
            num += ((s[None, :] >= TAU_GRID[:, None]) == y[None, :]).sum(axis=1)
        per_seed_tau.append(float(TAU_GRID[int(np.argmax(num))]))

    ref_wrong = (ref >= 0) != truth
    out = {
        "arm": arm,
        "vintage": vintage,
        "scored_increment": f"v{nxt} minus v{vintage}",
        "n_rows_in_vintage": len(prev),
        "n_rows_in_next_vintage": len(cur),
        "n_rows_lost_from_vintage": n_lost,
        "seeds": seeds,
        "n_new_rows": len(new),
        "n_source": int(src.sum()),
        "n_generated": int(is_gen.sum()),
        "n_source_fp_hunt": int(fp_dir.sum()),
        "n_source_fn_hunt": int(fn_dir.sum()),
        "n_source_found_at_vintage": int(cohorts["source_found_at_vintage"].sum()),
        "tau_grid": [float(t) for t in TAU_GRID],
        "curves": {k: [float(v) for v in c] for k, c in curves.items()},
        "eval_curves": {k: [float(v) for v in c] for k, c in eval_curves.items()},
        "operating_points": {
            "deployed": 0.0,
            "max_pooled_source_success": tau_star,
            "max_direction_balanced_success": tau_bal,
            "max_eval_accuracy": tau_eval,
            "per_seed_max_eval_accuracy": per_seed_tau,
        },
        # Per-seed rates at each operating point, so a headline difference can be read
        # against the seed spread it has to clear (the fit is underdetermined — 546 rows
        # in 5376 dimensions — so seed spread is the relevant noise floor, not row noise).
        "per_seed_at_points": {
            name: {
                cohort: _per_seed_rates(L[:, m], truth[m], tau)
                for cohort, m in cohorts.items()
            }
            for name, tau in (("deployed", 0.0), ("max_pooled", tau_star),
                              ("max_eval_accuracy", tau_eval))
        },
        # Local sensitivity: how many percentage points the rate moves per unit of logit
        # threshold at the deployed point, and how much of the cohort sits within one
        # logit unit of the boundary (the rows a small recalibration would flip).
        "sensitivity_at_deployed": {
            cohort: {
                "d_rate_per_logit": float((_at(curves[cohort], 0.5)
                                           - _at(curves[cohort], -0.5))),
                "frac_within_1_logit": float((np.abs(L[:, m]) < 1.0).mean())
                if m.any() else None,
                "frac_within_2_logits": float((np.abs(L[:, m]) < 2.0).mean())
                if m.any() else None,
            }
            for cohort, m in cohorts.items()
        },
        "reference_anchor_wrong": {
            "probe": f"probe_iter{vintage}.pkl",
            "source": float(ref_wrong[src].mean()),
            "source_found_at_vintage": float(
                ref_wrong[cohorts["source_found_at_vintage"]].mean())
            if cohorts["source_found_at_vintage"].any() else None,
            "generated": float(ref_wrong[is_gen].mean()),
        },
        "logit_stats": {
            "source_fp_hunt": _logit_stats(L[:, fp_dir]),
            "source_fn_hunt": _logit_stats(L[:, fn_dir]),
        },
    }

    # Independent cross-check against the other set of refits. That sidecar exists
    # only for the v2-on-new-v3 question, so it is consulted only there.
    prog = (A.REPO / "results_hu_harm_gemma27b_batch_ablation/vintage"
            / f"{arm}_v2probe_on_new_v3.progress.jsonl")
    if vintage == 2 and prog.exists():
        other = {}
        for line in prog.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            other[int(r["seed"])] = np.asarray(r["logits"], dtype=np.float64)
        common = [s for s in seeds if s in other and len(other[s]) == len(new)]
        if common:
            O = np.stack([other[s] for s in common])
            M = np.stack([lg[f"v2_s{s}"] for s in common])
            agree = float(((O >= 0) == (M >= 0)).mean())
            out["cross_check_vs_refits"] = {
                "seeds": common,
                "sign_agreement": agree,
                "pearson_r": float(np.corrcoef(O.ravel(), M.ravel())[0, 1]),
                # over ALL new rows (source + generated), matching v2_probe_on_new_v3's
                # "all" column — not the source-only headline of this script.
                "refit_rate_at_deployed_all_new": float(((O >= 0) != truth[None, :]).mean()),
                "saved_rate_at_deployed_all_new": float(((M >= 0) != truth[None, :]).mean()),
            }
    return out


def _per_seed_rates(L: np.ndarray, truth: np.ndarray, tau: float) -> dict:
    """Mean/sd/min/max over seeds of the misclassification rate at one ``tau``."""
    if L.shape[1] == 0:
        return {}
    per = ((L >= tau) != truth[None, :]).mean(axis=1)
    return {"mean": float(per.mean()),
            "sd": float(per.std(ddof=1)) if len(per) > 1 else 0.0,
            "min": float(per.min()), "max": float(per.max()),
            # kept per seed so a between-threshold contrast can be PAIRED: the same fit
            # scored at two thresholds shares everything except the cut, so the paired sd
            # is the right error bar and it is far tighter than the across-seed one.
            "per_seed": [float(v) for v in per]}


def _logit_stats(L: np.ndarray) -> dict:
    if L.size == 0:
        return {}
    v = L.ravel()
    return {
        "mean": float(v.mean()),
        "p10": float(np.percentile(v, 10)),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "frac_positive_side": float((v >= 0).mean()),
    }


# --- reporting ----------------------------------------------------------------------


def report(results: dict) -> None:
    def prob(tau):
        return 1.0 / (1.0 + np.exp(-tau))

    for arm, r in results.items():
        g = np.asarray(r["tau_grid"])
        c = {k: np.asarray(v) for k, v in r["curves"].items()}
        e = {k: np.asarray(v) for k, v in r["eval_curves"].items()}
        ops = r["operating_points"]

        print(f"\n\n=== {arm}: v{r['vintage']} probes on the {r['n_source']} source rows "
              f"{r['scored_increment']} adds, as the threshold moves ===")
        print(f"    {r['n_source_fp_hunt']} false-positive hunts (judge label negative) "
              f"/ {r['n_source_fn_hunt']} false-negative hunts (judge label positive)")
        print(f"    the probe they were found against "
              f"({r['reference_anchor_wrong']['probe']}) gets "
              f"{r['reference_anchor_wrong']['source']:.1%} of them wrong "
              f"(the definition that admitted them)\n")
        hdr = (f"{'tau':>7} {'t=sigmoid':>10} {'pooled':>8} {'FP-hunt':>8} {'FN-hunt':>8} "
               f"{'best dir':>9} {'balanced':>9} {'generated':>10} {'eval acc':>9}")
        print(hdr)
        print("-" * len(hdr))
        for tau in [-25, -10, -5, -2, -1, 0, 1, 2, 5, 10, 25]:
            mark = "  <- deployed" if tau == 0 else ""
            print(f"{tau:>7.2f} {prob(tau):>10.4f} "
                  f"{_at(c['source_all'], tau):>8.1%} {_at(c['source_fp_hunt'], tau):>8.1%} "
                  f"{_at(c['source_fn_hunt'], tau):>8.1%} "
                  f"{_at(c['attacker_best_direction'], tau):>9.1%} "
                  f"{_at(c['direction_balanced'], tau):>9.1%} "
                  f"{_at(c['generated'], tau):>10.1%} "
                  f"{_at(e['ALL'], tau):>9.1%}{mark}")
        print("    'best dir' = max of the two directions: the attacker picks which error "
              "to hunt, so that is the\n    rate a rotation aimed at this operating point "
              "could reach. 'balanced' = mean of the two, i.e.\n    the pooled rate at a "
              "50/50 FP/FN mix instead of this cohort's measured mix.")

        print(f"\n    operating points, head to head")
        hdr2 = (f"{'point':28s} {'tau':>7} {'t':>8} {'pooled':>8} {'FP-hunt':>8} "
                f"{'FN-hunt':>8} {'best dir':>9} {'eval acc':>9}")
        print("    " + hdr2)
        print("    " + "-" * len(hdr2))
        for name, tau in (("deployed (t=0.5)", 0.0),
                          ("max pooled success", ops["max_pooled_source_success"]),
                          ("max balanced-direction", ops["max_direction_balanced_success"]),
                          ("max eval accuracy", ops["max_eval_accuracy"])):
            print(f"    {name:28s} {tau:>7.2f} {prob(tau):>8.4f} "
                  f"{_at(c['source_all'], tau):>8.1%} {_at(c['source_fp_hunt'], tau):>8.1%} "
                  f"{_at(c['source_fn_hunt'], tau):>8.1%} "
                  f"{_at(c['attacker_best_direction'], tau):>9.1%} "
                  f"{_at(e['ALL'], tau):>9.1%}")

        print(f"\n    seed spread of the pooled source rate (ten fits of the same vintage)")
        for point in ("deployed", "max_pooled", "max_eval_accuracy"):
            st = r["per_seed_at_points"][point].get("source_all") or {}
            if st:
                print(f"      {point:20s} {st['mean']:.1%} +/- {st['sd']:.1%} "
                      f"(min {st['min']:.1%}, max {st['max']:.1%})")
        # The practically meaningful question: can the rate be raised WITHOUT making the
        # probe worse? A pooled maximum that costs 30 points of eval accuracy is the probe
        # being broken, not the attack being better, so the frontier is reported under an
        # explicit eval-accuracy floor.
        dep_acc = _at(e["ALL"], 0.0)
        print(f"\n    constrained frontier — best reachable success subject to an eval "
              f"accuracy floor\n      (deployed eval accuracy is {dep_acc:.1%})")
        hdr4 = (f"{'floor':>11} {'objective':>14} {'tau':>7} {'t':>8} {'pooled':>8} "
                f"{'FP-hunt':>8} {'FN-hunt':>8} {'eval acc':>9}")
        print("      " + hdr4)
        print("      " + "-" * len(hdr4))
        for label, floor in (("no loss", dep_acc), ("-2 points", dep_acc - 0.02),
                             ("-5 points", dep_acc - 0.05)):
            ok = e["ALL"] >= floor
            if not ok.any():
                continue
            # Each objective gets its OWN tau — reporting one row's tau next to another
            # row's rate would be a different threshold in every column.
            for obj in ("source_all", "attacker_best_direction"):
                j = int(np.nanargmax(np.where(ok, c[obj], -np.inf)))
                print(f"      {label:>11} {obj.replace('source_all', 'pooled')[:14]:>14} "
                      f"{g[j]:>7.2f} {prob(g[j]):>8.4f} {c['source_all'][j]:>8.1%} "
                      f"{c['source_fp_hunt'][j]:>8.1%} {c['source_fn_hunt'][j]:>8.1%} "
                      f"{e['ALL'][j]:>9.1%}")

        dep = np.asarray(r["per_seed_at_points"]["deployed"]["source_all"]["per_seed"])
        for point, label in (("max_pooled", "max pooled"),
                             ("max_eval_accuracy", "max eval accuracy")):
            alt = np.asarray(r["per_seed_at_points"][point]["source_all"]["per_seed"])
            d = alt - dep
            print(f"      PAIRED {label} - deployed: {d.mean():+.1%} +/- {d.std(ddof=1):.1%} "
                  f"(same fit, two cuts; {int((d > 0).sum())}/{len(d)} seeds improve)")

        print(f"\n    local sensitivity at the deployed point")
        for cohort in ("source_all", "source_fp_hunt", "source_fn_hunt"):
            s = r["sensitivity_at_deployed"][cohort]
            if s.get("frac_within_1_logit") is None:
                continue
            print(f"      {cohort:16s} {s['d_rate_per_logit']:+.1%} per unit of logit "
                  f"threshold;  {s['frac_within_1_logit']:.1%} of (row, seed) scores lie "
                  f"within 1 logit of the boundary, {s['frac_within_2_logits']:.1%} within 2")

        pst = np.asarray(ops["per_seed_max_eval_accuracy"], dtype=float)
        print(f"\n    per-seed eval-optimal tau: mean {pst.mean():+.2f} "
              f"(min {pst.min():+.2f}, max {pst.max():+.2f}) — the deployed point is 0")

        for d, key in (("FP-hunt", "source_fp_hunt"), ("FN-hunt", "source_fn_hunt")):
            st = r["logit_stats"].get(key) or {}
            if st:
                print(f"    {d:8s} logits across seeds: p10 {st['p10']:+.2f}  "
                      f"median {st['median']:+.2f}  p90 {st['p90']:+.2f}  "
                      f"{st['frac_positive_side']:.1%} on the positive side")

        print(f"\n    per-split eval accuracy at the three points")
        hdr3 = f"{'split':26s}" + "".join(f"{n:>12}" for n in ("deployed", "max succ", "max eval"))
        print("    " + hdr3)
        print("    " + "-" * len(hdr3))
        for sp in list(e):
            print(f"    {sp:26s}"
                  + "".join(f"{_at(e[sp], t):>12.1%}" for t in
                            (0.0, ops["max_pooled_source_success"], ops["max_eval_accuracy"])))

        cc = r.get("cross_check_vs_refits")
        if cc:
            print(f"\n    cross-check vs the independent refits' committed logits "
                  f"({len(cc['seeds'])} seeds): sign agreement {cc['sign_agreement']:.1%}, "
                  f"r={cc['pearson_r']:.4f}; rate at the deployed point over ALL new rows "
                  f"{cc['refit_rate_at_deployed_all_new']:.1%} (refits) vs "
                  f"{cc['saved_rate_at_deployed_all_new']:.1%} (saved probes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=["gptoss120b", "deepseekv4pro"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--chunk", type=int, default=24,
                    help="red-team rows per padded scoring block")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4, help="eval rows per fused pass")
    ap.add_argument("--vintage", type=int, default=DEFAULT_VINTAGE,
                    help="probe vintage k; scored on the rows vintage k+1 adds")
    args = ap.parse_args()
    if not 0 <= args.vintage < ITERATION:
        raise SystemExit(f"--vintage must be in [0, {ITERATION})")

    results = {}
    for arm in args.arm:
        results[arm] = analyse_arm(arm, args.seeds, args.chunk, args.device,
                                   args.batch_size, args.vintage)
        gc.collect()
        torch.cuda.empty_cache()

    report(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"v{args.vintage}_to_v{args.vintage + 1}"
    p = OUT_DIR / f"threshold_sensitivity_{tag}.json"
    p.write_text(json.dumps(results, indent=1), encoding="utf-8")

    csv = OUT_DIR / f"threshold_curves_{tag}.csv"
    with csv.open("w", encoding="utf-8") as fh:
        fh.write("arm,tau,prob_threshold,cohort,rate\n")
        for arm, r in results.items():
            for name, curve in list(r["curves"].items()) + [
                    (f"eval_{k}", v) for k, v in r["eval_curves"].items()]:
                for tau, val in zip(r["tau_grid"], curve):
                    fh.write(f"{arm},{tau},{1/(1+np.exp(-tau))},{name},{val}\n")
    print(f"\nwrote {p}\nwrote {csv}")


if __name__ == "__main__":
    main()
