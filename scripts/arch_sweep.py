"""Does a different probe *architecture* reach the eval rows red-teaming cannot?

``docs/heldout_v3_vs_v2_overlap.md`` ends with 31 eval rows misclassified by all eight
probe families it built — 2 attacker arms x 2 disjoint red-team vintages x with/without
the base training data x 5 seeds — and concludes that what fixes them "is not more of the
same red-teaming, and it is not the base training data either. It is a different eval
concept boundary or a different probe architecture." This script is the architecture half
of that, and the 31-row core is its primary readout: an architecture that only moves the
mean AUROC has done what a reseed does, whereas one that recovers core rows has reached
something no red-team vintage did.

Every fit runs off activations already on disk, so no gemma-3-27b forward is ever
executed and the whole sweep is minutes-per-fit rather than the days extraction cost.

Design
------
**The full training set, always.** Unlike ``heldout_v3_vs_v2_eval_tags.py``, nothing is
held out: every fit trains on the base data plus the whole iteration-3 red-team dump, so
the *only* thing that varies within an arm is the architecture. That makes the comparison
against the deployed ``linear_then_softmax`` head an architecture contrast and nothing
else.

**Two axes, separated deliberately** (see ``probe_architectures``' docstring):
pooling (how per-token evidence becomes a sequence logit) and feature-wise
non-linearity (a linear vs. an MLP readout). ``why_last_iteration_adds_nothing.md`` §2
argues these are not equally promising — the per-split *linear* ceiling on
``eval_ant_hh`` is 0.96 AUROC against 0.77 achieved, so capacity is not the binding
constraint — and the architecture set is built to let that prediction be checked rather
than assumed:

===========================  =========================  ==========================
architecture                 pooling                    readout
===========================  =========================  ==========================
``linear_then_softmax``      logit-weighted softmax     linear    (the baseline)
``attention``                decoupled attention query  linear
``pre_mean``                 mean                       linear
``difference_of_means``      mean                       linear, closed form
``lda_shrinkage``            mean                       linear, closed form whitened
``mlp_then_softmax``         logit-weighted softmax     MLP
``mean_then_mlp``            mean                       MLP
``attention_then_mlp``       decoupled attention query  MLP
===========================  =========================  ==========================

Reading down the pooling column with the readout held fixed isolates pooling; reading
across a row isolates non-linearity.

**Two decision rules**, as in the heldout sweep: ``raw`` (``logit >= 0``, what a deployed
probe does) and ``balanced`` (predict the top half of each split by logit). The probes
here are badly off-centre — one heldout fit ranked ``eval_ai_dilemmas`` at AUROC 0.987
while scoring 0.581 accuracy — so the raw rule alone measures a global offset as much as
it measures understanding, and architectures differ in how well-centred they leave the
logits. Every hu_ha split is exactly 50/50, so the split median is the operating point
matching the true prevalence.

**The best-epoch fix is on for every arm.** ``build_probe`` restores the epoch early
stopping selects (``docs/attribution_findings.md`` §1 documents the shallow-copy no-op).
Applying it uniformly is required for fairness — early stopping is the main overfitting
control for the MLP heads — and it means these numbers are **not** comparable to the
committed comparison CSVs. ``--legacy-best-epoch`` re-runs the baseline the old way so
the size of that effect is measured rather than assumed.

Usage
-----
    .venv_claude/bin/python scripts/arch_sweep.py                 # both arms, 5 seeds
    .venv_claude/bin/python scripts/arch_sweep.py --summarize-only
    .venv_claude/bin/python scripts/arch_sweep.py --sensitivity   # hidden_dim x wd
    .venv_claude/bin/python scripts/arch_sweep.py --legacy-best-epoch

The sweep resumes at ``(arm, architecture, seed)`` granularity from
``arch_progress.jsonl``; each row carries that probe's fp32 logits on all 866 eval rows,
so every table is re-derivable with ``--summarize-only`` and no refitting.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# tuberlens' trainer wraps every epoch in tqdm on stderr; ~90 fits would bury the log.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_refit as R
import attribution_vintage as V

OUT_DIR = A.REPO / "results_hu_harm_gemma27b_batch_ablation/arch_sweep"
PROGRESS = OUT_DIR / "arch_progress.jsonl"
HELDOUT_DIR = A.REPO / "results_hu_harm_gemma27b_batch_ablation/heldout_v3_vs_v2"
NOBASE_DIR = A.REPO / "results_hu_harm_gemma27b_batch_ablation/heldout_v3_vs_v2_nobase"

SEEDS = [42, 43, 44, 45, 46]
ITERATION = 3

#: The baseline is first so a resumed run establishes the comparison point early, and
#: the two closed-form architectures are last because they cost seconds rather than
#: minutes — if the box dies mid-sweep, the expensive arms are the ones already banked.
ARCHITECTURES = [
    "linear_then_softmax",   # the deployed head
    "attention",             # decoupled pooling query, linear readout
    "mlp_then_softmax",      # deployed pooling, MLP readout
    "attention_then_mlp",    # both
    "mean_then_mlp",         # fixed pooling, MLP readout
    "pre_mean",              # fixed pooling, linear readout
    "difference_of_means",   # mass-mean, closed form
    "lda_shrinkage",         # whitened mass-mean, closed form
]

#: --sensitivity: is an MLP result an artefact of its capacity / regularisation rather
#: than of the architecture? Run on one arm only; the point is the spread, not a second
#: full comparison.
SENSITIVITY_GRID = [
    {"hidden_dim": 16, "weight_decay": 1e-3},
    {"hidden_dim": 16, "weight_decay": 1e-2},
    {"hidden_dim": 64, "weight_decay": 1e-2},
    {"hidden_dim": 256, "weight_decay": 1e-3},
]
SENSITIVITY_ARCH = "mlp_then_softmax"


# --- eval side ----------------------------------------------------------------------


def load_eval_truth(probe) -> dict[str, dict]:
    """``split -> {y, sha16, n}``: labels and a join key per eval row, loaded once.

    The key is the sha16 of the canonical text *after* the config's message transforms,
    which is what ``heldout_v3_vs_v2_eval_tags.py`` writes into its per-row tag dumps —
    so the hard core can be joined to these fits without carrying any text around.
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
        out[split] = {
            "y": ds.labels_torch().float().cpu().numpy(),
            "sha16": [A.sha16(A.canon(m)) for m in ds.inputs],
            "n": len(ds),
        }
    return out


def eval_logits(probe, eval_dir: Path) -> dict[str, list[float]]:
    """fp32 logits per eval split, one blob in memory at a time.

    Mirrors ``heldout_v3_vs_v2_eval_tags.eval_logits``' memory discipline: the four eval
    blobs are 4.5 GB and this box has 31 GB with the padded red-team activations already
    resident, so each is mmap'd and dropped before the next is opened.
    """
    import contextlib
    import io

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
        blob = torch.load(
            eval_dir / f"{split}-acts_full.pt", weights_only=False, mmap=True
        )
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


# --- progress -----------------------------------------------------------------------


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


def _variant_key(row: dict) -> tuple:
    return (row["arm"], row["architecture"], row["seed"], row.get("variant", ""))


# --- the sweep ----------------------------------------------------------------------


def run_arm(arm: str, seeds: list[int], architectures: list[str], eval_dir: Path,
            resume: bool = True, variants: list[tuple[str, dict]] | None = None,
            legacy_best_epoch: bool = False) -> None:
    """Fit every (architecture, seed) for one arm on the full iteration-3 training set."""
    variants = variants or [("", {})]
    print(f"\n=== {arm} ===", flush=True)

    done = {_variant_key(r) for r in read_progress(PROGRESS)} if resume else set()
    todo = [
        (s, arch, vname, vhp)
        for s in seeds
        for arch in architectures
        for vname, vhp in variants
        if (arm, arch, s, vname) not in done
    ]
    if not todo:
        print("  every (architecture, seed) already recorded — nothing to do", flush=True)
        return
    print(f"  {len(todo)} fit(s) to run", flush=True)

    asm = V.assemble_train_only(arm, ITERATION)
    truth = load_eval_truth(asm.probe)

    # Seeds outer, architectures inner: after seed s lands, the sweep holds a COMPLETE
    # comparison across every architecture at s seeds. The other order would leave whole
    # architectures unmeasured until the end, which is the wrong thing to own if the box
    # dies mid-run.
    for seed, arch, vname, vhp in todo:
        hp = dict(vhp)
        # weight_decay lives inside optimizer_args, which the grid names flat.
        if "weight_decay" in hp:
            hp["optimizer_args"] = {
                **_default_optimizer_args(arch),
                "weight_decay": hp.pop("weight_decay"),
            }
        t0 = time.time()
        try:
            probe, n_tr, n_val = R.refit(
                asm, seed=seed, arch=arch, hyperparams=hp or None
            )
        except Exception as exc:  # noqa: BLE001 — one arm failing must not kill the sweep
            print(f"  seed {seed} {arch:22s} FAILED: {exc!r}", flush=True)
            append_progress(PROGRESS, {
                "arm": arm, "architecture": arch, "seed": seed, "variant": vname,
                "error": repr(exc),
            })
            continue
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

        n_params = int(sum(p.numel() for p in probe._classifier.model.parameters()))
        tag = f"{arch}{('/' + vname) if vname else ''}"
        print(
            f"  seed {seed} {tag:28s} fit {fit_s:6.1f}s score {score_s:4.1f}s "
            f"params={n_params:>8d} "
            f"best_epoch={getattr(probe._classifier, 'best_epoch', None)}",
            flush=True,
        )
        print(
            "     " + "  ".join(
                f"{sp.replace('eval_', ''):14s} acc={accs[sp]:.3f} "
                f"auc={aurocs[sp]['pipeline']:.4f}"
                for sp in A.EVAL_SPLITS
            ),
            flush=True,
        )
        append_progress(PROGRESS, {
            "arm": arm,
            "architecture": arch,
            "variant": vname,
            "hyperparams": {k: v for k, v in hp.items()},
            "seed": seed,
            "legacy_best_epoch": legacy_best_epoch,
            "n_train": n_tr,
            "n_val": n_val,
            "n_params": n_params,
            "best_epoch": (
                None if getattr(probe._classifier, "best_epoch", None) is None
                else int(probe._classifier.best_epoch)
            ),
            "fit_seconds": fit_s,
            "accuracy": accs,
            "auroc": aurocs,
            "logits": logits,
        })
        del probe
        gc.collect()
        torch.cuda.empty_cache()

    del asm
    gc.collect()
    torch.cuda.empty_cache()


def _default_optimizer_args(arch: str) -> dict:
    from agentic_redteam.probe_architectures import default_hyperparams

    return dict(default_hyperparams(arch).get("optimizer_args", {}))


# --- analysis -----------------------------------------------------------------------


def _balanced_pred(logits: np.ndarray, n_pos: int) -> np.ndarray:
    """Predict the top ``n_pos`` logits positive — the bias-free rule.

    Same construction as ``heldout_v3_vs_v2_eval_tags._balanced_pred``, and load-bearing
    for the same reason: these probes are badly off-centre, and architectures differ in
    how off-centre they leave the logits, so the raw rule would rank architectures partly
    by calibration. Every hu_ha split is exactly 50/50, so the split median is the
    operating point matching the true prevalence, and this rule depends only on ordering.
    """
    out = np.zeros(len(logits), dtype=bool)
    out[np.argsort(-logits, kind="stable")[:n_pos]] = True
    return out


def _flat_truth(truth: dict) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Concatenate the four splits onto one row axis, keeping offsets recoverable."""
    offsets, y_all, key_all, split_all = {}, [], [], []
    at = 0
    for sp in A.EVAL_SPLITS:
        offsets[sp] = at
        y_all.append(truth[sp]["y"])
        key_all += truth[sp]["sha16"]
        split_all += [sp] * truth[sp]["n"]
        at += truth[sp]["n"]
    return np.concatenate(y_all), np.array(split_all), key_all, offsets


def load_hard_core() -> tuple[dict[str, set[tuple[str, int]]], dict]:
    """The rows every probe family in the heldout experiment got wrong.

    Recomputed from the committed per-row tag dumps rather than hard-coded, so it tracks
    those files if they are ever re-derived. Returns the **eight-family** core (both
    attacker arms x both red-team vintages x with and without the base data), which under
    the balanced rule is the 31-row set ``heldout_v3_vs_v2_overlap.md`` §5 reports.

    Keyed on ``(split, idx_in_split)``, **not** on the content hash. The four hu_ha eval
    splits hold 866 rows but only 825 distinct conversations — 41 rows repeat, nearly all
    across splits — so a content join silently pulls in the duplicate's twin and
    over-counts the core (40 rows for a 31-row core, measured). Position is exact and
    matches the counts the heldout write-up published.
    """
    def four_way(d: Path, rule: str) -> set[tuple[str, int]]:
        cores = []
        for arm in sorted(A.ARMS):
            path = d / f"{arm}_eval_tags.jsonl"
            if not path.exists():
                return set()
            rows = [json.loads(line) for line in path.open(encoding="utf-8")]
            cores.append({
                (r["split"], r["idx_in_split"]) for r in rows
                if r[f"{rule}_v2_tag"] == "incorrect"
                and r[f"{rule}_v3new_tag"] == "incorrect"
            })
        return cores[0] & cores[1] if len(cores) == 2 else set()

    info = {}
    cores = {}
    for rule in ("raw", "balanced"):
        with_base = four_way(HELDOUT_DIR, rule)
        no_base = four_way(NOBASE_DIR, rule)
        cores[rule] = with_base & no_base
        info[rule] = {
            "four_way_with_base": len(with_base),
            "four_way_no_base": len(no_base),
            "eight_family_core": len(cores[rule]),
        }
    return cores, info


def analyse(rows: list[dict], truth: dict) -> dict:
    """Per-architecture performance, per-arm and pooled, plus hard-core recovery."""
    y, split_all, key_all, offsets = _flat_truth(truth)
    truth_bool = y > 0.5
    key_arr = np.array(key_all)
    cores, core_info = load_hard_core()

    def core_mask_for(rule: str) -> np.ndarray:
        """Boolean mask over the flat eval axis for one rule's hard core."""
        mask = np.zeros(len(y), dtype=bool)
        for split, idx in cores[rule]:
            mask[offsets[split] + idx] = True
        return mask

    core_masks = {rule: core_mask_for(rule) for rule in ("raw", "balanced")}

    by: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if "error" in r or r.get("variant"):
            continue
        by[r["arm"]][r["architecture"]][r["seed"]] = r

    def logit_matrix(arm: str, arch: str, seeds: list[int]) -> np.ndarray:
        return np.stack([
            np.concatenate([np.array(by[arm][arch][s]["logits"][sp])
                            for sp in A.EVAL_SPLITS])
            for s in seeds
        ])

    def correctness(mat: np.ndarray, rule: str) -> np.ndarray:
        if rule == "raw":
            return (mat >= 0) == truth_bool[None, :]
        pred = np.zeros(mat.shape, dtype=bool)
        for sp in A.EVAL_SPLITS:
            m = split_all == sp
            n_pos = int(truth_bool[m].sum())
            for k in range(mat.shape[0]):
                pred[k, m] = _balanced_pred(mat[k, m], n_pos)
        return pred == truth_bool[None, :]

    out: dict = {
        "n_eval_rows": int(len(y)),
        "hard_core": core_info,
        "per_arm": {},
    }

    # correctness[arm][arch][rule] -> [n_seeds, n_rows] bool, reused by the cross-arm block
    corr: dict = defaultdict(dict)

    for arm in sorted(by):
        out["per_arm"][arm] = {}
        for arch in [a for a in ARCHITECTURES if a in by[arm]]:
            seeds = sorted(by[arm][arch])
            mat = logit_matrix(arm, arch, seeds)
            entry = {
                "seeds": seeds,
                "n_params": by[arm][arch][seeds[0]].get("n_params"),
                "n_train": by[arm][arch][seeds[0]].get("n_train"),
                "best_epochs": [by[arm][arch][s].get("best_epoch") for s in seeds],
                "fit_seconds_mean": float(np.mean(
                    [by[arm][arch][s]["fit_seconds"] for s in seeds]
                )),
            }
            corr[arm][arch] = {}
            for rule in ("raw", "balanced"):
                c = correctness(mat, rule)
                corr[arm][arch][rule] = c
                per_split = {}
                for sp in A.EVAL_SPLITS:
                    m = split_all == sp
                    auc = np.array([
                        by[arm][arch][s]["auroc"][sp]["pipeline"] for s in seeds
                    ])
                    per_split[sp] = {
                        "acc_mean": float(c[:, m].mean(1).mean()),
                        "acc_sd": float(c[:, m].mean(1).std(ddof=1)),
                        "auroc_mean": float(auc.mean()),
                        "auroc_sd": float(auc.std(ddof=1)),
                    }
                auc_all = np.array([
                    np.mean([by[arm][arch][s]["auroc"][sp]["pipeline"]
                             for sp in A.EVAL_SPLITS])
                    for s in seeds
                ])
                # Hard-core recovery: of the rows every heldout probe family got wrong,
                # how many does this architecture get right? Reported as a per-seed mean
                # AND as a majority tag, because one lucky seed is not a recovery.
                core_mask = core_masks[rule]
                n_correct_core = (
                    c[:, core_mask].sum(1) if core_mask.any() else np.zeros(len(seeds))
                )
                maj = c.sum(0) * 2 > len(seeds)
                entry[rule] = {
                    "per_split": per_split,
                    "acc_mean": float(c.mean(1).mean()),
                    "acc_sd": float(c.mean(1).std(ddof=1)),
                    "auroc_mean": float(auc_all.mean()),
                    "auroc_sd": float(auc_all.std(ddof=1)),
                    "core_n": int(core_mask.sum()),
                    "core_recovered_mean": float(np.mean(n_correct_core)),
                    "core_recovered_sd": float(np.std(n_correct_core, ddof=1))
                    if len(seeds) > 1 else 0.0,
                    "core_recovered_majority": int((maj & core_mask).sum()),
                    "n_always_wrong": int((c.sum(0) == 0).sum()),
                }
            out["per_arm"][arm][arch] = entry

    out["cross_arm"] = _cross_arm_block(corr, core_masks)
    return out


def _cross_arm_block(corr: dict, core_masks: dict) -> dict:
    """Rows an architecture gets wrong under *both* attacker arms, and the new core.

    The heldout experiment's core was defined by intersecting across arms; the same
    construction here says whether an architecture's residue is also arm-independent, and
    how much of the old core it clears. An architecture that only helps one arm has found
    something about that arm's red-team data, not about the concept.
    """
    arms = sorted(corr)
    if len(arms) < 2:
        return {}
    out = {}
    for rule in ("raw", "balanced"):
        per_arch = {}
        core = set(np.flatnonzero(core_masks[rule]).tolist())
        for arch in ARCHITECTURES:
            if not all(arch in corr[a] for a in arms):
                continue
            # majority-wrong under each arm, as positions on the flat eval axis, then
            # intersected. Positions, not content hashes: 41 of the 866 rows are repeats
            # of another row (see load_hard_core).
            wrong = []
            for a in arms:
                c = corr[a][arch][rule]
                wrong.append(set(np.flatnonzero(~(c.sum(0) * 2 > len(c))).tolist()))
            both = wrong[0] & wrong[1]
            per_arch[arch] = {
                f"wrong_{arms[0]}": len(wrong[0]),
                f"wrong_{arms[1]}": len(wrong[1]),
                "wrong_both_arms": len(both),
                "old_core_n": len(core),
                "old_core_still_wrong": len(both & core) if core else None,
                "old_core_recovered": len(core - both) if core else None,
            }
        out[rule] = per_arch
    return out


def print_report(res: dict) -> None:
    print(f"\n\n########## architecture sweep — {res['n_eval_rows']} eval rows ##########")
    ci = res["hard_core"]
    print(f"  hard core (balanced): {ci['balanced']['eight_family_core']} rows wrong under "
          f"all 8 heldout probe families; raw: {ci['raw']['eight_family_core']}")

    for arm, archs in res["per_arm"].items():
        for rule in ("raw", "balanced"):
            print(f"\n=== {arm} — decision rule: {rule} ===")
            print(f"{'architecture':24s} {'params':>9s} {'AUROC':>16s} {'accuracy':>16s} "
                  f"{'core recovered':>16s} {'always wrong':>13s}")
            for arch, e in archs.items():
                d = e[rule]
                core = (f"{d['core_recovered_mean']:.1f}+/-{d['core_recovered_sd']:.1f}"
                        f"/{d['core_n']}")
                print(f"{arch:24s} {e['n_params'] or 0:>9d} "
                      f"{d['auroc_mean']:.4f}+/-{d['auroc_sd']:.4f} "
                      f"{d['acc_mean']:.4f}+/-{d['acc_sd']:.4f} "
                      f"{core:>16s} {d['n_always_wrong']:>13d}")

        print(f"\n--- {arm}: per-split AUROC (balanced-rule fits, pipeline scale) ---")
        print(f"{'architecture':24s} " + " ".join(
            f"{sp.replace('eval_', ''):>17s}" for sp in A.EVAL_SPLITS))
        for arch, e in archs.items():
            ps = e["balanced"]["per_split"]
            print(f"{arch:24s} " + " ".join(
                f"{ps[sp]['auroc_mean']:.4f}+/-{ps[sp]['auroc_sd']:.3f}"
                for sp in A.EVAL_SPLITS))

    if res.get("cross_arm"):
        for rule, per_arch in res["cross_arm"].items():
            print(f"\n=== cross-arm ({rule}): rows wrong under BOTH attacker arms ===")
            print(f"{'architecture':24s} {'wrong both arms':>16s} "
                  f"{'old core still wrong':>21s} {'old core recovered':>19s}")
            for arch, d in per_arch.items():
                print(f"{arch:24s} {d['wrong_both_arms']:>16d} "
                      f"{str(d['old_core_still_wrong']):>21s} "
                      f"{str(d['old_core_recovered']):>19s}")


def print_sensitivity(rows: list[dict], truth: dict) -> None:
    """Is an MLP result an artefact of capacity / regularisation?"""
    var_rows = [r for r in rows if r.get("variant") and "error" not in r]
    if not var_rows:
        return
    y, split_all, _keys, _off = _flat_truth(truth)
    print("\n\n########## sensitivity: capacity and regularisation ##########")
    print("  (same architecture, varying hidden_dim / weight_decay — if the spread here")
    print("   covers the architecture differences above, the sweep measured tuning)")
    by = defaultdict(lambda: defaultdict(list))
    for r in var_rows:
        auc = np.mean([r["auroc"][sp]["pipeline"] for sp in A.EVAL_SPLITS])
        by[(r["arm"], r["architecture"])][r["variant"]].append(auc)
    for (arm, arch), variants in by.items():
        print(f"\n  {arm} / {arch}")
        for vname, aucs in sorted(variants.items()):
            a = np.array(aucs)
            print(f"    {vname:28s} AUROC {a.mean():.4f}"
                  f"+/-{a.std(ddof=1) if len(a) > 1 else 0.0:.4f}  ({len(a)} seed(s))")


def summarize() -> None:
    rows = read_progress(PROGRESS)
    if not rows:
        print("no progress rows yet")
        return
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"\n{len(errs)} failed fit(s):")
        for r in errs[:10]:
            print(f"  {r['arm']} {r['architecture']} seed {r['seed']}: {r['error']}")

    probe = A.load_probe(A.ARMS[sorted(A.ARMS)[0]] / f"probe_iter{ITERATION}.pkl")
    truth = load_eval_truth(probe)
    del probe
    gc.collect()

    res = analyse(rows, truth)
    print_report(res)
    print_sensitivity(rows, truth)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.pop("_correctness_keys", None)
    path = OUT_DIR / "arch_sweep.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--architectures", nargs="+", default=ARCHITECTURES)
    ap.add_argument("--eval-dir", type=Path, default=A.EVAL_ACTIVATIONS_DIR)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help=f"vary hidden_dim / weight_decay for {SENSITIVITY_ARCH} on one "
                         "arm, to separate architecture from tuning")
    ap.add_argument("--legacy-best-epoch", action="store_true",
                    help="reproduce tuberlens' non-restoring early stopping, to measure "
                         "what the fix itself is worth")
    args = ap.parse_args()

    if args.legacy_best_epoch:
        from agentic_redteam.probe_architectures import LEGACY_BEST_EPOCH_ENV

        os.environ[LEGACY_BEST_EPOCH_ENV] = "1"
        print("--legacy-best-epoch: early stopping will NOT restore the selected epoch")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        if args.sensitivity:
            variants = [
                (f"h{g['hidden_dim']}_wd{g['weight_decay']:g}", g)
                for g in SENSITIVITY_GRID
            ]
            run_arm(args.arm[0], args.seeds[:3], [SENSITIVITY_ARCH], args.eval_dir,
                    resume=not args.no_resume, variants=variants,
                    legacy_best_epoch=args.legacy_best_epoch)
        else:
            for arm in args.arm:
                run_arm(arm, args.seeds, args.architectures, args.eval_dir,
                        resume=not args.no_resume,
                        legacy_best_epoch=args.legacy_best_epoch)
    summarize()


if __name__ == "__main__":
    main()
