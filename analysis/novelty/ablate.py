#!/usr/bin/env python
"""Phase 3 -- REMOVE red-team rows, refit, and measure. The causal test.

    .venv_claude/bin/python analysis/novelty/ablate.py --experiment instructions --arm gptoss

Phases 1 and 2 can only *propose* which rows sit off the eval manifold. Geometry cannot
say whether a far row helps, hurts, or is invisible to eval -- only removing it and
refitting can. Every condition below is one refit of the probe on
``base union redteam[keep]`` with the run's own validation set (dev, whole), scored on
the eval splits exactly the way ``get_performances`` scores them.

CONDITIONS

``full``              every red-team row. The reference. NOT the published number -- see
                      "row order" below -- so all deltas are measured against this refit.
``base_only``         no red-team rows at all.
``drop_top_qN``       drop the N% most novel rows by ``knn_eval``. The hypothesis.
``drop_bottom_qN``    drop the N% *least* novel rows. The inverse control, and the one
                      that matters most: if dropping the nearest rows helps as much as
                      dropping the farthest, novelty explains nothing and the effect is
                      training-set size.
``drop_random_qN``    drop N% at random, label-stratified, matched n, several seeds. The
                      size control. "Novelty-ordered removal beats random" is the only
                      evidence that would show the geometry is predictive.
``drop_outside``      drop every row past eval's own self-kNN p95 -- the calibrated
                      threshold rather than a hand-picked quantile.
``drop_relative_q20`` drop by ``relative`` instead of raw distance: targets *coherent*
                      novel families rather than isolated one-offs.
``drop_region_K``     drop one k-means region at a time, for attribution.
``full_permM``        identical data, shuffled row order. This is the NOISE FLOOR: the
                      fit is order-sensitive, so the spread across these is the smallest
                      difference any other condition may be read as producing.

WHAT IS MEASURED, per condition: per-split and macro AUROC / accuracy / TPR@1%FPR on
eval; AUROC on dev; and ``cross_auroc`` -- AUROC on the OTHER attacker arm's red-team
rows, held out entirely from this arm's training. That last one is what stops the
central misreading: if removing a region leaves eval flat, that does not show the region
was useless, only that eval has no support there. Cross-attacker robustness is the
metric that can still see what eval is blind to.

ROW ORDER. ``_activate_redteam_cached`` emits cache hits before newly-computed rows, so
a run's training-set row order encodes its box's cache history and is not recoverable
from the snapshot alone (see analysis/ceiling/README.md). Every condition here uses
snapshot file order, so all conditions are mutually comparable, and none reproduces the
published probe bit-identically. Compare conditions to ``full``, never to the CSV.

FIT-SIZE FLOOR. ``batch_size 16`` x ``gradient_accumulation_steps 4`` means a training
set under 64 rows takes ZERO optimizer steps -- the fit runs its epoch budget and
changes nothing. ``base_only`` (50 rows) is below that floor on both concepts. It is
reported with ``below_step_floor: true`` rather than silently.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ceiling"))

import experiments as X  # noqa: E402
import loaders as L  # noqa: E402

QUANTILES = (5, 10, 20, 40)
RANDOM_SEEDS = (0, 1, 2)
PERM_SEEDS = (1, 2, 3)
STEP_FLOOR = 64  # batch_size * gradient_accumulation_steps


# ------------------------------------------------------------------ conditions


def build_conditions(nov: dict, regions: dict, n: int, quantiles=QUANTILES) -> list[dict]:
    """Every condition as {name, keep (bool mask over snapshot rows), meta}."""
    knn = nov["knn_eval"]
    rel = nov["relative"]
    labels = nov["labels"]
    conds: list[dict] = [{"name": "full", "keep": np.ones(n, bool), "meta": {}}]
    conds.append({"name": "base_only", "keep": np.zeros(n, bool), "meta": {}})

    for q in quantiles:
        n_drop = max(1, int(round(n * q / 100)))
        top = np.argsort(-knn)[:n_drop]
        bot = np.argsort(knn)[:n_drop]
        for nm, idx in (("drop_top", top), ("drop_bottom", bot)):
            keep = np.ones(n, bool)
            keep[idx] = False
            conds.append({"name": f"{nm}_q{q}", "keep": keep, "meta": {"n_dropped": int(n_drop)}})
        for s in RANDOM_SEEDS:
            keep = np.ones(n, bool)
            keep[_stratified_draw(labels, n_drop, s)] = False
            conds.append(
                {"name": f"drop_random_q{q}_s{s}", "keep": keep, "meta": {"n_dropped": int(n_drop), "seed": s}}
            )

    keep = nov["outside"] == 0
    conds.append({"name": "drop_outside", "keep": keep, "meta": {"n_dropped": int((~keep).sum())}})

    n_drop = max(1, int(round(n * 0.20)))
    keep = np.ones(n, bool)
    keep[np.argsort(-rel)[:n_drop]] = False
    conds.append({"name": "drop_relative_q20", "keep": keep, "meta": {"n_dropped": int(n_drop)}})

    for r in regions["regions"]:
        if r["id"] < 0:
            continue
        keep = np.ones(n, bool)
        keep[np.array(r["indices"], dtype=int)] = False
        conds.append(
            {
                "name": f"drop_region_{r['id']}",
                "keep": keep,
                "meta": {"n_dropped": r["n"], "knn_eval": r["knn_eval"], "outside_pct": r["outside_pct"]},
            }
        )

    for s in PERM_SEEDS:
        conds.append({"name": f"full_perm{s}", "keep": np.ones(n, bool), "meta": {"perm_seed": s}})
    return conds


def _stratified_draw(labels: np.ndarray, n_drop: int, seed: int) -> np.ndarray:
    """Draw `n_drop` row indices keeping the positive/negative balance of the whole set."""
    rng = np.random.default_rng(seed)
    out = []
    for cls in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == cls)
        share = int(round(n_drop * len(idx) / len(labels)))
        out.extend(rng.choice(idx, size=min(share, len(idx)), replace=False).tolist())
    out = list(dict.fromkeys(out))
    if len(out) < n_drop:  # rounding shortfall: top up from whatever is left
        rest = np.setdiff1d(np.arange(len(labels)), np.array(out, dtype=int))
        out.extend(rng.choice(rest, size=min(n_drop - len(out), len(rest)), replace=False).tolist())
    return np.array(out[:n_drop], dtype=int)


# ------------------------------------------------------------------ fit / score


_PAD_FIELDS = ("activations", "attention_mask", "input_ids")


def build_train(base, rt_full, idx: np.ndarray, chunk: int = 64):
    """base ++ rt_full[idx] WITHOUT consuming either part.

    ``retrain._concatenate_consuming`` pops each part's pad fields as it copies them --
    correct for a one-shot retrain, fatal here, where ``base`` and ``rt_full`` are shared
    across every condition and would be gutted by the first fit. tuberlens'
    ``LabelledDataset.concatenate`` does not consume but pads every part first and then
    cats, so it peaks at ~2x the result *on top of* the parts, which does not fit beside
    a staged dev set.

    So: allocate the output once and copy into it, reading the parts without popping,
    and gather the red-team rows in chunks so the fancy-index temporary stays bounded
    rather than materialising the whole subset a second time.
    """
    import torch as _t

    n = len(base) + len(idx)
    idx_t = _t.as_tensor(np.asarray(idx, dtype=np.int64))
    fields = {}
    for f in _PAD_FIELDS:
        b, r = base.other_fields[f], rt_full.other_fields[f]
        idx_dev = idx_t.to(r.device)
        max_len = max(b.shape[1], r.shape[1] if len(idx) else 0)
        out = _t.empty((n, max_len, *b.shape[2:]), dtype=b.dtype, device=b.device)
        out[: len(base), : b.shape[1]] = b
        if b.shape[1] < max_len:
            out[: len(base), b.shape[1] :] = 0
        start = len(base)
        for s0 in range(0, len(idx), chunk):
            sel = idx_dev[s0 : s0 + chunk]
            part = r.index_select(0, sel).to(b.device)
            out[start : start + len(sel), : r.shape[1]] = part
            if r.shape[1] < max_len:
                out[start : start + len(sel), r.shape[1] :] = 0
            start += len(sel)
            del part
        fields[f] = out

    keys = set(base.other_fields) & set(rt_full.other_fields)
    for key in keys - set(_PAD_FIELDS):
        bv, rv = base.other_fields[key], rt_full.other_fields[key]
        rsel = [rv[int(i)] for i in idx]
        fields[key] = list(bv) + list(rsel)
    return type(base)(
        inputs=list(base.inputs) + [rt_full.inputs[int(i)] for i in idx],
        ids=list(base.ids) + [rt_full.ids[int(i)] for i in idx],
        other_fields=fields,
    )


def fit_condition(exp: X.Experiment, base, rt_full, dev, cond: dict, verbose: bool = True):
    """One condition -> one probe (an ensemble when the concept fits one)."""
    import harness as H
    from agentic_redteam.ensemble import ENSEMBLE_SEEDS

    idx = np.flatnonzero(cond["keep"])
    if "perm_seed" in cond["meta"]:
        idx = np.random.default_rng(cond["meta"]["perm_seed"]).permutation(idx)

    train = build_train(base, rt_full, idx)
    n_train = len(train)

    members, seeds = [], list(ENSEMBLE_SEEDS[: exp.ensemble_size])
    for i, sd in enumerate(seeds):
        t0 = time.monotonic()
        with H.Quiet() as q:
            m = _build_member(exp, train, dev, sd)
        members.append(m)
        if verbose:
            print(
                f"    [{cond['name']}] member {i + 1}/{len(seeds)} seed={sd} "
                f"{q.epochs_run()} ep {time.monotonic() - t0:.0f}s",
                flush=True,
            )
    probe = members[0] if len(members) == 1 else _ensemble(members, seeds)
    del train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probe, {"n_train": n_train, "n_redteam": int(len(idx)), "below_step_floor": n_train < STEP_FLOOR}


def _build_member(exp: X.Experiment, train, dev, seed: int):
    from agentic_redteam.evaluation import seed_everything
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.probes.probe_factory import ProbeFactory

    seed_everything(seed)
    return ProbeFactory.build(
        probe_spec=ProbeSpec(name=ProbeType.linear_then_softmax),
        train_dataset=train,
        model_name=X.MODEL,
        layer=X.LAYER,
        validation_dataset=dev,
        use_store=False,
        pos_class_label=exp.pos,
        neg_class_label=exp.neg,
        probe_description=None,
    )


def _ensemble(members, seeds):
    from agentic_redteam.ensemble import EnsembleProbe

    return EnsembleProbe.from_members(members, list(seeds))


def score_all(exp: X.Experiment, probes: dict, dev, cross_ds) -> dict:
    """Score every fitted probe. Eval splits are loaded ONE AT A TIME and scored by all
    probes before being freed -- on high-stakes a split is up to 30 GiB, so paying that
    read once per split instead of once per condition is the difference between minutes
    and hours."""
    import harness as H

    out = {name: {"per_split": {}} for name in probes}
    for split in exp.splits():
        t0 = time.monotonic()
        ds = L.load_eval_split(exp, split)
        y = L.labels_of(ds)
        for name, probe in probes.items():
            p = np.asarray(probe.predict_proba(ds))
            out[name]["per_split"][split] = H.split_metrics(y, p)
        del ds
        gc.collect()
        print(f"  scored eval/{split} for {len(probes)} probes ({time.monotonic() - t0:.0f}s)", flush=True)

    y_dev = L.labels_of(dev)
    for name, probe in probes.items():
        out[name]["macro"] = H.macro(out[name]["per_split"])
        out[name]["dev_auroc"] = H.split_metrics(y_dev, np.asarray(probe.predict_proba(dev)))["auroc"]
    if cross_ds is not None:
        y_x = L.labels_of(cross_ds)
        for name, probe in probes.items():
            out[name]["cross_auroc"] = H.split_metrics(y_x, np.asarray(probe.predict_proba(cross_ds)))["auroc"]
    return out


# ------------------------------------------------------------------ driver


def run_arm(exp: X.Experiment, arm: X.Arm, quantiles, only: list[str] | None, verbose: bool = True) -> list[dict]:
    import harness as H

    H.silence_tqdm()

    nov = dict(np.load(X.RESULTS / f"novelty_{exp.key}_{arm.name}.npz", allow_pickle=True))
    regions = json.loads((X.RESULTS / f"regions_{exp.key}_{arm.name}_kmeans.json").read_text())
    n = len(nov["knn_eval"])

    conds = build_conditions(nov, regions, n, quantiles)
    if only:
        conds = [c for c in conds if c["name"] in set(only)]
    print(f"=== {exp.key}/{arm.name}: {len(conds)} conditions over {n} red-team rows ===", flush=True)

    t0 = time.monotonic()
    base = L.load_base(exp)
    dev = L.load_dev(exp)
    n_dev_all = len(dev)
    if exp.dev_fit_rows and n_dev_all > exp.dev_fit_rows:
        dev = dev[[int(i) for i in dev_subsample(exp, exp.dev_fit_rows)]]
    rt_full = L.load_redteam(exp, arm)
    print(
        f"loaded base={len(base)} dev={len(dev)}"
        + (f" (stratified subsample of {n_dev_all} for the fit)" if len(dev) != n_dev_all else "")
        + f" redteam={len(rt_full)} in {time.monotonic() - t0:.0f}s",
        flush=True,
    )

    # What goes on the card, and why. Validation and the training set are BOTH read
    # every epoch, so a host-resident one costs a full copy per epoch; the goal is to
    # have both resident for the whole fit. On the heavy concept that is only possible
    # because `dev` was subsampled above. `rt_full` deliberately stays on the host
    # there: build_train allocates its output on `base`'s device and copies the selected
    # rows in, so the card holds the condition's training set without ever also holding
    # the whole red-team pool.
    H.stage(dev, base, verbose=verbose)
    if not exp.heavy:
        H.stage(rt_full, verbose=verbose)
    print(f"staged; {H.gpu_free_gib():.1f} GiB allocatable", flush=True)

    probes, meta = {}, {}
    for c in conds:
        t = time.monotonic()
        probe, m = fit_condition(exp, base, rt_full, dev, c, verbose)
        probes[c["name"]] = probe
        m.update(c["meta"])
        m["fit_seconds"] = round(time.monotonic() - t, 1)
        meta[c["name"]] = m
        if m["below_step_floor"]:
            print(f"    !! {c['name']}: {m['n_train']} train rows < {STEP_FLOOR}; fit takes no optimizer step",
                  flush=True)

    # Free the fit-time residents before scoring: on high-stakes the biggest eval split
    # is 30 GiB of host RAM and the dev set is still holding 19.6 GiB of the card.
    cross_ds = _load_cross(exp, arm)
    del rt_full
    H.free()

    scored = score_all(exp, probes, dev, cross_ds)

    rows = []
    for name, s in scored.items():
        rows.append(
            {
                "experiment": exp.key,
                "arm": arm.name,
                "condition": name,
                **meta[name],
                "macro_auroc": s["macro"]["auroc"],
                "macro_accuracy": s["macro"]["accuracy"],
                "macro_tpr": s["macro"]["tpr_at_fpr_le"],
                "dev_auroc": s["dev_auroc"],
                "cross_auroc": s.get("cross_auroc"),
                "per_split_auroc": {k: v["auroc"] for k, v in s["per_split"].items()},
            }
        )
    return rows


def dev_subsample(exp: X.Experiment, want: int) -> np.ndarray:
    """A split- and class-stratified subsample of the dev rows, fixed across conditions.

    Stratified on (dev split, label) so the reduced validation set keeps the shape of the
    full one. ``_load_dev_dataset`` concatenates the splits in sorted-glob order, so row
    ranges give each row its split identity. Seeded on the run's own seed, and returned
    sorted, so every condition -- and every re-run -- validates against the same rows.
    """
    split_of, lab = [], []
    for f in sorted(exp.dev_dir.glob("*.jsonl")):
        ds = L.load_jsonl_dataset(f, exp)
        split_of.extend([f.stem] * len(ds))
        lab.extend(L.labels_of(ds).tolist())
    strata = [f"{s}|{y}" for s, y in zip(split_of, lab)]
    rng = np.random.default_rng(X.SEED)
    buckets = []
    for k in sorted(set(strata)):
        idx = np.array([i for i, s in enumerate(strata) if s == k])
        rng.shuffle(idx)
        buckets.append(list(idx))
    out: list[int] = []
    while len(out) < want and any(buckets):
        for b in buckets:
            if b and len(out) < want:
                out.append(b.pop(0))
    return np.array(sorted(out), dtype=int)


def _load_cross(exp: X.Experiment, arm: X.Arm):
    """The other arm's red-team rows, as a held-out attack-generalisation set."""
    others = [a for k, a in exp.arms.items() if k != arm.name]
    if not others:
        return None
    try:
        return L.load_redteam(exp, others[0])
    except FileNotFoundError as e:
        print(f"  (no cross-attacker set: {e})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=sorted(X.EXPERIMENTS))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--quantiles", type=int, nargs="*", default=list(QUANTILES))
    ap.add_argument("--only", nargs="*", default=None, help="run just these condition names")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    exp = X.get(args.experiment)
    arms = [exp.arms[args.arm]] if args.arm else list(exp.arms.values())
    dest = Path(args.out) if args.out else X.RESULTS / f"ablation_{exp.key}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        rows = run_arm(exp, arm, args.quantiles, args.only)
        with dest.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        base = next(r for r in rows if r["condition"] == "full")
        print(f"\n--- {exp.key}/{arm.name}: full = {base['macro_auroc']:.4f} macro AUROC ---")
        for r in sorted(rows, key=lambda r: -r["macro_auroc"]):
            d = r["macro_auroc"] - base["macro_auroc"]
            cross = f"{r['cross_auroc']:.4f}" if r.get("cross_auroc") is not None else "  n/a "
            print(
                f"{r['condition']:>24} n={r['n_train']:>5} macro {r['macro_auroc']:.4f} "
                f"({d:+.4f})  dev {r['dev_auroc']:.4f}  cross {cross}"
            )
        print(f"\nAppended {len(rows)} rows -> {dest}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
