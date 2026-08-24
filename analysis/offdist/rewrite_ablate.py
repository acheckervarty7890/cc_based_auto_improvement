#!/usr/bin/env python
"""Extract activations for the rewritten successes, then refit and score them.

`rewrite_successes.py` asked gpt-5.1 — the model that wrote this run's contrastive partners
— to re-express each of arm 1's attacker successes in its own voice, holding the scenario,
the assistant's behaviour and the label fixed. This is the other half: put those rewrites
through the same extraction, the same fit and the same eval as every other condition, so
the number lands next to `drop_generated` on the same axis.

The comparison the whole thing exists for, all three trained on 50 base rows plus 294
red-team rows and validated on the ceiling study's reserved dev slice:

    drop_generated        base + the attacker's own 294 successes        (already measured)
    rewritten_sources     base + the same 294, rewritten by gpt-5.1      (this script)
    drop_sources          base + the 294 generated partners             (already measured)

If voice is what separates the first from the third, rewriting closes the gap. If the gap
survives, what the partners contribute is the pairing itself, not the writing.

**A caveat that belongs on any number this produces.** The rewrites' labels are asserted by
the rewrite prompt, not verified by the judge, so a rewrite that drifted across the boundary
enters as a mislabelled row. That biases this condition *downward* — against the hypothesis
— so a rewritten score that beats `drop_generated` is still evidence; a score that does not
is only suggestive until the rewrites are re-judged.

Everything runs through the ceiling harness: `ca_common` for the fit, the dev partition and
the eval sources, `retrain._activate_redteam_cached` for the extraction. The rewrites are
content-keyed into the shared per-conversation cache like any other conversation, so the
extraction is resumable and a second run costs nothing.

    analysis/offdist/rewrite_ablate.py [--dry-run] [--conditions ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_common as O  # noqa: E402
from ablate import KEY_FIELDS, N_BASE, evaluate  # noqa: E402

sys.path.insert(0, str(O.REPO / "ceiling_analysis" / "scripts"))
import ca_common as C  # noqa: E402
import ca_data as D  # noqa: E402

ARM = "gptoss120b"
# One blob per condition-set, beside the arm's own pool. Keyed on the arm rather than the
# shared `acts_name`, exactly as `redteam_pool_blob` is: the rewrites belong to this arm.
POOL = C.ACTS_ROOT / "hu_ha_dd_gptoss120b" / "rewritten_pool.pt"


def rewritten_dataset(arm: O.Arm, concept: C.Concept):
    """The rewrites as a LabelledDataset, with the blobs' own message transforms."""
    path = O.RESULTS / f"rewritten_{arm.key}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} — run rewrite_successes.py first")
    ds = C.load_jsonl_dataset(path, concept)
    bad = [lbl for lbl in ds.other_fields["labels"] if lbl is None]
    if bad:
        raise SystemExit(
            f"{path}: {len(bad)} rows whose `labels` matched neither class label — "
            f"tuberlens would load them as None and the fit would score nothing"
        )
    return ds


def extract(ds, concept: C.Concept, *, dry_run: bool) -> int:
    """Fill the shared per-conversation cache for whatever is not in it yet."""
    from agentic_redteam.retrain import (
        _activate_redteam_cached,
        _apply_message_transforms,
        _redteam_activation_cache_path,
    )

    cache_dir = concept.redteam_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = _apply_message_transforms(ds, C.COMBINE, C.CONVERT)
    miss = sum(
        0 if _redteam_activation_cache_path(
            cache_dir, m, C.MODEL_NAME, C.LAYER, C.COMBINE, C.CONVERT
        ).exists() else 1
        for m in ds.inputs
    )
    print(f"  {len(ds)} rewrites, {miss} uncached", flush=True)
    if dry_run or miss == 0:
        return miss

    from agentic_redteam.model_loading import load_extraction_model

    loaded = {"model": None}

    def get_model():
        if loaded["model"] is None:
            print("  loading extraction model ...", flush=True)
            t0 = time.time()
            loaded["model"] = load_extraction_model(C.MODEL_NAME, C.LAYER, verbose=True)
            print(f"  model loaded in {time.time() - t0:.0f}s", flush=True)
        return loaded["model"]

    t0 = time.time()
    _activate_redteam_cached(
        ds, cache_dir, C.MODEL_NAME, C.LAYER, C.COMBINE, C.CONVERT, get_model, True
    )
    print(f"  extraction done in {time.time() - t0:.0f}s", flush=True)
    loaded["model"] = None
    C.free_gpu()
    return miss


def rewritten_source(ds, concept: C.Concept) -> D.BlobSource:
    """The rewrites as one consolidated, mmap-able blob, built on first use.

    Same construction as `ca_common.redteam_source`, including the left-compaction the pool
    builder does — a chunked extraction comes back left-padded and every width trim in this
    analysis assumes right-padding.
    """
    if not POOL.exists():
        C.build_pool_blob(ds, concept.redteam_cache_dir, POOL)
    return D.BlobSource("rewritten", POOL, ds)


def conditions(rt_src, rw_src, flags) -> dict[str, list]:
    """condition -> the `(source, row indices)` parts its training set is made of.

    `base` is rows 0..49 of the arm's own pool, identical to every condition in
    `ablate.py`; `generated` is the partner half of the pairs, addressed in that same pool.
    """
    base = (rt_src, list(range(N_BASE)))
    all_rw = (rw_src, list(range(len(rw_src))))
    generated = (rt_src, [N_BASE + f["i"] for f in flags if f.get("pair_role") == "generated"])
    return {
        # The answer to "what if the attacker's successes were written by gpt-5.1?"
        "rewritten_sources": [base, all_rw],
        # And the pairing restored on top of them: rewrites + the partners already generated
        # for the originals. Free (one more fit, no extra extraction) and it separates
        # "the writing was the problem" from "the pairing was the problem".
        "rewritten_plus_generated": [base, all_rw, generated],
    }


STRUCT_FIELDS = ("chars_total", "chars_assistant", "n_newlines", "has_bullets",
                 "has_numbered", "n_messages")


def write_stats(arm: O.Arm, flags) -> dict:
    """What the rewrite actually changed, beside what it was meant to change.

    The rewrite prompt asks for "similar structure and length", but a model's voice carries
    its own length and formatting, and this analysis has already shown (Q1, and the
    `drop_longest_assistant_30pct` condition in Q3) that both matter here. So the profile of
    the rewrites is recorded next to the two halves it sits between and the eval rows it is
    scored against — otherwise a null result is unreadable: it could mean voice does not
    matter, or that the rewrite did not move the voice.
    """
    rw = {json.loads(l)["source_i"]: json.loads(l)
          for l in (O.RESULTS / f"rewritten_{arm.key}.jsonl").read_text(
              encoding="utf-8").splitlines() if l.strip()}
    groups = {
        "sources": [f["structural"] for f in flags if f.get("pair_role") == "source"],
        "partners": [f["structural"] for f in flags if f.get("pair_role") == "generated"],
        "rewrites": [O.structural_features(json.loads(rw[f["i"]]["inputs"]))
                     for f in flags if f.get("pair_role") == "source" and f["i"] in rw],
        "eval": [O.structural_features(e["messages"]) for e in O.load_eval()],
    }
    sims = sorted(r["similarity"] for r in rw.values())
    turns_kept = sum(len(json.loads(r["inputs"])) == len(json.loads(r["original_inputs"]))
                     for r in rw.values())
    out = {
        "arm": arm.key,
        "n_rewrites": len(rw),
        "rewrite_model": next(iter(rw.values()))["rewrite_model"] if rw else "",
        "similarity_median": sims[len(sims) // 2] if sims else None,
        "similarity_min": sims[0] if sims else None,
        "similarity_max": sims[-1] if sims else None,
        "turn_count_preserved": turns_kept,
        "structural_means": {
            g: {f: float(sum(r[f] for r in rows) / len(rows)) for f in STRUCT_FIELDS}
            for g, rows in groups.items() if rows
        },
    }
    O.write_json(O.RESULTS / f"rewrite_stats_{arm.key}.json", out)
    m = out["structural_means"]
    print("\n  structural profile (chars_total / chars_assistant):")
    for g in ("sources", "rewrites", "partners", "eval"):
        if g in m:
            print(f"    {g:10s} {m[g]['chars_total']:7.1f} / {m[g]['chars_assistant']:7.1f}")
    return out


def run(args) -> int:
    arm = O.ARMS[ARM]
    concept = C.CONCEPTS[arm.concept]
    flags = [json.loads(l) for l in
             (O.RESULTS / f"flags_{arm.key}.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]

    ds = rewritten_dataset(arm, concept)
    extract(ds, concept, dry_run=args.dry_run)
    if args.dry_run:
        print("  --dry-run: nothing fitted")
        return 0
    # A pool that no longer matches the rewrite file (a re-run of rewrite_successes.py with
    # more rows, or a changed prompt) is rebuilt rather than silently reused: BlobSource
    # addresses rows positionally, so a stale pool would mislabel every row after the first
    # difference.
    if POOL.exists():
        import torch

        n_pool = int(torch.load(POOL, map_location="cpu", mmap=True)["activations"].shape[0])
        if n_pool != len(ds):
            print(f"  pool has {n_pool} rows against {len(ds)} rewrites; rebuilding")
            POOL.unlink()

    rt_src = C.redteam_source(concept)
    rw_src = rewritten_source(ds, concept)
    eval_srcs = C.eval_sources(concept)
    dev_src, val_idx, _pool = C.dev_partition(concept)
    val_d = C.ragged_from_parts([(dev_src, val_idx)])

    log_path = O.RESULTS / f"ablation_{arm.key}.jsonl"
    done = C.done_keys(log_path, KEY_FIELDS) if args.resume else set()
    conds = conditions(rt_src, rw_src, flags)
    wanted = args.conditions or list(conds)

    for name in wanted:
        if name not in conds:
            raise SystemExit(f"unknown condition {name!r}; have {sorted(conds)}")
        if (arm.key, name, 0) in done:
            print(f"[{arm.key}] {name}: already in {log_path.name}, skipping")
            continue
        parts = conds[name]
        n_train = sum(len(idx) for _src, idx in parts)
        # Original red-team rows this condition does NOT carry over. These conditions add
        # rows as well as drop them, so the matched-random controls in `report.controls`
        # do not apply — nothing random removes rows and substitutes different text — and
        # `frac_removed` is left NaN to keep that visible.
        n_orig_used = sum(len(idx) for src, idx in parts
                          if src is rt_src and idx and min(idx) >= N_BASE)
        t0 = time.time()
        train = C.ragged_from_parts(parts)
        probe = C.fit(train, val_d, concept, seed=C.FIT_SEED)
        del train
        C.free_gpu()
        m = evaluate(probe, eval_srcs)
        row = {
            "arm": arm.key, "condition": name, "seed": 0,
            "n_removed": len(flags) - n_orig_used,
            "n_train": n_train,
            "n_rewritten": len(rw_src),
            "frac_removed": float("nan"),
            "val_auroc": C.ragged_val_auroc(probe, val_d),
            "seconds": round(time.time() - t0, 1),
            "mean": m["mean"], "per_split": m["per_split"],
        }
        C.append_jsonl(log_path, row)
        print(f"[{arm.key}] {name:26s} train={n_train:>4} "
              f"eval AUROC {m['mean']['auroc']:.4f} ({row['seconds']}s)", flush=True)
        del probe
        C.free_gpu()

    write_stats(arm, flags)

    # The comparison this script exists for, printed rather than left to be looked up.
    rows = {r["condition"]: r for r in
            [json.loads(l) for l in
             log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if r["arm"] == arm.key}
    print("\n  condition                     eval AUROC")
    for name in ("full", "drop_generated", "rewritten_sources", "drop_sources",
                 "rewritten_plus_generated", "base_only"):
        r = rows.get(name)
        if r:
            print(f"  {name:28s} {r['mean']['auroc']:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
