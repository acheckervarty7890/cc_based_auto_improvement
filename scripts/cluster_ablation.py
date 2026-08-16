#!/usr/bin/env python
"""Cluster-subsampling ablation on the 8July run's red-team successes.

Question this answers: **how does keeping only `n` samples per near-duplicate
cluster of red-team successes change the retrained probe's eval scores?** i.e.
do the clone families help the probe, hurt it, or wash out?

For each `n` in `--n-per-cluster` it:

  1. clusters the red-team successes (difflib ratio on the first user message,
     greedy single-linkage >= --tau), independently for each error-type JSONL;
  2. keeps the first `min(n, cluster_size)` members of every cluster, using a
     *seeded, nested* ordering (so the n=1 pick is a subset of n=2, etc.);
  3. writes those success rows verbatim to a temp JSONL;
  4. calls `retrain_probe` on base-data + that subsample, running the SAME
     production preprocessing as the run (`filter_dataset` +
     `generate_contrastive_dataset`), inheriting architecture/metadata from the
     base probe (`probe_iter0`);
  5. scores the retrained probe on the local eval splits via `evaluate_probe`;
  6. records per-split + mean auroc/accuracy.

Everything runs OFF THE 8July CACHES — no gemma load, no API calls — *provided*
the success pool is restricted to the run's post-filter survivors (the samples
that actually have a cached activation blob and a cached contrastive pair). The
~170 FP "confounders" that production's `filter_dataset` dropped before
activation are therefore excluded by default (they never trained a probe and
aren't cached); pass --allow-uncached to include them (needs OPENROUTER_API_KEY
and will load gemma to compute the missing activations — likely OOM on this box).

Usage (from repo root):

  .venv_claude/bin/python scripts/cluster_ablation.py configs/gemma27_hs_prompt.md \
      --base-training-data data/hs_ls_200.jsonl \
      --n-per-cluster 1 2 3 5 10 all \
      --out-csv results_hs_prompt/cluster_ablation.csv
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# --------------------------------------------------------------------------- #
# Success-pool loading + clustering (stdlib only)
# --------------------------------------------------------------------------- #


def first_user(sample) -> str:
    """First user message content of a sample (list of msgs or {messages:[...]})."""
    msgs = sample if isinstance(sample, list) else (sample or {}).get("messages", [])
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return (m.get("content") or "").strip()
    return (msgs[0].get("content", "") if msgs and isinstance(msgs[0], dict) else "").strip()


def canon(sample) -> str:
    """Canonical text of the whole conversation, for exact dedup."""
    msgs = sample if isinstance(sample, list) else (sample or {}).get("messages", [])
    return json.dumps(
        [{"role": m.get("role"), "content": m.get("content")} for m in msgs],
        sort_keys=True,
        ensure_ascii=False,
    )


def load_successes(path: Path):
    """Return distinct successful rows: list of {raw, sample, text}. Exact-deduped."""
    out = []
    seen = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("success"):
                continue
            sample = d.get("sample")
            key = canon(sample)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "raw": line,
                "sample": sample,
                "text": first_user(sample),
                "judge_label": d.get("judge_label", ""),
            })
    return out


def cluster_orderings(rows, tau: float, prefix: int, rng):
    """Greedy single-linkage clustering by difflib ratio >= tau on the first user
    message (truncated to `prefix`). Returns a list of clusters, each a list of
    row-indices *shuffled once* (seeded) so slicing [:n] gives nested subsets."""
    texts = [r["text"][:prefix] for r in rows]
    used = [False] * len(rows)
    clusters = []
    for i in range(len(rows)):
        if used[i]:
            continue
        members = [i]
        used[i] = True
        for j in range(i + 1, len(rows)):
            if used[j]:
                continue
            if difflib.SequenceMatcher(None, texts[i], texts[j]).ratio() >= tau:
                members.append(j)
                used[j] = True
        rng.shuffle(members)  # seeded → stable nested ordering across n
        clusters.append(members)
    return clusters


def select(rows, clusters, n) -> list[str]:
    """Raw JSONL lines kept when we take min(n, size) of each cluster."""
    keep = math.inf if n == math.inf else int(n)
    picked = []
    for members in clusters:
        take = members if keep == math.inf else members[:keep]
        picked.extend(rows[idx]["raw"] for idx in take)
    return picked


# --------------------------------------------------------------------------- #
# Cache-coverage restriction (keep the ablation fully offline)
# --------------------------------------------------------------------------- #


def restrict_to_cached(rows, cache_dir, model, layer, combine, convert,
                       pos_label, neg_label, truth_label, contrastive_cache,
                       preprocessing=None):
    """Keep only rows that are fully cache-backed for the offline retrain path.

    A row survives iff BOTH exist on disk already:
      * its (transformed) activation blob  — so no gemma forward is needed, and
      * its contrastive pair in `contrastive_cache` — so `generate_contrastive_dataset`
        makes no LLM call (a lone missing pair forces a client build → API key).
    These are exactly the run's post-filter survivors whose contrastive generation
    also succeeded. `truth_label` is the error-type's implied true class, used as the
    label (hence contrastive target) when the judge label isn't one of the two
    classes — matching `retrain._success_human_label`.
    """
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLM

    from agentic_redteam.preprocessing import (
        _cache_key,
        _extract_messages,
        _guidance_fingerprint,
    )
    from agentic_redteam.retrain import _redteam_activation_cache_path

    # The contrastive cache key folds in the config's concept guidance (empty
    # fingerprint when the config sets none), so mirror it here or every lookup
    # misses for a guidance-carrying config.
    description = getattr(preprocessing, "concept_description", "") or ""
    guidance = getattr(preprocessing, "label_guidance", None)

    def transform(msgs):
        d = msgs
        if convert:
            d = LabelledDataset._convert_tool_to_assistant(d)
        if combine:
            d = LabelledDataset._combine_consecutive_messages(d)
        return d

    kept = []
    dropped_act = dropped_pair = 0
    for r in rows:
        sample = r["sample"]
        raw_msgs = sample if isinstance(sample, list) else (sample or {}).get("messages", [])
        tl = [TLM(role=m.get("role", "user"), content=m.get("content", "")) for m in raw_msgs]
        if not _redteam_activation_cache_path(
            cache_dir, transform(tl), model, layer, combine, convert
        ).exists():
            dropped_act += 1
            continue
        label = r["judge_label"] if r["judge_label"] in (pos_label, neg_label) else truth_label
        target = neg_label if label == pos_label else pos_label
        msgs = _extract_messages({"inputs": raw_msgs}, "inputs")
        fingerprint = _guidance_fingerprint(description, target, guidance)
        # Pinned to "v1" deliberately: this reads a contrastive_cache.jsonl written by a
        # COMPLETED run, and every such cache predates GENERATION_PROMPT_VERSION. Taking
        # the live default instead would put the current version in the key, miss every
        # row, and silently report the whole run as having no pairs.
        if _cache_key(msgs, target, fingerprint, "v1") not in contrastive_cache:
            dropped_pair += 1
            continue
        kept.append(r)
    return kept, dropped_act, dropped_pair


# --------------------------------------------------------------------------- #
# Cached-activation eval (no model load)
# --------------------------------------------------------------------------- #


def _eval_cache_stem(max_samples, seed) -> str:
    """Filename stem evaluate_probe/get_performances write per split (see evaluation.py)."""
    return f"acts_n{max_samples}_seed{seed}.pt" if max_samples is not None else "acts_full.pt"


def offline_eval_feasible(eval_dataset_dir, activations_cache_dir, max_samples, seed):
    """Return (ok, splits, missing) — whether every split has a cached activation blob."""
    eval_dataset_dir = Path(eval_dataset_dir)
    activations_cache_dir = Path(activations_cache_dir)
    splits = sorted(p.stem for p in eval_dataset_dir.glob("*.jsonl"))
    stem = _eval_cache_stem(max_samples, seed)
    missing = [n for n in splits if not (activations_cache_dir / f"{n}-{stem}").exists()]
    return (bool(splits) and not missing), splits, missing


def evaluate_probe_offline(
    probe_path, eval_dataset_dir, activations_cache_dir, *,
    max_samples, seed, combine, convert, splits=None,
):
    """Score a probe on the eval splits using ONLY the cached activation blobs.

    Mirrors agentic_redteam.evaluation.evaluate_probe, but pre-loads each split's
    cached ``<name>-<stem>.pt`` (via ``LLMModel.load_activations`` — no model
    instance) and assigns it to the dataset *before* calling ``get_performances``.
    That makes ``"activations" in dataset.other_fields`` true, so get_performances
    never enters its ``LLMModel.load`` branch — zero gemma loads. Row order matches
    the cache because the dataset is rebuilt with the same deterministic
    ``load_from`` (+ seeded ``subsample_balanced_subset``) that wrote it.
    """
    import pickle

    from tuberlens.evaluation import get_performances
    from tuberlens.interfaces.dataset import LabelledDataset, subsample_balanced_subset
    from tuberlens.model import LLMModel

    from agentic_redteam.evaluation import seed_everything

    eval_dataset_dir = Path(eval_dataset_dir)
    activations_cache_dir = Path(activations_cache_dir)
    if splits is None:
        splits = sorted(p.stem for p in eval_dataset_dir.glob("*.jsonl"))
    with open(probe_path, "rb") as f:
        probe = pickle.load(f)

    seed_everything(seed)  # same ordering guarantee as evaluate_probe (subsample determinism)
    stem = _eval_cache_stem(max_samples, seed)
    eval_datasets = {}
    for name in splits:
        ds = LabelledDataset.load_from(
            eval_dataset_dir / f"{name}.jsonl",
            pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label,
            combine_consecutive_messages=combine,
            convert_tool_to_assistant=convert,
        )
        if max_samples is not None:
            ds = subsample_balanced_subset(ds, n_per_class=max_samples // 2)
        acts = LLMModel.load_activations(activations_cache_dir / f"{name}-{stem}")
        ds = ds.assign(
            activations=acts.activations,
            attention_mask=acts.attention_mask,
            input_ids=acts.input_ids,
        )
        eval_datasets[name] = ds
    # activations already present on every dataset → get_performances loads no model
    return get_performances(probe, eval_datasets)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_n(tok: str):
    if tok.lower() in ("all", "inf", "full", "max"):
        return math.inf
    return int(tok)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path, help="run config (e.g. configs/gemma27_hs_prompt.md)")
    ap.add_argument("--base-training-data", type=Path, default=REPO_ROOT / "data/hs_ls_200.jsonl")
    ap.add_argument(
        "--base-probe",
        type=Path,
        default=REPO_ROOT / "probes/8July_hs_concept/probe_iter0.pkl",
        help="probe whose architecture + metadata the retrains inherit",
    )
    ap.add_argument(
        "--probe-out-dir",
        type=Path,
        default=REPO_ROOT / "probes/8July_hs_concept",
        help="dir holding contrastive_cache.jsonl; ablation probes go in <dir>/cluster_ablation",
    )
    ap.add_argument("--eval-dataset-dir", type=Path, default=REPO_ROOT / "eval_datasets")
    ap.add_argument(
        "--n-per-cluster", nargs="+", default=["1", "2", "3", "5", "10", "all"],
        help="cluster caps to sweep; 'all' = keep every member (default: 1 2 3 5 10 all)",
    )
    ap.add_argument("--tau", type=float, default=0.75, help="near-dup similarity threshold")
    ap.add_argument("--prefix", type=int, default=600, help="chars of first-user msg compared")
    ap.add_argument("--seed", type=int, default=42, help="split seed + cluster-shuffle seed")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument(
        "--allow-uncached", action="store_true",
        help="do NOT restrict to cache-backed successes (may load gemma + call the API)",
    )
    ap.add_argument(
        "--offline-eval", action=argparse.BooleanOptionalAction, default=True,
        help="score probes off cached eval activations with NO gemma load (default on); "
        "auto-falls back to online eval if any split's activation cache is missing",
    )
    args = ap.parse_args(argv)

    import random

    from agentic_redteam.config import load_config
    from agentic_redteam.evaluation import evaluate_probe, seed_everything
    from agentic_redteam.retrain import retrain_probe

    config = load_config(args.config)
    combine = config.eval.combine_consecutive_messages
    convert = config.eval.convert_tool_to_assistant
    eval_max = config.eval.eval_max_samples if (config.eval.eval_max_samples or 0) > 0 else None
    base_cache = config.output.base_activation_cache_dir
    eval_cache = config.output.activations_cache_dir
    model = config.probe.model
    layer = config.probe.layer
    contrastive_cache = args.probe_out_dir / "contrastive_cache.jsonl"
    ablation_dir = args.probe_out_dir / "cluster_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ablation_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_csv or (base_cache.parent / "cluster_ablation.csv"
                               if base_cache else ablation_dir / "cluster_ablation.csv")

    if base_cache is None:
        sys.exit("config has no output.base_activation_cache_dir — needed for cached training")
    for pth, what in [(base_cache, "base activation cache"), (eval_cache, "eval activation cache"),
                      (args.base_probe, "base probe"), (contrastive_cache, "contrastive cache")]:
        if pth is None or not Path(pth).exists():
            sys.exit(f"missing {what}: {pth}")

    ns = [parse_n(t) for t in args.n_per_cluster]
    print(f"Config: {args.config}")
    print(f"  model={model} layer={layer}  combine={combine} convert={convert}  eval_max={eval_max}")
    print(f"  base_cache={base_cache}\n  eval_cache={eval_cache}")
    print(f"  n-per-cluster sweep: {[('all' if n==math.inf else int(n)) for n in ns]}  tau={args.tau}")

    # ---- Build the (cache-restricted) success pool per error type, cluster it. ----
    error_types = list(config.probe.error_types)
    pos_label, neg_label = config.probe.pos_class_label, config.probe.neg_class_label

    from agentic_redteam.preprocessing import _load_cache
    contrastive_map = _load_cache(contrastive_cache)
    print(f"  contrastive cache entries: {len(contrastive_map)}")

    pools = {}  # error_type value -> (rows, clusters)
    for et in error_types:
        jp = config.jsonl_path_for(et)
        et_name = getattr(et, "value", str(et))
        if not jp.exists():
            print(f"  [{et_name}] JSONL missing ({jp}); skipping")
            continue
        rows = load_successes(jp)
        n_raw = len(rows)
        # truth class implied by the error type: false_positive → negative, false_negative → positive
        truth_label = neg_label if et_name == "false_positive" else pos_label
        if not args.allow_uncached:
            rows, drop_act, drop_pair = restrict_to_cached(
                rows, base_cache, model, layer, combine, convert,
                pos_label, neg_label, truth_label, contrastive_map,
                config.preprocessing,
            )
            bits = []
            if drop_act:
                bits.append(f"{drop_act} no cached activation")
            if drop_pair:
                bits.append(f"{drop_pair} no cached contrastive pair")
            note = f"  ({', '.join(bits)} dropped)" if bits else ""
        else:
            note = "  (--allow-uncached: pool NOT restricted)"
        rng = random.Random(f"{args.seed}:{et_name}")
        clusters = cluster_orderings(rows, args.tau, args.prefix, rng)
        pools[et_name] = (rows, clusters)
        sizes = sorted((len(c) for c in clusters), reverse=True)
        print(f"  [{et_name}] {n_raw} distinct successes → {len(rows)} usable{note}; "
              f"{len(clusters)} clusters (largest {sizes[0] if sizes else 0})")

    if not pools:
        sys.exit("no usable success pools")

    # Decide eval mode once: offline (cached activations, no gemma load) if requested
    # and every split has a cached activation blob; else fall back to online eval.
    use_offline_eval = args.offline_eval
    if args.offline_eval:
        ok, _splits, missing = offline_eval_feasible(
            args.eval_dataset_dir, eval_cache, eval_max, args.seed
        )
        use_offline_eval = ok
        if ok:
            print(f"  eval: OFFLINE (cached activations, no gemma load) — {len(_splits)} splits")
        else:
            print(f"  eval: online (falling back — missing cached activations: {missing})")

    # ---- Sweep n: subsample → retrain (cached) → eval. ----
    import gc

    import pandas as pd
    import torch

    all_rows = []
    for n in ns:
        n_label = "all" if n == math.inf else str(int(n))
        selected_lines = []
        per_et = {}
        for et_name, (rows, clusters) in pools.items():
            lines = select(rows, clusters, n)
            per_et[et_name] = len(lines)
            selected_lines.extend(lines)
        n_selected = len(selected_lines)

        tmp_jsonl = tmp_dir / f"successes_n{n_label}.jsonl"
        tmp_jsonl.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")

        probe_out = ablation_dir / f"probe_n{n_label}.pkl"
        print(f"\n{'='*70}\nn={n_label}: selected {n_selected} successes "
              f"({', '.join(f'{k}={v}' for k,v in per_et.items())}) → retrain")

        seed_everything(args.seed)
        result = retrain_probe(
            jsonl_path=tmp_jsonl,
            base_probe_path=args.base_probe,
            base_training_data_path=args.base_training_data,
            new_probe_path=probe_out,
            preprocessing=config.preprocessing,
            contrastive_cache_path=contrastive_cache,
            base_activation_cache_dir=base_cache,
            combine_consecutive_messages=combine,
            convert_tool_to_assistant=convert,
            seed=args.seed,
            test_size=args.test_size,
            verbose=True,
        )

        print(f"n={n_label}: eval {probe_out}"
              + ("  [offline]" if use_offline_eval else "  [online]"))
        if use_offline_eval:
            df = evaluate_probe_offline(
                probe_out,
                args.eval_dataset_dir,
                eval_cache,
                max_samples=eval_max,
                seed=args.seed,
                combine=combine,
                convert=convert,
            )
        else:
            df = evaluate_probe(
                probe_out,
                args.eval_dataset_dir,
                eval_cache,
                max_samples=eval_max,
                seed=args.seed,
                combine_consecutive_messages=combine,
                convert_tool_to_assistant=convert,
            )
        df = df.copy()
        df.insert(0, "n_per_cluster", n_label)
        df.insert(1, "n_success_selected", n_selected)
        df.insert(2, "n_redteam_trained", result.n_redteam_samples)
        df.insert(3, "n_train_total", result.n_training_samples_total)
        # get_performances already appends its own dataset="mean" aggregate row.
        print(df.to_string(index=False))
        all_rows.append(df)

        # Release the eval model's GPU/CPU memory before the next n loads it again —
        # every tuberlens load re-infers its device_map from *free* memory, so a
        # resident copy would force the next into deeper offload (or OOM).
        gc.collect()
        torch.cuda.empty_cache()

    final = pd.concat(all_rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_csv, index=False)
    print(f"\nSaved {len(final)} rows to {out_csv}")

    # Compact summary: mean auroc/accuracy vs n
    summary = final[final["dataset"] == "mean"][
        ["n_per_cluster", "n_success_selected", "n_redteam_trained", "n_train_total",
         "auroc", "accuracy"]
    ]
    print("\n### mean-across-splits vs n-per-cluster ###")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
