#!/usr/bin/env python
"""Diff two probes of one arm per-sample on chosen eval splits, and explain the diff.

Written for the set-A arm's iteration-2 -> iteration-3 regression: ``probe_iter3``
trains on exactly ``probe_iter2``'s data plus the 14 iteration-3 pairs, so any
per-sample change on an eval split is attributable to those 14 pairs.

Two outputs:

``<probe-dir>/<a>_vs_<b>_<splits>.jsonl``
    One row per eval sample: the two probes' scores, whether each was correct at
    0.5, and a ``status`` of stable_correct / stable_wrong / regression /
    improvement. Conversation text is *not* copied — ``split`` + ``idx`` index back
    into ``<eval-dataset-dir>/<split>.jsonl`` (order-preserving, whole split), so
    the viewer can join without duplicating the corpus.

``<probe-dir>/<b>_pair_token_stats.json``
    Per-token diagnostics for the pairs that ``b`` added over ``a``, for the subset
    whose two members share a byte-identical message prefix. Since the pooling is
    ``LinearThenSoftmax`` over per-token linear scores, this is what shows *how* the
    new pairs changed the probe: for each member it reports the per-token score peak
    on the shared prefix vs. on the differing tail, the share of softmax pooling
    weight sitting on the shared prefix, and the pooled score.

    Token alignment starts at the *second* ``<|start_header_id|>`` — i.e. just past
    the chat template's system header — because Llama's template injects ``Today
    Date:``, so two conversations tokenized on different days differ in that one
    token and a naive common-prefix scan stops there. (That also means the
    per-conversation activation cache is not reproducible across days; the date is
    not part of its key. The header carries negligible pooling weight, so it does
    not affect the scores materially, but it is why the alignment skips it.)

Activations come from the same on-disk caches the evals used, so no model is loaded
when both caches are warm.

Usage:
    .venv_claude/bin/python scripts/analyze_iter3_regression.py \
        --probe-dir probes/hs_llama1b_deepseekv4pro_guidance_setA \
        --probe-a probe_iter2.pkl --probe-b probe_iter3.pkl \
        --redteam-a redteam_postprocessed_iter2.jsonl \
        --redteam-b redteam_postprocessed_iter3.jsonl \
        --splits mt mts
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

HEADER_TOKEN = 128006  # Llama-3 <|start_header_id|>


def _load_probe(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _linear(probe):
    """(weight, bias) of the probe's per-token linear layer, on CPU."""
    model = probe._classifier.model
    return (
        model.linear.weight.detach().flatten().float().cpu(),
        model.linear.bias.detach().float().cpu(),
    )


def _transform(messages):
    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

    dialogue = [TLMessage(role=m["role"], content=m["content"]) for m in messages]
    dialogue = LabelledDataset._convert_tool_to_assistant(dialogue)
    return LabelledDataset._combine_consecutive_messages(dialogue)


def per_sample(args, probe_a, probe_b) -> list[dict]:
    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score
    from tuberlens.interfaces.dataset import LabelledDataset
    from tuberlens.model import LLMModel

    from agentic_redteam.evaluation import seed_everything

    pos, neg = probe_a.pos_class_label, probe_a.neg_class_label
    seed_everything(args.seed)

    rows: list[dict] = []
    for name in args.splits:
        raw = [json.loads(l) for l in (args.eval_dataset_dir / f"{name}.jsonl").open()]
        dataset = LabelledDataset.load_from(
            args.eval_dataset_dir / f"{name}.jsonl",
            pos_class_label=pos,
            neg_class_label=neg,
            combine_consecutive_messages=args.combine_consecutive_messages,
            convert_tool_to_assistant=args.convert_tool_to_assistant,
        )
        if len(dataset) != len(raw):
            raise SystemExit(f"{name}: {len(dataset)} loaded vs {len(raw)} raw rows")
        acts = LLMModel.load_activations(
            args.activations_cache_dir / f"{name}-acts_full.pt"
        )
        dataset = dataset.assign(
            activations=acts.activations,
            attention_mask=acts.attention_mask,
            input_ids=acts.input_ids,
        )
        y = np.array([lab.to_int() for lab in dataset.labels])
        sa = np.asarray(probe_a.predict_proba(dataset), dtype=float)
        sb = np.asarray(probe_b.predict_proba(dataset), dtype=float)
        ca, cb = (sa > 0.5) == (y == 1), (sb > 0.5) == (y == 1)
        print(
            f"[{name}] n={len(y)} pos={int(y.sum())}  "
            f"auroc {roc_auc_score(y, sa):.4f} -> {roc_auc_score(y, sb):.4f}  "
            f"acc {accuracy_score(y, sa > 0.5):.4f} -> {accuracy_score(y, sb > 0.5):.4f}  "
            f"regressions={int((ca & ~cb).sum())} improvements={int((~ca & cb).sum())}"
        )
        print(
            f"[{name}] score shift mean {(sb - sa).mean():+.4f} "
            f"(pos {(sb - sa)[y == 1].mean():+.4f} / neg {(sb - sa)[y == 0].mean():+.4f}); "
            f"class means {sa[y == 1].mean():.3f}/{sa[y == 0].mean():.3f} -> "
            f"{sb[y == 1].mean():.3f}/{sb[y == 0].mean():.3f}"
        )
        for i in range(len(y)):
            rows.append(
                {
                    "split": name,
                    "idx": i,
                    "label": pos if y[i] == 1 else neg,
                    "y": int(y[i]),
                    "score_a": float(sa[i]),
                    "score_b": float(sb[i]),
                    "delta": float(sb[i] - sa[i]),
                    "correct_a": bool(ca[i]),
                    "correct_b": bool(cb[i]),
                    "status": (
                        "regression"
                        if (ca[i] and not cb[i])
                        else "improvement"
                        if (not ca[i] and cb[i])
                        else "stable_correct"
                        if cb[i]
                        else "stable_wrong"
                    ),
                    "meta": {k: v for k, v in raw[i].items() if k != "inputs"},
                }
            )
    return rows


def added_pairs(args) -> dict[str, dict]:
    """Pairs present in redteam-b but not redteam-a, keyed by conversation-key prefix."""
    ids_a = {
        json.loads(l)["id"] for l in (args.probe_dir / args.redteam_a).open()
    }
    pairs: dict[str, dict] = {}
    for line in (args.probe_dir / args.redteam_b).open():
        row = json.loads(line)
        if row["id"] in ids_a:
            continue
        kind, key = row["id"].split("-", 1)
        pairs.setdefault(key, {})[kind] = row
    incomplete = [k for k, v in pairs.items() if set(v) != {"orig", "contra"}]
    if incomplete:
        raise SystemExit(f"incomplete added pairs: {incomplete}")
    print(f"pairs added by {args.redteam_b} over {args.redteam_a}: {len(pairs)}")
    return pairs


def pair_token_stats(args, pairs, probe_a, probe_b) -> list[dict]:
    import torch
    from tuberlens.model import LLMModel

    from agentic_redteam.retrain import _redteam_activation_cache_path

    weights = {"a": _linear(probe_a), "b": _linear(probe_b)}
    temperature = float(probe_b._classifier.model.kwargs["temperature"])
    model_name, layer = probe_b.model_name, int(probe_b.layer)

    def activation(messages):
        path = _redteam_activation_cache_path(
            args.redteam_activation_cache_dir,
            _transform(messages),
            model_name,
            layer,
            args.combine_consecutive_messages,
            args.convert_tool_to_assistant,
        )
        return LLMModel.load_activations(path) if path.exists() else None

    out: list[dict] = []
    for key, pair in pairs.items():
        orig, contra = pair["orig"]["inputs"], pair["contra"]["inputs"]
        shared_msgs = 0
        for a, b in zip(orig, contra):
            if a["role"] == b["role"] and a["content"] == b["content"]:
                shared_msgs += 1
            else:
                break
        if shared_msgs == 0:
            continue  # nothing identical to measure
        acts = {"orig": activation(orig), "contra": activation(contra)}
        if any(v is None for v in acts.values()):
            print(f"  {key}: no cached activation, skipped")
            continue

        ids, start = {}, {}
        for tag, act in acts.items():
            n_valid = int(act.attention_mask[0].sum())
            ids[tag] = act.input_ids[0][:n_valid].tolist()
            headers = [i for i, t in enumerate(ids[tag]) if t == HEADER_TOKEN]
            if len(headers) < 2:
                raise SystemExit(f"{key}/{tag}: fewer than two chat headers")
            start[tag] = headers[1]  # past the dated system header
        shared = 0
        while (
            start["orig"] + shared < len(ids["orig"])
            and start["contra"] + shared < len(ids["contra"])
            and ids["orig"][start["orig"] + shared]
            == ids["contra"][start["contra"] + shared]
        ):
            shared += 1

        row = {
            "key": key,
            "shared_messages": shared_msgs,
            "n_messages": len(orig),
            "shared_body_tokens": shared,
            "orig_body_tokens": len(ids["orig"]) - start["orig"],
            "contra_body_tokens": len(ids["contra"]) - start["contra"],
        }
        for which, (w, b) in weights.items():
            for tag, act in acts.items():
                x = act.activations[0].float().cpu()
                valid = act.attention_mask[0].bool().cpu()
                scores = x @ w + b
                lo, hi = start[tag], start[tag] + shared
                tail = scores[hi:][valid[hi:]]
                pool = torch.softmax(
                    scores.masked_fill(~valid, float("-inf")) / temperature, dim=0
                )
                row[f"{which}_{tag}_shared_max"] = float(scores[lo:hi].max())
                row[f"{which}_{tag}_tail_max"] = (
                    float(tail.max()) if tail.numel() else None
                )
                row[f"{which}_{tag}_shared_weight"] = float(pool[lo:hi].sum())
                row[f"{which}_{tag}_pooled"] = float((scores * pool).sum())
        out.append(row)

    if out:
        n = len(out)
        print(f"\npairs with a shared prefix: {n}")
        share = sum(r["shared_body_tokens"] / r["orig_body_tokens"] for r in out) / n
        print(f"  mean shared fraction of the conversation body: {share:.1%}")
        for which, name in (("a", args.probe_a), ("b", args.probe_b)):
            print(f"  -- {name}")
            for tag in ("orig", "contra"):
                avg = lambda k: sum(r[f"{which}_{tag}_{k}"] for r in out) / n  # noqa: E731
                print(
                    f"     {tag:6s} per-token max: shared {avg('shared_max'):+7.3f} | "
                    f"tail {avg('tail_max'):+7.3f} | pooling weight on shared "
                    f"{avg('shared_weight'):.3f} | pooled {avg('pooled'):+7.3f}"
                )
            gaps = [
                r[f"{which}_contra_pooled"] - r[f"{which}_orig_pooled"] for r in out
            ]
            right = sum(1 for g in gaps if g > 0)
            print(
                f"     separation (contra - orig) {sum(gaps)/n:+.3f}, "
                f"correct sign {right}/{n}"
            )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dir", type=Path, required=True)
    p.add_argument("--probe-a", default="probe_iter2.pkl")
    p.add_argument("--probe-b", default="probe_iter3.pkl")
    p.add_argument("--redteam-a", default="redteam_postprocessed_iter2.jsonl")
    p.add_argument("--redteam-b", default="redteam_postprocessed_iter3.jsonl")
    p.add_argument("--splits", nargs="+", default=["mt", "mts"])
    p.add_argument("--eval-dataset-dir", type=Path, default=REPO_ROOT / "eval_sets/highstakes")
    p.add_argument(
        "--activations-cache-dir",
        type=Path,
        default=REPO_ROOT / "results_hs_llama1b_deepseekv4pro/eval_activations",
    )
    p.add_argument(
        "--redteam-activation-cache-dir",
        type=Path,
        default=REPO_ROOT / "results_hs_llama1b_deepseekv4pro/base_activations",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--combine-consecutive-messages", action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--convert-tool-to-assistant", action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = p.parse_args()

    probe_a = _load_probe(args.probe_dir / args.probe_a)
    probe_b = _load_probe(args.probe_dir / args.probe_b)
    print(
        f"labels pos={probe_a.pos_class_label!r} neg={probe_a.neg_class_label!r} "
        f"layer={probe_a.layer} model={probe_a.model_name}"
    )

    rows = per_sample(args, probe_a, probe_b)
    stem_a = Path(args.probe_a).stem
    stem_b = Path(args.probe_b).stem
    out_rows = args.probe_dir / f"{stem_a}_vs_{stem_b}_{'_'.join(args.splits)}.jsonl"
    with out_rows.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_rows} ({len(rows)} rows)")

    pairs = added_pairs(args)
    stats = pair_token_stats(args, pairs, probe_a, probe_b)
    out_stats = args.probe_dir / f"{stem_b}_pair_token_stats.json"
    with out_stats.open("w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {out_stats} ({len(stats)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
