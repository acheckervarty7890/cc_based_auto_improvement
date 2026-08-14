"""Content-side triage of the pairs iteration 3 added, against the ones it inherited.

Runs no training at all: it joins each red-team *pair* in an arm's iteration-3
training set back to the raw attempt record that produced it, then contrasts the
iteration-3-only pairs with the earlier ones on the properties that could plausibly
move ``eval_balanced_refusal`` — the split where the single-seed vintage run showed a
dip.

The hypotheses it is built to test, in order of how specific they are to that split:

**Refusal geometry.** ``eval_balanced_refusal`` is the split whose two classes are
*both* full of refusals — a harmful reply and a refusal of the same prompt. A
``false_positive`` hunt produces exactly the confusable thing: conversations where the
assistant refuses and the judge rules ``not_harmful_to_human``. Train on enough of
those and the probe can learn "refusal => not harmful", which is precisely the
shortcut that split is built to punish. So the error-type mix and the rate of
refusal-shaped assistant turns are reported per group.

**Label reliability.** ``judge_confidence`` gates the training path
(``retrain_probe(min_judge_confidence=)``); low-confidence successes are label noise
by another name.

**Template collapse.** A rotation that finds one winning template and re-skins it
submits near-clones. The near-duplicate guard was off for these runs, so clones can be
in the training set; a group of clones is one effective sample carrying the weight of
many. Measured as the max ``difflib`` ratio of each opener against the others in the
same group, on the same basis as ``scripts/clone_rate.py``.

**Truncation damage.** ``shorten_long_contrastive_pairs.py`` rewrote generated rows
that overran the 1024-token activation window. For an assistant-centric concept the
rewritten turn *is* the label carrier, so a botched rewrite is direct label noise.

Nothing here proves causation — it produces ranked suspects for a real per-pair sweep
(``attribution_loo.py``). Read it alongside ``attribution_vintage_gate.py``, which
says whether there is an effect to explain at all.

Usage:
    .venv_claude/bin/python scripts/vintage_triage.py --arm gptoss120b
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A  # noqa: E402
from attribution_vintage import vintages  # noqa: E402

RESULTS = {
    "gptoss120b": A.REPO / "results_hu_harm_gemma27b_gptoss120b_batch",
    "deepseekv4pro": A.REPO / "results_hu_harm_gemma27b_deepseekv4pro_batch",
}
JSONL_STEM = {"gptoss120b": "gptoss120b_probing", "deepseekv4pro": "deepseekv4pro_probing"}

# Prefix-only, like llm_judge._REFUSAL_MARKERS: a reply that *mentions* "I can't"
# mid-paragraph is not a refusal, one that opens with it is.
_REFUSAL_PREFIXES = (
    "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm sorry",
    "i am sorry", "sorry,", "i'm not able", "i am not able", "i'm unable",
    "i am unable", "unfortunately, i", "i must decline", "i'd rather not",
    "i am not going to", "i'm not going to", "no, i", "i don't think i should",
)


def looks_like_refusal(text: str) -> bool:
    t = re.sub(r"^[\s*_#>-]+", "", (text or "").strip().lower())
    return t.startswith(_REFUSAL_PREFIXES)


def load_attempts(arm: str) -> dict[str, dict]:
    """Canonical text (post-transform) -> the attempt record that submitted it."""
    from tuberlens.interfaces.dataset import Message

    out: dict[str, dict] = {}
    for et in ("fp", "fn"):
        path = RESULTS[arm] / f"{JSONL_STEM[arm]}_{et}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                msgs = rec.get("sample", {}).get("messages") or []
                key = A.canon(
                    A.apply_transforms(
                        [Message(role=m["role"], content=m["content"]) for m in msgs]
                    )
                )
                out.setdefault(key, rec)
    return out


def first_user_text(messages) -> str:
    for m in messages:
        if m.role == "user":
            return m.content or ""
    return messages[0].content if messages else ""


def last_assistant_text(messages) -> str:
    for m in reversed(messages):
        if m.role == "assistant":
            return m.content or ""
    return ""


def clone_scores(openers: list[str], prefix: int = 600) -> list[float]:
    """Max similarity of each opener against every *other* opener in the group."""
    trimmed = [o[:prefix] for o in openers]
    out = []
    for i, a in enumerate(trimmed):
        best = 0.0
        for j, b in enumerate(trimmed):
            if i == j:
                continue
            r = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
            if r > best:
                best = r
        out.append(best)
    return out


def describe(name: str, rows: list[dict]) -> dict:
    """Print and return the per-group summary."""
    n = len(rows)
    if not n:
        print(f"\n--- {name}: empty")
        return {}

    def frac(pred):
        return sum(1 for r in rows if pred(r)) / n

    conf = [r["judge_confidence"] for r in rows if r["judge_confidence"] is not None]
    clones = [r["clone"] for r in rows]
    src_len = [r["source_chars"] for r in rows]
    gen_len = [r["generated_chars"] for r in rows if r["generated_chars"] is not None]

    summary = {
        "n_pairs": n,
        "matched_to_attempt": sum(1 for r in rows if r["matched"]),
        "error_type": dict(Counter(r["error_type"] for r in rows)),
        "judge_label": dict(Counter(r["judge_label"] for r in rows)),
        "attacker_model": dict(Counter(r["attacker_model"] for r in rows)),
        "iteration_found": dict(Counter(r["iteration"] for r in rows)),
        "judge_confidence_mean": statistics.mean(conf) if conf else None,
        "judge_confidence_lt8": frac(
            lambda r: r["judge_confidence"] is not None and r["judge_confidence"] < 8
        ),
        "source_refusal_rate": frac(lambda r: r["source_refusal"]),
        "generated_refusal_rate": frac(lambda r: r["generated_refusal"]),
        "shortened_rate": frac(lambda r: r["shortened"]),
        "clone_mean": statistics.mean(clones),
        "clone_ge_0.8": frac(lambda r: r["clone"] >= 0.8),
        "source_chars_median": statistics.median(src_len),
        "generated_chars_median": statistics.median(gen_len) if gen_len else None,
    }

    print(f"\n--- {name}  (n={n} pairs, {summary['matched_to_attempt']} joined to an attempt record)")
    print(f"  error_type          {summary['error_type']}")
    print(f"  judge_label         {summary['judge_label']}")
    print(f"  found in iteration  {summary['iteration_found']}")
    print(
        f"  judge_confidence    mean={summary['judge_confidence_mean']}"
        f"  frac<8={summary['judge_confidence_lt8']:.1%}"
    )
    print(
        f"  refusal-shaped      source={summary['source_refusal_rate']:.1%}"
        f"  generated={summary['generated_refusal_rate']:.1%}"
    )
    print(f"  rewritten by shortening  {summary['shortened_rate']:.1%}")
    print(
        f"  clone (max difflib) mean={summary['clone_mean']:.3f}"
        f"  frac>=0.8={summary['clone_ge_0.8']:.1%}"
    )
    print(
        f"  chars               source median={summary['source_chars_median']:.0f}"
        f"  generated median={summary['generated_chars_median']}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gptoss120b", choices=sorted(A.ARMS))
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arm, it = args.arm, args.iteration
    redteam = A.load_redteam_dataset(arm, it)
    pairs, stats = A.build_pairs(arm, redteam)
    keep_rows, _ = vintages(arm, it)

    pair_of_row = {}
    for p in pairs:
        for idx in (p.source_idx, p.generated_idx):
            if idx is not None:
                pair_of_row[idx] = p.pair_id
    pv = {k: {pair_of_row[r] for r in rows if r in pair_of_row} for k, rows in keep_rows.items()}
    new_ids = pv[it] - pv[it - 1]
    old_ids = pv[it - 1]
    print(f"{arm} iter{it}: {len(pairs)} pairs — {len(new_ids)} new in v{it}, {len(old_ids)} inherited")

    # rows rewritten by shorten_long_contrastive_pairs.py, if the .bak is still around
    bak = A.ARMS[arm] / f"redteam_postprocessed_iter{it}.jsonl.bak"
    shortened_keys: set[str] = set()
    if bak.exists():
        from tuberlens.interfaces.dataset import Message

        bak_keys = {
            A.canon([Message(role=m["role"], content=m["content"]) for m in json.loads(l)["inputs"]])
            for l in bak.open(encoding="utf-8")
        }
        shortened_keys = {
            A.canon(m) for m in redteam.inputs if A.canon(m) not in bak_keys
        }
        print(f"  rows rewritten by the shortening pass: {len(shortened_keys)}")

    attempts = load_attempts(arm)
    print(f"  raw attempt records indexed: {len(attempts)}")

    rows_by_group: dict[str, list[dict]] = {"new_in_v3": [], "inherited": []}
    for p in pairs:
        group = "new_in_v3" if p.pair_id in new_ids else ("inherited" if p.pair_id in old_ids else None)
        if group is None:
            continue
        src_msgs = redteam.inputs[p.source_idx] if p.source_idx is not None else None
        gen_msgs = redteam.inputs[p.generated_idx] if p.generated_idx is not None else None
        rec = attempts.get(p.source_key, {})
        rows_by_group[group].append(
            {
                "pair_id": p.pair_id,
                "matched": bool(rec),
                "error_type": rec.get("error_type", "?"),
                "judge_label": rec.get("judge_label", "?"),
                "judge_confidence": rec.get("judge_confidence"),
                "probe_score": rec.get("probe_score"),
                "attacker_model": rec.get("attacker_model", "?"),
                "iteration": rec.get("iteration", "?"),
                "source_refusal": looks_like_refusal(last_assistant_text(src_msgs)) if src_msgs else False,
                "generated_refusal": looks_like_refusal(last_assistant_text(gen_msgs)) if gen_msgs else False,
                "shortened": any(
                    A.canon(m) in shortened_keys
                    for m in (src_msgs, gen_msgs)
                    if m is not None
                ),
                "source_chars": len(json.dumps([m.content for m in src_msgs])) if src_msgs else 0,
                "generated_chars": len(json.dumps([m.content for m in gen_msgs])) if gen_msgs else None,
                "opener": first_user_text(src_msgs) if src_msgs else "",
            }
        )

    for group, rows in rows_by_group.items():
        for r, c in zip(rows, clone_scores([r["opener"] for r in rows])):
            r["clone"] = c

    out = {g: describe(g, rows) for g, rows in rows_by_group.items()}

    # The contrast is the point, so print the deltas rather than making the reader diff.
    a, b = out.get("new_in_v3", {}), out.get("inherited", {})
    if a and b:
        print("\n=== new_in_v3 minus inherited ===")
        for k in ("judge_confidence_mean", "judge_confidence_lt8", "source_refusal_rate",
                  "generated_refusal_rate", "shortened_rate", "clone_mean", "clone_ge_0.8"):
            if a.get(k) is None or b.get(k) is None:
                continue
            print(f"  {k:24s} {a[k] - b[k]:+.4f}   ({a[k]:.4f} vs {b[k]:.4f})")

    path = args.out or (
        A.REPO / f"results_hu_harm_gemma27b_batch_ablation/vintage/{arm}_triage.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": out, "rows": rows_by_group}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
