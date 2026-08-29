#!/usr/bin/env python
"""Stop-word-filtered word frequencies over the five corpora this run touches.

The corpora are the loop's own training-set ingredients plus the two fixed sets it is
scored on:

- ``base``     — ``data/instructions_llama70b_50.jsonl`` (50 rows, the initial probe)
- ``accepted`` — the 62 generator samples the loop accepted (``accepted_iter13.jsonl``)
- ``imitated`` — the 107 rows written later from those 62 (``instructions_like_accepted62.jsonl``)
- ``eval_pos`` / ``eval_neg`` — that same 107 split by the sign of each family's own eval
  delta when added to ``base ∪ 62`` alone (50 rows vs 57). They partition ``imitated``.
- ``poison``   — the 60 rejected samples fed in by ``poison_curve.py`` (it0b1..it1b3)
- ``dev``      — ``dev_samples/instructions`` (early stopping AND the ΔAUROC scoring set)
- ``eval``     — ``eval_sets/instructions`` (the held-out splits)

Only message *content* is counted (never the role keys), lowercased, split on a word
regex, and filtered against scikit-learn's ENGLISH_STOP_WORDS. Rates are per 10k
content words so corpora of wildly different sizes are comparable, which is the only
way ``accepted`` (62 rows) and ``eval`` (1302 rows) can be read side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

REPO = Path(__file__).resolve().parent.parent
RUN_DIR = REPO / "results_gen_gemma27b_instructions_nemotron"

# The six rejected batches poison_curve.py walked through, in its own order.
POISON_BATCHES = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3)]

# Families whose own eval delta was positive / negative when added to base ∪ 62 alone
# (like62_directions_results.csv). Together they are the 107.
EVAL_POSITIVE = {"it11b3", "it4b3", "it1b1", "it5b4"}
EVAL_NEGATIVE = {"it7b2", "it2b0", "it9b1", "it0b4"}

WORD_RE = re.compile(r"[a-z][a-z'\-]*")
# Contraction tails and single letters survive the stop list; they are not content.
EXTRA_STOP = {"ve", "ll", "re", "don", "doesn", "isn", "didn", "won", "wouldn",
              "couldn", "shouldn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
              "let", "s", "t", "d", "m"}
STOP = set(ENGLISH_STOP_WORDS) | EXTRA_STOP


def tokenize(text: str) -> list[str]:
    out = []
    for w in WORD_RE.findall(text.lower()):
        w = w.strip("-'")
        if len(w) < 3 or w in STOP:
            continue
        out.append(w)
    return out


def _messages(inputs) -> list[dict]:
    """``inputs`` is a list of messages, or a JSON-encoded string of one."""
    if isinstance(inputs, str):
        inputs = json.loads(inputs)
    return list(inputs)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_slot_corpora() -> dict[str, list[tuple[str, list[dict]]]]:
    """The eight Δdev rank slots, pooled over the seven same-prompt draws.

    ``data/union_slot{k}.jsonl`` holds the family that ranked k-th by Δdev in each of
    reps 1, 2, 4-8 — 85-105 rows each, so the eight are size-comparable and the only
    thing that varies is which families the ranking put in that slot.
    """
    out: dict[str, list[tuple[str, list[dict]]]] = {}
    for k in range(1, 9):
        p = REPO / f"data/union_slot{k}.jsonl"
        if p.exists():
            out[f"slot{k}"] = [(r["labels"], _messages(r["inputs"])) for r in _read_jsonl(p)]
    return out


def load_corpora() -> dict[str, list[tuple[str, list[dict]]]]:
    """corpus name → [(label, messages), ...]"""
    corpora: dict[str, list[tuple[str, list[dict]]]] = {}

    corpora["base"] = [
        (r["labels"], _messages(r["inputs"]))
        for r in _read_jsonl(REPO / "data/instructions_llama70b_50.jsonl")
    ]
    corpora["accepted"] = [
        (r["labels"], _messages(r["inputs"]))
        for r in _read_jsonl(RUN_DIR / "accepted_iter13.jsonl")
    ]

    latest: dict[tuple[int, int], dict] = {}
    for r in _read_jsonl(RUN_DIR / "batches.jsonl"):
        latest[(r["iteration"], r["batch_index"])] = r
    poison: list[tuple[str, list[dict]]] = []
    for key in POISON_BATCHES:
        for s in latest[key]["samples"]:
            poison.append((s["label"], list(s["messages"])))
    corpora["poison"] = poison

    imitated = _read_jsonl(REPO / "data/instructions_like_accepted62.jsonl")
    corpora["imitated"] = [(r["labels"], _messages(r["inputs"])) for r in imitated]
    for name, fams in (("eval_pos", EVAL_POSITIVE), ("eval_neg", EVAL_NEGATIVE)):
        corpora[name] = [(r["labels"], _messages(r["inputs"]))
                         for r in imitated if r["family"] in fams]

    for name, d in (("dev", REPO / "dev_samples/instructions"),
                    ("eval", REPO / "eval_sets/instructions")):
        docs: list[tuple[str, list[dict]]] = []
        for f in sorted(d.glob("*.jsonl")):
            for r in _read_jsonl(f):
                docs.append((r["labels"], _messages(r["inputs"])))
        corpora[name] = docs
    return corpora


def counts_for(docs, role: str | None = None) -> Counter:
    c: Counter = Counter()
    for _label, messages in docs:
        for m in messages:
            if role and m.get("role") != role:
                continue
            c.update(tokenize(str(m.get("content", ""))))
    return c


def rate(c: Counter, w: str) -> float:
    total = sum(c.values())
    return 10000.0 * c[w] / total if total else 0.0


def _write_csv(path: Path, order: list[str], counts: dict[str, Counter]) -> None:
    """Every word in any corpus' top-200, with each corpus' count and per-10k rate."""
    keep: set[str] = set()
    for name in order:
        keep.update(w for w, _ in counts[name].most_common(200))
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["word"] + [f"{n}_count" for n in order] + [f"{n}_rate_per10k" for n in order])
        for w in sorted(keep, key=lambda w: -rate(counts["eval"], w)):
            wr.writerow([w] + [counts[n][w] for n in order]
                        + [f"{rate(counts[n], w):.2f}" for n in order])
    print(f"\nwrote {Path(path).resolve()} ({len(keep)} words)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=25, help="rows per corpus in the report")
    ap.add_argument("--role", choices=["user", "assistant"], default=None,
                    help="restrict to one speaker (default: both)")
    ap.add_argument("--slots", action="store_true",
                    help="analyse the eight Δdev rank slots against dev and eval instead")
    ap.add_argument("--csv", type=Path, default=RUN_DIR / "word_frequency.csv")
    args = ap.parse_args()

    corpora = load_corpora()
    if args.slots:
        slots = load_slot_corpora()
        corpora = {**slots, "dev": corpora["dev"], "eval": corpora["eval"]}
        order = list(slots) + ["dev", "eval"]
    else:
        order = ["base", "accepted", "imitated", "eval_pos", "eval_neg", "poison", "dev", "eval"]
    counts = {name: counts_for(docs, args.role) for name, docs in corpora.items()}

    print(f"corpus     docs   content words   vocab   words/doc")
    for name in order:
        c, docs = counts[name], corpora[name]
        tot = sum(c.values())
        print(f"{name:<10} {len(docs):>5}   {tot:>13,}   {len(c):>5}   {tot/max(len(docs),1):>9.1f}")

    for name in order:
        c = counts[name]
        print(f"\n=== {name} — top {args.top} (rate = per 10k content words) ===")
        print(f"{'word':<18}{'count':>7}{'rate':>9}   {'eval rate':>9}  {'x eval':>7}")
        for w, n in c.most_common(args.top):
            er = rate(counts["eval"], w)
            ratio = f"{rate(c, w)/er:>7.1f}" if er else "      —"
            print(f"{w:<18}{n:>7}{rate(c, w):>9.1f}   {er:>9.1f}  {ratio}")

    # How far each corpus' unigram distribution sits from the set it is judged on.
    print("\n=== overlap with eval (unigram distributions) ===")
    print(f"{'corpus':<10}{'cosine':>8}{'top100 shared':>15}{'OOV tokens':>12}{'OOV types':>11}")
    ev = counts["eval"]
    ev_top = {w for w, _ in ev.most_common(100)}
    ev_total = sum(ev.values())
    for name in order:
        c = counts[name]
        tot = sum(c.values())
        dot = sum(c[w] * ev[w] for w in c if w in ev)
        norm = (sum(v * v for v in c.values()) ** 0.5) * (sum(v * v for v in ev.values()) ** 0.5)
        cos = dot / norm if norm else 0.0
        shared = len(ev_top & {w for w, _ in c.most_common(100)})
        oov_tok = sum(n for w, n in c.items() if w not in ev)
        oov_typ = sum(1 for w in c if w not in ev)
        print(f"{name:<10}{cos:>8.3f}{shared:>13}/100{100*oov_tok/max(tot,1):>11.1f}%{100*oov_typ/max(len(c),1):>10.1f}%")

    print("\n=== pairwise cosine of unigram rate vectors ===")
    print(" " * 10 + "".join(f"{n:>10}" for n in order))
    for a in order:
        row = []
        for b in order:
            ca, cb = counts[a], counts[b]
            dot = sum(ca[w] * cb[w] for w in ca if w in cb)
            norm = (sum(v * v for v in ca.values()) ** 0.5) * (sum(v * v for v in cb.values()) ** 0.5)
            row.append(dot / norm if norm else 0.0)
        print(f"{a:<10}" + "".join(f"{v:>10.3f}" for v in row))

    # Words that most distinguish each generated corpus from eval, by log-odds with a
    # +1 prior: high count AND high rate ratio, so a single freak word can't top it.
    for name in [n for n in ("base", "accepted", "imitated", "eval_pos", "eval_neg", "poison") if n in counts]:
        c = counts[name]
        tot = sum(c.values())
        scored = sorted(
            ((w, n, (1e4 * n / tot) / (1e4 * (ev[w] + 1) / ev_total)) for w, n in c.items() if n >= 5),
            key=lambda t: -t[2],
        )
        print(f"\n=== {name} — most over-represented vs eval (count >= 5) ===")
        for w, n, r in scored[:15]:
            print(f"{w:<18}{n:>6}{1e4*n/tot:>9.1f} vs {rate(ev, w):>7.1f}   x{r:>7.1f}")

    # The imitations were written FROM the accepted rows, so the sharper question for
    # them is not distance from eval but what they did to their own source vocabulary.
    if "accepted" not in counts:
        _write_csv(args.csv, order, counts)
        return
    ac, im = counts["accepted"], counts["imitated"]
    ac_tot, im_tot = sum(ac.values()), sum(im.values())
    print("\n=== imitated vs its source (accepted): shifts on words common to both ===")
    print(f"{'word':<18}{'accepted':>10}{'imitated':>10}{'ratio':>8}")
    common = [w for w in set(ac) | set(im) if ac[w] + im[w] >= 12]
    scored = sorted(common, key=lambda w: -(1e4 * im[w] / im_tot + 1) / (1e4 * ac[w] / ac_tot + 1))
    print("-- amplified by the imitations --")
    for w in scored[:8]:
        print(f"{w:<18}{1e4*ac[w]/ac_tot:>10.1f}{1e4*im[w]/im_tot:>10.1f}"
              f"{(1e4*im[w]/im_tot+1)/(1e4*ac[w]/ac_tot+1):>8.2f}")
    print("-- dropped by the imitations --")
    for w in scored[-8:]:
        print(f"{w:<18}{1e4*ac[w]/ac_tot:>10.1f}{1e4*im[w]/im_tot:>10.1f}"
              f"{(1e4*im[w]/im_tot+1)/(1e4*ac[w]/ac_tot+1):>8.2f}")
    shared_types = len(set(ac) & set(im))
    print(f"\nvocabulary: {len(ac)} accepted types, {len(im)} imitated types, "
          f"{shared_types} shared ({100*shared_types/len(ac):.0f}% of accepted's, "
          f"{100*shared_types/len(im):.0f}% of the imitations')")
    print(f"imitated tokens drawn from accepted's vocabulary: "
          f"{100*sum(n for w, n in im.items() if w in ac)/im_tot:.1f}%")

    # Does anything in the vocabulary mark the families that helped eval on their own?
    pc, nc = counts["eval_pos"], counts["eval_neg"]
    pt, nt = sum(pc.values()), sum(nc.values())
    dot = sum(pc[w] * nc[w] for w in pc if w in nc)
    norm = (sum(v * v for v in pc.values()) ** 0.5) * (sum(v * v for v in nc.values()) ** 0.5)
    print(f"\n=== eval_pos vs eval_neg (cosine {dot / norm:.3f}) ===")
    print(f"{'word':<18}{'pos':>8}{'neg':>8}{'ratio':>8}")
    both = [w for w in set(pc) | set(nc) if pc[w] + nc[w] >= 10]
    ranked = sorted(both, key=lambda w: -(1e4 * pc[w] / pt + 1) / (1e4 * nc[w] / nt + 1))
    print("-- leans to the families that HELPED eval --")
    for w in ranked[:10]:
        print(f"{w:<18}{1e4*pc[w]/pt:>8.1f}{1e4*nc[w]/nt:>8.1f}"
              f"{(1e4*pc[w]/pt+1)/(1e4*nc[w]/nt+1):>8.2f}")
    print("-- leans to the families that HURT eval --")
    for w in ranked[-10:]:
        print(f"{w:<18}{1e4*pc[w]/pt:>8.1f}{1e4*nc[w]/nt:>8.1f}"
              f"{(1e4*pc[w]/pt+1)/(1e4*nc[w]/nt+1):>8.2f}")

    _write_csv(args.csv, order, counts)


if __name__ == "__main__":
    main()
