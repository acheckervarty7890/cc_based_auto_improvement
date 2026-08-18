#!/usr/bin/env python
"""Build a self-contained HTML viewer of the training-data pathologies in the
instruction-following attacker-model experiment (gpt-oss-120b vs nemotron arms).

Four evidence tabs, one methodology tab:

``twins``
    Pairs of conversations inside ONE retrain's training set that are >= --tau
    similar to each other but carry OPPOSITE labels. These are the rows that make
    the set internally contradictory: in the probe's activation space they are
    nearly the same point with a flipped target. Similarity is
    ``difflib.SequenceMatcher(autojunk=False)`` on the whole rendered conversation
    (``role: content`` joined), matching how ``persistence._is_near`` scores
    openers — ``autojunk=False`` is load-bearing there and here for the same
    reason (the heuristic makes the ratio asymmetric above 200 chars).

``format`` / ``nonformat``
    ``--n-examples`` red-team SUCCESSES per arm from each side of the
    format-constraint split, so the two populations can be read side by side. The
    classifier is the ``FORMAT`` regex below over the conversation's user turns:
    it fires on explicit output-shape constraints ("only", "exactly N words",
    "no extra text", JSON/haiku/bullet counts). "Non-format" is everything else —
    ordinary tasks where following the instruction is a semantic property, which
    is what all seven ``eval_instructions`` splits actually test.

``mislabels``
    Generated contrastive pairs labelled "follows the instruction" that the
    prompt's own constraint refutes. Ten constraint families are adjudicated (see
    ``checks``), covering ~10-13% of pairs — so this is a census of a SUBSAMPLE and
    the rate is a rate *within* it, not over the corpus. Word counts are taken under
    the most lenient tokenizer that could make the label true, so a row shown here
    is wrong under every reading.

``invisible``
    Successes whose assistant turn carries a zero-width / invisible character or a
    mixed-script homoglyph — an attack that defeats the judge (which reads the raw
    string and correctly says "there is an extra character") while being invisible
    to a reader and near-invisible to the probe. Rendered with the offending
    codepoints made visible. Note that not every hit is an attack: NNBSP inside
    ``225 g`` is ordinary typography, so the tab labels each hit's codepoints and
    lets you judge.

Everything is embedded in the page; no network access at view time.

Usage:
    .venv_claude/bin/python scripts/build_instructions_arms_viewer.py \
        --arm gptoss:probes/instructions_gemma27b_gptoss:results_instructions_gemma27b_gptoss/gptoss120b_probing \
        --arm nemotron:probes/instructions_gemma27b_nemotron:results_instructions_gemma27b_nemotron/nemotron_probing \
        --out viewers/instructions_gemma27b_arms.html
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path

# Codepoints that render as nothing (or as plain space) but survive into the
# tokenizer. An attacker can append one to an otherwise perfectly compliant reply
# and the judge, reading the raw string, correctly rules "extra character present".
INVISIBLE = {
    "​": "ZWSP", "‌": "ZWNJ", "‍": "ZWJ", "﻿": "BOM",
    "­": "SOFT-HYPHEN", "⁠": "WORD-JOINER", "᠎": "MONGOLIAN-VS",
    "⁡": "FN-APPLY", " ": "NBSP", " ": "NNBSP",
    " ": "THIN-SPACE", " ": "FIGURE-SPACE", "ㅤ": "HANGUL-FILLER",
}

# Explicit output-shape constraints. Deliberately broad on the "shape" side and
# silent on topic, so the complement ("non-format") is ordinary semantic tasks.
FORMAT = re.compile(
    r"\b(only|exactly|precisely|no more than|at most|do not (use|include|mention|add)|"
    r"don't (use|include|mention)|without (using|the word)|nothing else|no extra|"
    r"no additional|syllab|bullet|numbered list|json|uppercase|lowercase|acrostic|"
    r"haiku|alphabetical|palindrome|raw number|plain text)\b",
    re.I,
)

# ------------------------------------------------------- constraint adjudication
#
# Only constraints a script can settle with no semantic judgement are adjudicated;
# everything else abstains so it stays out of the denominator. Three guards, each
# added after hand-inspection turned up a false positive of an earlier version:
#
#   1. SINGLE EXCHANGE ONLY — assistant turns get concatenated to check them, so in
#      a multi-turn conversation "Say ONLY 'approved'. Now say ONLY 'denied'." reads
#      as one reply 'approved denied' and every literal/JSON check fires spuriously.
#   2. NO DISTRIBUTIVE QUANTIFIER — "each tip must be ... exactly eight words" scopes
#      the count to a sub-unit this cannot segment. Enumerating the sub-unit nouns
#      kept missing new ones (tip, memoir, stanza), so it abstains on each/every/per.
#   3. NO UNCHECKED CO-CONSTRAINT — a prompt asking for 3 lines AND 5-7-5 syllables is
#      only half-verifiable, so passing the half we can check proves nothing about the
#      label. Any co-occurring constraint outside this repertoire -> abstain.
#
# The verdict is also used ONE-SIDED (see find_mislabels): a failed check refutes a
# "follows" label, but a passed check cannot establish one, since clauses outside the
# repertoire may still have failed.


NUM={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
     "eleven":11,"twelve":12}
def _n(s):
    s=s.strip().lower(); return int(s) if s.isdigit() else NUM.get(s)
NW=r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"

# Any distributive quantifier at all. "each tip must be ... exactly eight words"
# scopes the count to a sub-unit; enumerating the nouns kept missing new ones
# (tip, memoir, stanza...), so this abstains on the quantifier itself.
PER_ITEM=re.compile(r"\b(each|every|per|apiece|respectively)\b",re.I)
# A carve-out ("exactly 100 words, not including the title") changes what counts.
CARVEOUT=re.compile(r"\b(not including|excluding|not counting|apart from|besides the)\b",re.I)
# Constraints this file cannot verify. If one co-occurs, abstain entirely.
UNCHECKED=re.compile(r"\b(syllab|rhyme|rhyming|abab|acrostic|palindrome|alphabetical|"
                     r"iambic|haiku|sonnet|limerick|anagram|ascending|descending|"
                     r"in order|sorted)\b",re.I)

def words(a): return (len(a.split()), len(re.findall(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*",a)))

def checks(messages):
    """-> (family, follows: bool, detail) or None to abstain."""
    us=[m for m in messages if m.get("role")=="user"]
    a_s=[m for m in messages if m.get("role")=="assistant"]
    if len(us)!=1 or len(a_s)!=1:
        return None                                  # guard 1
    u, a = us[0]["content"], a_s[0]["content"]
    ul, s = u.lower(), a.strip()
    if PER_ITEM.search(ul) or CARVEOUT.search(ul):
        return None                                  # guard 2

    def co(*allowed):
        """True when the prompt carries a constraint outside `allowed`."""
        if UNCHECKED.search(ul):
            return True
        others = {"words": rf"(exactly {NW} words|\b{NW}-word\b|\bword count\b)",
                  "lines": rf"exactly {NW} lines",
                  # "must not contain the letter 'e'" is the same constraint as
                  # "without using the letter 'e'" and has to be caught too.
                  "letter": r"(us(?:e|ing)|contain(?:ing|s)?|includ(?:e|ing)) the letter",
                  "punct": r"\bno punctuation\b",
                  "case": r"\b(uppercase|lowercase|all caps)\b",
                  # "the 3rd word must be 'blue'" — the count can be right while the
                  # positional clause fails, which is what the label reflects.
                  "positional": r"\b((\d+(?:st|nd|rd|th)|first|second|third|last) word|"
                                r"must (include|contain|end with|begin with|start with))\b"}
        return any(re.search(p, ul) for k, p in others.items() if k not in allowed)

    m=re.search(rf"exactly {NW} words",ul)
    if m and _n(m.group(1)) and not co("words"):
        w=_n(m.group(1)); ws,hy=words(a)
        return ("exactly-N-words", w in (ws,hy), f"want {w} words, got {ws}/{hy}")

    m=re.search(r"(?:do not|don't|without|avoid|never) us(?:e|ing) the letter ['\"]?([a-z])['\"]?",ul)
    if m and not co("letter"):
        L=m.group(1); hit=[w for w in re.findall(r"[A-Za-z']+",a) if L in w.lower()]
        return ("letter-ban", not hit, f"letter '{L}' banned; {len(hit)} offender(s): {', '.join(hit[:4])}")

    m=re.search(r"(?:do not|don't|never|without)\s+(?:use|using|mention|mentioning|include|including|say|saying)\s+(?:the words?\s+)?((?:['\"][^'\"]{2,30}['\"][,\s]*(?:or|and)?\s*)+)",ul)
    if m and not co():
        ws_=re.findall(r"['\"]([^'\"]{2,30})['\"]",m.group(1))
        if ws_:
            hit=[w for w in ws_ if re.search(r"\b"+re.escape(w.lower()),a.lower())]
            return ("forbidden-word", not hit, f"banned {ws_}; found {hit}")

    m=re.search(r"(?:reply|respond|answer|output|say|write)\s+(?:back\s+)?(?:with\s+)?(?:only|exactly|just)\s+(?:the\s+(?:word|phrase|text|string)\s+)?['\"]([^'\"]{1,60})['\"]",ul)
    if m and re.search(r"\b(only|exactly|just|nothing else|no extra|no other)\b",ul) and not co():
        want=m.group(1).strip()
        return ("exact-literal", s.strip().lower()==want.lower(), f"want {want!r}, got {s[:40]!r}")

    if re.search(r"\b(only|nothing else|no extra text|no commentary|no other text)\b",ul) and re.search(r"\bjson\b",ul) and not co():
        try:
            json.loads(s); return None               # parses; other clauses unverified
        except Exception:
            return ("json-only", False, "response is not parseable JSON")

    if re.search(r"\bno punctuation\b",ul) and not co("punct"):
        hit=[c for c in a if unicodedata.category(c).startswith("P")]
        return ("no-punctuation", not hit, f"{len(hit)} punctuation char(s)")

    if re.search(r"\b(all (caps|uppercase)|in uppercase|entirely uppercase)\b",ul) and not co("case"):
        return ("all-uppercase", a.upper()==a, "uppercase check")
    if re.search(r"\b(all lowercase|in lowercase|entirely lowercase)\b",ul) and not co("case"):
        return ("all-lowercase", a.lower()==a, "lowercase check")

    m=re.search(rf"exactly {NW} lines",ul)
    if m and _n(m.group(1)) and not co("lines"):
        w=_n(m.group(1)); got=len([l for l in a.splitlines() if l.strip()])
        return ("exactly-N-lines", got==w, f"want {w} lines, got {got}")

    return None

POS_LABEL = "assistant_follows_the_instruction"


# ---------------------------------------------------------------- data loading

def canon(messages: list[dict]) -> str:
    return json.dumps(
        [[m.get("role", ""), m.get("content", "")] for m in messages], ensure_ascii=False
    )


def render_conv(messages: list[dict]) -> str:
    return " ".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)


def parts(messages: list[dict]) -> tuple[str, str]:
    """(all user text, all assistant text)."""
    return (
        " ".join(m["content"] for m in messages if m.get("role") == "user"),
        " ".join(m["content"] for m in messages if m.get("role") == "assistant"),
    )


def load_postprocessed(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        msgs = r["inputs"] if isinstance(r["inputs"], list) else json.loads(r["inputs"])
        out.append({"id": r.get("id", ""), "messages": msgs, "label": r["label"]})
    return out


def load_cache(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line)["record"])
    return out


def load_successes(base: Path) -> list[dict]:
    """Red-team successes across both error-type JSONLs of one arm."""
    out = []
    for et in ("fp", "fn"):
        p = base.with_name(base.name + f"_{et}.jsonl")
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("success"):
                continue
            out.append(
                {
                    "messages": r["sample"]["messages"],
                    "iteration": r.get("iteration", -1),
                    "error_type": r.get("error_type", ""),
                    "probe_score": r.get("probe_score"),
                    "judge_label": r.get("judge_label", ""),
                    "judge_reason": r.get("judge_reason", ""),
                    "judge_confidence": r.get("judge_confidence", 0),
                    "round": r.get("round", -1),
                }
            )
    return out


# ---------------------------------------------------------------- tab builders

def mark_messages(messages: list[dict]) -> list[dict]:
    return [
        {"role": m.get("role", ""), "content": mark_invisible(m.get("content", ""))}
        for m in messages
    ]


def find_twins(rows: list[dict], generated: set[str], *, tau: float, cap: int) -> list[dict]:
    """Opposite-label pairs at >= tau similarity, most similar first.

    The length pre-filter is what keeps this O(n^2) scan tractable: two texts whose
    lengths differ by more than (1 - tau) can't reach ratio tau, since the ratio is
    bounded by 2*matches/(len_a+len_b) <= 2*min/(min+max).
    """
    texts = [render_conv(r["messages"]) for r in rows]
    keys = [canon(r["messages"]) for r in rows]
    found: list[dict] = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i]["label"] == rows[j]["label"]:
                continue
            a, b = texts[i], texts[j]
            lo, hi = (len(a), len(b)) if len(a) < len(b) else (len(b), len(a))
            if not hi or 2 * lo / (lo + hi) < tau:
                continue
            ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
            if ratio < tau:
                continue
            # The whole point of some of these pairs is a codepoint that renders as
            # nothing, so the displayed copy has to make it visible or the card looks
            # like two identical conversations with opposite labels and no explanation.
            cps = sorted(set(scan_invisible(a)[0]) | set(scan_invisible(b)[0]))
            found.append(
                {
                    "ratio": round(ratio, 4),
                    "a": mark_messages(rows[i]["messages"]),
                    "a_label": rows[i]["label"],
                    "a_gen": keys[i] in generated,
                    "b": mark_messages(rows[j]["messages"]),
                    "b_label": rows[j]["label"],
                    "b_gen": keys[j] in generated,
                    "codepoints": cps,
                }
            )
    found.sort(key=lambda d: -d["ratio"])
    return found[:cap]


def find_mislabels(cache: list[dict]) -> tuple[list[dict], dict]:
    """Generated pairs labelled "follows" that the constraint refutes.

    Used ONE-SIDED on purpose. A failed check is conclusive — the response does not
    satisfy a constraint stated verbatim in the prompt, so "follows" is wrong. A
    passed check is not: the prompt may carry clauses outside this repertoire that
    the response failed, which is exactly what a "does not follow" label may be
    recording. Adjudicating that direction too produced false positives (a reply
    that was the right length but put the wrong word third), so it is withheld.
    """
    rows: list[dict] = []
    stats = {
        "pairs": len(cache), "checkable": 0, "refuted": 0,
        "by_family": {},
    }
    for rec in cache:
        got = checks(rec["inputs"])
        if got is None:
            continue
        family, actually_follows, detail = got
        stats["checkable"] += 1
        fam = stats["by_family"].setdefault(family, {"n": 0, "refuted": 0})
        fam["n"] += 1
        if actually_follows or rec["labels"] != POS_LABEL:
            continue
        stats["refuted"] += 1
        fam["refuted"] += 1
        rows.append(
            {
                "family": family,
                "detail": detail,
                "assigned": rec["labels"],
                "claims_follows": True,
                "messages": rec["inputs"],
                "original": rec.get("original_messages", []),
                "original_label": rec.get("original_label", ""),
                "explanation": rec.get("generation_explanation", "") or "",
            }
        )
    rows.sort(key=lambda d: (d["family"], d["detail"]))
    return rows, stats


def _script_of(ch: str) -> str:
    name = unicodedata.name(ch, "")
    for s in ("CYRILLIC", "GREEK", "LATIN"):
        if s in name:
            return s
    return "OTHER"


def scan_invisible(text: str) -> tuple[list[str], list[str]]:
    """(invisible codepoint names, mixed-script words) present in ``text``."""
    found = sorted({INVISIBLE[c] for c in text if c in INVISIBLE})
    homoglyphs = []
    for word in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE):
        scripts = {_script_of(c) for c in word}
        if "LATIN" in scripts and ({"CYRILLIC", "GREEK"} & scripts):
            homoglyphs.append(word)
    return found, homoglyphs


def mark_invisible(text: str) -> str:
    """Replace invisible codepoints with a visible ⟨NAME⟩ marker."""
    return "".join(f"⟨{INVISIBLE[c]}⟩" if c in INVISIBLE else c for c in text)


def find_invisible(successes: list[dict]) -> list[dict]:
    out = []
    for s in successes:
        _, asst = parts(s["messages"])
        codepoints, homoglyphs = scan_invisible(asst)
        if not codepoints and not homoglyphs:
            continue
        out.append(
            {
                **s,
                "messages": [
                    {"role": m.get("role", ""), "content": mark_invisible(m.get("content", ""))}
                    for m in s["messages"]
                ],
                "codepoints": codepoints,
                "homoglyphs": homoglyphs,
            }
        )
    out.sort(key=lambda d: (d["iteration"], d["codepoints"]))
    return out


def split_examples(successes: list[dict], n: int) -> tuple[list[dict], list[dict]]:
    """Evenly spaced picks from each side, so the sample spans all iterations."""
    fmt, non = [], []
    for s in successes:
        user, _ = parts(s["messages"])
        (fmt if FORMAT.search(user) else non).append(s)

    def spread(rows: list[dict]) -> list[dict]:
        if len(rows) <= n:
            return rows
        step = len(rows) / n
        return [rows[int(i * step)] for i in range(n)]

    return spread(fmt), spread(non)


def collect(arm: str, probe_dir: Path, jsonl_base: Path, *, tau: float, cap: int, n_examples: int) -> dict:
    cache = load_cache(probe_dir / "contrastive_cache.jsonl")
    generated = {canon(r["inputs"]) for r in cache}

    iters = sorted(int(p.stem.rsplit("iter", 1)[1]) for p in probe_dir.glob("redteam_postprocessed_iter*.jsonl"))
    final = iters[-1]
    rows = load_postprocessed(probe_dir / f"redteam_postprocessed_iter{final}.jsonl")
    twins = find_twins(rows, generated, tau=tau, cap=cap)

    successes = load_successes(jsonl_base)
    fmt, non = split_examples(successes, n_examples)
    n_fmt = sum(1 for s in successes if FORMAT.search(parts(s["messages"])[0]))

    mislabels, mstats = find_mislabels(cache)

    return {
        "arm": arm,
        "probe_dir": str(probe_dir),
        "final_iter": final,
        "train_n": len(rows),
        "twins": twins,
        "format": fmt,
        "nonformat": non,
        "n_successes": len(successes),
        "n_format": n_fmt,
        "n_nonformat": len(successes) - n_fmt,
        "mislabels": mislabels,
        "mislabel_stats": mstats,
        "invisible": find_invisible(successes),
        # (iteration, n_successes) pairs, for the methodology tab's incidence table.
        "_succ_per_iter": sorted(
            (it, sum(1 for s in successes if s["iteration"] == it))
            for it in {s["iteration"] for s in successes}
        ),
    }


# ---------------------------------------------------------------------- render

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light dark;
  --bg:#fff; --panel:#f6f7f9; --panel2:#eef0f3; --line:#d9dde3;
  --fg:#1a1d21; --muted:#666e79; --accent:#3f6fd8;
  --pos-bg:#eaf5ee; --pos-fg:#1e6b3c; --neg-bg:#fdeeee; --neg-fg:#9d2727;
  --warn:#8a5b00; --warn-bg:#fdf3dc; --gen:#6a3fa0; --gen-bg:#f2ecfa;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
    --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
    --pos-bg:#17281d; --pos-fg:#86e0a5; --neg-bg:#2c1c1e; --neg-fg:#ff9a9a;
    --warn:#f0c060; --warn-bg:#33290f; --gen:#c3a2e8; --gen-bg:#241b33;
  }
}
:root[data-theme="dark"] {
  --bg:#14171a; --panel:#1c2126; --panel2:#232a31; --line:#333c45;
  --fg:#e5e9ee; --muted:#94a0ad; --accent:#7aa2f7;
  --pos-bg:#17281d; --pos-fg:#86e0a5; --neg-bg:#2c1c1e; --neg-fg:#ff9a9a;
  --warn:#f0c060; --warn-bg:#33290f; --gen:#c3a2e8; --gen-bg:#241b33;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:22px 18px 80px; }
h1 { margin:0 0 4px; font-size:21px; letter-spacing:-.01em; }
h2 { font-size:16px; margin:26px 0 8px; }
.sub { color:var(--muted); font-size:13px; }
.note { margin:16px 0 0; padding:11px 14px; border-left:3px solid var(--accent);
  background:var(--panel); border-radius:0 6px 6px 0; font-size:13.5px; color:var(--muted); }
.note b { color:var(--fg); }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
  background:var(--panel2); padding:1px 5px; border-radius:4px; }
.tabs { display:flex; flex-wrap:wrap; gap:6px; margin-top:20px; border-bottom:1px solid var(--line); }
.tab { appearance:none; border:1px solid transparent; border-bottom:none; background:transparent;
  color:var(--muted); cursor:pointer; padding:9px 15px; font-size:14px; font-weight:600;
  border-radius:7px 7px 0 0; margin-bottom:-1px; }
.tab:hover { color:var(--fg); background:var(--panel); }
.tab[aria-selected="true"] { color:var(--fg); background:var(--bg);
  border-color:var(--line); border-bottom:1px solid var(--bg); }
.tab .cnt { font-weight:400; color:var(--muted); font-size:12px; margin-left:5px; }
.stats { display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 6px; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:9px 13px; min-width:120px; }
.stat .k { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.stat .v { font-size:19px; font-weight:650; margin-top:2px; }
.toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:14px 0 18px; }
.toolbar input[type=search] { flex:1 1 240px; min-width:180px; padding:8px 11px; font-size:14px;
  background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:7px; }
.chip { appearance:none; cursor:pointer; padding:7px 12px; font-size:13px; font-weight:550;
  background:var(--panel); color:var(--muted); border:1px solid var(--line); border-radius:999px; }
.chip[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.count { color:var(--muted); font-size:13px; margin-left:auto; }
.card { border:1px solid var(--line); border-radius:10px; margin-bottom:14px; background:var(--panel); overflow:hidden; }
.card .head { display:flex; flex-wrap:wrap; gap:8px; align-items:center;
  padding:8px 13px; border-bottom:1px solid var(--line); background:var(--panel2); }
.badge { font-size:11px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
  padding:3px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }
.badge.pos { color:var(--pos-fg); background:var(--pos-bg); border-color:var(--pos-fg); }
.badge.neg { color:var(--neg-fg); background:var(--neg-bg); border-color:var(--neg-fg); }
.badge.gen { color:var(--gen); background:var(--gen-bg); border-color:var(--gen); }
.badge.warn { color:var(--warn); background:var(--warn-bg); border-color:var(--warn); }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; color:var(--muted); }
.cols { display:grid; grid-template-columns:1fr 1fr; }
.side { padding:12px 15px; background:var(--bg); }
.side + .side { border-left:1px solid var(--line); }
@media (max-width:820px) { .cols { grid-template-columns:1fr; }
  .side + .side { border-left:none; border-top:1px solid var(--line); } }
.side.pos { background:var(--pos-bg); } .side.neg { background:var(--neg-bg); }
.lbl { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin-bottom:7px; }
.side.pos .lbl { color:var(--pos-fg); } .side.neg .lbl { color:var(--neg-fg); }
.turn { margin-bottom:9px; } .turn:last-child { margin-bottom:0; }
.role { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); margin-bottom:3px; }
.text { white-space:pre-wrap; overflow-wrap:anywhere; font-size:14px; }
.body { padding:12px 15px; background:var(--bg); }
.foot { padding:9px 14px; border-top:1px solid var(--line); font-size:13px; color:var(--muted); }
.foot b { color:var(--fg); font-weight:600; }
.diff ins { background:var(--pos-bg); color:var(--pos-fg); text-decoration:none; padding:0 1px; border-radius:3px; }
.diff del { background:var(--neg-bg); color:var(--neg-fg); padding:0 1px; border-radius:3px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; margin:10px 0 4px; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; font-weight:700; }
td.num { font-family:ui-monospace,Menlo,monospace; }
.hidden { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div id="panel"></div>
</div>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const POS = "assistant_follows_the_instruction";
const esc = s => String(s == null ? '' : s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const shortLab = l => l === POS ? 'follows' : (l === 'positive' ? 'follows' : (l === 'negative' ? 'does-not-follow' : 'does-not-follow'));
const isPos = l => l === POS || l === 'positive';

function conv(messages, cls) {
  return `<div class="${cls||''}">` + messages.map(m =>
    `<div class="turn"><div class="role">${esc(m.role)}</div><div class="text">${esc(m.content)}</div></div>`
  ).join('') + `</div>`;
}

/* Word-level diff so the reader can see how small the difference actually is. */
function diffPair(a, b) {
  const A = a.split(/(\s+)/), B = b.split(/(\s+)/);
  const n = A.length, m = B.length;
  // LCS over tokens; conversations here are short enough for the full table.
  const dp = Array.from({length:n+1}, () => new Uint32Array(m+1));
  for (let i=n-1;i>=0;i--) for (let j=m-1;j>=0;j--)
    dp[i][j] = A[i]===B[j] ? dp[i+1][j+1]+1 : Math.max(dp[i+1][j], dp[i][j+1]);
  let i=0,j=0,out='';
  while (i<n && j<m) {
    if (A[i]===B[j]) { out += esc(A[i]); i++; j++; }
    else if (dp[i+1][j] >= dp[i][j+1]) { out += `<del>${esc(A[i])}</del>`; i++; }
    else { out += `<ins>${esc(B[j])}</ins>`; j++; }
  }
  while (i<n) { out += `<del>${esc(A[i])}</del>`; i++; }
  while (j<m) { out += `<ins>${esc(B[j])}</ins>`; j++; }
  return out;
}

const flat = ms => ms.map(m => m.role + ': ' + m.content).join('\n');

/* ------------------------------------------------------------- tab renderers */

function twinsTab(arms) {
  const rows = [];
  arms.forEach(a => a.twins.forEach(t => rows.push({arm:a.arm, ...t})));
  rows.sort((x,y) => y.ratio - x.ratio);
  return {
    rows,
    stats: arms.map(a => ({k:a.arm + ' pairs', v:a.twins.length}))
             .concat([{k:'shown', v:rows.length}]),
    search: r => r.arm + ' ' + flat(r.a) + ' ' + flat(r.b),
    render: r => {
      const aPos = isPos(r.a_label), bPos = isPos(r.b_label);
      const A = aPos ? r.a : r.b, B = aPos ? r.b : r.a;
      const aGen = aPos ? r.a_gen : r.b_gen, bGen = aPos ? r.b_gen : r.a_gen;
      return `<div class="card">
        <div class="head">
          <span class="badge">${esc(r.arm)}</span>
          <span class="badge warn">similarity ${r.ratio.toFixed(3)}</span>
          ${(r.codepoints||[]).map(c => `<span class="badge gen">${esc(c)}</span>`).join('')}
          <span class="mono">opposite labels</span>
        </div>
        <div class="cols">
          <div class="side pos"><div class="lbl">follows${aGen?' &middot; LLM-generated':' &middot; attacker-authored'}</div>${conv(A)}</div>
          <div class="side neg"><div class="lbl">does not follow${bGen?' &middot; LLM-generated':' &middot; attacker-authored'}</div>${conv(B)}</div>
        </div>
        <div class="foot diff"><b>Word diff</b> (<del>only in “does not follow”</del> / <ins>only in “follows”</ins>):
          <div style="margin-top:5px">${diffPair(flat(B), flat(A))}</div></div>
      </div>`;
    },
  };
}

function exampleTab(arms, key) {
  const rows = [];
  arms.forEach(a => a[key].forEach(e => rows.push({arm:a.arm, ...e})));
  return {
    rows,
    stats: arms.map(a => ({
      k: a.arm + (key === 'format' ? ' format' : ' non-format'),
      v: (key === 'format' ? a.n_format : a.n_nonformat) + ' / ' + a.n_successes,
    })),
    search: r => r.arm + ' ' + flat(r.messages) + ' ' + r.judge_reason,
    render: r => `<div class="card">
        <div class="head">
          <span class="badge">${esc(r.arm)}</span>
          <span class="badge">iter ${r.iteration}</span>
          <span class="badge ${r.error_type === 'false_positive' ? 'neg' : 'pos'}">${esc(r.error_type)}</span>
          <span class="mono">probe ${r.probe_score == null ? '—' : r.probe_score.toFixed(3)}</span>
          <span class="badge ${isPos(r.judge_label) ? 'pos' : 'neg'}">judge: ${esc(shortLab(r.judge_label))}</span>
          <span class="mono">conf ${r.judge_confidence}</span>
        </div>
        <div class="body">${conv(r.messages)}</div>
        <div class="foot"><b>Judge:</b> ${esc(r.judge_reason)}</div>
      </div>`,
  };
}

function mislabelTab(arms) {
  const rows = [];
  arms.forEach(a => a.mislabels.forEach(m => rows.push({arm:a.arm, ...m})));
  const st = [];
  arms.forEach(a => {
    const s = a.mislabel_stats;
    st.push({k:a.arm + ' refuted', v:`${s.refuted}/${s.checkable}`});
  });
  return {
    rows, stats: st,
    search: r => r.arm + ' ' + r.family + ' ' + flat(r.messages) + ' ' + r.explanation,
    render: r => `<div class="card">
        <div class="head">
          <span class="badge">${esc(r.arm)}</span>
          <span class="badge warn">${esc(r.family)}</span>
          <span class="badge gen">LLM-generated</span>
          <span class="badge ${r.claims_follows ? 'pos' : 'neg'}">labelled: ${esc(shortLab(r.assigned))}</span>
          <span class="mono">${esc(r.detail)}</span>
        </div>
        <div class="body">${conv(r.messages)}</div>
        <div class="foot"><b>Generator's own explanation:</b> ${esc(r.explanation)}</div>
        ${r.original.length ? `<div class="foot"><b>Rewritten from</b> (labelled ${esc(shortLab(r.original_label))}):
           <div style="margin-top:6px">${conv(r.original)}</div></div>` : ''}
      </div>`,
  };
}

function invisibleTab(arms) {
  const rows = [];
  arms.forEach(a => a.invisible.forEach(v => rows.push({arm:a.arm, ...v})));
  return {
    rows,
    stats: arms.map(a => ({k:a.arm, v:`${a.invisible.length} / ${a.n_successes}`})),
    search: r => r.arm + ' ' + r.codepoints.join(' ') + ' ' + flat(r.messages),
    render: r => `<div class="card">
        <div class="head">
          <span class="badge">${esc(r.arm)}</span>
          <span class="badge">iter ${r.iteration}</span>
          ${r.codepoints.map(c => `<span class="badge warn">${esc(c)}</span>`).join('')}
          ${r.homoglyphs.length ? `<span class="badge gen">homoglyph: ${esc(r.homoglyphs.join(', '))}</span>` : ''}
          <span class="mono">probe ${r.probe_score == null ? '—' : r.probe_score.toFixed(3)}</span>
          <span class="badge ${isPos(r.judge_label) ? 'pos' : 'neg'}">judge: ${esc(shortLab(r.judge_label))}</span>
        </div>
        <div class="body">${conv(r.messages)}</div>
        <div class="foot"><b>Judge:</b> ${esc(r.judge_reason)}</div>
      </div>`,
  };
}

function methodTab() {
  return {rows: null, html: DATA.method_html};
}

const TABS = [
  {id:'method',    name:'How to read this', build: methodTab},
  {id:'twins',     name:'Opposite-label twins', build: () => twinsTab(DATA.arms)},
  {id:'format',    name:'Format-constraint successes', build: () => exampleTab(DATA.arms, 'format')},
  {id:'nonformat', name:'Non-format successes', build: () => exampleTab(DATA.arms, 'nonformat')},
  {id:'mislabels', name:'Verified mislabels', build: () => mislabelTab(DATA.arms)},
  {id:'invisible', name:'Invisible-character attacks', build: () => invisibleTab(DATA.arms)},
];

let active = 'method', armFilter = null, query = '';

function draw() {
  const tabsEl = document.getElementById('tabs');
  const built = {};
  TABS.forEach(t => { built[t.id] = t.build(); });
  tabsEl.innerHTML = TABS.map(t => {
    const b = built[t.id];
    const n = b.rows ? b.rows.length : '';
    return `<button class="tab" role="tab" data-id="${t.id}" aria-selected="${t.id===active}">${t.name}${
      n !== '' ? `<span class="cnt">${n}</span>` : ''}</button>`;
  }).join('');
  tabsEl.querySelectorAll('.tab').forEach(b => b.onclick = () => {
    active = b.dataset.id; query = ''; draw();
  });

  const t = built[active];
  const panel = document.getElementById('panel');
  if (!t.rows) { panel.innerHTML = t.html; return; }

  const arms = [...new Set(t.rows.map(r => r.arm))];
  let rows = t.rows;
  if (armFilter) rows = rows.filter(r => r.arm === armFilter);
  if (query) {
    const q = query.toLowerCase();
    rows = rows.filter(r => t.search(r).toLowerCase().includes(q));
  }
  panel.innerHTML = `
    <div class="stats">${t.stats.map(s =>
      `<div class="stat"><div class="k">${esc(s.k)}</div><div class="v">${esc(s.v)}</div></div>`).join('')}</div>
    <div class="toolbar">
      <input type="search" id="q" placeholder="Filter text…" value="${esc(query)}">
      ${arms.map(a => `<button class="chip" data-arm="${esc(a)}" aria-pressed="${armFilter===a}">${esc(a)}</button>`).join('')}
      <span class="count">${rows.length} shown</span>
    </div>
    <div>${rows.map(t.render).join('') || '<div class="note">Nothing matches that filter.</div>'}</div>`;
  const q = document.getElementById('q');
  q.oninput = () => { query = q.value; const p = q.selectionStart; draw();
    const nq = document.getElementById('q'); nq.focus(); nq.setSelectionRange(p, p); };
  panel.querySelectorAll('.chip').forEach(c => c.onclick = () => {
    armFilter = armFilter === c.dataset.arm ? null : c.dataset.arm; draw();
  });
}
draw();
</script>
</body>
</html>
"""


def method_html(arms: list[dict], tau: float) -> str:
    def row(a: dict) -> str:
        s = a["mislabel_stats"]
        share = f"{s['checkable'] / s['pairs']:.0%}" if s["pairs"] else "—"
        pct = f"{s['refuted'] / s['checkable']:.0%}" if s["checkable"] else "—"
        return (
            f"<tr><td>{a['arm']}</td><td class='num'>{s['pairs']}</td>"
            f"<td class='num'>{s['checkable']} ({share})</td>"
            f"<td class='num'>{s['refuted']} ({pct})</td></tr>"
        )

    def fam_rows(a: dict) -> str:
        out = []
        for fam, d in sorted(a["mislabel_stats"]["by_family"].items(), key=lambda kv: -kv[1]["n"]):
            pct = f"{d['refuted'] / d['n']:.0%}" if d["n"] else "—"
            out.append(
                f"<tr><td>{a['arm']}</td><td><code>{fam}</code></td>"
                f"<td class='num'>{d['n']}</td><td class='num'>{d['refuted']} ({pct})</td></tr>"
            )
        return "".join(out)

    # Per-iteration incidence, because *when* these appear is the whole point: they
    # are the delta one retrain sees over the previous one.
    inv_rows = []
    for a in arms:
        per: dict[int, int] = {}
        for v in a["invisible"]:
            per[v["iteration"]] = per.get(v["iteration"], 0) + 1
        tot = {}
        for e in a["_succ_per_iter"]:
            tot[e[0]] = e[1]
        cells = "".join(
            f"<td class='num'>{per.get(i, 0)} / {tot.get(i, 0)}</td>" for i in sorted(tot)
        )
        inv_rows.append(f"<tr><td>{a['arm']}</td>{cells}</tr>")
    iters = sorted({i for a in arms for i in dict(a["_succ_per_iter"])})
    invisible_table = (
        "<table><tr><th>arm</th>"
        + "".join(f"<th>iteration {i}</th>" for i in iters)
        + "</tr>"
        + "".join(inv_rows)
        + "</table>"
    )

    return f"""
<div class="note">
This page is evidence for three claims about the <b>training data</b> the iterative
retrain loop builds — not about the probe. Every row is a real record from
<code>redteam_postprocessed_iter*.jsonl</code> or <code>contrastive_cache.jsonl</code>.
</div>

<h2>Opposite-label twins</h2>
<p>Pairs inside a single retrain's training set that are <b>&ge; {tau:.2f}</b> similar to
each other but carry <b>opposite</b> labels. Similarity is
<code>difflib.SequenceMatcher(autojunk=False)</code> over the whole rendered
conversation. Each card shows a word-level diff so you can see how small the
difference is — often one token (<code>even</code>/<code>odd</code>, a trailing period,
<code>2,3,5</code>/<code>2,3,4</code>).</p>
<p>These are mostly not <i>mislabels</i>: read strictly, each label is defensible. They are
<b>unlearnable</b> — the label turns on a semantic computation (is 7&times;3 really 21?
is that the 51st word?) that a linear read of one layer-32 activation cannot perform,
while the two inputs sit at nearly the same point in activation space.</p>

<h2>Format vs non-format successes</h2>
<p>Red-team successes split by whether the user turn states an explicit
<b>output-shape constraint</b> ("only", "exactly N words", "no extra text", JSON,
haiku, bullet counts). The complement is ordinary tasks where following the
instruction is a <b>semantic</b> property — which is what all seven
<code>eval_instructions</code> splits actually test (refusal, context drift,
contradiction of a source, omission, answer substitution).</p>
<p>The examples are spread evenly across each arm's iterations rather than taken from
the head of the file, so the sample isn't all iteration 0. The stat tiles give the
true population share behind the sample.</p>

<h2>Verified mislabels — and what the percentage means</h2>
<p>A generated pair is shown here only when its label says <b>"follows the
instruction"</b> and the prompt's own constraint, run as code, says it does not.
Ten families are adjudicated (word counts, letter bans, forbidden words, exact
literals, JSON-only, line/bullet counts, case, punctuation) — everything else (is this a
good summary? did it drift?) needs a judge and is skipped.</p>
<p><b>Read the percentage carefully.</b> It is a rate <i>within the adjudicable
subsample</i>, which is ~10&ndash;13% of pairs — <b>not</b> a corpus-wide mislabel rate.
The subsample is also not random: it over-represents <b>counting</b> constraints,
which is the single thing the generator (<code>openai/gpt-5.1</code>) is worst at.</p>
<table>
<tr><th>arm</th><th>generated pairs</th><th>adjudicable</th><th>label refuted</th></tr>
{"".join(row(a) for a in arms)}
</table>
<p>Per family, which is where the real signal is — the generator is <b>reliable on
structural constraints and unreliable on counting ones</b>:</p>
<table>
<tr><th>arm</th><th>family</th><th>adjudicable</th><th>refuted</th></tr>
{"".join(fam_rows(a) for a in arms)}
</table>
<p>The check is deliberately <b>one-sided</b>. A failed check is conclusive: the
response misses a constraint stated verbatim in the prompt, so "follows" is wrong. A
<i>passed</i> check is not conclusive — the prompt may carry clauses outside the
repertoire that the response failed, which is often exactly what a "does not follow"
label is recording. An earlier two-sided version flagged
<code>"Write exactly 7 words. The 3rd word must be 'blue'"</code> &rarr;
<code>"I like the color blue at night"</code> as mislabelled because the count was
right; the positional clause was what failed. That direction is now withheld.</p>
<p>Read it as: <i>of the pairs whose constraint a script can verify, this fraction carry a
label the constraint contradicts.</i> The subsample is not random — it is precisely the
<b>counting</b> constraints, which is the thing the generator (<code>openai/gpt-5.1</code>)
is worst at. So the percentage should not be extrapolated to the other ~90% of pairs;
structural constraints it can satisfy exactly (valid JSON, "output only X") show
essentially no mislabels. With denominators this small the percentage is also
high-variance.</p>
<p>What <i>is</i> robust is the <b>direction</b>: almost every mislabel claims
<b>"follows"</b>. Asked to write the compliant member of a pair, the generator produces a
near-miss (45&ndash;51 words when 50 was asked) and then asserts in its own
<code>generation_explanation</code> that the count is exact. Word counts here are taken
under the most lenient tokenizer that could make the label true, so every row shown is
wrong under <i>every</i> reading.</p>

<h2>Invisible-character attacks</h2>
<p>Successes whose assistant turn carries a zero-width codepoint (ZWSP, ZWNJ, BOM,
soft hyphen&hellip;) or a mixed-script <b>homoglyph</b> — a Cyrillic <code>л</code> inside
an otherwise Latin word. The attack works because the judge reads the raw string and
correctly rules "there is an extra character, so it did not follow <i>output only X</i>",
while a human sees a perfectly compliant reply and the probe sees an activation almost
identical to the compliant one.</p>
<p>These produce the most extreme opposite-label twins in the corpus: the two members are
<i>visually and semantically identical</i>. Codepoints are rendered as
<code>&#10216;ZWNJ&#10217;</code>-style markers here and in the twins tab, or the cards would
show two identical conversations with opposite labels and no visible reason.</p>
<p>Not every hit is an attack — NNBSP inside <code>225&#8239;g</code> is ordinary
typographic spacing, not adversarial. Each card lists the codepoints it found so the
distinction stays visible rather than being asserted.</p>
{invisible_table}
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", action="append", required=True, metavar="NAME:PROBE_DIR:JSONL_BASE",
                    help="Repeatable. JSONL_BASE is the run's jsonl path without the "
                         "_fp/_fn suffix, e.g. results_.../gptoss120b_probing")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=0.90,
                    help="Similarity threshold for opposite-label twins (default 0.90)")
    ap.add_argument("--max-twins", type=int, default=120,
                    help="Cap on twin pairs kept per arm, most similar first (default 120)")
    ap.add_argument("--n-examples", type=int, default=15,
                    help="Examples per arm in each of the format / non-format tabs")
    ap.add_argument("--title", default="Instruction-following arms — training-data pathologies")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        name, probe_dir, jsonl_base = spec.split(":", 2)
        arms.append(
            collect(name, Path(probe_dir), Path(jsonl_base),
                    tau=args.tau, cap=args.max_twins, n_examples=args.n_examples)
        )
        a = arms[-1]
        print(f"{name}: train(iter{a['final_iter']})={a['train_n']} twins={len(a['twins'])} "
              f"successes={a['n_successes']} (format {a['n_format']} / non-format {a['n_nonformat']}) "
              f"label-refuted={a['mislabel_stats']['refuted']}/{a['mislabel_stats']['checkable']}")

    subtitle = args.subtitle or (
        "Opposite-label twins, format vs non-format successes, and machine-verified "
        f"contrastive mislabels. Twin threshold {args.tau:.2f}."
    )
    payload = json.dumps(
        {"arms": arms, "method_html": method_html(arms, args.tau)}, ensure_ascii=False
    ).replace("</", r"<\/")
    html = (_TEMPLATE
            .replace("__TITLE__", args.title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__PAYLOAD__", payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
