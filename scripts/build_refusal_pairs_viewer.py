#!/usr/bin/env python
"""Build a self-contained HTML viewer of the REFUSAL-carrying contrastive pairs
that trained a given probe iteration.

Motivation: on the ``harmful_to_human`` concept the contrastive generator's idea
of "not harmful" is frequently "the assistant declines" — so a refusal-shaped
assistant turn enters the training set almost exclusively on the
``not_harmful_to_human`` side. That is a shortcut the probe can learn instead of
the concept, and it is invisible in the raw success logs because the refusal is
written by ``generate_contrastive_dataset``, not by the attacker. This viewer
surfaces every such pair with BOTH sides shown, so the refusal can be read
against the conversation it was generated from.

A "refusal" is a match of ``_REFUSAL_RE`` anywhere in a conversation's
**assistant** turns. A pair is included when *either* side matches; the side(s)
that matched are labelled, and the matched phrases are highlighted inline.

Pairing is taken from ``contrastive_cache.jsonl``, whose records carry
``original_messages`` / ``original_label`` alongside the generated ``inputs`` /
``labels`` — the authoritative link. (Positional alignment of the postprocessed
dump is wrong from iteration 2 on, since ``generate_contrastive_dataset``
returns cache hits before newly generated rows.)

Membership in ``redteam_postprocessed_iter{N}.jsonl`` is tested on the
**transformed** conversation: ``_apply_message_transforms`` (convert tool ->
assistant, then combine consecutive messages) runs *after* contrastive
generation, so the dumped rows can differ from the cache's raw messages. A pair
counts as present in iteration N only when both of its sides are.

Red-team metadata for the original side (probe score, judge label/confidence/
reason, error type, attacker model, round) is joined from the run's
``*_probing_{fp,fn}.jsonl`` when a matching success is found. Generated
conversations have no such row, and neither do originals whose sibling arm
produced them, so the field is optional.

The page has two tabs: the pairs themselves, and an **AUROC** tab plotting each
run's eval AUROC per split across iterations (from the run's comparison CSV,
supplied with ``--comparison``). Chart design follows the dataviz skill — one
categorical hue per split, the mean drawn as neutral emphasis rather than a 5th
slot, direct labels + a table view (the light-mode relief rule), and a
crosshair/tooltip hover layer. The palette is the reference instance's slots 1-4,
validated on the adjacent pairlist in both modes.

Usage:
    .venv_claude/bin/python scripts/build_refusal_pairs_viewer.py \
        --arm "gpt-oss-120b:probes/hu_harm_gemma27b_gptoss120b_batch:results_hu_harm_gemma27b_gptoss120b_batch/gptoss120b_probing" \
        --arm "deepseek-v4-pro:probes/hu_harm_gemma27b_deepseekv4pro_batch:results_hu_harm_gemma27b_deepseekv4pro_batch/deepseekv4pro_probing" \
        --comparison "gpt-oss-120b:results_hu_harm_gemma27b_gptoss120b_batch/gptoss120b_comparison.csv" \
        --comparison "deepseek-v4-pro:results_hu_harm_gemma27b_deepseekv4pro_batch/deepseekv4pro_comparison.csv" \
        --iteration 1 --iteration 2 --iteration 3 \
        --git-ref experiment11_cloud \
        --title "harmful_to_human — refusal-carrying contrastive pairs" \
        --out viewers/hu_harm_refusal_pairs.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# Refusal markers, matched against assistant turns only. Deliberately broader
# than an "explicit decline" list: the point is to surface candidates for
# reading, and every match is shown highlighted so a false positive is obvious
# on sight rather than silently counted.
#
# `_APOS` is load-bearing: these conversations are LLM-written and use the
# TYPOGRAPHIC apostrophe (U+2019) about 20x more often than the ASCII one, so a
# pattern spelled with a plain `'` misses most of "I'm sorry" / "I can't" /
# "I won't" and silently under-reports the refusal rate.
_APOS = r"['‘’ʼ]?"
_REFUSAL_RE = re.compile(
    rf"(I{_APOS}m sorry|I am sorry|I apologi[sz]e|I can{_APOS}t|I cannot|I won{_APOS}t|"
    rf"I{_APOS}m not able|I am not able|I{_APOS}m unable|I am unable|I must decline|"
    rf"I have to decline|not something I can|I{_APOS}d rather not|"
    rf"can{_APOS}t help with|cannot help with)",
    re.I,
)


# --------------------------------------------------------------------------- io


def _read_text(path: str, git_ref: str | None) -> str | None:
    """Read a file, either from the worktree or from a git ref."""
    if git_ref:
        proc = subprocess.run(
            ["git", "show", f"{git_ref}:{path}"], capture_output=True, text=True
        )
        return proc.stdout if proc.returncode == 0 else None
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None


def _read_jsonl(path: str, git_ref: str | None) -> list[dict]:
    text = _read_text(path, git_ref)
    if text is None:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_comparison_csv(path: str, git_ref: str | None) -> dict[str, Any] | None:
    """Parse an iterative-retrain comparison CSV into chart-ready series.

    Rows are ``round,dataset,auroc,accuracy,tpr_at_fpr,fpr`` where ``round`` is
    ``iter{N}`` — the eval of ``probe_iter{N}.pkl``, so ``iter0`` is the baseline
    probe and iteration N's retrain produces ``iter{N}``. The ``mean`` dataset is
    kept separate from the per-split series: it is an aggregate, and giving it a
    categorical hue would imply it is another component.
    """
    import csv as _csv

    text = _read_text(path, git_ref)
    if text is None:
        return None
    rows = list(_csv.DictReader(text.splitlines()))
    if not rows:
        return None

    def _key(r: str) -> int:
        m = re.search(r"(\d+)", r)
        return int(m.group(1)) if m else 0

    iters = sorted({r["round"] for r in rows}, key=_key)
    splits = [d for d in dict.fromkeys(r["dataset"] for r in rows) if d != "mean"]
    table = {(r["round"], r["dataset"]): r for r in rows}

    def _series(dataset: str) -> list[float | None]:
        out: list[float | None] = []
        for it in iters:
            r = table.get((it, dataset))
            try:
                out.append(float(r["auroc"]) if r else None)
            except (TypeError, ValueError):
                out.append(None)
        return out

    return {
        "iters": iters,
        "splits": splits,
        "series": {d: _series(d) for d in splits},
        "mean": _series("mean"),
    }


# ------------------------------------------------------------------- conversation


def _messages(raw: Any) -> list[dict[str, str]]:
    """Normalize a conversation to ``[{role, content}, ...]``.

    Handles the three shapes in play: a plain list, the JSON-encoded string the
    eval splits use, and the ``{"messages": [...]}`` wrapper of an AttemptRecord
    sample.
    """
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        raw = raw.get("messages", [])
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in raw]


def _transform(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Apply the same loader transforms ``retrain`` applies before dumping.

    Order matches ``LabelledDataset.load_from``: tool -> assistant first, then
    combine consecutive same-role messages.
    """
    from tuberlens.interfaces.dataset import LabelledDataset, Message

    dialogue = [Message(role=m["role"], content=m["content"]) for m in messages]
    dialogue = LabelledDataset._convert_tool_to_assistant(dialogue)
    dialogue = LabelledDataset._combine_consecutive_messages(dialogue)
    return [{"role": m.role, "content": m.content} for m in dialogue]


def _canon(messages: Sequence[dict[str, str]]) -> str:
    return json.dumps(list(messages), sort_keys=True)


def _assistant_text(messages: Sequence[dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "assistant")


def _refusal_phrases(messages: Sequence[dict[str, str]]) -> list[str]:
    """Distinct refusal phrases found in the assistant turns, in order."""
    seen: dict[str, None] = {}
    for match in _REFUSAL_RE.findall(_assistant_text(messages)):
        seen.setdefault(match.strip(), None)
    return list(seen)


# ----------------------------------------------------------------- data assembly


def _load_arm(
    label: str, probe_dir: str, probing_stem: str, iterations: Sequence[int],
    git_ref: str | None,
) -> tuple[list[dict], dict[str, Any]]:
    """Return this arm's refusal pairs plus a stats dict for the header."""
    cache = _read_jsonl(f"{probe_dir}/contrastive_cache.jsonl", git_ref)
    if not cache:
        print(f"  [{label}] WARNING: no contrastive cache at {probe_dir}", file=sys.stderr)

    # Membership sets, keyed on the transformed conversation.
    members: dict[int, set[str]] = {}
    for n in iterations:
        rows = _read_jsonl(f"{probe_dir}/redteam_postprocessed_iter{n}.jsonl", git_ref)
        members[n] = {_canon(_transform(_messages(r["inputs"]))) for r in rows}
        if not rows:
            print(f"  [{label}] WARNING: no postprocessed_iter{n}.jsonl", file=sys.stderr)

    # Red-team metadata for the original side, keyed on the raw conversation.
    meta: dict[str, dict] = {}
    for et in ("fp", "fn"):
        for r in _read_jsonl(f"{probing_stem}_{et}.jsonl", git_ref):
            if not r.get("success"):
                continue
            meta[_canon(_messages(r["sample"]))] = {
                "probe_score": r.get("probe_score"),
                "judge_label": r.get("judge_label", ""),
                "judge_confidence": r.get("judge_confidence"),
                "judge_reason": r.get("judge_reason", ""),
                "error_type": r.get("error_type", ""),
                "iteration": r.get("iteration"),
                "round": r.get("round"),
                "attacker_model": r.get("attacker_model", ""),
            }

    pairs: list[dict] = []
    n_pairs_in_scope = 0
    for entry in cache:
        rec = entry.get("record", {})
        orig = _messages(rec.get("original_messages", []))
        gen = _messages(rec.get("inputs", []))
        if not orig or not gen:
            continue

        present = {
            n: (_canon(_transform(orig)) in members[n] and _canon(_transform(gen)) in members[n])
            for n in iterations
        }
        if not any(present.values()):
            continue
        n_pairs_in_scope += 1

        ref_orig = _refusal_phrases(orig)
        ref_gen = _refusal_phrases(gen)
        if not (ref_orig or ref_gen):
            continue

        side = "both" if (ref_orig and ref_gen) else ("original" if ref_orig else "contrastive")
        pairs.append(
            {
                "arm": label,
                "side": side,
                "present": {str(n): present[n] for n in iterations},
                "original": {
                    "messages": orig,
                    "label": rec.get("original_label", ""),
                    "phrases": ref_orig,
                    "meta": meta.get(_canon(orig)),
                },
                "contrastive": {
                    "messages": gen,
                    "label": rec.get("labels", ""),
                    "phrases": ref_gen,
                    "explanation": rec.get("generation_explanation", ""),
                    "model": rec.get("generation_model", ""),
                },
            }
        )

    stats = {
        "arm": label,
        "cache_pairs": len(cache),
        "pairs_in_scope": n_pairs_in_scope,
        "refusal_pairs": len(pairs),
        "per_iteration": {
            str(n): sum(1 for p in pairs if p["present"][str(n)]) for n in iterations
        },
        "scope_per_iteration": {str(n): len(members[n]) // 2 for n in iterations},
    }
    return pairs, stats


# ------------------------------------------------------------------------ render


def _highlight(text: str) -> str:
    """Wrap refusal phrases in <mark>, escaping each segment as it is emitted.

    Matching happens on the RAW text and escaping is applied per segment —
    escaping first would rewrite ``'`` to ``&#x27;`` and stop every
    apostrophe-bearing marker from ever matching.
    """
    out: list[str] = []
    pos = 0
    for m in _REFUSAL_RE.finditer(text):
        out.append(html.escape(text[pos : m.start()]))
        out.append(f"<mark>{html.escape(m.group(0))}</mark>")
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _render_messages(messages: Iterable[dict[str, str]], highlight_assistant: bool) -> str:
    out = []
    for m in messages:
        role = m["role"]
        body = (
            _highlight(m["content"])
            if (highlight_assistant and role == "assistant")
            else html.escape(m["content"])
        )
        out.append(
            f'<div class="msg role-{html.escape(role)}">'
            f'<span class="role">{html.escape(role)}</span>'
            f'<div class="content">{body}</div></div>'
        )
    return "".join(out)


def _render_meta(meta: dict | None) -> str:
    if not meta:
        return '<div class="meta none">no matching red-team success row</div>'
    score = meta.get("probe_score")
    bits = []
    if score is not None:
        bits.append(f'<span class="chip">probe {score:.3f}</span>')
    if meta.get("judge_label"):
        bits.append(f'<span class="chip">judge: {html.escape(meta["judge_label"])}</span>')
    if meta.get("judge_confidence"):
        bits.append(f'<span class="chip">conf {meta["judge_confidence"]}</span>')
    if meta.get("error_type"):
        bits.append(f'<span class="chip">{html.escape(meta["error_type"])}</span>')
    if meta.get("iteration") is not None:
        bits.append(f'<span class="chip">rt-iter {meta["iteration"]}</span>')
    if meta.get("attacker_model"):
        bits.append(f'<span class="chip">{html.escape(meta["attacker_model"])}</span>')
    reason = meta.get("judge_reason") or ""
    detail = (
        f'<details><summary>judge reasoning</summary><p>{html.escape(reason)}</p></details>'
        if reason
        else ""
    )
    return f'<div class="meta">{"".join(bits)}{detail}</div>'


def _render_card(pair: dict, idx: int, iterations: Sequence[int]) -> str:
    o, c = pair["original"], pair["contrastive"]
    present_chips = "".join(
        f'<span class="chip iter {"on" if pair["present"][str(n)] else "off"}">'
        f'iter{n} {"✓" if pair["present"][str(n)] else "✗"}</span>'
        for n in iterations
    )
    side_note = {
        "original": "refusal on the ORIGINAL (attacker-written) side",
        "contrastive": "refusal on the GENERATED contrastive side",
        "both": "refusal on BOTH sides",
    }[pair["side"]]
    expl = (
        f'<details><summary>generation rationale</summary><p>{html.escape(c["explanation"])}</p></details>'
        if c.get("explanation")
        else ""
    )
    phrases = sorted(set(o["phrases"]) | set(c["phrases"]))
    phrase_chips = "".join(f'<span class="chip phrase">{html.escape(p)}</span>' for p in phrases)

    return f"""
<article class="card" data-arm="{html.escape(pair['arm'])}" data-side="{pair['side']}"
         data-iters="{','.join(str(n) for n in iterations if pair['present'][str(n)])}">
  <header class="card-head">
    <span class="idx">#{idx}</span>
    <span class="chip arm">{html.escape(pair['arm'])}</span>
    {present_chips}
    <span class="chip side side-{pair['side']}">{html.escape(side_note)}</span>
    {phrase_chips}
  </header>
  <div class="cols">
    <section class="col original">
      <h3>Original <span class="lbl lbl-{'pos' if 'not_' not in o['label'] else 'neg'}">{html.escape(o['label'])}</span></h3>
      {_render_meta(o['meta'])}
      <div class="convo">{_render_messages(o['messages'], True)}</div>
    </section>
    <section class="col contrastive">
      <h3>Generated contrastive <span class="lbl lbl-{'pos' if 'not_' not in c['label'] else 'neg'}">{html.escape(c['label'])}</span></h3>
      <div class="meta"><span class="chip">{html.escape(c.get('model', '') or 'generator')}</span>{expl}</div>
      <div class="convo">{_render_messages(c['messages'], True)}</div>
    </section>
  </div>
</article>"""


_CSS = """
:root{--bg:#fbfbfd;--fg:#16181d;--muted:#666e7a;--line:#e2e5ea;--card:#fff;
  --pos:#b4341f;--neg:#1f6f4a;--mark:#ffe8a3;--chip:#eef1f5;--accent:#2f5fd0;}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--fg:#e6e8ec;--muted:#98a1ad;
  --line:#2a2f38;--card:#181b21;--pos:#ff8f77;--neg:#63d29d;--mark:#6b5416;--chip:#232830;--accent:#7ba1ff;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
header.page{padding:28px 32px 16px;border-bottom:1px solid var(--line)}
h1{margin:0 0 6px;font-size:21px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--muted);max-width:80ch;margin:0 0 4px}
main{padding:20px 32px 64px;max-width:1600px}
table.stats{border-collapse:collapse;margin:14px 0 4px;font-size:13px}
table.stats th,table.stats td{border:1px solid var(--line);padding:5px 10px;text-align:right}
table.stats th{background:var(--chip);font-weight:600}
table.stats td:first-child,table.stats th:first-child{text-align:left}
.filters{position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0;
  border-bottom:1px solid var(--line);display:flex;gap:18px;flex-wrap:wrap;align-items:center}
.filters label{font-size:12px;color:var(--muted);display:flex;gap:6px;align-items:center}
select{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;
  background:var(--card);color:var(--fg)}
.count{margin-left:auto;color:var(--muted);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  margin:16px 0;overflow:hidden}
.card-head{display:flex;gap:8px;flex-wrap:wrap;align-items:center;
  padding:10px 14px;border-bottom:1px solid var(--line);background:var(--chip)}
.idx{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px;font-weight:600}
.chip{font-size:11px;padding:2px 8px;border-radius:20px;background:var(--bg);
  border:1px solid var(--line);color:var(--muted);white-space:nowrap}
.chip.arm{font-weight:600;color:var(--fg)}
.chip.iter.on{color:var(--accent);border-color:var(--accent)}
.chip.iter.off{opacity:.5;text-decoration:line-through}
.chip.side{font-weight:600;color:var(--fg)}
.chip.phrase{background:var(--mark);color:#3b2f05;border-color:transparent}
@media (prefers-color-scheme:dark){.chip.phrase{color:#ffe8a3}}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:0}
@media (max-width:1000px){.cols{grid-template-columns:1fr}}
.col{padding:14px 16px}
.col.original{border-right:1px solid var(--line)}
@media (max-width:1000px){.col.original{border-right:0;border-bottom:1px solid var(--line)}}
.col h3{margin:0 0 8px;font-size:13px;font-weight:650;text-transform:uppercase;
  letter-spacing:.04em;color:var(--muted);display:flex;gap:8px;align-items:center}
.lbl{font-size:11px;padding:2px 8px;border-radius:20px;text-transform:none;letter-spacing:0}
.lbl-pos{background:color-mix(in srgb,var(--pos) 16%,transparent);color:var(--pos)}
.lbl-neg{background:color-mix(in srgb,var(--neg) 16%,transparent);color:var(--neg)}
.meta{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
.meta.none{color:var(--muted);font-size:12px;font-style:italic}
details{width:100%;font-size:12px;color:var(--muted)}
summary{cursor:pointer;padding:2px 0}
details p{margin:6px 0 0;padding:8px 10px;background:var(--bg);border-radius:6px;
  border:1px solid var(--line);white-space:pre-wrap}
.msg{margin:0 0 10px;padding-left:10px;border-left:2px solid var(--line)}
.msg .role{display:inline-block;font-size:10px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);font-weight:700;margin-bottom:3px}
.role-assistant{border-left-color:var(--accent)}
.msg .content{white-space:pre-wrap;word-break:break-word}
mark{background:var(--mark);color:inherit;padding:0 2px;border-radius:3px}
.empty{color:var(--muted);padding:40px 0;font-style:italic}

/* ---- tabs ---- */
nav.tabs{display:flex;gap:4px;padding:12px 32px 0;border-bottom:1px solid var(--line)}
nav.tabs button{font:inherit;font-size:13px;font-weight:600;padding:8px 16px;border:0;
  background:transparent;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent}
nav.tabs button:hover{color:var(--fg)}
nav.tabs button.active{color:var(--fg);border-bottom-color:var(--accent)}
[hidden]{display:none !important}

/* ---- charts: roles from the dataviz reference palette (validated, adjacent
       pairlist, both modes). Light mode puts slots 3-4 under 3:1 vs the chart
       surface, so the relief rule applies — direct labels and the table view
       below are mandatory, not optional. ---- */
.viz-root{color-scheme:light;
  --surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e6e6e2;
  --series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  color-scheme:dark;
  --surface-1:#1a1a19;--text-primary:#ffffff;--text-secondary:#c3c2b7;--grid:#33332f;
  --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
  --surface-1:#1a1a19;--text-primary:#ffffff;--text-secondary:#c3c2b7;--grid:#33332f;
  --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px;margin:18px 0 8px}
.panel{background:var(--surface-1);border:1px solid var(--line);border-radius:10px;padding:14px 16px 8px;position:relative}
.panel h3{margin:0 0 2px;font-size:13px;font-weight:650;color:var(--text-primary)}
.panel .cap{margin:0 0 8px;font-size:11.5px;color:var(--text-secondary)}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 6px}
.legend span{display:flex;gap:5px;align-items:center;font-size:11px;color:var(--text-secondary)}
.legend i{width:12px;height:2.5px;border-radius:2px;display:inline-block}
.legend i.dash{background:repeating-linear-gradient(90deg,currentColor 0 4px,transparent 4px 7px)}
.panel svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--line);
  border-radius:8px;padding:8px 10px;font-size:11.5px;box-shadow:0 4px 14px rgba(0,0,0,.14);
  z-index:9;min-width:170px;opacity:0;transition:opacity .08s}
.tip.on{opacity:1}
.tip b{display:block;margin-bottom:4px;font-size:11px;color:var(--text-secondary);font-weight:600}
.tip .row{display:flex;align-items:center;gap:6px;justify-content:space-between}
.tip .row span{display:flex;align-items:center;gap:5px;color:var(--text-primary)}
.tip .row i{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}
.tip .row b.v{margin:0;font-variant-numeric:tabular-nums;color:var(--text-primary);font-weight:600}
table.data{border-collapse:collapse;font-size:12px;margin:10px 0 30px;width:100%}
table.data th,table.data td{border:1px solid var(--line);padding:4px 9px;text-align:right;
  font-variant-numeric:tabular-nums}
table.data th{background:var(--chip);font-weight:600}
table.data td:first-child,table.data th:first-child{text-align:left}
table.data tr.mean td{font-weight:650;background:var(--chip)}
table.data caption{caption-side:top;text-align:left;font-weight:650;padding:10px 0 6px;font-size:13px}
.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:baseline}
"""

_JS = """
const ITERS=__ITERS__;
const cards=[...document.querySelectorAll('.card')];
const sel={arm:'all',iter:'any',side:'all'};
function iterMatch(iters,want){
  if(want==='any')return iters.length>0;
  if(want==='all')return iters.length===ITERS.length;          // present in every file
  if(want.startsWith('only'))return iters.length===1&&iters[0]===want.slice(4);
  return iters.includes(want);
}
function apply(){
  let n=0;
  for(const c of cards){
    const iters=(c.dataset.iters||'').split(',').filter(Boolean);
    const ok=(sel.arm==='all'||c.dataset.arm===sel.arm)
          && (sel.side==='all'||c.dataset.side===sel.side)
          && iterMatch(iters,sel.iter);
    c.style.display=ok?'':'none';
    if(ok)n++;
  }
  document.getElementById('count').textContent=n+' of '+cards.length+' pairs shown';
  document.getElementById('empty').style.display=n?'none':'';
}
for(const id of ['arm','iter','side']){
  document.getElementById('f-'+id).addEventListener('change',e=>{sel[id]=e.target.value;apply();});
}
apply();

/* ---------------------------------------------------------------- tabs */
const tabs=[...document.querySelectorAll('nav.tabs button')];
tabs.forEach(b=>b.addEventListener('click',()=>{
  tabs.forEach(x=>x.classList.toggle('active',x===b));
  document.querySelectorAll('main > section').forEach(s=>{s.hidden=(s.id!=='tab-'+b.dataset.tab);});
  if(b.dataset.tab==='auroc')drawAll();
}));

/* ------------------------------------------------------- AUROC line charts
   Form: trend over time, distinct components -> multi-line, categorical hue per
   split. The mean is drawn as EMPHASIS (neutral ink, dashed) rather than a 5th
   categorical slot: it is an aggregate of the other four, not a peer component. */
const NS='http://www.w3.org/2000/svg';
const W=560,H=330,M={t:14,r:104,b:36,l:46};
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function fmt(v){return v==null?'—':v.toFixed(3);}

function draw(panel){
  const spec=CHARTS[panel.dataset.chart];
  const host=panel.querySelector('.plot');
  host.textContent='';
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':spec.title});
  const iw=W-M.l-M.r, ih=H-M.t-M.b;
  const n=spec.iters.length;
  const x=i=>M.l+(n===1?iw/2:iw*i/(n-1));
  const y=v=>M.t+ih*(1-(v-spec.ymin)/(spec.ymax-spec.ymin));

  // recessive grid + y ticks
  for(const t of spec.ticks){
    svg.appendChild(el('line',{x1:M.l,x2:M.l+iw,y1:y(t),y2:y(t),stroke:'var(--grid)','stroke-width':1}));
    const lb=el('text',{x:M.l-8,y:y(t)+3.5,'text-anchor':'end',fill:'var(--text-secondary)','font-size':10.5});
    lb.textContent=t.toFixed(2); svg.appendChild(lb);
  }
  spec.iters.forEach((it,i)=>{
    const lb=el('text',{x:x(i),y:M.t+ih+16,'text-anchor':'middle',fill:'var(--text-secondary)','font-size':11});
    lb.textContent=it; svg.appendChild(lb);
  });

  // series
  const labels=[];
  spec.series.forEach(s=>{
    const pts=s.values.map((v,i)=>v==null?null:[x(i),y(v)]).filter(Boolean);
    if(!pts.length)return;
    svg.appendChild(el('path',{d:'M'+pts.map(p=>p.join(' ')).join('L'),fill:'none',stroke:s.color,
      'stroke-width':s.emphasis?2.5:2,'stroke-linejoin':'round','stroke-linecap':'round',
      ...(s.emphasis?{'stroke-dasharray':'6 4'}:{})}));
    pts.forEach(p=>{                       // 2px surface ring keeps overlaps readable
      svg.appendChild(el('circle',{cx:p[0],cy:p[1],r:4,fill:s.color,stroke:'var(--surface-1)','stroke-width':2}));
    });
    const last=s.values.reduce((a,v,i)=>v==null?a:i,-1);
    if(last>=0)labels.push({y:y(s.values[last]),y0:y(s.values[last]),x0:x(last),
                            text:s.short,color:s.color,emphasis:s.emphasis});
  });

  // direct labels (mandatory at 4 series, and by the relief rule in light mode):
  // de-overlap greedily so no two collide.
  labels.sort((a,b)=>a.y-b.y);
  const GAP=13;
  for(let i=1;i<labels.length;i++)
    if(labels[i].y-labels[i-1].y<GAP)labels[i].y=labels[i-1].y+GAP;
  const over=labels.length?labels[labels.length-1].y-(M.t+ih):0;
  if(over>0)labels.forEach(l=>l.y-=over);
  labels.forEach(l=>{
    // a displaced label needs a leader back to its own endpoint, or three stacked
    // labels stop saying which line each belongs to
    if(Math.abs(l.y-l.y0)>2){
      svg.appendChild(el('polyline',{points:`${l.x0+5},${l.y0} ${M.l+iw+3},${l.y0} ${M.l+iw+6},${l.y}`,
        fill:'none',stroke:l.color,'stroke-width':1,opacity:.55}));
    }
    const t=el('text',{x:M.l+iw+9,y:l.y+3.5,fill:'var(--text-primary)','font-size':10.5,
      'font-weight':l.emphasis?700:600});
    t.textContent=l.text; svg.appendChild(t);
  });

  // hover: crosshair + one tooltip listing every series at that iteration
  const cross=el('line',{x1:0,x2:0,y1:M.t,y2:M.t+ih,stroke:'var(--text-secondary)',
    'stroke-width':1,'stroke-dasharray':'3 3',opacity:0});
  svg.appendChild(cross);
  const hit=el('rect',{x:M.l-14,y:M.t,width:iw+28,height:ih,fill:'transparent'});
  svg.appendChild(hit);
  const tip=panel.querySelector('.tip');
  hit.addEventListener('mousemove',ev=>{
    const r=svg.getBoundingClientRect(), sx=(ev.clientX-r.left)*W/r.width;
    let i=Math.round((sx-M.l)/(n===1?1:iw/(n-1))); i=Math.max(0,Math.min(n-1,i));
    cross.setAttribute('x1',x(i));cross.setAttribute('x2',x(i));cross.setAttribute('opacity',1);
    tip.innerHTML='<b>'+spec.iters[i]+'</b>'+spec.series.map(s=>
      `<div class="row"><span><i style="background:${s.color}"></i>${s.short}</span><b class="v">${fmt(s.values[i])}</b></div>`).join('');
    tip.classList.add('on');
    const px=r.width*(x(i)/W);
    tip.style.left=Math.min(Math.max(px-85,4),r.width-186)+'px';
    tip.style.top=(ev.clientY-r.top+14)+'px';
  });
  hit.addEventListener('mouseleave',()=>{cross.setAttribute('opacity',0);tip.classList.remove('on');});
  host.appendChild(svg);
}
let drawn=false;
function drawAll(){ if(drawn)return; document.querySelectorAll('.panel[data-chart]').forEach(draw); drawn=true; }
"""


_SERIES_VARS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"]
_SERIES_HEX = {  # for the static table swatches, which sit outside the SVG
    "var(--series-1)": ("#2a78d6", "#3987e5"),
    "var(--series-2)": ("#eb6834", "#d95926"),
    "var(--series-3)": ("#1baf7a", "#199e70"),
    "var(--series-4)": ("#eda100", "#c98500"),
}


def _short_split(name: str) -> str:
    """Compact split name for direct labels (they live in a 104px gutter)."""
    return re.sub(r"^eval[_-]", "", name)


def _y_scale(values: Iterable[float]) -> tuple[float, float, list[float]]:
    vals = [v for v in values if v is not None]
    lo = min(vals + [1.0])
    ymin = max(0.0, math.floor(lo * 20 - 1) / 20)  # nearest 0.05 below
    ymax = 1.0
    step = 0.05 if (ymax - ymin) <= 0.35 else 0.1
    ticks, t = [], ymin
    while t <= ymax + 1e-9:
        ticks.append(round(t, 2))
        t += step
    return ymin, ymax, ticks


def _chart_specs(runs: list[dict]) -> dict[str, dict]:
    """One chart per run (splits + mean), plus one cross-run mean comparison."""
    every = [v for r in runs for s in r["data"]["series"].values() for v in s]
    every += [v for r in runs for v in r["data"]["mean"]]
    ymin, ymax, ticks = _y_scale(every)

    specs: dict[str, dict] = {}
    for r in runs:
        d = r["data"]
        series = [
            {
                "short": _short_split(name),
                "color": _SERIES_VARS[i % len(_SERIES_VARS)],
                "values": d["series"][name],
                "emphasis": False,
            }
            for i, name in enumerate(d["splits"])
        ]
        series.append(
            {"short": "mean", "color": "var(--text-primary)", "values": d["mean"], "emphasis": True}
        )
        specs[r["key"]] = {
            "title": f"{r['label']} — AUROC per eval split",
            "iters": d["iters"], "series": series,
            "ymin": ymin, "ymax": ymax, "ticks": ticks,
        }

    if len(runs) > 1:
        specs["__mean__"] = {
            "title": "Mean AUROC — run comparison",
            "iters": runs[0]["data"]["iters"],
            "series": [
                {"short": r["label"], "color": _SERIES_VARS[i % len(_SERIES_VARS)],
                 "values": r["data"]["mean"], "emphasis": False}
                for i, r in enumerate(runs)
            ],
            "ymin": ymin, "ymax": ymax, "ticks": ticks,
        }
    return specs


def _render_panel(key: str, spec: dict, caption: str) -> str:
    legend = "".join(
        f'<span><i class="{"dash" if s["emphasis"] else ""}" '
        f'style="{"color" if s["emphasis"] else "background"}:{s["color"]}"></i>'
        f'{html.escape(s["short"])}</span>'
        for s in spec["series"]
    )
    return f"""
<div class="panel" data-chart="{html.escape(key)}">
  <h3>{html.escape(spec['title'])}</h3>
  <p class="cap">{html.escape(caption)}</p>
  <div class="legend">{legend}</div>
  <div class="plot"></div>
  <div class="tip"></div>
</div>"""


def _render_data_table(run: dict) -> str:
    d = run["data"]
    head = "".join(f"<th>{html.escape(i)}</th>" for i in d["iters"])
    body = ""
    for i, name in enumerate(d["splits"]):
        light, dark = _SERIES_HEX[_SERIES_VARS[i % len(_SERIES_VARS)]]
        cells = "".join(f"<td>{'' if v is None else f'{v:.4f}'}</td>" for v in d["series"][name])
        body += (
            f'<tr><td><span class="swatch" style="background:{light}"></span>'
            f'{html.escape(name)}</td>{cells}</tr>'
        )
    cells = "".join(f"<td>{'' if v is None else f'{v:.4f}'}</td>" for v in d["mean"])
    body += f'<tr class="mean"><td>mean</td>{cells}</tr>'
    return f"""
<table class="data">
  <caption>{html.escape(run['label'])} — AUROC (table view of the chart above)</caption>
  <tr><th>eval split</th>{head}</tr>
  {body}
</table>"""


def _render_auroc_tab(runs: list[dict]) -> tuple[str, dict[str, dict]]:
    if not runs:
        return (
            '<p class="empty">No comparison CSV supplied — pass --comparison LABEL:PATH '
            "to enable this tab.</p>",
            {},
        )
    specs = _chart_specs(runs)
    caption = (
        "iterN = eval of probe_iterN.pkl; iter0 is the baseline probe, so each later "
        "point is one retrain on that iteration's red-team successes."
    )
    panels = "".join(_render_panel(r["key"], specs[r["key"]], caption) for r in runs)
    if "__mean__" in specs:
        panels += _render_panel(
            "__mean__", specs["__mean__"],
            "The mean row of each run's comparison CSV, drawn on one axis for direct comparison.",
        )
    tables = "".join(_render_data_table(r) for r in runs)
    return f'<div class="charts">{panels}</div>{tables}', specs


def _render_page(
    pairs: list[dict], stats: list[dict], iterations: Sequence[int], title: str, subtitle: str,
    runs: list[dict],
) -> str:
    arms = [s["arm"] for s in stats]
    arm_opts = "".join(f'<option value="{html.escape(a)}">{html.escape(a)}</option>' for a in arms)
    # "all" = present in every iteration file; "onlyN" = present in that one alone.
    iter_opts = "".join(f'<option value="{n}">iter{n}</option>' for n in iterations)
    iter_opts += (
        f'<option value="all">all {len(iterations)} ('
        + " ∩ ".join(f"iter{n}" for n in iterations)
        + ")</option>"
    )
    iter_opts += "".join(f'<option value="only{n}">iter{n} only</option>' for n in iterations)

    head = "".join(f"<th>iter{n} refusal pairs</th>" for n in iterations)
    head += "".join(f"<th>iter{n} total pairs</th>" for n in iterations)
    body = ""
    for s in stats:
        cells = "".join(f'<td>{s["per_iteration"][str(n)]}</td>' for n in iterations)
        cells += "".join(f'<td>{s["scope_per_iteration"][str(n)]}</td>' for n in iterations)
        body += (
            f'<tr><td>{html.escape(s["arm"])}</td><td>{s["cache_pairs"]}</td>'
            f'<td>{s["refusal_pairs"]}</td>{cells}</tr>'
        )

    cards = "".join(_render_card(p, i + 1, iterations) for i, p in enumerate(pairs))
    auroc_html, specs = _render_auroc_tab(runs)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style></head>
<body class="viz-root">
<header class="page">
  <h1>{html.escape(title)}</h1>
  <p class="sub">{html.escape(subtitle)}</p>
  <p class="sub">A pair is listed when a refusal phrase appears in the <strong>assistant</strong>
  turns of either side. Both sides are always shown. Matched phrases are highlighted.
  <code>iterN ✓</code> means both sides of the pair are present in
  <code>redteam_postprocessed_iterN.jsonl</code>, i.e. the pair trained <code>probe_iterN.pkl</code>.</p>
  <table class="stats">
    <tr><th>arm</th><th>cache pairs</th><th>refusal pairs</th>{head}</tr>
    {body}
  </table>
</header>
<nav class="tabs">
  <button data-tab="pairs" class="active">Refusal pairs</button>
  <button data-tab="auroc">AUROC</button>
</nav>
<main>
<section id="tab-pairs">
  <div class="filters">
    <label>arm <select id="f-arm"><option value="all">all</option>{arm_opts}</select></label>
    <label>present in <select id="f-iter"><option value="any">any iteration</option>{iter_opts}</select></label>
    <label>refusal side <select id="f-side">
      <option value="all">all</option>
      <option value="contrastive">generated contrastive</option>
      <option value="original">original</option>
      <option value="both">both</option>
    </select></label>
    <span class="count" id="count"></span>
  </div>
  <p class="empty" id="empty" style="display:none">No pairs match these filters.</p>
  {cards}
</section>
<section id="tab-auroc" hidden>{auroc_html}</section>
</main>
<script>
const CHARTS={json.dumps(specs)};
{_JS.replace("__ITERS__", json.dumps([str(n) for n in iterations]))}
</script>
</body></html>"""


# -------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--arm", action="append", required=True,
        help="LABEL:PROBE_DIR:PROBING_STEM (repeatable). PROBING_STEM is the JSONL "
             "path without the _fp/_fn suffix.",
    )
    ap.add_argument("--iteration", type=int, action="append", required=True,
                    help="Postprocessed iteration to include (repeatable)")
    ap.add_argument("--git-ref", default=None,
                    help="Read inputs from this git ref instead of the worktree")
    ap.add_argument("--comparison", action="append", default=[],
                    help="LABEL:PATH of a run's eval comparison CSV (repeatable). Each "
                         "becomes a panel in the AUROC tab; omit to drop that tab.")
    ap.add_argument("--out", type=Path, required=True, help="Output HTML path")
    ap.add_argument("--title", default="Refusal-carrying contrastive pairs")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    iterations = sorted(set(args.iteration))
    all_pairs: list[dict] = []
    all_stats: list[dict] = []
    for spec in args.arm:
        parts = spec.split(":")
        if len(parts) != 3:
            ap.error(f"--arm must be LABEL:PROBE_DIR:PROBING_STEM, got {spec!r}")
        label, probe_dir, probing_stem = parts
        pairs, stats = _load_arm(label, probe_dir, probing_stem, iterations, args.git_ref)
        all_pairs.extend(pairs)
        all_stats.append(stats)
        print(
            f"  [{label}] {stats['refusal_pairs']} refusal pairs of "
            f"{stats['pairs_in_scope']} in scope ({stats['cache_pairs']} cached)"
        )

    runs: list[dict] = []
    for spec in args.comparison:
        label, _, path = spec.partition(":")
        if not path:
            ap.error(f"--comparison must be LABEL:PATH, got {spec!r}")
        data = _read_comparison_csv(path, args.git_ref)
        if data is None:
            print(f"  [{label}] WARNING: no comparison CSV at {path}", file=sys.stderr)
            continue
        runs.append({"key": f"run{len(runs)}", "label": label, "data": data})
        print(f"  [{label}] AUROC: {len(data['splits'])} splits x {len(data['iters'])} iterations")

    # Contrastive-side refusals first — that is the asymmetry worth reading.
    order = {"contrastive": 0, "both": 1, "original": 2}
    all_pairs.sort(key=lambda p: (order[p["side"]], p["arm"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        _render_page(all_pairs, all_stats, iterations, args.title, args.subtitle, runs),
        encoding="utf-8",
    )
    print(f"Wrote {args.out} ({len(all_pairs)} pairs, {args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
