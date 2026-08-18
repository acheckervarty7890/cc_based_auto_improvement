#!/usr/bin/env python
"""Build a self-contained HTML viewer of the EXACT prompts behind every submission.

Reads the verbatim per-turn capture written by ``PromptTraceStore``
(``<jsonl>.prompts.jsonl``, enabled by ``attacker.capture_prompts``) and joins it to the
attempt log, then answers: *what was the attacker looking at when it produced each of
these conversations?*

**Every** captured submission is included. They are grouped into tabs by clustering on
the first user turn (single-link, difflib ratio over the opening ``--opener-chars``
characters), largest cluster first, with all size-1 clusters collected into a final
"unclustered" tab. ``--cluster-threshold`` is sensitive — single-link chains, so a value
slightly too low merges every template into one tab; sweep it if the tab bar degenerates. Tab labels are taken from each cluster's medoid opener, so the tab bar
doubles as a summary of which templates the run actually produced.

Each submission renders as: the exact message array sent to the model on that turn
(system prompt, and every prior user/assistant turn of that session, verbatim), the raw
reply, the parsed submission, and the probe/judge verdict. The message array is ground
truth, not a reconstruction — which matters because the attempt JSONL records no session
or turn identifier, so with ``sessions_per_model > 1`` the in-session context of any
submission after a session's first is otherwise unrecoverable after the fact.

    scripts/build_prompt_trace_viewer.py \
        --prompts results_memotest_noview/memotest_probing.prompts.jsonl \
        --out viewers/memotest_prompt_trace_viewer.html
"""

from __future__ import annotations

import argparse
import difflib
import html
import itertools
import json
import re
from pathlib import Path

DEFAULT_CLUSTER_THRESHOLD = 0.50
DEFAULT_OPENER_CHARS = 120

WORD = re.compile(r"\S+")


# --------------------------------------------------------------------------- #
# Loading + clustering
# --------------------------------------------------------------------------- #


def _first_user(messages: list[dict]) -> str:
    return next((m.get("content", "") for m in messages if m.get("role") == "user"), "")


def load_memos(summaries_path: Path | None) -> list[dict]:
    """Rolling round memos from ``<jsonl>.summaries.jsonl``, oldest first.

    The store holds ONE rolling memo that ``update()`` replaces each round, so the memo
    written after round R is the memo — and the only one — injected into round R+1's
    attacker system prompts. It is not cumulative and it is not shown to any later round.
    The final round is never summarized (nothing would consume it), so a 3-round rotation
    leaves 2 memos.
    """
    if summaries_path is None or not summaries_path.exists():
        return []
    memos = [
        json.loads(line)
        for line in summaries_path.read_text().splitlines()
        if line.strip()
    ]
    memos.sort(key=lambda m: (m.get("iteration", 0), m.get("round", 0)))
    return memos


def _explode_batches(rows: list[dict]) -> list[dict]:
    """Split batch-mode rows into one row per submission.

    Under ``attacker.batch_submissions`` a single API call produces several
    conversations, so ``PromptTraceStore`` writes them as one row carrying plural
    ``submissions``/``results`` rather than repeating the (large) prompt once per
    conversation. Everything downstream here — clustering, rendering, the per-row
    verdict — is written against one submission per row, so undo that here.

    The copies share a prompt: they carry the same ``turn`` and the same ``messages``,
    and are distinguished by ``batch_index``. Rows without ``submissions`` (per-turn
    mode, and every capture written before batch mode existed) pass through untouched.
    """
    out: list[dict] = []
    for r in rows:
        subs = r.get("submissions")
        if not subs:
            out.append(r)
            continue
        keys = r.get("submission_keys") or []
        results = r.get("results") or []
        for i, sub in enumerate(subs):
            copy = dict(r)
            copy.pop("submissions", None)
            copy.pop("submission_keys", None)
            copy.pop("results", None)
            copy["submission"] = sub
            copy["submission_key"] = keys[i] if i < len(keys) else ""
            copy["result"] = results[i] if i < len(results) else None
            copy["batch_index"] = i
            copy["batch_size"] = len(subs)
            out.append(copy)
    return out


def load_rows(prompts_path: Path, attempts_path: Path | None) -> list[dict]:
    """Trace rows that produced a parsed submission, enriched from the attempt log."""
    rows = [
        json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()
    ]
    rows = _explode_batches(rows)
    rows = [r for r in rows if r.get("parsed") and r.get("submission")]

    by_key: dict[str, dict] = {}
    if attempts_path and attempts_path.exists():
        from agentic_redteam.persistence import AttemptRecord

        for line in attempts_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = AttemptRecord.from_jsonl_row(line)
            by_key[rec.sample.to_canonical_text()] = {
                "judge_reason": rec.judge_reason,
                "judge_confidence": rec.judge_confidence,
                "probe_score": rec.probe_score,
                "judge_label": rec.judge_label,
                "probe_label": rec.probe_label,
                "success": rec.success,
            }
    for r in rows:
        r["attempt"] = by_key.get(r.get("submission_key", ""), {})
        r["_opener"] = _first_user(r["submission"])
    # Stable, readable order within a cluster.
    rows.sort(key=lambda r: (r["round"], r["session_id"], r["turn"], r.get("batch_index", 0)))
    return rows


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def cluster(rows: list[dict], threshold: float, opener_chars: int) -> list[list[int]]:
    """Single-link clustering of rows on their submission's opening user text."""
    openers = [r["_opener"][:opener_chars] for r in rows]
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(rows)), 2):
        if _sim(openers[i], openers[j]) >= threshold:
            parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def medoid(rows: list[dict], idxs: list[int], opener_chars: int) -> int:
    """The member most similar to all the others — the cluster's representative."""
    if len(idxs) == 1:
        return idxs[0]
    openers = {i: rows[i]["_opener"][:opener_chars] for i in idxs}
    return max(
        idxs,
        key=lambda i: sum(_sim(openers[i], openers[j]) for j in idxs if j != i) / (len(idxs) - 1),
    )


def mean_sim(rows: list[dict], idxs: list[int], opener_chars: int) -> float:
    openers = [rows[i]["_opener"][:opener_chars] for i in idxs]
    pairs = [_sim(a, b) for a, b in itertools.combinations(openers, 2)]
    return sum(pairs) / len(pairs) if pairs else float("nan")


def label_for(rows: list[dict], idxs: list[int], opener_chars: int, words: int = 6) -> str:
    """Short tab label from the medoid's opening words."""
    text = rows[medoid(rows, idxs, opener_chars)]["_opener"]
    toks = WORD.findall(text)[:words]
    return " ".join(toks) + ("…" if len(WORD.findall(text)) > words else "")


def build_tabs(rows: list[dict], args) -> list[dict]:
    """Cluster into tabs: multi-member clusters largest-first, then the singletons."""
    groups = cluster(rows, args.cluster_threshold, args.opener_chars)
    multi = [g for g in groups if len(g) > 1]
    singles = [i for g in groups if len(g) == 1 for i in g]

    tabs = []
    for n, g in enumerate(multi, 1):
        tabs.append({
            "id": f"c{n}",
            "label": f"{n}. {label_for(rows, g, args.opener_chars)}",
            "count": len(g),
            "idxs": g,
            "kind": "cluster",
        })
    if singles:
        tabs.append({
            "id": "singletons",
            "label": "unclustered",
            "count": len(singles),
            "idxs": sorted(singles),
            "kind": "singletons",
        })
    return tabs


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--card:#fafafa;--accent:#1d4ed8;
--sys:#6b46c1;--usr:#1d4ed8;--asst:#047857;--warn:#b45309;--ok:#047857;--bad:#b91c1c}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--muted:#9aa0a6;
--line:#2c2f36;--card:#1d2026;--accent:#93c5fd;--sys:#c4b5fd;--usr:#93c5fd;--asst:#6ee7b7;
--warn:#fcd34d;--ok:#6ee7b7;--bad:#fca5a5}}
:root[data-theme=dark]{--bg:#16181c;--fg:#e6e6e6;--muted:#9aa0a6;--line:#2c2f36;--card:#1d2026;
--accent:#93c5fd;--sys:#c4b5fd;--usr:#93c5fd;--asst:#6ee7b7;--warn:#fcd34d;--ok:#6ee7b7;--bad:#fca5a5}
:root[data-theme=light]{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--card:#fafafa;
--accent:#1d4ed8;--sys:#6b46c1;--usr:#1d4ed8;--asst:#047857;--warn:#b45309;--ok:#047857;--bad:#b91c1c}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,sans-serif;margin:0;padding:24px;max-width:1180px;margin-inline:auto}
h1{font-size:22px;margin:0 0 6px}
.sub{color:var(--muted);font-size:13px;margin-bottom:16px}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
padding:12px 14px;border-radius:6px;margin:14px 0;font-size:13.5px}
table.stats{border-collapse:collapse;font-size:13px;margin:10px 0 20px;width:100%}
table.stats td,table.stats th{border:1px solid var(--line);padding:5px 10px;text-align:left}
table.stats th{background:var(--card);white-space:nowrap}
.tabwrap{overflow-x:auto;border-bottom:2px solid var(--line);margin:18px 0 0}
.tabbar{display:flex;gap:2px;min-width:max-content}
.tabbar button{font:13px inherit;padding:8px 13px;border:1px solid var(--line);
border-bottom:none;border-radius:7px 7px 0 0;background:var(--card);color:var(--muted);
cursor:pointer;white-space:nowrap}
.tabbar button[aria-selected=true]{background:var(--bg);color:var(--fg);font-weight:600;
box-shadow:inset 0 3px 0 var(--accent)}
.tabbar .n{color:var(--muted);font-weight:400}
.panel{display:none;padding-top:14px}.panel.on{display:block}
details.item{border:1px solid var(--line);border-radius:8px;margin:11px 0;background:var(--card)}
details.item>summary{cursor:pointer;padding:11px 14px;font-size:14px;list-style:none}
details.item>summary::-webkit-details-marker{display:none}
details.item>summary::before{content:"\\25B8 ";color:var(--muted)}
details.item[open]>summary::before{content:"\\25BE "}
.body{padding:4px 14px 14px}
.badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;
border:1px solid var(--line);margin-right:6px;color:var(--muted)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.msg{border:1px solid var(--line);border-radius:6px;margin:7px 0;overflow:hidden;background:var(--bg)}
.msg .role{font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.06em;
text-transform:uppercase;padding:6px 10px;border-bottom:1px solid var(--line)}
.role-system{color:var(--sys)}.role-user{color:var(--usr)}.role-assistant{color:var(--asst)}
pre{margin:0;padding:10px;white-space:pre-wrap;word-break:break-word;
font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-x:auto}
.scroll{max-height:340px;overflow-y:auto}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
margin:18px 0 6px;font-weight:600}
.verdict{font-size:13px;background:var(--bg);border:1px solid var(--line);
border-radius:6px;padding:9px 11px;margin-top:6px}
.tools{margin:12px 0}
.tools button{font:13px inherit;padding:5px 11px;margin-right:7px;border:1px solid var(--line);
border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
.medoid{border-left:3px solid var(--accent)}
"""

JS = """
const bar=document.querySelector('.tabbar');
bar.addEventListener('click',e=>{
  const b=e.target.closest('button[data-tab]'); if(!b) return;
  bar.querySelectorAll('button').forEach(x=>x.setAttribute('aria-selected',x===b));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id===b.dataset.tab));
  window.scrollTo({top:0,behavior:'instant'});
});
document.querySelectorAll('[data-toggle]').forEach(b=>b.addEventListener('click',()=>{
  const open=b.dataset.toggle==='open';
  document.querySelectorAll('.panel.on details.item').forEach(d=>d.open=open);
}));
"""


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def render_messages(messages: list[dict], scroll_system: bool) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        cls = f"role-{role}" if role in ("system", "user", "assistant") else ""
        pre_cls = "scroll" if scroll_system and role in ("system", "user") else ""
        out.append(
            f'<div class="msg"><div class="role {cls}">{esc(role)}'
            f' &middot; {len(content):,} chars</div>'
            f'<pre class="{pre_cls}">{esc(content)}</pre></div>'
        )
    return "\n".join(out)


def render_item(rows: list[dict], i: int, n: int, is_medoid: bool) -> str:
    r = rows[i]
    a = r.get("attempt", {})
    res = r.get("result") or {}
    sub = r["submission"]
    success = a.get("success", res.get("success"))
    score = a.get("probe_score", res.get("probe_score"))

    head = (
        f'<span class="badge">#{n}</span>'
        f'<span class="badge">round {r["round"]}</span>'
        f'<span class="badge">{esc(r["session_id"])}</span>'
        f'<span class="badge">turn {r["turn"]}</span>'
        + (
            f'<span class="badge">batch {r["batch_index"] + 1}/{r["batch_size"]}</span>'
            if "batch_index" in r
            else ""
        )
        + ('<span class="badge">medoid</span>' if is_medoid else "")
        + f'<span class="{"ok" if success else "bad"}">{"SUCCESS" if success else "fail"}</span>'
        + (f' <span class="badge">probe {score:.3f}</span>' if isinstance(score, (int, float)) else "")
        + f' &nbsp; {esc(r["_opener"][:92])}&hellip;'
    )

    verdict = (
        f'<div class="verdict"><b>probe:</b> {esc(a.get("probe_label", "?"))}'
        + (f" (score {score:.3f})" if isinstance(score, (int, float)) else "")
        + f' &nbsp;|&nbsp; <b>judge:</b> {esc(a.get("judge_label", "?"))}'
        + (f' (confidence {a["judge_confidence"]})' if a.get("judge_confidence") else "")
        + (f'<br><b>judge reason:</b> {esc(a.get("judge_reason"))}' if a.get("judge_reason") else "")
        + "</div>"
    )

    if "batch_index" in r:
        # Batch mode: one prompt produced the whole batch, and no verdict was ever fed
        # back — so the prompt below is the entire context for every member of it.
        turn_note = (
            f"Batch submission {r['batch_index'] + 1} of {r['batch_size']}, all written "
            "from the one prompt below in a single reply. The attacker saw no probe or "
            "judge verdict for any of them."
            + (
                ""
                if r["turn"] == 0
                else f" (Top-up call {r['turn'] + 1}: the prompt already contains this "
                "session's earlier batch, but not how it scored.)"
            )
        )
    else:
        turn_note = (
            "Session's FIRST turn — the prompt below is the entire context the model had."
            if r["turn"] == 0
            else f"Turn {r['turn']} of this session: the prompt below already contains this "
            f"session's own {r['turn']} earlier submission(s) and the verdicts they got back."
        )

    return f"""<details class="item{' medoid' if is_medoid else ''}">
<summary>{head}</summary>
<div class="body">
  <div class="note">{turn_note}</div>
  <h3>Exact prompt sent to the attacker &mdash; {len(r['messages'])} messages</h3>
  {render_messages(r['messages'], scroll_system=True)}
  <h3>Raw reply</h3>
  <div class="msg"><div class="role role-assistant">assistant &middot; raw</div>
  <pre class="scroll">{esc(r['response_text'])}</pre></div>
  <h3>Parsed submission</h3>
  {render_messages(sub, scroll_system=False)}
  <h3>Verdict</h3>
  {verdict}
</div>
</details>"""


def render_memo_panel(memos: list[dict], rows: list[dict]) -> str:
    """The judge's rolling round memos, in the order the attacker met them."""
    rounds = sorted({r["round"] for r in rows})
    last_round = rounds[-1] if rounds else None
    n_by_round = {rd: sum(1 for r in rows if r["round"] == rd) for rd in rounds}

    blocks = []
    for m in memos:
        rd = m.get("round")
        seen_by = rd + 1 if rd is not None else None
        words = len(m.get("text", "").split())
        complete = m.get("text", "").rstrip().endswith((".", "!", "?", '"', "”", ")"))
        where = (
            f"injected into <b>round {seen_by}</b>'s system prompts"
            f" ({n_by_round.get(seen_by, 0)} submissions)"
            if seen_by is not None and seen_by in n_by_round
            else "written but never shown (no later round ran)"
        )
        blocks.append(
            f'<details class="item" open><summary>'
            f'<span class="badge">after round {rd}</span>'
            f'<span class="badge">{words} words</span>'
            f'<span class="badge">{m.get("n_successes", "?")}/{m.get("n_attempts", "?")} succeeded</span>'
            f'<span class="{"ok" if complete else "bad"}">'
            f'{"complete" if complete else "TRUNCATED"}</span>'
            f'</summary><div class="body">'
            f'<div class="note">{where}. Written by the judge from that round\'s '
            f'{m.get("n_attempts", "?")} attempts plus the previous memo, which it '
            f'rewrites rather than appends to.</div>'
            f'<pre>{esc(m.get("text", ""))}</pre>'
            f'</div></details>'
        )

    intro = (
        "<p class='sub'>The judge's <b>rolling strategy memo</b> — the only cross-attempt "
        "signal in a <code>view_limit: 0</code> run. The store holds ONE memo that each "
        "round's update <em>replaces</em>, so the memo written after round R is seen by "
        "round R+1 and by no later round. The final round is never summarized, which is "
        f"why {len(memos)} memo(s) cover {len(rounds)} rounds.</p>"
        "<div class='note'>These are also visible verbatim inside the system prompt of "
        "every submission in the cluster tabs — this tab just puts them side by side so "
        "the memo can be read against what the next round actually produced.</div>"
    )
    return intro + "\n".join(blocks)


def tab_stats(rows: list[dict], idxs: list[int], opener_chars: int) -> str:
    n = len(idxs)
    succ = sum(1 for i in idxs if (rows[i].get("attempt", {}) or rows[i].get("result") or {}).get("success"))
    first = sum(1 for i in idxs if rows[i]["turn"] == 0)
    sessions = {rows[i]["session_id"] for i in idxs}
    rounds = sorted({rows[i]["round"] for i in idxs})
    ms = mean_sim(rows, idxs, opener_chars)
    ms_txt = f"{ms:.3f}" if ms == ms else "n/a (single member)"
    return f"""<table class="stats">
<tr><th>submissions</th><td>{n}</td>
    <th>successes</th><td>{succ} ({100 * succ / n:.0f}%)</td></tr>
<tr><th>mean pairwise opener similarity</th><td>{ms_txt}</td>
    <th>written on a session's first turn</th><td>{first} of {n}</td></tr>
<tr><th>distinct sessions</th><td>{len(sessions)}</td>
    <th>rounds</th><td>{', '.join(str(r) for r in rounds)}</td></tr>
</table>"""


def build_html(rows: list[dict], tabs: list[dict], args, src: Path,
               memos: list[dict] | None = None) -> str:
    total = len(rows)
    memos = memos or []
    biggest = max((t["count"] for t in tabs if t["kind"] == "cluster"), default=0)
    n_sessions = len({r["session_id"] for r in rows})
    overall = mean_sim(rows, list(range(total)), args.opener_chars)

    summary_rows = "\n".join(
        f'<tr><td><b>{esc(t["label"])}</b></td><td>{t["count"]}</td>'
        f'<td>{100 * t["count"] / total:.0f}%</td>'
        f'<td>{mean_sim(rows, t["idxs"], args.opener_chars):.3f}</td></tr>'
        if t["count"] > 1 else
        f'<tr><td><b>{esc(t["label"])}</b></td><td>{t["count"]}</td>'
        f'<td>{100 * t["count"] / total:.0f}%</td><td>&mdash;</td></tr>'
        for t in tabs
    )

    # The memo tab leads: it is the run's cross-round context, and the cluster tabs read
    # differently once you have seen what the judge told the next round to do. Cluster
    # numbering in the summary table above is unaffected — that table lists `tabs`.
    all_tabs = tabs
    if memos:
        all_tabs = [{
            "id": "tab-memos",
            "label": "Judge memos",
            "count": len(memos),
            "kind": "memos",
            "idxs": [],
        }] + tabs

    tabbar = "\n".join(
        f'<button data-tab="{t["id"]}" aria-selected="{"true" if n == 0 else "false"}">'
        f'{esc(t["label"])} <span class="n">({t["count"]})</span></button>'
        for n, t in enumerate(all_tabs)
    )

    panels = []
    for n, t in enumerate(all_tabs):
        if t["kind"] == "memos":
            panels.append(
                f'<div class="panel{" on" if n == 0 else ""}" id="{t["id"]}">'
                f'{render_memo_panel(memos, rows)}</div>'
            )
            continue
        med = medoid(rows, t["idxs"], args.opener_chars) if t["kind"] == "cluster" else -1
        shown = t["idxs"] if args.per_tab <= 0 else t["idxs"][: args.per_tab]
        cap = (
            f'<div class="note">Showing {len(shown)} of {t["count"]} '
            f'(--per-tab {args.per_tab}).</div>' if len(shown) < t["count"] else ""
        )
        head = (
            "<p class='sub'>Every submission whose opener did not cluster with any other. "
            "These are the run's genuinely distinct attempts.</p>"
            if t["kind"] == "singletons" else ""
        )
        items = "\n".join(
            render_item(rows, i, k, i == med) for k, i in enumerate(shown, 1)
        )
        panels.append(
            f'<div class="panel{" on" if n == 0 else ""}" id="{t["id"]}">'
            f'{head}{tab_stats(rows, t["idxs"], args.opener_chars)}{cap}'
            f'<div class="tools"><button data-toggle="open">expand all</button>'
            f'<button data-toggle="closed">collapse all</button></div>'
            f'{items}</div>'
        )

    return f"""<title>Attacker prompt trace — every submission by cluster</title>
<style>{CSS}</style>
<h1>Attacker prompt trace</h1>
<div class="sub">Source: <code>{esc(src)}</code> &middot; {total:,} submissions across
{n_sessions} sessions &middot; clustered at difflib &ge; {args.cluster_threshold}
on the first {args.opener_chars} opener chars</div>

<div class="note"><b>These prompts are verbatim, not reconstructed.</b> Each is the exact
message array sent to the model on that turn, captured by <code>PromptTraceStore</code> at
call time. That matters here: the attempt JSONL records no session or turn identifier, and
with <code>sessions_per_model &gt; 1</code> the concurrent sessions interleave in one file,
so the in-session context of any submission after a session's first cannot be recovered
after the fact.</div>

<table class="stats">
<tr><th>total submissions</th><td>{total}</td>
    <th>mean pairwise opener similarity (all)</th><td>{overall:.3f}</td></tr>
<tr><th>clusters (size &gt; 1)</th><td>{sum(1 for t in tabs if t["kind"] == "cluster")}</td>
    <th>largest cluster</th><td>{biggest} of {total} ({100 * biggest / total:.0f}%)</td></tr>
</table>

<table class="stats">
<tr><th>cluster</th><th>n</th><th>share</th><th>mean opener similarity</th></tr>
{summary_rows}
</table>

<div class="tabwrap"><div class="tabbar">{tabbar}</div></div>
{''.join(panels)}
<script>{JS}</script>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--prompts", type=Path, required=True, help="<jsonl>.prompts.jsonl")
    p.add_argument("--attempts", type=Path, default=None,
                   help="attempt JSONL for judge reasons (default: derived from --prompts)")
    p.add_argument("--summaries", type=Path, default=None,
                   help="rolling round-memo sidecar <jsonl>.summaries.jsonl for the "
                        "'Judge memos' tab (default: derived from --prompts; the tab is "
                        "omitted when the file is absent)")
    p.add_argument("--out", type=Path, default=Path("viewers/prompt_trace_viewer.html"))
    p.add_argument("--per-tab", type=int, default=0,
                   help="cap submissions rendered per tab (0 = all, the default)")
    p.add_argument("--cluster-threshold", type=float, default=DEFAULT_CLUSTER_THRESHOLD,
                   help="difflib ratio at which two openers join a cluster (default 0.50). "
                        "Sensitive: linkage is single-link, so a threshold slightly too low "
                        "chains every template into one blob (on arm C, 0.45 gave one cluster "
                        "of 45/50, 0.50 gave 5 clusters of 17/10/6/2/2). Sweep it if the tab "
                        "bar looks degenerate.")
    p.add_argument("--opener-chars", type=int, default=DEFAULT_OPENER_CHARS)
    args = p.parse_args(argv)

    attempts = args.attempts
    if attempts is None:
        attempts = args.prompts.with_name(
            args.prompts.name.replace(".prompts.jsonl", ".jsonl")
        )
    summaries = args.summaries
    if summaries is None:
        summaries = args.prompts.with_name(
            args.prompts.name.replace(".prompts.jsonl", ".summaries.jsonl")
        )

    if not args.prompts.exists():
        print(f"No prompt capture at {args.prompts}.")
        print("Run the arm with `capture_prompts: true` under `attacker:` first.")
        return 1

    rows = load_rows(args.prompts, attempts)
    if not rows:
        print(f"{args.prompts} has no parsed submissions.")
        return 1

    memos = load_memos(summaries)
    tabs = build_tabs(rows, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_html(rows, tabs, args, args.prompts, memos), encoding="utf-8"
    )

    print(f"{len(rows)} submissions across {len({r['session_id'] for r in rows})} sessions")
    if memos:
        detail = ", ".join(
            "after r{}={}w".format(m.get("round"), len(m.get("text", "").split()))
            for m in memos
        )
        print(f"  judge memos: {len(memos)} ({detail})")
    else:
        print(f"  judge memos: none found at {summaries} — tab omitted")
    for t in tabs:
        ms = mean_sim(rows, t["idxs"], args.opener_chars)
        ms_txt = f"{ms:.3f}" if ms == ms else "  -  "
        print(f"  {t['count']:>4}  sim={ms_txt}  {t['label']}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
