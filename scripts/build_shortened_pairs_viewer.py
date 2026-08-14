#!/usr/bin/env python
"""Build a self-contained HTML viewer for the refusal-related shortened pairs.

``scripts/shorten_long_contrastive_pairs.py`` re-generated every contrastive pair
that overran tuberlens' 1024-token activation window. Shortening is not neutral:
under a word budget the cheapest way to write a "not_harmful_to_human" assistant
turn is a **refusal**, so some pairs that used to answer with a safe substitute
(often a short code block) now decline instead. That is a distribution shift in
the negative class, and it has to be eyeballed rather than trusted to a summary
statistic — hence this viewer.

It reads, per arm:

* ``probes/<arm>/shortened_pairs_iter3.jsonl``      — the run report (before/after tokens)
* ``probes/<arm>/redteam_postprocessed_iter3.jsonl`` — the NEW conversations
* ``probes/<arm>/…iter3.jsonl.bak``                  — the OLD ones (or git, see --old-from-git)
* ``probes/<arm>/contrastive_cache.jsonl``           — the red-team SOURCE each pair came from

keeps only the pairs where a refusal is present **before or after**, classifies each
as ``flipped`` (compliant → refusal), ``both`` (refused either way — usually just a
change of rhetorical order) or ``dropped`` (refusal → compliant), and embeds them as
JSON in one HTML file that opens by double-click.

The ``.bak`` files are not committed, so on a clean checkout pass ``--old-from-git``
to read the pre-shortening dumps out of git instead::

    .venv_claude/bin/python scripts/build_shortened_pairs_viewer.py --old-from-git HEAD
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from publish_kaggle_redteam_activations import ARMS  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "viewers" / "shortened_refusal_pairs.html"

# A refusal anywhere in the assistant text — deliberately not anchored at position 0.
# Anchoring it was what made the shift look far larger than it is: most of the old
# pairs already refused, they just opened with the helpful clause first.
REFUSAL = re.compile(
    r"(i can(?:no|')?t|i can’t|i cannot|i won'?t|i’m not able|i'?m not able|i am not able"
    r"|can'?t provide|can’t provide|won'?t provide|unable to provide|i'?m sorry, but)",
    re.I,
)
CODE_FENCE = re.compile(r"```")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_old_rows(probe_dir: Path, iteration: int, from_git: str | None) -> list[dict]:
    """The pre-shortening dump: the .bak beside it, or the same path out of a git rev."""
    dump = probe_dir / f"redteam_postprocessed_iter{iteration}.jsonl"
    if from_git:
        rel = dump.relative_to(REPO_ROOT)
        blob = subprocess.run(
            ["git", "show", f"{from_git}:{rel.as_posix()}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [json.loads(line) for line in blob.splitlines() if line.strip()]
    backup = dump.with_suffix(dump.suffix + ".bak")
    if not backup.exists():
        raise SystemExit(
            f"{backup} not found — the originals are only in git on a clean checkout. "
            "Re-run with --old-from-git <rev> (e.g. the commit before the shortening)."
        )
    return load_jsonl(backup)


def assistant_text(messages: list[dict]) -> str:
    return " ".join(m["content"] for m in messages if m.get("role") == "assistant")


def collect(iteration: int, from_git: str | None) -> tuple[list[dict], list[dict]]:
    """``(entries, per-arm stats)`` for every refusal-related regenerated pair."""
    entries: list[dict] = []
    stats: list[dict] = []

    for arm in ARMS.values():
        probe_dir = arm.probe_path
        report = {r["id"]: r for r in load_jsonl(probe_dir / f"shortened_pairs_iter{iteration}.jsonl")}
        if not report:
            continue
        new_rows = load_jsonl(probe_dir / f"redteam_postprocessed_iter{iteration}.jsonl")
        old_rows = load_old_rows(probe_dir, iteration, from_git)
        cache = {}
        for row in load_jsonl(probe_dir / "contrastive_cache.jsonl"):
            cache[row["key"]] = row["record"]  # last write wins, as _load_cache does

        old_by_id = {r["id"]: r for r in old_rows}
        n_regen = n_old_ref = n_new_ref = n_old_code = n_new_code = n_flip = 0

        for row in new_rows:
            rep = report.get(row["id"])
            if rep is None:
                continue
            old = old_by_id.get(row["id"])
            if old is None or old["inputs"] == row["inputs"]:
                continue
            n_regen += 1
            old_txt, new_txt = assistant_text(old["inputs"]), assistant_text(row["inputs"])
            old_ref, new_ref = bool(REFUSAL.search(old_txt)), bool(REFUSAL.search(new_txt))
            old_code, new_code = bool(CODE_FENCE.search(old_txt)), bool(CODE_FENCE.search(new_txt))
            n_old_ref += old_ref
            n_new_ref += new_ref
            n_old_code += old_code
            n_new_code += new_code
            if not old_ref and new_ref:
                n_flip += 1
            if not (old_ref or new_ref):
                continue  # not refusal-related; out of scope for this viewer

            record = cache.get(rep.get("cache_key"), {})
            entries.append(
                {
                    "arm": arm.name,
                    "id": row["id"],
                    "label": row["label"],
                    "category": "flipped" if (new_ref and not old_ref)
                    else ("dropped" if (old_ref and not new_ref) else "both"),
                    "before_tokens": rep["before_tokens"],
                    "after_tokens": rep["after_tokens"],
                    "attempts": rep["attempts"],
                    "status": rep["status"],
                    "old": old["inputs"],
                    "new": row["inputs"],
                    "source": record.get("original_messages", []),
                    "source_label": record.get("original_label", ""),
                    "old_code": old_code,
                    "new_code": new_code,
                }
            )

        stats.append(
            {
                "arm": arm.name,
                "regenerated": n_regen,
                "refusal_old": n_old_ref,
                "refusal_new": n_new_ref,
                "code_old": n_old_code,
                "code_new": n_new_code,
                "flipped": n_flip,
                "shown": sum(1 for e in entries if e["arm"] == arm.name),
            }
        )

    order = {"flipped": 0, "both": 1, "dropped": 2}
    entries.sort(key=lambda e: (order[e["category"]], e["arm"], e["id"]))
    return entries, stats


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shortened contrastive pairs — refusal shift</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --border:#2a2f3a;
    --fg:#e6e8ec; --muted:#9aa3b2; --accent:#6ea8fe;
    --pos:#ff6b6b; --pos-bg:#2a1618; --neg:#4dd08a; --neg-bg:#142218;
    --user:#7aa2ff; --asst:#c9a0ff; --old:#c98b3f; --new:#4dd08a;
    --mark:#5a3a00; --mark-fg:#ffd489;
  }
  @media (prefers-color-scheme: light){
    :root{
      --bg:#f6f7f9; --panel:#ffffff; --panel2:#f0f2f5; --border:#d9dde4;
      --fg:#1a1d23; --muted:#5b6472; --accent:#2f6fed;
      --pos:#c0392b; --pos-bg:#fdecea; --neg:#1e8e5a; --neg-bg:#e8f6ee;
      --user:#2f5fed; --asst:#7a3ff0; --old:#9a6b00; --new:#1e8e5a;
      --mark:#ffe9a8; --mark-fg:#6b4a00;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       background:var(--bg);color:var(--fg)}
  header{padding:18px 22px 6px}
  h1{margin:0 0 6px;font-size:18px}
  .sub{color:var(--muted);font-size:13px;max-width:1100px;margin-bottom:4px}
  .sub code{background:var(--panel2);padding:1px 5px;border-radius:4px}
  table.stats{border-collapse:collapse;margin:12px 22px 0;font-size:12.5px}
  table.stats th,table.stats td{border:1px solid var(--border);padding:5px 11px;text-align:right}
  table.stats th:first-child,table.stats td:first-child{text-align:left;font-weight:600}
  table.stats th{background:var(--panel2);color:var(--muted);font-weight:600}
  .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:14px 22px;background:var(--panel);
            margin-top:14px;border-top:1px solid var(--border);border-bottom:1px solid var(--border);
            position:sticky;top:0;z-index:5}
  .controls input[type=search]{flex:1;min-width:220px;padding:8px 12px;border:1px solid var(--border);
                               border-radius:8px;background:var(--bg);color:var(--fg)}
  .seg{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
  .seg button{padding:8px 13px;background:var(--bg);color:var(--muted);border:none;cursor:pointer;
              font-weight:600;font-size:12.5px}
  .seg button.active{background:var(--accent);color:#fff}
  .meta{color:var(--muted);font-size:12.5px}
  .list{padding:10px 22px 70px;max-width:1500px;margin:0 auto}
  .pair{margin:16px 0;border:1px solid var(--border);border-radius:12px;overflow:hidden;background:var(--panel2)}
  .pair-head{display:flex;align-items:center;gap:9px;padding:9px 14px;font-size:12px;color:var(--muted);
             border-bottom:1px solid var(--border);flex-wrap:wrap}
  .pair-head .idx{font-weight:700;color:var(--fg);font-family:ui-monospace,Menlo,monospace}
  .pill{font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px;font-family:ui-monospace,Menlo,monospace}
  .pill.positive{color:var(--pos);background:var(--pos-bg)}
  .pill.negative{color:var(--neg);background:var(--neg-bg)}
  .badge{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
         padding:2px 8px;border-radius:5px;border:1px solid var(--border);color:var(--muted)}
  .badge.flipped{color:var(--pos);border-color:var(--pos)}
  .badge.dropped{color:var(--accent);border-color:var(--accent)}
  .tok{font-family:ui-monospace,Menlo,monospace}
  .tok b{color:var(--fg)}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border)}
  @media (max-width:900px){.cols{grid-template-columns:1fr}}
  .col{background:var(--panel)}
  .col-head{display:flex;align-items:center;gap:8px;padding:8px 14px;background:var(--panel2);
            border-bottom:1px solid var(--border);font-size:11px;font-weight:700;
            text-transform:uppercase;letter-spacing:.05em}
  .col.old{border-top:3px solid var(--old)} .col.old .col-head{color:var(--old)}
  .col.new{border-top:3px solid var(--new)} .col.new .col-head{color:var(--new)}
  .msgs{padding:6px 14px 12px}
  .msg{padding:8px 0;border-bottom:1px dashed var(--border)}
  .msg:last-child{border-bottom:none}
  .role{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
  .role.user{color:var(--user)} .role.assistant{color:var(--asst)}
  .content{white-space:pre-wrap;word-wrap:break-word;font-size:13px}
  mark{background:var(--mark);color:var(--mark-fg);padding:0 2px;border-radius:3px}
  details.src{border-top:1px solid var(--border);background:var(--panel)}
  details.src summary{padding:8px 14px;cursor:pointer;font-size:12px;color:var(--muted);font-weight:600}
  details.src .msgs{background:var(--panel)}
  .empty{padding:40px 22px;color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>Shortened contrastive pairs — the refusal shift</h1>
  <div class="sub">
    Every pair here was re-generated by <code>scripts/shorten_long_contrastive_pairs.py</code> because it
    overran tuberlens' 1024-token activation window, and involves a refusal before or after. Under a word
    budget the cheapest "not harmful" assistant turn is a refusal, so pairs that used to answer with a safe
    substitute — often a code block — may now decline instead. <b>Left = old (truncated at extraction),
    right = new.</b> Refusal phrases are highlighted; the red-team conversation each pair was generated
    <i>from</i> is under "source".
  </div>
  __STATS__
</header>
<div class="controls">
  <div class="seg" id="cat">
    <button data-v="all" class="active">All</button>
    <button data-v="flipped">Compliant → refusal</button>
    <button data-v="both">Refused either way</button>
    <button data-v="dropped">Refusal → compliant</button>
  </div>
  <div class="seg" id="arm"></div>
  <input type="search" id="q" placeholder="Search id or text…">
  <span class="meta" id="count"></span>
</div>
<div class="list" id="list"></div>
<script>
const ENTRIES = __ENTRIES__;
const REFUSAL = /(i can(?:no|')?t|i can’t|i cannot|i won'?t|i’m not able|i'?m not able|i am not able|can'?t provide|can’t provide|won'?t provide|unable to provide|i'?m sorry, but)/ig;
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const mark = s => esc(s).replace(REFUSAL, m => '<mark>' + m + '</mark>');

const arms = [...new Set(ENTRIES.map(e => e.arm))];
const armBox = document.getElementById('arm');
armBox.innerHTML = ['all', ...arms].map((a, i) =>
  `<button data-v="${a}" class="${i === 0 ? 'active' : ''}">${a === 'all' ? 'Both arms' : a}</button>`).join('');

let state = {cat: 'all', arm: 'all', q: ''};
for (const box of [document.getElementById('cat'), armBox]) {
  box.addEventListener('click', ev => {
    const b = ev.target.closest('button'); if (!b) return;
    [...box.children].forEach(c => c.classList.toggle('active', c === b));
    state[box.id] = b.dataset.v; render();
  });
}
document.getElementById('q').addEventListener('input', ev => { state.q = ev.target.value.toLowerCase(); render(); });

const msgs = (list, hl) => '<div class="msgs">' + list.map(m =>
  `<div class="msg"><div class="role ${m.role}">${m.role}</div>` +
  `<div class="content">${hl ? mark(m.content) : esc(m.content)}</div></div>`).join('') + '</div>';

function card(e) {
  const catLabel = {flipped: 'compliant → refusal', both: 'refused either way', dropped: 'refusal → compliant'}[e.category];
  return `<div class="pair">
    <div class="pair-head">
      <span class="idx">${e.id}</span>
      <span class="pill ${e.label}">${e.label}</span>
      <span class="badge ${e.category}">${catLabel}</span>
      <span class="tok"><b>${e.before_tokens}</b> → <b>${e.after_tokens}</b> tokens</span>
      <span class="meta">${e.attempts} attempt${e.attempts > 1 ? 's' : ''} · ${e.arm}</span>
      ${e.old_code ? '<span class="badge">old had code</span>' : ''}
      ${e.new_code ? '<span class="badge">new has code</span>' : ''}
    </div>
    <div class="cols">
      <div class="col old"><div class="col-head">old · ${e.before_tokens} tokens (cut at 1024)</div>${msgs(e.old, true)}</div>
      <div class="col new"><div class="col-head">new · ${e.after_tokens} tokens</div>${msgs(e.new, true)}</div>
    </div>
    <details class="src"><summary>source red-team conversation — label ${e.source_label} (the pair was generated from this)</summary>
      ${msgs(e.source, false)}</details>
  </div>`;
}

function render() {
  const shown = ENTRIES.filter(e =>
    (state.cat === 'all' || e.category === state.cat) &&
    (state.arm === 'all' || e.arm === state.arm) &&
    (!state.q || e.id.toLowerCase().includes(state.q) ||
      JSON.stringify(e.old).toLowerCase().includes(state.q) ||
      JSON.stringify(e.new).toLowerCase().includes(state.q)));
  document.getElementById('count').textContent = `${shown.length} of ${ENTRIES.length} pairs`;
  document.getElementById('list').innerHTML =
    shown.length ? shown.map(card).join('') : '<div class="empty">Nothing matches those filters.</div>';
}
render();
</script>
</body>
</html>
"""


def stats_table(stats: list[dict]) -> str:
    head = (
        "<tr><th>arm</th><th>regenerated</th><th>refusal old → new</th>"
        "<th>compliant → refusal</th><th>code old → new</th><th>shown here</th></tr>"
    )
    rows = "".join(
        f"<tr><td>{s['arm']}</td><td>{s['regenerated']}</td>"
        f"<td>{s['refusal_old']} → {s['refusal_new']}</td><td>{s['flipped']}</td>"
        f"<td>{s['code_old']} → {s['code_new']}</td><td>{s['shown']}</td></tr>"
        for s in stats
    )
    return f'<table class="stats">{head}{rows}</table>'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iteration", type=int, default=3, help="Which iteration's dumps to read (default 3).")
    parser.add_argument(
        "--old-from-git",
        default=None,
        metavar="REV",
        help="Read the pre-shortening dumps from this git revision instead of the .bak files "
        "(the .bak files are not committed).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output HTML (default {DEFAULT_OUT}).")
    args = parser.parse_args(argv)

    entries, stats = collect(args.iteration, args.old_from_git)
    if not entries:
        raise SystemExit("No refusal-related regenerated pairs found — nothing to build.")

    # Escape the angle brackets before inlining: red-team conversations contain markup
    # and comment openers, and a bare `<!--` (or `</script`) inside a <script> element
    # flips the HTML parser into script-data-escaped state and eats the rest of the
    # file. `<` `>` `&` never appear as JSON structure, so this only touches strings,
    # and \u escapes are read back identically by the JS parser.
    payload = json.dumps(entries, ensure_ascii=False)
    for ch in "&<>":
        # chr(92) is a backslash: spelled this way so the escape sequence being built
        # (< and friends) never appears literally in this file.
        payload = payload.replace(ch, f"{chr(92)}u{ord(ch):04x}")
    html = HTML_TEMPLATE.replace("__STATS__", stats_table(stats)).replace("__ENTRIES__", payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    for s in stats:
        print(
            f"{s['arm']:>14}: {s['regenerated']:3d} regenerated, {s['shown']:3d} refusal-related "
            f"(refusal {s['refusal_old']}→{s['refusal_new']}, flips {s['flipped']}, "
            f"code {s['code_old']}→{s['code_new']})"
        )
    print(f"\nWrote {len(entries)} pairs to {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
