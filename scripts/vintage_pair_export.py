"""Export every iteration-3 red-team pair — both conversations — into ``report.html``.

The sweep and the transfer analysis both report *rates*. This makes the underlying
conversations readable: for each pair, the attacker-written success and the LLM-written
opposite-class counterpart side by side, tagged with the vintage it entered at, the error
the attacker was hunting, and how many of the ten v2-vintage probes it defeated.

Why it is embedded rather than written alongside
------------------------------------------------
A published artifact is one self-contained file under a strict CSP — it cannot fetch a
sibling JSON. The whole iteration-3 dump is ~1.4 MB of conversation text across both
arms, which is well inside the size budget, so the payload is inlined into a
``<script type="application/json">`` block between markers in ``report.html`` and the
page's own code renders from it.

Compactness is still worth a little effort, because this is the only part of the page
whose size scales with the data: roles are interned into a table and each message is a
``[role_index, text]`` pair rather than an object with repeated keys.

The conversations are taken from ``redteam_postprocessed_iter3.jsonl``, i.e. **after**
the config's message transforms — the same text whose activations were extracted and
whose rows every number on the page was computed from. Reading the raw attempt log
instead would show slightly different text from what was actually trained on.

Usage:
    .venv_claude/bin/python scripts/vintage_pair_export.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
from attribution_vintage import dropped_rows, vintages

BEGIN = "<!-- PAIRDATA:BEGIN -->"
END = "<!-- PAIRDATA:END -->"

ROLES = ["system", "user", "assistant", "tool"]


def _metrics(path: Path) -> dict[tuple[str, int], dict]:
    """``{(arm, row): record}`` from the transfer analysis' per-row sidecar."""
    out: dict[tuple[str, int], dict] = {}
    if not path.exists():
        raise SystemExit(f"missing {path} — run vintage_new_sample_success.py first")
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["arm"], r["row"])] = r
    return out


def build_arm(arm: str, iteration: int, drop_mode: str, metrics: dict) -> dict:
    ds = A.load_redteam_dataset(arm, iteration)
    pairs, stats = A.build_pairs(arm, ds)
    exclude, _ = dropped_rows(arm, iteration, drop_mode)
    keep, _ = vintages(arm, iteration, exclude)
    v1, v2 = set(keep[1]), set(keep[2])

    def vintage_of(row: int) -> str:
        """Which vintage set the row belongs to, with the non-nested case broken out.

        ``readd`` — present at iteration 1, **absent** from iteration 2, back at
        iteration 3 — exists because the vintages are not strictly nested:
        ``filter_dataset`` refits each cycle and drops a different top percentile. Those
        pairs are part of "in v3 but not v2" (the transfer cohort) even though they are
        not new work, so collapsing them into ``v1`` would make the viewer's iteration-3
        count disagree with the 70 / 87 reported for that cohort.
        """
        if row in exclude:
            return "excl"
        if row in v1:
            return "v1" if row in v2 else "readd"
        if row in v2:
            return "v2"
        return "new"

    def side(row: int | None) -> dict | None:
        if row is None:
            return None
        m = metrics.get((arm, row), {})
        return {
            "row": row,
            "label": ds.other_fields["labels"][row],
            "tok": A.blob_width(A.redteam_blob_path(ds.inputs[row])),
            # None rather than 0 when unscored, so the viewer can say "not scored"
            # instead of silently rendering a hole as a perfect score.
            "nw": m.get("n_misclassified"),
            "ns": m.get("n_seeds"),
            "p2": (None if m.get("pipeline_iter2_prob") is None
                   else round(m["pipeline_iter2_prob"], 4)),
            "w2": (None if m.get("pipeline_iter2_wrong") is None
                   else int(m["pipeline_iter2_wrong"])),
            "msgs": [
                [ROLES.index(msg.role) if msg.role in ROLES else 1, msg.content]
                for msg in ds.inputs[row]
            ],
        }

    out = []
    for p in pairs:
        anchor = p.source_idx if p.source_idx is not None else p.generated_idx
        m = metrics.get((arm, anchor), {})
        out.append({
            "id": p.pair_id,
            "v": vintage_of(anchor),
            "et": {"false_positive": "fp", "false_negative": "fn"}.get(
                m.get("found_error_type")),
            "found": m.get("found_iteration"),
            "src": side(p.source_idx),
            "gen": side(p.generated_idx),
        })
    return {"pairs": out, "n_orphan": stats["n_orphan"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--drop-overlong", choices=("none", "row", "pair"), default="pair")
    ap.add_argument("--out-dir", type=Path,
                    default=A.REPO / "results_hs_gemma27b_batch_ablation/vintage")
    args = ap.parse_args()

    metrics = _metrics(args.out_dir / "new_sample_success.jsonl")
    payload = {"roles": ROLES, "arms": {}}
    for arm in sorted(A.ARMS):
        payload["arms"][arm] = build_arm(arm, args.iteration, args.drop_overlong, metrics)
        from collections import Counter
        n = payload["arms"][arm]["pairs"]
        c = Counter(p["v"] for p in n)
        print(f"{arm:14s} {len(n):4d} pair(s)  "
              + "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
              + f"   [in v3 not v2 = {c['new'] + c['readd']}]"
              + f"  orphan={payload['arms'][arm]['n_orphan']}", flush=True)

    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # A ``</script>`` anywhere in a conversation would end the tag early and spill the
    # rest of the payload into the document as markup. ``\/`` is a valid JSON escape for
    # ``/`` and parses back identically, so this is unconditional rather than a scan —
    # the content is attacker-written text and no current absence of the sequence is a
    # guarantee about the next export. Same for the comment delimiters.
    blob = blob.replace("</", "<\\/").replace("<!--", "<\\u0021--")
    print(f"payload {len(blob) / 1e6:.2f} MB")

    report = args.out_dir / "report.html"
    html = report.read_text(encoding="utf-8")
    if BEGIN not in html or END not in html:
        raise SystemExit(f"markers {BEGIN} / {END} not found in {report}")
    head, rest = html.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    report.write_text(
        head + BEGIN
        + '\n<script id="pairdata" type="application/json">' + blob + "</script>\n"
        + END + tail,
        encoding="utf-8",
    )
    print(f"injected into {report} ({report.stat().st_size / 1e6:.2f} MB total)")


if __name__ == "__main__":
    main()
