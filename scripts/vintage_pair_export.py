"""Export the transfer cohort's red-team pairs — both conversations — into ``report.html``.

The sweep and the transfer analysis both report *rates*. This makes the underlying
conversations readable: for each pair, the attacker-written success and the LLM-written
opposite-class counterpart side by side, tagged with the vintage it entered at, the error
the attacker was hunting, and how many of the ten v2-vintage probes it defeated.

By default only the **vintage 3 minus vintage 2** pairs are exported — the 157 pairs the
transfer section is about — not all 659 in the iteration-3 dump. ``--cohort all`` exports
everything, at roughly four times the page weight.

Why it is embedded rather than written alongside
------------------------------------------------
A published artifact is one self-contained file under a strict CSP — it cannot fetch a
sibling JSON. The cohort is ~0.35 MB of conversation text across both arms (the whole
dump would be ~1.5 MB), well inside the size budget, so the payload is inlined into a
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


def _v2_pairs(path: Path) -> dict[tuple[str, str], dict]:
    """``{(arm, source_key): record}`` of the v2 minimal-edit rewrites.

    ``source_key`` is ``sha16`` of the source's canonical text **after** the config's
    message transforms, which is exactly ``sha16(canon(dump row))`` — so this joins to
    the dump-derived pairs without re-deriving anything. Absent file → no v2 side, and
    the viewer simply doesn't offer it.
    """
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["arm"], r["source_key"])] = r
    return out


def _v2_scores(path: Path) -> dict[tuple[str, str, str], dict]:
    """``{(arm, side, source_key): record}`` from ``score_contrastive_v2.py``."""
    out: dict[tuple[str, str, str], dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["arm"], r["side"], r["source_key"])] = r
    return out


def build_arm(arm: str, iteration: int, drop_mode: str, metrics: dict,
              cohort: str = "notv2", v2_pairs: dict | None = None,
              v2_scores: dict | None = None) -> dict:
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

    v2_pairs = v2_pairs or {}
    v2_scores = v2_scores or {}

    def counterpart(source_row: int | None) -> dict | None:
        """This pair's minimal-edit counterpart.

        Its conversation comes from the generation output and its numbers from the
        scoring run — joined on the same ``source_key`` the viewer's pairs are keyed by.
        ``tok`` is read off its extracted blob, so a counterpart that overran the
        1024-token window would be visible here exactly as a source one is.
        """
        if source_row is None:
            return None
        skey = A.sha16(A.canon(ds.inputs[source_row]))
        rec = v2_pairs.get((arm, skey))
        if rec is None:
            return None
        from tuberlens.interfaces.dataset import Message

        msgs = A.apply_transforms(
            [Message(role=m["role"], content=m["content"]) for m in rec["new_messages"]]
        )
        blob = A.redteam_blob_path(msgs)
        sc = v2_scores.get((arm, "new", skey), {})
        # A counterpart identical to its source is the one degenerate outcome a
        # minimal-edit instruction can produce: the training set would then carry the
        # same conversation twice under opposite labels. It happens when the generator
        # disagrees with the source's judge label and says so instead of editing.
        # Flagged here so the viewer can badge and filter it, and so it can never be
        # quietly folded into a retrain.
        return {
            "same": A.canon(msgs) == A.canon(ds.inputs[source_row]),
            "label": rec["target_label"],
            "tok": (A.blob_width(blob) if blob.exists() else None),
            "sim": round(rec["sim_new"], 4),
            "why": rec.get("explanation", ""),
            "nw": sc.get("n_misclassified"),
            "ns": sc.get("n_seeds"),
            "p2": (None if sc.get("pipeline_iter2_prob") is None
                   else round(sc["pipeline_iter2_prob"], 4)),
            "w2": (None if sc.get("pipeline_iter2_wrong") is None
                   else int(sc["pipeline_iter2_wrong"])),
            "msgs": [
                [ROLES.index(m.role) if m.role in ROLES else 1, m.content] for m in msgs
            ],
        }

    out = []
    for p in pairs:
        anchor = p.source_idx if p.source_idx is not None else p.generated_idx
        # ``notv2`` (the default) keeps exactly the transfer cohort — vintage 3 minus
        # vintage 2 — which is what every number in the section above is computed over.
        # Exporting all 659 pairs instead would quadruple the page for material the
        # section does not discuss.
        if cohort == "notv2" and vintage_of(anchor) not in ("new", "readd"):
            continue
        m = metrics.get((arm, anchor), {})
        out.append({
            "id": p.pair_id,
            "v": vintage_of(anchor),
            "et": {"false_positive": "fp", "false_negative": "fn"}.get(
                m.get("found_error_type")),
            "found": m.get("found_iteration"),
            "src": side(p.source_idx),
            # The pair's counterpart is the minimal-edit rewrite. The dump also holds
            # the counterpart the pipeline itself generated, but the viewer does not
            # show it: it is superseded, and carrying it doubled the payload for a
            # conversation the page makes no claim about.
            "cp": counterpart(p.source_idx),
        })
    return {"pairs": out, "n_orphan": stats["n_orphan"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--drop-overlong", choices=("none", "row", "pair"), default="pair")
    ap.add_argument(
        "--cohort", choices=("notv2", "all"), default="notv2",
        help="notv2 (default) exports only the vintage-3-minus-vintage-2 pairs, i.e. the "
             "cohort the transfer section measures. 'all' exports every iteration-3 pair.",
    )
    ap.add_argument("--out-dir", type=Path,
                    default=A.REPO / "results_hs_gemma27b_batch_ablation/vintage")
    ap.add_argument("--v2-dir", type=Path,
                    default=A.REPO / "results_hs_gemma27b_batch_ablation/contrastive_v2",
                    help="regen_cohort_contrastive.py output (the v2 conversations)")
    ap.add_argument("--v2-scored-dir", type=Path,
                    default=A.REPO / "results_hs_gemma27b_batch_ablation/contrastive_v2_scored",
                    help="score_contrastive_v2.py output (the v2 numbers)")
    args = ap.parse_args()

    metrics = _metrics(args.out_dir / "new_sample_success.jsonl")
    v2_pairs = _v2_pairs(args.v2_dir / "cohort_contrastive_v2.jsonl")
    v2_scores = _v2_scores(args.v2_scored_dir / "contrastive_v2_success.jsonl")
    print(f"counterparts: {len(v2_pairs)}   v2 scored rows: {len(v2_scores)}", flush=True)

    payload = {"roles": ROLES, "cohort": args.cohort, "arms": {}}
    for arm in sorted(A.ARMS):
        payload["arms"][arm] = build_arm(arm, args.iteration, args.drop_overlong, metrics,
                                         args.cohort, v2_pairs, v2_scores)
        from collections import Counter
        n = payload["arms"][arm]["pairs"]
        c = Counter(p["v"] for p in n)
        print(f"{arm:14s} {len(n):4d} pair(s)  "
              + "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
              + f"   [in v3 not v2 = {c['new'] + c['readd']}]"
              + f"  counterparts={sum(1 for p in n if p['cp'])}"
              + f"  UNCHANGED={sum(1 for p in n if p['cp'] and p['cp']['same'])}"
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
