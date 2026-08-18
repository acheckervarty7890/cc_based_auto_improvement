"""Render the JSON produced by analyze_redteam_successes.py as markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLITS = [("combined", "combined"), ("false_positive", "false-positive"), ("false_negative", "false-negative")]


def length_row(tag: str, st: dict) -> str:
    u = st["by_role"].get("user", {})
    a = st["by_role"].get("assistant", {})
    return (
        f"| {tag} | {st['n_conversations']} | {st['turns_per_conversation']:.2f} | "
        f"{u.get('turns_per_conversation', 0):.2f} | {u.get('words_per_turn', 0):.1f} | {u.get('chars_per_turn', 0):.0f} | "
        f"{a.get('turns_per_conversation', 0):.2f} | {a.get('words_per_turn', 0):.1f} | {a.get('chars_per_turn', 0):.0f} | "
        f"{st['words_per_conversation']:.1f} | {st['chars_per_conversation']:.0f} |"
    )


LEN_HEADER = (
    "| set | n convs | turns/conv | user turns/conv | user words/turn | user chars/turn | "
    "asst turns/conv | asst words/turn | asst chars/turn | words/conv | chars/conv |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def render(report: dict) -> str:
    out: list[str] = []
    out.append("# Red-team success analysis: topics & conversation length\n")
    out.append(
        "Per iteration, per arm. **Original** = the success conversation the attacker submitted. "
        "**Contrastive** = the opposite-class conversation the preprocessing LLM generated from it "
        "(matched by the same `sha256(source messages + target label)` cache key the pipeline uses, "
        "so the pairing is exact). Coverage < 100% is `filter_dataset` dropping sources before "
        "contrastive generation, not a matching failure.\n"
    )
    out.append("Lengths are means. A *turn* is one message; `words` splits on whitespace.\n")

    tm = report["topic_model"]
    out.append(f"\n## Topic model\n")
    out.append(
        f"TF-IDF (1-2 grams) over each conversation's **user turns**, KMeans k={tm['k']}, seed {tm['seed']}, "
        "fit **once over all arms/iterations** so a topic id means the same thing in every table. "
        "Cluster names were assigned by reading each cluster's top terms and sampled members.\n"
    )
    out.append("| id | topic | top terms |\n|---|---|---|")
    for cid, c in sorted(tm["clusters"].items(), key=lambda kv: int(kv[0])):
        out.append(f"| {cid} | {c['name']} | {', '.join(c['top_terms'][:6])} |")

    for arm_name, arm in report["arms"].items():
        out.append(f"\n\n## {arm_name}\n")
        out.append(f"`{arm['results_dir']}` — {arm['n_successes']} successes total\n")
        for it, itd in sorted(arm["iterations"].items(), key=lambda kv: int(kv[0])):
            out.append(f"\n### Iteration {it}  ({itd['n_attempts']} attempts)\n")
            if not itd["splits"]:
                out.append("\n**No successes this iteration.**\n")
                continue
            for key, pretty in SPLITS:
                sp = itd["splits"].get(key)
                if not sp:
                    continue
                out.append(
                    f"\n**{pretty}** — {sp['n_successes']} successes, "
                    f"contrastive coverage {100 * sp['contrastive_coverage']:.0f}%\n"
                )
                out.append(LEN_HEADER)
                out.append(length_row("original", sp["original"]))
                if sp["contrastive"]["n_conversations"]:
                    out.append(length_row("contrastive", sp["contrastive"]))
                out.append("\n_Top 10 topics_\n")
                out.append("| # | topic | n | % |\n|---|---|---|---|")
                for i, t in enumerate(sp["topics"], 1):
                    out.append(f"| {i} | {t['topic']} | {t['count']} | {t['pct']:.1f}% |")
                if sp["keywords"]:
                    out.append(f"\n_Top TF-IDF keywords_: {', '.join(sp['keywords'])}\n")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.write_text(render(json.loads(args.report.read_text())), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
