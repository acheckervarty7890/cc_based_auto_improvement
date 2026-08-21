"""Render the ceiling + dev-sweep results into tables (stdout, markdown)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SPLITS = [
    "anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
    "hc_contradiction", "mm_substitution", "oig_context_drift", "oig_omission",
]


def rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def md_table(header, body):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in body:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    md = []
    md.append("## Part 1 — ceiling on eval_sets/instructions\n")
    ceil = {}
    for arm in ("gptoss", "nemotron"):
        for r in rows(RES / f"ceiling_{arm}.jsonl"):
            ceil[(arm, r["condition"])] = r
    order = [
        ("gptoss", "redteam_only", "red-team only (gptoss, iter5 training set)"),
        ("nemotron", "redteam_only", "red-team only (nemotron, iter5 training set)"),
        ("gptoss", "cv_eval", "CV on eval alone (5-fold, out-of-fold)"),
        ("gptoss", "cv_eval_rt", "CV on eval + red-team (gptoss)"),
        ("nemotron", "cv_eval_rt", "CV on eval + red-team (nemotron)"),
        ("gptoss", "oracle", "oracle: fit and scored on all 1302 eval rows"),
    ]
    body = []
    for arm, cond, label in order:
        r = ceil.get((arm, cond))
        if not r:
            continue
        m = r["macro"]
        body.append([label, f"{m['auroc']:.4f}", f"{m['accuracy']:.4f}",
                     f"{m['tpr_at_fpr']:.4f}", f"{m['tpr_at_fpr_le']:.4f}"])
    md.append(md_table(["condition", "AUROC", "acc", "TPR@1%FPR (tuberlens)",
                        "TPR@FPR<=1%"], body))

    md.append("\n### per split (AUROC)\n")
    body = []
    for arm, cond, label in order:
        r = ceil.get((arm, cond))
        if not r:
            continue
        body.append([label] + [f"{r['per_split'][s]['auroc']:.3f}" for s in SPLITS])
    md.append(md_table(["condition"] + [s.replace("_", " ") for s in SPLITS], body))

    md.append("\n## Part 2 — dev-sample sweep\n")
    for arm in ("gptoss", "nemotron"):
        rs = rows(RES / f"sweep_{arm}.jsonl")
        if not rs:
            continue
        md.append(f"\n### arm = {arm}\n")
        ns = sorted({r["n_dev"] for r in rs})
        ways = []
        for r in rs:
            key = (r["way"], r.get("ft_lr"))
            if key not in ways:
                ways.append(key)
        for metric, title in (("auroc", "AUROC"), ("tpr_at_fpr_le", "TPR@FPR<=1%")):
            body = []
            for way, lr in ways:
                cells = []
                for n in ns:
                    hit = [r for r in rs if r["way"] == way and r["n_dev"] == n
                           and r.get("ft_lr") == lr]
                    cells.append(f"{hit[0]['macro'][metric]:.4f}" if hit else "-")
                name = way if way == "joint" else f"finetune (lr={lr or 'default 5e-3'})"
                body.append([name] + cells)
            md.append(f"\n**{title}**\n")
            md.append(md_table(["N dev samples"] + [str(n) for n in ns], body))
    text = "\n".join(md)
    print(text)
    (RES / "summary.md").write_text(text)


if __name__ == "__main__":
    main()
