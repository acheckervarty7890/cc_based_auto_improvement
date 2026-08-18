#!/usr/bin/env python
"""Collect every AUROC + sample-count number behind the hu_harm/llama-1b writeups
into one JSON, for scripts/build_hu_harm_auroc_viewer.py to template.

Two families, deliberately kept separate because they are NOT cell-comparable:

  cross_experiment — the 9 arms of experiment{1,3,4,5,10}_cloud, read from each
      branch's committed comparison CSV + probing JSONLs + postprocessed
      snapshots via `git show`. Scored on whatever eval activations that run's
      machine had.
  ablation — the 5 preprocessing/confidence variants of experiment10_cloud's two
      batch arms, read from the local working tree. All scored on one identical
      local eval-activation blob set.

Usage:  .venv_claude/bin/python scripts/collect_hu_harm_auroc_data.py [-o PATH]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPLITS = ["eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal",
          "eval_daily_dilemmas", "mean"]

# --- cross-experiment arms ---------------------------------------------------
# (exp, arm label, branch, results dir, jsonl stem, probe dir, comparison csv, params)
CROSS = [
    ("exp1", "Llama-3.3-70B", "experiment1_cloud",
     "results_hu_harm_llama70b50", "llama70b_probing",
     "probes/llama70b50_llama70b", "llama70b_comparison.csv",
     dict(attacker="meta-llama/Llama-3.3-70B-Instruct", feedback="per-turn",
          view_limit=4, memo="off (not yet implemented)", judge="gpt-5.1-chat",
          contrastive="gpt-5.1-chat")),
    ("exp1", "gpt-oss-120b", "experiment1_cloud",
     "results_hu_harm_llama70b50_gptoss120b", "gptoss120b_probing",
     "probes/llama70b50_gptoss120b", "gptoss120b_comparison.csv",
     dict(attacker="openai/gpt-oss-120b", feedback="per-turn", view_limit=4,
          memo="off (not yet implemented)", judge="gpt-5.1-chat",
          contrastive="gpt-5.1-chat")),
    ("exp3", "gpt-oss-120b run2", "experiment3_cloud",
     "results_hu_harm_llama70b50_gptoss120b_run2", "gptoss120b_probing",
     "probes/llama70b50_gptoss120b_run2", "gptoss120b_comparison.csv",
     dict(attacker="openai/gpt-oss-120b", feedback="per-turn", view_limit=4,
          memo="ON", judge="gpt-5.1-chat", contrastive="gpt-5.1")),
    ("exp4", "gpt-5.1 MEMO", "experiment4_cloud",
     "results_hu_harm_llama70b50_gpt51_memo", "gpt51_probing",
     "probes/llama70b50_gpt51_memo", "gpt51_comparison.csv",
     dict(attacker="openai/gpt-5.1", feedback="per-turn", view_limit=4,
          memo="ON", judge="gpt-5.1", contrastive="gpt-5.1")),
    ("exp4", "gpt-5.1 nomemo", "experiment4_cloud",
     "results_hu_harm_llama70b50_gpt51_nomemo", "gpt51_probing",
     "probes/llama70b50_gpt51_nomemo", "gpt51_comparison.csv",
     dict(attacker="openai/gpt-5.1", feedback="per-turn", view_limit=4,
          memo="off", judge="gpt-5.1", contrastive="gpt-5.1")),
    ("exp5", "deepseek-v4-flash MEMO", "experiment5_cloud",
     "results_hu_harm_llama70b50_deepseekv4_memo", "deepseekv4_probing",
     "probes/llama70b50_deepseekv4_memo", "deepseekv4_comparison.csv",
     dict(attacker="deepseek/deepseek-v4-flash", feedback="per-turn", view_limit=4,
          memo="ON", judge="gpt-5.1-chat", contrastive="deepseek-v4-flash")),
    ("exp5", "deepseek-v4-flash nomemo", "experiment5_cloud",
     "results_hu_harm_llama70b50_deepseekv4_nomemo", "deepseekv4_probing",
     "probes/llama70b50_deepseekv4_nomemo", "deepseekv4_comparison.csv",
     dict(attacker="deepseek/deepseek-v4-flash", feedback="per-turn", view_limit=4,
          memo="off", judge="gpt-5.1-chat", contrastive="deepseek-v4-flash")),
    ("exp10", "deepseek-v4-pro BATCH", "experiment10_cloud",
     "results_hu_harm_llama70b50_deepseekv4pro_batch", "deepseekv4pro_probing",
     "probes/hu_harm_llama1b_deepseekv4pro_batch", "deepseekv4pro_comparison.csv",
     dict(attacker="deepseek/deepseek-v4-pro", feedback="BATCH (blind)",
          view_limit=0, memo="off", judge="gpt-5.1", contrastive="gpt-5.1")),
    ("exp10", "gpt-oss-120b BATCH", "experiment10_cloud",
     "results_hu_harm_llama70b50_gptoss120b_batch", "gptoss120b_probing",
     "probes/hu_harm_llama1b_gptoss120b_batch", "gptoss120b_comparison.csv",
     dict(attacker="openai/gpt-oss-120b", feedback="BATCH (blind)", view_limit=0,
          memo="off", judge="gpt-5.1", contrastive="gpt-5.1")),
]

# --- exp10 ablation variants -------------------------------------------------
ABLATION_ARMS = [
    ("deepseek-v4-pro BATCH", "hu_harm_llama1b_deepseekv4pro_batch",
     "results_hu_harm_llama70b50_deepseekv4pro_batch/deepseekv4pro_probing_%s.jsonl"),
    ("gpt-oss-120b BATCH", "hu_harm_llama1b_gptoss120b_batch",
     "results_hu_harm_llama70b50_gptoss120b_batch/gptoss120b_probing_%s.jsonl"),
]
# (label, comparison-csv suffix, probe-dir suffix, min judge confidence, contrastive?)
ABLATION_VARIANTS = [
    ("contrastive · conf≥7 (original)", "_WITHcontrastive_localacts", "", 7, True),
    ("contrastive · conf≥7 (cache-only)", "_cachedcontrastive_conf7check",
     "_cachedcontrastive_conf7check", 7, True),
    ("contrastive · conf=10", "_cachedcontrastive_conf10",
     "_cachedcontrastive_conf10", 10, True),
    ("no contrastive · conf≥7", "_nocontrastive", "_nocontrastive", 7, False),
    ("no contrastive · conf=10", "_nocontrastive_conf10",
     "_nocontrastive_conf10", 10, False),
]
ABLATION_DIR = REPO / "results_hu_harm_llama70b50_batch_nocontrastive"
BASE_TRAIN_N = 50  # data/hu_harm_llama70b_50.jsonl


def git_show(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True,
                       text=True, cwd=REPO)
    return r.stdout if r.returncode == 0 else None


def parse_auroc(text: str) -> dict:
    """{iterN: {split: {auroc, accuracy, tpr_at_fpr}}}"""
    out: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(text)):
        out.setdefault(row["round"], {})[row["dataset"]] = {
            "auroc": float(row["auroc"]),
            "accuracy": float(row["accuracy"]),
            "tpr_at_fpr": float(row["tpr_at_fpr"]),
        }
    return out


def jsonl_rows(text: str):
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def label_counts(text: str) -> dict:
    c = Counter(r["label"] for r in jsonl_rows(text))
    return {"total": sum(c.values()), "positive": c.get("positive", 0),
            "negative": c.get("negative", 0)}


def collect_cross() -> list[dict]:
    arms = []
    for exp, label, branch, rdir, stem, pdir, csv_name, params in CROSS:
        auroc = parse_auroc(git_show(branch, f"{rdir}/{csv_name}"))
        attempts: dict[int, dict] = {}
        for et in ("fp", "fn"):
            text = git_show(branch, f"{rdir}/{stem}_{et}.jsonl") or ""
            for r in jsonl_rows(text):
                it = r.get("iteration", -1)
                cell = attempts.setdefault(it, {"fp_att": 0, "fp_succ": 0,
                                                "fn_att": 0, "fn_succ": 0})
                cell[f"{et}_att"] += 1
                if r.get("success"):
                    cell[f"{et}_succ"] += 1
        train = {}
        for i in range(1, 6):
            text = git_show(branch, f"{pdir}/redteam_postprocessed_iter{i}.jsonl")
            if text is None:
                continue
            train[f"iter{i}"] = label_counts(text)
        arms.append({
            "exp": exp, "label": label, "branch": branch, "params": params,
            "auroc": auroc,
            "rounds": {str(k): v for k, v in sorted(attempts.items()) if k >= 0},
            "train": train,
        })
    return arms


def collect_ablation() -> list[dict]:
    out = []
    for arm_label, stem, jsonl_tpl in ABLATION_ARMS:
        variants = []
        for vlabel, csv_suffix, probe_suffix, min_conf, has_contra in ABLATION_VARIANTS:
            csv_path = ABLATION_DIR / f"{stem}{csv_suffix}_comparison.csv"
            if not csv_path.exists():
                print(f"  skip (missing): {csv_path}")
                continue
            auroc = parse_auroc(csv_path.read_text())
            probe_dir = REPO / "probes" / f"{stem}{probe_suffix}"
            train = {}
            for k in (0, 1, 2):
                it = f"iter{k + 1}"
                pp = probe_dir / f"redteam_postprocessed_{it}.jsonl"
                cell = {"successes_raw": 0, "successes_after_conf": 0}
                for et in ("fp", "fn"):
                    for r in jsonl_rows((REPO / (jsonl_tpl % et)).read_text()):
                        if not r.get("success"):
                            continue
                        if not (0 <= r.get("iteration", -1) <= k):
                            continue
                        cell["successes_raw"] += 1
                        if r.get("judge_confidence", 0) >= min_conf:
                            cell["successes_after_conf"] += 1
                if pp.exists():
                    cell.update(label_counts(pp.read_text()))
                    cell["total_training"] = cell["total"] + BASE_TRAIN_N
                train[it] = cell
            variants.append({
                "label": vlabel, "min_judge_confidence": min_conf,
                "contrastive": has_contra, "auroc": auroc, "train": train,
            })
        out.append({"label": arm_label, "stem": stem, "variants": variants})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path,
                    default=REPO / "scripts" / "hu_harm_auroc_data.json")
    args = ap.parse_args()

    print("collecting cross-experiment arms (git show over 5 branches) ...")
    cross = collect_cross()
    print("collecting exp10 ablation variants (local) ...")
    ablation = collect_ablation()

    payload = {
        "splits": SPLITS,
        "base_training_samples": BASE_TRAIN_N,
        "cross_experiment": cross,
        "ablation": ablation,
    }
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    print(f"  cross-experiment arms: {len(cross)}")
    print(f"  ablation arms: {len(ablation)} "
          f"({sum(len(a['variants']) for a in ablation)} variants total)")


if __name__ == "__main__":
    main()
