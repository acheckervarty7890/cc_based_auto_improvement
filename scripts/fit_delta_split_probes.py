#!/usr/bin/env python3
"""Split a red-team run's successes by the SIGN of the iteration that produced them,
then retrain and evaluate on each half.

THE QUESTION. Over ten iterations a run's eval AUROC moves up on some cycles and down
on others. Iteration i red-teams probe_iter{i} and its retrain writes probe_iter{i+1},
so the delta ATTRIBUTABLE to iteration i is mean(iter i+1) - mean(iter i), and the
red-team samples attributable to it are the JSONL rows carrying `iteration == i`.
This script trains one probe on the samples from the POSITIVE-delta iterations and one
on the samples from the NEGATIVE-delta iterations, each on the same 50-row base and
under the same recipe as the run's own retrains, and scores both on the eval splits.

Everything except WHICH successes are included is held to the arm's own configuration:
the same base probe (probe_iter0, for architecture + metadata), the same base training
data, the same preprocessing (filter + contrastive) and contrastive cache, the same
judge-confidence gate, the same held-out dev set, the same single-probe fit, the same
seed and the same message transforms. So the two halves differ only in their data.

Note the two halves are NOT the same size — that is the finding, not a flaw, and the
row counts are printed alongside the AUROCs so a size effect can be read off directly.

Usage:
  fit_delta_split_probes.py                       # all eight arms, both signs
  fit_delta_split_probes.py --arms arm7 --signs positive
"""
from __future__ import annotations
import argparse, csv, json, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_redteam.config import load_config          # noqa: E402
from agentic_redteam.retrain import retrain_probe       # noqa: E402
from agentic_redteam.evaluation import evaluate_probe   # noqa: E402

EVAL_DIR = ROOT / "eval_sets/highstakes"
OUT_PROBES = ROOT / "probes/hs_nm_deltasplit"
OUT_RES = ROOT / "results_hs_nm_deltasplit"

def _arm(label, cfg, res, stem, probes, base, log=None):
    return dict(label=label, config=ROOT / f"configs/{cfg}",
                res=ROOT / f"results_hs_gemma27b_{res}",
                stem=stem,
                csv=ROOT / f"results_hs_gemma27b_{res}/{stem.removesuffix('_probing')}_comparison.csv",
                log=log, probes=ROOT / f"probes/hs_gemma27b_{probes}",
                base=f"data/{base}")


ARMS = {
    "arm1": _arm("gpt-oss \u00b7 memo", "gptoss120b_hs_gemma27b_gptossbase_itermemo150.md",
                 "gptoss120b_gptossbase_itermemo150", "gptossbase_itermemo150_probing",
                 "gptoss120b_gptossbase_itermemo150", "highstakes_gptoss_50.jsonl"),
    "arm2": _arm("gpt-oss \u00b7 +eval-desc", "gptoss120b_hs_gemma27b_gptossbase_itermemo150_evaldesc.md",
                 "gptoss120b_gptossbase_evaldesc", "gptossbase_evaldesc_probing",
                 "gptoss120b_gptossbase_evaldesc", "highstakes_gptoss_50.jsonl"),
    "arm3": _arm("deepseek \u00b7 memo", "deepseekv4pro_hs_gemma27b_dsbase_itermemo150.md",
                 "deepseekv4pro_dsbase_itermemo150", "dsbase_itermemo150_probing",
                 "deepseekv4pro_dsbase_itermemo150", "highstakes_deepseekv4pro_50.jsonl"),
    "arm4": _arm("deepseek \u00b7 +eval-desc", "deepseekv4pro_hs_gemma27b_dsbase_itermemo150_evaldesc.md",
                 "deepseekv4pro_dsbase_evaldesc", "dsbase_evaldesc_probing",
                 "deepseekv4pro_dsbase_evaldesc", "highstakes_deepseekv4pro_50.jsonl"),
    "arm5": _arm("llama70b \u00b7 memo", "llama70b_hs_gemma27b_llamabase_itermemo150.md",
                 "llama70b_llamabase_itermemo150", "llamabase_itermemo150_probing",
                 "llama70b_llamabase_itermemo150", "highstakes_llama70b_50.jsonl"),
    "arm6": _arm("llama70b \u00b7 +eval-desc", "llama70b_hs_gemma27b_llamabase_itermemo150_evaldesc.md",
                 "llama70b_llamabase_evaldesc", "llamabase_evaldesc_probing",
                 "llama70b_llamabase_evaldesc", "highstakes_llama70b_50.jsonl"),
    # arm 7's comparison CSV was rewritten by the resumed process and covers iter4..iter10
    # only; iter0..iter3 come from the recovered pre-abort run log.
    "arm7": _arm("nemotron \u00b7 memo", "nemotron_hs_gemma27b_nemobase_itermemo150.md",
                 "nemotron_nemobase_itermemo150", "nemobase_itermemo150_probing",
                 "nemotron_nemobase_itermemo150", "highstakes_nemotron_50.jsonl",
                 log=ROOT / "logs_archive/preabort/run_hs_gemma27b_nemotron_nemobase_itermemo150.preabort.log"),
    "arm8": _arm("nemotron \u00b7 +eval-desc", "nemotron_hs_gemma27b_nemobase_itermemo150_evaldesc.md",
                 "nemotron_nemobase_evaldesc", "nemobase_evaldesc_probing",
                 "nemotron_nemobase_evaldesc", "highstakes_nemotron_50.jsonl"),
}


def probe_means(spec) -> dict[int, float]:
    """{N: eval mean of probe_iterN}, from the comparison CSV plus, where the CSV is
    short, the run log's printed eval tables."""
    means: dict[int, float] = {}
    with Path(spec["csv"]).open() as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] == "mean":
                means[int(row["round"].removeprefix("iter"))] = float(row["auroc"])
    log = spec.get("log")
    if log and Path(log).exists():
        text = Path(log).read_text(errors="replace").replace("\r", "\n")
        cur = None
        for line in text.splitlines():
            m = re.match(r"=====\s*EVALUATING iter(\d+):", line)
            if m:
                cur = int(m.group(1))
                continue
            m = re.match(r"\s*mean\s+([0-9.]+)\s", line)
            if m and cur is not None:
                means.setdefault(cur, float(m.group(1)))
                cur = None
    return means


def split_iterations(means: dict[int, float], n_iter: int = 10):
    pos, neg, deltas = [], [], {}
    for i in range(n_iter):
        if i not in means or i + 1 not in means:
            raise SystemExit(f"missing eval mean for probe_iter{i} or probe_iter{i+1}")
        d = means[i + 1] - means[i]
        deltas[i] = d
        (pos if d > 0 else neg).append(i)
    return pos, neg, deltas


def write_subset(spec, iterations: set[int], out_prefix: Path) -> tuple[list[Path], dict]:
    """Copy the rows of the chosen iterations into fresh fp/fn JSONLs.

    Whole rows, not just successes: retrain_probe applies its own success and
    judge-confidence filters, and keeping the failures makes the subset auditable."""
    paths, stats = [], {"rows": 0, "succ": 0, "succ_conf": 0}
    for et in ("fp", "fn"):
        src = Path(spec["res"]) / f"{spec['stem']}_{et}.jsonl"
        dst = out_prefix.with_name(out_prefix.name + f"_{et}.jsonl")
        n = 0
        with src.open() as fh, dst.open("w") as out:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("iteration") in iterations:
                    out.write(line)
                    n += 1
                    stats["rows"] += 1
                    if r.get("success"):
                        stats["succ"] += 1
                        if int(r.get("judge_confidence") or 0) >= 7:
                            stats["succ_conf"] += 1
        paths.append(dst)
    return paths, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    # "full" is not a half: it takes ALL ten iterations' successes, i.e. exactly the training
    # set the arm's own last retrain used to produce probe_iter10. Paired with --no-dev it is
    # the clean one-variable comparison against that probe: same data, same recipe, same seed,
    # only the validation set differs.
    ap.add_argument("--signs", nargs="+", default=["positive", "negative"],
                    choices=["positive", "negative", "full"])
    ap.add_argument("--seed", type=int, default=42)
    # VALIDATION SOURCE. Default: the arm's own held-out dev set (dev_samples/highstakes, 1908
    # rows), which is what every arm and every refit above used. --no-dev instead holds out
    # --test-size of the TRAINING data — base and red-team are split independently at the same
    # fraction by stable_train_test_split and recombined per side, so the validation set is
    # ~10% of each. That is a different probe, not the same probe differently scored: tuberlens
    # selects the best-val-AUROC epoch, so changing the validation set changes which epoch is
    # kept. Writes to its own probe dir and CSV so the dev-set fits are never overwritten.
    ap.add_argument("--no-dev", action="store_true",
                    help="ignore validation.dev_data; hold out --test-size of the training data")
    ap.add_argument("--test-size", type=float, default=0.1)
    args = ap.parse_args()
    suffix = "_nodev" if args.no_dev else ""

    OUT_PROBES.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    summary_csv = OUT_RES / f"delta_split{suffix}_comparison.csv"
    probe_dir = Path(str(OUT_PROBES) + suffix)
    probe_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        spec = ARMS[arm]
        cfg = load_config(spec["config"])
        means = probe_means(spec)
        pos, neg, deltas = split_iterations(means)
        print(f"\n{'='*78}\n{arm}  ({spec['label']})")
        for i in sorted(deltas):
            print(f"  iter {i}: {means[i]:.5f} -> {means[i+1]:.5f}  delta {deltas[i]:+.5f}"
                  f"  {'POSITIVE' if deltas[i] > 0 else 'negative'}")
        print(f"  positive iterations: {pos}\n  negative iterations: {neg}")

        base_probe = Path(spec["probes"]) / "probe_iter0.pkl"
        # The arm's own contrastive cache, COPIED so the fits here cannot mutate the
        # original run's. Both signs of one arm share the copy: it is keyed per source
        # conversation, the two subsets are disjoint, and sharing avoids paying twice
        # for any pair they happen to have in common.
        cache = OUT_RES / f"{arm}_contrastive_cache.jsonl"
        if not cache.exists():
            src_cache = Path(spec["probes"]) / "contrastive_cache.jsonl"
            shutil.copy2(src_cache, cache) if src_cache.exists() else cache.touch()

        for sign in args.signs:
            iters = set(range(10)) if sign == "full" else set(pos if sign == "positive" else neg)
            tag = f"{arm}_{sign}"
            prefix = OUT_RES / f"{tag}_probing"
            paths, stats = write_subset(spec, iters, prefix)
            probe_out = probe_dir / f"{tag}.pkl"
            val = (f"{args.test_size:.0%} of the training data"
                   if args.no_dev else "dev_samples/highstakes (1908 rows)")
            it_txt = "ALL 0-9" if sign == "full" else str(sorted(iters))
            print(f"\n--- {tag}{suffix}: iterations {it_txt} | rows {stats['rows']} "
                  f"| successes {stats['succ']} | successes at judge confidence >= "
                  f"{cfg.judge.confidence_threshold}: {stats['succ_conf']} | validation: {val}")

            result = retrain_probe(
                jsonl_path=paths,
                base_probe_path=base_probe,
                base_training_data_path=ROOT / spec["base"],
                new_probe_path=probe_out,
                layer=None,
                probe_spec=None,                       # inherit the arm's architecture
                preprocessing=cfg.preprocessing,
                contrastive_cache_path=cache,
                postprocessed_out_path=OUT_RES / f"{tag}{suffix}_postprocessed.jsonl",
                min_judge_confidence=cfg.judge.confidence_threshold,
                test_size=(args.test_size if args.no_dev else 0.2),  # non-dev: the held-out
                                                        #   fraction; dev path forces it to 0.0
                split_field=None,
                dev_data_path=(None if args.no_dev else cfg.validation.dev_data),
                seed=args.seed,
                ensemble_size=1,
                base_activation_cache_dir=cfg.output.base_activation_cache_dir,
                combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant,
                eval_data_description=cfg.eval.data_description,
                verbose=True,
            )
            print(f"    trained {probe_out.name}: {getattr(result, 'n_train', '?')} train rows")

            df = evaluate_probe(
                probe_out,
                EVAL_DIR,
                cfg.output.activations_cache_dir,
                max_samples=None,                       # eval_max_samples: 0 => full splits
                seed=args.seed,
                combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant,
            )
            print(df.to_string(index=False))
            df.insert(0, "validation", "trainsplit" if args.no_dev else "dev")
            df.insert(1, "arm", arm)
            df.insert(2, "sign", sign)
            df.insert(3, "iterations", " ".join(str(i) for i in sorted(iters)))
            df.insert(4, "redteam_successes", stats["succ_conf"])
            header = not summary_csv.exists()
            df.to_csv(summary_csv, mode="a", header=header, index=False)
            print(f"    appended to {summary_csv}")


if __name__ == "__main__":
    main()
