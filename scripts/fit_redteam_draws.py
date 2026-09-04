#!/usr/bin/env python3
"""Resample each high-stakes arm's red-team successes and refit, N times, to measure how
much of a retrain's eval score is the DATA and how much is the draw.

For every arm: take `--fraction` (default 0.9) of that arm's successes at random, fit a probe
on the 50-row base plus that subsample under the arm's own recipe, score it on the four eval
splits, and repeat `--draws` times. Report mean and sd per arm.

Everything except which successes are included is the arm's own configuration: the same
probe_iter0 for architecture and metadata, the same filter + contrastive preprocessing and
contrastive cache, the same judge-confidence gate, the same seed, the same message
transforms. Preprocessing runs INSIDE each draw, as it would in a real retrain, so a draw
changes the filter's bag-of-words fit and which contrastive pairs are minted too.

Validation is `--dev-dir`, by default the 500-row doubly-balanced cut of
dev_samples/highstakes (scripts/make_balanced_dev.py): 125 rows from each of the four
splits, 250 high-stakes / 250 low-stakes. The full 1908-row set is 54% anthropic_hh, so the
smaller one is balanced in a way the larger is not — and it costs ~4x less per epoch.

Draws are seeded on (arm, draw index, --seed), so the same subsets come back on a re-run and
the grid extends without recomputing what is already there.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, random, shutil, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
_sp = importlib.util.spec_from_file_location("fds", ROOT / "scripts/fit_delta_split_probes.py")
fds = importlib.util.module_from_spec(_sp)
_argv = sys.argv[:]; sys.argv = [_argv[0]]
_sp.loader.exec_module(fds); sys.argv = _argv

from agentic_redteam.config import load_config          # noqa: E402
from agentic_redteam.retrain import retrain_probe       # noqa: E402
from agentic_redteam.evaluation import evaluate_probe   # noqa: E402

OUT_PROBES = ROOT / "probes/hs_draws"
OUT_RES = ROOT / "results_hs_draws"
EVAL_DIR = ROOT / "eval_sets/highstakes"


def draw_subset(spec, frac: float, rng: random.Random, out_prefix: Path):
    """Write `frac` of this arm's SUCCESSES to fresh fp/fn JSONLs.

    Successes only: retrain_probe ignores everything else, and writing just the sampled rows
    makes the file an exact record of what the draw trained on. The two error types are
    sampled TOGETHER (one pool, one draw) so a draw cannot accidentally rebalance fp against
    fn — that ratio is a property of the arm."""
    pool = []
    for et in ("fp", "fn"):
        path = Path(spec["res"]) / f"{spec['stem']}_{et}.jsonl"
        with path.open() as fh:
            for line in fh:
                if line.strip() and json.loads(line).get("success"):
                    pool.append((et, line))
    k = max(1, round(len(pool) * frac))
    keep = rng.sample(pool, k)
    paths = []
    for et in ("fp", "fn"):
        dst = out_prefix.with_name(out_prefix.name + f"_{et}.jsonl")
        dst.write_text("".join(l for e, l in keep if e == et))
        paths.append(dst)
    return paths, len(pool), k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(fds.ARMS), choices=list(fds.ARMS))
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-dir", default="dev_samples/highstakes_500")
    args = ap.parse_args()

    OUT_PROBES.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    dev_dir = ROOT / args.dev_dir
    if not dev_dir.is_dir():
        raise SystemExit(f"missing dev dir {dev_dir}")
    n_dev = sum(1 for p in dev_dir.glob("*.jsonl") for l in p.read_text().splitlines() if l.strip())
    csv_path = OUT_RES / "draws_comparison.csv"
    print(f"validation: {args.dev_dir} ({n_dev} rows) | {args.draws} draws x "
          f"{args.fraction:.0%} of each arm's successes | seed {args.seed}\n")

    done = set()
    if csv_path.exists():                      # resumable: skip (arm, draw) already scored
        with csv_path.open() as fh:
            for row in csv.reader(fh):
                if len(row) > 2 and row[0] in fds.ARMS:
                    done.add((row[0], row[1]))

    for arm in args.arms:
        spec = fds.ARMS[arm]
        cfg = load_config(spec["config"])
        cache = OUT_RES / f"{arm}_contrastive_cache.jsonl"
        if not cache.exists():
            src = Path(spec["probes"]) / "contrastive_cache.jsonl"
            shutil.copy2(src, cache) if src.exists() else cache.touch()
        for d in range(args.draws):
            if (arm, str(d)) in done:
                print(f"--- {arm} draw {d}: already scored, skipping")
                continue
            rng = random.Random(f"{arm}:{d}:{args.seed}")
            tag = f"{arm}_d{d}"
            paths, n_pool, n_keep = draw_subset(spec, args.fraction, rng, OUT_RES / f"{tag}_probing")
            probe_out = OUT_PROBES / f"{tag}.pkl"
            print(f"\n--- {arm} draw {d}  ({spec['label']}): {n_keep} of {n_pool} successes")
            retrain_probe(
                jsonl_path=paths,
                base_probe_path=Path(spec["probes"]) / "probe_iter0.pkl",
                base_training_data_path=ROOT / spec["base"],
                new_probe_path=probe_out,
                layer=None, probe_spec=None,
                preprocessing=cfg.preprocessing,
                contrastive_cache_path=cache,
                min_judge_confidence=cfg.judge.confidence_threshold,
                test_size=0.2, split_field=None,     # ignored: dev_data_path forces 0.0
                dev_data_path=dev_dir,
                seed=args.seed, ensemble_size=1,
                base_activation_cache_dir=cfg.output.base_activation_cache_dir,
                combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant,
                eval_data_description=cfg.eval.data_description,
                verbose=True,
            )
            df = evaluate_probe(probe_out, EVAL_DIR, cfg.output.activations_cache_dir,
                                max_samples=None, seed=args.seed,
                                combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                                convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant)
            print(df.to_string(index=False))
            df.insert(0, "arm", arm); df.insert(1, "draw", d)
            df.insert(2, "n_kept", n_keep); df.insert(3, "n_pool", n_pool)
            df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            m = float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])
            print(f"    {tag}: mean {m:.5f}  -> {csv_path.name}")

    # summary over whatever is on disk
    vals: dict[str, list[float]] = {}
    if csv_path.exists():
        with csv_path.open() as fh:
            for row in csv.reader(fh):
                if len(row) > 5 and row[0] in fds.ARMS and row[4] == "mean":
                    vals.setdefault(row[0], []).append(float(row[5]))
    if vals:
        print(f"\n{'arm':26}{'draws':>7}{'mean':>10}{'sd':>9}{'min':>10}{'max':>10}{'range':>9}")
        for a in fds.ARMS:
            v = vals.get(a, [])
            if not v:
                continue
            sd = st.stdev(v) if len(v) > 1 else float("nan")
            print(f"{fds.ARMS[a]['label']:26}{len(v):7d}{st.mean(v):10.5f}{sd:9.5f}"
                  f"{min(v):10.5f}{max(v):10.5f}{max(v)-min(v):9.5f}")


if __name__ == "__main__":
    main()
