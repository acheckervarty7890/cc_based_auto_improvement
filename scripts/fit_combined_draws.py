#!/usr/bin/env python3
"""COMBINATION analysis: pool the red-team successes of all four attackers within one
configuration, refit, and resample — the same 90%/80% draw grid fit_redteam_draws.py runs
per arm, but with the arm axis collapsed.

THE QUESTION. The per-arm grid answers "does showing the eval description to the attacker
help *this* attacker?", and the answer was attacker-dependent (gpt-oss +4.75 sd, the other
three nothing). This script asks the pooled version: if a practitioner ran all four
attackers under one configuration and trained on everything they found, would the
configuration itself show up? Three configurations, one probe each per draw:

  combo_memo  = arms 1,3,5,7   (rolling + cross-iteration memos only)
  combo_desc  = arms 2,4,6,8   (+ eval.data_description, reaching the attacker only
                                through the judge's memos and the scope check)
  combo_att   = arms 9-12      (+ attacker.show_eval_data_description: the same paragraph
                                verbatim in the attacker's own system prompt)

WHAT IS HELD FIXED. The base training data is the UNION of the four arms' 50-row bases —
they are pairwise disjoint (verified: 0 overlap, 100 high / 100 low), so the union is
exactly the base data those four attackers collectively had, and it is the same 200 rows
for all three combos. The base probe (architecture + metadata template only; retrain_probe
does not warm-start weights) is arm1's probe_iter0 for all three. The dev set, the eval
splits, the judge-confidence gate, the seed, the message transforms and the filter +
contrastive recipe are the same as in the per-arm grid.

WHAT NECESSARILY VARIES. Each combo keeps its own group's config, because
`eval.data_description` reaches `generate_contrastive_dataset`'s prompt and is folded into
the contrastive cache key — so combo_memo's pairs are minted without it and combo_desc /
combo_att's with it. That is the arms' own recipe, and it is the same asymmetry the
per-arm grid already carries between arm1 and arm2. combo_desc and combo_att have
byte-identical retrain recipes (verified: the `preprocessing:` block is identical across
all twelve configs and the `data_description` text across all eight that carry it), so
those two differ in their DATA alone.

Everything else follows fit_redteam_draws.py: the two error types are drawn together as
one pool, draws are seeded on (combo, draw, --seed) so a re-run extends the grid instead
of recomputing it, and each --fraction gets its own CSV and probe dir.
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

from agentic_redteam.config import load_config              # noqa: E402
from agentic_redteam.retrain import retrain_probe, train_initial_probe   # noqa: E402
from agentic_redteam.evaluation import evaluate_probe       # noqa: E402

OUT_PROBES = ROOT / "probes/hs_combined_draws"
OUT_RES = ROOT / "results_hs_combined_draws"
EVAL_DIR = ROOT / "eval_sets/highstakes"
BASE = "data/highstakes_combined_200.jsonl"          # union of the four disjoint 50-row bases
TEMPLATE_PROBE = Path(fds.ARMS["arm1"]["probes"]) / "probe_iter0.pkl"

COMBOS = {
    "combo_memo": dict(label="ALL 4 · memo", arms=["arm1", "arm3", "arm5", "arm7"],
                       config=ROOT / "configs/gptoss120b_hs_gemma27b_gptossbase_itermemo150.md"),
    "combo_desc": dict(label="ALL 4 · +eval-desc", arms=["arm2", "arm4", "arm6", "arm8"],
                       config=ROOT / "configs/gptoss120b_hs_gemma27b_gptossbase_itermemo150_evaldesc.md"),
    "combo_att": dict(label="ALL 4 · +eval-desc→attacker", arms=["arm10", "arm11", "arm12", "arm9"],
                      config=ROOT / "configs/gptoss120b_hs_gemma27b_gptossbase_evaldesc_attacker.md"),
}


def seed_contrastive_cache(combo: str, arms: list[str], dst: Path) -> int:
    """Concatenate the member arms' contrastive caches into this combo's cache.

    Both the runs' own caches and the per-arm draws caches, since between them they cover
    nearly every pool member. Keys are sha256(source messages + target label + guidance
    fingerprint) and _load_cache is last-row-wins, so concatenating cannot corrupt an
    entry — a duplicated key just resolves to the later copy. Only pairs that no member
    arm ever minted (a record the pooled filter keeps but every per-arm filter dropped)
    are regenerated."""
    if dst.exists():
        return sum(1 for _ in dst.open())
    n = 0
    with dst.open("w") as out:
        for a in arms:
            for src in (Path(fds.ARMS[a]["probes"]) / "contrastive_cache.jsonl",
                        ROOT / f"results_hs_draws/{a}_contrastive_cache.jsonl"):
                if src.exists():
                    for line in src.open():
                        if line.strip():
                            out.write(line if line.endswith("\n") else line + "\n")
                            n += 1
    return n


def draw_subset(arms: list[str], frac: float, rng: random.Random, out_prefix: Path):
    """Write `frac` of the POOLED successes of all member arms to fresh fp/fn JSONLs.

    One pool over both error types AND all four attackers, drawn once: a draw must not be
    able to rebalance fp against fn, nor one attacker against another — those ratios are
    properties of the configuration, which is the thing under test."""
    pool, per_arm = [], {}
    for a in arms:
        spec = fds.ARMS[a]
        n = 0
        for et in ("fp", "fn"):
            path = Path(spec["res"]) / f"{spec['stem']}_{et}.jsonl"
            with path.open() as fh:
                for line in fh:
                    if line.strip() and json.loads(line).get("success"):
                        pool.append((et, line))
                        n += 1
        per_arm[a] = n
    k = max(1, round(len(pool) * frac))
    keep = rng.sample(pool, k)
    paths = []
    for et in ("fp", "fn"):
        dst = out_prefix.with_name(out_prefix.name + f"_{et}.jsonl")
        dst.write_text("".join(l for e, l in keep if e == et))
        paths.append(dst)
    return paths, len(pool), k, per_arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", nargs="+", default=list(COMBOS), choices=list(COMBOS))
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-dir", default="dev_samples/highstakes_500")
    ap.add_argument("--base-only", action="store_true",
                    help="also fit the 200-row base ALONE (no red-team data) as the reference "
                         "line every combo is read against; written as combo 'base' draw 0")
    args = ap.parse_args()

    OUT_PROBES.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    dev_dir = ROOT / args.dev_dir
    if not dev_dir.is_dir():
        raise SystemExit(f"missing dev dir {dev_dir}")
    if not (ROOT / BASE).exists():
        raise SystemExit(f"missing combined base {BASE}")
    n_dev = sum(1 for p in dev_dir.glob("*.jsonl") for l in p.read_text().splitlines() if l.strip())
    # Same rule as fit_redteam_draws.py: the resume key is (combo, draw) and carries no
    # fraction, so each fraction needs its own CSV and probe dir.
    suffix = "" if abs(args.fraction - 0.9) < 1e-9 else f"_f{round(args.fraction * 100)}"
    probe_dir = Path(str(OUT_PROBES) + suffix)
    probe_dir.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_RES / f"combined_draws{suffix}.csv"
    print(f"combination analysis | base {BASE} (200 rows) | validation {args.dev_dir} "
          f"({n_dev} rows)\n  {args.draws} draws x {args.fraction:.0%} of each configuration's "
          f"POOLED successes | seed {args.seed}\n  -> {csv_path.name} , {probe_dir.name}/\n")

    done = set()
    if csv_path.exists():
        with csv_path.open() as fh:
            for row in csv.reader(fh):
                if len(row) > 2 and (row[0] in COMBOS or row[0] == "base"):
                    done.add((row[0], row[1]))

    def score(probe_out: Path, cfg, tag: str, combo: str, d: int, n_keep: int, n_pool: int):
        df = evaluate_probe(probe_out, EVAL_DIR, cfg.output.activations_cache_dir,
                            max_samples=None, seed=args.seed,
                            combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                            convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant)
        print(df.to_string(index=False))
        df.insert(0, "combo", combo); df.insert(1, "draw", d)
        df.insert(2, "n_kept", n_keep); df.insert(3, "n_pool", n_pool)
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        m = float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])
        print(f"    {tag}: mean {m:.5f}  -> {csv_path.name}")

    if args.base_only and ("base", "0") not in done:
        cfg = load_config(COMBOS["combo_memo"]["config"])
        probe_out = probe_dir / "base_only.pkl"
        print("\n--- base only: 200 base rows, no red-team data")
        train_initial_probe(
            base_training_data_path=ROOT / BASE,
            model_name=cfg.probe.model, layer=cfg.probe.layer,
            new_probe_path=probe_out,
            pos_class_label=cfg.probe.pos_class_label,
            neg_class_label=cfg.probe.neg_class_label,
            probe_description=cfg.probe.description,
            probe_spec=cfg.probe.architecture,
            test_size=0.2, split_field=None, dev_data_path=dev_dir,
            seed=args.seed, ensemble_size=1,
            base_activation_cache_dir=cfg.output.base_activation_cache_dir,
            combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
            convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant,
            verbose=True,
        )
        score(probe_out, cfg, "base_only", "base", 0, 0, 0)

    for combo in args.combos:
        spec = COMBOS[combo]
        cfg = load_config(spec["config"])
        cache = OUT_RES / f"{combo}_contrastive_cache.jsonl"
        n_seed = seed_contrastive_cache(combo, spec["arms"], cache)
        print(f"\n### {combo} ({spec['label']}): arms {'+'.join(spec['arms'])}, "
              f"contrastive cache {n_seed} rows")
        for d in range(args.draws):
            if (combo, str(d)) in done:
                print(f"--- {combo} draw {d}: already scored, skipping")
                continue
            rng = random.Random(f"{combo}:{d}:{args.seed}")
            tag = f"{combo}_d{d}{suffix}"
            paths, n_pool, n_keep, per_arm = draw_subset(
                spec["arms"], args.fraction, rng, OUT_RES / f"{tag}_probing")
            probe_out = probe_dir / f"{tag}.pkl"
            print(f"\n--- {combo} draw {d}  ({spec['label']}): {n_keep} of {n_pool} "
                  f"pooled successes  {per_arm}")
            retrain_probe(
                jsonl_path=paths,
                base_probe_path=TEMPLATE_PROBE,
                base_training_data_path=ROOT / BASE,
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
            score(probe_out, cfg, tag, combo, d, n_keep, n_pool)

    vals: dict[str, list[float]] = {}
    if csv_path.exists():
        with csv_path.open() as fh:
            for row in csv.reader(fh):
                if len(row) > 5 and (row[0] in COMBOS or row[0] == "base") and row[4] == "mean":
                    vals.setdefault(row[0], []).append(float(row[5]))
    if vals:
        print(f"\n{'configuration':30}{'draws':>7}{'mean':>10}{'sd':>9}{'min':>10}{'max':>10}{'range':>9}")
        for c in ["base"] + list(COMBOS):
            v = vals.get(c, [])
            if not v:
                continue
            sd = st.stdev(v) if len(v) > 1 else float("nan")
            lbl = "200-row base only" if c == "base" else COMBOS[c]["label"]
            print(f"{lbl:30}{len(v):7d}{st.mean(v):10.5f}{sd:9.5f}"
                  f"{min(v):10.5f}{max(v):10.5f}{max(v)-min(v):9.5f}")


if __name__ == "__main__":
    main()
