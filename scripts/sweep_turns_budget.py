#!/usr/bin/env python
"""Fixed-budget sweep over round length (max_turns) for agentic red-teaming.

Holds the total attacker-turn budget T = rounds * max_turns constant across
every cell and varies how it is split, so any difference in unique successes is
attributable to *round granularity* (how often the rolling memo refreshes)
rather than to one cell simply running longer.

Each cell reuses an existing config's prompt body (via the repo's own
`_split_frontmatter`), overriding only the swept frontmatter knobs, then runs
`scripts/run_redteam.py` against ONE fixed probe (no retrain, no eval). No new
prompts are introduced. Results are parsed straight from each cell's JSONL.

The probe, error_type, and base config are CLI args. Outputs (under --out-dir):
  configs/cell_*.md              generated per-cell config (base prompts kept)
  jsonl/cell_*[_fp|_fn].jsonl    raw attempts
  jsonl/cell_*.done              per-cell completion marker (written on rc=0)
  sweep_results.csv              one row per (cell, error_type, repeat)
  sweep_per_round.csv            one row per (cell, error_type, repeat, round)

Resume: the sweep is safely restartable at CELL granularity. Re-run the SAME
command after a crash — cells with a `.done` marker are skipped (parse-only);
the cell that crashed mid-run (partial JSONL, no marker) is deleted and rerun
from scratch; unstarted cells run normally. You lose at most the in-flight
cell's work, and the fixed-budget invariant (T = rounds*max_turns per cell) is
preserved. `--force` ignores markers and reruns every cell. Result CSVs are
flushed after every cell, so a crash still leaves partial aggregates on disk.

Example:
  .venv_claude/bin/python scripts/sweep_turns_budget.py \
      --probe probes/run_099/probe_iter3.pkl \
      --base-config configs/dry_run_config.md \
      --error-type false_positive \
      --total-budget 48 --out-dir results_sweep/T48_fp
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from agentic_redteam.config import _split_frontmatter, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = REPO_ROOT / ".venv_claude" / "bin" / "python"
RUN_REDTEAM = REPO_ROOT / "scripts" / "run_redteam.py"

ERROR_TYPE_MAP = {
    "fp": ["false_positive"],
    "false_positive": ["false_positive"],
    "fn": ["false_negative"],
    "false_negative": ["false_negative"],
    "both": ["false_positive", "false_negative"],
}


def build_cell_config(
    base_fm: dict,
    body: str,
    *,
    attacker_model: str,
    judge_model: str,
    max_turns: int,
    batch_target: int,
    rounds: int,
    concurrency: int,
    sessions_per_model: int,
    interface: str,
    view_limit: int,
    probe_path: Path,
    error_type_value,
    jsonl_path: Path,
    run_id: int,
) -> str:
    """Override only the swept knobs on a copy of the base frontmatter; keep the
    base config's prompt `body` verbatim (no new prompts)."""
    fm = copy.deepcopy(base_fm)
    atk = fm.setdefault("attacker", {})
    atk["models"] = [attacker_model]
    atk["max_turns"] = max_turns
    atk["batch_target"] = batch_target
    atk["rounds"] = rounds
    atk["concurrency"] = concurrency
    atk["sessions_per_model"] = sessions_per_model
    atk["round_summaries"] = True
    atk["view_reshuffle"] = False
    atk["interface"] = interface
    atk["view_limit"] = view_limit
    fm.setdefault("judge", {})["model"] = judge_model
    fm.setdefault("probe", {})["path"] = str(probe_path)
    fm["probe"]["error_type"] = error_type_value
    # Replace output wholesale so stray base keys (comparison_csv, caches) don't leak.
    fm["output"] = {"jsonl_path": str(jsonl_path), "run_id": run_id}
    return "---\n" + yaml.safe_dump(fm, sort_keys=False) + "---\n" + body


def parse_jsonl(path: Path) -> dict:
    """Aggregate one cell/error-type JSONL into summary + per-round stats."""
    per_round: dict[int, dict[str, int]] = {}
    total = succ = 0
    if path.exists():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                slot = per_round.setdefault(
                    int(row.get("round", -1)), {"submissions": 0, "successes": 0}
                )
                slot["submissions"] += 1
                if row.get("success"):
                    succ += 1
                    slot["successes"] += 1
    return {"total": total, "successes": succ, "per_round": per_round}


def memo_lift(per_round: dict[int, dict[str, int]]) -> tuple[float, float, float]:
    """Success rate in the first vs second half of rounds, and their diff.

    A positive lift is weak evidence the rolling memo is helping later rounds.
    Returns (first_rate, second_rate, lift); NaNs when <2 rounds ran.
    """
    rounds = sorted(per_round)
    if len(rounds) < 2:
        return (float("nan"), float("nan"), float("nan"))
    mid = len(rounds) // 2

    def rate(rs):
        sub = sum(per_round[r]["submissions"] for r in rs)
        su = sum(per_round[r]["successes"] for r in rs)
        return (su / sub) if sub else float("nan")

    a, b = rate(rounds[:mid]), rate(rounds[mid:])
    return (a, b, b - a)


def write_csvs(
    out: Path, result_rows: list[dict], per_round_rows: list[dict]
) -> tuple[Path, Path]:
    """(Re)write the two result CSVs from the rows accumulated so far.

    Called after every cell (so a crash mid-sweep still leaves partial
    aggregates on disk) and once more at the end. Idempotent: it rewrites both
    files wholesale from the current row lists.
    """
    res_csv, pr_csv = out / "sweep_results.csv", out / "sweep_per_round.csv"
    if result_rows:
        with open(res_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(result_rows[0]))
            w.writeheader()
            w.writerows(result_rows)
    if per_round_rows:
        with open(pr_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_round_rows[0]))
            w.writeheader()
            w.writerows(per_round_rows)
    return res_csv, pr_csv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fixed-budget (T = rounds*max_turns) sweep over round length.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--probe", type=Path, required=True,
                   help="Path to the llama-1b (or any) probe pickle to attack.")
    p.add_argument("--base-config", type=Path,
                   default=REPO_ROOT / "configs" / "dry_run_config.md",
                   help="Existing config whose prompt body (and default knobs) "
                        "are reused. Its probe family should match --probe.")
    p.add_argument("--error-type", default="false_positive",
                   choices=sorted(ERROR_TYPE_MAP),
                   help="Target error type. 'both' runs each cell twice.")
    p.add_argument("--total-budget", type=int, default=30,
                   help="T = rounds * max_turns, held constant across cells.")
    p.add_argument("--grid", default="3,5,10,15,30",
                   help="Comma-separated max_turns values; rounds = T // max_turns.")
    p.add_argument("--attacker-model", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--judge-model", default="openai/gpt-5.1-chat")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--sessions-per-model", type=int, default=5,
                   help="Concurrent copies of the attacker model launched per "
                        "round. Needs --concurrency >= this value or copies queue.")
    p.add_argument("--interface", default="prompt", choices=["tools", "prompt"],
                   help="Attacker interface. 'prompt' (default) = classical no-tool "
                        "mode (OpenRouter only): the model emits one conversation per "
                        "turn as text and view_past_attempts is injected each turn. "
                        "'tools' = the model calls the three tools itself.")
    p.add_argument("--view-limit", type=int, default=4,
                   help="Past-attempts sample size injected each turn in prompt mode.")
    p.add_argument("--repeats", type=int, default=2,
                   help="Independent runs per cell (LLM sampling variance).")
    p.add_argument("--couple-batch-target", action="store_true",
                   help="Set batch_target = max_turns (your original coupling). "
                        "Default leaves batch_target unreachable so max_turns "
                        "binds and every cell burns the same compute.")
    p.add_argument("--batch-target", type=int, default=1_000_000,
                   help="Early-stop successes per round when NOT coupling.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory for generated configs, JSONLs, and result CSVs.")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON,
                   help="Interpreter used to run scripts/run_redteam.py.")
    p.add_argument("--force", action="store_true",
                   help="Delete an existing cell JSONL and rerun it. Default: "
                        "reuse existing JSONLs (parse-only) so you can re-aggregate.")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate configs and print the plan; do not run.")
    args = p.parse_args(argv)

    probe_path = args.probe.resolve()
    if not probe_path.exists():
        p.error(f"probe not found: {probe_path}")
    if not args.base_config.exists():
        p.error(f"base config not found: {args.base_config}")
    if not args.dry_run and not Path(args.python).exists():
        p.error(f"interpreter not found: {args.python}")
    if "OPENROUTER_API_KEY" not in os.environ and not args.dry_run:
        print("WARNING: OPENROUTER_API_KEY is not set — runs will fail.",
              file=sys.stderr)
    if args.sessions_per_model > args.concurrency:
        print(f"WARNING: sessions_per_model ({args.sessions_per_model}) > "
              f"concurrency ({args.concurrency}); copies will queue on the "
              f"semaphore instead of running in parallel.", file=sys.stderr)

    error_types = ERROR_TYPE_MAP[args.error_type]
    et_value = error_types[0] if len(error_types) == 1 else error_types
    base_fm, body = _split_frontmatter(args.base_config.read_text())
    T = args.total_budget
    max_turns_grid = [int(x) for x in args.grid.split(",") if x.strip()]

    out = args.out_dir.resolve()
    cfg_dir, jsonl_dir = out / "configs", out / "jsonl"
    for d in (cfg_dir, jsonl_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Probe:        {probe_path}")
    print(f"Base config:  {args.base_config}  (prompts reused verbatim)")
    print(f"Error types:  {error_types}")
    print(f"Attacker:     {args.attacker_model}")
    print(f"Judge:        {args.judge_model}")
    print(f"Budget T:     {T}   grid(max_turns)={max_turns_grid}   "
          f"repeats={args.repeats}   sessions/model={args.sessions_per_model}")
    print(f"Interface:    {args.interface}   view_limit={args.view_limit}")
    print(f"batch_target: {'=max_turns (coupled)' if args.couple_batch_target else args.batch_target}\n")

    result_rows: list[dict] = []
    per_round_rows: list[dict] = []
    run_times: dict[str, float] = {}  # tag -> wall seconds for that run (NaN if reused)
    run_id = 9000  # distinct namespace; bumped per cell/repeat
    sweep_t0 = time.time()

    for mt in max_turns_grid:
        rounds = max(1, T // mt)
        actual_T = rounds * mt
        batch_target = mt if args.couple_batch_target else args.batch_target
        for rep in range(args.repeats):
            run_id += 1
            tag = f"mt{mt}_r{rounds}_rep{rep}"
            jsonl_base = jsonl_dir / f"cell_{tag}.jsonl"
            cfg_path = cfg_dir / f"cell_{tag}.md"

            cfg_path.write_text(build_cell_config(
                base_fm, body,
                attacker_model=args.attacker_model, judge_model=args.judge_model,
                max_turns=mt, batch_target=batch_target, rounds=rounds,
                concurrency=args.concurrency,
                sessions_per_model=args.sessions_per_model,
                interface=args.interface, view_limit=args.view_limit,
                probe_path=probe_path,
                error_type_value=et_value, jsonl_path=jsonl_base, run_id=run_id,
            ))

            # Reuse the repo's config to resolve the exact per-error-type JSONL paths.
            cell_cfg = load_config(cfg_path)
            et_files = {et: cell_cfg.jsonl_path_for(et) for et in error_types}

            print(f"=== cell {tag}  (T={actual_T}, budget-matched={actual_T == T}) ===")
            if args.dry_run:
                print(f"    config: {cfg_path}")
                continue

            # Resume is per-CELL and gated on an explicit completion marker, not
            # on "a JSONL exists": a cell that crashed mid-run leaves a partial
            # JSONL, and reusing that would silently record truncated counts and
            # break the fixed-budget invariant. Only a `.done` marker (written on
            # clean rc=0 completion below) means the cell is safe to parse-only.
            done_marker = jsonl_dir / f"cell_{tag}.done"
            if done_marker.exists() and not args.force:
                print(f"    cell complete (.done present), parse-only "
                      f"(use --force to rerun): {done_marker}")
                elapsed = float("nan")
            else:
                # Never ran, crashed mid-run, or --force: start clean so
                # run_redteam can't dedup against a stale partial JSONL.
                for f in et_files.values():
                    f.unlink(missing_ok=True)
                Path(str(jsonl_base) + ".summaries.jsonl").unlink(missing_ok=True)
                done_marker.unlink(missing_ok=True)
                t0 = time.time()
                proc = subprocess.run(
                    [str(args.python), str(RUN_REDTEAM), str(cfg_path)],
                    cwd=str(REPO_ROOT),
                )
                elapsed = time.time() - t0
                if proc.returncode != 0:
                    # Leave no marker so this cell reruns on the next restart.
                    print(f"    run_redteam FAILED (rc={proc.returncode}); "
                          f"recording partial JSONL if any (no .done written, "
                          f"cell will rerun).", file=sys.stderr)
                else:
                    done_marker.write_text(
                        f"tag={tag} run_id={run_id} T={actual_T} "
                        f"finished={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                print(f"    took {elapsed:.1f}s")
            run_times[tag] = elapsed

            for et, f in et_files.items():
                agg = parse_jsonl(f)
                first, second, lift = memo_lift(agg["per_round"])
                rate = (agg["successes"] / agg["total"]) if agg["total"] else 0.0
                result_rows.append({
                    "cell": tag, "max_turns": mt, "rounds": rounds,
                    "total_budget": actual_T, "batch_target": batch_target,
                    "error_type": et, "repeat": rep, "run_id": run_id,
                    "attacker_model": args.attacker_model,
                    "judge_model": args.judge_model, "probe": str(probe_path),
                    "submissions": agg["total"],
                    "unique_successes": agg["successes"],
                    "success_rate": round(rate, 4),
                    "first_half_rate": round(first, 4) if first == first else "",
                    "second_half_rate": round(second, 4) if second == second else "",
                    "memo_lift": round(lift, 4) if lift == lift else "",
                    "wall_seconds": round(elapsed, 1) if elapsed == elapsed else "",
                })
                for r in sorted(agg["per_round"]):
                    slot = agg["per_round"][r]
                    per_round_rows.append({
                        "cell": tag, "max_turns": mt, "rounds": rounds,
                        "error_type": et, "repeat": rep, "round": r,
                        "submissions": slot["submissions"],
                        "successes": slot["successes"],
                        "success_rate": round(
                            slot["successes"] / slot["submissions"], 4
                        ) if slot["submissions"] else 0.0,
                    })
                print(f"    [{et}] submissions={agg['total']} "
                      f"successes={agg['successes']} rate={rate:.3f}")

            # Flush after each cell so a crash mid-sweep still leaves partial
            # aggregates on disk (the JSONLs remain the source of truth).
            write_csvs(out, result_rows, per_round_rows)

    if args.dry_run:
        print("\n(dry run — no CSVs written)")
        return 0

    res_csv, pr_csv = write_csvs(out, result_rows, per_round_rows)
    print(f"\nWrote {res_csv}\nWrote {pr_csv}")

    print("\n=== summary (best success_rate first) ===")
    hdr = f"{'cell':<20}{'et':<16}{'subs':>6}{'succ':>6}{'rate':>8}{'lift':>8}{'time':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    for row in sorted(result_rows, key=lambda r: r["success_rate"], reverse=True):
        lift = row["memo_lift"]
        secs = run_times.get(row["cell"], float("nan"))
        print(f"{row['cell']:<20}{row['error_type']:<16}"
              f"{row['submissions']:>6}{row['unique_successes']:>6}"
              f"{row['success_rate']:>8.3f}"
              f"{(f'{lift:+.3f}' if lift != '' else '   n/a'):>8}"
              f"{(f'{secs:.1f}s' if secs == secs else 'reused'):>9}")

    # Per-cell wall-time tally (summed over repeats; error-type shares one run).
    print("\n=== time per cell (summed over repeats) ===")
    thdr = f"{'cell':<14}{'runs':>5}{'total':>10}"
    print(thdr + "\n" + "-" * len(thdr))
    cell_totals: dict[str, list[float]] = {}
    for tag, secs in run_times.items():
        base = tag.rsplit("_rep", 1)[0]  # strip repeat suffix -> the max_turns/rounds cell
        cell_totals.setdefault(base, []).append(secs)
    for base in sorted(cell_totals, key=lambda b: int(b.split("_")[0][2:])):
        times = [s for s in cell_totals[base] if s == s]  # drop NaN (reused runs)
        total = sum(times)
        label = f"{total:.1f}s" if times else "reused"
        print(f"{base:<14}{len(cell_totals[base]):>5}{label:>10}")

    total_wall = time.time() - sweep_t0
    h, rem = divmod(int(total_wall), 3600)
    m, s = divmod(rem, 60)
    print(f"\nTotal sweep wall time: {total_wall:.1f}s ({h:02d}:{m:02d}:{s:02d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
