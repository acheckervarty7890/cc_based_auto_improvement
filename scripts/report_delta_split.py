#!/usr/bin/env python3
"""Read results_hs_nm_deltasplit/delta_split_comparison.csv and print every arm's
delta-split probes against that arm's own reference points.

References, both from the arm's own run:
  probe_iter0   the 50-row base alone, no red-team data at all
  probe_iter10  the arm's actual endpoint, trained on ALL its successes

The arm table (paths, labels, which comparison CSV, which run log) is imported from
fit_delta_split_probes so the two scripts can never disagree about what an arm is.

  --csv  also print a machine-readable block for the artifact build.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("fds", ROOT / "scripts/fit_delta_split_probes.py")
fds = importlib.util.module_from_spec(_spec)
_argv = sys.argv[:]          # fit_delta_split_probes parses argv at import; restore ours after
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(fds)
sys.argv = _argv

CSV = ROOT / "results_hs_nm_deltasplit/delta_split_comparison.csv"
SPLITS = ["anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced"]


def arm_probe_rows(spec):
    """{N: {dataset: auroc}} for probe_iterN — CSV, topped up from the run log where short."""
    out: dict[int, dict[str, float]] = {}
    with Path(spec["csv"]).open() as fh:
        for row in csv.DictReader(fh):
            out.setdefault(int(row["round"].removeprefix("iter")), {})[row["dataset"]] = float(row["auroc"])
    log = spec.get("log")
    if log and Path(log).exists():
        import re
        text = Path(log).read_text(errors="replace").replace("\r", "\n")
        cur = None
        for line in text.splitlines():
            m = re.match(r"=====\s*EVALUATING iter(\d+):", line)
            if m:
                cur = int(m.group(1)); continue
            m = re.match(r"\s*(\S+)\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s*$", line)
            if m and cur is not None and (m.group(1) in SPLITS or m.group(1) == "mean"):
                out.setdefault(cur, {}).setdefault(m.group(1), float(m.group(2)))
                if m.group(1) == "mean":
                    cur = None
    return out


def total_successes(spec) -> int:
    n = 0
    for et in ("fp", "fn"):
        p = Path(spec["res"]) / f"{spec['stem']}_{et}.jsonl"
        with p.open() as fh:
            for line in fh:
                if line.strip() and json.loads(line).get("success"):
                    n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="also emit a JSON block for the artifact build")
    args = ap.parse_args()
    if not CSV.exists():
        raise SystemExit(f"{CSV} not written yet")

    fits: dict[tuple[str, str], dict] = {}
    meta: dict[tuple[str, str], tuple[str, int]] = {}
    with CSV.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["arm"], row["sign"])
            fits.setdefault(key, {})[row["dataset"]] = float(row["auroc"])
            meta[key] = (row["iterations"], int(row["redteam_successes"]))

    out_json = {}
    print(f"{'arm':6}{'attacker / knob':24}{'base':>9}{'positive':>10}{'negative':>10}{'full run':>10}"
          f"{'pos-neg':>10}{'best':>10}   n(pos/neg/all)")
    for arm, spec in fds.ARMS.items():
        ref = arm_probe_rows(spec)
        base = ref.get(0, {}).get("mean")
        full = ref.get(10, {}).get("mean")
        pos = fits.get((arm, "positive"), {}).get("mean")
        neg = fits.get((arm, "negative"), {}).get("mean")
        npos = meta.get((arm, "positive"), ("", 0))[1]
        nneg = meta.get((arm, "negative"), ("", 0))[1]
        nall = total_successes(spec)
        cells = [base, pos, neg, full]
        best = max(v for v in cells if v is not None)
        names = ["base", "positive", "negative", "full run"]
        best_name = names[cells.index(best)]
        f = lambda v: f"{v:>10.5f}" if v is not None else f"{'pending':>10}"
        gap = f"{pos-neg:>+10.5f}" if (pos and neg) else f"{'':>10}"
        print(f"{arm:6}{spec['label']:24}{base:>9.4f}{f(pos)}{f(neg)}{f(full)}{gap}{best_name:>10}"
              f"   {npos}/{nneg}/{nall}")
        out_json[arm] = dict(label=spec["label"], base=base, full=full, pos=pos, neg=neg,
                             n_pos=npos, n_neg=nneg, n_all=nall,
                             pos_iters=meta.get((arm, "positive"), ("", 0))[0],
                             neg_iters=meta.get((arm, "negative"), ("", 0))[0],
                             splits={s: {"base": ref.get(0, {}).get(s), "full": ref.get(10, {}).get(s),
                                         "pos": fits.get((arm, "positive"), {}).get(s),
                                         "neg": fits.get((arm, "negative"), {}).get(s)} for s in SPLITS})

    print(f"\nPER SPLIT\n{'arm':6}{'variant':10}" + "".join(f"{s[:14]:>16}" for s in SPLITS) + f"{'mean':>10}")
    for arm, spec in fds.ARMS.items():
        ref = arm_probe_rows(spec)
        rows = [("base", ref.get(0, {})), ("positive", fits.get((arm, "positive"), {})),
                ("negative", fits.get((arm, "negative"), {})), ("full run", ref.get(10, {}))]
        for name, d in rows:
            if not d:
                continue
            print(f"{arm:6}{name:10}" + "".join(f"{d.get(s, float('nan')):>16.5f}" for s in SPLITS)
                  + f"{d.get('mean', float('nan')):>10.5f}")
        print()

    if args.json:
        print("\n===JSON===")
        print(json.dumps(out_json, indent=1))


if __name__ == "__main__":
    main()
