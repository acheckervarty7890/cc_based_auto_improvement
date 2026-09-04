#!/usr/bin/env python3
"""Compare the high-stakes delta-split refits under two VALIDATION SETS.

Same 16 refits, same training data, same seed, same architecture — only what the fit
early-stops against differs:

  dev        dev_samples/highstakes, 1908 held-out rows (what every arm and every refit
             in the artifact used)
  trainsplit 10% of the training data itself, held out by stable_train_test_split, so
             ~10% of the base and ~10% of the postprocessed red-team set

tuberlens selects the best-val-AUROC epoch, so this changes which epoch is kept and
therefore the probe. It also changes the base activation cache key (test_size 0.0 -> 0.1).
The eval splits are identical in both, so the two columns are directly comparable.
"""
from __future__ import annotations
import csv, re, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_hs_nm_deltasplit"
SPLITS = ["anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced"]
LAB = {"arm1": "gpt-oss · memo", "arm2": "gpt-oss · +eval-desc",
       "arm3": "deepseek · memo", "arm4": "deepseek · +eval-desc",
       "arm5": "llama70b · memo", "arm6": "llama70b · +eval-desc",
       "arm7": "nemotron · memo", "arm8": "nemotron · +eval-desc"}


def load(path: Path):
    """Tolerant CSV reader.

    delta_split_comparison.csv was written in two layouts: the original delta-split rows have
    (arm, sign, iterations, redteam_successes, round, dataset, auroc, ...) and the later
    full-data rows carry an extra leading `validation` column. Appending the second to the
    first left no new header, so a positional read is wrong for half the file. Locate the
    fields by CONTENT instead — the arm id matches ^arm[1-8]$, the split name is one of the
    four eval splits or "mean", and the AUROC is the field after it."""
    out: dict[tuple[str, str], dict] = {}
    n: dict[tuple[str, str], int] = {}
    if not path.exists():
        return out, n
    names = set(SPLITS) | {"mean"}
    with path.open() as fh:
        for row in csv.reader(fh):
            ai = next((i for i, v in enumerate(row) if re.fullmatch(r"arm[1-8]", v)), None)
            di = next((i for i, v in enumerate(row) if v in names), None)
            if ai is None or di is None or di + 1 >= len(row):
                continue                      # header, or a row we do not recognise
            try:
                auroc = float(row[di + 1])
            except ValueError:
                continue
            arm, sign = row[ai], row[ai + 1]
            k = (arm, sign)
            out.setdefault(k, {})[row[di]] = auroc
            try:
                n[k] = int(row[ai + 3])       # redteam_successes
            except (ValueError, IndexError):
                pass
    return out, n


def epochs(path: Path) -> list[int]:
    if not path.exists():
        return []
    t = path.read_text(errors="replace").replace("\r", "\n")
    return [int(m.group(1)) for m in re.finditer(r"Early stopping triggered after (\d+) epochs", t)]


def main() -> None:
    dev, nd = load(RES / "delta_split_comparison.csv")
    tsp, _ = load(RES / "delta_split_nodev_comparison.csv")
    halves = [h for h in ("full", "positive", "negative")
              if any((a, h) in dev or (a, h) in tsp for a in LAB)]
    order = [(a, h) for a in LAB for h in halves]

    # the arm's own published endpoint, for reference only — it comes from the iterative CLI,
    # not from this script, and the arm1 control measured that cross-pipeline gap at -0.0095.
    iter10 = {}
    import importlib.util, sys as _sys
    _sp = importlib.util.spec_from_file_location("fds", ROOT / "scripts/fit_delta_split_probes.py")
    _m = importlib.util.module_from_spec(_sp); _argv = _sys.argv[:]; _sys.argv = [_argv[0]]
    _sp.loader.exec_module(_m); _sys.argv = _argv
    for a in LAB:
        iter10[a] = _m.probe_means(_m.ARMS[a])[10]

    print(f"{'arm':26}{'half':10}{'n':>5}{'dev':>10}{'trainsplit':>12}{'diff':>10}{'probe_iter10':>14}")
    diffs, fdiffs = [], []
    for k in order:
        d, t = dev.get(k, {}).get("mean"), tsp.get(k, {}).get("mean")
        if d is None and t is None:
            continue
        dc = f"{d:10.5f}" if d is not None else f"{'pending':>10}"
        tc = f"{t:12.5f}" if t is not None else f"{'pending':>12}"
        gap = f"{t-d:+10.5f}" if (t is not None and d is not None) else " " * 10
        if t is not None and d is not None:
            diffs.append(t - d)
            if k[1] == "full":
                fdiffs.append((k[0], d, t, t - d))
        ref = f"{iter10[k[0]]:14.5f}" if k[1] == "full" else " " * 14
        print(f"{LAB[k[0]]:26}{k[1]:10}{nd.get(k,0):5d}{dc}{tc}{gap}{ref}")

    if not diffs:
        print("\n(no trainsplit rows yet)")
        return
    if fdiffs:
        print(f"\nFULL-DATA FITS ONLY — the iteration-10 training set, {len(fdiffs)} arms")
        w = sum(1 for _, d, t, g in fdiffs if g > 0)
        gs = [g for *_, g in fdiffs]
        print(f"  trainsplit higher in {w}/{len(fdiffs)}   "
              f"mean {st.mean(gs):+.5f}   median {st.median(gs):+.5f}   "
              f"range {min(gs):+.5f} .. {max(gs):+.5f}")
        if len(gs) > 1:
            print(f"  sd {st.stdev(gs):.5f}   |diff|>0.01 in {sum(abs(g)>0.01 for g in gs)}/{len(gs)}")

    print(f"\ntrainsplit − dev over {len(diffs)} refits:")
    sd = f"   sd {st.stdev(diffs):.5f}" if len(diffs) > 1 else ""
    print(f"  mean {st.mean(diffs):+.5f}   median {st.median(diffs):+.5f}{sd}"
          f"   range {min(diffs):+.5f} .. {max(diffs):+.5f}")
    print(f"  trainsplit higher in {sum(d>0 for d in diffs)}/{len(diffs)}")
    print(f"  |diff| > 0.01 in {sum(abs(d)>0.01 for d in diffs)}/{len(diffs)}   "
          f"> 0.02 in {sum(abs(d)>0.02 for d in diffs)}/{len(diffs)}")

    # does the finding survive?
    print("\nDoes the positive half still beat the negative half?")
    for src, tbl in (("dev", dev), ("trainsplit", tsp)):
        ok = tot = 0
        gaps = []
        for a in LAB:
            p, n2 = tbl.get((a, "positive"), {}).get("mean"), tbl.get((a, "negative"), {}).get("mean")
            if p is None or n2 is None:
                continue
            tot += 1
            ok += p > n2
            gaps.append(p - n2)
        if tot:
            print(f"  {src:11} {ok}/{tot} = {100*ok/tot:3.0f}%   "
                  f"mean gap {st.mean(gaps):+.4f}   range {min(gaps):+.4f} .. {max(gaps):+.4f}")

    e1, e2 = epochs(ROOT / "logs/fit_delta_split.log"), epochs(ROOT / "logs/fit_delta_split_nodev.log")
    if e1 and e2:
        print(f"\nEarly stopping — epochs before the fit stopped:")
        print(f"  dev        (1908-row validation) mean {st.mean(e1):5.1f}  range {min(e1)}-{max(e1)}  n={len(e1)}")
        print(f"  trainsplit (~30-row validation)  mean {st.mean(e2):5.1f}  range {min(e2)}-{max(e2)}  n={len(e2)}")

    print(f"\nPER SPLIT · trainsplit − dev")
    print(f"{'arm':26}{'half':10}" + "".join(f"{s.replace('_balanced',''):>16}" for s in SPLITS))
    for k in order:
        if k not in tsp or k not in dev:
            continue
        print(f"{LAB[k[0]]:26}{k[1]:10}" +
              "".join(f"{tsp[k][s]-dev[k][s]:>+16.4f}" for s in SPLITS))


if __name__ == "__main__":
    main()
