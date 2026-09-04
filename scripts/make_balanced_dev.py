#!/usr/bin/env python3
"""Cut a 500-row, doubly-balanced dev set out of dev_samples/highstakes (1908 rows).

The full dev set is class-balanced but not split-balanced: anthropic_hh is 1028 of its 1908
rows, 54% of the validation signal. This takes an equal 125 rows from each of the four
splits, class-balanced inside each, so the result is 250 high-stakes / 250 low-stakes AND
125 per split. Deterministic: rows are chosen by sha256 of their own content under a fixed
seed, so the same 500 rows come out on every machine and every run.
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dev_samples/highstakes"
OUT = ROOT / "dev_samples/highstakes_500"
PER_SPLIT, SEED = 125, "balanced500:42"
POS, NEG = "high-stakes", "low-stakes"


def key(row: str) -> str:
    return hashlib.sha256((SEED + row).encode()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tot = {POS: 0, NEG: 0}
    # 63/62 alternating so the four splits sum to exactly 250/250
    for i, path in enumerate(sorted(SRC.glob("*.jsonl"))):
        rows = [l for l in path.read_text().splitlines() if l.strip()]
        by = {POS: [], NEG: []}
        for l in rows:
            lab = json.loads(l)["labels"]
            if lab not in by:
                raise SystemExit(f"{path.name}: unexpected label {lab!r}")
            by[lab].append(l)
        n_pos = 63 if i < 2 else 62
        want = {POS: n_pos, NEG: PER_SPLIT - n_pos}
        keep = []
        for lab, n in want.items():
            pool = sorted(by[lab], key=key)
            if len(pool) < n:
                raise SystemExit(f"{path.name}: only {len(pool)} {lab} rows, need {n}")
            keep += pool[:n]
            tot[lab] += n
        keep.sort(key=key)                    # interleave the classes
        (OUT / path.name).write_text("\n".join(keep) + "\n")
        print(f"  {path.name:28} {len(rows):5d} -> {len(keep):4d}  "
              f"({want[POS]} {POS} / {want[NEG]} {NEG})")
    print(f"  {'TOTAL':28} {'':5} -> {sum(tot.values()):4d}  ({tot[POS]} {POS} / {tot[NEG]} {NEG})")

    # every kept row must exist in the source, and the set must stay disjoint from eval
    src_all = {l for p in SRC.glob("*.jsonl") for l in p.read_text().splitlines() if l.strip()}
    out_all = {l for p in OUT.glob("*.jsonl") for l in p.read_text().splitlines() if l.strip()}
    assert out_all <= src_all, "subsample contains rows not in the source dev set"
    ev = {json.loads(l)["inputs"] for p in (ROOT / "eval_sets/highstakes").glob("*.jsonl")
          for l in p.read_text().splitlines() if l.strip()}
    overlap = sum(1 for l in out_all if json.loads(l)["inputs"] in ev)
    print(f"  subset of the 1908: yes | overlap with eval_sets/highstakes: {overlap}")


if __name__ == "__main__":
    main()
