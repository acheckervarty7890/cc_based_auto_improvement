#!/usr/bin/env python
"""Pool the oig_omission dev and eval splits and re-cut them 30 / 70.

The shipped split gives dev only 32 rows. AUROC over its 16x16 label pairs is a count
over 256, so it moves in steps of 0.0039 and cannot express anything finer — which is
what made the targeted run's one acceptance (a 10-pair dev move) fail to transfer to
eval. Pooling both splits and re-cutting 30/70 buys dev 44 rows (22x22 = 484 pairs,
step 0.0021) at the cost of 12 eval rows.

TWO PROPERTIES OF THIS DATA GOVERN HOW THE CUT IS MADE.

**Split by source, never by row.** The concept is omission: the same multi-part request
appears once answered in full and once with a part dropped, and the two rows share the
split's ``original_text`` provenance column. All 146 pooled rows sit in 73 such couples.
Assigning rows independently would put a conversation in dev whose twin — same source,
opposite label — sits in eval, so the fit would early-stop and accept batches against a
near-copy of a test row. Whole couples move together, which is what the shipped split
does: it shares zero sources between dev and eval, and this preserves that (asserted
below, not assumed). See ``_group_key`` for why the first user turn is the wrong key.

**Stratify, or the class balance drifts.** The pool is exactly 73/73 and every source is
one row of each class, so taking whole couples keeps both sides balanced by construction;
the singleton strata below stay in place for a file that lacks the column.

Assignment is content-addressed rather than RNG-shuffled: a group's side is
``sha256(f"{namespace}:{seed}:{source}")``, groups are ordered by that digest and a
prefix is taken. Deterministic across machines and Python versions, exact in its counts,
and independent of the order the files happen to be read in.

    .venv_claude/bin/python scripts/resplit_oig_omission.py [--dev-fraction 0.30] [--seed 42]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_DEV = REPO / "dev_samples/oig_omission/oig_omission.jsonl"
SRC_EVAL = REPO / "eval_sets/oig_omission/oig_omission.jsonl"
OUT_DEV = REPO / "dev_samples/oig_omission_mixed/oig_omission.jsonl"
OUT_EVAL = REPO / "eval_sets/oig_omission_mixed/oig_omission.jsonl"
POS = "assistant_follows_the_instruction"
NAMESPACE = "oig-omission-mix"


GROUP_FIELD = "original_text"


def _group_key(row: dict) -> str:
    """What must not be split across dev and eval.

    ``original_text`` is the split's own provenance column and couples ALL 146 pooled rows
    into 73 sources of two — one reply complete, one omitting. An earlier version of this
    script grouped on the first user turn instead, which recovers only 49 of those couples
    and leaves 48 rows looking like singletons; that cut put **14 sources on both sides**,
    so a dev row's twin — same source, opposite label — sat in eval. Group on the column
    when it exists and fall back to the user turn only when it does not.
    """
    if GROUP_FIELD in row and row[GROUP_FIELD]:
        return str(row[GROUP_FIELD])
    for m in json.loads(row["inputs"]):
        if m["role"] == "user":
            return m["content"]
    return row["inputs"]


def _digest(seed: int, key: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}:{seed}:{key}".encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-fraction", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 0.0 < args.dev_fraction < 1.0:
        ap.error("--dev-fraction must be in (0, 1)")

    pool = [json.loads(l) for l in SRC_DEV.read_text().splitlines() if l.strip()]
    n_dev_src = len(pool)
    pool += [json.loads(l) for l in SRC_EVAL.read_text().splitlines() if l.strip()]
    print(f"pooled {n_dev_src} dev + {len(pool) - n_dev_src} eval = {len(pool)} rows")

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        groups[_group_key(r)].append(r)

    # three strata: true pairs (self-balancing), then singletons by label
    strata: dict[str, list[str]] = {"pair": [], "single_pos": [], "single_neg": []}
    for key, rows in groups.items():
        labels = {r["labels"] for r in rows}
        if len(labels) == 2:
            strata["pair"].append(key)
        elif POS in labels:
            strata["single_pos"].append(key)
        else:
            strata["single_neg"].append(key)

    dev_keys: set[str] = set()
    for name, keys in strata.items():
        ordered = sorted(keys, key=lambda k: _digest(args.seed, k))
        take = math.ceil(len(ordered) * args.dev_fraction)
        dev_keys.update(ordered[:take])
        print(f"  stratum {name:11s} {len(ordered):>3} groups -> {take} to dev")

    dev_rows = [r for k, rs in groups.items() if k in dev_keys for r in rs]
    eval_rows = [r for k, rs in groups.items() if k not in dev_keys for r in rs]

    # Invariants. Each of these would be invisible in any downstream AUROC.
    assert len(dev_rows) + len(eval_rows) == len(pool), "rows lost or duplicated"
    dk = {_group_key(r) for r in dev_rows}
    ek = {_group_key(r) for r in eval_rows}
    shared = dk & ek
    assert not shared, f"{len(shared)} source(s) land in BOTH sides — couples were split"
    dc = collections.Counter(r["labels"] for r in dev_rows)
    ec = collections.Counter(r["labels"] for r in eval_rows)

    for path, rows in ((OUT_DEV, dev_rows), (OUT_EVAL, eval_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    frac = len(dev_rows) / len(pool)
    print(
        f"\ndev  {len(dev_rows):>3} rows ({frac:.1%})  "
        f"{dc[POS]} pos / {sum(dc.values()) - dc[POS]} neg  "
        f"{len(dk)} sources  -> {OUT_DEV.relative_to(REPO)}"
    )
    print(
        f"eval {len(eval_rows):>3} rows ({1 - frac:.1%})  "
        f"{ec[POS]} pos / {sum(ec.values()) - ec[POS]} neg  "
        f"{len(ek)} sources  -> {OUT_EVAL.relative_to(REPO)}"
    )
    n_pairs = (len(dev_rows) // 2) ** 2 if dc[POS] == sum(dc.values()) - dc[POS] else 0
    if n_pairs:
        print(
            f"\ndev AUROC granularity: {dc[POS]} x {sum(dc.values()) - dc[POS]} = "
            f"{dc[POS] * (sum(dc.values()) - dc[POS])} label pairs, "
            f"step {1 / (dc[POS] * (sum(dc.values()) - dc[POS])):.4f} "
            f"(was 1/256 = 0.0039 on the shipped 32-row dev set)"
        )
    print("\nSources shared between the new dev and eval: 0 (asserted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
