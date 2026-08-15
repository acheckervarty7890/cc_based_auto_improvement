"""Re-run an already-recorded fit with AGENTIC_FAST_ACTS on and demand the same AUROC.

The fast path (``scripts/fast_activations.py``) claims to change only where the activation
tensors live and how they reach the GPU, never the arithmetic. That claim is cheap to make
and expensive to be wrong about: a sweep whose early fits used one path and later fits
another would compare vintages across a numerical discontinuity, which is precisely the
confound the sweep exists to remove.

So the claim is *tested*, against a fit already in the progress sidecar, on the same seed
and vintage. Exit code 0 only if every one of the seven splits matches to the last bit of
the recorded double. Anything else and the fast path must not be adopted.

Usage:
    AGENTIC_FAST_ACTS=1 .venv_claude/bin/python scripts/verify_fast_activations.py \
        --arm gptoss120b --vintage 1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A  # noqa: E402
import attribution_refit as R  # noqa: E402
import attribution_vintage as V  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(A.ARMS), default="gptoss120b")
    ap.add_argument("--vintage", type=int, default=1)
    ap.add_argument("--seed", type=int, default=A.SEED)
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--drop-long", default="pair")
    ap.add_argument("--progress", type=Path,
                    default=A.REPO / "results_instructions_gemma27b_vintage/vintage_progress.jsonl")
    args = ap.parse_args()

    recorded = None
    for line in args.progress.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r["arm"], r["vintage"], r["seed"]) == (args.arm, args.vintage, args.seed):
            recorded = r
    if recorded is None:
        print(f"no recorded fit for {args.arm} v{args.vintage} seed {args.seed}", file=sys.stderr)
        raise SystemExit(2)

    import fast_activations as F

    print(f"fast path enabled: {F.enabled()}")
    print(f"reference fit: {recorded['fit_seconds']:.0f}s, "
          f"MEAN={recorded['auroc']['mean']['pipeline']:.6f}")

    keep, _ = V.vintages(args.arm, args.iteration, args.drop_long)
    asm = V.assemble_train_only(args.arm, args.iteration)
    drop = set(range(len(asm.redteam))) - set(keep[args.vintage])

    t0 = time.time()
    probe, n_tr, n_val = R.refit(asm, drop_rows=drop, seed=args.seed)
    fit_s = time.time() - t0
    res = V.score_streaming(asm, probe, A.EVAL_ACTIVATIONS_DIR)

    assert n_tr == recorded["n_train"] and n_val == recorded["n_val"], (
        f"different training set: {n_tr}/{n_val} vs recorded "
        f"{recorded['n_train']}/{recorded['n_val']}"
    )

    print(f"\nfit {fit_s:.0f}s vs recorded {recorded['fit_seconds']:.0f}s "
          f"— speedup {recorded['fit_seconds'] / fit_s:.1f}x\n")
    ok = True
    for sp in list(A.EVAL_SPLITS) + ["mean"]:
        for scale in ("pipeline", "rank"):
            new, old = res[sp][scale], recorded["auroc"][sp][scale]
            same = new == old
            ok &= same
            if scale == "pipeline":
                print(f"  {sp:28s} {old:.10f} -> {new:.10f}  {'OK' if same else 'DIFFERS'}")
            elif not same:
                print(f"  {sp:28s} (rank) {old:.10f} -> {new:.10f}  DIFFERS")

    print()
    if ok:
        print("IDENTICAL — the fast path changes timing only. Safe to adopt.")
        raise SystemExit(0)
    print("DIFFERS — do NOT adopt; the sweep would straddle two numerical paths.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
