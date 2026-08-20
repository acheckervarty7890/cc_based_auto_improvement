"""Verify the offline harness reproduces the run's own numbers before anything is fit."""
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H

t0 = time.time()
ev = H.load_eval_splits()
print(f"eval: {sum(len(d) for d in ev.values())} rows across {len(ev)} splits "
      f"({time.time()-t0:.0f}s)", flush=True)
for k, d in ev.items():
    a = d.other_fields["activations"]
    print(f"  {k:32s} n={len(d):4d} seq={a.shape[1]:4d} hid={a.shape[2]} dtype={a.dtype} "
          f"pos={int(H.labels_of(d).sum())}")

dev = H.load_dev()
a = dev.other_fields["activations"]
print(f"dev: n={len(dev)} seq={a.shape[1]} pos={int(H.labels_of(dev).sum())}")
base = H.load_base()
a = base.other_fields["activations"]
print(f"base: n={len(base)} seq={a.shape[1]} pos={int(H.labels_of(base).sum())}")
for arm in H.ARMS:
    rt = H.load_redteam(arm)
    a = rt.other_fields["activations"]
    print(f"redteam[{arm}]: n={len(rt)} seq={a.shape[1]} pos={int(H.labels_of(rt).sum())}")
    del rt

print(f"\nloaded in {time.time()-t0:.0f}s; host RSS "
      f"{torch.cuda.memory_allocated()/2**30:.1f} GiB cuda", flush=True)

# --- reproduce the published per-split numbers for every saved probe ---
for arm, d in H.ARMS.items():
    csv = pd.read_csv(sorted((H.ROOT / f"results_instructions_gemma27b_ens10dev_{arm}").glob("*_comparison.csv"))[0])
    for it in range(6):
        p = d / f"probe_iter{it}.pkl"
        if not p.exists():
            continue
        with p.open("rb") as f:
            probe = pickle.load(f)
        got = H.score_splits(probe, ev)
        want = csv[csv["round"] == f"iter{it}"].set_index("dataset")
        bad = []
        for name, m in got.items():
            if name not in want.index:
                continue
            ref = float(want.loc[name, "auroc"])
            if abs(ref - m["auroc"]) > 1e-6:
                bad.append((name, ref, m["auroc"]))
        mm = H.macro(got)
        ref_mean = float(want.loc["mean", "auroc"]) if "mean" in want.index else float("nan")
        flag = "OK " if not bad and abs(ref_mean - mm["auroc"]) < 1e-6 else "MISMATCH"
        print(f"{flag} {arm:9s} iter{it}  macro AUROC ours={mm['auroc']:.6f} csv={ref_mean:.6f} "
              f"tpr={mm['tpr_at_fpr']:.4f}")
        for b in bad:
            print("     ", b)
print(f"\ntotal {time.time()-t0:.0f}s")
