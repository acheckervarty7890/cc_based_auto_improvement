"""Build every representation the metric sweep needs, in one pass over the blobs.

``why_close_but_wrong{,_centered}.py`` measure one representation — mean-pooled layer-32
activations — under three linear rescalings. The open question they leave is whether a
*different* geometry would change their verdict: that the new-in-v3 successes sit as
close to the v2 training set as a deliberately opposite-label counterpart does, so
proximity carries no label information and cannot drive a novelty guard or an
acquisition rule.

Answering that means comparing many candidate metrics on the same rows, and the
expensive half is shared: reading 1776 per-conversation blobs (6.7 GB) off disk. So this
script does that once and caches everything a metric could be built from.

What is cached, per arm, in ``metric_reps/{arm}.npz``:

**Five poolings.** The pooling *is* part of the geometry — mean pooling averages a
decisive token over as many as 1024 positions, which is one candidate explanation for
why the label is nearly invisible in it.

    mean     mean over unpadded tokens (what the published numbers use)
    last     the final unpadded token
    last32   mean over the final 32 unpadded tokens
    probe    the deployed head's OWN pooling, sum_u p_u h_u with p = softmax(z/T)
             and z = h.w + b from probe_iter2 — a fixed feature space here, because
             w is frozen at the attacked probe rather than being refitted
    topz16   mean over the 16 tokens with the largest |z| under that same w

**The probe's own readout.** ``seq_logit`` is probe_iter2's sequence logit s (fp32, via
``attribution_lib._forward``), and ``jac`` is ds/dw — the exact derivative through the
softmax pooling. Two conversations are close in ``jac`` when they push the probe's
weights the same way, which is the natural "probe's own metric" for influence.

**The text**, rendered role-by-role, for the metrics that never look at activations
(TF-IDF, sentence embeddings).

Eval-split reps (mean and probe pooling) are cached alongside so §5's "does red-team
data live where eval lives" question can be re-asked in each candidate metric.

Usage:
    .venv_claude/bin/python scripts/metric_geometry_features.py
    .venv_claude/bin/python scripts/metric_geometry_features.py --arm gptoss120b
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import attribution_vintage as V

OUT_DIR = Path("results_hu_harm_gemma27b_batch_ablation/vintage/metric_reps")
POOLINGS = ("mean", "last", "last32", "probe", "topz16")
TOPZ_K = 16
LASTK = 32


def _pool_all(
    acts: torch.Tensor, mask: torch.Tensor, w: torch.Tensor, b: float, T: float
) -> dict[str, np.ndarray]:
    """Every pooling of one batch of conversations, plus the probe's own per-token logits.

    ``acts`` is ``[N, T, E]`` fp16 and ``mask`` ``[N, T]``; padding is on the right (the
    red-team blobs are written one row per call and so are unpadded, the eval blobs are
    padded to the split's max). Everything is computed in fp32 for the same reason
    ``attribution_lib.sequence_logits`` does — bf16 ULP at these magnitudes is ~0.1.
    """
    h = acts.float()
    m = mask.bool()
    h = h * m[..., None]
    n_tok = m.sum(-1).clamp(min=1)
    # the probe unpickles onto whatever device it was saved from, so w may be on cuda
    w = w.detach().to(h.device, torch.float32)

    out: dict[str, np.ndarray] = {}
    out["mean"] = (h.sum(-2) / n_tok[:, None]).numpy()

    # right padding, so the last real token is at index n_tok - 1
    last_idx = (n_tok - 1).clamp(min=0)
    out["last"] = h[torch.arange(h.shape[0]), last_idx].numpy()

    lastk = np.empty((h.shape[0], h.shape[-1]), dtype=np.float32)
    for i in range(h.shape[0]):
        t = int(n_tok[i])
        lastk[i] = h[i, max(0, t - LASTK) : t].mean(0).numpy()
    out["last32"] = lastk

    # the probe's own pooling, at the frozen w of probe_iter2
    z = (h @ w + b).masked_fill(~m, 0.0)
    p = torch.softmax(z.masked_fill(~m, float("-inf")) / T, dim=-1)
    out["probe"] = torch.einsum("bt,bte->be", p, h).numpy()

    topz = np.empty((h.shape[0], h.shape[-1]), dtype=np.float32)
    for i in range(h.shape[0]):
        t = int(n_tok[i])
        k = min(TOPZ_K, t)
        sel = z[i, :t].abs().topk(k).indices
        topz[i] = h[i, sel].mean(0).numpy()
    out["topz16"] = topz

    out["_n_tok"] = n_tok.numpy().astype(np.int32)
    return out


def _render(messages) -> str:
    return "\n\n".join(f"{m.role}: {m.content}" for m in messages)


def build_arm(arm: str, iteration: int = 3, device: str = "cpu") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{arm}.npz"

    print(f"\n########## {arm} ##########", flush=True)
    ds = A.load_redteam_dataset(arm, iteration)
    gen2src = V._generated_to_source(arm)
    probe2 = A.load_probe(A.ARMS[arm] / "probe_iter2.pkl")
    w, b, T = A.probe_params(probe2)
    keep, vreport = V.vintages(arm, iteration)

    n = len(ds.inputs)
    cols = {p: np.empty((n, w.numel()), dtype=np.float32) for p in POOLINGS}
    jac = np.empty((n, w.numel()), dtype=np.float32)
    seq = np.empty(n, dtype=np.float32)
    n_tok = np.empty(n, dtype=np.int32)
    src_keys, is_gen, texts = [], [], []

    t0 = time.time()
    for i, msgs in enumerate(ds.inputs):
        blob = torch.load(A.redteam_blob_path(msgs), weights_only=False)
        acts, mask = blob["activations"], blob["attention_mask"]
        pooled = _pool_all(acts, mask, w, b, T)
        for p in POOLINGS:
            cols[p][i] = pooled[p][0]
        n_tok[i] = pooled["_n_tok"][0]
        s, j = A.sequence_logits_and_jacobians(acts, mask, w, b, T, device=device, batch_size=1)
        seq[i], jac[i] = s[0], j[0]
        key = A.canon(msgs)
        src_keys.append(gen2src.get(key, key))
        is_gen.append(key in gen2src)
        texts.append(_render(msgs))
        del blob, acts, mask, pooled
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n} rows  {el:.0f}s  ({el/(i+1):.3f} s/row)", flush=True)

    y = np.array(
        [1.0 if l == "positive" else 0.0 for l in ds.other_fields["labels"]],
        dtype=np.float32,
    )
    is_val = np.array([A.is_val(m) for m in ds.inputs], dtype=bool)

    payload = {f"X_{p}": cols[p] for p in POOLINGS}
    payload.update(
        {
            "jac": jac,
            "seq_logit": seq,
            "n_tok": n_tok,
            "y": y,
            "is_gen": np.array(is_gen, dtype=bool),
            "is_val": is_val,
            "src_keys": np.array(src_keys, dtype=object),
            "texts": np.array(texts, dtype=object),
            "idx_v2": np.array(keep[2], dtype=np.int64),
            "idx_v3": np.array(keep[3], dtype=np.int64),
        }
    )
    np.savez(out_path, **payload)
    print(f"  wrote {out_path}  ({out_path.stat().st_size/1e6:.0f} MB)", flush=True)
    (OUT_DIR / f"{arm}_vintage_report.json").write_text(json.dumps(vreport, indent=1))
    del cols, jac, payload
    gc.collect()


def build_eval(device: str = "cpu") -> None:
    """Mean- and probe-pooled reps for the four eval splits.

    The probe pooling needs a ``w``; gptoss120b's ``probe_iter2`` is used for both arms
    so the eval reps are shared, and every comparison that reads them stays within one
    arm's own red-team rows anyway.
    """
    out_path = OUT_DIR / "eval.npz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe2 = A.load_probe(A.ARMS["gptoss120b"] / "probe_iter2.pkl")
    w, b, T = A.probe_params(probe2)

    from tuberlens.interfaces.dataset import LabelledDataset

    payload = {}
    for split in A.EVAL_SPLITS:
        ds = LabelledDataset.load_from(
            A.EVAL_DATASET_DIR / f"{split}.jsonl",
            pos_class_label="harmful_to_human",
            neg_class_label="not_harmful_to_human",
            combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
            convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
        )
        blob = torch.load(
            A.EVAL_ACTIVATIONS_DIR / f"{split}-acts_full.pt", weights_only=False, mmap=True
        )
        acts, mask = blob["activations"], blob["attention_mask"]
        chunks: dict[str, list[np.ndarray]] = {p: [] for p in POOLINGS}
        for i in range(0, acts.shape[0], 16):
            pooled = _pool_all(acts[i : i + 16], mask[i : i + 16], w, b, T)
            for p in POOLINGS:
                chunks[p].append(pooled[p])
        for p in POOLINGS:
            payload[f"X_{p}_{split}"] = np.concatenate(chunks[p])
        payload[f"y_{split}"] = ds.labels_torch().float().cpu().numpy()
        payload[f"texts_{split}"] = np.array([_render(m) for m in ds.inputs], dtype=object)
        print(f"  {split}: {payload[f'X_mean_{split}'].shape}", flush=True)
        del blob, acts, mask, chunks
        gc.collect()
    np.savez(out_path, **payload)
    print(f"  wrote {out_path}  ({out_path.stat().st_size/1e6:.0f} MB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=["gptoss120b", "deepseekv4pro"])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--only-eval", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    for arm in [] if args.only_eval else args.arm:
        build_arm(arm, device=args.device)
    if not args.skip_eval:
        print("\n########## eval splits ##########", flush=True)
        build_eval(device=args.device)


if __name__ == "__main__":
    main()
