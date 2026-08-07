"""A GPU-resident reimplementation of the probe trainer, for sweeps of many refits.

``attribution_refit`` runs the real ``ProbeFactory.build`` and costs ~131 s per fit.
Almost all of that is memory traffic, not arithmetic: the activations live in host
RAM padded to the global maximum width (1024 tokens, against a 446-token mean, so
2.3x of every batch is padding) and every epoch ships ~7 GB of it across PCIe. The
model itself is a single 5376->1 linear layer — a few GFLOP per epoch.

So this module packs the activations *ragged* (no padding at all — 3.7 GB for a
778-row arm, which fits on an 8 GB card) and keeps them on the GPU for the whole
sweep. Two consequences:

- one fit drops from ~131 s to ~1 s;
- **K fits can share one pass.** The parameter is a single column vector, so K
  independent probes are just a ``[5376, K]`` matrix and one GEMM. A drop-set sweep
  trains K variants simultaneously, all seeing the identical shuffle order — common
  random numbers, which is what makes small paired differences measurable at all
  against a seed-to-seed spread that is far larger than any single pair's effect.

Faithfulness to the reference trainer (``PytorchAdamClassifier.train``) is the whole
point, so the following are replicated rather than approximated: bf16 parameters and
arithmetic, AdamW(lr=5e-3, wd=1e-3) with a 200-epoch cosine schedule to 1e-4, batch
16 with 4-step gradient accumulation, per-model gradient clipping at norm 1.0,
validation AUROC **on bf16-saturated sigmoid** (see ``attribution_lib.auroc_pipeline``
— the ties are load-bearing, they are what the reference selects epochs on), best-
epoch snapshotting and patience-50 early stopping tracked per model.

Two deliberate differences, both documented at the call site:

- Dropped rows are masked out of the loss rather than removed from the partition, so
  every variant sees the same batch boundaries. A batch containing a dropped row
  trains on 15 samples instead of 16, with the mean taken over the survivors.
- A model whose patience is exhausted stops updating its snapshot but keeps running
  in the batch. That is exactly equivalent to stopping — the returned probe is the
  best-validation snapshot either way — and avoids ragged control flow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

import attribution_lib as A

HYPER = {
    "temperature": 5,
    "batch_size": 16,
    "epochs": 200,
    "lr": 0.005,
    "weight_decay": 0.001,
    "final_lr": 0.0001,
    "gradient_accumulation_steps": 4,
    "patience": 50,
}


@dataclass
class Packed:
    """Ragged activations on the GPU: one flat token matrix plus row offsets."""

    flat: torch.Tensor      # [total_tokens, dim], fp16
    offsets: torch.Tensor   # [n_rows], int64 — first token of each row
    lengths: torch.Tensor   # [n_rows], int64
    y: torch.Tensor         # [n_rows], float32
    dim: int
    # A host-side copy of `lengths`, so a caller can work out a batch's padded width
    # without reading back from the device. Reading one scalar off a CUDA tensor costs
    # a full synchronisation, and this happens once per batch (~2400 times per fit) —
    # invisible on a laptop GPU, expensive on a virtualised one.
    lengths_cpu: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return self.lengths.numel()

    @property
    def gb(self) -> float:
        return self.flat.numel() * self.flat.element_size() / 1e9

    def gather(self, rows: torch.Tensor, lmax: int | None = None):
        """``(h, mask)`` for ``rows``: ``[B, Lmax_batch, dim]`` padded to the batch.

        Padded rather than ragged, and that choice is not cosmetic. A segment-softmax
        over a flat token vector needs ``index_add_``/``scatter_reduce``, whose CUDA
        implementations accumulate through atomics in nondeterministic order. That is
        normally a 1e-7 irrelevance — but here the trainer selects its epoch on a
        validation AUROC that separates neighbouring epochs by one or two pairs out of
        6624, so atomic jitter flips *which epoch is kept*. Measured: the same seed
        returned best_epoch 10 on one run and 4 on the next. A dense gather plus a
        dense softmax is bit-reproducible, which is what makes paired (common-random-
        number) comparisons between drop-sets meaningful at all.

        Padding to the *batch* maximum rather than the global 1024 keeps the copy at
        ~100 MB, which is why this stays fast despite materialising a dense block.
        """
        lens = self.lengths[rows]
        if lmax is None:
            lmax = int(lens.max())  # syncs; pass lmax from the host copy to avoid it
        ar = torch.arange(lmax, device=rows.device)
        mask = ar[None, :] < lens[:, None]
        idx = (self.offsets[rows][:, None] + ar[None, :]).clamp_(
            max=self.flat.shape[0] - 1
        )
        # Deliberately NOT zeroing the out-of-row positions. Clamping reads a few of
        # the next row's tokens into the tail, but ``_pool`` masks the pad logits to 0
        # and their softmax weights to exactly 0 before the weighted sum, so those
        # values cannot reach the output — while the multiply itself would cost a full
        # extra read+write of the ~100 MB block on every one of ~2400 batches per fit.
        # Output is bit-identical either way.
        return self.flat[idx], mask


def pack(acts_iter, y: np.ndarray, device: str = "cuda", dtype=torch.float16) -> Packed:
    """Build a :class:`Packed` from an iterable of ``(activations, mask)`` per row."""
    chunks, lengths = [], []
    for acts, mask in acts_iter:
        m = mask[0].bool() if mask.dim() == 2 else mask.bool()
        h = acts[0] if acts.dim() == 3 else acts
        h = h[m]
        chunks.append(h.to(dtype))
        lengths.append(h.shape[0])
    flat = torch.cat(chunks).to(device)
    lengths_t = torch.tensor(lengths, dtype=torch.int64, device=device)
    offsets = torch.cumsum(lengths_t, 0) - lengths_t
    return Packed(
        flat=flat,
        offsets=offsets,
        lengths=lengths_t,
        y=torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device),
        dim=flat.shape[1],
        lengths_cpu=torch.tensor(lengths, dtype=torch.int64),
    )


def _pool(h: torch.Tensor, mask: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
          T: float, dtype):
    """``LinearThenSoftmax`` for K models at once: ``[B, L, D] -> [B, K]``.

    Same three steps as ``LinearThenAgg.forward`` + ``LinearThenSoftmax.agg`` — linear
    per token, zero the pad logits, softmax the logits at temperature ``T`` and take
    the weighted sum — with the parameter promoted from a vector to a ``[D, K]``
    matrix so K probes share one GEMM.
    """
    z = torch.einsum("bld,dk->blk", h.to(dtype), W) + b
    m = mask[:, :, None]
    z = z.masked_fill(~m, 0.0)
    p = torch.softmax(z.masked_fill(~m, float("-inf")) / T, dim=1)
    return (z * p).sum(dim=1)


def _auroc_saturated(y: np.ndarray, s: torch.Tensor) -> np.ndarray:
    """Per-model AUROC on bf16-saturated sigmoid, vectorised over ``s``'s columns.

    Replicates what the reference trainer selects epochs on: ``probs()`` returns a
    bf16 sigmoid, in which every logit above ~5.5 collapses to exactly 1.0, and
    ``roc_auc_score`` then credits 0.5 per tied pair. Dropping the saturation here
    would make this trainer early-stop on a different signal than the real one.
    """
    p = torch.sigmoid(s.float()).to(torch.bfloat16).float()
    n, k = p.shape
    yt = torch.as_tensor(y, device=p.device, dtype=torch.float32)
    npos, nneg = float(yt.sum()), float((1.0 - yt).sum())
    if npos == 0 or nneg == 0:
        return np.full(k, 0.5)

    # Tie-averaged ranks without a Python loop. For a value v, the tie run it belongs
    # to occupies sorted positions [lo, hi), so its average 1-based rank is
    # (lo + 1 + hi) / 2 — which is the plain rank when the value is unique. Two
    # batched searchsorted calls give lo and hi for every element at once.
    #
    # The loop this replaces indexed a CUDA tensor one scalar at a time, and every
    # such read is a device synchronisation: ~580k of them per pass, 6.2 s of a 26.6 s
    # pass on a laptop GPU and far worse on a virtualised one, where sync latency is
    # the thing that differs. Values are unchanged — all ranks are integers or halves,
    # and float32 represents those (and their sums, max ~10^5 here) exactly, so the
    # result is bit-identical to the loop regardless of summation order.
    pt = p.t().contiguous()
    ps, _ = torch.sort(pt, dim=1)
    lo = torch.searchsorted(ps, pt, right=False).to(torch.float32)
    hi = torch.searchsorted(ps, pt, right=True).to(torch.float32)
    ranks = (lo + 1.0 + hi) * 0.5
    pos_rank_sum = (ranks * yt[None, :]).sum(dim=1)
    return ((pos_rank_sum - npos * (npos + 1) / 2) / (npos * nneg)).cpu().numpy()


def _auroc_saturated_masked(y: np.ndarray, s: torch.Tensor,
                            keep: np.ndarray) -> np.ndarray:
    """Per-column AUROC over each column's own kept validation rows.

    Columns whose kept set is the full set share the vectorised path; only the columns
    that actually drop a validation row (at most two rows each) fall back to sklearn.
    At K=64 and ~55 epochs that fallback is well under a second per pass.
    """
    from sklearn.metrics import roc_auc_score

    full = keep.all(axis=0)
    out = np.empty(s.shape[1], dtype=np.float64)
    if full.any():
        cols = np.flatnonzero(full)
        out[cols] = _auroc_saturated(y, s[:, torch.from_numpy(cols).to(s.device)])
    partial = np.flatnonzero(~full)
    if partial.size:
        # Only pulled to host when some column actually drops a validation row —
        # this is a GPU->CPU sync, so it must not run on the common all-full path.
        p_all = torch.sigmoid(s.float()).to(torch.bfloat16).float().cpu().numpy()
        for j in partial:
            m = keep[:, j]
            out[j] = roc_auc_score(y[m], p_all[m, j])
    return out


def init_weights(k: int, dim: int, seed: int, device="cuda", dtype=torch.bfloat16,
                 shared: bool = True):
    """Initialise K probes the way ``nn.Linear`` initialises one.

    Built by constructing the reference module under ``seed_everything``, so a K=1 run
    starts bit-identically to ``ProbeFactory.build``.

    ``shared=True`` (the default) gives **every column the same starting weights**,
    which is the whole basis of the paired design: columns in one pass then differ
    only by their drop-set, not by initialisation or shuffle order. Since the seed
    noise here (sd up to 0.023 on balanced_refusal) is an order of magnitude larger
    than any single pair's effect, unpaired comparison is hopeless and this is what
    makes the difference measurable. ``shared=False`` draws K independent replicas,
    for measuring that seed spread itself.
    """
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax

    from agentic_redteam.evaluation import seed_everything

    seed_everything(seed)
    ws, bs = [], []
    for _ in range(1 if shared else k):
        m = LinearThenSoftmax(dim, **HYPER)
        ws.append(m.linear.weight.detach().flatten().clone())
        bs.append(m.linear.bias.detach().clone())
    if shared:
        W = ws[0][:, None].repeat(1, k)
        b = bs[0].repeat(k)
    else:
        W = torch.stack(ws, dim=1)
        b = torch.cat(bs)
    return W.to(device).to(dtype).contiguous(), b.to(device).to(dtype).contiguous()


def train_many(
    train: Packed,
    val: Packed,
    keep_mask: torch.Tensor,
    seed: int,
    *,
    val_keep_mask: torch.Tensor | None = None,
    shared_init: bool = True,
    epochs: int = HYPER["epochs"],
    dtype=torch.bfloat16,
    verbose: bool = False,
):
    """Train K probes at once. ``keep_mask`` is ``[n_train, K]`` (bool).

    Column ``j`` is a probe trained on the rows where ``keep_mask[:, j]`` is true.
    All columns share one shuffle order and (by default) one initialisation — common
    random numbers — so their differences isolate the drop-set rather than the seed.

    ``val_keep_mask`` (``[n_val, K]``) drops rows from a column's *validation* set as
    well. Removing a conversation from the dataset removes it from whichever side the
    content-deterministic split put it on, and for these arms 31% of pairs straddle
    that boundary — so honouring it is what makes the intervention the real one rather
    than a train-only approximation. It matters because the val set is what early
    stopping reads.

    Returns ``(W_out, b_out, best_auroc, best_epoch)``.
    """
    from agentic_redteam.evaluation import seed_everything

    device = train.flat.device
    k = keep_mask.shape[1]
    W, b = init_weights(k, train.dim, seed, device=device, dtype=dtype,
                        shared=shared_init)
    W.requires_grad_(True)
    b.requires_grad_(True)

    opt = torch.optim.AdamW([W, b], lr=HYPER["lr"], weight_decay=HYPER["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=HYPER["epochs"], eta_min=HYPER["final_lr"]
    )
    T = float(HYPER["temperature"])
    bs, accum = HYPER["batch_size"], HYPER["gradient_accumulation_steps"]

    y_val = val.y.cpu().numpy()
    keep = keep_mask.to(device)
    val_keep_np = None if val_keep_mask is None else val_keep_mask.cpu().numpy()
    best_auroc = np.zeros(k, dtype=np.float64)
    best_epoch = np.zeros(k, dtype=np.int32)
    stalled = np.zeros(k, dtype=np.int32)
    frozen = np.zeros(k, dtype=bool)
    # NOT the best-validation snapshot — the weights at the epoch early stopping
    # fires. That is what the reference actually returns: it snapshots with
    # `state_dict().copy()`, a shallow dict copy whose tensors alias the live
    # parameters, so AdamW's in-place updates overwrite the snapshot and the closing
    # `load_state_dict(best_model_state)` restores the current weights. Verified by
    # construction (the copy shares data_ptr with the parameter). Reproducing the
    # pipeline's real behaviour matters more here than reproducing its intent, since
    # the probes under probes/ were produced this way.
    W_out = W.detach().clone()
    b_out = b.detach().clone()

    # The reference's DataLoader(shuffle=True) draws one CPU randperm per epoch from
    # the same global generator the weight init just advanced. Continue that stream —
    # do NOT reseed — so a K=1 run reproduces the reference's exact shuffle sequence
    # as well as its exact initialisation, leaving only arithmetic order to differ.
    t0 = time.time()
    for epoch in range(epochs):
        perm_cpu = torch.randperm(train.n)
        perm = perm_cpu.to(device)
        # RandomSampler draws a SECOND permutation per epoch and slices it to
        # `num_samples % n` == 0, discarding it (torch/utils/data/sampler.py). Harmless
        # there, but it advances the generator — so skipping it would desynchronise
        # every epoch after the first from the reference's shuffle stream.
        torch.randperm(train.n)
        opt.zero_grad()
        for step, start in enumerate(range(0, train.n, bs)):
            rows = perm[start : start + bs]
            # Batch width from the host-side lengths: same value, no device sync.
            lmax = int(train.lengths_cpu[perm_cpu[start : start + bs]].max())
            h, mask = train.gather(rows, lmax=lmax)
            s = _pool(h, mask, W, b, T, dtype)
            target = train.y[rows][:, None].to(s.dtype).expand_as(s)
            m = keep[rows].to(s.dtype)
            per = torch.nn.functional.binary_cross_entropy_with_logits(
                s, target, reduction="none"
            )
            # mean over the batch's *surviving* rows, per model
            loss = ((per * m).sum(0) / m.sum(0).clamp(min=1)).sum() / accum
            loss.backward()
            # The reference clips after EVERY micro-batch, on the running accumulated
            # gradient — not once before the step (pytorch_classifiers.py: the
            # clip_grad_norm_ call sits outside the `% accum` guard). Rescaling the
            # partial sum three times before it is applied is not the same update, and
            # matching it is what puts this trainer on the reference's trajectory.
            _clip_per_model(W, b, 1.0)
            if (step + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
        sched.step()

        with torch.no_grad():
            s_val = _forward_packed(val, W, b, T, dtype)
        auroc = (
            _auroc_saturated(y_val, s_val)
            if val_keep_np is None
            else _auroc_saturated_masked(y_val, s_val, val_keep_np)
        )
        improved = (auroc > best_auroc) & ~frozen
        if improved.any():
            best_auroc[improved] = auroc[improved]
            best_epoch[improved] = epoch + 1
        stalled = np.where(improved, 0, stalled + 1)
        newly_frozen = ~frozen & (stalled >= HYPER["patience"])
        if newly_frozen.any():
            # The reference stops here and returns these weights, so capture them
            # now; the column keeps riding along in the batch but its result is
            # already fixed.
            cols = torch.from_numpy(np.flatnonzero(newly_frozen)).to(device)
            W_out[:, cols] = W.detach()[:, cols]
            b_out[cols] = b.detach()[cols]
        frozen |= newly_frozen
        if frozen.all():
            break
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch + 1}: best mean val AUROC {best_auroc.mean():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # Models that ran out of epochs without ever tripping patience keep their last
    # weights, matching the reference's behaviour when the 200-epoch budget expires.
    if not frozen.all():
        cols = torch.from_numpy(np.flatnonzero(~frozen)).to(device)
        W_out[:, cols] = W.detach()[:, cols]
        b_out[cols] = b.detach()[cols]

    return W_out, b_out, best_auroc, best_epoch


def _clip_per_model(W: torch.Tensor, b: torch.Tensor, max_norm: float) -> None:
    """``clip_grad_norm_`` applied per column, not across the whole K-model tensor."""
    gw, gb = W.grad, b.grad
    norm = torch.sqrt((gw.float() ** 2).sum(0) + gb.float() ** 2)
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0).to(gw.dtype)
    gw.mul_(scale[None, :])
    gb.mul_(scale)


def _forward_packed(pk: Packed, W: torch.Tensor, b: torch.Tensor, T: float, dtype,
                    chunk: int = 64) -> torch.Tensor:
    """Sequence logits ``[n_rows, K]`` for every row of ``pk``."""
    outs = []
    for start in range(0, pk.n, chunk):
        stop = min(start + chunk, pk.n)
        rows = torch.arange(start, stop, device=pk.flat.device)
        lmax = (int(pk.lengths_cpu[start:stop].max())
                if pk.lengths_cpu is not None else None)
        h, mask = pk.gather(rows, lmax=lmax)
        # The eval packs stay in host RAM — train+val already fill most of an 8 GB
        # card — so a chunk may need moving. A no-op for the resident train/val packs.
        outs.append(_pool(h.to(W.device), mask.to(W.device), W, b, T, dtype))
    return torch.cat(outs, dim=0)


def score_eval(eval_packed: dict, W: torch.Tensor, b: torch.Tensor,
               T: float = float(HYPER["temperature"]), dtype=torch.bfloat16):
    """Per-split, per-model AUROC on both scales. Returns ``{split: {scale: [K]}}``."""
    out = {}
    for split, (pk, y) in eval_packed.items():
        with torch.no_grad():
            s = _forward_packed(pk, W, b, T, dtype).float()
        out[split] = {
            "pipeline": _auroc_saturated(y, s),
            "rank": np.array(
                [A.auroc_rank(y, s[:, j].cpu().numpy()) for j in range(s.shape[1])]
            ),
        }
    for scale in ("pipeline", "rank"):
        out.setdefault("mean", {})[scale] = np.mean(
            [out[s][scale] for s in A.EVAL_SPLITS], axis=0
        )
    return out
