"""A ragged-storage probe fit that is arithmetically identical to tuberlens', and fits on the card.

`get_activations` pads every row of a call to that call's longest row, capped at 1024. The
high-stakes pools are the pathological case: the mean conversation is ~370 tokens and the
longest is 1024, so a dense pool tensor is ~2.7x larger than the activations it actually
holds, and the 2323-row training pool at the top of the sweep is ~25 GB — just past a 24 GB
card. Falling back to host residency costs ~100x on the fit (see the note in
`retrain._to_device_for_fit`), and there are ~20 such fits in this analysis.

So the activations are stored **packed**: one `[total_real_tokens, dim]` tensor plus per-row
offsets, and each batch is padded to *its own* longest row when it is assembled. Both the
storage and the per-batch compute then scale with real tokens instead of with the pool's
longest conversation.

**This is not an approximation.** `LinearThenAgg.forward` masks the head's per-token output
to 0 and `LinearThenSoftmax.agg` fills masked positions with `-inf` before the softmax, so a
padded position contributes exactly `0.0` to `(x * weights).sum(dim=1)` — and adding exact
zeros to a float sum changes nothing, at any summation order. Trimming a batch to its own
width therefore returns the same numbers as padding it to 1024. `verify_fast_fit.py` checks
that end to end against the stock `ProbeFactory.build` path rather than taking the argument
on trust.

The training loop is a transcription of `PytorchAdamClassifier.train`: same AdamW args, same
cosine schedule to `final_lr`, same `BCEWithLogitsLoss`, same per-batch gradient clipping
with `gradient_accumulation_steps`-wise stepping, same per-epoch validation AUROC with
best-checkpoint retention and the same patience. Batch order is `torch.randperm` chunked by
`batch_size`, which is exactly what `DataLoader(shuffle=True)`'s `RandomSampler` draws from
the same global generator, so a given seed produces the same batches in both paths.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

# Overridable so the whole path can be exercised on the CPU (smoke tests, or a box whose
# card is busy) without touching any call site.
DEVICE = "cuda"
DTYPE = torch.bfloat16


@dataclass
class RaggedActivations:
    """Row-packed activations on one device, plus the labels they belong to."""

    packed: torch.Tensor  # [total_tokens, dim]
    offsets: torch.Tensor  # [n_rows + 1], int64, on the same device
    lengths: torch.Tensor  # [n_rows], int64
    y: torch.Tensor  # [n_rows]
    dim: int
    dtype: torch.dtype
    device: str

    @classmethod
    def from_dataset(cls, dataset, device: str | None = None,
                     dtype: torch.dtype | None = None, slab: int = 64):
        device, dtype = device or DEVICE, dtype or DTYPE
        acts = dataset.other_fields["activations"]
        mask = dataset.other_fields["attention_mask"].bool()
        lengths = mask.sum(-1).to(torch.int64)
        # Packing and unpacking both assume each row's real tokens are a PREFIX, i.e. that
        # the tokenizer right-pads. It does (checked against the published blobs), but a
        # left-padded blob would pack silently and shift every row, so it is asserted.
        prefix = torch.arange(mask.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        if not bool((prefix == mask).all()):
            raise ValueError("activations are not right-padded; ragged packing is unsafe")
        total = int(lengths.sum().item())
        dim = acts.shape[2]
        packed = torch.empty((total, dim), dtype=dtype, device=device)
        offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64)
        offsets[1:] = torch.cumsum(lengths, 0)
        for start in range(0, acts.shape[0], slab):
            stop = min(start + slab, acts.shape[0])
            a = acts[start:stop].to(device, dtype, non_blocking=True)
            m = mask[start:stop].to(device, non_blocking=True)
            sel = a[m]  # [sum(lengths in slab), dim], in row-major row order
            lo = int(offsets[start].item())
            packed[lo : lo + sel.shape[0]] = sel
            del a, m, sel
        y = torch.tensor(
            [lab.to_int() for lab in dataset.labels], dtype=dtype, device=device
        )
        return cls(packed, offsets.to(device), lengths.to(device), y, dim, dtype, device)

    @classmethod
    def from_parts(cls, parts, labels, device: str | None = None,
                   dtype: torch.dtype | None = None):
        """Pack `(source, row indices)` parts straight onto the card, with no dense pool.

        The dense intermediate is what makes the high-stakes side expensive: the full
        training fold of the ceiling CV is 3526 rows padded to 1024 tokens — 39 GB — of
        which ~14 GB is real. Streaming each source's slabs through to the packed buffer
        keeps the host-side peak at one slab.
        """
        device, dtype = device or DEVICE, dtype or DTYPE
        lengths_np = np.concatenate([src.lengths()[list(idx)] for src, idx in parts])
        lengths = torch.tensor(lengths_np, dtype=torch.int64)
        total = int(lengths.sum().item())
        dim = parts[0][0].dim
        packed = torch.empty((total, dim), dtype=dtype, device=device)
        offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64)
        offsets[1:] = torch.cumsum(lengths, 0)
        at = 0
        row = 0
        for src, idx in parts:
            for a, m, _i in src.slabs(idx):
                m = m.bool()
                n = a.shape[0]
                prefix = (
                    torch.arange(m.shape[1]).unsqueeze(0) < m.sum(-1).unsqueeze(1)
                )
                if not bool((prefix == m).all()):
                    raise ValueError(
                        "activations are not right-padded; ragged packing is unsafe"
                    )
                sel = a.to(device, dtype)[m.to(device)]
                packed[at : at + sel.shape[0]] = sel
                at += sel.shape[0]
                row += n
                del a, m, sel
        y = torch.tensor(labels, dtype=dtype, device=device)
        return cls(packed, offsets.to(device), lengths.to(device), y, dim, dtype, device)

    def __len__(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def nbytes(self) -> int:
        return self.packed.element_size() * self.packed.nelement()

    def batch(self, idx: torch.Tensor):
        """Assemble rows `idx` as a padded `(activations, mask, y)` batch of their own width."""
        lens = self.lengths[idx]
        width = int(lens.max().item())
        n = idx.shape[0]
        out = torch.zeros((n, width, self.dim), dtype=self.dtype, device=self.device)
        mask = (
            torch.arange(width, device=self.device).unsqueeze(0) < lens.unsqueeze(1)
        )
        starts = self.offsets[idx]
        # gather every token of the batch in one indexing op: for row r, position p < len(r)
        # the source token is starts[r] + p.
        pos = torch.arange(width, device=self.device).unsqueeze(0).expand(n, width)
        src = (starts.unsqueeze(1) + pos)[mask]
        out[mask] = self.packed[src]
        return out, mask, self.y[idx]


def _probs(model, ragged: RaggedActivations, batch_size: int = 32) -> np.ndarray:
    model.eval()
    out = np.empty(len(ragged), dtype=float)
    with torch.no_grad():
        for start in range(0, len(ragged), batch_size):
            idx = torch.arange(start, min(start + batch_size, len(ragged)),
                               device=ragged.device)
            a, m, _ = ragged.batch(idx)
            logits = model(a, m)
            out[start : start + idx.shape[0]] = (
                logits.float().sigmoid().cpu().numpy()
            )
    model.train()
    return out


def train_head(train: RaggedActivations, val: RaggedActivations | None,
               hyperparams: dict, *, arch, verbose: bool = False):
    """Transcription of `PytorchAdamClassifier.train` over ragged batches."""
    from sklearn.metrics import roc_auc_score

    model = arch(train.dim, **hyperparams).to(train.device).to(train.dtype)
    opt = torch.optim.AdamW(model.parameters(), **hyperparams["optimizer_args"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=hyperparams["epochs"], eta_min=hyperparams["final_lr"]
    )
    crit = torch.nn.BCEWithLogitsLoss()
    accum = hyperparams.get("gradient_accumulation_steps", 1)
    patience = hyperparams["patience"]

    val_y = val.y.float().cpu().numpy() if val is not None else None
    best_auroc, best_state, best_epoch, stale = 0.0, None, None, 0

    n = len(train)
    model.train()
    for epoch in range(hyperparams["epochs"]):
        perm = torch.randperm(n)  # matches DataLoader(shuffle=True)'s RandomSampler
        opt.zero_grad()
        for b, start in enumerate(range(0, n, hyperparams["batch_size"])):
            idx = perm[start : start + hyperparams["batch_size"]].to(train.device)
            a, m, y = train.batch(idx)
            loss = crit(model(a, m), y) / accum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (b + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            del a, m, y
        sched.step()
        if val is None:
            continue
        auroc = float(roc_auc_score(val_y, _probs(model, val)))
        if auroc > best_auroc:
            best_auroc, best_epoch, stale = auroc, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"    early stop after {epoch + 1} epochs", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_auroc": best_auroc}


def finetune_head(model, train: RaggedActivations, val: RaggedActivations,
                  hyperparams: dict, *, verbose: bool = False):
    """Continue training an existing head, keeping the better of the two stages' checkpoints.

    Mirrors `PytorchAdamClassifier.train(initialize_model=False)`: fresh optimizer and
    schedule, same everything else. Its best-checkpoint tracking restarts at AUROC 0, so a
    stage that only ever hurt would still return its least-bad epoch; since both stages
    early-stop against the *same* fixed validation set the two are directly comparable, and
    the incoming checkpoint is seeded in as the one to beat.
    """
    from sklearn.metrics import roc_auc_score

    val_y = val.y.float().cpu().numpy()
    before = float(roc_auc_score(val_y, _probs(model, val)))
    best_auroc, best_state, best_epoch, stale = before, copy.deepcopy(
        model.state_dict()
    ), None, 0

    opt = torch.optim.AdamW(model.parameters(), **hyperparams["optimizer_args"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=hyperparams["epochs"], eta_min=hyperparams["final_lr"]
    )
    crit = torch.nn.BCEWithLogitsLoss()
    accum = hyperparams.get("gradient_accumulation_steps", 1)
    n = len(train)
    model.train()
    for epoch in range(hyperparams["epochs"]):
        perm = torch.randperm(n)
        opt.zero_grad()
        for b, start in enumerate(range(0, n, hyperparams["batch_size"])):
            idx = perm[start : start + hyperparams["batch_size"]].to(train.device)
            a, m, y = train.batch(idx)
            loss = crit(model(a, m), y) / accum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (b + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            del a, m, y
        sched.step()
        auroc = float(roc_auc_score(val_y, _probs(model, val)))
        if auroc > best_auroc:
            best_auroc, best_epoch, stale = auroc, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= hyperparams["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {
        "val_auroc_stage1": before,
        "val_auroc_finetuned": best_auroc,
        "checkpoint_kept": "stage1" if best_epoch is None else "finetuned",
        "finetune_best_epoch": best_epoch,
    }


def wrap_probe(model, hyperparams: dict, *, model_name: str, layer: int,
               pos_class_label: str, neg_class_label: str, description: str,
               best_epoch=None):
    """Put a trained head into a real `PytorchProbe`.

    Everything downstream — `predict_proba`, the metric helpers, pickling — then works on it
    exactly as it works on a probe that came out of `ProbeFactory.build`. Scoring goes
    through the stock dense path, which is fine: it is chunked, and only the *fit* was
    memory-bound.
    """
    from tuberlens.probes.pytorch_classifiers import PytorchAdamClassifier
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax
    from tuberlens.probes.pytorch_probes import PytorchProbe

    classifier = PytorchAdamClassifier(
        training_args=dict(hyperparams),
        probe_architecture=LinearThenSoftmax,
        model=model,
        device=DEVICE,
        dtype=DTYPE,
    )
    classifier.best_epoch = best_epoch
    return PytorchProbe(
        hyper_params=dict(hyperparams),
        _classifier=classifier,
        model_name=model_name,
        layer=layer,
        description=description,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
    )
