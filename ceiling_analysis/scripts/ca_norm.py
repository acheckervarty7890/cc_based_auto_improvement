"""A normalization step in front of the probe head, and nothing else changed.

`LinearThenSoftmax` reads the layer-32 residual stream **raw**: `nn.Linear(5376, 1)`
straight onto the activation, then a temperature-5 softmax over positions. Every probe in
this repo's experiments was trained that way. This module inserts one normalization step
between the activation and that linear, so the ceiling can be re-measured under each
choice with everything else — the harness, the seeds, the fold assignment, the reserved dev
validation slice, all seven hyperparameters — held fixed.

**Why the raw stream is not obviously the right input.** Measured over the real tokens of
`eval_ant_hh`: per-token L2 norm 94.4 (p5 79, p95 103) and per-token RMS across the 5376
dims 1.29, so the *overall* scale is already near unity and this is not a "the inputs are
huge" problem. The heterogeneity is across features: mean per-feature std 0.79 against a
max of 32.9. A handful of gemma's massive-activation dims carry ~40x the spread of a
typical one, and a single linear layer under one global learning rate sees them all through
the same step size.

That measurement is also what makes the variants below distinguishable rather than
redundant, and it is why they are separated into affine and non-affine forms:

* `layernorm_noaffine` — per token, subtract the mean over the 5376 dims and divide by
  their std. No parameters. Note what this does *not* fix: the per-token statistics are
  themselves dominated by the massive-activation dims, so this rescales each token by
  roughly the inverse of its outliers' magnitude — a per-token gain that varies with
  content, not a per-feature reweighting.
* `layernorm` — the same, plus a learnable per-feature scale and shift. Those 2 x 5376
  parameters *can* undo the feature heterogeneity, but they are learned under the same
  `lr=5e-3`, `weight_decay=1e-3` and global grad-norm clip as the linear.
* `rmsnorm` — divide by the per-token RMS with no mean subtraction, learnable scale. The
  mean-subtraction-free half of layernorm, which is the half that matters if the useful
  signal sits in the per-token *direction* rather than in its offset.
* `standardize` — the per-**feature** answer: subtract the mean and divide by the std of
  each of the 5376 dims, computed once over the training fold's real tokens and frozen.
  This is the only variant that equalizes the feature scales the measurement above
  actually found, and the only one whose statistics are data-dependent (see `fit_norm`).

**All five preserve the harness's exactness guarantees**, which is not automatic and is why
no variant here normalizes across positions or across the batch:

* *Padding still contributes exactly nothing.* `LinearThenAgg.forward` masks the per-token
  output to 0 and `LinearThenSoftmax.agg` fills masked positions with `-inf` before the
  softmax, so whatever a norm does to a padded position is discarded downstream. (It does
  do something: layernorm of an all-zero vector is `bias`, not zero.) A norm that pooled
  over positions would break this, and with it `ca_fit`'s ragged packing.
* *A batch's width cannot change its scores.* Every normalization here is per token, so a
  row's output is a function of that row alone.
* *`none` is bit-identical to the unnormalized baseline.* `nn.Identity` adds no parameters,
  and — the part that has to be checked rather than assumed — the norm is constructed
  **after** `super().__init__` has drawn `self.linear`'s init, and none of LayerNorm /
  RMSNorm / the standardizer draws from the RNG at all (they init to ones and zeros). So
  the global RNG stream reaching the DataLoader is untouched and the control reproduces the
  existing `ceiling_<concept>.json` exactly. `run_ceiling_norm.py --norm none` is that
  check.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tuberlens.probes.pytorch_modules import LinearThenSoftmax

KINDS = ("none", "layernorm", "layernorm_noaffine", "rmsnorm", "standardize")


class Standardize(nn.Module):
    """Freeze per-feature mean/std of the training activations. Fit, not learned.

    Buffers are kept in float32 and cast at use. The head runs in bf16, whose ~3 significant
    digits are plenty for the *result* but not for accumulating a mean over ~10^6 tokens, and
    a bf16 reciprocal of a small std would quantize the very features this is meant to lift.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(embed_dim, dtype=torch.float32))
        self.register_buffer("inv_std", torch.ones(embed_dim, dtype=torch.float32))
        self.fitted = False

    @torch.no_grad()
    def fit(self, packed: torch.Tensor, eps: float = 1e-5) -> None:
        """`packed` is `[total_real_tokens, embed_dim]` — real tokens only, no padding.

        The buffers are **assigned**, not `copy_`d into. `train_head` builds the head and
        then calls `.to(dtype)`, and `nn.Module.to` casts floating-point *buffers* as well
        as parameters — so by the time this runs the fp32 buffers registered in `__init__`
        are already bf16, and copying into them would quantize the statistics after all.
        Assigning a tensor to a registered buffer name replaces the entry in `_buffers`,
        keeping the name a buffer and the dtype fp32.
        """
        x = packed.to(torch.float32)
        self.mean = x.mean(0)
        self.inv_std = 1.0 / (x.std(0) + eps)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) * self.inv_std.to(x.dtype)


def _make_norm(kind: str, embed_dim: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "layernorm":
        return nn.LayerNorm(embed_dim)
    if kind == "layernorm_noaffine":
        return nn.LayerNorm(embed_dim, elementwise_affine=False)
    if kind == "rmsnorm":
        return nn.RMSNorm(embed_dim)
    if kind == "standardize":
        return Standardize(embed_dim)
    raise ValueError(f"unknown normalization {kind!r}; expected one of {KINDS}")


class NormedLinearThenSoftmax(LinearThenSoftmax):
    """`LinearThenSoftmax` with `self.norm` applied to the activations first.

    Subclasses of this are what `train_head` instantiates; `NORM` is set by `arch_for`.
    """

    NORM = "none"

    def __init__(self, embed_dim: int, **kwargs):
        # `super().__init__` draws `self.linear`'s init from the global RNG. Constructing the
        # norm afterwards keeps that draw at the same point in the stream, which is what
        # makes `--norm none` reproduce the baseline bit for bit.
        super().__init__(embed_dim, **kwargs)
        self.norm = _make_norm(self.NORM, embed_dim)

    def fit_norm(self, packed: torch.Tensor) -> None:
        """Hook `ca_fit.train_head` calls once, with the training fold's packed tokens."""
        if isinstance(self.norm, Standardize):
            self.norm.fit(packed)

    def forward(self, x, mask):
        return super().forward(self.norm(x), mask)


_CACHE: dict[str, type] = {}


def arch_for(kind: str) -> type:
    """The head class for one normalization kind. Cached so `isinstance` stays meaningful."""
    if kind not in KINDS:
        raise ValueError(f"unknown normalization {kind!r}; expected one of {KINDS}")
    if kind not in _CACHE:
        _CACHE[kind] = type(
            f"LinearThenSoftmax_{kind}", (NormedLinearThenSoftmax,), {"NORM": kind}
        )
    return _CACHE[kind]
