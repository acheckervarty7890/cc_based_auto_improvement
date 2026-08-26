"""A probe head that reads out k positional segments instead of one pooled score.

Motivation. tuberlens' ``LinearThenSoftmax`` scores every token, softmax-weights those scores
by themselves, and sums:  ``out = sum_t softmax(z/T)_t * z_t``. At this project's temperature
(5) and score scale that behaves nearly like a max -- measured on a real conversation, one
token of 179 carried 93% of the weight and 101% of the output. So the head answers "how strong
is the single most probe-positive moment", and *where* that moment sits is discarded.

For a concept like "the reply dropped one of the questions asked", position is not incidental:
the couples this project collects differ from their partners mainly in the FINAL third of the
reply (cosine distance 0.0046 / 0.0197 / 0.0361 across thirds). An earlier attempt to expose
that by mean-pooling the *inputs* into k segments failed, and the reason was diagnosable: it
also destroyed the head's ability to select, replacing a near-max over ~180 tokens with a flat
average, which is a much bigger loss than the positional gain.

This head keeps the selection and adds the position. Steps 1-4 are unchanged -- per-token
scores, masked, softmax-weighted. Step 5 is split: each token's contribution ``z_t * w_t`` is
routed to one of k buckets by where it sits in the VALID span, giving k partial sums, and a
final ``Linear(k -> 1)`` learns what to do with them. The logit goes to
``BCEWithLogitsLoss`` / ``probs()`` exactly as before, so the sigmoid is applied downstream and
must not be applied here.

Two properties worth keeping:

* **It strictly generalizes the old head.** The buckets partition the same sum, so with the
  final layer at weight 1 / bias 0 the output is ``LinearThenSoftmax``'s to the bit -- for any
  k. That is the initialization, so training starts from the old probe's function and can only
  depart from it deliberately. ``verify_segmented_head_identity`` asserts this.
* **k=1 is the old head plus two scalars.** Useful as a control: it isolates what the extra
  affine buys from what the segmentation buys.

Hyperparameters (passed through ``ProbeSpec.hyperparams`` alongside the usual training args):
  n_segments      k, the number of positional buckets (default 2).
  segment_softmax False (default) = one softmax over the whole sequence, then split the sum --
                  buckets compete for weight, so a bucket holding no high-scoring token
                  contributes ~0. True = renormalize within each bucket, so every bucket
                  reports its own local near-max regardless of the others. The first is the
                  literal split of the existing computation; the second gives quiet regions a
                  voice, at the cost of no longer being a partition of the old output.
"""

from typing import Any

import torch
from torch import nn


class LinearThenSegmentedSoftmax(nn.Module):
    """Per-token linear scores, softmax-weighted, summed into k positional buckets, then Linear(k->1)."""

    def __init__(self, embed_dim: int, **kwargs: Any):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        self.kwargs = kwargs
        k = int(kwargs.get("n_segments", 2))
        if k < 1:
            raise ValueError(f"n_segments must be >= 1, got {k}")
        self.n_segments = k
        self.segment_softmax = bool(kwargs.get("segment_softmax", False))
        self.head = nn.Linear(k, 1)
        # Start as LinearThenSoftmax: the buckets partition its sum, so weight 1 / bias 0
        # reproduces it exactly at any k. Training departs from there on purpose, not by
        # accident, and a k-segment probe is never worse than the old one at init.
        with torch.no_grad():
            self.head.weight.fill_(1.0)
            self.head.bias.zero_()

    def segment_ids(self, mask: torch.Tensor) -> torch.Tensor:
        """Bucket index per position: token t of a length-L span goes to floor(t*k/L).

        Computed from the mask, so buckets divide the REAL tokens and padding is excluded --
        the same span the pooling study measured over. Padding is parked in bucket 0, where it
        contributes exactly 0 because its score was zeroed and its weight is 0.
        """
        k = self.n_segments
        lens = mask.sum(dim=1).clamp(min=1).long()
        pos = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        seg = (pos * k) // lens.unsqueeze(1)
        return seg.clamp(max=k - 1).masked_fill(~mask, 0).long()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:  # (batch, seq, embed), (batch, seq) -> (batch,)
        temperature = self.kwargs["temperature"]
        z = self.linear(x).squeeze(-1)                       # (B, S) per-token scores
        z = z.masked_fill(~mask, 0)                          # padding contributes nothing
        seg = self.segment_ids(mask)                         # (B, S) in [0, k-1]

        if self.segment_softmax:
            # One softmax per bucket: a position competes only with its own segment, so each
            # bucket reports its local near-max. Buckets are disjoint and cover the span, so
            # building the weights bucket-by-bucket and adding is exact.
            w = torch.zeros_like(z, dtype=torch.float32)
            zf = z.float()
            for j in range(self.n_segments):
                m = mask & (seg == j)
                if not bool(m.any()):
                    continue
                wj = torch.softmax(zf.masked_fill(~m, float("-inf")) / temperature, dim=1)
                # rows with no token in this bucket produce NaN from an all -inf row
                w = w + torch.nan_to_num(wj, nan=0.0)
            w = w.to(z.dtype)
        else:
            w = torch.softmax(
                z.masked_fill(~mask, float("-inf")).float() / temperature, dim=1
            ).to(z.dtype)

        contrib = z * w                                      # (B, S)
        parts = torch.zeros(
            z.shape[0], self.n_segments, dtype=contrib.dtype, device=contrib.device
        ).scatter_add_(1, seg, contrib)                      # (B, k) partial sums
        return self.head(parts).squeeze(-1)                  # (B,) logit


def verify_segmented_head_identity(embed_dim: int = 64, seq: int = 40, batch: int = 6,
                                   ks=(1, 2, 3, 5), seed: int = 0) -> None:
    """Assert that, at initialization, this head reproduces LinearThenSoftmax for every k.

    The buckets partition one sum and the final layer starts at weight 1 / bias 0, so the two
    must agree to floating-point error. If they ever stop agreeing, the segmentation is
    dropping or double-counting contributions.
    """
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax

    torch.manual_seed(seed)
    x = torch.randn(batch, seq, embed_dim)
    lens = torch.randint(3, seq + 1, (batch,))
    mask = torch.arange(seq)[None, :] < lens[:, None]
    ref = LinearThenSoftmax(embed_dim, temperature=5)
    for k in ks:
        m = LinearThenSegmentedSoftmax(embed_dim, temperature=5, n_segments=k)
        m.linear.load_state_dict(ref.linear.state_dict())
        a, b = ref(x, mask), m(x, mask)
        d = (a - b).abs().max().item()
        assert d < 1e-4, f"k={k}: segmented head differs from LinearThenSoftmax by {d}"
        print(f"  k={k}: matches LinearThenSoftmax at init (max diff {d:.2e})")


if __name__ == "__main__":
    verify_segmented_head_identity()
    print("segmented head verified")


class _ReLUSquared(nn.Module):
    """``relu(x)**2`` -- the one activation here that SHARPENS rather than flattens the pooling.

    Every other choice compresses the score range and so spreads the softmax; squaring widens
    the gap between the top token and its runners-up, making selection more max-like. Worth
    testing precisely because it pushes the opposite way; the risk is saturating into a hard
    argmax that starves every other position of gradient.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.relu(x)
        return r * r


_CHANNEL_ACTIVATIONS = {
    "identity": nn.Identity,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "tanh": nn.Tanh,
    "relu2": _ReLUSquared,
}


class MultiChannelLinearThenSoftmax(nn.Module):
    """``p`` parallel scoring channels, each softmax-pooled over tokens, then ``Linear(p -> 1)``.

    Where ``LinearThenSoftmax`` projects each token to ONE score and pools it, this projects to
    ``p`` scores and pools each independently. Because the pooling weights are derived from the
    scores themselves, each channel ends up attending to a different token: channel j asks its
    own question of the conversation, finds the position that answers it most strongly, and
    reports that. ``Linear(p -> 1)`` then decides how the p answers combine.

    The motivation is the same measurement that motivated the segmented head, read the other
    way. The stock head is nearly a max -- one token of 179 carried 93% of the weight -- so the
    probe's entire verdict rests on a single position and a single direction in activation
    space. That is a severe bottleneck for a concept like "the reply answered two of the three
    questions", where the evidence is plausibly several places at once: where each answer began,
    and where the third should have. Segmenting the sum forced coverage by position; this
    instead lets the model learn ``p`` things to look for and look for each of them everywhere.

    Shapes, against the stock head:

        step 1   Linear(embed -> p)          (B, S, p)      stock: (B, S, 1)
        step 3   mask                        (B, S, p)      padding zeroed, broadcast over p
        step 4   softmax over S, PER CHANNEL (B, S, p)      stock: one distribution, not p
        step 5   (z * w).sum(dim=1)          (B, p)         stock: (B,)
        step 6   Linear(p -> 1)              (B,)           logit; sigmoid stays downstream

    At ``p = 1`` with the final layer at weight 1 / bias 0 this is ``LinearThenSoftmax``
    exactly, which ``verify_multichannel_head_identity`` asserts -- so p is a strict widening
    of the existing probe and p=1 is the control that separates "more channels" from "one extra
    affine".

    Hyperparameters (via ``ProbeSpec.hyperparams``):
      n_channels            p, the number of parallel scoring channels (default 4).
      channel_activation    pointwise nonlinearity on the per-token channel scores, applied
                            between step 1 and the pooling: identity (default, the original
                            head), gelu, silu, relu, leaky_relu, tanh, relu2. With anything but
                            identity this becomes a one-hidden-layer MLP over the token vector
                            whose p hidden units are pooled SEPARATELY and combined afterwards
                            -- the mirror image of ``MLPThenSoftmax``, which combines first and
                            pools once. Given that pooling is nearly a max, that ordering is
                            the whole point: here each hidden unit may select its own token.
      input_norm            normalization applied to the activation vector BEFORE step 1:
                            none (default), layernorm (learnable affine, 2*embed params) or
                            layernorm_noaffine (0 params). Normalizing over the embedding axis
                            makes every token's vector unit-scale, which changes what the head
                            sees in two ways worth separating: it removes the ACTIVATION-NORM
                            signal (a token's magnitude, which in a residual stream tends to
                            grow with depth and salience, can no longer contribute), and it
                            standardizes the scale of the per-token scores, which is the scale
                            the softmax temperature is measured against -- so it interacts with
                            `temperature` rather than being orthogonal to it, and both should
                            move together when either is tuned.
      channel_weights_from  which scores the softmax weights are computed from: "activated"
                            (default, the literal reading) or "raw" (pre-activation).
                            This matters more than the activation choice. The weights are
                            ``softmax(z/T)`` of the scores themselves, so squashing the
                            NEGATIVE tail -- which gelu/relu/silu/elu all do -- replaces a
                            denominator term of ``exp(-8/5) = 0.20`` with ``exp(0) = 1`` at
                            every low-scoring position. The denominator drifts toward the
                            sequence length, the top token's share falls, and the pooling moves
                            from near-max toward near-mean over the valid span -- which this
                            project measured to be costly (segment-mean-pooled inputs 0.611 vs
                            0.714 full-sequence). Taking the weights from the raw linear scores
                            decouples WHERE to look (linear, spread preserved, selection as
                            sharp as the stock head) from WHAT to report (nonlinear). Monotone
                            activations do not move the argmax at all under "raw", so the
                            nonlinearity then buys only a reshaped report -- e.g. tanh bounds
                            each channel to +-1, turning the readout into a weighted vote no
                            single outlier token can dominate.
    """

    def __init__(self, embed_dim: int, **kwargs: Any):
        super().__init__()
        p = int(kwargs.get("n_channels", 4))
        if p < 1:
            raise ValueError(f"n_channels must be >= 1, got {p}")
        self.n_channels = p
        self.kwargs = kwargs
        name = str(kwargs.get("channel_activation", "identity")).lower()
        if name not in _CHANNEL_ACTIVATIONS:
            raise ValueError(
                f"channel_activation must be one of {sorted(_CHANNEL_ACTIVATIONS)}, got {name!r}"
            )
        self.channel_activation = name
        self.act = _CHANNEL_ACTIVATIONS[name]()
        wf = str(kwargs.get("channel_weights_from", "activated")).lower()
        if wf not in ("activated", "raw"):
            raise ValueError(f"channel_weights_from must be activated|raw, got {wf!r}")
        self.channel_weights_from = wf
        norm = str(kwargs.get("input_norm", "none")).lower()
        if norm not in ("none", "layernorm", "layernorm_noaffine"):
            raise ValueError(
                f"input_norm must be none|layernorm|layernorm_noaffine, got {norm!r}"
            )
        self.input_norm = norm
        self.norm = (
            nn.Identity() if norm == "none"
            else nn.LayerNorm(embed_dim, elementwise_affine=(norm == "layernorm"))
        )
        self.linear = nn.Linear(embed_dim, p)
        self.head = nn.Linear(p, 1)
        # At p=1 this makes the module identical to LinearThenSoftmax; at p>1 it starts as the
        # plain sum of p randomly-initialized channels, which is an unbiased neutral start
        # rather than an arbitrary weighting.
        with torch.no_grad():
            self.head.weight.fill_(1.0)
            self.head.bias.zero_()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:  # (batch, seq, embed), (batch, seq) -> (batch,)
        temperature = self.kwargs["temperature"]
        m = mask.unsqueeze(-1)                                # (B, S, 1), broadcasts over p
        raw = self.linear(self.norm(x))                       # (B, S, p) linear scores
        z = self.act(raw)                                     # identity by default
        z = z.masked_fill(~m, 0)
        # Padding carries a zeroed activation vector, so its raw score is the bias and its
        # activated score is act(bias) -- neither is 0 in general. Both are masked before use.
        src = z if self.channel_weights_from == "activated" else raw.masked_fill(~m, 0)
        # dim=1 is the SEQUENCE axis, so each of the p channels gets its own distribution over
        # tokens. Pooling in float32: at p>1 some channels can saturate, and bf16 softmax over
        # ~1000 positions loses more than the extra cast costs.
        w = torch.softmax(
            src.masked_fill(~m, float("-inf")).float() / temperature, dim=1
        ).to(z.dtype)
        pooled = (z * w).sum(dim=1)                           # (B, p)
        return self.head(pooled).squeeze(-1)                  # (B,)


def verify_multichannel_head_identity(embed_dim: int = 64, seq: int = 40, batch: int = 6,
                                      seed: int = 0) -> None:
    """Assert that at ``p = 1`` this head reproduces ``LinearThenSoftmax`` at initialization."""
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax

    torch.manual_seed(seed)
    x = torch.randn(batch, seq, embed_dim)
    lens = torch.randint(3, seq + 1, (batch,))
    mask = torch.arange(seq)[None, :] < lens[:, None]
    ref = LinearThenSoftmax(embed_dim, temperature=5)
    m = MultiChannelLinearThenSoftmax(embed_dim, temperature=5, n_channels=1)
    m.linear.load_state_dict(ref.linear.state_dict())
    d = (ref(x, mask) - m(x, mask)).abs().max().item()
    assert d < 1e-4, f"p=1: multi-channel head differs from LinearThenSoftmax by {d}"
    print(f"  p=1: matches LinearThenSoftmax at init (max diff {d:.2e})")
    for p in (2, 4, 8):
        mp = MultiChannelLinearThenSoftmax(embed_dim, temperature=5, n_channels=p)
        out = mp(x, mask)
        assert out.shape == (batch,), out.shape
        print(f"  p={p}: forward ok, output {tuple(out.shape)}")


_HIDDEN_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "identity": nn.Identity,
}


class MLPThenSoftmax(nn.Module):
    """A hidden layer with a nonlinearity between the activations and the per-token score.

    The other two heads in this module widen the readout while keeping each scoring direction
    LINEAR in the activation vector: ``LinearThenSegmentedSoftmax`` splits one linear score by
    position, ``MultiChannelLinearThenSoftmax`` runs p linear scores in parallel. Both leave
    intact the assumption that "how instruction-following is this token" is a dot product with
    one direction. This head drops that assumption instead: each token is scored by
    ``Linear(embed -> h) -> GELU -> Linear(h -> 1)``, so the score can depend on a conjunction
    of directions (present AND not-present, or a threshold on one feature gated by another).

    Everything after the score is untouched -- mask, softmax over the sequence at the same
    temperature, weighted sum -- so this is a strictly local change to step 1 and is directly
    comparable to every other fit here. The output is a logit; the sigmoid stays downstream in
    ``BCEWithLogitsLoss`` / ``probs()``.

    Unlike the other two heads this one does NOT start at ``LinearThenSoftmax``: a random MLP
    is not a random linear map, and forcing it to would mean pinning the hidden layer to an
    identity, which is only possible at h=1. The control is instead the ``identity`` activation
    at h=1, which reduces the composite to a rescaled single linear score -- a
    reparameterization of the stock head, trainable but functionally the same family.
    ``verify_mlp_head_identity`` asserts that reduction holds.

    Note the parameter count: at embed 5376 the hidden layer alone is ``5376 * h``, so h=64 is
    344k parameters. Against this project's 116-row training set that is far more capacity than
    the p-channel ladder had when its mean AUROC began to erode, so treat overfitting as the
    expected outcome to be measured rather than a surprise.

    Hyperparameters (via ``ProbeSpec.hyperparams``):
      n_hidden           h, the hidden width (default 64).
      hidden_activation  one of gelu (default), relu, silu, tanh, identity.
    """

    def __init__(self, embed_dim: int, **kwargs: Any):
        super().__init__()
        h = int(kwargs.get("n_hidden", 64))
        if h < 1:
            raise ValueError(f"n_hidden must be >= 1, got {h}")
        name = str(kwargs.get("hidden_activation", "gelu")).lower()
        if name not in _HIDDEN_ACTIVATIONS:
            raise ValueError(
                f"hidden_activation must be one of {sorted(_HIDDEN_ACTIVATIONS)}, got {name!r}"
            )
        self.n_hidden = h
        self.hidden_activation = name
        self.kwargs = kwargs
        self.fc1 = nn.Linear(embed_dim, h)
        self.act = _HIDDEN_ACTIVATIONS[name]()
        self.fc2 = nn.Linear(h, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:  # (batch, seq, embed), (batch, seq) -> (batch,)
        temperature = self.kwargs["temperature"]
        z = self.fc2(self.act(self.fc1(x))).squeeze(-1)       # (B, S) per-token scores
        z = z.masked_fill(~mask, 0)
        w = torch.softmax(
            z.masked_fill(~mask, float("-inf")).float() / temperature, dim=1
        ).to(z.dtype)
        return (z * w).sum(dim=1)                             # (B,) logit


def verify_mlp_head_identity(embed_dim: int = 64, seq: int = 40, batch: int = 6,
                             seed: int = 0) -> None:
    """Assert the h=1 / identity-activation reduction, and that the real head runs at each h."""
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax

    torch.manual_seed(seed)
    x = torch.randn(batch, seq, embed_dim)
    lens = torch.randint(3, seq + 1, (batch,))
    mask = torch.arange(seq)[None, :] < lens[:, None]
    ref = LinearThenSoftmax(embed_dim, temperature=5)
    m = MLPThenSoftmax(embed_dim, temperature=5, n_hidden=1, hidden_activation="identity")
    m.fc1.load_state_dict(ref.linear.state_dict())
    with torch.no_grad():
        m.fc2.weight.fill_(1.0)
        m.fc2.bias.zero_()
    d = (ref(x, mask) - m(x, mask)).abs().max().item()
    assert d < 1e-4, f"h=1/identity: MLP head differs from LinearThenSoftmax by {d}"
    print(f"  h=1, identity: matches LinearThenSoftmax (max diff {d:.2e})")
    for h in (8, 32, 128):
        mh = MLPThenSoftmax(embed_dim, temperature=5, n_hidden=h)
        out = mh(x, mask)
        assert out.shape == (batch,), out.shape
        n = sum(p.numel() for p in mh.parameters())
        print(f"  h={h}: forward ok, output {tuple(out.shape)}, {n} params at embed {embed_dim}")
