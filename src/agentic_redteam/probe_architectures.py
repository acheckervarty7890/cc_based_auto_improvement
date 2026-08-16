"""Non-linear probe heads, and one builder that trains any of them off cached activations.

Why this lives here and not in tuberlens
----------------------------------------
``ProbeType`` is a ``str, Enum`` and ``ProbeFactory.build`` dispatches on it with a
``match``, so a new architecture cannot be registered from outside — the enum is closed
and an unmatched name falls through to an unbound ``classifier``. Forking tuberlens is
the other option, but its checkout lives under ``.venv_claude/src/`` and is **not**
committed with this repo, so a fork would silently not survive a fresh box (which is
exactly how the activation caches get lost). So the new heads are defined here and
:func:`build_probe` mirrors ``ProbeFactory.build`` for every architecture, stock or new.
Nothing in tuberlens is modified.

What is being tested
--------------------
``docs/heldout_v3_vs_v2_overlap.md`` closes with 31 eval rows misclassified by all eight
probe families it built (2 attacker arms × 2 disjoint red-team vintages × with/without the
base data × 5 seeds) and concludes the residue is reachable only off that axis — a
different eval concept boundary, or a different probe architecture. This module is the
architecture half.

Two axes are varied independently, because ``why_last_iteration_adds_nothing.md`` §2 says
they are not equally promising:

- **Pooling** — how per-token evidence becomes one sequence logit. The deployed head
  (``linear_then_softmax``) weights each token by its *own* logit, so the pooling cannot
  attend to anything the classifier does not already score highly. ``AttnLite``
  decouples the two with a separate query. This is the axis with visible headroom: the
  same activations read out with mean-pooling score 0.64 on ``eval_ai_dilemmas`` where
  the deployed softmax pooling reaches 0.998.
- **Feature-wise non-linearity** — an MLP readout instead of a linear one. §2's per-split
  linear ceiling is 0.96 AUROC on ``eval_ant_hh`` against 0.77 achieved, i.e. the signal
  is already linearly extractable and capacity is *not* the binding constraint. These
  heads are therefore expected to buy little, and are here to establish that rather than
  to assume it. ``scripts/nonlinear_ceiling.py`` measures the same thing directly.
- **Positional pooling** — :class:`TokenPool` and the two heads built on it
  (``token_linear_then_softmax``, ``token_mlp_then_softmax``) weight each token by
  *where it is*, not by what it says. Every other head here pools by content; mean
  pooling is the special case of this one with the weights frozen at ``1/L``. It is a
  separate axis because "which positions carry the label" is a hypothesis the
  content-based heads cannot express and the fixed mean cannot learn. Note what it can
  and cannot reach: the blobs are **right**-padded, so index ``s`` is "the s-th token
  from the start" and the map can learn e.g. *the opening turns do not matter* — but it
  cannot align to the **end** of a conversation, which is where an assistant-centric
  label actually lives, because the last real token sits at a different index in every
  row (median 300, max 1024). End-alignment would need a per-row gather, and is the
  obvious follow-up if position turns out to carry anything at all.
  ``scripts/token_pool_vintage.py`` runs it across the three red-team vintages.

Capacity is deliberately small. Both arms already fit their training sets perfectly
(train accuracy 1.0000, ~600-900 rows in 5376 dimensions — ``attribution_findings.md``),
so an unregularised MLP buys variance, not signal. ``hidden_dim`` defaults to 64 and the
optimizer args are **inherited unchanged from the linear counterpart**, so a head-to-head
against the baseline changes the architecture and nothing else; ``scripts/arch_sweep.py
--sensitivity`` then varies ``hidden_dim``/``weight_decay`` separately to check the
result is not an artefact of that choice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# --- the new heads -----------------------------------------------------------------

#: Width of the MLP readout's hidden layer. Small on purpose — see the module docstring.
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.0


def _mlp_head(embed_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """``Linear -> GELU -> [Dropout] -> Linear`` returning one logit.

    Applied to the last dimension, so it is a drop-in for the ``nn.Linear(embed_dim, 1)``
    that ``LinearThenAgg`` and ``AttnLite`` use — per token in the first case, on the
    pooled vector in the second.
    """
    layers: list[nn.Module] = [nn.Linear(embed_dim, hidden_dim), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


def _head_kwargs(kwargs: dict[str, Any]) -> tuple[int, float]:
    return (
        int(kwargs.get("hidden_dim", DEFAULT_HIDDEN_DIM)),
        float(kwargs.get("dropout", DEFAULT_DROPOUT)),
    )


# --- learned pooling over the TOKEN axis ---------------------------------------------

#: Width the token axis is treated as. ``get_activations`` truncates at 1024
#: (``tuberlens/model.py:433``), so every blob this repo holds is at most this wide and a
#: fixed-size map over token positions is well defined.
DEFAULT_MAX_TOKENS = 1024

#: Hidden width of the token-axis MLP. Deliberately the same 64 as the embedding-axis
#: readout, so ``token_mlp_then_softmax`` differs from ``token_linear_then_softmax`` by
#: the non-linearity and nothing else.
DEFAULT_TOKEN_HIDDEN_DIM = 64

#: How many pooled vectors the token map produces. 1 is the design being tested — the
#: whole sequence collapses to one activation vector, and the softmax pooling downstream
#: is then a no-op over a length-1 sequence. >1 leaves the downstream head a real pooling
#: job over learned slots; it exists so that variant is one flag away rather than a
#: rewrite.
DEFAULT_N_SLOTS = 1


def _token_pool_kwargs(kwargs: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)),
        int(kwargs.get("token_hidden_dim", DEFAULT_TOKEN_HIDDEN_DIM)),
        int(kwargs.get("n_slots", DEFAULT_N_SLOTS)),
    )


class TokenPool(nn.Module):
    """Reduce ``[B, S, E]`` to ``[B, R, E]`` with a learned map over token *positions*.

    The contrast with every pooling head already here: ``pre_mean`` / ``mean_then_mlp``
    average the token axis with fixed uniform weights, ``linear_then_softmax`` weights
    each token by its own logit, and ``attention`` weights it by a learned query — all
    three are functions of a token's **content**. This one is a function of its
    **position**: weight ``w[r, s]`` multiplies whatever sits at index ``s``, and the
    same weight applies to every conversation. Mean pooling is the special case
    ``w = 1/L``.

    Three properties are load-bearing and were each checked rather than assumed:

    - **Right padding.** Every activation blob in this repo is right-padded — real tokens
      occupy indices ``0..L-1``, so index ``s`` means "the s-th token of the
      conversation" for every row. (Verified across all four eval blobs, both base blobs
      and the per-conversation red-team cache: no row has a leading mask zero.)
    - **Slicing the weight is exactly zero-padding the input.** Blobs arrive at whatever
      width their extraction produced — 84 for the base split, 121/144/288/859 for the
      four eval splits, 1024 for the merged red-team set — so a fixed 1024-wide map has
      to be reconciled with the tensor in hand. Materialising ``F.pad(x, ...)`` out to
      1024 would copy a ``[B, 1024, 5376]`` tensor per forward (352 MB at batch 16, fp32)
      to multiply the added columns by weights and get zero. Taking ``weight[:, :S]``
      instead is the identical function at no cost, *because* padding is on the right and
      the contribution of a zero column is zero. Both layers of the MLP variant stay
      exact under this, since a zero pre-activation contribution passes through the sum
      before the GELU, not after it.
    - **Pad positions hold exact zeros.** ``get_activations`` zero-pads,
      ``_concatenate_consuming`` zero-fills, and ``attribution_refit._attach_redteam``
      builds from ``torch.zeros`` — measured ``max|activation|`` at masked positions is
      0.0 across every blob. So ``mask`` never has to be applied here, which is what
      keeps the forward allocation-free. ``scripts/token_pool_vintage.py --self-test``
      re-checks the invariant on the real tensors rather than trusting this comment.

    What this head does **not** do is normalise by length: ``pre_mean`` divides by the
    token count, this does not, so a longer conversation contributes a larger pooled
    vector. That is inherent to "a linear layer over the token dimension" and is left in
    rather than quietly corrected — ``n_slots``-style, the length-normalised variant is
    available as ``normalize=True`` so the difference can be measured instead of argued.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        n_slots: int = DEFAULT_N_SLOTS,
        hidden_dim: int | None = None,
        normalize: bool = False,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.n_slots = n_slots
        self.normalize = normalize
        if hidden_dim is None:
            self.first = nn.Linear(max_tokens, n_slots)
            self.rest: nn.Module | None = None
        else:
            self.first = nn.Linear(max_tokens, hidden_dim)
            self.rest = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, n_slots))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq = x.shape[1]
        if seq > self.max_tokens:
            # Unreachable at the 1024-token extraction cap, but a silent wrong answer if
            # it ever were reached, so it is truncated rather than assumed away.
            x = x[:, : self.max_tokens]
            seq = self.max_tokens
        # [B, E, R'] — the token axis is contracted away, the embedding axis is carried
        # through untouched, and `first.bias` broadcasts over the trailing slot axis.
        h = torch.einsum("bse,os->beo", x, self.first.weight[:, :seq])
        if self.normalize:
            # Before the bias, not after: the bias is a per-slot constant, not part of
            # the weighted sum, so dividing it by the length would make the head's
            # offset depend on how long the conversation happened to be.
            h = h / mask.sum(dim=1).clamp(min=1).to(h.dtype)[:, None, None]
        h = h + self.first.bias
        if self.rest is not None:
            h = self.rest(h)
        return h.transpose(1, 2)


_MODULE_CLASSES: dict[str, type[nn.Module]] | None = None


def new_module_classes() -> dict[str, type[nn.Module]]:
    """The three new heads, built once and cached.

    They subclass tuberlens modules, so they cannot be defined at import time without
    making ``agentic_redteam`` import torch eagerly — every other module here imports
    tuberlens lazily for the same reason. Caching keeps the class *identity* stable, so
    ``probe_architecture`` comparisons and ``isinstance`` behave as they would for a
    stock architecture.
    """
    global _MODULE_CLASSES
    if _MODULE_CLASSES is not None:
        return _MODULE_CLASSES

    from tuberlens.probes.pytorch_modules import AttnLite, LinearThenSoftmax

    class MLPThenSoftmax(LinearThenSoftmax):
        """Per-token MLP readout, then the deployed head's own softmax pooling.

        Isolates feature-wise non-linearity: the pooling is byte-identical to
        ``linear_then_softmax``'s, so a difference against it is the readout alone.
        """

        def __init__(self, embed_dim: int, **kw: Any):
            super().__init__(embed_dim, **kw)
            h, d = _head_kwargs(kw)
            self.linear = _mlp_head(embed_dim, h, d)

    class MeanThenMLP(nn.Module):
        """Mean-pool, then an MLP — the non-linear counterpart of ``pre_mean``.

        The complement of ``MLPThenSoftmax``: pooling is fixed and label-independent, so
        anything this gains over ``pre_mean`` is feature-wise non-linearity acting on the
        pooled vector, with no interaction with token selection.
        """

        def __init__(self, embed_dim: int, **kw: Any):
            super().__init__()
            h, d = _head_kwargs(kw)
            self.mlp = _mlp_head(embed_dim, h, d)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # Masked mean. tuberlens' MeanThenLinear sums x unmasked and relies on pad
            # positions being exactly zero; they are, but masking explicitly matches
            # PytorchDifferenceOfMeansClassifier and costs nothing.
            m = mask.bool()
            x = x.masked_fill(~m.unsqueeze(-1), 0.0)
            x = x.sum(dim=1) / m.sum(dim=1, keepdim=True).clamp(min=1)
            return self.mlp(x).squeeze(-1)

    class AttentionThenMLP(AttnLite):
        """``AttnLite`` with an MLP classifier in place of its linear one.

        Both axes at once. Worth separating from the other two because ``AttnLite``
        applies ``self.classifier`` twice — per token (for the reported token logits) and
        once on the attention-pooled context (for the sequence logit it is trained on) —
        so the MLP here is non-linear in the *pooled* representation, which the decoupled
        query is free to shape.
        """

        def __init__(self, embed_dim: int, **kw: Any):
            super().__init__(embed_dim, **kw)
            h, d = _head_kwargs(kw)
            self.classifier = _mlp_head(embed_dim, h, d)

    class _TokenPoolThenSoftmax(LinearThenSoftmax):
        """Pool the token axis by learned position weights, then the deployed head.

        ``LinearThenSoftmax`` is reached through ``super().forward`` on the pooled
        ``[B, R, E]`` tensor with an all-ones mask, so the readout and the aggregation
        are byte-identical to the baseline's and the only change is what produced the
        sequence they see. At the default ``n_slots=1`` that aggregation is a softmax
        over one element — weight 1.0 — so the sequence logit reduces exactly to
        ``w . pooled + b``. That degeneracy is the design, not an oversight: the request
        was to replace mean pooling with a learned map over token positions and then read
        out, and keeping the real ``LinearThenSoftmax`` in the path is what makes
        ``n_slots > 1`` a one-flag change rather than a different class.
        """

        #: Subclass switch: whether the token map has a hidden layer. Its *width* comes
        #: from the ``token_hidden_dim`` hyperparameter either way, so a sensitivity run
        #: varies one knob rather than editing a class.
        use_token_hidden: bool = False

        def __init__(self, embed_dim: int, **kw: Any):
            super().__init__(embed_dim, **kw)
            max_tokens, h_tok, n_slots = _token_pool_kwargs(kw)
            self.token_pool = TokenPool(
                max_tokens,
                n_slots,
                hidden_dim=h_tok if self.use_token_hidden else None,
                normalize=bool(kw.get("normalize", False)),
            )

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            pooled = self.token_pool(x, mask)
            pooled_mask = torch.ones(
                pooled.shape[:2], dtype=torch.bool, device=pooled.device
            )
            return super().forward(pooled, pooled_mask)

    class TokenLinearThenSoftmax(_TokenPoolThenSoftmax):
        """One learned weight per token position, then the deployed readout."""

        use_token_hidden = False

    class TokenMLPThenSoftmax(_TokenPoolThenSoftmax):
        """``Linear -> GELU -> Linear`` over the token axis, then the deployed readout.

        The token-axis counterpart of ``mlp_then_softmax``: the same non-linearity, moved
        from the embedding axis to the position axis. It can express "attend to the tail
        only when the head is quiet", which the single-layer version cannot.
        """

        use_token_hidden = True

    _MODULE_CLASSES = {
        "mlp_then_softmax": MLPThenSoftmax,
        "mean_then_mlp": MeanThenMLP,
        "attention_then_mlp": AttentionThenMLP,
        "token_linear_then_softmax": TokenLinearThenSoftmax,
        "token_mlp_then_softmax": TokenMLPThenSoftmax,
    }
    return _MODULE_CLASSES


# --- architecture registry ----------------------------------------------------------

#: Architectures defined here, and the stock architecture each inherits its training
#: hyperparameters from. Inheriting rather than choosing fresh ones is what makes the
#: head-to-head an architecture comparison instead of a hyperparameter comparison.
MLP_READOUT_ARCHITECTURES: dict[str, str] = {
    "mlp_then_softmax": "linear_then_softmax",
    "mean_then_mlp": "pre_mean",
    "attention_then_mlp": "attention",
}


#: Heads that replace mean pooling with a learned map over token **positions** (see
#: :class:`TokenPool`). Both inherit ``linear_then_softmax``'s optimizer settings — not
#: ``pre_mean``'s — because what follows the pool *is* ``LinearThenSoftmax``, and because
#: the two stock heads ship identical settings anyway (batch 16 / accumulation 4 /
#: final_lr 1e-4), so the choice costs nothing and keeps the head-to-head against the
#: deployed baseline a change of architecture alone.
TOKEN_POOL_ARCHITECTURES: dict[str, str] = {
    "token_linear_then_softmax": "linear_then_softmax",
    "token_mlp_then_softmax": "linear_then_softmax",
}


NEW_ARCHITECTURES: dict[str, str] = {
    **MLP_READOUT_ARCHITECTURES,
    **TOKEN_POOL_ARCHITECTURES,
}


#: Closed-form architectures defined here (no Adam, no epochs).
#:
#: ``lda_shrinkage`` exists because **tuberlens' ``ProbeType.lda`` is not LDA**:
#: ``PytorchDifferenceOfMeansClassifier`` declares ``use_lda`` and never reads it, so
#: ``lda`` and ``difference_of_means`` train the identical mass-mean probe and return
#: bit-identical logits. Keeping both in a sweep would silently report one estimator
#: twice, which is worse than not testing it.
CLOSED_FORM_ARCHITECTURES: dict[str, str] = {
    "lda_shrinkage": "difference_of_means",
}


def default_hyperparams(arch: str) -> dict[str, Any]:
    """Default training args for any architecture name, stock or new."""
    from tuberlens.interfaces.probes import ProbeType

    if arch in TOKEN_POOL_ARCHITECTURES:
        base = ProbeType(TOKEN_POOL_ARCHITECTURES[arch]).default_hyperparams.copy()
        base["max_tokens"] = DEFAULT_MAX_TOKENS
        base["n_slots"] = DEFAULT_N_SLOTS
        base["normalize"] = False
        if arch == "token_mlp_then_softmax":
            base["token_hidden_dim"] = DEFAULT_TOKEN_HIDDEN_DIM
        return base
    if arch in NEW_ARCHITECTURES:
        base = ProbeType(NEW_ARCHITECTURES[arch]).default_hyperparams.copy()
        base["hidden_dim"] = DEFAULT_HIDDEN_DIM
        base["dropout"] = DEFAULT_DROPOUT
        return base
    if arch in CLOSED_FORM_ARCHITECTURES:
        base = ProbeType(CLOSED_FORM_ARCHITECTURES[arch]).default_hyperparams.copy()
        base["shrinkage"] = DEFAULT_SHRINKAGE
        return base
    return ProbeType(arch).default_hyperparams.copy()


def all_architectures() -> list[str]:
    from tuberlens.interfaces.probes import ProbeType

    return (
        [t.value for t in ProbeType]
        + list(NEW_ARCHITECTURES)
        + list(CLOSED_FORM_ARCHITECTURES)
    )


# --- the best-epoch restore fix -----------------------------------------------------

#: Set to 1 to reproduce tuberlens' behaviour, in which early stopping selects an epoch
#: but cannot restore it. Exists so the fix itself can be measured as a control arm
#: rather than silently changing every number.
LEGACY_BEST_EPOCH_ENV = "AGENTIC_REDTEAM_LEGACY_BEST_EPOCH"


def _restore_default() -> bool:
    return os.environ.get(LEGACY_BEST_EPOCH_ENV, "") not in ("1", "true", "True")


def _clone_on_state_dict(model: nn.Module) -> None:
    """Make ``model.state_dict()`` return detached clones, per instance.

    ``PytorchAdamClassifier.train`` snapshots the best epoch as
    ``self.model.state_dict().copy()``. ``state_dict()`` returns *references* to the live
    parameter tensors and ``dict.copy()`` is shallow, so the snapshot tracks training and
    the closing ``load_state_dict(best_model_state)`` restores the final epoch's weights
    onto themselves — early stopping picks an epoch and then cannot go back to it
    (``docs/attribution_findings.md`` §1).

    Cloning at the ``state_dict`` boundary fixes it without touching tuberlens: the
    snapshot becomes a real copy and ``load_state_dict`` then does what its name says.
    The patch is an instance attribute (``nn.Module.__setattr__`` passes plain functions
    through to ``object.__setattr__``) and is removed after training, since a closure
    over a bound method would make the probe unpicklable.
    """
    original = model.state_dict

    def state_dict(*args: Any, **kwargs: Any) -> dict:
        return {k: v.detach().clone() for k, v in original(*args, **kwargs).items()}

    model.state_dict = state_dict  # type: ignore[method-assign]


def _make_adam_classifier(architecture, hyperparams: dict, restore_best_epoch: bool):
    """``PytorchAdamClassifier`` whose early-stopping snapshot actually restores."""
    from tuberlens.probes.pytorch_classifiers import PytorchAdamClassifier

    @dataclass(kw_only=True)
    class BestEpochAdamClassifier(PytorchAdamClassifier):
        restore_best_epoch: bool = True

        def train(self, activations, y, validation_activations=None,
                  validation_y=None, **kwargs):
            if self.restore_best_epoch and kwargs.get("initialize_model", True):
                # Build the model here so its state_dict can be patched before the
                # trainer takes its first snapshot, then hand it over already
                # initialised. The trainer's initialize_model=False branch is otherwise
                # identical (it calls .train(), which the training loop does anyway).
                self.model = (
                    self.probe_architecture(activations.embed_dim, **self.training_args)
                    .to(self.device)
                    .to(self.dtype)
                )
                _clone_on_state_dict(self.model)
                kwargs["initialize_model"] = False
            try:
                return super().train(
                    activations, y, validation_activations, validation_y, **kwargs
                )
            finally:
                if self.model is not None:
                    self.model.__dict__.pop("state_dict", None)

    return BestEpochAdamClassifier(
        training_args=hyperparams,
        probe_architecture=architecture,
        restore_best_epoch=restore_best_epoch,
    )


# --- shrinkage LDA (closed form) ----------------------------------------------------

#: ``"auto"`` uses the Ledoit-Wolf coefficient; a float in [0, 1] pins it. Shrinkage is
#: not optional here: the pooled covariance is 5376x5376 estimated from ~700 samples, so
#: the sample estimate is singular and unshrunk LDA is undefined.
DEFAULT_SHRINKAGE: Any = "auto"

#: Rows cast to float64 at a time when mean-pooling for the LDA fit. At 1024 tokens x 5376
#: dims x 8 B that is ~176 MB per chunk; the whole tensor at once is ~33 GB.
_LDA_POOL_CHUNK = 4


def _ledoit_wolf_alpha(x: torch.Tensor, cov: torch.Tensor) -> float:
    """Ledoit-Wolf shrinkage intensity toward ``(tr(S)/d) I``.

    ``x`` is the within-class-centred data ``[n, d]`` whose covariance is ``cov``. The
    per-sample term is expanded rather than materialised, since ``sum_i ||x_i x_i^T -
    S||_F^2`` over 5376 dimensions would otherwise build ``n`` dense ``d x d`` outer
    products:

        sum_i ||x_i x_i^T||_F^2 = sum_i (x_i . x_i)^2
        sum_i <x_i x_i^T, S>    = sum_i x_i^T S x_i
    """
    n, d = x.shape
    mu = torch.diagonal(cov).mean()
    # ||S - mu I||_F^2 expanded as ||S||_F^2 - 2 mu tr(S) + d mu^2, which avoids
    # materialising a 5376x5376 float64 identity (231 MB) purely to subtract a diagonal.
    cov_sq_norm = (cov**2).sum()
    delta_sq = (cov_sq_norm - 2 * mu * torch.diagonal(cov).sum() + d * mu**2) / d

    sq_norms = (x * x).sum(dim=1)
    term_xx = (sq_norms**2).sum()
    term_xs = ((x @ cov) * x).sum()
    beta_sq = (term_xx - 2 * term_xs + n * cov_sq_norm) / (n**2 * d)

    if delta_sq <= 0:
        return 1.0
    return float(torch.clamp(beta_sq / delta_sq, 0.0, 1.0))


def _fit_shrinkage_lda(activations, y, shrinkage: Any) -> tuple[torch.Tensor, float]:
    """``(w, b)`` for ``w = S^-1 (mu_pos - mu_neg)`` on mean-pooled activations.

    The contrast with ``difference_of_means``, which is the *unwhitened* ``mu_pos -
    mu_neg``. Marks & Tegmark (2023) report that the unwhitened direction transfers
    better across distributions than whitened/logistic ones, and distribution shift is
    precisely this pipeline's failure mode — so having both estimators is the point,
    and a result where mass-mean wins is a real result rather than a null.

    Computed in float64 on CPU: the Cholesky solve of a 5376x5376 system is seconds
    there, whereas consumer GPUs run float64 at 1/64 rate, and float32 on a matrix this
    ill-conditioned is where a shrinkage estimator would quietly lose its point.
    """
    # Pool in chunks. Casting the whole [n, seq, embed] tensor to float64 at once is
    # 747 x 1024 x 5376 x 8 B = 32.9 GB on a real training set — it OOM-killed the sweep.
    # The pooled result is only [n, embed] (32 MB), so only the cast needs bounding, and
    # a mean over the token axis is per-row so chunking is exact.
    mask = activations.attention_mask.bool().cpu()
    acts = activations.activations
    n_rows, _seq, embed_dim = acts.shape
    x = torch.empty((n_rows, embed_dim), dtype=torch.float64)
    for i in range(0, n_rows, _LDA_POOL_CHUNK):
        m = mask[i : i + _LDA_POOL_CHUNK]
        h = acts[i : i + _LDA_POOL_CHUNK].cpu().to(torch.float64)
        h = h.masked_fill(~m.unsqueeze(-1), 0.0)
        x[i : i + _LDA_POOL_CHUNK] = (
            h.sum(dim=1) / m.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float64)
        )
        del h, m

    y = y.cpu().flatten() > 0.5
    if not y.any() or y.all():
        raise ValueError("shrinkage LDA needs both classes present")

    mu_pos = x[y].mean(dim=0)
    mu_neg = x[~y].mean(dim=0)

    centred = x.clone()
    centred[y] -= mu_pos
    centred[~y] -= mu_neg
    n, d = centred.shape
    # Pooled within-class covariance; n-2 because two class means were estimated.
    cov = centred.T @ centred / max(n - 2, 1)

    alpha = (
        _ledoit_wolf_alpha(centred, cov)
        if shrinkage == "auto"
        else float(shrinkage)
    )
    mu = torch.diagonal(cov).mean()
    cov.mul_(1.0 - alpha).diagonal().add_(alpha * mu)

    delta = mu_pos - mu_neg
    try:
        w = torch.cholesky_solve(
            delta.unsqueeze(1), torch.linalg.cholesky(cov)
        ).squeeze(1)
    except Exception:
        # A pinned shrinkage of 0 (or a degenerate class) can leave cov indefinite.
        w = torch.linalg.lstsq(cov, delta.unsqueeze(1)).solution.squeeze(1)

    b = float(-0.5 * w @ (mu_pos + mu_neg))
    print(f"  shrinkage LDA: alpha={alpha:.4f} (n={n}, d={d})")
    return w, b


@dataclass(kw_only=True)
class ClosedFormMeanClassifier:
    """A ``PytorchClassifier`` whose weights come from a closed form, not from Adam.

    Wraps the fitted direction in tuberlens' own ``MeanThenLinear`` so scoring goes
    through the standard ``PytorchProbe`` path. That is exact rather than convenient:
    ``MeanThenLinear`` computes ``w . mean_t(h_t) + b``, and mean-pooling commutes with a
    linear map, so this is the same function the direction was fitted for.
    """

    training_args: dict
    model: nn.Module | None = None
    best_epoch: int | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    shrinkage: Any = DEFAULT_SHRINKAGE

    def train(self, activations, y, validation_activations=None, validation_y=None,
              **_kwargs):
        from tuberlens.config import global_settings
        from tuberlens.probes.pytorch_modules import MeanThenLinear

        self.device = global_settings.DEVICE
        self.dtype = global_settings.DTYPE
        w, b = _fit_shrinkage_lda(activations, y, self.shrinkage)

        model = MeanThenLinear(w.numel())
        with torch.no_grad():
            model.linear.weight.copy_(w.reshape(1, -1))
            model.linear.bias.fill_(b)
        self.model = model.to(self.device).to(self.dtype)
        return self

    @torch.no_grad()
    def logits(self, activations, per_token: bool = False) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not trained")
        if per_token:
            raise NotImplementedError("per-token logits are not defined for mean pooling")
        self.model.eval()
        acts = activations.activations.to(self.device, self.dtype)
        mask = activations.attention_mask.to(self.device).bool()
        return self.model(acts, mask)

    @torch.no_grad()
    def probs(self, activations, per_token: bool = False) -> torch.Tensor:
        return self.logits(activations, per_token=per_token).sigmoid()


# --- the builder --------------------------------------------------------------------


def _stock_architecture(arch: str):
    """The ``nn.Module`` class a stock Adam-trained ``ProbeType`` uses."""
    from tuberlens.probes.pytorch_modules import (
        AttnLite,
        LinearThenLast,
        LinearThenMax,
        LinearThenMean,
        LinearThenRollingMax,
        LinearThenSoftmax,
        MeanThenLinear,
    )

    return {
        "pre_mean": MeanThenLinear,
        "attention": AttnLite,
        "linear_then_mean": LinearThenMean,
        "linear_then_max": LinearThenMax,
        "linear_then_softmax": LinearThenSoftmax,
        "linear_then_rolling_max": LinearThenRollingMax,
        "linear_then_last": LinearThenLast,
    }[arch]


def build_probe(
    arch: str,
    train_dataset,
    validation_dataset,
    *,
    model_name: str,
    layer: int,
    pos_class_label: str,
    neg_class_label: str,
    probe_description: str | None = None,
    hyperparams: dict | None = None,
    restore_best_epoch: bool | None = None,
):
    """Train one probe of architecture ``arch`` on pre-activated datasets.

    Mirrors ``ProbeFactory.build`` — same ``PytorchProbe`` wrapper, same
    ``probe.fit(train, val)`` call, same hyperparameter defaults — with two differences,
    both required for this comparison to mean anything:

    1. The three architectures in :data:`NEW_ARCHITECTURES` can be named, which the
       closed ``ProbeType`` enum cannot express.
    2. Every Adam-trained architecture, stock or new, is trained through
       :func:`_make_adam_classifier`, so early stopping restores the epoch it selected.
       Applying the fix to *all* arms is the point: it would otherwise handicap exactly
       the higher-capacity heads this sweep is testing, since early stopping is their
       main overfitting control.

    ``use_store`` is not exposed — the probe store keys on ``FullProbeSpec``, which has no
    room for an architecture it does not know about.
    """
    from tuberlens.interfaces.probes import ProbeType
    from tuberlens.probes.probe_factory import has_activations
    from tuberlens.probes.pytorch_probes import PytorchProbe
    from tuberlens.probes.sklearn_probes import SklearnProbe

    if restore_best_epoch is None:
        restore_best_epoch = _restore_default()
    hp = dict(default_hyperparams(arch))
    if hyperparams:
        hp.update(hyperparams)

    for name, ds in (("train", train_dataset), ("validation", validation_dataset)):
        if ds is not None and not has_activations(ds):
            raise ValueError(f"{name} dataset must carry activations")

    probe_kwargs = {
        "model_name": model_name,
        "layer": layer,
        "pos_class_label": pos_class_label,
        "neg_class_label": neg_class_label,
    }
    if probe_description is not None:
        probe_kwargs["description"] = probe_description

    if arch == ProbeType.sklearn.value:
        return SklearnProbe(hyper_params=hp, **probe_kwargs).fit(train_dataset)

    if arch in CLOSED_FORM_ARCHITECTURES:
        classifier = ClosedFormMeanClassifier(
            training_args=hp, shrinkage=hp.get("shrinkage", DEFAULT_SHRINKAGE)
        )
    elif arch in (ProbeType.difference_of_means.value, ProbeType.lda.value):
        from tuberlens.probes.pytorch_classifiers import (
            PytorchDifferenceOfMeansClassifier,
        )

        classifier = PytorchDifferenceOfMeansClassifier(
            use_lda=(arch == ProbeType.lda.value), training_args=hp
        )
    else:
        architecture = (
            new_module_classes()[arch]
            if arch in NEW_ARCHITECTURES
            else _stock_architecture(arch)
        )
        classifier = _make_adam_classifier(architecture, hp, restore_best_epoch)

    probe = PytorchProbe(hyper_params=hp, _classifier=classifier, **probe_kwargs)
    probe.fit(train_dataset, validation_dataset)
    return probe
