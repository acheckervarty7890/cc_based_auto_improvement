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

    _MODULE_CLASSES = {
        "mlp_then_softmax": MLPThenSoftmax,
        "mean_then_mlp": MeanThenMLP,
        "attention_then_mlp": AttentionThenMLP,
    }
    return _MODULE_CLASSES


# --- architecture registry ----------------------------------------------------------

#: Architectures defined here, and the stock architecture each inherits its training
#: hyperparameters from. Inheriting rather than choosing fresh ones is what makes the
#: head-to-head an architecture comparison instead of a hyperparameter comparison.
NEW_ARCHITECTURES: dict[str, str] = {
    "mlp_then_softmax": "linear_then_softmax",
    "mean_then_mlp": "pre_mean",
    "attention_then_mlp": "attention",
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
    mask = activations.attention_mask.bool().cpu()
    acts = activations.activations.cpu().to(torch.float64)
    acts = acts.masked_fill(~mask.unsqueeze(-1), 0.0)
    x = acts.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).to(torch.float64)
    del acts

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
