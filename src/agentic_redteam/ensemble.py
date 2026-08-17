"""Score-averaging deep ensemble of tuberlens probes.

A *deep ensemble* here is the standard construction: fit ``n`` probes of the same
architecture on the **same** activations, varying only the training seed, and use
the mean of their scores as the probe's output. Everything upstream of the fit —
the base/red-team split, the message transforms, the activation extraction — is
shared and computed once, so the marginal cost of a member is one probe-head fit
over already-materialized activations, not another pass through the extraction
LLM.

:class:`EnsembleProbe` is deliberately *duck-typed* against tuberlens' ``Probe``
rather than a subclass of it: it is pickled to the same ``probe_iter{N}.pkl``
path a single probe would be, and read back by everything that consumes a probe
here (``ProbeJudge``, ``evaluation.evaluate_probe`` → tuberlens'
``get_performances``, and ``retrain._infer_probe_spec`` on the next cycle). Those
call ``model_name`` / ``layer`` / ``pos_class_label`` / ``neg_class_label`` /
``description`` / ``predict_proba`` / ``predict_proba_from_inputs`` and nothing
else, so matching that surface is what makes an ensemble a drop-in replacement.
Not subclassing also keeps this module importable without pulling in tuberlens
(``probe_judge`` imports it eagerly; tuberlens is imported lazily throughout).

The averaging happens on **probabilities**, before thresholding — so the probe's
predicted class is ``mean_i p_i >= threshold``, not a vote over the members'
individual predictions. That is what makes the judge see one averaged score and
one prediction derived from it, exactly as it would for a single probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Upper bound on `probe.ensemble_size` / `--ensemble-size`. Members are cheap to
# fit (the activations are already in memory) but each one is another full
# training run over the training split, so the ceiling keeps a typo from turning
# one retrain into an overnight job.
MAX_ENSEMBLE_SIZE = 10

# The ensemble's training seeds, drawn once at random and pinned here — member `i`
# is always fit under ENSEMBLE_SEEDS[i], for every run, config and box.
#
# The alternative (walking `--seed + i`) makes a member's identity depend on a flag
# that also governs the train/val split and the eval subsampling, so two runs that
# differ only in `--seed` produce ensembles that differ in *two* ways at once and
# can't be compared member-for-member. Pinning the fit seeds here separates the
# concerns: `--seed` still moves the data (split, subsample), and this list alone
# fixes the weight init and batch order of each member. It also means an
# n-member ensemble and an (n+1)-member one share their first n members' seeds, so
# growing an ensemble adds a member rather than reshuffling all of them.
#
# Treat these as frozen: changing a value silently changes every ensemble probe
# trained afterwards, and nothing in the probe pickle would flag the mismatch
# except `EnsembleProbe.member_seeds`, which records what was actually used.
ENSEMBLE_SEEDS = (3699, 14431, 23529, 26229, 26660, 42624, 43521, 54184, 65963, 69051)

assert len(ENSEMBLE_SEEDS) == MAX_ENSEMBLE_SIZE, (
    "ENSEMBLE_SEEDS must supply one seed per allowed member; raising "
    "MAX_ENSEMBLE_SIZE means appending seeds (never reordering the existing ones)."
)

# ProbeType names whose fit is a deterministic function of the training data:
# difference-of-means and LDA are closed-form, and SklearnProbe's logistic
# regression is solved by lbfgs with a fixed `random_state` hyperparam. Varying
# the seed does not diversify these, so an ensemble of them is n identical
# members — worth warning about rather than silently burning the compute.
DETERMINISTIC_ARCHS = frozenset({"sklearn", "difference_of_means", "lda"})


@dataclass
class EnsembleProbe:
    """``n`` independently-seeded probes whose scores are averaged into one score.

    ``members`` share architecture, training data and metadata; they differ only
    in the seed their weight initialization and batch shuffling were drawn from
    (``member_seeds[i]``, recorded so a run is reproducible from the pickle
    alone). The class-label / model / layer metadata is copied off member 0 at
    construction so this object answers metadata questions without unwrapping —
    see the "probe metadata is the source of truth" convention.
    """

    members: list[Any]
    member_seeds: list[int] = field(default_factory=list)
    model_name: str | None = None
    layer: int | None = None
    description: str | None = None
    pos_class_label: str = "positive"
    neg_class_label: str = "negative"

    @classmethod
    def from_members(
        cls, members: list[Any], member_seeds: list[int] | None = None
    ) -> "EnsembleProbe":
        """Wrap already-fitted probes, inheriting metadata from the first member."""
        if not members:
            raise ValueError("EnsembleProbe needs at least one member probe.")
        first = members[0]
        return cls(
            members=list(members),
            member_seeds=list(member_seeds or []),
            model_name=getattr(first, "model_name", None),
            layer=getattr(first, "layer", None),
            description=getattr(first, "description", None),
            pos_class_label=getattr(first, "pos_class_label", "positive"),
            neg_class_label=getattr(first, "neg_class_label", "negative"),
        )

    def __len__(self) -> int:
        return len(self.members)

    @property
    def hyper_params(self) -> dict:
        """The members' shared training hyperparameters (read off member 0)."""
        return getattr(self.members[0], "hyper_params", None) or {}

    # ------------------------------------------------------------------ #
    # Probe surface
    # ------------------------------------------------------------------ #

    def predict_proba(self, dataset):
        """Mean positive-class probability across members, shape ``(batch_size,)``.

        ``dataset`` must already carry activations (as it does when it reaches
        tuberlens' ``get_performances``); each member reads them off the same
        dataset, so the activations are never recomputed per member.
        """
        return self._mean_proba(dataset)

    def predict(self, dataset):
        """Labels from the *averaged* score, not a vote over member predictions."""
        from tuberlens.interfaces.dataset import Label

        return [Label.from_int(p > 0.5) for p in self._mean_proba(dataset)]

    def predict_proba_from_inputs(
        self,
        inputs,
        model,
        layer: int | None = None,
        start_turn_index: int | None = None,
        end_turn_index: int | None = None,
    ):
        """Score raw conversations, extracting activations **once** for all members.

        Mirrors ``PytorchProbe.predict_proba_from_inputs``, but the forward pass
        through the extraction LLM — which dominates the cost of scoring a
        red-team submission on a gemma-sized probe — is done a single time and
        the resulting activations are handed to every member. Delegating to the
        members' own ``predict_proba_from_inputs`` instead would run ``n``
        extractions per submission.
        """
        from tuberlens.interfaces.dataset import Dataset
        from tuberlens.probes.pytorch_probes import filter_activations_by_turns

        layer_used = layer if layer is not None else self.layer
        if layer_used is None:
            raise ValueError("EnsembleProbe has no layer and none was passed.")

        activations = model.get_activations(inputs, layer=layer_used)

        # Turn filtering is a property of how the probe was built, so it is
        # identical across members; read it off member 0 as PytorchProbe reads
        # it off itself.
        first = self.members[0]
        if start_turn_index is None:
            start_turn_index = getattr(first, "start_turn_index", None)
        if end_turn_index is None:
            end_turn_index = getattr(first, "end_turn_index", None)
        if start_turn_index is not None or end_turn_index is not None:
            activations = filter_activations_by_turns(
                activations, inputs, model, start_turn_index, end_turn_index
            )
        activations.attention_mask = activations.attention_mask.bool()

        dataset = Dataset(
            inputs=list(inputs),
            ids=[str(i) for i in range(len(inputs))],
            other_fields={
                "activations": activations.activations,
                "attention_mask": activations.attention_mask,
                "input_ids": activations.input_ids,
            },
        )
        return self._mean_proba(dataset)

    def per_token_predictions(self, dataset):
        """Not defined for an ensemble — iterate ``.members`` for per-member maps.

        The per-token output is architecture-dependent (``PytorchAdamClassifier``
        returns a 3-tuple of logits/attention scores/weights), so there is no
        single averaging rule that is correct across architectures. Nothing in
        this repo calls it; raising is honest where guessing would not be.
        """
        raise NotImplementedError(
            "EnsembleProbe does not average per-token predictions; call "
            "per_token_predictions on the individual probes in .members."
        )

    def _mean_proba(self, dataset):
        import numpy as np

        per_member = [
            np.asarray(m.predict_proba(dataset), dtype=float) for m in self.members
        ]
        return np.mean(per_member, axis=0)


def iter_probe_members(probe: Any) -> list[Any]:
    """The individual probes behind ``probe`` — ``[probe]`` when it is not an ensemble.

    Lets code that has to reach *inside* a probe (e.g. moving the classifier's
    torch module onto the right device/dtype after a CPU unpickle) treat the
    single-probe and ensemble cases uniformly.
    """
    if isinstance(probe, EnsembleProbe):
        return list(probe.members)
    return [probe]


def ensemble_size(probe: Any) -> int:
    """Number of members behind ``probe`` (1 for an ordinary single probe)."""
    return len(iter_probe_members(probe))
