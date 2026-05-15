"""Wrap a tuberlens probe so we can score Conversation objects."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_redteam.persistence import Conversation


@dataclass
class ProbeJudge:
    """Loads a pickled tuberlens probe and exposes (score, predicts_positive) for a Conversation."""

    probe: Any
    threshold: float
    error_type: str  # "false_positive" | "false_negative"
    _model: Any = None

    @classmethod
    def load(
        cls, probe_path: str | Path, threshold: float, error_type: str
    ) -> "ProbeJudge":
        probe_path = Path(probe_path)
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe file not found: {probe_path}")
        with probe_path.open("rb") as f:
            probe = pickle.load(f)
        if probe.model_name is None:
            raise ValueError("Loaded probe has no model_name; cannot run inference.")
        if probe.layer is None:
            raise ValueError("Loaded probe has no layer; cannot run inference.")
        return cls(probe=probe, threshold=threshold, error_type=error_type)

    @property
    def pos_class_label(self) -> str:
        return getattr(self.probe, "pos_class_label", "positive") or "positive"

    @property
    def neg_class_label(self) -> str:
        return getattr(self.probe, "neg_class_label", "negative") or "negative"

    @property
    def description(self) -> str:
        return getattr(self.probe, "description", "") or ""

    @property
    def model_name(self) -> str:
        return self.probe.model_name

    @property
    def layer(self) -> int:
        return int(self.probe.layer)

    def _ensure_model(self):
        if self._model is None:
            from tuberlens.model import LLMModel  # imported lazily — heavy

            self._model = LLMModel.load(self.probe.model_name)
        return self._model

    def score(self, conversation: Conversation) -> float:
        from tuberlens.interfaces.dataset import Message as TLMessage

        model = self._ensure_model()
        tl_messages = [
            TLMessage(role=m.role, content=m.content) for m in conversation.messages
        ]
        # tuberlens probes expect a list of conversations (each a list of Messages)
        probas = self.probe.predict_proba_from_inputs([tl_messages], model=model)
        return float(probas[0])

    def evaluate(self, conversation: Conversation) -> tuple[float, bool]:
        """Return (probe_score, predicts_positive)."""
        s = self.score(conversation)
        return s, bool(s >= self.threshold)

    @property
    def true_class_label(self) -> str:
        """The class label a successful red-team sample is expected to actually belong to.

        For `error_type=false_positive` the probe says positive on a sample whose
        true class is negative — so the true class for a successful find is the
        negative class. (Used in the attacker prompt as a directional hint;
        success is determined separately by comparing probe vs judge labels.)
        """
        if self.error_type == "false_positive":
            return self.neg_class_label
        return self.pos_class_label

    def label_for(self, predicts_positive: bool) -> str:
        return self.pos_class_label if predicts_positive else self.neg_class_label
