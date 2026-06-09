"""Wrap a tuberlens probe so we can score Conversation objects."""

from __future__ import annotations

import io
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from agentic_redteam.persistence import Conversation


class ProbeScoringError(Exception):
    """Raised when the probe cannot score a conversation (e.g. the model's chat
    template rejects the submitted messages). Caught by the submit handler so a
    single un-scoreable submission is reported back to the attacker instead of
    crashing the whole rotation."""


def _cpu_unpickle(f: io.BufferedIOBase) -> Any:
    """Unpickle a torch-containing object, forcing all tensors to CPU."""
    _orig = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    try:
        return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = _orig


@dataclass
class ProbeJudge:
    """Loads a pickled tuberlens probe and exposes (score, predicts_positive) for a Conversation."""

    probe: Any
    threshold: float
    error_type: str  # "false_positive" | "false_negative"
    # Message-loader transforms mirroring the training/eval path (see
    # retrain._apply_message_transforms / evaluation.evaluate_probe). Applied to
    # the attacker's submitted conversation before tokenizing so the probe scores
    # it on the same message representation it was trained/evaluated on — and so
    # chat templates that require strict role alternation (e.g. gemma-3-27b-it)
    # don't raise "Conversation roles must alternate" on consecutive same-role
    # turns the attacker emits. convert_tool_to_assistant is applied first (so it
    # doesn't create consecutive assistant turns), then combine_consecutive_messages.
    combine_consecutive_messages: bool = False
    convert_tool_to_assistant: bool = False
    _model: Any = None
    # Serializes the GPU forward pass. One ProbeJudge is shared across all
    # concurrent attacker sessions (see attacker.run_redteam), so without this
    # up to `concurrency` threads would run model forwards at once. For a model
    # loaded with device_map="auto" + CPU/disk offload (e.g. gemma-3-27b),
    # accelerate's offload hook resets each weight to the `meta` device after
    # every forward; concurrent forwards then collide ("Tensor on device meta
    # is not on the expected device cpu!"). The GPU is serial regardless, so
    # this lock costs nothing — the judge LLM calls still overlap.
    _score_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def load(
        cls,
        probe_path: str | Path,
        threshold: float,
        error_type: str,
        combine_consecutive_messages: bool = False,
        convert_tool_to_assistant: bool = False,
    ) -> "ProbeJudge":
        probe_path = Path(probe_path)
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe file not found: {probe_path}")
        with probe_path.open("rb") as f:
            probe = _cpu_unpickle(f)
        # We unpickle forcing CPU (safe across boxes), but tuberlens feeds
        # activations through ActivationDataset on global_settings.DEVICE /
        # .DTYPE (e.g. cuda/bfloat16 when a GPU is present). Reconcile the
        # classifier with that device+dtype so the linear layer and the
        # incoming activations live on the same device.
        if (
            hasattr(probe, "_classifier")
            and getattr(probe._classifier, "model", None) is not None
        ):
            from tuberlens.config import global_settings

            probe._classifier.model.to(
                device=global_settings.DEVICE, dtype=global_settings.DTYPE
            )
        if probe.model_name is None:
            raise ValueError("Loaded probe has no model_name; cannot run inference.")
        if probe.layer is None:
            raise ValueError("Loaded probe has no layer; cannot run inference.")
        return cls(
            probe=probe,
            threshold=threshold,
            error_type=error_type,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
        )

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

            # offload_buffers=True: when device_map="auto" offloads layers to
            # CPU/disk (e.g. gemma-3-27b), buffers must be offloaded too or
            # accelerate warns about insufficient GPU buffer space / risks OOM.
            self._model = LLMModel.load(
                self.probe.model_name,
                model_kwargs={"offload_buffers": True},
            )
        return self._model

    def warmup(self) -> None:
        """Force-load the underlying LLM so concurrent sessions don't race on lazy init."""
        self._ensure_model()

    def release(self) -> None:
        """Drop the loaded LLM and return its GPU memory to the CUDA driver.

        Models are loaded with ``device_map="auto"`` + ``max_memory=None``, so
        accelerate re-infers the GPU/CPU layer split from whatever GPU memory is
        *free at load time*. torch's caching allocator keeps freed memory
        reserved, so a model that is merely dereferenced still makes the next
        load see less free GPU — pushing more layers to CPU/disk offload and
        slowing every subsequent forward pass. Call this when a phase is done
        with the probe (e.g. after red-teaming, before retrain/eval reload the
        same base model) so each phase starts from a clean GPU.
        """
        if self._model is None:
            return
        self._model = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score(self, conversation: Conversation) -> float:
        from tuberlens.interfaces.dataset import Message as TLMessage

        model = self._ensure_model()
        tl_messages = [
            TLMessage(role=m.role, content=m.content) for m in conversation.messages
        ]
        # Reshape to match the training/eval message representation (and satisfy
        # alternation-strict chat templates). Order matches load_from /
        # retrain._apply_message_transforms: convert tool→assistant, then combine.
        if self.convert_tool_to_assistant or self.combine_consecutive_messages:
            from tuberlens.interfaces.dataset import LabelledDataset

            if self.convert_tool_to_assistant:
                tl_messages = LabelledDataset._convert_tool_to_assistant(tl_messages)
            if self.combine_consecutive_messages:
                tl_messages = LabelledDataset._combine_consecutive_messages(tl_messages)
        # tuberlens probes expect a list of conversations (each a list of Messages).
        # Hold the lock for the whole forward: HookedModel mutates the model
        # in-place (layer truncation) and accelerate's offload hooks juggle
        # weight devices, neither of which is thread-safe.
        with self._score_lock:
            try:
                probas = self.probe.predict_proba_from_inputs(
                    [tl_messages], model=model
                )
            except Exception as e:
                # DEBUG: dump the exact (post-transform) messages so we can see
                # what tripped the chat template (e.g. gemma role-alternation),
                # then surface it as a ProbeScoringError so the submit handler
                # can skip this one conversation instead of crashing the run.
                import sys

                raw_roles = [m.role for m in conversation.messages]
                print(
                    "\n[probe_judge.score] tokenization FAILED. "
                    f"transforms: convert_tool_to_assistant="
                    f"{self.convert_tool_to_assistant}, "
                    f"combine_consecutive_messages={self.combine_consecutive_messages}",
                    file=sys.stderr,
                )
                print(f"  raw roles      : {raw_roles}", file=sys.stderr)
                print(
                    f"  post-transform : {[m.role for m in tl_messages]}",
                    file=sys.stderr,
                )
                print("  full conversation that failed:", file=sys.stderr)
                for i, m in enumerate(tl_messages):
                    print(f"    --- message[{i}] role={m.role!r} ---", file=sys.stderr)
                    print(m.content, file=sys.stderr)
                raise ProbeScoringError(str(e)) from e
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
