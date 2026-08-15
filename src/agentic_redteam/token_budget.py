"""Length safeguard against tuberlens' 1024-token activation cap.

``LLMModel.get_activations`` pads *or truncates* every conversation to
``max_length=1024`` (``tuberlens/model.py:394``), so a longer conversation is
scored — and, if it becomes training data, *trained on* — from its first 1024
tokens only, with the tail silently dropped. That is not a hypothetical: 26% of
the ``mt`` eval split is over the cap, and those rows regress at three times the
rate of intact ones.

Red-team data is generated, not given, so the fix upstream is simply not to
produce over-long samples in the first place:

- the attacker's submissions are **dropped** before the probe/judge ever see
  them (``tools.handle_submit_conversation``), and
- the contrastive generator is **asked again**, told how long its last attempt
  was (``preprocessing.generate_contrastive_dataset``).

Counting has to match what tuberlens actually tokenizes, which has two traps:

1. ``tokenize_inputs`` looks like it strips the leading ``<bos>``
   (``token_dict[k] = v[:, 1:]``) but the following ``.to(device)`` line
   reassigns the original tensor and overwrites the slice — the strip is a
   no-op. The width reaching the 1024 cap therefore **includes** ``<bos>``;
   never subtract 1.
2. The chat template already emits the special tokens, so the tokenizer is
   called with ``add_special_tokens=False`` (tuberlens'
   ``default_tokenize_kwargs``). Counting with the default ``True`` adds a
   second ``<bos>`` and reads one token high.

:func:`count_tokens` mirrors ``tokenize_inputs`` exactly on both points, and the
message transforms (``convert_tool_to_assistant`` then
``combine_consecutive_messages``, in that order) are applied first so the count
is taken on the same representation the extraction path will see.

Everything degrades to "don't know" (``None``) rather than raising: a tokenizer
that can't be loaded, or a conversation the chat template rejects, must not cost
a submission that would otherwise have been scored. The probe's own error path
already reports malformed conversations back to the attacker.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

# tuberlens' get_activations max_length default — the width a conversation is
# padded or truncated to before the probe ever sees it.
MAX_ACTIVATION_TOKENS = 1024


@lru_cache(maxsize=8)
def _load_tokenizer(model_name: str) -> Any | None:
    """Load (and cache) the extraction model's tokenizer, or None if unavailable.

    Cached per model name: a rotation builds a budget per model run, and the
    tokenizer is the only expensive part. Returns None — rather than raising —
    when the tokenizer can't be resolved (offline, gated repo, bad name), which
    turns the whole safeguard into a no-op instead of failing the run.
    """
    try:
        from transformers import AutoTokenizer
        from tuberlens.config import global_settings

        # cache_dir mirrors LLMModel.load's, so we resolve the same files as the
        # extraction model does.
        return AutoTokenizer.from_pretrained(
            model_name, cache_dir=global_settings.CACHE_DIR
        )
    except Exception as exc:  # noqa: BLE001 — never block a run on the safeguard
        print(
            f"[token_budget] could not load tokenizer for {model_name!r} "
            f"({exc}); length safeguard disabled."
        )
        return None


def _as_dicts(messages: Sequence[Mapping[str, Any] | Any]) -> list[dict[str, str]]:
    """Normalize Conversation Messages / tuberlens Messages / dicts to plain dicts."""
    out: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, Mapping):
            role, content = msg.get("role", "user"), msg.get("content", "")
        else:
            role, content = getattr(msg, "role", "user"), getattr(msg, "content", "")
        out.append({"role": str(role), "content": str(content)})
    return out


def apply_message_transforms(
    messages: Sequence[Mapping[str, Any] | Any],
    *,
    combine_consecutive_messages: bool = False,
    convert_tool_to_assistant: bool = False,
) -> list[dict[str, str]]:
    """Reshape messages the way the extraction path will.

    Order matches ``LabelledDataset.load_from`` / ``retrain._apply_message_transforms``
    / ``ProbeJudge.score``: convert ``tool``→``assistant`` first (so it doesn't
    create consecutive assistant turns), then combine same-role neighbours.
    ``_fix_assistant_first`` is deliberately *not* applied — none of the three
    call sites above apply it either.
    """
    dicts = _as_dicts(messages)
    if not (combine_consecutive_messages or convert_tool_to_assistant):
        return dicts

    from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

    dialogue = [TLMessage(role=m["role"], content=m["content"]) for m in dicts]
    if convert_tool_to_assistant:
        dialogue = LabelledDataset._convert_tool_to_assistant(dialogue)
    if combine_consecutive_messages:
        dialogue = LabelledDataset._combine_consecutive_messages(dialogue)
    return [{"role": m.role, "content": m.content} for m in dialogue]


def count_tokens(
    model_name: str,
    messages: Sequence[Mapping[str, Any] | Any],
    *,
    combine_consecutive_messages: bool = False,
    convert_tool_to_assistant: bool = False,
) -> int | None:
    """Chat-template token width of ``messages`` under ``model_name``'s tokenizer.

    This is the number tuberlens' ``pad_or_truncate`` compares against 1024, so
    it includes the template's ``<bos>`` and every role header. Returns None when
    the tokenizer is unavailable or the chat template rejects the conversation
    (e.g. gemma's strict role alternation) — the caller then skips the check.
    """
    tokenizer = _load_tokenizer(model_name)
    if tokenizer is None:
        return None
    try:
        dicts = apply_message_transforms(
            messages,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
        )
        # Batched exactly like tokenize_inputs (one dialogue in a list), and with
        # add_special_tokens=False because the template already added them.
        rendered = tokenizer.apply_chat_template([dicts], tokenize=False)
        if isinstance(rendered, str):  # older transformers unwrap single batches
            rendered = [rendered]
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"][0])
    except Exception as exc:  # noqa: BLE001 — unknown length, not a failure
        print(f"[token_budget] could not count tokens ({exc}); allowing sample through.")
        return None


@dataclass(frozen=True)
class TokenBudget:
    """A per-run token cap bound to the extraction model and its transforms.

    ``max_tokens <= 0`` disables the safeguard entirely (``enabled`` is False and
    :meth:`count` returns None without touching a tokenizer).
    """

    model_name: str
    max_tokens: int = MAX_ACTIVATION_TOKENS
    combine_consecutive_messages: bool = False
    convert_tool_to_assistant: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.model_name) and self.max_tokens > 0

    def warmup(self) -> None:
        """Load the tokenizer now rather than on the first submission.

        The count runs on the asyncio loop thread (unlike the probe forward, which
        goes to a worker), so a first-call tokenizer download would stall every
        concurrent session. Cheap and idempotent — the load is cached per model.
        """
        if self.enabled:
            _load_tokenizer(self.model_name)

    def count(self, messages: Sequence[Mapping[str, Any] | Any]) -> int | None:
        if not self.enabled:
            return None
        return count_tokens(
            self.model_name,
            messages,
            combine_consecutive_messages=self.combine_consecutive_messages,
            convert_tool_to_assistant=self.convert_tool_to_assistant,
        )

    def overage(self, messages: Sequence[Mapping[str, Any] | Any]) -> int | None:
        """Token count if ``messages`` is over the cap, else None.

        None means "fine, or unknown" — the two cases callers treat identically,
        since an uncountable conversation must not be dropped.
        """
        n_tokens = self.count(messages)
        if n_tokens is not None and n_tokens > self.max_tokens:
            return n_tokens
        return None
