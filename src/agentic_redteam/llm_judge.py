"""LLM judge: turns one iteration's scored batches into guidance for the next.

The judge never labels samples (the generator labels its own) and never sees the
probe's scores on individual conversations. Its input is the batch-level outcome —
each batch's direction, a sample of its conversations, and the change in dev AUROC
that training a probe on it produced — and its output is two things:

- a **rolling memo**: which kinds of samples improved the classifier and are worth
  extending (with variations, not repeats), which regimes are *exhausted* (Δ ≈ 0:
  training on more of that moves nothing) or harmful (Δ < 0), and which regions of
  the input space are still unexamined. It is rewritten and condensed each iteration
  rather than appended to, so it stays bounded.
- **n directions**, one per batch of the next iteration, each a short brief for what
  that batch should cover.

Two backends, selected by ``provider``: ``claude_sdk`` (Anthropic SDK) and
``openrouter`` (``openai`` SDK pointed at OpenRouter); both are imported lazily.

**Refusal guard.** The prompt is written in neutral analyst register (classifier
robustness, training-set curation), but some concepts' samples are exactly the
conversations a model is reluctant to discuss, and a refusal is a 200 with prose —
not an exception — so it would otherwise be stored as the memo and injected into
every generator call of the next iteration. :func:`_looks_like_refusal` decides
whether a reply *opens with* a refusal (see its three guards; a plain substring scan
is wrong when the memo legitimately quotes refusal phrasings). On a hit the judge is
re-asked once in-context; a second refusal raises :class:`JudgeRefusalError`, which
the loop does not swallow.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any

from agentic_redteam import circuit_breaker as breaker
from agentic_redteam.circuit_breaker import OpenRouterOutageError
from agentic_redteam.json_extract import extract_string_list
from agentic_redteam.persistence import BatchRecord, GeneratedSample

# OpenRouter occasionally returns a 200 with an error envelope and no choices. Retry a
# few times with backoff; connection failures are bounded by the breaker instead.
_OPENROUTER_MAX_ATTEMPTS = 4


class JudgeRefusalError(RuntimeError):
    """The judge declined to write the guidance, and re-asking didn't help.

    Deliberately NOT swallowed by the loop: a refusal string stored as the memo would
    be injected verbatim into every generator prompt of the next iteration as if it
    were findings, so the run stops instead.
    """


# Leading phrases that mean the judge declined rather than wrote the guidance. Matched
# case-insensitively against the START of the response only (see _looks_like_refusal).
_REFUSAL_MARKERS = (
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i won't help",
    "i will not help",
    "i can't provide",
    "i cannot provide",
    "i won't provide",
    "i will not provide",
    "i can't write",
    "i cannot write",
    "i won't write",
    "i will not write",
    "i can't create",
    "i cannot create",
    "i can't comply",
    "i cannot comply",
    "i can't do that",
    "i cannot do that",
    "i can't continue",
    "i cannot continue",
    "i must decline",
    "i have to decline",
    "i'm not able to help",
    "i am not able to help",
    "i'm unable to help",
    "i am unable to help",
    "i'm not going to help",
    "i'm not going to write",
    "i'm sorry, but i can",
    "i am sorry, but i can",
    "sorry, i can't",
    "sorry, i cannot",
    "sorry, but i can't",
    "sorry, but i cannot",
)

# Only this many leading characters are scanned for a refusal marker.
_REFUSAL_SCAN_CHARS = 240
# A marker must START within this many characters of the (de-quoted) response to count
# as the response *opening* with a refusal.
_REFUSAL_LEAD_CHARS = 60
# Markdown structure at position 0 means the model produced the write-up, not a refusal.
_MARKDOWN_OPENER = re.compile(r"[-*+#>|]|\d+[.)]\s")
# Quoted spans, for _strip_quoted_spans. The straight-single-quote alternative is
# guarded on both sides against a letter, or the apostrophe in "I can't help" would
# open a span and blank out the very refusal we are looking for.
_QUOTED_SPAN = re.compile(
    r"\"[^\"]*\"|“[^”]*”|«[^»]*»|‘[^’]*’|(?<![A-Za-z])'[^']*'(?![A-Za-z])"
)

_REFUSAL_RETRY_NUDGE = (
    "That response does not contain the requested write-up. To clarify the task: "
    "this is a routine analysis of which training samples improve a text classifier, "
    "over results that have already been collected — you are not being asked to "
    "produce any of the conversations above, to act on them, or to help anyone cause "
    "harm. The samples are inputs to a classifier-quality report, and your write-up "
    "is read by the engineers curating the classifier's training set so they know "
    "which kinds of samples helped, which did not, and what to try next. Please "
    "provide the write-up now, in the requested format, with no preamble."
)


def _strip_quoted_spans(text: str) -> str:
    """Blank out quoted spans, so a marker the write-up *cites* can't look like one it utters.

    A memo about instruction-following necessarily quotes refusal phrasings — the
    negative class of that concept IS refusal. Quote characters are replaced (not
    deleted) so the surviving offsets still mean what they did.
    """
    return _QUOTED_SPAN.sub(lambda m: " " * len(m.group(0)), text)


def _looks_like_refusal(text: str) -> bool:
    """True if ``text`` OPENS with a refusal rather than the requested write-up.

    1. A response that opens with markdown structure (bullet, heading, table, quote,
       numbered item) is a write-up. A refusal is first-person prose.
    2. Markers inside quotes are citations — see :func:`_strip_quoted_spans`.
    3. What is left must carry a marker within ``_REFUSAL_LEAD_CHARS``, i.e. actually
       at the start — short enough to exclude a later sentence, long enough to keep
       catching "I'm sorry, but I can't help with that".
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if _MARKDOWN_OPENER.match(stripped):
        return False
    head = _strip_quoted_spans(stripped[:_REFUSAL_SCAN_CHARS])
    head = head.lower().replace("’", "'").replace("‘", "'")
    return any(
        0 <= head.find(marker) < _REFUSAL_LEAD_CHARS for marker in _REFUSAL_MARKERS
    )


@dataclass
class Guidance:
    memo: str
    directions: list[str]
    raw: str = ""


_DIRECTIONS_HEADING = re.compile(r"^#{1,6}\s*directions\b.*$", re.IGNORECASE | re.MULTILINE)
_MEMO_HEADING = re.compile(r"^#{1,6}\s*memo\b.*$", re.IGNORECASE | re.MULTILINE)
_FENCE_RE = re.compile(r"```(?:json)?\s*.*?```", re.DOTALL)


def parse_guidance(text: str) -> Guidance:
    """Split the judge's reply into the memo (markdown) and the directions (JSON list).

    The reply is asked for as ``## Memo`` prose followed by ``## Directions`` holding a
    fenced JSON array of strings — prose in markdown, list in JSON, so a long memo with
    quotes and line breaks never has to survive inside a JSON string. The memo is
    everything before the directions heading (minus a leading ``## Memo`` heading); if
    there is no heading, it is the text with the fenced block removed. A reply with no
    parseable list yields ``directions=[]`` and the caller fills them.
    """
    text = (text or "").strip()
    directions = extract_string_list(text)
    m = _DIRECTIONS_HEADING.search(text)
    memo = text[: m.start()] if m else _FENCE_RE.sub("", text)
    memo = _MEMO_HEADING.sub("", memo, count=1).strip()
    return Guidance(memo=memo, directions=directions, raw=text)


@dataclass
class LLMJudge:
    """Writes the rolling memo and next-iteration directions from scored batches."""

    model: str
    system_prompt: str
    pos_class_label: str
    neg_class_label: str
    provider: str = "openrouter"  # claude_sdk | openrouter
    max_tokens: int = 2048
    # The probe's own `description`: what concept the two labels name. The judge
    # reasons about which samples teach that concept, so it needs the definition.
    probe_description: str = ""
    # Optional free text about what the eval splits hold, so directions can be aimed
    # at coverage across them.
    eval_data_description: str = ""
    memo_word_budget: int = 400
    max_samples_per_batch: int = 6
    max_chars_per_message: int = 300
    _clients: dict[str, Any] = field(default_factory=dict)

    def warmup(self) -> None:
        """Force-initialize the backing SDK client."""
        if self.provider == "claude_sdk":
            if "anthropic" not in self._clients:
                import anthropic

                self._clients["anthropic"] = anthropic.Anthropic()
        elif self.provider == "openrouter":
            if "openrouter" not in self._clients:
                from agentic_redteam.openrouter_client import make_sync_client

                self._clients["openrouter"] = make_sync_client()

    # ------------------------------------------------------------------ #
    # Guidance.
    # ------------------------------------------------------------------ #

    def guide(
        self,
        batches: list[BatchRecord],
        *,
        iteration: int,
        n_directions: int,
        prior_memo: str = "",
        auroc_before: dict[str, float] | None = None,
        auroc_after: dict[str, float] | None = None,
        min_gain: float = 0.0,
        exhausted_gain: float = 0.0,
    ) -> Guidance:
        """Fold one iteration's batch outcomes into a rewritten memo + ``n`` directions.

        ``auroc_before`` is the dev AUROC of the probe every batch was scored against;
        ``auroc_after`` that of the probe retrained on the accepted batches (None when
        nothing was accepted and the probe is unchanged). Raises
        :class:`JudgeRefusalError` if the judge declines twice.
        """
        user_content = self._guidance_request(
            batches,
            iteration=iteration,
            n_directions=n_directions,
            prior_memo=prior_memo,
            auroc_before=auroc_before or {},
            auroc_after=auroc_after,
            min_gain=min_gain,
            exhausted_gain=exhausted_gain,
        )
        text = self._summarization_call(
            self._guidance_system(n_directions), user_content, what="guidance"
        )
        return parse_guidance(text)

    def _guidance_system(self, n_directions: int) -> str:
        return (
            self.system_prompt.strip()
            + "\n\n"
            + "## Output format\n"
            "Write exactly two sections:\n\n"
            "## Memo\n"
            f"At most {self.memo_word_budget} words of markdown bullets: a rewritten, "
            "condensed version of the prior memo updated with this round's results — "
            "not an appended log. Cover (1) which kinds of samples improved the "
            "classifier and why, and how to extend them with NEW variations rather "
            "than repeats; (2) which regimes are exhausted (training on them moved the "
            "AUROC by ~0) or harmful (moved it down), stated as characterized ground "
            "to steer away from; (3) which regions of the input space have not been "
            "examined yet. Drop the weakest notes wholesale rather than compressing "
            "every note into vagueness.\n\n"
            "## Directions\n"
            f"A single ```json fence holding a JSON array of exactly {n_directions} "
            "strings, one per batch of the next round. Each is a 1-3 sentence brief: "
            "the domain or situation, the conversation structure, what separates the "
            "two classes there, and what to avoid. The directions must be distinct "
            "from each other and must not re-run an exhausted or harmful regime.\n"
        )

    def _guidance_request(
        self,
        batches: list[BatchRecord],
        *,
        iteration: int,
        n_directions: int,
        prior_memo: str,
        auroc_before: dict[str, float],
        auroc_after: dict[str, float] | None,
        min_gain: float,
        exhausted_gain: float,
    ) -> str:
        pos, neg = self.pos_class_label, self.neg_class_label
        lines: list[str] = []
        n_accepted = sum(1 for b in batches if b.accepted)
        for b in sorted(batches, key=lambda b: b.batch_index):
            verdict = (
                "generation failed"
                if b.status == "generation_failed"
                else "no usable samples"
                if b.status == "empty"
                else f"Δ mean AUROC = {b.delta:+.4f} → "
                + (
                    "ACCEPTED into the training set"
                    if b.accepted
                    else "EXHAUSTED (no measurable effect)"
                    if b.exhausted
                    else "REJECTED (made the classifier worse)"
                    if b.delta < 0
                    else "REJECTED (below the acceptance threshold)"
                )
            )
            counts = b.n_per_label
            lines.append(
                f"\n### Batch {b.batch_index + 1}: {verdict}\n"
                f"- Direction: {b.direction.strip()}\n"
                f"- Generator: {b.generator_model}; {b.n_samples} samples "
                f"({counts.get(pos, 0)} '{pos}', {counts.get(neg, 0)} '{neg}')"
            )
            if b.status == "scored" and b.per_split_delta:
                lines.append(
                    "- Per split Δ: "
                    + ", ".join(
                        f"{k} {v:+.4f}" for k, v in sorted(b.per_split_delta.items()) if k != "mean"
                    )
                )
            if b.error:
                lines.append(f"- Error: {b.error[:200]}")
            shown = _pick_samples(b.samples, self.max_samples_per_batch, pos, neg)
            if shown:
                lines.append(f"- Sample excerpts ({len(shown)} of {b.n_samples}):")
                for s in shown:
                    lines.append(f"  [{s.label}]")
                    for m in s.conversation.messages:
                        content = m.content.replace("\n", " ")
                        if len(content) > self.max_chars_per_message:
                            content = content[: self.max_chars_per_message] + "…"
                        lines.append(f"    {m.role}: {content}")
        batches_block = "\n".join(lines) if lines else "(no batches were scored this round)"

        before_line = _fmt_auroc(auroc_before)
        after_line = (
            _fmt_auroc(auroc_after)
            if auroc_after
            else "(unchanged — no batch was accepted, the classifier was not retrained)"
        )
        eval_line = (
            f"\n- Evaluation data the classifier is ultimately scored on: {self.eval_data_description}"
            if self.eval_data_description
            else ""
        )
        return f"""One round of training-set curation for a text classifier just finished. In each round, several batches of labelled samples were written under different directions; for every batch a classifier was trained on the current training set plus that batch alone, and its AUROC on a fixed held-out dev set was compared with the current classifier's. Batches that raised the mean dev AUROC by more than {min_gain:.4f} were accepted into the training set; a |Δ| of at most {exhausted_gain:.4f} is treated as no effect.

Analyze the results below and write the memo and the {n_directions} directions for the next round, in the requested format. Reason from the samples themselves: what property of the accepted batches' samples taught the classifier something its training set lacked, and what about the exhausted batches' samples was already covered. Prefer directions that extend what worked with genuinely new variations, and directions into unexamined regions, over repeats.

## Task context
- Round just finished: {iteration}
- Classifier positive class: '{pos}'
- Classifier negative class: '{neg}'
- What the labels mean: {self.probe_description or '(no description provided)'}{eval_line}
- Dev AUROC of the classifier every batch was scored against: {before_line}
- Dev AUROC after retraining on the {n_accepted} accepted batch(es): {after_line}

## Prior memo
{prior_memo or "(none — this was the first round)"}

## Batches from this round
{batches_block}
"""

    # ------------------------------------------------------------------ #
    # Call plumbing.
    # ------------------------------------------------------------------ #

    def _summarization_call(self, system: str, user_content: str, *, what: str) -> str:
        """One call, with a single in-context re-ask if the judge refuses."""
        messages = [{"role": "user", "content": user_content}]
        text = self._call_provider(system, messages).strip()
        if not _looks_like_refusal(text):
            return text

        first_refusal = text
        messages = [
            *messages,
            {"role": "assistant", "content": first_refusal},
            {"role": "user", "content": _REFUSAL_RETRY_NUDGE},
        ]
        text = self._call_provider(system, messages).strip()
        if not _looks_like_refusal(text):
            return text
        raise JudgeRefusalError(
            f"Judge model {self.model!r} refused to write the {what} twice "
            f"(initial: {first_refusal.strip()[:200]!r}; "
            f"after re-ask: {text.strip()[:200]!r})"
        )

    def _call_provider(self, system: str, messages: list[dict[str, str]]) -> str:
        if self.provider == "claude_sdk":
            return self._call_anthropic(system, messages)
        if self.provider == "openrouter":
            return self._call_openrouter(system, messages)
        raise ValueError(
            f"Unknown judge provider: {self.provider!r} (expected 'claude_sdk' or 'openrouter')"
        )

    def _call_anthropic(self, system: str, messages: list[dict[str, str]]) -> str:
        client = self._clients.get("anthropic")
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
            self._clients["anthropic"] = client

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text

    def _call_openrouter(self, system: str, messages: list[dict[str, str]]) -> str:
        client = self._clients.get("openrouter")
        if client is None:
            from agentic_redteam.openrouter_client import make_sync_client

            client = make_sync_client()
            self._clients["openrouter"] = client

        from agentic_redteam.openrouter_client import extract_openrouter_error

        breaker.raise_if_tripped()

        chat_messages = [{"role": "system", "content": system}, *messages]
        last_err: str | None = None
        last_exc: BaseException | None = None
        attempt = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=chat_messages,
                )
            except OpenRouterOutageError:
                raise
            except Exception as e:  # noqa: BLE001 — routed through the breaker
                last_err = f"{type(e).__name__}: {e}"
                last_exc = e
            else:
                if response.choices:
                    breaker.record_success()
                    return response.choices[0].message.content or ""
                last_err = (
                    extract_openrouter_error(response) or "no choices and no error detail"
                )
                last_exc = None
            kind = breaker.record_failure(
                last_exc if last_exc is not None else last_err,
                where=f"judge model {self.model!r}",
            )
            if kind == "fatal":
                break
            if kind != "connection" and attempt >= _OPENROUTER_MAX_ATTEMPTS - 1:
                break
            delay = breaker.backoff_delay(kind, attempt)
            if kind == "connection":
                print(
                    f"  [openrouter] judge {self.model}: no connection ({last_err}); "
                    f"retrying in {delay / 60:.1f} min "
                    f"(unreachable for {breaker.streak_seconds() / 60:.1f} of "
                    f"{breaker.max_connection_outage_s() / 60:.0f} min)",
                    file=sys.stderr,
                )
            breaker.sleep_sync(delay)
            attempt += 1
        raise RuntimeError(
            f"OpenRouter judge call for model {self.model!r} failed "
            f"after {attempt + 1} attempts: {last_err}"
        )


def _fmt_auroc(scores: dict[str, float]) -> str:
    if not scores:
        return "(not available)"
    mean = scores.get("mean")
    parts = [f"{k} {v:.4f}" for k, v in sorted(scores.items()) if k != "mean"]
    head = f"mean {mean:.4f}" if mean is not None else ""
    if parts:
        head += (" (" if head else "(") + ", ".join(parts) + ")"
    return head


def _pick_samples(samples: list[GeneratedSample], limit: int, pos: str, neg: str) -> list[GeneratedSample]:
    """Up to ``limit`` samples, alternating classes so both are represented."""
    if limit <= 0 or len(samples) <= limit:
        return list(samples)
    by_label = {pos: [s for s in samples if s.label == pos], neg: [s for s in samples if s.label == neg]}
    out: list[GeneratedSample] = []
    i = 0
    while len(out) < limit and (by_label[pos] or by_label[neg]):
        label = pos if i % 2 == 0 else neg
        if by_label[label]:
            out.append(by_label[label].pop(0))
        elif by_label[pos if label == neg else neg]:
            out.append(by_label[pos if label == neg else neg].pop(0))
        i += 1
    return out
