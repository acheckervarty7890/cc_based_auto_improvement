"""The generator: writes labelled training samples for a probe, one batch per call.

Nothing adversarial happens here. The generator is shown the probe's concept (its
two class labels and ``description``), the judge's rolling memo, and one *direction*
— a short brief for what this batch should cover — and is asked for ``m`` samples,
``m/2`` per class, each a conversation plus the label the generator itself assigns.
Whether a batch was worth writing is decided downstream, by training a probe on it
and reading the dev AUROC (see ``cli.iterative_generate_main``); the generator never
sees a verdict within an iteration, only the judge's memo at the next one.

Two providers, picked per model: ``claude_sdk`` (Anthropic Messages API — no Agent
SDK or tools are involved, the model just answers) and ``openrouter`` (the ``openai``
SDK pointed at OpenRouter). Batches of one iteration run concurrently under
``generator.concurrency``; batch ``k`` is written by ``models[k % len(models)]``.

Every generated sample passes three guards before it counts: a **length** guard
(``generator.max_sample_tokens`` through :class:`TokenBudget` — the probe reads at
most 1024 tokens, so a longer sample would be trained on truncated), a **label**
guard (the label must normalize to one of the probe's two), and a **novelty** guard
(never seen in any earlier batch of this run, accepted or not, nor in another batch
in flight — :meth:`BatchStore.reserve`). A reply short of ``m`` usable samples gets
up to ``generator.max_retries`` in-context top-up asks naming only what is missing.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any

from agentic_redteam import circuit_breaker as breaker
from agentic_redteam.circuit_breaker import OpenRouterOutageError
from agentic_redteam.config import GeneratorConfig, GeneratorModel
from agentic_redteam.json_extract import extract_json_values, extract_string_list
from agentic_redteam.persistence import (
    BatchStore,
    Conversation,
    GeneratedSample,
    RunLogger,
)
from agentic_redteam.token_budget import TokenBudget

# Retries for transient OpenRouter failures (429/5xx/empty choices). Connection
# failures are bounded by the circuit breaker's outage clock instead — see
# _openrouter_create_with_retry.
_OPENROUTER_MAX_ATTEMPTS = 4

_VALID_ROLES = ("user", "assistant", "system")


@dataclass(frozen=True)
class ProbeMeta:
    """What the generator and judge are told about the probe — read off the probe."""

    pos_class_label: str
    neg_class_label: str
    description: str = ""
    model_name: str = ""


@dataclass
class BatchGeneration:
    """What one batch's generation produced, before it is scored."""

    samples: list[GeneratedSample] = field(default_factory=list)
    n_calls: int = 0
    n_dropped_too_long: int = 0
    n_dropped_duplicate: int = 0
    n_dropped_bad_label: int = 0
    error: str = ""
    replies: list[str] = field(default_factory=list)

    def count(self, label: str) -> int:
        return sum(1 for s in self.samples if s.label == label)


# --------------------------------------------------------------------------- #
# Prompts.
# --------------------------------------------------------------------------- #


def build_generator_system_prompt(
    config: GeneratorConfig, probe: ProbeMeta, memo: str = ""
) -> str:
    """The config's ``# Generator`` prompt plus the concrete probe context.

    The concept is stated verbatim from the probe's ``description`` — the generator
    labels its own samples, so it has to work from the same definition the probe is
    meant to learn. The judge's ``memo`` (what helped, what is exhausted) goes last, as
    the most recent signal.
    """
    prompt = (
        config.system_prompt.strip()
        + "\n\n"
        + "## Classifier being trained\n"
        + f"- Positive class label: '{probe.pos_class_label}'\n"
        + f"- Negative class label: '{probe.neg_class_label}'\n"
        + f"- What the labels mean: {probe.description or '(no description provided)'}\n"
    )
    if config.max_sample_tokens > 0:
        prompt += (
            "\n## Length limit\n"
            f"- Each sample must be at most {config.max_sample_tokens} tokens in total "
            f"(roughly {config.max_sample_tokens * 3} characters across all of its "
            "messages, roles and formatting included). Longer samples are discarded, "
            "so keep every sample comfortably under the limit.\n"
        )
    prompt += (
        "\n## Output format\n"
        "Return ONLY a JSON array inside a single ```json fence. Each element is one "
        "sample:\n"
        "```json\n"
        "[\n"
        f'  {{"label": "{probe.pos_class_label}", "messages": [{{"role": "user", '
        '"content": "..."}, {"role": "assistant", "content": "..."}]},\n'
        f'  {{"label": "{probe.neg_class_label}", "messages": [...]}}\n'
        "]\n"
        "```\n"
        f"- `label` must be exactly '{probe.pos_class_label}' or "
        f"'{probe.neg_class_label}'.\n"
        "- `messages` is the conversation, in order; roles are 'user', 'assistant' "
        "or 'system'.\n"
        "- No commentary outside the fence.\n"
    )
    if memo:
        prompt += "\n## Guidance from the judge (what has and has not helped so far)\n" + memo.strip() + "\n"
    return prompt


def _batch_request(probe: ProbeMeta, direction: str, m: int, batch_index: int, n_batches: int) -> str:
    half = m // 2
    return (
        f"Write {m} training samples for this batch (batch {batch_index + 1} of "
        f"{n_batches} in this round): exactly {half} labelled "
        f"'{probe.pos_class_label}' and {half} labelled '{probe.neg_class_label}'.\n\n"
        f"## Direction for this batch\n{direction.strip()}\n\n"
        "## Requirements\n"
        "- Every sample is a realistic conversation that fits the direction above.\n"
        "- Assign each label honestly according to the definition of the classes — "
        "the label is the ground truth the classifier will be trained on, so a "
        "mislabelled sample teaches it the wrong thing.\n"
        "- Make the two classes hard to tell apart on surface features alone: pair "
        "similar topics, tones, lengths and structures across the two labels, so what "
        "separates them is the concept itself.\n"
        "- Vary topics, personas, conversation lengths and turn counts within the "
        "direction; no near-duplicates of each other.\n"
    )


def _topup_request(
    probe: ProbeMeta,
    need_pos: int,
    need_neg: int,
    got_pos: int,
    got_neg: int,
    dropped: dict[str, int],
) -> str:
    reasons = []
    if dropped.get("too_long"):
        reasons.append(f"{dropped['too_long']} over the length limit")
    if dropped.get("duplicate"):
        reasons.append(f"{dropped['duplicate']} duplicating samples already in the training pool")
    if dropped.get("bad_label"):
        reasons.append(f"{dropped['bad_label']} with an unrecognized label or malformed messages")
    dropped_note = f" ({', '.join(reasons)} were discarded)" if reasons else ""
    asks = []
    if need_pos > 0:
        asks.append(f"{need_pos} more labelled '{probe.pos_class_label}'")
    if need_neg > 0:
        asks.append(f"{need_neg} more labelled '{probe.neg_class_label}'")
    return (
        f"Received {got_pos} usable '{probe.pos_class_label}' and {got_neg} usable "
        f"'{probe.neg_class_label}' samples so far{dropped_note}. Write "
        + " and ".join(asks)
        + ", different from the ones you already wrote, under the same direction and "
        "in the same JSON format. Return only the new samples."
    )


def _proposal_request(probe: ProbeMeta, n: int, existing: list[str]) -> str:
    text = (
        f"Before any samples are written, propose {n} distinct directions for {n} "
        "batches of training samples for this classifier. A direction is a 1-3 "
        "sentence brief for one batch: the domain or situation type, the conversation "
        "structure, and what separates the two classes there. Cover different regions "
        "of the space of conversations the classifier will meet — different domains, "
        "registers, lengths, and ways the concept can be present or absent — rather "
        "than variations on one theme.\n\n"
        "Return ONLY a JSON array of strings inside a single ```json fence, one string "
        "per direction."
    )
    if existing:
        text += "\n\nDirections already taken (propose ones that differ from these):\n" + "\n".join(
            f"- {d}" for d in existing
        )
    return text


# --------------------------------------------------------------------------- #
# Reply parsing and sample guards.
# --------------------------------------------------------------------------- #


def normalize_label(raw: Any, pos: str, neg: str) -> str | None:
    """Map a generator-written label onto one of the probe's two, or None."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    for label in (pos, neg):
        if text == label or text.lower() == label.lower():
            return label
    low = text.lower().replace("_", " ").replace("-", " ")
    if low in ("positive", "pos", "1", "true"):
        return pos
    if low in ("negative", "neg", "0", "false"):
        return neg
    # A label string with the class name embedded, e.g. "label: high-stakes".
    hits = [label for label in (pos, neg) if label.lower() in text.lower()]
    if len(hits) == 1:
        return hits[0]
    return None


def _coerce_sample(obj: Any, pos: str, neg: str) -> tuple[GeneratedSample | None, str]:
    """One parsed JSON element → (sample, "") or (None, reason)."""
    if not isinstance(obj, dict):
        return None, "bad_label"
    msgs = obj.get("messages")
    if msgs is None:
        msgs = obj.get("conversation")
    if not isinstance(msgs, list) or not msgs:
        return None, "bad_label"
    label = normalize_label(obj.get("label"), pos, neg)
    if label is None:
        return None, "bad_label"
    out = []
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return None, "bad_label"
        role = str(m["role"]).strip().lower()
        if role not in _VALID_ROLES:
            return None, "bad_label"
        content = m["content"]
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        out.append({"role": role, "content": content})
    if not any(m["content"].strip() for m in out):
        return None, "bad_label"
    return GeneratedSample(conversation=Conversation.from_messages(out), label=label), ""


def parse_samples(text: str | None, pos: str, neg: str) -> tuple[list[GeneratedSample], int]:
    """Every well-formed sample in a generator reply, plus how many elements were malformed.

    Accepts a JSON array of samples, a ``{"samples": [...]}`` wrapper, or — when a
    model forgot the array — a run of bare sample objects. Elements that fail the
    shape check are counted and skipped rather than failing the whole reply.
    """
    malformed = 0

    def _accept(value: Any):
        nonlocal malformed
        if isinstance(value, dict):
            for key in ("samples", "examples", "conversations", "data"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
            else:
                # A bare sample object is accepted only when it IS a sample; anything
                # else dict-shaped is not this reply's payload.
                sample, _ = _coerce_sample(value, pos, neg)
                return [sample] if sample else None
        if not isinstance(value, list):
            return None
        found: list[GeneratedSample] = []
        for item in value:
            sample, reason = _coerce_sample(item, pos, neg)
            if sample is None:
                malformed += 1
            else:
                found.append(sample)
        return found if found else None

    groups = extract_json_values(text, _accept)
    out: list[GeneratedSample] = []
    seen: set[str] = set()
    for group in groups:
        for s in group:
            if s.key not in seen:
                seen.add(s.key)
                out.append(s)
    return out, malformed


# --------------------------------------------------------------------------- #
# Provider calls.
# --------------------------------------------------------------------------- #


async def _openrouter_create_with_retry(client, *, model, messages, max_tokens):
    """``chat.completions.create`` with transient retries reported to the breaker.

    Retries a 200 with empty ``choices`` (rate-limit / provider blip envelope) and
    ``openai.APIError`` / ``json.JSONDecodeError`` raised during the call. Connection
    failures are retried on the breaker's minutes-scale schedule until its outage
    clock trips (so a network back at minute 12 resumes this same batch); transient
    ones for ``_OPENROUTER_MAX_ATTEMPTS``; fatal ones (401/402) not at all.
    """
    import openai

    from agentic_redteam.openrouter_client import extract_openrouter_error

    retryable_exc = (openai.APIError, json.JSONDecodeError)
    breaker.raise_if_tripped()

    last_err: str | None = None
    last_exc: BaseException | None = None
    attempt = 0
    while True:
        try:
            response = await client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
        except retryable_exc as e:
            last_err = f"{type(e).__name__}: {e}"
            last_exc = e
        else:
            if response.choices:
                breaker.record_success()
                return response
            last_err = extract_openrouter_error(response) or "no choices and no error detail"
            last_exc = None
        kind = breaker.record_failure(
            last_exc if last_exc is not None else last_err,
            where=f"generator model {model!r}",
        )
        if kind == "fatal":
            break
        if kind != "connection" and attempt >= _OPENROUTER_MAX_ATTEMPTS - 1:
            break
        delay = breaker.backoff_delay(kind, attempt)
        if kind == "connection":
            print(
                f"  [openrouter] {model}: no connection ({last_err}); retrying in "
                f"{delay / 60:.1f} min (unreachable for {breaker.streak_seconds() / 60:.1f} "
                f"of {breaker.max_connection_outage_s() / 60:.0f} min)",
                file=sys.stderr,
            )
        await breaker.sleep_async(delay)
        attempt += 1
    raise RuntimeError(
        f"OpenRouter generator call for model {model!r} failed after {attempt + 1} "
        f"attempts: {last_err}"
    )


@dataclass
class Generator:
    """Binds a :class:`GeneratorConfig` to a probe and the SDK clients.

    ``call`` is the only thing that touches a provider; everything else is prompt
    assembly and reply parsing, which is what makes the fake-provider tests in
    ``scripts/verify_generation_loop.py`` cover the real code path.
    """

    config: GeneratorConfig
    probe: ProbeMeta
    token_budget: TokenBudget | None = None
    runlog: RunLogger | None = None
    _clients: dict[str, Any] = field(default_factory=dict)

    def warmup(self) -> None:
        """Build the SDK clients (and load the probe tokenizer) before the fan-out."""
        providers = {m.provider for m in self.config.models}
        if "claude_sdk" in providers and "anthropic" not in self._clients:
            import anthropic

            self._clients["anthropic"] = anthropic.AsyncAnthropic()
        if "openrouter" in providers and "openrouter" not in self._clients:
            from agentic_redteam.openrouter_client import make_async_client

            self._clients["openrouter"] = make_async_client()
        if self.token_budget is not None:
            self.token_budget.warmup()

    async def call(self, model: GeneratorModel, system: str, messages: list[dict[str, str]]) -> str:
        """One completion. Returns the reply text (``""`` for an empty reply)."""
        if model.provider == "claude_sdk":
            client = self._clients.get("anthropic")
            if client is None:
                import anthropic

                client = anthropic.AsyncAnthropic()
                self._clients["anthropic"] = client
            response = await client.messages.create(
                model=model.name,
                max_tokens=self.config.max_tokens,
                system=system,
                messages=messages,
            )
            return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        if model.provider == "openrouter":
            client = self._clients.get("openrouter")
            if client is None:
                from agentic_redteam.openrouter_client import make_async_client

                client = make_async_client()
                self._clients["openrouter"] = client
            response = await _openrouter_create_with_retry(
                client,
                model=model.name,
                messages=[{"role": "system", "content": system}, *messages],
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""
        raise ValueError(f"Unknown generator provider: {model.provider!r}")

    def model_for_batch(self, batch_index: int) -> GeneratorModel:
        return self.config.models[batch_index % len(self.config.models)]

    # ------------------------------------------------------------------ #
    # Batch generation.
    # ------------------------------------------------------------------ #

    def _admit(
        self, candidates: list[GeneratedSample], store: BatchStore | None, gen: BatchGeneration,
        have: dict[str, int], half: int,
    ) -> None:
        """Run the guards over a reply's samples and append the survivors to ``gen``.

        Class caps are enforced here too: once a class has ``half`` samples, extra ones
        of it are dropped (counted as duplicates of the ask, not the pool — they are
        released back so a later batch may still use them).
        """
        for s in candidates:
            if self.token_budget is not None and self.token_budget.overage(
                [m.to_dict() for m in s.conversation.messages]
            ) is not None:
                gen.n_dropped_too_long += 1
                continue
            if have[s.label] >= half:
                # Over-delivery of one class; not a defect worth reporting back.
                continue
            if store is not None and not store.reserve(s):
                gen.n_dropped_duplicate += 1
                continue
            gen.samples.append(s)
            have[s.label] += 1

    async def generate_batch(
        self,
        *,
        batch_index: int,
        n_batches: int,
        direction: str,
        memo: str,
        store: BatchStore | None,
        iteration: int = 0,
    ) -> BatchGeneration:
        """Generate one batch of ``config.batch_size`` samples under ``direction``.

        The first call asks for the whole batch; up to ``config.max_retries`` in-context
        follow-ups ask for whatever is still missing per class. Any exception other than
        an :class:`OpenRouterOutageError` is captured on the returned
        :class:`BatchGeneration` (``error``) so one dead batch never aborts the
        iteration; the outage error propagates, as everywhere in this repo.
        """
        model = self.model_for_batch(batch_index)
        pos, neg = self.probe.pos_class_label, self.probe.neg_class_label
        m = self.config.batch_size
        half = m // 2
        gen = BatchGeneration()
        have = {pos: 0, neg: 0}
        system = build_generator_system_prompt(self.config, self.probe, memo)
        messages: list[dict[str, str]] = [
            {"role": "user", "content": _batch_request(self.probe, direction, m, batch_index, n_batches)}
        ]
        try:
            for call_idx in range(self.config.max_retries + 1):
                reply = await self.call(model, system, messages)
                gen.n_calls += 1
                gen.replies.append(reply)
                candidates, malformed = parse_samples(reply, pos, neg)
                gen.n_dropped_bad_label += malformed
                before = len(gen.samples)
                self._admit(candidates, store, gen, have, half)
                if self.runlog is not None:
                    self.runlog.log(
                        "generator_call",
                        iteration=iteration,
                        batch=batch_index,
                        model=model.name,
                        call=call_idx,
                        parsed=len(candidates),
                        admitted=len(gen.samples) - before,
                        malformed=malformed,
                    )
                need_pos, need_neg = half - have[pos], half - have[neg]
                if need_pos <= 0 and need_neg <= 0:
                    break
                if call_idx == self.config.max_retries:
                    break
                messages = [
                    *messages,
                    {"role": "assistant", "content": reply or "(empty reply)"},
                    {
                        "role": "user",
                        "content": _topup_request(
                            self.probe,
                            max(need_pos, 0),
                            max(need_neg, 0),
                            have[pos],
                            have[neg],
                            {
                                "too_long": gen.n_dropped_too_long,
                                "duplicate": gen.n_dropped_duplicate,
                                "bad_label": gen.n_dropped_bad_label,
                            },
                        ),
                    },
                ]
        except OpenRouterOutageError:
            raise
        except Exception as e:  # noqa: BLE001 — one dead batch must not abort the iteration
            gen.error = f"{type(e).__name__}: {e}"
            if self.runlog is not None:
                self.runlog.log(
                    "generator_error", iteration=iteration, batch=batch_index,
                    model=model.name, error=gen.error,
                )
        return gen

    async def propose_directions(
        self, n: int, *, memo: str = "", existing: list[str] | None = None, iteration: int = 0,
    ) -> list[str]:
        """Ask the generator for ``n`` distinct directions (iteration 0, or to fill a
        judge reply that came back short). Pads with a free-choice brief if the model
        still under-delivers after one retry, so the loop always has ``n``."""
        existing = list(existing or [])
        model = self.config.models[0]
        system = build_generator_system_prompt(self.config, self.probe, memo)
        out: list[str] = []
        try:
            for _ in range(2):
                want = n - len(out)
                if want <= 0:
                    break
                reply = await self.call(
                    model, system,
                    [{"role": "user", "content": _proposal_request(self.probe, want, existing + out)}],
                )
                for d in extract_string_list(reply):
                    if d not in out and d not in existing and len(out) < n:
                        out.append(d)
                if self.runlog is not None:
                    self.runlog.log(
                        "directions_proposed", iteration=iteration, model=model.name,
                        wanted=want, got=len(out),
                    )
        except OpenRouterOutageError:
            raise
        except Exception as e:  # noqa: BLE001
            if self.runlog is not None:
                self.runlog.log("directions_error", iteration=iteration, error=f"{type(e).__name__}: {e}")
        while len(out) < n:
            out.append(
                f"Free choice #{len(out) + 1}: pick a domain, register and conversation "
                "structure not covered by the other batches of this round, and write "
                "samples where the two classes differ only in the concept itself."
            )
        return out


async def generate_batches(
    generator: Generator,
    *,
    iteration: int,
    directions: list[str],
    memo: str,
    store: BatchStore | None,
    batch_indices: list[int] | None = None,
) -> dict[int, BatchGeneration]:
    """Generate every batch of an iteration concurrently (bounded by ``concurrency``).

    ``batch_indices`` restricts the fan-out to a subset (a resumed iteration only
    regenerates the batches that were never scored). Returns ``{batch_index: result}``.
    """
    n = len(directions)
    indices = list(range(n)) if batch_indices is None else list(batch_indices)
    sem = asyncio.Semaphore(max(1, generator.config.concurrency))

    async def _one(k: int) -> tuple[int, BatchGeneration]:
        async with sem:
            result = await generator.generate_batch(
                batch_index=k, n_batches=n, direction=directions[k], memo=memo,
                store=store, iteration=iteration,
            )
            return k, result

    tasks = [asyncio.ensure_future(_one(k)) for k in indices]
    try:
        done = await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        raise
    return dict(done)


def generate_batches_sync(generator: Generator, **kwargs) -> dict[int, BatchGeneration]:
    return asyncio.run(generate_batches(generator, **kwargs))


def propose_directions_sync(generator: Generator, n: int, **kwargs) -> list[str]:
    return asyncio.run(generator.propose_directions(n, **kwargs))
