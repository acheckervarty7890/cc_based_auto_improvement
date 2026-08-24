"""Collation-style preprocessing for red-team successes.

Ports the two collation steps of tuberlens' ``collate_train_evaluate.py`` that
are applied to the "extra" data before it is merged with the base training set:

1. ``filter_dataset`` — drop confounders. A bag-of-words logistic-regression
   classifier is fit on the records; the examples it predicts most confidently
   (top ``filter_percentile``) are removed, on the theory that they are
   separable by surface lexical cues rather than the underlying concept.
2. ``generate_contrastive_dataset`` — for each record, ask an LLM to write a
   similar-looking conversation belonging to the opposite class, yielding
   contrastive pairs.

Both functions are generalized to arbitrary class labels (the originals
hard-coded ``"high-stakes"``/``"low-stakes"``): the caller passes
``pos_class_label`` / ``neg_class_label``, which come from the probe metadata.

Records are plain dicts: ``{text_key: [{"role", "content"}, ...], label_key: <label string>}``.
The contrastive generator uses this repo's own LLM clients (Anthropic SDK or
the OpenAI SDK pointed at OpenRouter) rather than litellm, runs requests on a
thread pool, and caches generated pairs to disk keyed by the source
conversation so accumulating red-team successes are not re-generated every
iteration.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from agentic_redteam import circuit_breaker as breaker
from agentic_redteam.circuit_breaker import OpenRouterOutageError
from agentic_redteam.token_budget import TokenBudget


# --------------------------------------------------------------------------- #
# Shared record helpers
# --------------------------------------------------------------------------- #


def _extract_messages(record: dict, text_key: str) -> list[dict[str, str]]:
    """Extract a list of ``{"role", "content"}`` messages from a record."""
    value = record.get(text_key, "")
    if isinstance(value, list):
        out = []
        for msg in value:
            if isinstance(msg, dict):
                out.append(
                    {
                        "role": str(msg.get("role", "user")),
                        "content": str(msg.get("content", "")),
                    }
                )
            else:
                out.append({"role": "user", "content": str(msg)})
        return out
    return [{"role": "user", "content": str(value)}]


def _is_well_formed_conversation(messages: Sequence[dict[str, str]]) -> bool:
    """True if ``messages`` is a valid chat: optional leading ``system`` message,
    then a user-first, alternating user/assistant body.

    Contrastive generation occasionally yields malformed conversations — most
    commonly when the provider *refuses* the generation request and returns its
    refusal as a single ``assistant`` message with no ``user`` turn. Such a
    conversation has no valid chat-template rendering: model chat templates
    require the (non-system) turns to start with ``user`` and alternate
    user/assistant, so it raises "Conversation roles must alternate ..." at
    activation time. Reject these here so they're skipped (and not cached) rather
    than poisoning the red-team training set.

    A single leading ``system`` message is allowed and skipped before checking
    alternation — it's standard and present in both the base training data and
    prompt-mode red-team scenarios (which are ``[{system}, {user}]``). Rejecting
    it here silently dropped every contrastive pair for such runs.
    """
    if not messages:
        return False
    body = messages[1:] if messages[0].get("role") == "system" else list(messages)
    if not body:
        return False
    for i, msg in enumerate(body):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg.get("role") != expected:
            return False
    return True


def _extract_text(record: dict, text_key: str) -> str:
    """Flatten a record's messages into one string for the bag-of-words classifier."""
    messages = _extract_messages(record, text_key)
    return " ".join(msg["content"] for msg in messages)


def _render_transcript(messages: Sequence[dict[str, str]]) -> str:
    return "\n".join(f"[{m['role']}] {m['content']}" for m in messages)


# --------------------------------------------------------------------------- #
# 1. Confounder filtering
# --------------------------------------------------------------------------- #


class BagOfWordsClassifier:
    """A simple bag-of-words classifier for filtering confounded examples."""

    def __init__(self) -> None:
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            max_features=20000,
            min_df=3,
            max_df=0.9,
            binary=False,
        )
        self.clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
        self._is_fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "BagOfWordsClassifier":
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self._is_fitted = True
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted first")
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)


def filter_dataset(
    records: Sequence[dict],
    pos_class_label: str,
    filter_percentile: float = 0.8,
    text_key: str = "inputs",
    label_key: str = "labels",
) -> list[dict]:
    """Drop the examples a bag-of-words classifier predicts most confidently.

    Generalized from tuberlens (which hard-coded ``"high-stakes"`` as the
    positive class): ``pos_class_label`` defines the positive class; everything
    else is treated as negative. If fewer than two classes are present (e.g.
    single-error-type successes), the records are returned unchanged.
    """
    if not records:
        return []

    labels = np.array(
        [1 if record.get(label_key) == pos_class_label else 0 for record in records]
    )
    if len(np.unique(labels)) < 2:
        return list(records)

    texts = [_extract_text(record, text_key) for record in records]
    classifier = BagOfWordsClassifier().fit(texts, labels.tolist())
    probs = classifier.predict_proba(texts)
    confidence = np.max(probs, axis=1)

    threshold = np.percentile(confidence, filter_percentile * 100)
    return [record for record, conf in zip(records, confidence) if conf <= threshold]


# --------------------------------------------------------------------------- #
# 2. Contrastive pair generation
# --------------------------------------------------------------------------- #

# The dataset carries the long class labels; we present shorter, plainer names to
# the LLM in the generation prompt only. This is display-only: the raw labels are
# still used everywhere that matters (cache key, the generated record's `labels` /
# `original_label` fields). Unknown labels (e.g. the high-stakes concept) pass
# through unchanged.
LABEL_SHORT = {
    "harmful_to_human": "unsafe",
    "not_harmful_to_human": "safe",
}

def _short_label(label: str) -> str:
    """Map a raw class label to its short prompt name (pass through if unknown)."""
    return LABEL_SHORT.get(label, label)


def _label_guidance_for(
    label: str, label_guidance: dict[str, str] | None
) -> str:
    """Look up per-label prompt guidance, accepting the raw or short label as key."""
    if not label_guidance:
        return ""
    text = label_guidance.get(label)
    if text is None:
        text = label_guidance.get(_short_label(label))
    return (text or "").strip()


def _length_instruction(max_sample_tokens: int) -> str:
    """Bullet telling the generator the hard length cap (empty when uncapped).

    Stated in the prompt as well as enforced afterwards: a pair that comes back
    over the cap costs a full regeneration, and the instruction removes most of
    them. It sits next to the "similar structure and length" bullet because it
    overrides it — a long source must yield a *shorter* pair, not an equally
    truncated one.
    """
    if max_sample_tokens <= 0:
        return ""
    return (
        f"- Keep the ENTIRE conversation under {max_sample_tokens} tokens "
        f"(roughly {max_sample_tokens * 3} characters across all messages). This is a "
        "hard limit: if the original is longer than that, write a shorter version "
        "rather than matching its length\n"
    )


def _length_retry_feedback(n_tokens: int, max_sample_tokens: int) -> str:
    """Follow-up nudge after an over-long generation, naming the measured length."""
    target = max(1, int(max_sample_tokens * 0.85))
    return (
        f"\n\nNote: your previous attempt was {n_tokens} tokens, over the "
        f"{max_sample_tokens}-token limit, so it was rejected. Write a shorter version "
        f"this time — aim for at most {target} tokens (about {target * 3} characters "
        "across all messages). Cut or condense actual content (fewer turns, shorter "
        "turns); the class of the conversation must stay the same."
    )


def _eval_data_instruction(eval_data_description: str) -> str:
    """Block constraining the generated pair to the evaluation data's form.

    The generated pairs become training data for the same probe the eval splits
    score, so a pair shaped unlike that data trains the probe on a distribution it is
    never measured on. The description is injected verbatim (as ``concept_description``
    is) and the instruction binds only what the description *states*: the same task,
    its class decided on the same terms, with everything the description leaves open
    (topic, wording, length, layout, number of parts, supplied material or not) free to
    vary and its examples explicitly not requirements. ``""`` when unset, so the prompt
    — and therefore the contrastive cache key — is exactly what it was before this
    existed.
    """
    text = (eval_data_description or "").strip()
    if not text:
        return ""
    return (
        "The data this classifier is evaluated on looks like this:\n"
        f"{text}\n\n"
        "Your generated conversation must stay within what that description actually "
        "constrains: it must be the same task, with its class decided on the same "
        "terms. Change the class, not the task — do not switch to a different task or "
        "to a way of belonging to the class that the description does not describe. "
        "Everything the description leaves open may vary freely: the topic, the "
        "wording, the length, the layout, how many parts the request has, whether the "
        "user supplies material to work from. Do not treat details of its examples as "
        "requirements.\n\n"
    )


def _generation_system_prompt(
    current_label: str,
    target_label: str,
    *,
    assistant_centric: bool = False,
    concept_description: str = "",
    label_guidance: dict[str, str] | None = None,
    eval_data_description: str = "",
    max_sample_tokens: int = 0,
) -> str:
    cur = _short_label(current_label)
    tgt = _short_label(target_label)
    # For concepts defined by the assistant's reply (e.g. harmful_to_human), tell
    # the model the label lives in the assistant turns; skip this for whole-scenario
    # concepts (e.g. high-stakes). Gated by the caller's assistant_centric flag.
    emphasis = (
        "- Crucially, it is the assistant's response (not the user's message) that "
        "determines this label, so focus your changes on the assistant turn(s)\n"
        if assistant_centric
        else ""
    )
    # Optional concept detail from the config: what the concept is (shown for both
    # directions) and what the *target* class specifically means. Injected verbatim
    # so a config can describe the concept as concretely as a hand-written prompt.
    concept_block = (
        f"About this classification concept:\n{concept_description.strip()}\n\n"
        if concept_description.strip()
        else ""
    )
    target_detail = _label_guidance_for(target_label, label_guidance)
    guidance_block = (
        f'\nWhat "{tgt}" means here:\n{target_detail}\n' if target_detail else ""
    )
    return (
        "You are an AI-safety research assistant supporting a red-teaming team. "
        "Your task is to help generate contrastive training pairs used to build and "
        "evaluate a safety classifier.\n\n"
        f"{concept_block}"
        f"{_eval_data_instruction(eval_data_description)}"
        f'You are given a conversation/scenario labeled "{cur}". Generate a '
        f'similar-looking conversation/scenario that belongs to the "{tgt}" '
        "class instead.\n\n"
        "The new scenario should:\n"
        "- Have similar structure and length\n"
        f"{_length_instruction(max_sample_tokens)}"
        "- Use similar language and style\n"
        "- Preserve the conversational format (roles) if applicable\n"
        f'- But clearly belong to "{tgt}" rather than "{cur}"\n'
        f"{emphasis}"
        f"{guidance_block}"
        "\n"
        "Respond with a single JSON object using these keys:\n"
        "- generated_messages: the new conversation as a list of message objects, each "
        "with 'role' and 'content' string fields\n"
        f'- explanation: a brief explanation of why it is "{tgt}"\n'
        "Output only the JSON object, with no surrounding text."
    )


def _parse_json_object(text: str) -> dict | None:
    """Best-effort parse of a single JSON object, tolerating code fences / prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1 and text[:nl].strip().lower() in ("json", ""):
            text = text[nl + 1 :]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


class _ContrastiveLLM:
    """Sync LLM caller for contrastive generation, over either provider."""

    def __init__(
        self,
        provider: str,
        model: str,
        max_tokens: int,
        *,
        assistant_centric: bool = False,
        concept_description: str = "",
        label_guidance: dict[str, str] | None = None,
        eval_data_description: str = "",
        max_sample_tokens: int = 0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.assistant_centric = assistant_centric
        self.concept_description = concept_description
        self.label_guidance = label_guidance or {}
        self.eval_data_description = eval_data_description
        # Length cap stated in the generation prompt. The measurement/rejection side
        # lives in generate_contrastive_dataset (it needs the probe's tokenizer);
        # 0 means uncapped and drops the instruction entirely.
        self.max_sample_tokens = max_sample_tokens
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            if self.provider == "claude_sdk":
                import anthropic

                self._client = anthropic.Anthropic()
            elif self.provider == "openrouter":
                from agentic_redteam.openrouter_client import make_sync_client

                self._client = make_sync_client()
            else:
                raise ValueError(f"Unknown preprocessing provider: {self.provider!r}")
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]],
        current_label: str,
        target_label: str,
        feedback: str = "",
    ) -> dict | None:
        """One generation call. ``feedback`` is appended to the user turn.

        The retry loop uses it to tell the model why the previous attempt was
        rejected (currently: it was over the token limit, and by how much). The
        call is stateless otherwise — each attempt re-sends the original
        conversation rather than continuing a chat — so the feedback has to
        travel here rather than in a message history.
        """
        client = self._ensure_client()
        system = _generation_system_prompt(
            current_label,
            target_label,
            assistant_centric=self.assistant_centric,
            concept_description=self.concept_description,
            label_guidance=self.label_guidance,
            eval_data_description=self.eval_data_description,
            max_sample_tokens=self.max_sample_tokens,
        )
        user = (
            f'Original "{_short_label(current_label)}" conversation:\n\n'
            f"{_render_transcript(messages)}\n\n"
            f'Now produce the "{_short_label(target_label)}" version as instructed.'
            f"{feedback}"
        )
        openrouter = self.provider != "claude_sdk"
        if not openrouter:
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
            except Exception as e:  # noqa: BLE001 — one bad generation shouldn't kill the run
                print(f"  [contrastive] generation failed: {e}")
                return None
            return _parse_json_object(text)

        # OpenRouter path. A connection failure is retried here on the breaker's
        # minutes-long schedule rather than returned as None: the caller's `None`
        # handling *drops the record and its source*, so a one-minute network
        # blip during preprocessing would silently truncate the retraining set
        # (far worse than aborting, because the resulting probe still looks fine).
        attempt = 0
        while True:
            breaker.raise_if_tripped()
            failure: object
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except OpenRouterOutageError:
                # OpenRouter is durably down (record_failure or the interruptible
                # sleep tripped the breaker). Every remaining generation would fail
                # too, and silently dropping them all would retrain the probe on a
                # quietly truncated dataset — which is exactly what happened when a
                # run outlived its credits. Propagate out of the worker so the
                # pool.map in the caller stops.
                raise
            except Exception as e:  # noqa: BLE001 — one bad generation shouldn't kill the run
                print(f"  [contrastive] generation failed: {e}")
                failure = e
            else:
                if resp.choices:
                    breaker.record_success()
                    return _parse_json_object(resp.choices[0].message.content or "")
                # OpenRouter surfaces some upstream errors (rate limits, policy
                # blocks) as a 200 with an error body and no choices, rather than
                # raising — report it instead of an opaque subscript TypeError.
                failure = (resp.model_dump() or {}).get("error")
                print(f"  [contrastive] no choices returned; error={failure}")

            kind = breaker.record_failure(failure, where="contrastive generation")
            if kind != "connection":
                # Server-side problem: hand back to the caller's own retry loop.
                return None
            delay = breaker.backoff_delay(kind, attempt)
            print(
                f"  [contrastive] no connection; retrying in {delay / 60:.1f} min "
                f"(unreachable for {breaker.streak_seconds() / 60:.1f} of "
                f"{breaker.max_connection_outage_s() / 60:.0f} min)"
            )
            breaker.sleep_sync(delay)
            attempt += 1


def _guidance_fingerprint(
    concept_description: str,
    target_label: str,
    label_guidance: dict[str, str] | None,
    eval_data_description: str = "",
) -> str:
    """Short hash of the prompt detail that shaped this generation.

    Empty for the no-guidance case, so configs that don't use it keep the exact
    cache keys they had before this knob existed. When guidance *is* set, it goes
    into the key: the cache is loaded by key without re-checking the prompt, so
    otherwise an edited description would silently reuse pairs written under the
    old one. ``eval_data_description`` is folded in for exactly the same reason — it
    is now part of the generation prompt, so a run that adds or edits it must mint
    fresh pairs rather than reuse ones written to the older constraint. It is keyed
    alongside, not instead: a cache written before this argument existed still hits
    for a run that sets no eval-data description.
    """
    detail = _label_guidance_for(target_label, label_guidance)
    description = (concept_description or "").strip()
    eval_data = (eval_data_description or "").strip()
    if not detail and not description and not eval_data:
        return ""
    payload_obj = {"description": description, "target": detail}
    if eval_data:
        payload_obj["eval_data"] = eval_data
    payload = json.dumps(payload_obj, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_key(
    messages: Sequence[dict[str, str]],
    target_label: str,
    guidance_fingerprint: str = "",
) -> str:
    payload: dict[str, Any] = {"messages": list(messages), "target": target_label}
    if guidance_fingerprint:
        payload["guidance"] = guidance_fingerprint
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_cache(cache_path: Path | None) -> dict[str, dict]:
    if cache_path is None or not cache_path.exists():
        return {}
    cache: dict[str, dict] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" in row and "record" in row:
                cache[row["key"]] = row["record"]
    return cache


def _append_cache(cache_path: Path | None, key: str, record: dict) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "record": record}, ensure_ascii=False) + "\n")


def generate_contrastive_dataset(
    records: Sequence[dict],
    *,
    pos_class_label: str,
    neg_class_label: str,
    provider: str,
    model: str,
    max_concurrent: int = 50,
    max_tokens: int = 2048,
    max_retries: int = 2,
    text_key: str = "inputs",
    label_key: str = "labels",
    cache_path: str | Path | None = None,
    assistant_centric: bool = False,
    concept_description: str = "",
    label_guidance: dict[str, str] | None = None,
    eval_data_description: str = "",
    token_budget: TokenBudget | None = None,
) -> list[dict]:
    """Return ``records`` plus one opposite-class contrastive example per record.

    Generalized from tuberlens to arbitrary ``pos_class_label`` /
    ``neg_class_label``. Only records whose label is one of the two are eligible.
    Generated pairs are cached to ``cache_path`` keyed by the source conversation,
    so successes that were already processed in an earlier iteration are reused
    instead of re-queried.

    ``eval_data_description`` (what the evaluation data looks like) constrains the
    generated pair's FORM to that data — see :func:`_eval_data_instruction`. Like the
    two knobs below it is injected verbatim and folded into the cache key, so editing
    it mints fresh pairs instead of reusing ones written to the older constraint.

    ``concept_description`` (what the concept is) and ``label_guidance`` (what a
    given class specifically means, keyed by class label) are injected verbatim
    into the generation prompt, letting a config describe the concept as
    concretely as a hand-written prompt would. Non-empty guidance is folded into
    the cache key, so editing it regenerates the pairs rather than reusing ones
    written under the old prompt; with neither set the keys are unchanged from
    before the knobs existed.

    Each eligible record is retried up to ``max_retries`` times (i.e. up to
    ``max_retries + 1`` LLM calls) when a generation fails — an LLM exception,
    an unparseable/missing response, a malformed (non-well-formed) conversation,
    or a conversation **longer than ``token_budget``**. If it still fails after
    the retries, **both the source record and its (missing) contrastive pair are
    dropped** from the returned data, rather than keeping an unpaired source.
    Failed generations are never cached, so a dropped source is re-attempted on
    the next iteration's retrain.

    ``token_budget`` (a :class:`TokenBudget` over the *probe's* tokenizer) is the
    length safeguard: activation extraction truncates at 1024 tokens, so a longer
    pair would be trained on with its tail cut off. An over-long generation is
    rejected and regenerated with the measured length fed back, rather than
    dropped outright — unlike the attacker path, where a rejected submission
    costs one turn; here the source record has no pair without it, so both would
    be lost. ``None`` (or a disabled budget) skips the check.
    """
    records = list(records)
    valid = [r for r in records if r.get(label_key) in (pos_class_label, neg_class_label)]
    if not valid:
        print("No records with valid labels found; skipping contrastive generation.")
        return records

    # A typo'd label key would silently contribute no guidance (and leave the
    # prompt at its generic form), so say so rather than fail quietly.
    if label_guidance:
        known = {pos_class_label, neg_class_label}
        known |= {_short_label(lbl) for lbl in known}
        unknown = [lbl for lbl in label_guidance if lbl not in known]
        if unknown:
            print(
                f"  [contrastive] warning: label_guidance keys {unknown} match neither "
                f"class label ({pos_class_label!r} / {neg_class_label!r}); ignored"
            )

    cache_path = Path(cache_path) if cache_path is not None else None
    cache = _load_cache(cache_path)

    def _target(label: str) -> str:
        return neg_class_label if label == pos_class_label else pos_class_label

    def _key(messages: Sequence[dict[str, str]], target_label: str) -> str:
        return _cache_key(
            messages,
            target_label,
            _guidance_fingerprint(
                concept_description,
                target_label,
                label_guidance,
                eval_data_description,
            ),
        )

    length_capped = token_budget is not None and token_budget.enabled

    # Partition into cache hits and the records that still need an LLM call. A cached
    # pair is re-checked against the budget rather than trusted: caches written before
    # this cap existed (or under a larger one) hold over-long pairs, and the cache is
    # keyed by the *source* conversation, so nothing else would ever revisit them.
    # Regenerating appends a fresh row under the same key, and _load_cache is
    # last-row-wins, so the shorter pair supersedes it from the next run on.
    generated: list[dict] = []
    to_generate: list[tuple[int, dict, list[dict[str, str]], str, str]] = []
    n_cached_too_long = 0
    for idx, record in enumerate(valid):
        messages = _extract_messages(record, text_key)
        current_label = str(record.get(label_key, ""))
        target_label = _target(current_label)
        key = _key(messages, target_label)
        cached = cache.get(key)
        if cached is not None and length_capped:
            if token_budget.overage(_extract_messages(cached, text_key)) is not None:
                cached = None
                n_cached_too_long += 1
        if cached is not None:
            generated.append(cached)
        else:
            to_generate.append((idx, record, messages, current_label, target_label))
    if n_cached_too_long:
        print(
            f"  [contrastive] regenerating {n_cached_too_long} cached pairs that exceed "
            f"the {token_budget.max_tokens}-token limit"
        )
    if to_generate:
        llm = _ContrastiveLLM(
            provider,
            model,
            max_tokens,
            assistant_centric=assistant_centric,
            concept_description=concept_description,
            label_guidance=label_guidance,
            eval_data_description=eval_data_description,
            max_sample_tokens=token_budget.max_tokens if length_capped else 0,
        )
        llm._ensure_client()  # initialize once before fan-out

        def _attempt(messages, current_label, target_label, feedback=""):
            """One generation attempt.

            Returns ``(contrastive, feedback_for_next_attempt)``. ``contrastive`` is
            None on any failure; the second element is non-empty only when the
            failure is one the model can act on — currently an over-long generation,
            where the retry is told the measured length. An unparseable or malformed
            response carries no such signal, so it retries unchanged.
            """
            response = llm.generate(messages, current_label, target_label, feedback)
            if not response or "generated_messages" not in response:
                return None, ""
            new_messages = _extract_messages(
                {text_key: response["generated_messages"]}, text_key
            )
            # Reject malformed generations (e.g. a provider refusal returned as a
            # lone assistant message) — they have no valid chat-template rendering
            # and would crash activation extraction.
            if not _is_well_formed_conversation(new_messages):
                return None, ""
            # Length safeguard: a pair over the cap would be *trained on* truncated
            # at 1024 tokens, so regenerate it shorter rather than accept it.
            if length_capped:
                n_tokens = token_budget.overage(new_messages)
                if n_tokens is not None:
                    return None, _length_retry_feedback(n_tokens, token_budget.max_tokens)
            return {
                text_key: new_messages,
                label_key: target_label,
                "original_messages": messages,
                "original_label": current_label,
                "generation_explanation": str(response.get("explanation", "")),
                "generation_model": model,
                "is_generated": True,
            }, ""

        def _work(item):
            """Retry a record up to ``max_retries`` times.

            Returns ``("ok", record, key, contrastive)`` on success or
            ``("fail", record, None, None)`` once every attempt has failed — the
            caller drops the source of a failed record (and never caches junk).
            Any feedback an attempt produced (e.g. "that was N tokens, too long")
            is carried into the next attempt's prompt.
            """
            _idx, record, messages, current_label, target_label = item
            feedback = ""
            for attempt in range(max_retries + 1):
                contrastive, feedback = _attempt(
                    messages, current_label, target_label, feedback
                )
                if contrastive is not None:
                    return "ok", record, _key(messages, target_label), contrastive
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))  # brief backoff before retrying
            return "fail", record, None, None

        workers = max(1, min(max_concurrent, len(to_generate)))
        failed_source_ids: set[int] = set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for status, record, key, contrastive in pool.map(_work, to_generate):
                if status == "ok":
                    generated.append(contrastive)
                    _append_cache(cache_path, key, contrastive)
                else:
                    # Give up on this record after retries: drop its source too.
                    failed_source_ids.add(id(record))

        if failed_source_ids:
            records = [r for r in records if id(r) not in failed_source_ids]
            print(
                f"  [contrastive] dropped {len(failed_source_ids)} source records "
                f"(and their pairs) after {max_retries + 1} failed generation attempts"
            )

    label_counts: dict[str, int] = {}
    for record in generated:
        lbl = str(record.get(label_key, "unknown"))
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(
        f"Contrastive generation: {len(generated)} contrastive examples "
        f"from {len(valid)} source records; label distribution {label_counts}"
    )

    return records + generated
