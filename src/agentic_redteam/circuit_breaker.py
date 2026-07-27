"""Process-global circuit breaker for OpenRouter calls.

Motivation: every OpenRouter call site in this repo is individually
fault-tolerant by design — a failed attacker round is logged and the rotation
carries on (``attacker.run_one_model``), a failed contrastive generation drops
one record (``preprocessing``), a failed summarization is swallowed
(``attacker._summarize_round``). That is the right behaviour for a transient
blip, but it means a *durable* outage — an exhausted OpenRouter balance, a
revoked key — is invisible: the run keeps grinding through every remaining
round, retrains on nothing, evaluates, and exits 0 with a plausible-looking
comparison CSV.

This module adds the missing global signal. Every OpenRouter call reports its
outcome here; N consecutive failures across *all* call sites trip the breaker,
which raises :class:`OpenRouterOutageError` and keeps raising it, so the run
stops instead of burning the rest of its schedule against a dead endpoint.

Two thresholds, because the two failure classes deserve different patience:

* **transient** — 429/5xx/timeouts/empty-choices envelopes. May well recover,
  so allow ``OPENROUTER_MAX_CONSECUTIVE_ERRORS`` (default 10) in a row.
* **fatal** — 401/402/403, "Insufficient credits", "Invalid API key". These do
  not recover without human action, so allow only
  ``OPENROUTER_MAX_CONSECUTIVE_FATAL_ERRORS`` (default 3), and callers skip
  their backoff sleeps for them.

"Consecutive" is counted globally and reset by *any* successful OpenRouter
call, so one durably-broken model in a rotation whose siblings still work will
not trip the breaker — which is what we want, since the rotation genuinely can
continue in that case.

The counters are guarded by a :class:`threading.Lock` rather than an asyncio
primitive: call sites live in the asyncio attacker loop *and* in
``preprocessing``'s ``ThreadPoolExecutor`` workers, so the breaker has to be
safe from both.
"""

from __future__ import annotations

import os
import threading

__all__ = [
    "OpenRouterOutageError",
    "classify_error",
    "is_fatal_error",
    "record_success",
    "record_failure",
    "raise_if_tripped",
    "reset",
    "snapshot",
]


class OpenRouterOutageError(RuntimeError):
    """OpenRouter has failed too many times in a row; the run must stop.

    Deliberately **not** swallowed by the per-call-site ``except Exception``
    handlers in :mod:`agentic_redteam.attacker` and
    :mod:`agentic_redteam.preprocessing` — those re-raise it, mirroring how
    ``JudgeRefusalError`` is treated. A run that keeps going after this has
    nothing left to produce.
    """


DEFAULT_MAX_CONSECUTIVE_ERRORS = 10
DEFAULT_MAX_CONSECUTIVE_FATAL_ERRORS = 3

# HTTP statuses that never recover on their own: no credits, bad key, forbidden.
_FATAL_STATUSES = frozenset({401, 402, 403})

# Substrings that identify a non-recoverable failure when no status code is
# available (e.g. the message was pulled out of a 200-with-error-envelope body
# by ``openrouter_client.extract_openrouter_error``).
_FATAL_MARKERS = (
    "insufficient credits",
    "insufficient_quota",
    "invalid api key",
    "no auth credentials",
    "user not found",
    "quota exceeded",
    "payment required",
    "account has been disabled",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def classify_error(detail: object) -> str:
    """Return ``"fatal"`` or ``"transient"`` for an exception or message string.

    ``detail`` may be an exception (its ``status_code`` is preferred when
    present, as on ``openai.APIStatusError``) or any object whose ``str()`` is
    the error text.
    """
    status = getattr(detail, "status_code", None)
    if isinstance(status, int) and status in _FATAL_STATUSES:
        return "fatal"
    text = str(detail).lower()
    if any(marker in text for marker in _FATAL_MARKERS):
        return "fatal"
    # The status code often survives only inside the stringified message, e.g.
    # "APIStatusError: Error code: 402 - {...}".
    if any(f"error code: {code}" in text for code in _FATAL_STATUSES):
        return "fatal"
    return "transient"


def is_fatal_error(detail: object) -> bool:
    """True when retrying ``detail`` in-process cannot possibly help."""
    return classify_error(detail) == "fatal"


class _Breaker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consecutive = 0
        self._consecutive_fatal = 0
        self._tripped: str | None = None
        self._last_detail = ""

    def record_success(self) -> None:
        with self._lock:
            if self._tripped is None:
                self._consecutive = 0
                self._consecutive_fatal = 0

    def record_failure(self, detail: object, *, where: str) -> str:
        """Count one failed OpenRouter call; raise if that trips the breaker.

        Returns the classification (``"fatal"`` / ``"transient"``) so the caller
        can skip its backoff sleep on a fatal error.
        """
        kind = classify_error(detail)
        max_any = _env_int("OPENROUTER_MAX_CONSECUTIVE_ERRORS", DEFAULT_MAX_CONSECUTIVE_ERRORS)
        max_fatal = _env_int(
            "OPENROUTER_MAX_CONSECUTIVE_FATAL_ERRORS", DEFAULT_MAX_CONSECUTIVE_FATAL_ERRORS
        )
        with self._lock:
            self._consecutive += 1
            if kind == "fatal":
                self._consecutive_fatal += 1
            self._last_detail = f"{where}: {detail}"
            if self._tripped is None:
                if self._consecutive_fatal >= max_fatal:
                    self._tripped = (
                        f"{self._consecutive_fatal} consecutive non-recoverable "
                        f"OpenRouter errors (limit {max_fatal}). These do not resolve "
                        f"on their own — check your credit balance and API key. "
                        f"Last error — {self._last_detail}"
                    )
                elif self._consecutive >= max_any:
                    self._tripped = (
                        f"{self._consecutive} consecutive OpenRouter errors "
                        f"(limit {max_any}). Last error — {self._last_detail}"
                    )
            tripped = self._tripped
        if tripped is not None:
            raise OpenRouterOutageError(tripped)
        return kind

    def raise_if_tripped(self) -> None:
        with self._lock:
            tripped = self._tripped
        if tripped is not None:
            raise OpenRouterOutageError(tripped)

    def reset(self) -> None:
        with self._lock:
            self._consecutive = 0
            self._consecutive_fatal = 0
            self._tripped = None
            self._last_detail = ""

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "consecutive": self._consecutive,
                "consecutive_fatal": self._consecutive_fatal,
                "tripped": self._tripped,
            }


_BREAKER = _Breaker()

record_success = _BREAKER.record_success
record_failure = _BREAKER.record_failure
raise_if_tripped = _BREAKER.raise_if_tripped
reset = _BREAKER.reset
snapshot = _BREAKER.snapshot
