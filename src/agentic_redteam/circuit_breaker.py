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

Three classes, because they deserve very different patience:

* **transient** — 429/5xx/empty-choices envelopes. We *reached* OpenRouter and
  it answered badly. May well recover in seconds, so allow
  ``OPENROUTER_MAX_CONSECUTIVE_ERRORS`` (default 10) in a row, retried on the
  short 2/4/8s backoff.
* **connection** — the request never left the box (or never got an answer):
  ``APIConnectionError`` / ``APITimeoutError``, DNS failure, connection reset,
  network unreachable. A wifi drop, a DHCP renew or a container losing its NIC
  routinely lasts minutes, so this class is retried on a *minutes*-long
  schedule (``OPENROUTER_CONNECTION_BACKOFF_S``, default 60/120/480s) and is
  governed by **elapsed time, not a failure count** — see below.
* **fatal** — 401/402/403, "Insufficient credits", "Invalid API key". These do
  not recover without human action, so allow only
  ``OPENROUTER_MAX_CONSECUTIVE_FATAL_ERRORS`` (default 3), and callers skip
  their backoff sleeps for them.

"Consecutive" is counted globally and reset by *any* successful OpenRouter
call, so one durably-broken model in a rotation whose siblings still work will
not trip the breaker — which is what we want, since the rotation genuinely can
continue in that case.

**Why connection errors are timed, not counted.** A count-based threshold
counts *observations*, and with ``attacker.sessions_per_model: 10`` a single
network event is observed by ten concurrent sessions at once — so the counter
reached its limit of 10 on the first wave, before any backoff sleep had even
begun, and no amount of lengthening the backoff would have helped. That is
exactly how a ~2.5 minute connection blip killed a 10-hour run twice in one
night (``logs/run_hs_gemma27b_gptoss120b_noguidance.log``: two
``openrouter_outage`` aborts, followed by 6h and 30min of idle GPU waiting for
a human to restart). A wall-clock streak duration is immune to that
multiplication: ten simultaneous observers do not advance a clock. So once a
failure streak contains **any** connection error, the streak is governed by
``OPENROUTER_MAX_CONNECTION_OUTAGE_S`` (default 1800s = 30 min) instead of by
the consecutive count.

The counters are guarded by a :class:`threading.Lock` rather than an asyncio
primitive: call sites live in the asyncio attacker loop *and* in
``preprocessing``'s ``ThreadPoolExecutor`` workers, so the breaker has to be
safe from both.
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
import time

__all__ = [
    "OpenRouterOutageError",
    "classify_error",
    "is_fatal_error",
    "is_connection_error",
    "backoff_delay",
    "connection_backoff_schedule",
    "max_connection_outage_s",
    "sleep_sync",
    "sleep_async",
    "streak_seconds",
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

# How long a connection outage may persist (seconds of *continuous* failure,
# reset by any success) before the breaker gives up. Deliberately far longer
# than the retry schedule below: the schedule decides how often we probe, this
# decides when we stop. 30 min of patience is cheap next to the 6 hours of idle
# GPU that an unnecessary abort cost.
DEFAULT_MAX_CONNECTION_OUTAGE_S = 1800.0

# Backoff between connection retries. Connection revival (ISP, DHCP, NIC) is a
# minutes-scale event, so probing every couple of seconds is pure noise.
DEFAULT_CONNECTION_BACKOFF_S = (60.0, 120.0, 480.0)

# Backoff for ordinary transient (429/5xx) errors: these really are blips, and
# a minutes-long sleep here would just waste rotation time.
DEFAULT_TRANSIENT_BACKOFF_BASE_S = 2.0

# ±20% so ten sessions waking from the same outage don't stampede the endpoint
# in lockstep.
_JITTER_FRACTION = 0.2

# Granularity at which the long sleeps re-check the breaker, so that when one
# call site's streak finally trips, every other call site sleeping out its
# backoff wakes up and aborts instead of idling for another 8 minutes.
_SLEEP_CHUNK_S = 5.0

# HTTP statuses that never recover on their own: no credits, bad key, forbidden.
_FATAL_STATUSES = frozenset({401, 402, 403})

# Exception types (matched by name across the MRO, so we don't import openai or
# httpx here) that mean the request never completed a round trip.
_CONNECTION_EXC_NAMES = frozenset(
    {
        "APIConnectionError",  # openai — base of APITimeoutError too
        "APITimeoutError",
        "ConnectError",  # httpx
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadError",
        "RemoteProtocolError",
        "ConnectionError",  # stdlib / requests
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "TimeoutError",
    }
)

# Substrings identifying a transport failure when all we have is a message
# string. Deliberately narrower than the exception list: a *provider* timeout
# reported inside an OpenRouter 200-error-envelope is not our network going
# away, and must not earn the minutes-long backoff. Bare "timed out" is
# therefore excluded — real client timeouts arrive as APITimeoutError and are
# caught by type above.
_CONNECTION_MARKERS = (
    "connection error",
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection timeout",
    "server disconnected",
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname",
    "network is unreachable",
    "no route to host",
)

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
    """Return ``"fatal"``, ``"connection"`` or ``"transient"``.

    ``detail`` may be an exception (its ``status_code`` is preferred when
    present, as on ``openai.APIStatusError``) or any object whose ``str()`` is
    the error text.

    Fatal is checked first: a 402 whose body happens to mention a reset
    connection is still a drained balance, not a network problem.
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
    # A status code of any kind means we reached OpenRouter, so this is a
    # server-side problem however its message reads — never a transport one.
    if isinstance(detail, BaseException):
        if isinstance(status, int):
            return "transient"
        if any(cls.__name__ in _CONNECTION_EXC_NAMES for cls in type(detail).__mro__):
            return "connection"
    if any(marker in text for marker in _CONNECTION_MARKERS):
        return "connection"
    return "transient"


def is_fatal_error(detail: object) -> bool:
    """True when retrying ``detail`` in-process cannot possibly help."""
    return classify_error(detail) == "fatal"


def is_connection_error(detail: object) -> bool:
    """True when ``detail`` is a transport failure, not an OpenRouter answer."""
    return classify_error(detail) == "connection"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def max_connection_outage_s() -> float:
    """Seconds of continuous connection failure tolerated before tripping."""
    return _env_float("OPENROUTER_MAX_CONNECTION_OUTAGE_S", DEFAULT_MAX_CONNECTION_OUTAGE_S)


def connection_backoff_schedule() -> tuple[float, ...]:
    """Retry intervals for connection errors, from ``OPENROUTER_CONNECTION_BACKOFF_S``.

    A comma-separated list of seconds (e.g. ``"60,120,480"``). The final entry
    repeats for any further attempts, since the retry loop for this class is
    bounded by :func:`max_connection_outage_s`, not by an attempt count.
    """
    raw = os.environ.get("OPENROUTER_CONNECTION_BACKOFF_S")
    if not raw:
        return DEFAULT_CONNECTION_BACKOFF_S
    try:
        values = tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError:
        return DEFAULT_CONNECTION_BACKOFF_S
    positive = tuple(v for v in values if v > 0)
    return positive or DEFAULT_CONNECTION_BACKOFF_S


def backoff_delay(kind: str, attempt: int) -> float:
    """Seconds to wait before the retry following a failed ``attempt`` (0-based)."""
    if kind == "connection":
        schedule = connection_backoff_schedule()
        delay = schedule[min(attempt, len(schedule) - 1)]
        return delay * (1.0 + random.uniform(-_JITTER_FRACTION, _JITTER_FRACTION))
    return DEFAULT_TRANSIENT_BACKOFF_BASE_S * (2**attempt)


def sleep_sync(delay: float) -> None:
    """Sleep ``delay`` seconds, aborting early if the breaker trips meanwhile.

    Chunked rather than one long ``time.sleep`` for two reasons: a call site
    that is 7 minutes into a backoff should not keep waiting once another call
    site has already declared the outage terminal, and an interpreter shutdown
    should not have to join a thread parked for 8 minutes (the old code already
    warned "executor did not finishing joining its threads within 300 seconds").
    """
    deadline = time.monotonic() + delay
    while True:
        raise_if_tripped()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, _SLEEP_CHUNK_S))


async def sleep_async(delay: float) -> None:
    """Async twin of :func:`sleep_sync`."""
    deadline = time.monotonic() + delay
    while True:
        raise_if_tripped()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, _SLEEP_CHUNK_S))


class _Breaker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consecutive = 0
        self._consecutive_fatal = 0
        self._consecutive_connection = 0
        # monotonic timestamp of the first failure of the current streak, or
        # None when the last outcome was a success. This is what makes the
        # connection rule immune to N concurrent sessions observing one event.
        self._streak_started_at: float | None = None
        self._connection_in_streak = False
        self._tripped: str | None = None
        self._last_detail = ""

    def record_success(self) -> None:
        with self._lock:
            if self._tripped is None:
                self._consecutive = 0
                self._consecutive_fatal = 0
                self._consecutive_connection = 0
                self._streak_started_at = None
                self._connection_in_streak = False

    def record_failure(self, detail: object, *, where: str) -> str:
        """Count one failed OpenRouter call; raise if that trips the breaker.

        Returns the classification (``"fatal"`` / ``"connection"`` /
        ``"transient"``) so the caller can pick the right backoff — skipping it
        entirely for fatal, and using the minutes-long schedule for connection.
        """
        kind = classify_error(detail)
        max_any = _env_int("OPENROUTER_MAX_CONSECUTIVE_ERRORS", DEFAULT_MAX_CONSECUTIVE_ERRORS)
        max_fatal = _env_int(
            "OPENROUTER_MAX_CONSECUTIVE_FATAL_ERRORS", DEFAULT_MAX_CONSECUTIVE_FATAL_ERRORS
        )
        max_outage = max_connection_outage_s()
        now = time.monotonic()
        with self._lock:
            if self._streak_started_at is None:
                self._streak_started_at = now
            self._consecutive += 1
            if kind == "fatal":
                self._consecutive_fatal += 1
            elif kind == "connection":
                self._consecutive_connection += 1
                self._connection_in_streak = True
            elapsed = now - self._streak_started_at
            self._last_detail = f"{where}: {detail}"
            if self._tripped is None:
                if self._consecutive_fatal >= max_fatal:
                    self._tripped = (
                        f"{self._consecutive_fatal} consecutive non-recoverable "
                        f"OpenRouter errors (limit {max_fatal}). These do not resolve "
                        f"on their own — check your credit balance and API key. "
                        f"Last error — {self._last_detail}"
                    )
                elif self._connection_in_streak:
                    # Timed, not counted — see the module docstring. Ten
                    # sessions failing at once must not stand in for ten
                    # minutes of a dead network.
                    if elapsed >= max_outage:
                        self._tripped = (
                            f"OpenRouter has been unreachable for "
                            f"{elapsed / 60:.1f} minutes straight "
                            f"(limit {max_outage / 60:.0f} min, "
                            f"{self._consecutive} failed calls, "
                            f"{self._consecutive_connection} of them connection errors). "
                            f"The network did not come back — check connectivity/DNS on "
                            f"this host. Last error — {self._last_detail}"
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

    def streak_seconds(self) -> float:
        """How long the current unbroken failure streak has lasted (0 if none)."""
        with self._lock:
            if self._streak_started_at is None:
                return 0.0
            return time.monotonic() - self._streak_started_at

    def raise_if_tripped(self) -> None:
        with self._lock:
            tripped = self._tripped
        if tripped is not None:
            raise OpenRouterOutageError(tripped)

    def reset(self) -> None:
        with self._lock:
            self._consecutive = 0
            self._consecutive_fatal = 0
            self._consecutive_connection = 0
            self._streak_started_at = None
            self._connection_in_streak = False
            self._tripped = None
            self._last_detail = ""

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = (
                0.0
                if self._streak_started_at is None
                else time.monotonic() - self._streak_started_at
            )
            return {
                "consecutive": self._consecutive,
                "consecutive_fatal": self._consecutive_fatal,
                "consecutive_connection": self._consecutive_connection,
                "connection_in_streak": self._connection_in_streak,
                "streak_seconds": elapsed,
                "tripped": self._tripped,
            }


_BREAKER = _Breaker()

record_success = _BREAKER.record_success
record_failure = _BREAKER.record_failure
raise_if_tripped = _BREAKER.raise_if_tripped
streak_seconds = _BREAKER.streak_seconds
reset = _BREAKER.reset
snapshot = _BREAKER.snapshot
