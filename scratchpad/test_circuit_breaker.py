#!/usr/bin/env python
"""Exercise the OpenRouter circuit breaker against the failure the ablation hit.

Reproduces the shape of logs/run_hh_llama70b50_gpt51_memo.log: every OpenRouter
call returns 402 Insufficient credits. Before the breaker, each of the 300
model-rounds burned 4 attempts + ~14s of backoff and was absorbed into a
"FAILED" round summary, so the run marched through every iteration and produced
a comparison CSV from probes trained on nothing.

Case 6 covers the *second* failure this file has now seen in production
(logs/run_hs_gemma27b_gptoss120b_noguidance.log): a ~2.5 minute network drop,
observed simultaneously by 10 concurrent sessions, tripped the count-based
threshold on its first wave and aborted a 10-hour run — twice in one night.
Connection errors are therefore their own class now, governed by elapsed outage
time rather than a failure count. Note cases 3 and 4 used to raise
APIConnectionError to stand in for a "transient blip"; that conflation is
precisely what changed, so they now use a genuine server-side transient (a 200
with an error envelope and no choices).

Run: .venv_claude/bin/python scratchpad/test_circuit_breaker.py
"""

import asyncio
import os
import sys
import time

import openai

# Keep the connection schedule test-sized: 1/2/8 SECONDS standing in for the
# production 1/2/8 minutes, and a 10s outage window for the 30-minute default.
os.environ["OPENROUTER_CONNECTION_BACKOFF_S"] = "1,2,8"
os.environ["OPENROUTER_MAX_CONNECTION_OUTAGE_S"] = "10"

from agentic_redteam import circuit_breaker as breaker  # noqa: E402
from agentic_redteam.attacker import _openrouter_create_with_retry  # noqa: E402
from agentic_redteam.circuit_breaker import OpenRouterOutageError  # noqa: E402

FAILURES = 0


def _make_402() -> openai.APIStatusError:
    """An APIStatusError shaped like the one OpenRouter returns at zero balance."""
    import httpx

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        402,
        request=request,
        json={
            "error": {
                "message": (
                    "Insufficient credits. Add more using "
                    "https://openrouter.ai/settings/credits"
                ),
                "code": 402,
            }
        },
    )
    # Message verbatim from logs/run_hh_llama70b50_gpt51_memo.log.
    return openai.APIStatusError(
        "Error code: 402 - {'error': {'message': 'Insufficient credits. Add more "
        "using https://openrouter.ai/settings/credits', 'code': 402}}",
        response=response,
        body=response.json(),
    )


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _AlwaysFailsCompletions:
    async def create(self, **kwargs):
        global FAILURES
        FAILURES += 1
        raise _make_402()


class _AlwaysFailsClient:
    """Stands in for openai.AsyncOpenAI; every call raises the 402."""

    def __init__(self):
        self.chat = _Chat(_AlwaysFailsCompletions())


class _Msg:
    content = "ok"
    tool_calls = None


class _Choice:
    message = _Msg()


class _Resp:
    choices = [_Choice()]


class _FlakyCompletions:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        global FAILURES
        if self.owner.remaining > 0:
            self.owner.remaining -= 1
            FAILURES += 1
            import httpx

            raise openai.APIConnectionError(
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/x")
            )
        return _Resp()


class _FlakyThenOkClient:
    """Fails `n_fail` times with a CONNECTION error, then succeeds."""

    def __init__(self, n_fail: int):
        self.remaining = n_fail
        self.chat = _Chat(_FlakyCompletions(self))


class _EnvelopeResp:
    """A 200 with an OpenRouter error envelope and no choices — server-side."""

    choices: list = []
    error = {"message": "upstream provider is rate-limiting, please retry"}


class _TransientCompletions:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        global FAILURES
        if self.owner.remaining > 0:
            self.owner.remaining -= 1
            FAILURES += 1
            return _EnvelopeResp()
        return _Resp()


class _TransientThenOkClient:
    """Fails `n_fail` times with a genuine transient (429-ish) error, then succeeds."""

    def __init__(self, n_fail: int):
        self.remaining = n_fail
        self.chat = _Chat(_TransientCompletions(self))


def check(name: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}{(' — ' + detail) if detail else ''}")
    return condition


async def main() -> int:
    global FAILURES
    ok = True

    # ---- 1. Fatal (402) errors trip the breaker after 3 calls, not 300. ------
    print("\n1. 402 Insufficient credits trips the breaker quickly")
    breaker.reset()
    FAILURES = 0
    client = _AlwaysFailsClient()
    rounds_attempted = 0
    outage = None
    for _ in range(300):  # the ablation's 300 model-rounds
        rounds_attempted += 1
        try:
            await _openrouter_create_with_retry(
                client, model="openai/gpt-5.1", messages=[], tools=None
            )
        except OpenRouterOutageError as e:
            outage = e
            break
        except RuntimeError:
            continue  # one round absorbed, as before
    ok &= check("aborted instead of running all 300 rounds", outage is not None)
    ok &= check(
        "stopped within 5 rounds", rounds_attempted <= 5, f"took {rounds_attempted}"
    )
    ok &= check(
        "no wasted retry storm",
        FAILURES <= 5,
        f"{FAILURES} API calls (was 4 per round x 300 = 1200)",
    )
    ok &= check(
        "message names the cause",
        "credits" in str(outage).lower(),
        str(outage)[:110],
    )

    # ---- 2. Once tripped, queued work bails without touching the network. ----
    print("\n2. Tripped breaker short-circuits later calls")
    before = FAILURES
    raised = False
    try:
        await _openrouter_create_with_retry(
            client, model="openai/gpt-5.1", messages=[], tools=None
        )
    except OpenRouterOutageError:
        raised = True
    ok &= check("still raises", raised)
    ok &= check("made no further API calls", FAILURES == before)

    # ---- 3. Transient errors are tolerated up to the higher limit. -----------
    print("\n3. A transient blip is retried, not fatal")
    breaker.reset()
    FAILURES = 0
    flaky = _TransientThenOkClient(n_fail=2)  # recovers inside one retry loop
    got = None
    try:
        got = await _openrouter_create_with_retry(
            flaky, model="openai/gpt-5.1", messages=[], tools=None
        )
    except Exception as e:  # noqa: BLE001
        got = e
    ok &= check("call succeeded after retries", getattr(got, "choices", None) is not None)
    ok &= check(
        "success reset the counter", breaker.snapshot()["consecutive"] == 0,
        str(breaker.snapshot()),
    )

    # ---- 4. Transient errors still trip eventually (10 in a row). -----------
    print("\n4. Sustained transient failures trip at the higher threshold")
    breaker.reset()
    FAILURES = 0
    never = _TransientThenOkClient(n_fail=10**6)
    calls = 0
    outage = None
    for _ in range(50):
        calls += 1
        try:
            await _openrouter_create_with_retry(
                never, model="openai/gpt-5.1", messages=[], tools=None
            )
        except OpenRouterOutageError as e:
            outage = e
            break
        except RuntimeError:
            continue
    ok &= check("tripped", outage is not None)
    ok &= check(
        "tolerated more than the fatal budget first",
        FAILURES >= 10,
        f"{FAILURES} failed calls before tripping",
    )

    # ---- 5. Interleaved success prevents a trip (one bad model in a rotation).
    print("\n5. A healthy sibling model keeps the run alive")
    breaker.reset()
    bad, good = _AlwaysFailsClient(), _FlakyThenOkClient(n_fail=0)
    survived = True
    for _ in range(20):
        try:
            await _openrouter_create_with_retry(bad, model="bad/model", messages=[])
        except OpenRouterOutageError:
            survived = False
            break
        except RuntimeError:
            pass
        await _openrouter_create_with_retry(good, model="good/model", messages=[])
    ok &= check("did not trip on one durably-broken model", survived)

    # ---- 6. THE regression: a recoverable network drop must not kill the run.
    # logs/run_hs_gemma27b_gptoss120b_noguidance.log, rounds 10 and 13.
    print("\n6. A connection drop is survived, not counted")
    breaker.reset()
    FAILURES = 0
    # 6a. The exact shape that aborted the run: sessions_per_model=10 concurrent
    # sessions all observing one network event. Under the old count-based rule
    # this hit the limit of 10 on the first wave, before any backoff even began.
    clients = [_FlakyThenOkClient(n_fail=2) for _ in range(10)]
    t0 = time.monotonic()
    try:
        results = await asyncio.gather(
            *[
                _openrouter_create_with_retry(c, model="openai/gpt-oss-120b", messages=[])
                for c in clients
            ]
        )
        elapsed = time.monotonic() - t0
        ok &= check(
            "10 concurrent sessions all rode out the drop",
            len(results) == 10 and all(r.choices for r in results),
            f"{FAILURES} failed calls over {elapsed:.1f}s, breaker held",
        )
    except OpenRouterOutageError as e:
        ok &= check("10 concurrent sessions all rode out the drop", False, str(e)[:90])
    ok &= check(
        "recovery reset the streak", breaker.snapshot()["streak_seconds"] == 0.0
    )

    # 6b. ...but a network that never returns still aborts, on elapsed time.
    breaker.reset()
    FAILURES = 0
    dead = _FlakyThenOkClient(n_fail=10**6)
    t0 = time.monotonic()
    outage = None
    try:
        await _openrouter_create_with_retry(dead, model="openai/gpt-oss-120b", messages=[])
    except OpenRouterOutageError as e:
        outage = e
    waited = time.monotonic() - t0
    ok &= check("a dead network still trips", outage is not None)
    ok &= check(
        "tripped on the outage clock, not the count",
        outage is not None and "unreachable for" in str(outage),
        str(outage)[:100],
    )
    ok &= check(
        "waited out the full window first",
        waited >= 10.0,
        f"{waited:.1f}s (window 10s; production default 30 min)",
    )

    breaker.reset()
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
