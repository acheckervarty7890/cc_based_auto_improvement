"""Thin OpenRouter client factory.

OpenRouter exposes an OpenAI-compatible endpoint, so we use the official
``openai`` SDK and just point ``base_url`` at it. The same instance is reused by
both the OpenRouter judge and the OpenRouter attacker driver.

Env vars:
    OPENROUTER_API_KEY        — required.
    OPENROUTER_BASE_URL       — optional override (default https://openrouter.ai/api/v1).
    OPENROUTER_HTTP_REFERER   — optional, sent as HTTP-Referer (OpenRouter dashboard attribution).
    OPENROUTER_APP_TITLE      — optional, sent as X-Title (OpenRouter dashboard attribution).

``openai`` is imported lazily so configurations that never touch OpenRouter
don't need it loadable.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _resolve_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; required for provider='openrouter'."
        )
    return key


def _default_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_APP_TITLE")
    if title:
        headers["X-Title"] = title
    return headers


def make_sync_client() -> Any:
    """Return a configured ``openai.OpenAI`` instance pointed at OpenRouter."""
    from openai import OpenAI

    return OpenAI(
        api_key=_resolve_api_key(),
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        default_headers=_default_headers() or None,
    )


def make_async_client() -> Any:
    """Return a configured ``openai.AsyncOpenAI`` instance pointed at OpenRouter."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=_resolve_api_key(),
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        default_headers=_default_headers() or None,
    )
