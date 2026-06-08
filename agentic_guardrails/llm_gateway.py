"""
llm_gateway.py
--------------
Single switch for routing every LLM completion through the Otari gateway
(mozilla-ai/otari) instead of calling each provider directly.

Otari is an OpenAI-compatible LLM gateway built on any-llm — it adds virtual API
keys, budgets, and usage tracking in front of the real providers. any-llm v1 ships
a native ``otari`` provider, so routing through it is just a matter of rewriting the
``provider``/``model`` that we hand to ``any_llm.completion``.

Usage: every completion call site builds its kwargs via ``resolve_completion_kwargs``:

    from llm_gateway import resolve_completion_kwargs
    resp = completion(**resolve_completion_kwargs(provider="openai", model="gpt-5-nano"),
                      messages=[...])

Configuration (environment / .env):
    LLM_GATEWAY=direct        # default — call providers directly (unchanged behavior)
    LLM_GATEWAY=otari         # route everything through the gateway
    OTARI_API_BASE=http://localhost:8000/v1
    OTARI_API_KEY=<virtual key>          # (or OTARI_PLATFORM_TOKEN for hosted otari.ai)

If ``LLM_GATEWAY=otari`` but no ``OTARI_API_BASE`` is configured, calls fall back to
direct mode so a missing gateway never breaks a run. Legacy ``GATEWAY_API_BASE`` /
``GATEWAY_API_KEY`` env names are also accepted (any-llm's pre-Otari naming).
"""

from __future__ import annotations

import os

# any-llm's native gateway provider name.
_OTARI_PROVIDER = "otari"


def gateway_mode() -> str:
    """Return the configured gateway mode: 'otari' or 'direct' (default)."""
    return os.getenv("LLM_GATEWAY", "direct").strip().lower()


def _otari_api_base() -> str | None:
    return os.getenv("OTARI_API_BASE") or os.getenv("GATEWAY_API_BASE")


def _otari_api_key() -> str | None:
    return os.getenv("OTARI_API_KEY") or os.getenv("GATEWAY_API_KEY")


def resolve_completion_kwargs(*, provider: str, model: str) -> dict[str, str]:
    """
    Map a (provider, model) pair to the kwargs ``any_llm.completion`` should receive.

    - direct mode (default): ``{"provider": provider, "model": model}`` — unchanged.
    - otari mode: route through the gateway as
      ``{"provider": "otari", "model": "<provider>:<model>", "api_base": ..., "api_key": ...}``.
      The gateway expects the real upstream as a ``provider:model`` string.

    Falls back to direct mode when otari is requested but ``OTARI_API_BASE`` is unset,
    so a missing/unconfigured gateway never breaks a run.
    """
    if gateway_mode() != _OTARI_PROVIDER:
        return {"provider": provider, "model": model}

    api_base = _otari_api_base()
    if not api_base:
        # Otari requested but not configured — degrade gracefully to direct.
        return {"provider": provider, "model": model}

    kwargs: dict[str, str] = {
        "provider": _OTARI_PROVIDER,
        "model": f"{provider}:{model}",
        "api_base": api_base,
    }
    api_key = _otari_api_key()
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs
