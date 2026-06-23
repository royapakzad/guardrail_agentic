"""
llm_gateway.py
--------------
Optional Otari gateway routing for all LLM completions (PR #14).

How Otari routing works:
    The `otari` Python SDK routes calls through the Otari hosted gateway at
    https://api.otari.ai (default) or a self-hosted instance.

    To enable:
      1. Go to your Otari dashboard → Providers → add your provider API keys
         (Anthropic, OpenAI, etc.) so Otari can forward calls.
      2. Go to Otari dashboard → API Keys → generate a platform token.
      3. Set in your .env:
             LLM_GATEWAY=otari
             OTARI_AI_TOKEN=<token from dashboard>
      4. Optionally set OTARI_API_BASE to override the default https://api.otari.ai.

    When active, every completion call is rewritten as:
        provider="otari", model="<orig_provider>:<orig_model>"
    and the `otari` SDK forwards it through the gateway using your OTARI_AI_TOKEN.

    Leave LLM_GATEWAY unset to call providers directly (default).

Exports:
    resolve_completion_kwargs(provider, model) -> dict
        Returns {} in direct mode (no-op).
        Returns {"provider": "otari", "model": "<provider>:<model>"} in Otari mode.
        Emits RuntimeWarning if OTARI_AI_TOKEN is missing so misconfiguration
        is caught early without breaking a run.
"""

from __future__ import annotations

import os
import warnings
from typing import Any


def _patch_otari_provider() -> None:
    """
    Monkey-patch any-llm-sdk's OtariProvider to use AsyncOtariClient instead of
    OtariClient. Bug in any-llm-sdk <=1.17.0: _init_client creates the sync
    OtariClient but all _acompletion / _aembedding / etc. methods await it,
    raising "object ChatCompletion can't be used in 'await' expression".
    """
    try:
        from any_llm.providers.otari.otari import OtariProvider
        from otari import AsyncOtariClient

        _original_init_client = OtariProvider._init_client

        def _patched_init_client(
            self: Any, api_key: Any = None, api_base: Any = None, **kwargs: Any
        ) -> None:
            _original_init_client(self, api_key=api_key, api_base=api_base, **kwargs)
            # Replace the sync OtariClient with the async one so that
            # `await self.otari_client.completion(...)` works correctly.
            orig = self.otari_client
            self.otari_client = AsyncOtariClient(
                api_base=orig._base_url.removesuffix("/v1")
                if orig._base_url.endswith("/v1")
                else orig._base_url,
                api_key=getattr(orig, "_api_key", None),
                platform_token=getattr(orig, "_platform_token", None),
                default_headers=getattr(orig, "_default_headers", None),
            )
            # Re-bind the openai async client attribute used by base class helpers
            self.client = self.otari_client.openai

        OtariProvider._init_client = _patched_init_client  # type: ignore[method-assign]
    except Exception:
        pass  # If patch fails, the error surfaces at call time with the original message


def resolve_completion_kwargs(provider: str, model: str) -> dict[str, Any]:
    """
    Return extra kwargs to merge into any_llm.completion() calls.

    Direct mode (LLM_GATEWAY not set):
        Returns {} — provider and model are passed through unchanged.

    Otari mode (LLM_GATEWAY=otari):
        Returns {"provider": "otari", "model": "<provider>:<model>"} so the
        any-llm-sdk OtariProvider routes the call through the Otari gateway.
        The gateway uses OTARI_AI_TOKEN for auth (auto-read by the otari SDK).

    Fallback: if OTARI_AI_TOKEN is unset, emits a RuntimeWarning and returns {}
    so a misconfigured gateway never silently breaks a run.
    """
    gateway = os.getenv("LLM_GATEWAY", "").strip().lower()
    if gateway != "otari":
        return {}

    token = (
        os.getenv("OTARI_AI_TOKEN")
        or os.getenv("OTARI_PLATFORM_TOKEN")
        or os.getenv("GATEWAY_PLATFORM_TOKEN")
    )

    if not token:
        warnings.warn(
            "[llm_gateway] LLM_GATEWAY=otari but OTARI_AI_TOKEN is not set. "
            "Go to your Otari dashboard → API Keys to generate one. "
            "Falling back to direct provider calls.",
            RuntimeWarning,
            stacklevel=3,
        )
        return {}

    # Apply the async-client patch the first time Otari mode is activated.
    _patch_otari_provider()

    # any-llm-sdk's OtariProvider requires api_base explicitly — it does not inherit
    # the otari SDK's built-in default. Fall back to the hosted gateway URL.
    api_base = (
        os.getenv("OTARI_API_BASE") or os.getenv("GATEWAY_API_BASE") or "https://api.otari.ai"
    )

    return {
        "provider": "otari",
        "model": f"{provider}:{model}",
        "api_base": api_base,
    }
