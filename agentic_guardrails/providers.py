"""
providers.py
------------
LLM call helper built on mozilla-ai/any-llm-sdk.

any-llm-sdk provides a single unified interface for 40+ providers.
API keys are read automatically from environment variables:
  - OPENAI_API_KEY       (openai)
  - ANTHROPIC_API_KEY    (anthropic)
  - GEMINI_API_KEY       (gemini)
  - MISTRAL_API_KEY      (mistral)
  - COHERE_API_KEY       (cohere)
  - DEEPSEEK_API_KEY     (deepseek)
  - CEREBRAS_API_KEY     (cerebras)
  Ollama runs locally and needs no key.

PR #14: calls are optionally routed through the Otari gateway when
LLM_GATEWAY=otari is set. See llm_gateway.py.

Install:
  pip install 'any-llm-sdk[openai,anthropic,gemini,mistral]'
  pip install 'any-llm-sdk[all]'   # all providers
"""

from __future__ import annotations

from any_llm import completion as _completion
from llm_gateway import resolve_completion_kwargs  # PR #14

SUPPORTED_PROVIDERS = (
    "openai",
    "anthropic",
    "gemini",
    "mistral",
    "cohere",
    "deepseek",
    "cerebras",
    "ollama",
)


def call_llm(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float | None = None,
) -> str:
    """
    Plain chat completion (no tools). Returns the assistant message content.

    Args:
        provider:      any-llm provider name, e.g. "openai", "anthropic", "gemini".
        model:         Model identifier understood by that provider.
        system_prompt: System message text.
        user_message:  User turn text.
        temperature:   Sampling temperature. Defaults to None (use the model's
                       default). Some models reject any explicit temperature value.
    """
    # PR #14: gateway overrides (empty dict in direct mode — no-op)
    gateway_overrides = resolve_completion_kwargs(provider, model)

    kwargs: dict = {
        "provider": provider.lower(),
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    kwargs.update(gateway_overrides)

    if temperature is not None:
        kwargs["temperature"] = temperature

    resp = _completion(**kwargs)
    # Non-streaming call always returns a ChatCompletion (not a chunk iterator).
    return resp.choices[0].message.content or ""  # type: ignore[union-attr]
