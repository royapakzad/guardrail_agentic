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

Install:
  pip install 'any-llm-sdk[openai,anthropic,google,mistral]'
  pip install 'any-llm-sdk[all]'   # all providers
"""
from __future__ import annotations

from any_llm import completion as _completion

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
        provider:     any-llm provider name, e.g. "openai", "anthropic", "gemini".
        model:        Model identifier understood by that provider,
                      e.g. "gpt-5-mini", "claude-sonnet-4-6", "gemini-2.5-flash".
        system_prompt: System message text.
        user_message:  User turn text.
        temperature:   Sampling temperature. Defaults to None (use the model's
                       default). Some models (e.g. gpt-5-mini, o-series) reject
                       any explicit temperature value and require the default.
    """
    kwargs: dict = dict(
        provider=provider.lower(),
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    # Only send temperature when explicitly set — models like gpt-5-mini and
    # the o-series reject temperature=0.0 and require the API default.
    if temperature is not None:
        kwargs["temperature"] = temperature

    resp = _completion(**kwargs)
    return resp.choices[0].message.content or ""
