"""
providers.py
------------
LLM client factory and plain chat-completion helper.
Supports OpenAI, Gemini (via OpenAI-compatible endpoint), and Mistral.
"""
from __future__ import annotations

import os
from openai import OpenAI


SUPPORTED_PROVIDERS = ("openai", "gemini", "mistral")


def build_client(provider: str) -> OpenAI:
    """
    Return an OpenAI-compatible client for the given provider.

    Reads OPENAI_API_KEY / GEMINI_API_KEY / MISTRAL_API_KEY from the environment.
    Raises RuntimeError if the required key is absent.
    """
    provider = provider.lower()

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        return OpenAI(api_key=api_key)

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set in the environment.")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
        )

    raise ValueError(
        f"Unknown provider: {provider!r}. Must be one of: {SUPPORTED_PROVIDERS}"
    )


def call_llm(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
) -> str:
    """
    Plain chat completion (no tools). Returns the assistant message content.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content or ""
