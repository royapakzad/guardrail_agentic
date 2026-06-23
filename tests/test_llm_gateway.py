"""Tests for the Otari gateway resolver (PR #14)."""

import os
import warnings
from unittest.mock import patch

from llm_gateway import resolve_completion_kwargs


def test_direct_mode_returns_empty_dict():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LLM_GATEWAY", None)
        result = resolve_completion_kwargs("openai", "gpt-5-mini")
    assert result == {}


def test_otari_mode_overrides_provider_and_model():
    """With OTARI_AI_TOKEN set, calls are rewritten to provider=otari."""
    env = {"LLM_GATEWAY": "otari", "OTARI_AI_TOKEN": "test-token"}
    with patch.dict(os.environ, env, clear=False):
        for var in ("OTARI_API_BASE", "GATEWAY_API_BASE"):
            os.environ.pop(var, None)
        result = resolve_completion_kwargs("openai", "gpt-5-mini")
    assert result["provider"] == "otari"
    assert result["model"] == "openai:gpt-5-mini"
    assert result["api_base"] == "https://api.otari.ai"  # always present; default when not set


def test_otari_mode_includes_api_base_when_set():
    env = {
        "LLM_GATEWAY": "otari",
        "OTARI_AI_TOKEN": "test-token",
        "OTARI_API_BASE": "https://self-hosted.example.com",
    }
    with patch.dict(os.environ, env):
        result = resolve_completion_kwargs("anthropic", "claude-haiku-4-5-20251001")
    assert result["api_base"] == "https://self-hosted.example.com"
    assert result["provider"] == "otari"
    assert result["model"] == "anthropic:claude-haiku-4-5-20251001"


def test_otari_mode_falls_back_when_no_token():
    env = {"LLM_GATEWAY": "otari"}
    with patch.dict(os.environ, env, clear=False):
        for var in ("OTARI_AI_TOKEN", "OTARI_PLATFORM_TOKEN", "GATEWAY_PLATFORM_TOKEN"):
            os.environ.pop(var, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_completion_kwargs("anthropic", "claude-sonnet-4-6")
    assert result == {}
    assert any("OTARI_AI_TOKEN" in str(w.message) for w in caught)


def test_otari_mode_accepts_legacy_platform_token():
    env = {"LLM_GATEWAY": "otari", "OTARI_PLATFORM_TOKEN": "legacy-token"}
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("OTARI_AI_TOKEN", None)
        result = resolve_completion_kwargs("gemini", "gemini-2.5-flash")
    assert result["provider"] == "otari"
    assert result["model"] == "gemini:gemini-2.5-flash"


def test_unknown_gateway_value_returns_empty_dict():
    env = {"LLM_GATEWAY": "someother"}
    with patch.dict(os.environ, env):
        result = resolve_completion_kwargs("openai", "gpt-5-mini")
    assert result == {}
