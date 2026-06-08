"""Tests for the Otari gateway resolver (PR-7). Pure env-driven logic, no network."""

import pytest
from llm_gateway import gateway_mode, resolve_completion_kwargs

_GATEWAY_ENV = ("LLM_GATEWAY", "OTARI_API_BASE", "OTARI_API_KEY", "GATEWAY_API_BASE", "GATEWAY_API_KEY")


@pytest.fixture(autouse=True)
def _clear_gateway_env(monkeypatch):
    for var in _GATEWAY_ENV:
        monkeypatch.delenv(var, raising=False)


def test_direct_mode_is_default():
    assert gateway_mode() == "direct"
    assert resolve_completion_kwargs(provider="openai", model="gpt-5-nano") == {
        "provider": "openai",
        "model": "gpt-5-nano",
    }


def test_otari_mode_routes_through_gateway(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY", "otari")
    monkeypatch.setenv("OTARI_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("OTARI_API_KEY", "vk-test")
    assert resolve_completion_kwargs(provider="anthropic", model="claude-sonnet-4-6") == {
        "provider": "otari",
        "model": "anthropic:claude-sonnet-4-6",
        "api_base": "http://localhost:8000/v1",
        "api_key": "vk-test",
    }


def test_otari_without_api_base_falls_back_to_direct(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY", "otari")  # but no OTARI_API_BASE configured
    assert resolve_completion_kwargs(provider="openai", model="gpt-5-nano") == {
        "provider": "openai",
        "model": "gpt-5-nano",
    }


def test_legacy_gateway_env_names_are_accepted(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY", "otari")
    monkeypatch.setenv("GATEWAY_API_BASE", "http://gw:8000/v1")
    monkeypatch.setenv("GATEWAY_API_KEY", "legacy-key")
    out = resolve_completion_kwargs(provider="openai", model="gpt-5-nano")
    assert out["provider"] == "otari"
    assert out["api_base"] == "http://gw:8000/v1"
    assert out["api_key"] == "legacy-key"


def test_otari_mode_without_key_omits_api_key(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY", "otari")
    monkeypatch.setenv("OTARI_API_BASE", "http://localhost:8000/v1")
    out = resolve_completion_kwargs(provider="openai", model="gpt-5-nano")
    assert "api_key" not in out
    assert out["provider"] == "otari"
