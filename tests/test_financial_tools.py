"""Tests for the financial domain tools (PR-12). The cached lists are stubbed."""

import tools
from tools import get_tool_schemas


def test_financial_group_registered():
    names = [s["function"]["name"] for s in get_tool_schemas("financial")]
    assert names == ["search_web", "entity_registration", "sanctions_screen"]


def test_entity_registration_finds_ticker(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_SEC_TICKERS_CACHE",
        [{"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}],
    )
    result = tools.entity_registration("NVDA")
    assert result["registered"] is True
    assert result["matches"][0]["name"] == "NVIDIA CORP"
    assert result["matches"][0]["cik"] == 1045810


def test_entity_registration_unknown_is_unregistered(monkeypatch):
    monkeypatch.setattr(
        tools, "_SEC_TICKERS_CACHE", [{"cik_str": 1, "ticker": "AAA", "title": "Real Company Inc"}]
    )
    result = tools.entity_registration("Definitely Fake Capital LLC")
    assert result["registered"] is False
    assert result["matches"] == []


def test_sanctions_screen_flags_sdn_match(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_OFAC_SDN_CACHE",
        [{"name": "BANCO NACIONAL DE CUBA", "type": "entity", "program": "CUBA"}],
    )
    result = tools.sanctions_screen("Banco Nacional de Cuba")
    assert result["sanctioned"] is True
    assert result["matches"][0]["program"] == "CUBA"


def test_sanctions_screen_clean_name(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_OFAC_SDN_CACHE",
        [{"name": "BANCO NACIONAL DE CUBA", "type": "entity", "program": "CUBA"}],
    )
    result = tools.sanctions_screen("Acme Friendly Bakery")
    assert result["sanctioned"] is False


def test_sanctions_screen_short_query_is_safe(monkeypatch):
    monkeypatch.setattr(tools, "_OFAC_SDN_CACHE", [])
    result = tools.sanctions_screen("ab")
    assert result["sanctioned"] is None
