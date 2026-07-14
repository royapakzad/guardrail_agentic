"""Tests that the `language` parameter added to search_web() actually reaches
each backend, using whatever locale-biasing mechanism that backend supports.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

import tools


# ── Tavily ───────────────────────────────────────────────────────────────────

class _FakeTavilyClient:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return {"results": [{"title": "t", "url": "u", "content": "c"}]}


def test_tavily_passes_country_for_mapped_language(monkeypatch):
    fake_client = _FakeTavilyClient()
    monkeypatch.setattr(tools, "_tavily_client", lambda: fake_client)

    tools._search_tavily("query", max_results=3, language="fa")

    assert fake_client.calls[0][1].get("country") == "iran"


def test_tavily_omits_country_for_unmapped_language(monkeypatch):
    fake_client = _FakeTavilyClient()
    monkeypatch.setattr(tools, "_tavily_client", lambda: fake_client)

    tools._search_tavily("query", max_results=3, language="es")

    # Spanish spans many countries — deliberately not mapped, no guess made.
    assert "country" not in fake_client.calls[0][1]


def test_tavily_works_with_no_language_arg(monkeypatch):
    fake_client = _FakeTavilyClient()
    monkeypatch.setattr(tools, "_tavily_client", lambda: fake_client)

    results = tools._search_tavily("query", max_results=3)

    assert results[0]["title"] == "t"
    assert "country" not in fake_client.calls[0][1]


# ── SearXNG ──────────────────────────────────────────────────────────────────

def test_searxng_passes_language_param(monkeypatch):
    captured = {}

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"title": "t", "url": "u", "content": "c"}]}

    def fake_get(url, params=None, **kwargs):
        captured.update(params or {})
        return _FakeResp()

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    tools._search_searxng("query", max_results=3, language="fa")
    assert captured["language"] == "fa"


def test_searxng_defaults_to_all_languages(monkeypatch):
    captured = {}

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"title": "t", "url": "u", "content": "c"}]}

    def fake_get(url, params=None, **kwargs):
        captured.update(params or {})
        return _FakeResp()

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    tools._search_searxng("query", max_results=3)
    assert captured["language"] == "all"


# ── DuckDuckGo ───────────────────────────────────────────────────────────────

def test_duckduckgo_passes_region_for_mapped_language(monkeypatch):
    captured = {}

    class _FakeDDGS:
        def __init__(self, timeout=20):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=5, region="wt-wt"):
            captured["region"] = region
            return [{"title": "t", "href": "u", "body": "b"}]

    import types
    fake_module = types.ModuleType("ddgs")
    fake_module.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)

    tools._search_duckduckgo("query", max_results=3, language="fr")
    assert captured["region"] == "fr-fr"


def test_duckduckgo_falls_back_to_worldwide_for_unmapped_language(monkeypatch):
    captured = {}

    class _FakeDDGS:
        def __init__(self, timeout=20):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=5, region="wt-wt"):
            captured["region"] = region
            return [{"title": "t", "href": "u", "body": "b"}]

    import types
    fake_module = types.ModuleType("ddgs")
    fake_module.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)

    tools._search_duckduckgo("query", max_results=3, language="fa")
    assert captured["region"] == "wt-wt"
