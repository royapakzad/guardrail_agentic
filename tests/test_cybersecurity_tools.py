"""Tests for the cybersecurity / social-engineering domain tools (Issue #25). HTTP is mocked."""

import tools
from tools import get_tool_schemas


def test_cybersecurity_group_registered():
    names = [s["function"]["name"] for s in get_tool_schemas("cybersecurity")]
    assert names == [
        "search_web",
        "fetch_url",
        "check_url_validity",
        "check_acronym",
        "urlscan_check",
        "scam_guidance_lookup",
    ]


def test_urlscan_check_flags_malicious(monkeypatch):
    fake = {
        "results": [
            {
                "_id": "abc123",
                "task": {"time": "2026-07-01T00:00:00.000Z", "tags": ["phishing", "paypal"]},
                "verdicts": {"overall": {"malicious": True, "score": 90}},
            }
        ]
    }
    monkeypatch.setattr(tools, "_http_json", lambda *a, **k: fake)
    result = tools.urlscan_check("http://paypal-verify-account.tk/login")
    assert result["found"] is True
    assert result["malicious"] is True
    assert result["scan_report_url"] == "https://urlscan.io/result/abc123/"
    assert "phishing" in result["tags"]


def test_urlscan_check_quotes_url_in_lucene_query(monkeypatch):
    """Regression test for Issue #38: an unquoted `page.url:{url}` term is
    invalid Lucene syntax once the URL's `:` and `/` are in play, and
    URLScan rejects it with 400. The term must be a quoted phrase."""
    captured = {}

    def fake_http_json(url, *, params=None, timeout=20):
        captured["params"] = params
        return {"results": []}

    monkeypatch.setattr(tools, "_http_json", fake_http_json)
    tools.urlscan_check("http://paypal-verify-account.tk/login")
    assert captured["params"]["q"] == 'page.url:"http://paypal-verify-account.tk/login"'


def test_urlscan_check_no_scan_found(monkeypatch):
    monkeypatch.setattr(tools, "_http_json", lambda *a, **k: {"results": []})
    result = tools.urlscan_check("http://example.com")
    assert result["found"] is False
    assert result["malicious"] is None


def test_urlscan_check_empty_url_is_safe():
    result = tools.urlscan_check("")
    assert result["found"] is False


def test_urlscan_check_handles_tool_error(monkeypatch):
    def raise_error(*a, **k):
        raise tools.ToolError("network down")

    monkeypatch.setattr(tools, "_http_json", raise_error)
    result = tools.urlscan_check("http://example.com")
    assert result["found"] is False
    assert "network down" in result["note"]


def test_scam_guidance_lookup_matches_gift_card():
    result = tools.scam_guidance_lookup("gift card")
    assert result["matched"] is True
    assert result["results"][0]["authority"] == "FTC"
    assert result["results"][0]["url"].startswith("https://consumer.ftc.gov")


def test_scam_guidance_lookup_matches_acronym():
    result = tools.scam_guidance_lookup("BEC")
    assert result["matched"] is True
    assert result["results"][0]["scam_type"] == "Business email compromise (BEC)"


def test_scam_guidance_lookup_no_match_for_unrelated_query():
    result = tools.scam_guidance_lookup("unrelated topic xyz")
    assert result["matched"] is False
    assert result["results"] == []


def test_scam_guidance_lookup_empty_query_is_safe():
    result = tools.scam_guidance_lookup("")
    assert result["matched"] is False
