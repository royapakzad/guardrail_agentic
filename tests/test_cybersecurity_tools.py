"""Tests for the cybersecurity / social-engineering domain tools (Issue #25). HTTP is mocked."""

import tools
from tools import get_tool_schemas


def test_cybersecurity_group_registered():
    names = [s["function"]["name"] for s in get_tool_schemas("cybersecurity")]
    # check_acronym is intentionally excluded from the live default tool set
    # (see tools.TOOL_GROUPS) -- pre-run acronym checks cover it at zero
    # tool-call-budget cost instead.
    assert names == [
        "search_web",
        "fetch_url",
        "check_url_validity",
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


# ── urlscan_check verdict-source bugs (reported by an annotator, scenario 15,
# https://kkgt.3e558.sod777.com/): "overall" can under-report a malicious
# verdict the ML "engines" source already caught, and the search API's
# compact hits sometimes carry NO verdict data at all for a URL that DOES
# have a real verdict in its full result -- silently returning malicious=None
# looked identical to "checked, found clean". ─────────────────────────────────

def test_urlscan_check_catches_malicious_engines_verdict_overall_misses(monkeypatch):
    """Bug 1: "overall" says not-malicious, but "engines" (the ML verdict)
    flags it -- the old code only ever looked at "overall" and would have
    reported malicious=False here. Must check every source and take the worst."""
    fake = {
        "results": [{
            "_id": "abc123",
            "task": {"time": "t", "tags": []},
            "verdicts": {
                "overall": {"malicious": False, "score": 0},
                "engines": {"malicious": True, "score": 70, "hasVerdicts": True},
                "community": {},
                "urlscan": {},
            },
        }]
    }
    monkeypatch.setattr(tools, "_http_json", lambda *a, **k: fake)
    result = tools.urlscan_check("http://example.com")
    assert result["malicious"] is True
    assert result["score"] == 70
    assert result["verdict_source"] == "search"


def test_urlscan_check_no_verdict_data_without_api_key_reports_absence_not_safety(monkeypatch):
    """Bug 2 (the exact reported case): the search hit exists (found=True) but
    its `verdicts` object carries no data at all for any source. Without an
    API key there's nothing more we can do -- must report this as a data gap,
    not silently return malicious=None looking like a checked-clean result."""
    fake = {
        "results": [{
            "_id": "019ec7cb-a2df-72db-9c42-32b724f69834",
            "task": {"time": "t", "tags": []},
            "verdicts": None,
        }]
    }
    monkeypatch.setattr(tools, "_http_json", lambda *a, **k: fake)
    monkeypatch.delenv("URLSCAN_API_KEY", raising=False)
    result = tools.urlscan_check("https://kkgt.3e558.sod777.com/")
    assert result["found"] is True
    assert result["malicious"] is None
    assert result["verdict_source"] == "none"
    assert "absence of data" in result["note"]
    assert "not evidence the URL is safe" in result["note"]


def test_urlscan_check_falls_back_to_authenticated_result_when_key_set(monkeypatch):
    """When the search hit has no usable verdict AND URLSCAN_API_KEY is set,
    escalate to GET /api/v1/result/{uuid}/ with the API key and use ITS
    verdicts instead."""
    search_response = {
        "results": [{
            "_id": "abc123",
            "task": {"time": "t", "tags": []},
            "verdicts": None,
        }]
    }
    full_result = {
        "verdicts": {
            "overall": {},
            "engines": {"malicious": True, "score": 85, "hasVerdicts": True},
        }
    }
    calls = []

    def fake_http_json(url, *, params=None, timeout=20, headers=None):
        calls.append({"url": url, "params": params, "headers": headers})
        if "result/" in url:
            assert headers == {"API-Key": "test-key-123"}
            return full_result
        return search_response

    monkeypatch.setattr(tools, "_http_json", fake_http_json)
    monkeypatch.setenv("URLSCAN_API_KEY", "test-key-123")

    result = tools.urlscan_check("http://example.com")

    assert result["malicious"] is True
    assert result["score"] == 85
    assert result["verdict_source"] == "authenticated_result"
    assert len(calls) == 2  # search, then the authenticated result fetch
    assert calls[1]["url"] == "https://urlscan.io/api/v1/result/abc123/"


def test_urlscan_check_authenticated_fetch_failure_falls_back_gracefully(monkeypatch):
    """If the authenticated fetch itself fails (bad/expired key, network
    error, ...), don't crash -- fall back to the inconclusive search-hit
    result and say so."""
    search_response = {
        "results": [{"_id": "abc123", "task": {"time": "t", "tags": []}, "verdicts": None}]
    }

    def fake_http_json(url, *, params=None, timeout=20, headers=None):
        if "result/" in url:
            raise tools.ToolError("403 Forbidden")
        return search_response

    monkeypatch.setattr(tools, "_http_json", fake_http_json)
    monkeypatch.setenv("URLSCAN_API_KEY", "bad-key")

    result = tools.urlscan_check("http://example.com")
    assert result["found"] is True
    assert result["malicious"] is None
    assert result["verdict_source"] == "none"


def test_urlscan_check_does_not_escalate_when_search_hit_already_has_data(monkeypatch):
    """No wasted API call when the search hit's own verdicts are already
    conclusive -- even with an API key configured."""
    fake = {
        "results": [{
            "_id": "abc123",
            "task": {"time": "t", "tags": []},
            "verdicts": {"overall": {"malicious": True, "score": 90}},
        }]
    }
    calls = []
    monkeypatch.setattr(tools, "_http_json", lambda *a, **k: (calls.append(1), fake)[1])
    monkeypatch.setenv("URLSCAN_API_KEY", "test-key-123")

    result = tools.urlscan_check("http://example.com")
    assert result["malicious"] is True
    assert result["verdict_source"] == "search"
    assert len(calls) == 1  # only the search call, no escalation needed


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
