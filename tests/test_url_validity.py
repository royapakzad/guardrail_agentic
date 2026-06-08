"""Tests for check_url_validity (PR-10): timeouts are unverified, not broken."""

import json

import requests
import tools
from agentic_runner import _summarize_tool_result


class _Resp:
    def __init__(self, status, url="https://example.org", history=()):
        self.status_code = status
        self.url = url
        self.history = list(history)

    def close(self):
        return None


def test_timeout_is_unverified_not_broken(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.Timeout("slow government portal")

    monkeypatch.setattr(requests, "head", boom)
    monkeypatch.setattr(requests, "get", boom)
    result = tools.check_url_validity("https://www.ofpra.gouv.fr")
    assert result["valid"] is None  # NOT False — we couldn't confirm it's broken
    assert result["timed_out"] is True
    assert "do NOT apply" in result["note"]


def test_reachable_url_is_valid(monkeypatch):
    monkeypatch.setattr(requests, "head", lambda *a, **k: _Resp(200))
    result = tools.check_url_validity("https://example.org")
    assert result["valid"] is True
    assert result["timed_out"] is False


def test_404_is_broken(monkeypatch):
    monkeypatch.setattr(requests, "head", lambda *a, **k: _Resp(404))
    result = tools.check_url_validity("https://example.org/missing")
    assert result["valid"] is False


def test_head_rejected_falls_back_to_get(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "head", lambda *a, **k: (calls.append("head"), _Resp(405))[1])
    monkeypatch.setattr(requests, "get", lambda *a, **k: (calls.append("get"), _Resp(200))[1])
    result = tools.check_url_validity("https://example.org")
    assert calls == ["head", "get"]
    assert result["valid"] is True


def test_summarize_marks_unverified():
    summary = _summarize_tool_result(
        "check_url_validity",
        {"url": "https://x"},
        json.dumps({"valid": None, "status_code": None}),
    )
    assert "unverified" in summary
