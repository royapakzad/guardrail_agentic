"""Tests for clean-text fetch introduced in PR #16."""

from unittest.mock import MagicMock, patch

import pytest
from tools import ToolError, _fetch_main_text


def _mock_response(html: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = html
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


ARTICLE_HTML = """
<html><head><title>Test</title></head>
<body>
  <nav>Menu item 1 | Menu item 2</nav>
  <article>
    <h1>Real Article Title</h1>
    <p>This is the actual article content that matters.</p>
    <p>Second paragraph with important information.</p>
  </article>
  <footer>Cookie policy | Terms | Privacy</footer>
</body></html>
"""


def test_fetch_main_text_returns_article_content_with_trafilatura():
    with patch("requests.get", return_value=_mock_response(ARTICLE_HTML)):
        try:
            import trafilatura  # noqa: F401

            trafilatura_available = True
        except ImportError:
            trafilatura_available = False

        if not trafilatura_available:
            pytest.skip("trafilatura not installed")

        content = _fetch_main_text("http://example.com", max_chars=4000)
        assert "Real Article Title" in content or "article content" in content.lower()
        # Should not contain nav/footer boilerplate
        assert "Cookie policy" not in content


def test_fetch_main_text_fallback_to_bs4_when_trafilatura_absent():
    with patch("requests.get", return_value=_mock_response(ARTICLE_HTML)):
        with patch.dict("sys.modules", {"trafilatura": None}):
            content = _fetch_main_text("http://example.com", max_chars=4000)
    # BS4 fallback should at least strip script/style/nav/footer
    assert "article content" in content.lower() or "Real Article Title" in content


def test_fetch_main_text_raises_tool_error_on_network_failure():
    import requests

    with patch("requests.get", side_effect=requests.RequestException("connection refused")):
        with pytest.raises(ToolError, match="Failed to fetch"):
            _fetch_main_text("http://unreachable.example.com", max_chars=4000)


def test_fetch_main_text_respects_max_chars():
    long_html = "<html><body>" + "<p>word </p>" * 2000 + "</body></html>"
    with patch("requests.get", return_value=_mock_response(long_html)):
        content = _fetch_main_text("http://example.com", max_chars=500)
    assert len(content) <= 500
