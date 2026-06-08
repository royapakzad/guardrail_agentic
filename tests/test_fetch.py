"""Tests for clean-text fetching (PR-9): only readable text, never raw HTML."""

import json

import requests
import tools
from agentic_runner import _summarize_tool_result

_HTML = """<html><head><title>T</title>
<script>var secret = 1;</script><style>.x{color:red}</style></head>
<body>
<nav>MENU HOME ABOUT CONTACT</nav>
<article>
<h1>Asylum rights in Croatia</h1>
<p>You can request a written court order before unlocking your device, and you
are not obliged to disclose passwords without due legal process.</p>
</article>
<footer>Cookie banner junk and copyright notice</footer>
</body></html>"""


class _Resp:
    text = _HTML

    def raise_for_status(self):
        return None


def test_fetch_main_text_returns_clean_text_no_html(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    out = tools._fetch_main_text("https://example.org/article", 4000)

    # No HTML markup leaks into the result.
    assert "<" not in out and ">" not in out
    # Script/style contents are gone.
    assert "secret" not in out
    assert "color:red" not in out
    # The main content survives.
    assert "written court order" in out


def test_summarize_fetch_url_passes_full_clean_text_not_a_preview():
    long_text = "Relevant clean sentence. " * 50  # ~1200 chars
    summary = _summarize_tool_result(
        "fetch_url",
        {"url": "https://example.org"},
        json.dumps({"url": "https://example.org", "content": long_text}),
    )
    # The actual text reaches the conversation (not truncated to a ~200-char preview).
    assert long_text.strip() in summary
    assert len(summary) > 1000
