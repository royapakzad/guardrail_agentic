"""Tests for Issue #27 — acronym checker fails for non-Latin script (Farsi, Arabic).

Verifies that:
1. _extract_acronym_expansions() (agentic_runner.py) captures expansions that
   start with non-Latin script (Farsi, Arabic), not just Latin/accented-Latin.
2. check_acronym() (tools.py) builds a bilingual search query that includes a
   native-language term for non-Latin-script languages.
3. check_acronym() scores a correct non-Latin expansion as likely_correct by
   also matching against a native-language search, instead of always scoring 0.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

import tools
from agentic_runner import _extract_acronym_expansions


# ── Bug 1: extraction regex must accept non-Latin expansions ──────────────────

def test_extracts_farsi_expansion_in_parens():
    text = "UNHCR (کمیساریای عالی سازمان ملل برای پناهندگان) helps refugees."
    pairs = dict(_extract_acronym_expansions(text))
    assert pairs.get("UNHCR") == "کمیساریای عالی سازمان ملل برای پناهندگان"


def test_extracts_arabic_expansion_after_dash():
    text = "IOM – المنظمة الدولية للهجرة provides support to migrants."
    pairs = dict(_extract_acronym_expansions(text))
    assert pairs.get("IOM") == "المنظمة الدولية للهجرة provides support to migrants"


def test_still_extracts_latin_expansion():
    text = "NATO (North Atlantic Treaty Organization) was founded in 1949."
    pairs = dict(_extract_acronym_expansions(text))
    assert pairs.get("NATO") == "North Atlantic Treaty Organization"


def test_still_filters_non_expansion_parenthetical():
    # "UNHCR (2023)" is not an expansion — must not be extracted.
    text = "UNHCR (2023) published a new report."
    pairs = dict(_extract_acronym_expansions(text))
    assert "UNHCR" not in pairs


# ── Bug 3: search query includes a native-language term ───────────────────────

def test_query_includes_native_term_for_farsi(monkeypatch):
    captured_queries = []

    def fake_search_web(query, max_results=3):
        captured_queries.append(query)
        return [{"title": "UNHCR", "url": "https://unhcr.org", "snippet": "کمیساریای عالی"}]

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    tools.check_acronym("UNHCR", "کمیساریای عالی سازمان ملل برای پناهندگان", context_language="fa")

    assert any("مخفف" in q for q in captured_queries)


def test_query_unchanged_for_english():
    # No native term should be appended when the context language is English.
    lang_code = "en"
    assert tools._LANG_ACRONYM_TERMS.get(lang_code, "") == ""


# ── Bug 2: correct non-Latin expansions must not be scored 0 / likely_wrong ───

def test_correct_farsi_expansion_scores_via_native_search(monkeypatch):
    def fake_search_web(query, max_results=3):
        if "مخفف" in query or "Persian" in query:
            # Native-language search finds the correct Farsi expansion.
            return [
                {
                    "title": "UNHCR فارسی",
                    "url": "https://unhcr.org/fa",
                    "snippet": "کمیساریای عالی سازمان ملل برای پناهندگان",
                }
            ]
        return [{"title": "UNHCR", "url": "https://unhcr.org", "snippet": "UN Refugee Agency"}]

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    result = tools.check_acronym(
        "UNHCR",
        "کمیساریای عالی سازمان ملل برای پناهندگان",
        context_language="fa",
    )

    assert result["match_score"] >= 0.6
    assert result["verdict_hint"] == "likely_correct"


def test_wrong_farsi_expansion_still_flagged(monkeypatch):
    def fake_search_web(query, max_results=3):
        return [
            {
                "title": "UNHCR",
                "url": "https://unhcr.org",
                "snippet": "UN Refugee Agency کمیساریای عالی سازمان ملل برای پناهندگان",
            }
        ]

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    result = tools.check_acronym(
        "UNHCR",
        "سازمان بهداشت جهانی",  # wrong: World Health Organization, not UNHCR
        context_language="fa",
    )

    assert result["match_score"] < 0.6


def test_latin_expansion_scoring_unaffected(monkeypatch):
    def fake_search_web(query, max_results=3):
        return [
            {
                "title": "NATO",
                "url": "https://nato.int",
                "snippet": "North Atlantic Treaty Organization",
            }
        ]

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    result = tools.check_acronym(
        "NATO", "North Atlantic Treaty Organization", context_language="en"
    )

    assert result["match_score"] == 1.0
    assert result["verdict_hint"] == "likely_correct"
