"""Tests for the acronym checker: non-Latin script extraction (Issue #27) and
the two-query confirm/discover redesign (search-quality follow-up).

Verifies that:
1. _extract_acronym_expansions() (agentic_runner.py) captures expansions that
   start with non-Latin script (Farsi, Arabic), not just Latin/accented-Latin.
2. check_acronym() (tools.py) builds a confirmatory query from the claim's own
   words (whatever script they're in) — no per-language translation dict, so
   this works the same way for any language without special-casing.
3. check_acronym() scores a correct non-Latin expansion as likely_correct and
   a wrong one as likely_wrong, via word-level phrase matching rather than
   scattered word-bag overlap.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))

import tools
from agentic_runner import _extract_acronym_expansions


# ── extraction regex must accept non-Latin expansions ──────────────────────────

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


# ── confirmatory query is built from the claim's own words, any script ─────────

def test_confirm_query_uses_claims_own_farsi_words(monkeypatch):
    captured_queries = []

    def fake_search_web(query, max_results=3, language=""):
        captured_queries.append((query, language))
        return []

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    tools.check_acronym("UNHCR", "کمیساریای عالی سازمان ملل برای پناهندگان", context_language="fa")

    # No translation dictionary involved — the claim's own Farsi words appear
    # directly in the confirmatory query, and language is passed through.
    assert any("کمیساریای" in q for q, _ in captured_queries)
    assert all(lang == "fa" for _, lang in captured_queries)


def test_both_queries_run_for_english_too(monkeypatch):
    captured_queries = []

    def fake_search_web(query, max_results=3, language=""):
        captured_queries.append(query)
        return []

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    tools.check_acronym("NATO", "North Atlantic Treaty Organization", context_language="en")

    assert len(captured_queries) == 2
    assert any("North Atlantic Treaty Organization" in q for q in captured_queries)
    assert any("meaning" in q for q in captured_queries)


# ── scoring: correct vs wrong non-Latin expansions ──────────────────────────────

def test_correct_farsi_expansion_scores_likely_correct(monkeypatch):
    def fake_search_web(query, max_results=3, language=""):
        return [
            {
                "title": "UNHCR فارسی",
                "url": "https://unhcr.org/fa",
                "snippet": "کمیساریای عالی سازمان ملل برای پناهندگان",
            }
        ]

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    result = tools.check_acronym(
        "UNHCR", "کمیساریای عالی سازمان ملل برای پناهندگان", context_language="fa"
    )

    assert result["match_score"] >= 0.6
    assert result["verdict_hint"] == "likely_correct"


def test_wrong_farsi_expansion_flagged_likely_wrong(monkeypatch):
    def fake_search_web(query, max_results=3, language=""):
        # Both confirm and discover searches only turn up the *real* (and
        # different) expansion — nothing resembling the claim.
        return [
            {
                "title": "UNHCR",
                "url": "https://unhcr.org",
                "snippet": "کمیساریای عالی سازمان ملل برای پناهندگان به آوارگان کمک می‌کند",
            }
        ]

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    result = tools.check_acronym(
        "UNHCR",
        "سازمان بهداشت جهانی",  # wrong: World Health Organization, not UNHCR
        context_language="fa",
    )

    assert result["match_score"] < 0.6


def test_fabricated_expansion_scores_zero_not_a_partial_match(monkeypatch):
    # Regression: an earlier character-level scorer gave this fabricated
    # expansion 0.26 (borderline "unclear") purely from a coincidental
    # 6-character substring match ("ion of") that word-level matching can't
    # produce, since "Union" and "registration" are different words entirely.
    def fake_search_web(query, max_results=3, language=""):
        if "meaning" not in query:
            return []
        return [
            {
                "title": "GUDA official",
                "url": "https://x",
                "snippet": (
                    "GUDA stands for Guichet Unique pour Demandeurs d'Asile, "
                    "the French asylum registration office."
                ),
            }
        ]

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    result = tools.check_acronym("GUDA", "General Union of Displaced Asylees", context_language="en")

    assert result["match_score"] == 0.0
    assert result["verdict_hint"] == "likely_wrong"
    assert "Guichet Unique" in result["note"]  # surfaces what was actually found


def test_latin_expansion_scoring_unaffected(monkeypatch):
    def fake_search_web(query, max_results=3, language=""):
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


# ── domain hint anchoring ────────────────────────────────────────────────────

def test_domain_hint_appended_to_both_queries(monkeypatch):
    captured_queries = []

    def fake_search_web(query, max_results=3, language=""):
        captured_queries.append(query)
        return []

    monkeypatch.setattr(tools, "search_web", fake_search_web)
    tools.set_domain_hint_for_group("financial")
    try:
        tools.check_acronym("FCRA", "Fair Credit Reporting Act", context_language="en")
    finally:
        tools.set_domain_hint_for_group("default")  # don't leak into other tests

    assert all("finance credit regulation" in q for q in captured_queries)
