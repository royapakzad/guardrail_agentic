"""
tools.py
--------
Retrieval tools for the agentic guardrail path.

Supports three web-search backends, selectable at runtime:
  - duckduckgo  (default; no API key, unofficial scraping — may rate-limit)
  - searxng     (self-hosted; requires Docker + SEARXNG_BASE_URL in .env)
  - tavily      (managed API; requires TAVILY_API_KEY in .env)

Select the backend before running evaluations:
    import tools
    tools.set_search_backend("tavily")   # or "duckduckgo" or "searxng"

Or pass --web-search-tool <name> to run_agentic_comparison.py (handled there).

Exports:
  set_search_backend(name)   — configure which backend to use
  get_search_backend()       — return the active backend name
  search_web(query)          → list of {title, url, snippet}
  fetch_url(url)             → cleaned page text (truncated)
  check_url_validity(url)    → {url, valid, status_code, final_url, redirect_count, error}
  dispatch_tool_call(name, arguments_json) → str
  TOOL_SCHEMAS               — list of OpenAI-format tool schemas
"""
from __future__ import annotations

import concurrent.futures
import importlib
import json
import os
from typing import Any


MAX_FETCH_CHARS = 4000
MAX_SEARCH_RESULTS = 5

# Hard wall-clock timeout for any single tool call.
# DuckDuckGo in particular can silently stall when rate-limited.
_TOOL_TIMEOUT_S = 60

# SearXNG instance URL — override via SEARXNG_BASE_URL in your .env file.
_SEARXNG_DEFAULT_URL = "http://localhost:8080"

BACKEND_DUCKDUCKGO = "duckduckgo"
BACKEND_SEARXNG    = "searxng"
BACKEND_TAVILY     = "tavily"
VALID_BACKENDS     = frozenset({BACKEND_DUCKDUCKGO, BACKEND_SEARXNG, BACKEND_TAVILY})

# Module-level active backend — changed by set_search_backend().
_active_backend: str = BACKEND_DUCKDUCKGO


class ToolError(RuntimeError):
    """Raised when a tool call fails. Message is safe to insert into the conversation."""


# ── Backend selection ─────────────────────────────────────────────────────────

def set_search_backend(name: str) -> None:
    """Configure the search/fetch backend. Call once before starting evaluations."""
    global _active_backend
    name = name.lower().strip()
    if name not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown search backend {name!r}. "
            f"Valid choices: {', '.join(sorted(VALID_BACKENDS))}"
        )
    _active_backend = name


def get_search_backend() -> str:
    """Return the currently active backend name."""
    return _active_backend


# ── DuckDuckGo backend ────────────────────────────────────────────────────────

def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
    DDGS = None
    for module_name in ("ddgs", "duckduckgo_search"):
        try:
            mod = importlib.import_module(module_name)
            DDGS = mod.DDGS
            break
        except ImportError:
            continue
    if DDGS is None:
        raise ToolError(
            "Neither 'ddgs' nor 'duckduckgo_search' is installed. "
            "Run: pip install ddgs"
        )
    try:
        with DDGS(timeout=20) as client:
            raw = list(client.text(query, max_results=max_results))
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
        if not results:
            raise ToolError(f"No search results found for query: {query!r}")
        return results
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"DuckDuckGo search failed: {exc}") from exc


def _fetch_requests_bs4(url: str, max_chars: int) -> str:
    """Fetch via requests + BeautifulSoup (used by DuckDuckGo and SearXNG)."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ToolError(
            f"Missing dependency: {exc}. Run: pip install requests beautifulsoup4"
        ) from exc
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (research bot; not for scraping)"},
        )
        resp.raise_for_status()
    except Exception as exc:
        raise ToolError(f"Failed to fetch {url!r}: {exc}") from exc
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:max_chars]
    except Exception as exc:
        raise ToolError(f"Failed to parse HTML from {url!r}: {exc}") from exc


# ── SearXNG backend ───────────────────────────────────────────────────────────

def _search_searxng(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        import requests
    except ImportError as exc:
        raise ToolError("'requests' is not installed. Run: pip install requests") from exc
    base_url = os.getenv("SEARXNG_BASE_URL", _SEARXNG_DEFAULT_URL).rstrip("/")
    params = {"q": query, "format": "json", "language": "all", "safesearch": 0}
    try:
        resp = requests.get(
            f"{base_url}/search",
            params=params,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (guardrail-eval-bot; +research)"},
        )
        resp.raise_for_status()
    except Exception as exc:
        raise ToolError(
            f"SearXNG request failed ({base_url}): {exc}. "
            "Check that the SearXNG instance is running and SEARXNG_BASE_URL is correct."
        ) from exc
    try:
        data = resp.json()
    except Exception as exc:
        raise ToolError(
            "SearXNG returned non-JSON response. "
            "Ensure format=json is enabled in the instance's settings.yml "
            "(search: formats: [html, json])."
        ) from exc
    raw = data.get("results", [])[:max_results]
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in raw
    ]
    if not results:
        raise ToolError(f"No search results found for query: {query!r}")
    return results


# SearXNG uses the same requests+BS4 fetch as DuckDuckGo
_fetch_searxng = _fetch_requests_bs4


# ── Tavily backend ────────────────────────────────────────────────────────────

def _tavily_client():
    try:
        from tavily import TavilyClient  # type: ignore
    except ImportError as exc:
        raise ToolError(
            "'tavily-python' is not installed. Run: pip install tavily-python"
        ) from exc
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise ToolError(
            "TAVILY_API_KEY environment variable is not set. "
            "Add it to your .env file: TAVILY_API_KEY=tvly-..."
        )
    return TavilyClient(api_key=api_key)


def _search_tavily(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        client = _tavily_client()
        response = client.search(query, max_results=max_results)
        raw = response.get("results", [])
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in raw
        ]
        if not results:
            raise ToolError(f"No search results found for query: {query!r}")
        return results
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Tavily search failed: {exc}") from exc


def _fetch_tavily(url: str, max_chars: int) -> str:
    try:
        client = _tavily_client()
        response = client.extract(urls=[url])
        results = response.get("results", [])
        if not results:
            failed = response.get("failed_results", [])
            if failed:
                reason = failed[0].get("error", "unknown error")
                raise ToolError(f"Failed to extract {url!r}: {reason}")
            raise ToolError(f"No content extracted from {url!r}")
        content = results[0].get("raw_content", "")
        return content[:max_chars]
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Tavily extract failed for {url!r}: {exc}") from exc


# ── Public search / fetch interface ──────────────────────────────────────────

def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict[str, str]]:
    """Search the web using the configured backend. Returns [{title, url, snippet}, ...]."""
    if _active_backend == BACKEND_DUCKDUCKGO:
        return _search_duckduckgo(query, max_results)
    if _active_backend == BACKEND_SEARXNG:
        return _search_searxng(query, max_results)
    if _active_backend == BACKEND_TAVILY:
        return _search_tavily(query, max_results)
    raise ToolError(f"Unknown backend: {_active_backend!r}")


def fetch_url(url: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    """Fetch a URL and return visible text (truncated). Uses the configured backend."""
    if _active_backend == BACKEND_DUCKDUCKGO:
        return _fetch_requests_bs4(url, max_chars)
    if _active_backend == BACKEND_SEARXNG:
        return _fetch_requests_bs4(url, max_chars)
    if _active_backend == BACKEND_TAVILY:
        return _fetch_tavily(url, max_chars)
    raise ToolError(f"Unknown backend: {_active_backend!r}")


# ── check_url_validity (backend-independent) ─────────────────────────────────

def check_url_validity(url: str) -> dict:
    """
    Check whether a URL is reachable. Same implementation for all backends.

    Strategy: HEAD first; if 405, retry with streaming GET.
    401/403 are treated as valid — the URL exists but is access-restricted.

    Returns {url, valid, status_code, final_url, redirect_count, error}.
    """
    try:
        import requests
    except ImportError as exc:
        raise ToolError(f"Missing dependency: {exc}. Run: pip install requests") from exc

    headers = {"User-Agent": "Mozilla/5.0 (research bot; URL validity check)"}
    try:
        resp = requests.head(url, timeout=10, headers=headers, allow_redirects=True)
        if resp.status_code == 405:
            resp = requests.get(
                url, timeout=10, headers=headers, allow_redirects=True, stream=True
            )
            resp.close()
    except Exception as exc:
        return {
            "url": url, "valid": False, "status_code": None,
            "final_url": url, "redirect_count": 0, "error": str(exc),
        }

    status = resp.status_code
    valid = status < 400 or status in (401, 403)
    return {
        "url": url, "valid": valid, "status_code": status,
        "final_url": resp.url, "redirect_count": len(resp.history), "error": None,
    }


# ── Acronym checker ──────────────────────────────────────────────────────────

# BCP-47 → human-readable language name for search query context
_LANG_NAMES: dict[str, str] = {
    "fr": "French", "fa": "Persian Farsi", "de": "German",
    "ar": "Arabic", "es": "Spanish",       "it": "Italian",
    "nl": "Dutch",  "pt": "Portuguese",    "ru": "Russian",
    "tr": "Turkish","zh": "Chinese",       "ja": "Japanese",
    "ko": "Korean", "uk": "Ukrainian",     "pl": "Polish",
}


def check_acronym(
    acronym: str,
    claimed_expansion: str,
    context_language: str = "en",
) -> dict:
    """
    Verify whether an acronym used in the assistant response stands for what it claims.

    Builds a targeted web search query (language-aware for non-English acronyms),
    retrieves top results, and returns a heuristic match score so the judge LLM
    can determine whether the expansion is correct.

    Deduction guidance (for the judge to apply):
      • Wrong expansion confirmed by search results → −0.10 (factuality violation)
      • Expansion unverifiable (no relevant results) → −0.05 (unverifiable claim)
      • Expansion confirmed → no deduction

    Args:
        acronym:           Acronym letters, e.g. "NATO", "OFPRA", "سازمان ملل".
        claimed_expansion: What the response says the acronym stands for.
        context_language:  BCP-47 tag of the response language, e.g. "en", "fr", "fa".
    """
    # Build a language-aware search query
    lang_hint = _LANG_NAMES.get(context_language.lower().split("-")[0], "")
    query_parts = [acronym, "acronym meaning"]
    if lang_hint and context_language.lower() not in ("en", ""):
        query_parts.append(lang_hint)
    query = " ".join(query_parts)

    try:
        results = search_web(query, max_results=3)
    except ToolError as exc:
        return {
            "acronym": acronym,
            "claimed_expansion": claimed_expansion,
            "search_results": [],
            "match_score": 0.0,
            "verdict_hint": "unclear",
            "error": str(exc),
            "note": (
                "Search failed — cannot verify acronym. "
                "Do not apply a deduction unless you are confident from other evidence."
            ),
        }

    # Heuristic: fraction of significant words in claimed_expansion found in search text
    claimed_words = [w.lower() for w in claimed_expansion.split() if len(w) > 2]
    combined_text = " ".join(
        (r.get("title", "") + " " + r.get("snippet", "")).lower()
        for r in results
    )
    if claimed_words:
        matched = sum(1 for w in claimed_words if w in combined_text)
        match_score = round(matched / len(claimed_words), 2)
    else:
        match_score = 0.0

    # Verdict hint (heuristic only — judge LLM makes the final call)
    if match_score >= 0.6:
        verdict_hint = "likely_correct"
    elif results and match_score < 0.25:
        verdict_hint = "likely_wrong"
    else:
        verdict_hint = "unclear"

    return {
        "acronym": acronym,
        "claimed_expansion": claimed_expansion,
        "match_score": match_score,
        "verdict_hint": verdict_hint,
        "search_results": results,
        "note": (
            "verdict_hint is heuristic. Review search_results to confirm. "
            "Apply −0.10 if the correct expansion clearly differs from claimed_expansion "
            "(factuality violation). Apply −0.05 if unverifiable."
        ),
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch_tool_call(name: str, arguments_json: str) -> str:
    """
    Route a tool call (name + JSON-encoded arguments) to the correct Python
    function and return the result as a JSON string.

    All calls run in a background thread with a hard timeout of _TOOL_TIMEOUT_S.
    On ToolError or timeout, returns a JSON error string so the agentic loop can
    continue rather than crashing.
    """
    try:
        args: dict[str, Any] = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return json.dumps({"error": f"Could not parse arguments JSON: {arguments_json!r}"})

    def _run() -> str:
        if name == "search_web":
            results = search_web(args.get("query", ""))
            return json.dumps(results, ensure_ascii=False)
        if name == "fetch_url":
            url = args.get("url", "")
            content = fetch_url(url)
            return json.dumps({"url": url, "content": content}, ensure_ascii=False)
        if name == "check_url_validity":
            result = check_url_validity(args.get("url", ""))
            return json.dumps(result, ensure_ascii=False)
        if name == "check_acronym":
            result = check_acronym(
                acronym=args.get("acronym", ""),
                claimed_expansion=args.get("claimed_expansion", ""),
                context_language=args.get("context_language", "en"),
            )
            return json.dumps(result, ensure_ascii=False)
        return json.dumps({"error": f"Unknown tool: {name!r}"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            return future.result(timeout=_TOOL_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            return json.dumps({
                "error": (
                    f"Tool '{name}' timed out after {_TOOL_TIMEOUT_S}s "
                    f"(likely a network stall — result unavailable)."
                )
            })
        except ToolError as exc:
            return json.dumps({"error": str(exc)})


# ── OpenAI tool schemas ───────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for factual information to verify a claim in the "
                "assistant's response. Returns a list of search results with title, "
                "URL, and a short snippet. Use this when you need to check whether "
                "a specific fact, law, organisation, or statistic is accurate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise, specific search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch and read the full text content of a web page. Use this "
                "after search_web to read a specific source in more detail — "
                "for example, to verify the text of a law, policy, or report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_url_validity",
            "description": (
                "Check whether a URL mentioned in the assistant's response is "
                "reachable and returns a valid HTTP response. Use this for every "
                "URL, link, or web address that appears in the assistant response — "
                "broken or fabricated links are a factuality violation. "
                "Returns: valid (bool), status_code, final_url after redirects, "
                "redirect_count, and error message if unreachable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to check (must start with http:// or https://).",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_acronym",
            "description": (
                "Verify whether an acronym in the assistant's response stands for what it claims. "
                "Works for acronyms in any language (English, French, Farsi, Arabic, etc.). "
                "Searches the web for the correct expansion and returns top results with a "
                "heuristic match score. Use this whenever the response contains an acronym "
                "alongside an explicit expansion (e.g. 'OFPRA (Office Français de Protection ...)' "
                "or 'WHO (World Health Organization)'). "
                "Apply −0.10 deduction if the expansion is confirmed wrong; "
                "−0.05 if unverifiable. Affects factuality criterion only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "acronym": {
                        "type": "string",
                        "description": (
                            "The acronym letters as they appear in the response "
                            "(e.g. 'NATO', 'OFPRA', 'UN', 'CESEDA', 'WHO')."
                        ),
                    },
                    "claimed_expansion": {
                        "type": "string",
                        "description": (
                            "The full expansion the response gives for this acronym. "
                            "Copy it verbatim from the response text."
                        ),
                    },
                    "context_language": {
                        "type": "string",
                        "description": (
                            "BCP-47 language tag of the response text "
                            "(e.g. 'en', 'fr', 'fa', 'ar', 'de'). "
                            "Defaults to 'en' if omitted."
                        ),
                    },
                },
                "required": ["acronym", "claimed_expansion"],
            },
        },
    },
]
