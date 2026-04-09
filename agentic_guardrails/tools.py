"""
tools.py
--------
Retrieval tools for the agentic guardrail path.

  search_web(query)         → list of {title, url, snippet} dicts
  fetch_url(url)            → cleaned page text (truncated)
  check_url_validity(url)   → {url, valid, status_code, final_url, redirect_count, error}
  dispatch_tool_call(name, arguments_json) → str (tool result for conversation)

TOOL_SCHEMAS is the list of JSON schemas passed to OpenAI's tools= parameter.
"""
from __future__ import annotations

import json
from typing import Any


MAX_FETCH_CHARS = 4000
MAX_SEARCH_RESULTS = 5


class ToolError(RuntimeError):
    """Raised when a tool call fails. The message is safe to insert into the conversation."""


# ---------------------------------------------------------------------------
# search_web
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict[str, str]]:
    """
    Search the web using DuckDuckGo (no API key required).
    Returns a list of dicts: [{"title": ..., "url": ..., "snippet": ...}, ...]
    """
    # Support both old package name (duckduckgo_search) and new name (ddgs).
    DDGS = None
    for module_name in ("ddgs", "duckduckgo_search"):
        try:
            import importlib
            mod = importlib.import_module(module_name)
            DDGS = mod.DDGS
            break
        except ImportError:
            continue
    if DDGS is None:
        raise ToolError("Neither 'ddgs' nor 'duckduckgo_search' is installed. Run: pip install ddgs")

    try:
        with DDGS() as client:
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
            # DuckDuckGo returned nothing — surface this clearly
            raise ToolError(f"No search results found for query: {query!r}")
        return results
    except Exception as exc:
        raise ToolError(f"DuckDuckGo search failed: {exc}") from exc


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

def fetch_url(url: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    """
    Fetch a URL and return its visible text content (truncated to max_chars).

    Strips <script>, <style>, <nav>, <footer> tags before extracting text.
    Raises ToolError on HTTP errors, timeouts, or parse failures.
    """
    try:
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
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
        text = soup.get_text(separator="\n", strip=True)
        return text[:max_chars]
    except Exception as exc:
        raise ToolError(f"Failed to parse HTML from {url!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# check_url_validity
# ---------------------------------------------------------------------------

def check_url_validity(url: str) -> dict:
    """
    Check whether a URL is reachable and returns a non-error HTTP status.

    Strategy:
      1. Try HEAD first (fast, minimal bandwidth).
      2. If the server returns 405 Method Not Allowed, retry with GET.
      3. Follow redirects in both cases.

    Returns a dict with:
      url            – the original URL that was checked
      valid          – True if status_code < 400 and no network error
      status_code    – HTTP status code (int), or None if connection failed
      final_url      – the URL after all redirects (may differ from input)
      redirect_count – number of HTTP redirects followed (0 = no redirect)
      error          – network/timeout error message, or None on success
    """
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise ToolError(f"Missing dependency: {exc}. Run: pip install requests") from exc

    headers = {"User-Agent": "Mozilla/5.0 (research bot; URL validity check)"}

    try:
        resp = requests.head(url, timeout=10, headers=headers, allow_redirects=True)
        # Some servers reject HEAD; retry with GET (streaming so we don't download body)
        if resp.status_code == 405:
            resp = requests.get(
                url, timeout=10, headers=headers, allow_redirects=True, stream=True
            )
            resp.close()
    except Exception as exc:
        return {
            "url": url,
            "valid": False,
            "status_code": None,
            "final_url": url,
            "redirect_count": 0,
            "error": str(exc),
        }

    # Validity rule for URL fact-checking purposes:
    #   < 400         → valid  (2xx success or 3xx redirect that resolved)
    #   401 / 403     → valid  (server responded; the URL exists but requires
    #                           auth or is blocking automated requests — the
    #                           resource is real, just access-restricted)
    #   404 / 410     → invalid (page genuinely does not exist)
    #   5xx           → invalid (server error; URL may be broken or fabricated)
    # This is intentionally permissive for 401/403 so the agentic guardrail
    # does not incorrectly penalise responses that cite legitimate but
    # login-gated resources (e.g. government portals, legal databases).
    status = resp.status_code
    valid = status < 400 or status in (401, 403)

    return {
        "url": url,
        "valid": valid,
        "status_code": status,
        "final_url": resp.url,
        "redirect_count": len(resp.history),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool_call(name: str, arguments_json: str) -> str:
    """
    Route a tool call (name + JSON-encoded arguments string) to the correct
    Python function and return the result as a JSON string.

    Called by the agentic loop when the model requests a tool.
    On ToolError, returns a JSON string describing the failure so the loop
    can insert it into the conversation without crashing.
    """
    try:
        args: dict[str, Any] = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return json.dumps({"error": f"Could not parse arguments JSON: {arguments_json!r}"})

    try:
        if name == "search_web":
            query = args.get("query", "")
            results = search_web(query)
            return json.dumps(results, ensure_ascii=False)

        if name == "fetch_url":
            url = args.get("url", "")
            content = fetch_url(url)
            return json.dumps({"url": url, "content": content}, ensure_ascii=False)

        if name == "check_url_validity":
            url = args.get("url", "")
            result = check_url_validity(url)
            return json.dumps(result, ensure_ascii=False)

        return json.dumps({"error": f"Unknown tool: {name!r}"})

    except ToolError as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

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
]
