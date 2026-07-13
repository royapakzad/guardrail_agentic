"""
policy_criteria.py
-------------------
Explicit, policy-author-driven tool-need classification for guardrail criteria.

Replaces the keyword-guessing classifier (agentic_runner.criterion_needs_tools),
which mis-sorted criteria based on coincidental substring matches (e.g. "claim"
inside "disclaimers") and had no way to express a new domain's criteria at all
without extending a shared keyword list. Instead, the policy author marks each
numbered criterion that needs tool access directly in the policy text file,
right in its header line:

    1. ACTIONABILITY AND PRACTICALITY (potentially needs tool calls)
    - The response must provide practical, effective, and directly helpful...

    3. SAFETY, SECURITY, AND PRIVACY
    - The response must not expose the user to physical, emotional, ...

A criterion with no marker is treated as not needing tools — there's no need
to also tag every "no tools" criterion; most criteria in a policy are non-tool,
so marking only the minority that need tools keeps the policy file readable.

Exports:
    has_explicit_tool_tags(policy_text) -> bool
    parse_tagged_criteria(policy_text)  -> list[dict]  (number, name, needs_tools, header, body)
    split_tagged_policy(policy_text)    -> (tool_policy_text, nontool_policy_text)
"""
from __future__ import annotations

import re

_HEADER_RE = re.compile(r'^(\d+)\.\s+(.+?)\s*$', re.MULTILINE)

# Recognizes a few natural phrasings a policy author might use to mark a
# criterion as needing tools. Case-insensitive; matched anywhere in the
# header text and stripped out to get the clean criterion name.
_TOOL_TAG_RE = re.compile(
    r'\(\s*(?:potentially\s+)?(?:need|needs|require|requires)\s+tool(?:s)?(?:\s+call(?:s)?)?\s*\)',
    re.IGNORECASE,
)


def has_explicit_tool_tags(policy_text: str) -> bool:
    """True if at least one numbered criterion is marked as needing tools."""
    for m in _HEADER_RE.finditer(policy_text):
        if _TOOL_TAG_RE.search(m.group(2)):
            return True
    return False


def parse_tagged_criteria(policy_text: str) -> list[dict]:
    """
    Parse every numbered criterion into a structured record.

    Returns a list of dicts, each:
        {"number": int, "name": str, "needs_tools": bool, "header": str, "body": str}
    "body" is the full text from this criterion's header up to (not including)
    the next numbered header, with the tool-tag stripped from the header line
    so it isn't echoed back to the judge LLM.

    Raises ValueError if the policy text has no numbered criteria at all.
    """
    headers = list(_HEADER_RE.finditer(policy_text))
    if not headers:
        raise ValueError("No numbered criteria (e.g. '1. CRITERION NAME') found in policy text.")

    criteria: list[dict] = []
    for i, m in enumerate(headers):
        number = int(m.group(1))
        raw_name = m.group(2).strip()
        needs_tools = bool(_TOOL_TAG_RE.search(raw_name))
        name = _TOOL_TAG_RE.sub("", raw_name).strip()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(policy_text)
        header_line = f"{number}. {name}"
        body = header_line + policy_text[m.end():end].rstrip() + "\n"
        criteria.append(
            {
                "number": number,
                "name": name,
                "needs_tools": needs_tools,
                "header": header_line,
                "body": body,
            }
        )
    return criteria


def split_tagged_policy(policy_text: str) -> tuple[str, str]:
    """
    Split a policy into two standalone policy texts: one containing only
    tool-requiring criteria, one containing only non-tool criteria.

    Preamble text before the first numbered criterion (e.g. a "POLICY" title
    line) is preserved in both halves. Either half may be an empty string if
    no criteria fall on that side — callers should treat an empty half as
    "skip this evaluation, nothing to judge."
    """
    criteria = parse_tagged_criteria(policy_text)
    first_header_pos = _HEADER_RE.search(policy_text).start()
    preamble = policy_text[:first_header_pos].rstrip()

    tool_bodies = [c["body"] for c in criteria if c["needs_tools"]]
    nontool_bodies = [c["body"] for c in criteria if not c["needs_tools"]]

    def _assemble(bodies: list[str]) -> str:
        if not bodies:
            return ""
        parts = [preamble] if preamble else []
        parts.extend(bodies)
        return "\n\n".join(parts).strip() + "\n"

    return _assemble(tool_bodies), _assemble(nontool_bodies)
