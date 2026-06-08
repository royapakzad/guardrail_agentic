"""Tests for the tool registry + tool groups (PR-8)."""

import json

import pytest
import tools
from tools import REGISTRY, TOOL_SCHEMAS, Tool, dispatch_tool_call, get_tool_schemas

_DEFAULT_TOOLS = {"search_web", "fetch_url", "check_url_validity", "check_acronym"}


def test_default_group_has_the_four_web_tools():
    schemas = get_tool_schemas("default")
    names = {s["function"]["name"] for s in schemas}
    assert names == _DEFAULT_TOOLS


def test_default_group_matches_legacy_tool_schemas():
    # Back-compat: the default group is byte-for-byte the original TOOL_SCHEMAS.
    assert get_tool_schemas("default") == TOOL_SCHEMAS


def test_unknown_group_raises():
    with pytest.raises(ValueError, match="Unknown tool group"):
        get_tool_schemas("does_not_exist")


def test_registry_get_returns_tool_with_handler():
    tool = REGISTRY.get("search_web")
    assert tool is not None
    assert tool.name == "search_web"
    assert callable(tool.handler)


def test_register_new_tool_appears_in_schemas_and_dispatches(monkeypatch):
    schema = {"type": "function", "function": {"name": "echo_tool", "description": "", "parameters": {}}}
    REGISTRY.register(Tool(name="echo_tool", schema=schema, handler=lambda a: {"echoed": a.get("msg")}))
    monkeypatch.setitem(tools.TOOL_GROUPS, "echo_group", ["echo_tool"])

    assert get_tool_schemas("echo_group") == [schema]
    out = json.loads(dispatch_tool_call("echo_tool", json.dumps({"msg": "hi"})))
    assert out == {"echoed": "hi"}


def test_dispatch_unknown_tool_via_registry():
    out = json.loads(dispatch_tool_call("nope", "{}"))
    assert "Unknown tool" in out["error"]
