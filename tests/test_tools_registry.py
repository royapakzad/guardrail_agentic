"""Tests for the tool registry introduced in PR #15."""
import pytest

from tools import REGISTRY, TOOL_GROUPS, get_tool_schemas


EXPECTED_DEFAULT_TOOLS = {"search_web", "fetch_url", "check_url_validity", "check_acronym"}


def test_registry_contains_all_default_tools():
    assert EXPECTED_DEFAULT_TOOLS.issubset(REGISTRY.keys())


def test_default_group_in_tool_groups():
    assert "default" in TOOL_GROUPS


def test_get_tool_schemas_default_returns_four_schemas():
    schemas = get_tool_schemas("default")
    assert len(schemas) == 4
    names = {s["function"]["name"] for s in schemas}
    assert names == EXPECTED_DEFAULT_TOOLS


def test_get_tool_schemas_unknown_group_returns_empty():
    schemas = get_tool_schemas("nonexistent_group_xyz")
    assert schemas == []


def test_each_schema_has_required_fields():
    schemas = get_tool_schemas("default")
    for schema in schemas:
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]


def test_registry_tools_have_callable_handlers():
    for name, tool in REGISTRY.items():
        assert callable(tool.handler), f"handler for {name!r} is not callable"
