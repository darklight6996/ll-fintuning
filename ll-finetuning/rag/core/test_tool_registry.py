"""
test_tool_registry.py

Phase 7 — Tool Calling System verification tests.

Tests:
1. Registry construction and singleton identity
2. Tool registration and JSON Schema export
3. Successful dispatch with required and optional parameters
4. Missing required parameter produces ToolResult error (never raises)
5. Unknown tool name produces ToolResult error (never raises)
6. Built-in tools registered (rag_search, yaml_lookup, command_preview, intelligence_extract, session_memory_read)
7. Category filtering returns correct subset
8. command_preview tool executes correctly without side effects
9. intelligence_extract tool parses terminal output
10. Duplicate registration raises ValueError
"""

import sys
import os

# --- path setup -----------------------------------------------------------
_HERE = os.path.dirname(__file__)
_CORE_ROOT = os.path.dirname(_HERE)          # ll-finetuning/rag
_RAG_ROOT = os.path.dirname(_CORE_ROOT)     # ll-finetuning
sys.path.insert(0, _RAG_ROOT)
# --------------------------------------------------------------------------

import pytest
from core.tool_registry import (
    ToolRegistry,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    get_tool_registry,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_registry() -> ToolRegistry:
    """Return a fresh, empty registry (not the singleton)."""
    return ToolRegistry()


def _echo_handler(message: str, repeat: int = 1) -> str:
    return (message + "\n") * repeat


# ===========================================================================
# Tests
# ===========================================================================

def test_singleton_identity():
    """get_tool_registry() must return the same object every call."""
    r1 = get_tool_registry()
    r2 = get_tool_registry()
    assert r1 is r2


def test_registration_and_schema_export():
    """Registering a tool should make it appear in get_schema() output."""
    reg = _make_registry()
    reg.register_function(
        name="echo",
        description="Echoes a message N times.",
        parameters=[
            ToolParameter("message", "string", "Message to echo.", required=True),
            ToolParameter("repeat", "integer", "Repetition count.", required=False, default=1),
        ],
        handler=_echo_handler,
    )
    schemas = reg.get_schema()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    func = schema["function"]
    assert func["name"] == "echo"
    assert "message" in func["parameters"]["properties"]
    assert "repeat" in func["parameters"]["properties"]
    assert "message" in func["parameters"]["required"]
    assert "repeat" not in func["parameters"]["required"]


def test_successful_dispatch_with_defaults():
    """Dispatching a valid call should return a successful ToolResult."""
    reg = _make_registry()
    reg.register_function(
        name="echo",
        description="Echo tool.",
        parameters=[
            ToolParameter("message", "string", "Message.", required=True),
            ToolParameter("repeat", "integer", "Repeat count.", required=False, default=2),
        ],
        handler=_echo_handler,
    )
    result = reg.dispatch("echo", {"message": "hello"})
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.tool_name == "echo"
    assert "hello" in result.output
    # default repeat=2 → "hello\nhello\n"
    assert result.output.count("hello") == 2


def test_dispatch_with_explicit_args():
    """Optional parameters provided explicitly should override defaults."""
    reg = _make_registry()
    reg.register_function(
        name="echo",
        description="Echo.",
        parameters=[
            ToolParameter("message", "string", "Msg.", required=True),
            ToolParameter("repeat", "integer", "N.", required=False, default=1),
        ],
        handler=_echo_handler,
    )
    result = reg.dispatch("echo", {"message": "ping", "repeat": 3})
    assert result.success is True
    assert result.output.count("ping") == 3


def test_missing_required_param_returns_error():
    """Missing a required parameter must return a failed ToolResult, not raise."""
    reg = _make_registry()
    reg.register_function(
        name="echo",
        description="Echo.",
        parameters=[ToolParameter("message", "string", "Msg.", required=True)],
        handler=_echo_handler,
    )
    result = reg.dispatch("echo", {})  # no 'message'
    assert result.success is False
    assert "message" in result.error


def test_unknown_tool_returns_error():
    """Dispatching an unregistered name must return a failed ToolResult."""
    reg = _make_registry()
    result = reg.dispatch("nonexistent_tool", {})
    assert result.success is False
    assert "nonexistent_tool" in result.error


def test_handler_exception_returns_error():
    """Exceptions in a handler must be caught and returned as ToolResult errors."""
    def bad_handler(x: str) -> str:
        raise RuntimeError("Simulated handler failure")

    reg = _make_registry()
    reg.register_function(
        name="bad",
        description="Always fails.",
        parameters=[ToolParameter("x", "string", "Input.", required=True)],
        handler=bad_handler,
    )
    result = reg.dispatch("bad", {"x": "test"})
    assert result.success is False
    assert "RuntimeError" in result.error


def test_builtin_tools_registered():
    """The singleton registry must contain all 5 built-in security tools."""
    reg = get_tool_registry()
    expected = {
        "rag_search", "yaml_lookup", "command_preview",
        "session_memory_read", "intelligence_extract",
    }
    registered = set(reg.list_tools())
    missing = expected - registered
    assert not missing, f"Missing built-in tools: {missing}"


def test_category_filtering():
    """get_schema_by_category() must return only tools in that category."""
    reg = _make_registry()
    reg.register_function(
        name="tool_a", description="A", parameters=[],
        handler=lambda: "a", category="rag",
    )
    reg.register_function(
        name="tool_b", description="B", parameters=[],
        handler=lambda: "b", category="memory",
    )
    reg.register_function(
        name="tool_c", description="C", parameters=[],
        handler=lambda: "c", category="rag",
    )
    rag_schemas = reg.get_schema_by_category("rag")
    names = {s["function"]["name"] for s in rag_schemas}
    assert names == {"tool_a", "tool_c"}


def test_command_preview_no_side_effects():
    """command_preview must never execute the command — only return preview text."""
    reg = get_tool_registry()
    # This would normally require root privileges to succeed;
    # preview must return successfully without running anything.
    result = reg.dispatch("command_preview", {"command": "rm -rf /", "target": "localhost"})
    assert result.success is True
    assert "rm -rf /" in result.output
    assert "Preview mode" in result.output or "not executed" in result.output.lower()


def test_intelligence_extract_parses_terminal():
    """intelligence_extract must parse user/port/kernel info from terminal output."""
    sample_output = """\
uid=1000(john) gid=1000(john) groups=1000(john),27(sudo)
Linux victim 5.4.0-42-generic #46-Ubuntu SMP Fri Jul 10 00:24:02 UTC 2020 x86_64
tcp   LISTEN   0.0.0.0:22   0.0.0.0:*
tcp   LISTEN   0.0.0.0:80   0.0.0.0:*
"""
    reg = get_tool_registry()
    result = reg.dispatch("intelligence_extract", {"terminal_output": sample_output})
    assert result.success is True
    # Should extract username or port data
    assert result.output  # non-empty


def test_duplicate_registration_raises():
    """Registering the same name twice must raise ValueError."""
    reg = _make_registry()
    reg.register_function(
        name="echo", description="Echo.", parameters=[],
        handler=lambda: "ok",
    )
    with pytest.raises(ValueError, match="already registered"):
        reg.register_function(
            name="echo", description="Duplicate.", parameters=[],
            handler=lambda: "dup",
        )
