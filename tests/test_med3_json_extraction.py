"""
tests/test_med3_json_extraction.py

Regression tests for MED-3: AIPlanner._extract_json_object must correctly handle
deeply-nested JSON, markdown code fences, and leading/trailing commentary -- all
cases the old single-level-nesting regex silently mishandled.
"""

from core.scope import Scope
from core.action_contract import ActionRegistry
from core.ai_planner import AIPlanner


def _make_planner():
    scope = Scope(target="local", active_testing_allowed=True)
    registry = ActionRegistry(agent_name="test")
    return AIPlanner(registry=registry, chat_assistant=None, scope=scope)


def test_extracts_deeply_nested_parameters():
    """The old regex `\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}` only tolerates ONE level
    of nesting inside the outer object. A nested dict inside 'parameters' (two levels
    deep) previously broke extraction."""
    planner = _make_planner()
    raw = '''
{
  "decision": "record_finding",
  "action_id": null,
  "parameters": {
    "hypothesis_id": "hyp_1",
    "title": "x",
    "nested": {"a": {"b": {"c": 1}}}
  },
  "rationale": "deep nesting test",
  "confidence": 0.9
}
'''
    data = planner._extract_json_object(raw)
    assert data["decision"] == "record_finding"
    assert data["parameters"]["nested"]["a"]["b"]["c"] == 1


def test_extracts_from_markdown_fence_with_nesting():
    planner = _make_planner()
    raw = '''Sure, here is my decision:
```json
{
  "decision": "propose_action",
  "action_id": "port_scan",
  "parameters": {"ports": {"range": "1-1000"}},
  "rationale": "scan"
}
```
Let me know if you need anything else.'''
    data = planner._extract_json_object(raw)
    assert data["decision"] == "propose_action"
    assert data["parameters"]["ports"]["range"] == "1-1000"


def test_extracts_first_object_ignoring_trailing_commentary():
    planner = _make_planner()
    raw = '{"decision": "propose_completion"} -- that is my final answer.'
    data = planner._extract_json_object(raw)
    assert data["decision"] == "propose_completion"


def test_raises_value_error_when_no_json_present():
    planner = _make_planner()
    import pytest
    with pytest.raises(ValueError):
        planner._extract_json_object("I don't think I can decide right now.")
