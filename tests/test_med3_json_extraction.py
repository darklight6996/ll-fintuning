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


def test_strips_trailing_parenthetical_annotation_after_action_id():
    """
    Regression for a live GPU smoke-test failure (Qwen2.5-7B-Instruct + qwen-qlora,
    2026-11-11): the model echoed the OUTPUT FORMAT example's old placeholder wording
    as a literal trailing annotation -- `"action_id": "environment_recon" (from
    catalog),` -- which is invalid JSON. Without the repair, raw_decode correctly
    refuses the whole (malformed) outer object, the extraction loop moves on to the
    next '{' it can find (the empty "parameters": {}), parses THAT successfully, and
    silently returns a decision-less dict -- exactly what was observed:
    "Missing 'decision' field in AI proposal." This must now recover the real object.
    """
    planner = _make_planner()
    raw = '''{
  "decision": "propose_action",
  "action_id": "environment_recon" (from catalog),
  "target": null,
  "parameters": {},
  "rationale": "Gather initial system information to build a baseline understanding of the target.",
  "confidence": 0.9
}'''
    data = planner._extract_json_object(raw)
    assert data["decision"] == "propose_action"
    assert data["action_id"] == "environment_recon"
    assert data["target"] is None
    assert data["confidence"] == 0.9


def test_does_not_mangle_legitimate_parenthetical_text_inside_a_string_value():
    """The repair regex anchors on a `"` immediately followed by whitespace, '(',
    content, ')', then a JSON delimiter -- a shape that can't legitimately occur
    inside a string's own text (that `"` would already be the string's closing
    quote). A parenthetical remark that is genuinely part of the rationale text
    (not immediately after the field's closing quote) must survive untouched."""
    planner = _make_planner()
    raw = '''{"decision": "propose_completion", "action_id": null, "parameters": {},
  "rationale": "Scope requires manual review (see NIST SP 800-115 section 4).",
  "confidence": 1.0}'''
    data = planner._extract_json_object(raw)
    assert data["decision"] == "propose_completion"
    assert data["rationale"] == "Scope requires manual review (see NIST SP 800-115 section 4)."


def test_prefers_decision_bearing_object_over_earlier_trivial_nested_object():
    """Defensive robustness fix applied alongside the repair above: even when no
    repair is needed, the extraction loop must not settle for the first dict that
    merely parses successfully if a real, decision-bearing object exists later in
    the text."""
    planner = _make_planner()
    raw = 'Example of an empty call: {} -- but my actual answer is: {"decision": "propose_completion", "confidence": 1.0}'
    data = planner._extract_json_object(raw)
    assert data["decision"] == "propose_completion"


def test_still_raises_when_nothing_decision_bearing_and_nothing_parses():
    planner = _make_planner()
    import pytest
    with pytest.raises(ValueError):
        planner._extract_json_object("no braces here at all, sorry.")


def test_strips_trailing_line_comment_after_action_id():
    """
    Regression for a second live GPU smoke-test failure, same session as the
    parenthetical-annotation one above: the model's next attempt at the same
    annotation habit used a JS/JSON5-style `//` line comment instead --
    `"action_id": "environment_recon" // from catalog` -- equally invalid in
    standard JSON, and equally recoverable with the same closing-quote anchor.
    """
    planner = _make_planner()
    raw = '''{
  "decision": "propose_action",
  "action_id": "environment_recon" // from catalog
} // no parameters needed for this action type. Rationale is optional. Confidence is optional but recommended.'''
    data = planner._extract_json_object(raw)
    assert data["decision"] == "propose_action"
    assert data["action_id"] == "environment_recon"


def test_does_not_mangle_a_url_in_a_string_value():
    """`//` appears constantly in legitimate text (URLs). The repair must only
    strip it when it's directly after a value's closing quote, never mid-string."""
    planner = _make_planner()
    raw = '{"decision": "propose_completion", "rationale": "see http://example.com/docs for refs", "confidence": 1.0}'
    data = planner._extract_json_object(raw)
    assert data["rationale"] == "see http://example.com/docs for refs"


def test_strips_trailing_bracket_annotation_after_action_id():
    """Third variant of the same annotation habit, found live on 2026-11-11 after
    REGISTERED ACTIONS started showing each action's [risk, network, state] tag:
    the model copied that bracketed tag directly after action_id's closing quote,
    e.g. "action_id": "environment_recon" [low risk, network=none, state=none],
    -- invalid JSON, same as the earlier parenthetical and // comment variants."""
    planner = _make_planner()
    raw = '''{
  "decision": "propose_action",
  "action_id": "environment_recon" [low risk, network=none, state=none],
  "target": null,
  "parameters": {},
  "rationale": "environment_recon is the first REGISTERED ACTION still within its retry budget.",
  "confidence": 0.85
}'''
    data = planner._extract_json_object(raw)
    assert data["decision"] == "propose_action"
    assert data["action_id"] == "environment_recon"


def test_does_not_mangle_a_legitimate_array_or_bracketed_text_in_a_string_value():
    """The bracket-annotation repair must not touch a real JSON array value or a
    string that happens to contain square brackets as ordinary content."""
    planner = _make_planner()
    raw = '''{
  "decision": "record_finding",
  "action_id": null,
  "parameters": {
    "reproducible_steps": ["step1", "step2"],
    "note": "see [RFC 1234] section 2"
  },
  "rationale": "ok",
  "confidence": 0.9
}'''
    data = planner._extract_json_object(raw)
    assert data["parameters"]["reproducible_steps"] == ["step1", "step2"]
    assert data["parameters"]["note"] == "see [RFC 1234] section 2"
