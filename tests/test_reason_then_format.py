"""
tests/test_reason_then_format.py

Tests for the REASON-THEN-FORMAT planner path (core/ai_planner.py,
2026-11-25, PLANNER_REASON_THEN_FORMAT=1): an extra free-prose LLM call
before the strict JSON call, testing whether the model reasons better
unconstrained than when forced into "return exactly one JSON object" from
the first token. Off by default -- decide_next() must stay byte-identical
to the single-call path unless the operator explicitly opts in.
"""

import json

import pytest

from core.ai_planner import (
    AIPlanner,
    reason_then_format_enabled,
    _FINAL_DECISION_INSTRUCTION,
    _REASONING_ONLY_INSTRUCTION,
    _REASONING_SYSTEM_PROMPT,
)
from core.scope import Scope
from core.investigation import EvidenceStore
from agents.bug_bounty.actions import build_bug_bounty_registry
from agents.bug_bounty.completion import BUG_BOUNTY_COMPLETION


def _planner():
    return AIPlanner(
        registry=build_bug_bounty_registry(),
        chat_assistant=None,
        scope=Scope(target="https://x.web-security-academy.net/", in_scope=["x.web-security-academy.net"],
                    active_testing_allowed=True, active_network_allowed=True),
    )


class _RecordingChatAssistant:
    """Records every generate_raw() call (prompt, system_prompt) and returns
    a scripted response per call, in order."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_raw(self, prompt, max_new_tokens=512, system_prompt=None, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt,
                            "max_new_tokens": max_new_tokens})
        return self._responses.pop(0)


_VALID_JSON = json.dumps({
    "decision": "propose_action", "action_id": "recon_scan", "target": None,
    "parameters": {}, "rationale": "recon first", "confidence": 0.9,
})


# --- toggle -------------------------------------------------------------------

def test_reason_then_format_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PLANNER_REASON_THEN_FORMAT", raising=False)
    assert reason_then_format_enabled() is False


def test_reason_then_format_requires_exact_string_1(monkeypatch):
    monkeypatch.setenv("PLANNER_REASON_THEN_FORMAT", "true")
    assert reason_then_format_enabled() is False
    monkeypatch.setenv("PLANNER_REASON_THEN_FORMAT", "1")
    assert reason_then_format_enabled() is True


# --- disabled path: byte-identical, single call --------------------------------

def test_disabled_makes_exactly_one_generate_raw_call(monkeypatch):
    monkeypatch.delenv("PLANNER_REASON_THEN_FORMAT", raising=False)
    planner = _planner()
    chat = _RecordingChatAssistant([_VALID_JSON])
    planner.chat_assistant = chat

    proposal = planner.decide_next(
        session_state={}, evidence_store=EvidenceStore(), completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={},
    )
    assert proposal.action_id == "recon_scan"
    assert len(chat.calls) == 1
    assert _FINAL_DECISION_INSTRUCTION in chat.calls[0]["prompt"]


# --- enabled path: two calls, reasoning then format -----------------------------

def test_enabled_makes_two_calls_reasoning_then_strict(monkeypatch):
    monkeypatch.setenv("PLANNER_REASON_THEN_FORMAT", "1")
    planner = _planner()
    reasoning_text = "I observe a numeric productId parameter; I will propose self_script to test IDOR."
    chat = _RecordingChatAssistant([reasoning_text, _VALID_JSON])
    planner.chat_assistant = chat

    proposal = planner.decide_next(
        session_state={}, evidence_store=EvidenceStore(), completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={},
    )
    assert proposal.action_id == "recon_scan"
    assert len(chat.calls) == 2

    call1, call2 = chat.calls
    # Call 1: free-prose system prompt, reasoning-only instruction, NOT the strict one.
    assert call1["system_prompt"] == _REASONING_SYSTEM_PROMPT
    assert _REASONING_ONLY_INSTRUCTION in call1["prompt"]
    assert _FINAL_DECISION_INSTRUCTION not in call1["prompt"]

    # Call 2: strict system prompt (not the reasoning one), strict instruction
    # present, AND the Call 1 reasoning text threaded through as context.
    assert call2["system_prompt"] != _REASONING_SYSTEM_PROMPT
    assert _FINAL_DECISION_INSTRUCTION in call2["prompt"]
    assert reasoning_text in call2["prompt"]
    assert "YOUR OWN REASONING" in call2["prompt"]


def test_enabled_reasoning_call_failure_falls_back_to_single_call_prompt(monkeypatch, capsys):
    monkeypatch.setenv("PLANNER_REASON_THEN_FORMAT", "1")
    planner = _planner()

    class _FailFirstCallChatAssistant:
        def __init__(self):
            self.calls = 0

        def generate_raw(self, prompt, max_new_tokens=512, system_prompt=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Ollama unreachable")
            return _VALID_JSON

    chat = _FailFirstCallChatAssistant()
    planner.chat_assistant = chat

    proposal = planner.decide_next(
        session_state={}, evidence_store=EvidenceStore(), completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={},
    )
    assert proposal.action_id == "recon_scan"
    assert chat.calls == 2  # the failed reasoning attempt + the successful strict call
    assert "REASON-THEN-FORMAT Call 1 failed" in capsys.readouterr().out


def test_enabled_empty_reasoning_response_still_proceeds_with_strict_call(monkeypatch):
    monkeypatch.setenv("PLANNER_REASON_THEN_FORMAT", "1")
    planner = _planner()
    chat = _RecordingChatAssistant(["   ", _VALID_JSON])  # blank reasoning response
    planner.chat_assistant = chat

    proposal = planner.decide_next(
        session_state={}, evidence_store=EvidenceStore(), completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={},
    )
    assert proposal.action_id == "recon_scan"
    assert len(chat.calls) == 2
    # No reasoning text to thread through -- Call 2's prompt is the plain strict one.
    assert "YOUR OWN REASONING" not in chat.calls[1]["prompt"]


# --- _build_prompt's prior_reasoning parameter is a pure no-op when empty -------

def test_build_prompt_default_prior_reasoning_is_byte_identical():
    planner = _planner()
    kwargs = dict(session_state={}, evidence_store=EvidenceStore(), completion_criteria=BUG_BOUNTY_COMPLETION,
                  execution_history=[], retry_budget_status={})
    p1 = planner._build_prompt(**kwargs)
    p2 = planner._build_prompt(**kwargs, prior_reasoning="")
    assert p1 == p2


def test_build_prompt_prior_reasoning_appears_before_final_instruction():
    planner = _planner()
    prompt = planner._build_prompt(
        session_state={}, evidence_store=EvidenceStore(), completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={}, prior_reasoning="MARKER_REASONING_TEXT",
    )
    assert prompt.index("MARKER_REASONING_TEXT") < prompt.index(_FINAL_DECISION_INSTRUCTION)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
