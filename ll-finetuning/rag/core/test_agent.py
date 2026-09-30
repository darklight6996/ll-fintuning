"""
test_agent.py

Phase 8 — Local Agent Framework verification tests.

Tests:
1.  AgentRequest / AgentResponse dataclass construction
2.  CyberAgent instantiation has zero heavy side-effects
3.  Singleton identity: get_agent() returns same object per session_id
4.  Different session_ids return independent agents
5.  Fallback engine produces a valid AgentResponse
6.  Intent classification embedded in AgentResponse
7.  Reset intent is detected and gated (no confirm → no wipe)
8.  Reset intent with 'reset confirm' wipes state
9.  Tool call dispatch through agent.tool_call()
10. agent.run() returns AgentResponse with latency_ms > 0
11. force_fallback=True skips LLM regardless of availability
12. History retrieval via agent.get_history()
"""

import sys
import os

# --- path setup -----------------------------------------------------------
_HERE = os.path.dirname(__file__)
_CORE_ROOT = os.path.dirname(_HERE)
_RAG_ROOT = os.path.dirname(_CORE_ROOT)
sys.path.insert(0, _RAG_ROOT)
# --------------------------------------------------------------------------

import pytest
from core.agent import AgentRequest, AgentResponse, CyberAgent, get_agent


# ===========================================================================
# Helpers
# ===========================================================================

def _fresh_agent(session_id: str = "test-session") -> CyberAgent:
    """Return a fresh agent (not the singleton)."""
    return CyberAgent(session_id=session_id, verbose=False)


def _basic_state() -> dict:
    return {
        "target": "10.10.10.1",
        "type": "ctf",
        "notes": [],
        "evidence": [],
        "completed_steps": [],
        "chat_history": [],
    }


# ===========================================================================
# Tests
# ===========================================================================

def test_agent_request_construction():
    """AgentRequest should construct with defaults."""
    req = AgentRequest(query="hello")
    assert req.query == "hello"
    assert req.session_id == "default_session"
    assert req.force_fallback is False
    assert isinstance(req.state, dict)


def test_agent_response_construction():
    """AgentResponse must expose is_llm and is_fallback properties."""
    resp = AgentResponse(text="ok", intent="general", source="llm", latency_ms=42.0)
    assert resp.is_llm is True
    assert resp.is_fallback is False

    resp2 = AgentResponse(text="ok", intent="general", source="fallback")
    assert resp2.is_llm is False
    assert resp2.is_fallback is True


def test_agent_instantiation_no_side_effects():
    """CyberAgent.__init__ must not load any models or open DB connections."""
    agent = CyberAgent(session_id="test-init", verbose=False)
    # All lazy properties must be uninitialized at construction time
    assert agent._router is None
    assert agent._llm is None
    assert agent._memory is None
    assert agent._extractor is None
    assert agent._prompt_builder is None
    assert agent._tools is None


def test_singleton_identity():
    """get_agent() must return the same object per session_id."""
    a1 = get_agent("singleton-test")
    a2 = get_agent("singleton-test")
    assert a1 is a2


def test_different_sessions_independent():
    """Different session_ids must produce independent agents."""
    a1 = get_agent("session-alpha")
    a2 = get_agent("session-beta")
    assert a1 is not a2
    assert a1.session_id != a2.session_id


def test_fallback_produces_valid_response():
    """
    With force_fallback=True, agent.run() must return a valid AgentResponse
    without any model loading.
    """
    agent = _fresh_agent("fallback-test")
    req = AgentRequest(
        query="where do I start?",
        state=_basic_state(),
        session_id="fallback-test",
        force_fallback=True,
    )
    resp = agent.run(req)
    assert isinstance(resp, AgentResponse)
    assert resp.text
    assert resp.source == "fallback"
    assert resp.latency_ms >= 0.0


def test_intent_in_response():
    """Intent value should be embedded in AgentResponse."""
    agent = _fresh_agent("intent-test")
    req = AgentRequest(
        query="where do I start on this Ubuntu machine?",
        state=_basic_state(),
        force_fallback=True,
    )
    resp = agent.run(req)
    assert resp.intent in {
        "reset", "recon", "steering", "explain_output",
        "ctf_privesc", "ad_network", "web_api", "general",
    }


def test_recon_intent_classified():
    """'where do I start' must be classified as RECON intent."""
    agent = _fresh_agent("recon-intent-test")
    req = AgentRequest(
        query="where do I start?",
        state={},
        force_fallback=True,
    )
    resp = agent.run(req)
    assert resp.intent == "recon"


def test_reset_without_confirm_does_not_wipe():
    """
    Saying 'reset' without 'reset confirm' should return a clarification
    message — not wipe state.
    """
    agent = _fresh_agent("reset-no-confirm")
    state = _basic_state()
    state["notes"] = [{"content": "SUID binary found: /usr/bin/python3"}]
    req = AgentRequest(
        query="reset",
        state=state,
        force_fallback=True,
    )
    resp = agent.run(req)
    # State should NOT be wiped (notes still present in the dict reference)
    assert resp.intent == "reset"
    # The notes key in the passed state dict must not have been cleared
    assert len(state.get("notes", [])) > 0, "State was wiped without confirmation"


def test_reset_with_confirm_wipes_state():
    """
    'reset confirm' should wipe session state artifacts.
    """
    agent = _fresh_agent("reset-with-confirm")
    state = _basic_state()
    state["artifacts"] = ["evidence1", "evidence2"]
    req = AgentRequest(
        query="reset confirm",
        state=state,
        force_fallback=True,
    )
    resp = agent.run(req)
    assert resp.intent == "reset"
    # The IntentRouter.execute_reset clears state["artifacts"]
    assert state.get("artifacts") == []


def test_tool_call_through_agent():
    """agent.tool_call() must dispatch to the ToolRegistry correctly."""
    agent = _fresh_agent("tool-call-test")
    result = agent.tool_call("command_preview", {"command": "id"})
    assert result.success is True
    assert "id" in result.output


def test_agent_run_returns_latency():
    """AgentResponse.latency_ms must be positive."""
    agent = _fresh_agent("latency-test")
    req = AgentRequest(
        query="explain the output",
        state=_basic_state(),
        force_fallback=True,
    )
    resp = agent.run(req)
    assert resp.latency_ms > 0


def test_force_fallback_skips_llm():
    """force_fallback=True must always return source='fallback', even if LLM is loaded."""
    agent = _fresh_agent("force-fallback-test")
    req = AgentRequest(
        query="what is a SUID binary?",
        state={},
        force_fallback=True,
    )
    resp = agent.run(req)
    assert resp.source == "fallback"


def test_history_retrieval():
    """After a run, get_history() must return at least the last turn."""
    agent = _fresh_agent("history-test-" + str(os.getpid()))
    req = AgentRequest(
        query="show me the enumeration commands",
        state={},
        force_fallback=True,
    )
    agent.run(req)
    history = agent.get_history(limit=10)
    # Should have at least the turn we just added
    assert len(history) >= 1
