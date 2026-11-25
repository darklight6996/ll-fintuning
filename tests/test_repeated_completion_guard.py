"""
tests/test_repeated_completion_guard.py

Regression test for the repeated-completion guard in
agents/base_agent.py::BaseAgent.run_autonomous().

Found live (pentest agent, Qwen2.5-7B-Instruct + qwen-qlora smoke test,
2026-11-14): after every registered action was exhausted, the planner
correctly switched to propose_completion, but the required "exploitation"
phase was unsatisfiable on a bare loopback (no exploitable service), so
verify_completion() rejected completion every time. The model re-proposed
completion identically for 8 steps straight until max_steps. The guard ends
the run promptly with status="terminated_incomplete" after
MAX_CONSECUTIVE_COMPLETION_REJECTIONS back-to-back rejected completions,
surfacing the real blocking reason -- without ever letting the agent falsely
claim success.

The run_autonomous() loop builds its own AIPlanner (and a ChatAssistant that
would load the local model) internally, so this test patches those two seams:
core.ai_planner.AIPlanner -> an always-propose-completion stub, and
agents.base_agent.ChatAssistant -> a dummy (no model load). The completion
verifier is forced to always reject. Together these reproduce the live
unsatisfiable-phase state without a GPU or model.
"""

import types

import pytest

from core.scope import Scope
from core.action_contract import AIActionProposal, ActionRegistry

REJECT_REASON = "Required phase 'exploitation' has not been completed."


class _AlwaysCompletePlanner:
    """Stub AIPlanner: constructed with the same kwargs run_autonomous uses,
    and every decision is propose_completion."""
    def __init__(self, *args, **kwargs):
        pass

    def decide_next(self, **kwargs):
        return AIActionProposal(
            decision="propose_completion",
            rationale="every registered action is exhausted; declaring complete",
            confidence=0.6,
        )


class _DummyChatAssistant:
    """Stub ChatAssistant: no model load; only the members run_autonomous uses."""
    def __init__(self, *args, **kwargs):
        pass

    def synthesize_report(self, state, context=None):
        return "stub report"


def _always_reject(criteria, session_state, evidence_store):
    return False, REJECT_REASON


def test_run_terminates_via_guard_not_max_steps(monkeypatch):
    import agents.base_agent as base_agent_mod
    import core.ai_planner as ai_planner_mod
    from agents.pentest.agent import PentestAgent
    from core.session_manager import SessionManager

    # Seam 1: the planner the loop builds internally -> always propose completion.
    monkeypatch.setattr(ai_planner_mod, "AIPlanner", _AlwaysCompletePlanner)
    # Seam 2: the ChatAssistant the loop builds for the planner + report -> no model.
    monkeypatch.setattr(base_agent_mod, "ChatAssistant", _DummyChatAssistant)

    scope = Scope(target="127.0.0.1", in_scope=["127.0.0.1"],
                  active_testing_allowed=True, active_network_allowed=True)
    agent = PentestAgent(scope=scope, mode="automated", session_manager=SessionManager())
    # Force completion to always be rejected (simulates an unsatisfiable phase).
    agent._get_completion_verifier = lambda: _always_reject
    # Also stub the instance's own chat_assistant (built in __init__ before our patch).
    agent.chat_assistant = _DummyChatAssistant()

    result = agent.run_autonomous(
        registry=ActionRegistry(agent_name="pentest"),  # empty: no actions to offer
        completion_criteria=object(),  # truthy; content irrelevant (verifier stubbed)
        max_steps=50,
    )

    assert result["status"] == "terminated_incomplete", result
    # Guard threshold is 3, so it must stop well before max_steps=50.
    assert result["steps_run"] <= 5, result
    assert result.get("completion_block_reason") == REJECT_REASON, result


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
