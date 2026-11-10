"""
tests/test_med1_med2.py

Regression tests for:
  MED-1 -- consecutive planner-failure cap (abort early instead of burning max_steps)
  MED-2 -- stall / no-progress detector (added alongside, see base_agent.run_autonomous)

Both use a fake AIPlanner-shaped object so the real ChatAssistant/model is never touched.
"""

import pytest

from core.scope import Scope
from core.action_contract import ActionRegistry
from agents.base_agent import BaseAgent


class _MinimalAgent(BaseAgent):
    agent_type = "ctf"

    def _load_phases(self):
        self.phase_modules = []


class _AlwaysBrokenPlanner:
    """Every decide_next() call raises -- simulates a planner that can never produce
    parseable JSON (e.g. a severely under-sized local model)."""

    def __init__(self, *_a, **_kw):
        pass

    def decide_next(self, **_kwargs):
        raise ValueError("could not parse JSON")


def test_consecutive_planner_failures_abort_before_max_steps(monkeypatch):
    # run_autonomous() does `from core.ai_planner import AIPlanner` as a LOCAL import
    # inside the function body, re-resolved from core.ai_planner on every call -- so the
    # class must be patched at its source module, not on agents.base_agent.
    import core.ai_planner as ai_planner_module

    scope = Scope(target="local", active_testing_allowed=True)
    agent = _MinimalAgent(scope=scope, mode="automated")
    registry = ActionRegistry(agent_name="test")

    monkeypatch.setattr(ai_planner_module, "AIPlanner", _AlwaysBrokenPlanner)

    result = agent.run_autonomous(registry=registry, completion_criteria=None, max_steps=50)

    assert result["status"] == "aborted_planner_failures"
    # Aborted well before exhausting the 50-step budget -- locks in that this is an
    # early-exit path, not merely a relabeling of max_steps_reached.
    assert result["steps_run"] <= 5 + 1


class _AlwaysSameRejectedActionPlanner:
    """Always proposes the same unregistered action_id, which Gate 1
    (registry validation) rejects every time -- never produces an Observation,
    so the MED-2 stall detector should trip."""

    def __init__(self, *_a, **_kw):
        pass

    def decide_next(self, **_kwargs):
        from core.action_contract import AIActionProposal
        return AIActionProposal(
            decision="propose_action",
            action_id="nonexistent_action",
            target="local",
            parameters={},
            rationale="repeat the same thing forever",
        )


def test_stall_detector_aborts_after_no_progress_window(monkeypatch):
    import core.ai_planner as ai_planner_module

    scope = Scope(target="local", active_testing_allowed=True)
    agent = _MinimalAgent(scope=scope, mode="automated")
    registry = ActionRegistry(agent_name="test")  # intentionally has no actions registered

    monkeypatch.setattr(ai_planner_module, "AIPlanner", _AlwaysSameRejectedActionPlanner)

    result = agent.run_autonomous(registry=registry, completion_criteria=None, max_steps=50)

    assert result["status"] == "stalled_no_progress"
    assert result["steps_run"] == 10  # MAX_STEPS_WITHOUT_PROGRESS
    assert result["observations_count"] == 0
