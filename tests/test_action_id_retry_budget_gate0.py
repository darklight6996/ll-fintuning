"""
tests/test_action_id_retry_budget_gate0.py

Regression test for GATE 0 (action-id retry budget) in
core/command_executor.py::CommandExecutor.execute_action().

Found live (pentest agent, Qwen2.5-7B-Instruct + qwen-qlora smoke test,
2026-11-14): exploit_probe (max_retries=1) was re-proposed on steps 13-20
with varying invalid params. Because the old retry budget (Gate 6) keyed on
AIActionProposal.compute_identity() -- which hashes action_id + target +
params -- each varied-param re-proposal had a distinct identity and the
budget never fired; Gate 1 (validation) killed each one with a
`registry_validation` error instead. The planner already filtered the action
as exhausted by action_id, but the executor's hard gate disagreed. Gate 0
makes the executor count by action_id (matching the planner) and reject at/
over max_retries before validation.
"""

import sys
from pydantic import BaseModel, Field

from core.scope import Scope
from core.action_contract import ActionDefinition, AIActionProposal, ActionRegistry
from core.command_executor import CommandExecutor


class _ProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(...)


def _make_registry(max_retries):
    reg = ActionRegistry(agent_name="test")
    defn = ActionDefinition(
        action_id="probe",
        phase="recon",
        description="probe",
        params_model=_ProbeParams,
        allowed_target_kinds=["ip", "domain"],
        command_builder="build_probe",
        max_retries=max_retries,
    )
    reg.register(defn, lambda p: [[sys.executable, "-c", "print('ok')"]])
    return reg


def _scope():
    return Scope(target="10.0.0.5", in_scope=["10.0.0.5"],
                 active_testing_allowed=True, active_network_allowed=True)


def _proposal(**params):
    return AIActionProposal(
        decision="propose_action", action_id="probe",
        target=params.get("target", "10.0.0.5"), parameters=params,
    )


def test_first_attempt_is_allowed():
    ex = CommandExecutor(read_only=False)
    res = ex.execute_action(proposal=_proposal(target="10.0.0.5"),
                            registry=_make_registry(1), scope=_scope(),
                            execution_history=[], mode="automated")
    assert res["status"] == "success", res.get("rejection_reason")


def test_reproposal_over_budget_rejected_as_exhausted_not_validation():
    """The exact bug: a prior attempt exists; the re-proposal (even with
    different/invalid params) must be rejected as retry_budget_exhausted at
    Gate 0, NOT pass through to Gate 1 validation."""
    ex = CommandExecutor(read_only=False)
    history = [{"action_id": "probe", "status": "failed"}]  # one prior attempt
    # Re-propose with a bogus extra param that WOULD fail Gate 1 validation --
    # Gate 0 must catch it first so the reason is truthful.
    res = ex.execute_action(
        proposal=AIActionProposal(
            decision="propose_action", action_id="probe", target="10.0.0.5",
            parameters={"target": "10.0.0.5", "bogus": "x"},
        ),
        registry=_make_registry(1), scope=_scope(),
        execution_history=history, mode="automated",
    )
    assert res["status"] == "rejected"
    assert res["rejection_category"] == "retry_budget_exhausted"


def test_varying_params_do_not_evade_the_budget():
    """Each prior attempt used different params (distinct compute_identity).
    Gate 0 counts by action_id, so they still sum toward the budget."""
    ex = CommandExecutor(read_only=False)
    history = [
        {"action_id": "probe", "status": "rejected"},
        {"action_id": "probe", "status": "rejected"},
    ]
    res = ex.execute_action(proposal=_proposal(target="10.0.0.5"),
                            registry=_make_registry(2), scope=_scope(),
                            execution_history=history, mode="automated")
    assert res["status"] == "rejected"
    assert res["rejection_category"] == "retry_budget_exhausted"


def test_within_budget_still_executes():
    """max_retries=2 with only one prior attempt -- still within budget."""
    ex = CommandExecutor(read_only=False)
    history = [{"action_id": "probe", "status": "failed"}]
    res = ex.execute_action(proposal=_proposal(target="10.0.0.5"),
                            registry=_make_registry(2), scope=_scope(),
                            execution_history=history, mode="automated")
    assert res["status"] == "success", res.get("rejection_reason")


def test_other_actions_do_not_count_toward_this_actions_budget():
    ex = CommandExecutor(read_only=False)
    history = [
        {"action_id": "something_else", "status": "failed"},
        {"action_id": "another", "status": "failed"},
    ]
    res = ex.execute_action(proposal=_proposal(target="10.0.0.5"),
                            registry=_make_registry(1), scope=_scope(),
                            execution_history=history, mode="automated")
    assert res["status"] == "success", res.get("rejection_reason")
