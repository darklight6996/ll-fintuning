"""
tests/test_remediation_c1_c2_h1.py

Regression tests for the autonomous-loop remediation pass:
  C1 — deterministic evidence-chain producer (findings were previously unreachable)
  C2 — execution_history must actually reach Gate 6 via the keyword base_agent uses
  H1 — per-agent completion dispatch (no longer hardcoded to CTF's verify_completion)

These lock in the specific defects identified in AUDIT.md so a future refactor can't
silently reintroduce them.
"""

import sys
import pytest
from pydantic import BaseModel

from core.scope import Scope
from core.action_contract import ActionDefinition, AIActionProposal, ActionRegistry
from core.investigation import EvidenceStore, Observation
from core.command_executor import CommandExecutor
from agents.base_agent import BaseAgent

import agents.ctf.completion as ctf_completion
import agents.pentest.completion as pentest_completion


class _DummyParams(BaseModel):
    model_config = {"extra": "forbid"}


class _MinimalAgent(BaseAgent):
    """A BaseAgent subclass with no real phases, for exercising shared logic in isolation."""
    agent_type = "ctf"  # overridden per-instance in tests that need a different agent_type

    def _load_phases(self):
        self.phase_modules = []


def _make_agent(agent_type: str, scope: Scope) -> _MinimalAgent:
    agent = _MinimalAgent(scope=scope, mode="automated")
    agent.agent_type = agent_type
    return agent


# ---------------------------------------------------------------------------
# C2: Gate 6 (retry budget) must trip via base_agent's real calling convention
# ---------------------------------------------------------------------------
def test_retry_budget_trips_via_execution_history_kwarg():
    """
    base_agent.run_autonomous calls:
        executor.execute_action(proposal=..., registry=..., scope=...,
                                 execution_history=execution_history, mode=self.mode)
    Before the C2 fix, base_agent passed session_state=... instead, which landed in
    **kwargs and was never read — execution_history was always [] and Gate 6 never
    tripped. This test exercises the exact keyword form base_agent now uses.
    """
    scope = Scope(target="local", active_testing_allowed=True, state_changes_allowed=True)
    executor = CommandExecutor(read_only=False)
    registry = ActionRegistry(agent_name="test")

    action = ActionDefinition(
        action_id="retry_probe",
        phase="recon",
        description="Retry-budget probe",
        params_model=_DummyParams,
        allowed_target_kinds=["local"],
        command_builder="build_retry_probe",
        max_retries=2,
    )
    registry.register(action, lambda p: [["echo", "probe"]])

    proposal = AIActionProposal(
        decision="propose_action",
        action_id="retry_probe",
        target="local",
        parameters={},
        rationale="probe",
    )
    action_identity = proposal.compute_identity()

    execution_history = [
        {"action_identity": action_identity, "action_id": "retry_probe", "status": "success"},
        {"action_identity": action_identity, "action_id": "retry_probe", "status": "success"},
    ]

    result = executor.execute_action(
        proposal=proposal,
        registry=registry,
        scope=scope,
        execution_history=execution_history,
        mode="automated",
    )
    assert result["status"] == "rejected"
    assert result["rejection_category"] == "retry_budget_exhausted"


def test_retry_budget_does_not_trip_on_empty_history():
    """Sanity counterpart: with no prior attempts, the same action must be allowed to run."""
    scope = Scope(target="local", active_testing_allowed=True, state_changes_allowed=True)
    executor = CommandExecutor(read_only=False)
    registry = ActionRegistry(agent_name="test")

    action = ActionDefinition(
        action_id="retry_probe2",
        phase="recon",
        description="Retry-budget probe 2",
        params_model=_DummyParams,
        allowed_target_kinds=["local"],
        command_builder="build_retry_probe2",
        max_retries=2,
    )
    # Use sys.executable rather than "echo" — echo is a shell builtin on Windows
    # (no standalone echo.exe), so subprocess.run(["echo", ...], shell=False) fails
    # with FileNotFoundError there. python -c is a real executable on every platform.
    registry.register(action, lambda p: [[sys.executable, "-c", "print('probe')"]])

    proposal = AIActionProposal(
        decision="propose_action",
        action_id="retry_probe2",
        target="local",
        parameters={},
        rationale="probe",
    )

    result = executor.execute_action(
        proposal=proposal,
        registry=registry,
        scope=scope,
        execution_history=[],
        mode="automated",
    )
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# C1: deterministic evidence-chain producer
# ---------------------------------------------------------------------------
def test_evidence_promotion_creates_finding_for_satisfies_phase_action():
    scope = Scope(target="10.10.10.5")
    agent = _make_agent("ctf", scope)
    store = EvidenceStore()

    defn = ActionDefinition(
        action_id="synthetic_recon",
        phase="recon",
        description="Synthetic recon check",
        params_model=_DummyParams,
        allowed_target_kinds=["local"],
        command_builder="x",
        satisfies_phase=True,
    )
    proposal = AIActionProposal(
        decision="propose_action",
        action_id="synthetic_recon",
        target="10.10.10.5",
        parameters={},
        rationale="probe",
    )
    obs = Observation(
        action_id="synthetic_recon",
        argv=["id"],
        raw_stdout="uid=0(root) gid=0(root) groups=0(root)",
        raw_stderr="",
        exit_code=0,
    )

    agent._promote_evidence_if_warranted(
        defn=defn, proposal=proposal, observations=[obs], evidence_store=store,
    )

    assert len(store.hypotheses) == 1
    hyp = next(iter(store.hypotheses.values()))
    assert hyp.status == "confirmed"

    assert len(store.validations) == 1
    val = next(iter(store.validations.values()))
    assert val.verdict == "supports"

    assert len(store.findings) == 1
    finding = next(iter(store.findings.values()))
    assert finding.severity == "info"
    assert finding.affected_target == "10.10.10.5"


def test_evidence_promotion_skipped_when_satisfies_phase_false():
    scope = Scope(target="local")
    agent = _make_agent("ctf", scope)
    store = EvidenceStore()

    defn = ActionDefinition(
        action_id="low_value_enum",
        phase="enumeration",
        description="Low-value enumeration helper",
        params_model=_DummyParams,
        allowed_target_kinds=["local"],
        command_builder="x",
        satisfies_phase=False,
    )
    proposal = AIActionProposal(
        decision="propose_action", action_id="low_value_enum", target="local",
        parameters={}, rationale="probe",
    )
    obs = Observation(action_id="low_value_enum", argv=["id"], raw_stdout="output", exit_code=0)

    agent._promote_evidence_if_warranted(
        defn=defn, proposal=proposal, observations=[obs], evidence_store=store,
    )

    assert len(store.findings) == 0
    assert len(store.hypotheses) == 0


def test_evidence_promotion_skipped_on_nonzero_exit_or_empty_output():
    scope = Scope(target="local")
    agent = _make_agent("ctf", scope)
    store = EvidenceStore()

    defn = ActionDefinition(
        action_id="failed_action",
        phase="recon",
        description="Failed action",
        params_model=_DummyParams,
        allowed_target_kinds=["local"],
        command_builder="x",
        satisfies_phase=True,
    )
    proposal = AIActionProposal(
        decision="propose_action", action_id="failed_action", target="local",
        parameters={}, rationale="probe",
    )
    failed_obs = Observation(action_id="failed_action", argv=["false"], raw_stdout="", exit_code=1)
    empty_obs = Observation(action_id="failed_action", argv=["true"], raw_stdout="", exit_code=0)

    agent._promote_evidence_if_warranted(
        defn=defn, proposal=proposal, observations=[failed_obs, empty_obs], evidence_store=store,
    )

    assert len(store.findings) == 0


# ---------------------------------------------------------------------------
# H1: per-agent completion dispatch
# ---------------------------------------------------------------------------
def test_completion_verifier_resolves_per_agent_type():
    ctf_agent = _make_agent("ctf", Scope(target="local"))
    pentest_agent = _make_agent("pentest", Scope(target="local"))

    assert ctf_agent._get_completion_verifier() is ctf_completion.verify_completion
    assert pentest_agent._get_completion_verifier() is pentest_completion.verify_completion
    # The two must NOT resolve to the same function (that was the H1 bug).
    assert ctf_agent._get_completion_verifier() is not pentest_agent._get_completion_verifier()


def test_completion_verifier_raises_for_unknown_agent_type():
    bogus_agent = _make_agent("totally_not_a_real_agent", Scope(target="local"))
    with pytest.raises(ImportError):
        bogus_agent._get_completion_verifier()
