"""
tests/test_risk_tier_gate.py

BUILD SPEC v1 workstream B (2026-11-28) -- SAFETY REGRESSION GUARD for the
risk-tiered approval gate, the blast-radius cap, and loosened specialist-engine
access.

THE SAFETY INVARIANT under test: we loosen the gate for KNOWN-BENIGN actions
ONLY. Dangerous/offensive/irreversible actions ALWAYS require human approval on
a real target. FAIL-SAFE: an action that cannot be positively classified as
benign is Tier 2 and is never auto-run. "Fails closed" now means HALT into an
explicit awaiting-approval state -- never auto-proceed, never silently decline.
"""
import os
import sys

import pytest
from pydantic import BaseModel

from core.scope import Scope
from core.action_contract import ActionDefinition, AIActionProposal, ActionRegistry, action_is_unlocked
from core.command_executor import CommandExecutor


class _NoParams(BaseModel):
    model_config = {"extra": "forbid"}


def _action(action_id="probe", *, risk_level="low", network_effect="read_only",
            state_change="none", requires_active_testing=False):
    return ActionDefinition(
        action_id=action_id, phase="recon", description="x",
        risk_level=risk_level, network_effect=network_effect, state_change=state_change,
        requires_active_testing=requires_active_testing,
        params_model=_NoParams, allowed_target_kinds=["domain", "url", "local"],
        command_builder="noop",
    )


def _run(defn, *, target="h.example", in_scope=None):
    scope = Scope(
        target=target, in_scope=in_scope if in_scope is not None else [target],
        active_testing_allowed=True, active_network_allowed=True, state_changes_allowed=True,
    )
    executor = CommandExecutor(read_only=False)
    registry = ActionRegistry(agent_name="test")
    registry.register(defn, lambda p: [[sys.executable, "-c", "print('ok')"]])
    proposal = AIActionProposal(decision="propose_action", action_id=defn.action_id,
                                target=target, parameters={}, rationale="x")
    return executor.execute_action(proposal, registry, scope, {}, mode="automated")


def _clear_target_env(monkeypatch):
    for v in ("PENTEST_TARGET_KIND", "PENTEST_LAB_AUTO_APPROVE", "PENTEST_LAB_AUTO_APPROVE_DANGEROUS"):
        monkeypatch.delenv(v, raising=False)


# --- Tier classification (fail-safe) ---------------------------------------

def test_unknown_or_unclassified_action_is_tier2_failsafe():
    """An action that cannot be positively classified as benign -> Tier 2."""
    # A stub with NO risk metadata at all: getattr fail-safe -> high/active/modify.
    class _Bare:
        action_id = "mystery"
    assert CommandExecutor._classify_risk_tier(_Bare()) == 2
    # A high-risk active action is Tier 2.
    assert CommandExecutor._classify_risk_tier(
        _action(risk_level="high", network_effect="active", state_change="none")) == 2
    # Any target-state change is Tier 2 regardless of risk label.
    assert CommandExecutor._classify_risk_tier(
        _action(risk_level="low", state_change="target_modify")) == 2
    # Benign read-only recon is Tier 0.
    assert CommandExecutor._classify_risk_tier(
        _action(risk_level="low", network_effect="read_only", state_change="none")) == 0
    # Read-only credential exposure is Tier 1.
    assert CommandExecutor._classify_risk_tier(
        _action(risk_level="medium", network_effect="read_only", state_change="credential_access")) == 1


# --- Tier 0: auto-proceeds with no approver, lab AND real ------------------

def test_tier0_auto_proceeds_no_approver_on_real(monkeypatch):
    _clear_target_env(monkeypatch)  # default -> real
    result = _run(_action(risk_level="low", network_effect="read_only", state_change="none"))
    assert result["status"] == "success"
    assert result["approval_record"]["approved_by"] == "tier0_observational_auto"


def test_tier0_auto_proceeds_no_approver_on_lab(monkeypatch):
    _clear_target_env(monkeypatch)
    monkeypatch.setenv("PENTEST_TARGET_KIND", "lab")
    result = _run(_action(risk_level="low", network_effect="read_only", state_change="none"))
    assert result["status"] == "success"
    assert result["approval_record"]["approved_by"] == "tier0_observational_auto"


# --- Tier 2: REAL always needs approval; no approver -> HALT pending --------

def test_tier2_real_no_approver_halts_pending_never_executes(monkeypatch):
    _clear_target_env(monkeypatch)  # real
    result = _run(_action("rce", risk_level="high", network_effect="active", state_change="target_modify"))
    assert result["status"] == "pending_approval"           # NOT declined, NOT success
    assert result["rejection_category"] == "awaiting_human_approval"
    assert result["results"] == []                          # never executed
    assert result["approval_record"]["approved_by"] is None
    assert result["target_kind"] == "real"


# --- Tier 2 on a lab: approval by default; auto only with explicit opt-in ---

def test_tier2_lab_requires_approval_by_default(monkeypatch):
    _clear_target_env(monkeypatch)
    monkeypatch.setenv("PENTEST_TARGET_KIND", "lab")        # lab, but NO dangerous opt-in
    result = _run(_action("rce", risk_level="high", network_effect="active", state_change="target_modify"))
    assert result["status"] == "pending_approval"
    assert result["results"] == []
    assert result["target_kind"] == "lab"


def test_tier2_lab_auto_only_with_dangerous_optin(monkeypatch):
    _clear_target_env(monkeypatch)
    monkeypatch.setenv("PENTEST_LAB_AUTO_APPROVE_DANGEROUS", "1")   # explicit lab dangerous opt-in
    result = _run(_action("rce", risk_level="high", network_effect="active", state_change="target_modify"))
    assert result["status"] == "success"
    assert result["approval_record"]["approved_by"] == "lab_dangerous_auto"


def test_tier2_lab_legacy_flag_still_auto_approves(monkeypatch):
    """Backward compat: the legacy PENTEST_LAB_AUTO_APPROVE=1 still auto-approves
    Tier 2 on a lab (preserves existing lab automation)."""
    _clear_target_env(monkeypatch)
    monkeypatch.setenv("PENTEST_LAB_AUTO_APPROVE", "1")
    result = _run(_action("self_script", risk_level="high", network_effect="active", state_change="none"))
    assert result["status"] == "success"
    assert result["approval_record"]["approved_by"] == "lab_auto_approve"


# --- Tier 1: auto on lab, pauses on real -----------------------------------

def test_tier1_auto_on_lab(monkeypatch):
    _clear_target_env(monkeypatch)
    monkeypatch.setenv("PENTEST_TARGET_KIND", "lab")
    result = _run(_action("meta_read", risk_level="medium", network_effect="read_only",
                          state_change="credential_access"))
    assert result["status"] == "success"
    assert result["approval_record"]["approved_by"] == "tier1_lab_auto"


def test_tier1_pauses_on_real(monkeypatch):
    _clear_target_env(monkeypatch)  # real
    result = _run(_action("meta_read", risk_level="medium", network_effect="read_only",
                          state_change="credential_access"))
    assert result["status"] == "pending_approval"
    assert result["results"] == []
    assert result["target_kind"] == "real"


# --- Interactive approver present: declines on 'n', never executes ----------

def test_tier2_interactive_decline_does_not_execute(monkeypatch):
    _clear_target_env(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
    result = _run(_action("rce", risk_level="high", network_effect="active", state_change="target_modify"))
    assert result["status"] == "declined"
    assert result["results"] == []


# --- run_target.py refuses blanket auto-approve on a real target ------------

@pytest.mark.parametrize("flag", ["PENTEST_LAB_AUTO_APPROVE", "PENTEST_LAB_AUTO_APPROVE_DANGEROUS"])
def test_run_target_refuses_lab_auto_approve_flags(monkeypatch, flag):
    import run_target
    _clear_target_env(monkeypatch)
    monkeypatch.setenv(flag, "1")
    monkeypatch.setattr(sys, "argv", ["run_target.py", "http://example.com/"])
    with pytest.raises(SystemExit) as exc:
        run_target.main()
    assert exc.value.code == 2


def test_run_target_refuses_lab_target_kind(monkeypatch):
    import run_target
    _clear_target_env(monkeypatch)
    monkeypatch.setenv("PENTEST_TARGET_KIND", "lab")
    monkeypatch.setattr(sys, "argv", ["run_target.py", "http://example.com/"])
    with pytest.raises(SystemExit) as exc:
        run_target.main()
    assert exc.value.code == 2


# --- Part 2: blast-radius cap ----------------------------------------------

def test_blast_radius_per_action_cap_aborts(monkeypatch):
    _clear_target_env(monkeypatch)
    monkeypatch.setenv("PENTEST_REAL_MAX_REQUESTS_PER_ACTION", "5")
    executor = CommandExecutor(read_only=False)
    ok, reason, _cap = executor._blast_radius_check([["echo", "x"]] * 20, "real")
    assert ok is False
    assert "max_requests_per_action" in reason


def test_blast_radius_global_cap_aborts(monkeypatch):
    _clear_target_env(monkeypatch)
    monkeypatch.setenv("PENTEST_REAL_GLOBAL_REQUEST_CAP", "10")
    executor = CommandExecutor(read_only=False)
    executor._global_request_count = 9
    ok, reason, _cap = executor._blast_radius_check([["echo", "x"]] * 5, "real")
    assert ok is False
    assert "global_request_cap" in reason


def test_blast_radius_real_tighter_than_lab():
    executor = CommandExecutor(read_only=False)
    lab = executor._blast_radius_limits("lab")
    real = executor._blast_radius_limits("real")
    assert real["max_requests_per_action"] < lab["max_requests_per_action"]
    assert real["max_request_timeout"] < lab["max_request_timeout"]


# --- Part 3: specialist engine reachable without a class match -------------

def test_specialist_reachable_with_none_or_novel_class():
    """A class-gated specialist is reachable once ANY open hypothesis exists --
    including one whose vuln_class is None or a novel/unrecognized string. No
    class-string match is required (the SVG-lab lockout is gone). Suspicion is
    still required: with no hypothesis the specialist stays locked."""
    specialist = ActionDefinition(
        action_id="sqli_deep_scan", phase="vuln_validation", description="x",
        params_model=_NoParams, command_builder="noop",
        vuln_class="sqli", requires_suspicion=True,
    )
    # No suspicion at all -> still locked (suspicion-before-firing preserved).
    assert action_is_unlocked(specialist, set(), False) is False
    # An open hypothesis with a NOVEL class unlocks it (no 'sqli' match needed).
    assert action_is_unlocked(specialist, {"some-novel-uncatalogued-thing"}, True) is True
    # An open hypothesis with vuln_class=None (any_suspicion True, empty classes) unlocks it.
    assert action_is_unlocked(specialist, set(), True) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
