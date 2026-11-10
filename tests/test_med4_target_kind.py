"""
tests/test_med4_target_kind.py

Regression tests for MED-4: Gate 2 in command_executor.py must positively verify
that a non-local target's actual kind (ip/cidr/domain/url) is declared in the
action's allowed_target_kinds -- not just check the negative "is local disallowed"
case, which was all it did before this fix.
"""

import sys
from pydantic import BaseModel

from core.scope import Scope
from core.action_contract import ActionDefinition, AIActionProposal, ActionRegistry
from core.command_executor import CommandExecutor, _classify_target_kind


class _DummyParams(BaseModel):
    pass


def _make_action(allowed_target_kinds):
    return ActionDefinition(
        action_id="probe",
        phase="recon",
        description="probe",
        params_model=_DummyParams,
        allowed_target_kinds=allowed_target_kinds,
        command_builder="build_probe",
    )


def _run(target, allowed_target_kinds):
    scope = Scope(
        target=target,
        in_scope=[target],
        active_testing_allowed=True,
        active_network_allowed=True,
    )
    executor = CommandExecutor(read_only=False)
    registry = ActionRegistry(agent_name="test")
    action = _make_action(allowed_target_kinds)
    registry.register(action, lambda p: [[sys.executable, "-c", "print('ok')"]])

    proposal = AIActionProposal(
        decision="propose_action", action_id="probe", target=target,
        parameters={}, rationale="probe",
    )
    return executor.execute_action(
        proposal=proposal, registry=registry, scope=scope,
        execution_history=[], mode="automated",
    )


def test_classify_target_kind_ip():
    assert _classify_target_kind("192.168.1.10") == "ip"


def test_classify_target_kind_cidr():
    assert _classify_target_kind("192.168.1.0/24") == "cidr"


def test_classify_target_kind_url():
    assert _classify_target_kind("https://example.com/path") == "url"


def test_classify_target_kind_domain():
    assert _classify_target_kind("example.com") == "domain"


def test_classify_target_kind_malformed_ip_falls_back_to_domain():
    # Not a valid IP (octet > 255), and ipaddress correctly rejects it -- must not be
    # misclassified as "ip" by a looser regex-based check.
    assert _classify_target_kind("999.999.999.999") == "domain"


def test_ip_only_action_rejects_url_target():
    result = _run("https://example.com", allowed_target_kinds=["ip"])
    assert result["status"] == "rejected"
    assert result["rejection_category"] == "out_of_scope"


def test_ip_only_action_accepts_ip_target():
    result = _run("10.0.0.5", allowed_target_kinds=["ip"])
    assert result["status"] == "success"


def test_domain_action_rejects_ip_target():
    result = _run("10.0.0.5", allowed_target_kinds=["domain", "url"])
    assert result["status"] == "rejected"
    assert result["rejection_category"] == "out_of_scope"
