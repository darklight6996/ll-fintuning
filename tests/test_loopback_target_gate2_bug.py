"""
tests/test_loopback_target_gate2_bug.py

Regression tests for the Gate 2 "loopback is not a mere local sentinel" fix
in core/command_executor.py::CommandExecutor.execute_action().

Found live (pentest agent, Qwen2.5-7B-Instruct + qwen-qlora smoke test,
2026-11-14, same run as the target-param-fallback fix): once that fix let
proposals reach Gate 2, EVERY action was then rejected with "Local target is
not permitted for action '<id>'" -- because the old Gate 2 treated
"127.0.0.1"/"::1"/"localhost" identically to the bare "local"/""/path
sentinel, and none of pentest's actions declare allowed_target_kinds=["local"]
(that kind is reserved for CTF's filesystem-only actions). But
verify_pentest_smoke_test.py deliberately targets 127.0.0.1 -- a legitimate,
classifiable loopback IP that real nmap/curl commands run against -- not a
placeholder.

The fix: only "local"/""/a path-like string is treated as the un-classifiable
sentinel now. 127.0.0.1/::1/localhost are classified via _classify_target_kind
("ip"/"ip"/"domain") and checked against allowed_target_kinds like any other
real target, while STILL also satisfying a "local"-only action (CTF) so
agents/ctf's existing verify_autohunt_smoke_test.py (which also targets
"127.0.0.1" against allowed_target_kinds=["local"] actions) keeps working.
"""

import sys
from pydantic import BaseModel

from core.scope import Scope
from core.action_contract import ActionDefinition, AIActionProposal, ActionRegistry
from core.command_executor import CommandExecutor


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


def _run(target, allowed_target_kinds, in_scope=None):
    scope = Scope(
        target=target,
        in_scope=in_scope if in_scope is not None else [target],
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


def test_loopback_ip_accepted_by_ip_action_pentest_case():
    """The exact bug: a pentest-style action (allowed_target_kinds=["ip", ...],
    no "local") against target 127.0.0.1 must now succeed instead of being
    rejected as an impermissible 'local' target."""
    result = _run("127.0.0.1", allowed_target_kinds=["ip", "cidr", "domain"])
    assert result["status"] == "success", result.get("rejection_reason")


def test_loopback_ipv6_accepted_by_ip_action():
    result = _run("::1", allowed_target_kinds=["ip", "domain"])
    assert result["status"] == "success", result.get("rejection_reason")


def test_localhost_hostname_accepted_by_domain_action():
    result = _run("localhost", allowed_target_kinds=["domain"])
    assert result["status"] == "success", result.get("rejection_reason")


def test_loopback_ip_still_accepted_by_local_only_action_ctf_case():
    """Must not regress CTF's existing behavior: a "local"-only action
    (suid_audit/find_writable-style) targeted at 127.0.0.1 (exactly what
    verify_autohunt_smoke_test.py does) must still succeed."""
    result = _run("127.0.0.1", allowed_target_kinds=["local"])
    assert result["status"] == "success", result.get("rejection_reason")


def test_loopback_ip_rejected_when_out_of_declared_scope():
    """A loopback target is now routed through the real scope-membership
    check (previously skipped entirely for 'local'-ish strings) -- it must
    still be rejected if it's genuinely not in the authorized in_scope list."""
    result = _run("127.0.0.1", allowed_target_kinds=["ip"], in_scope=["10.0.0.0/8"])
    assert result["status"] == "rejected"
    assert result["rejection_category"] == "out_of_scope"


def test_bare_local_sentinel_still_requires_local_kind():
    """The literal sentinel "local" (no real host identity at all) must still
    be rejected for an action that doesn't declare allowed_target_kinds=
    ["local"] -- unchanged from before this fix."""
    result = _run("local", allowed_target_kinds=["ip", "domain"])
    assert result["status"] == "rejected"
    assert result["rejection_category"] == "out_of_scope"
    assert "Local target is not permitted" in result["rejection_reason"]


def test_empty_target_still_requires_local_kind():
    result = _run("", allowed_target_kinds=["ip", "domain"])
    assert result["status"] == "rejected"
    assert "Local target is not permitted" in result["rejection_reason"]


def test_path_like_target_still_requires_local_kind():
    result = _run("/etc/passwd", allowed_target_kinds=["ip", "domain"])
    assert result["status"] == "rejected"
    assert "Local target is not permitted" in result["rejection_reason"]
