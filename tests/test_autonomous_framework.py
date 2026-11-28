"""
tests/test_autonomous_framework.py

Comprehensive test suite for Autonomous AI Security Assessment System.
Verifies all 12 core safety, policy, retry, evidence chain, and execution invariants.
"""

import pytest
import os
import sys
import json
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field

from core.scope import Scope
from core.action_contract import ActionDefinition, AIActionProposal, ActionRegistry
from core.investigation import EvidenceStore, Observation, Hypothesis, Validation, Finding
from core.command_executor import CommandExecutor
from core.session_manager import SessionManager
from agents.ctf.actions import build_ctf_registry, SUID_AUDIT, PATH_HIJACK_VALIDATION
from agents.ctf.completion import CTF_COMPLETION, CompletionCriteria, verify_completion
import agents.ctf.command_builders as cb


# ---------------------------------------------------------------------------
# Test 1: Structural Scope Validation (rejects even inside longer path or URL)
# ---------------------------------------------------------------------------
def test_scope_validation_structural():
    scope = Scope(
        target="192.168.1.10",
        in_scope=["192.168.1.10"],
        out_of_scope=["10.0.0.1", "forbidden.internal", "evil.com"],
        active_testing_allowed=True,
        active_network_allowed=True,
    )
    executor = CommandExecutor(read_only=False)
    registry = build_ctf_registry()

    # Out-of-scope host in a URL target
    proposal = AIActionProposal(
        decision="propose_action",
        action_id="environment_recon",
        target="http://forbidden.internal/api/v1",
        parameters={},
        rationale="testing",
    )
    result = executor.execute_action(proposal, registry, scope, {}, mode="automated")
    assert result["status"] == "rejected"
    assert result["rejection_category"] in ("scope_violation", "out_of_scope")


# ---------------------------------------------------------------------------
# Test 2: Policy Gates (active network rejected when active_network_allowed=False)
# ---------------------------------------------------------------------------
def test_policy_gates():
    scope = Scope(
        target="local",
        active_network_allowed=False,  # Prohibits active network
        state_changes_allowed=False,
    )
    executor = CommandExecutor(read_only=False)
    registry = ActionRegistry(agent_name="test")

    class DummyParams(BaseModel):
        model_config = {"extra": "forbid"}

    network_action = ActionDefinition(
        action_id="network_probe",
        phase="recon",
        description="Active network probe",
        requires_active_testing=True,
        network_effect="active",
        state_change="none",
        risk_level="low",
        params_model=DummyParams,
        allowed_target_kinds=["local"],
        command_builder="build_dummy",
    )
    registry.register(network_action, lambda p: [["echo", "probe"]])

    proposal = AIActionProposal(
        decision="propose_action",
        action_id="network_probe",
        target="local",
        parameters={},
        rationale="Probe network",
    )
    result = executor.execute_action(proposal, registry, scope, {}, mode="automated")
    assert result["status"] == "rejected"
    assert result["rejection_category"] in ("policy_violation", "prohibited_action")
    assert "network" in result["rejection_reason"].lower()


# ---------------------------------------------------------------------------
# Test 3: Retry Budget Enforcement (exceeding budget rejected by framework)
# ---------------------------------------------------------------------------
def test_retry_budget_enforcement():
    scope = Scope(target="local")
    executor = CommandExecutor(read_only=False)
    registry = build_ctf_registry()

    # SUID_AUDIT has max_retries = 2
    proposal = AIActionProposal(
        decision="propose_action",
        action_id="suid_audit",
        target="local",
        parameters={"search_path": "/usr/bin"},
        rationale="Audit SUID",
    )
    action_ident = proposal.compute_identity()

    # Simulate 2 prior attempts for this action identity
    session_state = {
        "autonomous_execution_history": [
            {"action_identity": action_ident, "action_id": "suid_audit"},
            {"action_identity": action_ident, "action_id": "suid_audit"},
        ]
    }

    result = executor.execute_action(proposal, registry, scope, session_state, mode="automated")
    assert result["status"] == "rejected"
    assert result["rejection_category"] in ("retry_budget_exhausted", "retry_budget_exceeded")


# ---------------------------------------------------------------------------
# Test 4: High-Risk Requires Confirmation (autonomous fails closed on non-interactive)
# ---------------------------------------------------------------------------
def test_high_risk_requires_confirmation(monkeypatch):
    scope = Scope(target="local", active_testing_allowed=True, state_changes_allowed=True)
    executor = CommandExecutor(read_only=False)
    registry = ActionRegistry(agent_name="test")

    class HighRiskParams(BaseModel):
        model_config = {"extra": "forbid"}

    high_risk_action = ActionDefinition(
        action_id="nuke_all",
        phase="impact",
        description="Destructive action",
        risk_level="high",
        params_model=HighRiskParams,
        command_builder="build_nuke",
    )
    registry.register(high_risk_action, lambda p: [["echo", "nuke"]])

    proposal = AIActionProposal(
        decision="propose_action",
        action_id="nuke_all",
        target="local",
        parameters={},
        rationale="High risk action",
    )

    # Non-interactive + Tier 2 (high-risk, no lab marker -> real target): the
    # tiered gate HALTS into an explicit awaiting-approval pending state (BUILD
    # SPEC v1 workstream B) -- it must NOT auto-proceed and must NOT execute.
    # "Fails closed" now means "halts pending", not "silently declined".
    result = executor.execute_action(proposal, registry, scope, {}, mode="automated")
    assert result["status"] == "pending_approval"
    assert result["rejection_category"] == "awaiting_human_approval"
    assert result["results"] == []           # critically: nothing executed
    assert result["approval_record"]["approved_by"] is None


# ---------------------------------------------------------------------------
# Test 4b: PENTEST_LAB_AUTO_APPROVE (2026-11-27) -- LAB-ONLY switch for the
# exact high-risk/non-interactive deadlock Test 4 just proved exists. Must
# only ever be reachable through command_executor.py's autonomous+high-risk
# branch; run_target.py (real targets) refuses to even start when this is
# set -- see run_target.py's own hard exit, not re-tested here (separate file).
# ---------------------------------------------------------------------------
def test_lab_auto_approve_bypasses_the_non_interactive_deadlock(monkeypatch):
    monkeypatch.setenv("PENTEST_LAB_AUTO_APPROVE", "1")
    scope = Scope(target="local", active_testing_allowed=True, state_changes_allowed=True)
    executor = CommandExecutor(read_only=False)
    registry = ActionRegistry(agent_name="test")

    class HighRiskParams(BaseModel):
        model_config = {"extra": "forbid"}

    high_risk_action = ActionDefinition(
        action_id="self_script", phase="vuln_validation", description="model-authored probe",
        risk_level="high", params_model=HighRiskParams, command_builder="build_dummy",
    )
    registry.register(high_risk_action, lambda p: [["echo", "probe"]])
    proposal = AIActionProposal(decision="propose_action", action_id="self_script", target="local",
                                parameters={}, rationale="probe")

    result = executor.execute_action(proposal, registry, scope, {}, mode="automated")
    assert result["status"] == "success"
    assert result["approval_record"]["approved_by"] == "lab_auto_approve"


def test_lab_auto_approve_off_by_default_still_fails_closed(monkeypatch):
    # No lab markers -> treated as a REAL target (fail-safe). Tier 2 + no
    # approver -> HALT pending (never executes). Still fails closed; the new
    # behavior surfaces it for later approval instead of a blunt decline.
    monkeypatch.delenv("PENTEST_LAB_AUTO_APPROVE", raising=False)
    monkeypatch.delenv("PENTEST_LAB_AUTO_APPROVE_DANGEROUS", raising=False)
    monkeypatch.delenv("PENTEST_TARGET_KIND", raising=False)
    scope = Scope(target="local", active_testing_allowed=True, state_changes_allowed=True)
    executor = CommandExecutor(read_only=False)
    registry = ActionRegistry(agent_name="test")

    class HighRiskParams(BaseModel):
        model_config = {"extra": "forbid"}

    high_risk_action = ActionDefinition(
        action_id="self_script", phase="vuln_validation", description="model-authored probe",
        risk_level="high", params_model=HighRiskParams, command_builder="build_dummy",
    )
    registry.register(high_risk_action, lambda p: [["echo", "probe"]])
    proposal = AIActionProposal(decision="propose_action", action_id="self_script", target="local",
                                parameters={}, rationale="probe")

    result = executor.execute_action(proposal, registry, scope, {}, mode="automated")
    assert result["status"] == "pending_approval"
    assert result["rejection_category"] == "awaiting_human_approval"
    assert result["results"] == []           # never executed without approval
    assert result["target_kind"] == "real"


# ---------------------------------------------------------------------------
# Test 5: Completion Verification Rejects Early (framework-enforced gate)
# ---------------------------------------------------------------------------
def test_completion_verification_rejects_early():
    evidence_store = EvidenceStore()
    session_state = {"autonomous_execution_history": []}

    # Early completion with no required phases and no findings
    can_complete, reason = verify_completion(CTF_COMPLETION, session_state, evidence_store)
    assert can_complete is False
    assert "Required phase" in reason or "required" in reason.lower()


# ---------------------------------------------------------------------------
# Test 6: Hypothesis-to-Finding Chain (refuses finding without complete chain)
# ---------------------------------------------------------------------------
def test_hypothesis_to_finding_chain():
    store = EvidenceStore()

    # 1. Observation
    obs = Observation(
        action_id="suid_audit",
        argv=["find", "/usr/bin", "-perm", "-4000"],
        raw_stdout="-rwsr-xr-x /usr/local/bin/backup",
        exit_code=0,
    )
    store.add_observation(obs)

    # 2. Hypothesis
    hyp = Hypothesis(
        observation_ids=[obs.obs_id],
        description="Backup binary is SUID and calls relative tar command",
        validation_test="Test PATH injection on /usr/local/bin/backup",
    )
    store.add_hypothesis(hyp)

    # Cannot create finding yet (hypothesis is still open)
    finding, msg = store.record_finding(
        hypothesis_id=hyp.hyp_id,
        title="Privilege Escalation via SUID PATH Hijack",
        description="Relative tar command vulnerability",
        severity="high",
        affected_target="local",
        evidence_summary="SUID backup binary found",
    )
    assert finding is None
    assert "expected 'confirmed'" in msg

    # 3. Validation with verdict="supports"
    val = Validation(
        hypothesis_id=hyp.hyp_id,
        action_id="path_hijack_validation",
        observation=obs,
        verdict="supports",
        reasoning="Fake tar executed as root confirming privesc",
    )
    store.add_validation(val)

    # Now finding creation MUST succeed
    finding, msg = store.record_finding(
        hypothesis_id=hyp.hyp_id,
        title="Privilege Escalation via SUID PATH Hijack",
        description="Relative tar command vulnerability",
        severity="high",
        affected_target="local",
        evidence_summary="SUID backup binary found and fake tar executed",
    )
    assert finding is not None
    assert finding.finding_id in store.findings
    assert hyp.status == "confirmed"


# ---------------------------------------------------------------------------
# Test 7: False Positive Tracking (refuted hypothesis stored in refuted_hypotheses)
# ---------------------------------------------------------------------------
def test_false_positive_tracking():
    store = EvidenceStore()
    obs = Observation(
        action_id="strings_analysis",
        argv=["strings", "/bin/ping"],
        raw_stdout="root",
    )
    store.add_observation(obs)

    hyp = Hypothesis(
        observation_ids=[obs.obs_id],
        description="Ping binary contains hardcoded root password",
        validation_test="Try password",
    )
    store.add_hypothesis(hyp)

    # Add refuting validation
    val = Validation(
        hypothesis_id=hyp.hyp_id,
        action_id="test_password",
        observation=obs,
        verdict="refutes",
        reasoning="String 'root' was simply a comment in help text",
    )
    store.add_validation(val)

    assert hyp.status == "refuted"
    assert hyp.hyp_id in store.refuted_hypotheses
    assert len(store.findings) == 0


# ---------------------------------------------------------------------------
# Test 8: Ctrl+C Preserves State (atomic persistence to disk)
# ---------------------------------------------------------------------------
def test_ctrl_c_preserves_state(tmp_path):
    sm = SessionManager(base_dir=str(tmp_path))
    sm.create_session("ctrl_c_test", target="local")

    evidence_store = EvidenceStore()
    obs = Observation(
        action_id="environment_recon",
        argv=["id"],
        raw_stdout="uid=1000(ctf) gid=1000(ctf)",
    )
    evidence_store.add_observation(obs)

    # Save state as done during KeyboardInterrupt
    sm.save_current_state()
    sm.save_investigation_store(evidence_store)

    # Reload from fresh SessionManager
    sm2 = SessionManager(base_dir=str(tmp_path))
    sm2.load_session("ctrl_c_test")
    loaded_store = sm2.load_investigation_store()

    assert loaded_store is not None
    assert obs.obs_id in loaded_store.observations
    assert loaded_store.observations[obs.obs_id].raw_stdout == "uid=1000(ctf) gid=1000(ctf)"


# ---------------------------------------------------------------------------
# Test 9: Manual Mode Behavior Unchanged (regression check)
# ---------------------------------------------------------------------------
def test_manual_mode_unchanged():
    executor = CommandExecutor(read_only=True)
    # Read-only executor in manual mode prepares command previews
    previews = executor.prepare_step_commands({"commands": ["whoami"], "action": "Identify current user"})
    assert len(previews) == 1
    assert previews[0].command == "whoami"

    # execute_command with read_only=True returns skipped
    res = executor.execute_command("whoami")
    assert res.get("status") == "skipped"
    assert "Read-Only" in res.get("reason", "")

    # When read_only is False but not approved, returns declined
    active_executor = CommandExecutor(read_only=False)
    res_unapproved = active_executor.execute_command("whoami", approved=False)
    assert res_unapproved.get("status") == "declined" 


# ---------------------------------------------------------------------------
# Test 10: Command Builders Return List[List[str]] (No shell strings)
# ---------------------------------------------------------------------------
def test_command_builder_no_shell_strings():
    builders = [
        (cb.build_environment_recon_argv, None),
        (cb.build_suid_audit_argv, None),
        (cb.build_strings_analysis_argv, type("P", (), {"binary_path": "/bin/sh", "min_length": 8})()),
        (cb.build_read_file_argv, type("P", (), {"file_path": "/flag.txt"})()),
        (cb.build_path_hijack_argv, type("P", (), {"fake_binary_name": "fortune", "target_suid_binary": "/usr/bin/chk", "work_dir": "/tmp"})()),
        (cb.build_find_writable_argv, None),
        (cb.build_kernel_exploit_check_argv, None),
        (cb.build_process_list_argv, None),
        (cb.build_cron_job_audit_argv, None),
        (cb.build_flag_search_argv, None),
    ]

    for builder, params in builders:
        argv_list = builder(params)
        assert isinstance(argv_list, list), f"{builder.__name__} did not return a list"
        assert len(argv_list) > 0, f"{builder.__name__} returned an empty list"
        for argv in argv_list:
            assert isinstance(argv, list), f"{builder.__name__} item is not a list"
            for token in argv:
                assert isinstance(token, str), f"{builder.__name__} token is not a string"


# ---------------------------------------------------------------------------
# Test 11: Arg Injection Blocked (shell metacharacters safe with shell=False)
# ---------------------------------------------------------------------------
def test_arg_injection_blocked(tmp_path):
    # Injection payload that would create a file if passed to a shell
    canary_file = tmp_path / "injected.txt"
    payload = f"test; touch {canary_file} #"

    executor = CommandExecutor(read_only=False)
    registry = ActionRegistry(agent_name="test")

    class SearchParams(BaseModel):
        model_config = {"extra": "forbid"}
        pattern: str

    action = ActionDefinition(
        action_id="search_flag",
        phase="flag_capture",
        description="Search with pattern",
        params_model=SearchParams,
        command_builder="build_search",
    )
    # Builder passes payload directly as an argv argument
    registry.register(action, lambda p: [[sys.executable, "-c", "import sys; print(sys.argv[1])", p.pattern]])

    scope = Scope(target="local")
    proposal = AIActionProposal(
        decision="propose_action",
        action_id="search_flag",
        target="local",
        parameters={"pattern": payload},
        rationale="Injection test",
    )

    result = executor.execute_action(proposal, registry, scope, {}, mode="automated")
    assert result["status"] == "success"
    # The shell metacharacters were passed literally, NOT evaluated by a shell
    assert not canary_file.exists(), "Arg injection succeeded! Canary file was created!"


# ---------------------------------------------------------------------------
# Test 12: Retry Budget Per Action Identity
# ---------------------------------------------------------------------------
def test_retry_budget_per_action_identity():
    # Same action_id, but different parameters/targets produce distinct hashes
    proposal_a = AIActionProposal(
        decision="propose_action",
        action_id="suid_audit",
        target="local",
        parameters={"search_path": "/home"},
    )
    proposal_b = AIActionProposal(
        decision="propose_action",
        action_id="suid_audit",
        target="local",
        parameters={"search_path": "/etc"},
    )
    proposal_c = AIActionProposal(
        decision="propose_action",
        action_id="suid_audit",
        target="192.168.1.50",
        parameters={"search_path": "/home"},
    )

    id_a = proposal_a.compute_identity()
    id_b = proposal_b.compute_identity()
    id_c = proposal_c.compute_identity()

    assert id_a != id_b, "Different parameters must produce different identities"
    assert id_a != id_c, "Different targets must produce different identities"
    # Idempotence check
    assert id_a == proposal_a.compute_identity()


# ---------------------------------------------------------------------------
# Test 13: Evidence Graph Entities, Relationships, and Neighbor Queries
# ---------------------------------------------------------------------------
def test_evidence_graph_entities_and_relationships():
    from core.investigation import EvidenceStore, Entity, Relationship, EvidenceGraph

    store = EvidenceStore()
    assert isinstance(store.graph, EvidenceGraph)

    e_host = Entity(entity_type="host", name="web_server", value="10.10.10.50")
    e_port = Entity(entity_type="port", name="http_port", value="8080")
    e_user = Entity(entity_type="user", name="app_admin", value="admin")

    store.graph.add_entity(e_host)
    store.graph.add_entity(e_port)
    store.graph.add_entity(e_user)

    rel_expose = Relationship(
        source_id=e_host.entity_id,
        relation_type="exposes",
        target_id=e_port.entity_id
    )
    rel_auth = Relationship(
        source_id=e_port.entity_id,
        relation_type="authenticates_as",
        target_id=e_user.entity_id
    )
    store.graph.add_relationship(rel_expose)
    store.graph.add_relationship(rel_auth)

    assert len(store.graph.entities) == 3
    assert len(store.graph.relationships) == 2

    # Query neighbors
    neighbors = store.graph.get_neighbors(e_port.entity_id, direction="both")
    assert len(neighbors) == 2
    connected_ids = {ent.entity_id for _, ent in neighbors}
    assert e_host.entity_id in connected_ids
    assert e_user.entity_id in connected_ids

    # Serialization roundtrip
    dumped = store.to_dict()
    assert "graph" in dumped
    reloaded = EvidenceStore.from_dict(dumped)
    assert len(reloaded.graph.entities) == 3
    assert len(reloaded.graph.relationships) == 2


# ---------------------------------------------------------------------------
# Test 14: Evidence Graph Multi-Hop Attack Path Finding
# ---------------------------------------------------------------------------
def test_evidence_graph_multi_hop_path_finding():
    from core.investigation import EvidenceGraph, Entity, Relationship

    graph = EvidenceGraph()

    # Create a 4-hop attack path: Host -> Service -> User -> Credential -> Root
    e_host = Entity(entity_type="host", name="target_box", value="10.10.10.5")
    e_svc = Entity(entity_type="service", name="ssh_service", value="22")
    e_user = Entity(entity_type="user", name="low_priv", value="webuser")
    e_cred = Entity(entity_type="credential", name="suid_binary", value="/usr/bin/vuln")
    e_root = Entity(entity_type="user", name="privileged", value="root")

    for e in [e_host, e_svc, e_user, e_cred, e_root]:
        graph.add_entity(e)

    r1 = Relationship(source_id=e_host.entity_id, relation_type="runs_on", target_id=e_svc.entity_id)
    r2 = Relationship(source_id=e_svc.entity_id, relation_type="authenticates_as", target_id=e_user.entity_id)
    r3 = Relationship(source_id=e_user.entity_id, relation_type="exposes", target_id=e_cred.entity_id)
    r4 = Relationship(source_id=e_cred.entity_id, relation_type="leads_to", target_id=e_root.entity_id)

    for r in [r1, r2, r3, r4]:
        graph.add_relationship(r)

    # Find paths from host to root
    paths = graph.find_paths(e_host.entity_id, e_root.entity_id, max_depth=5)
    assert len(paths) == 1
    assert len(paths[0]) == 4
    rel_ids = [rel.rel_id for rel in paths[0]]
    assert rel_ids == [r1.rel_id, r2.rel_id, r3.rel_id, r4.rel_id]


# ---------------------------------------------------------------------------
# Test 15: Cross-Observation Correlation
# ---------------------------------------------------------------------------
def test_evidence_graph_cross_observation_correlation():
    from core.investigation import EvidenceGraph, Entity, Relationship

    graph = EvidenceGraph()

    obs_a = "obs_recon_users_123"
    obs_b = "obs_api_endpoint_456"

    # User entity discovered in Observation A
    e_user = Entity(
        entity_type="user",
        name="target_user",
        value="user_1002",
        source_observation_ids=[obs_a]
    )
    # Endpoint entity discovered in Observation B
    e_endpoint = Entity(
        entity_type="endpoint",
        name="profile_api",
        value="/api/v1/user/1002",
        source_observation_ids=[obs_b]
    )
    graph.add_entity(e_user)
    graph.add_entity(e_endpoint)

    # Connect them via exposes relationship
    r_idor = Relationship(
        source_id=e_endpoint.entity_id,
        relation_type="vulnerable_to",
        target_id=e_user.entity_id,
        evidence_ids=[obs_a, obs_b]
    )
    graph.add_relationship(r_idor)

    correlations = graph.correlate_cross_observations(obs_b, obs_a)
    assert len(correlations) >= 1
    assert correlations[0]["type"] == "attack_path"
    assert correlations[0]["source_entity"]["value"] == "/api/v1/user/1002"
    assert correlations[0]["target_entity"]["value"] == "user_1002"


# ---------------------------------------------------------------------------
# Test 16: Bug Bounty Action Registry & Pure Python Command Builders
# ---------------------------------------------------------------------------
def test_bug_bounty_action_registry_and_builders():
    from agents.bug_bounty.actions import (
        build_bug_bounty_registry,
        SubdomainEnumParams,
        PortScanParams,
        EndpointDiscoveryParams,
        DirBruteParams,
        ParamFuzzParams,
        HeaderAnalysisParams,
        IdorProbeParams,
        SqliProbeParams,
        XssProbeParams,
        VulnValidationParams,
    )
    import agents.bug_bounty.command_builders as bb_cb
    from agents.bug_bounty.completion import BUG_BOUNTY_COMPLETION, verify_completion
    from core.investigation import EvidenceStore, Observation

    registry = build_bug_bounty_registry()
    actions = registry.list_actions()
    # >= 10: the 10 original actions PLUS the detection engines (sqlmap, nuclei)
    # and the framework-level shared primitives (http_probe, browser_probe; and
    # self_script when opted in) added 2026-11. The original 10 must all remain.
    assert len(actions) >= 10
    action_ids = [a.action_id for a in actions]
    for original in ("subdomain_enum", "port_scan", "endpoint_discovery", "dir_brute",
                     "param_fuzz", "header_analysis", "idor_probe", "sqli_probe",
                     "xss_probe", "vuln_validation"):
        assert original in action_ids, f"original action {original} missing from registry"

    # Verify builders return List[List[str]] with NO shell strings
    builders_and_params = [
        (bb_cb.build_subdomain_enum, SubdomainEnumParams(target="example.com")),
        (bb_cb.build_port_scan, PortScanParams(target="example.com")),
        (bb_cb.build_endpoint_discovery, EndpointDiscoveryParams(target="https://example.com")),
        (bb_cb.build_dir_brute, DirBruteParams(target="https://example.com")),
        (bb_cb.build_param_fuzz, ParamFuzzParams(target="https://example.com", endpoint="/search")),
        (bb_cb.build_header_analysis, HeaderAnalysisParams(target="https://example.com")),
        (bb_cb.build_idor_probe, IdorProbeParams(target="https://example.com", endpoint="/api/user", user_a_id="101", user_b_id="102")),
        (bb_cb.build_sqli_probe, SqliProbeParams(target="https://example.com", endpoint="/item", test_param="id")),
        (bb_cb.build_xss_probe, XssProbeParams(target="https://example.com", endpoint="/search", param_name="q")),
        (bb_cb.build_vuln_validation, VulnValidationParams(target="https://example.com", endpoint="/api/test", vuln_type="idor", payload="id=102")),
    ]

    for builder, params in builders_and_params:
        cmds = builder(params)
        assert isinstance(cmds, list), f"{builder} must return a list"
        assert len(cmds) >= 1
        for argv in cmds:
            assert isinstance(argv, list), f"Command vector {argv} must be a list of strings"
            assert all(isinstance(arg, str) for arg in argv)
            # Ensure no shell command chaining or pipes in args
            for arg in argv:
                assert not any(op in arg for op in [" | ", " ; ", " && ", " || "]), f"Forbidden shell chaining in {arg}"

    # Verify completion verification rejects before minimum findings
    empty_store = EvidenceStore()
    ok, reason = verify_completion(BUG_BOUNTY_COMPLETION, {}, empty_store)
    assert not ok
    assert "Required phase" in reason or "Minimum" in reason


# ---------------------------------------------------------------------------
# Test 17: Pentest Action Registry & Pure Python Command Builders
# ---------------------------------------------------------------------------
def test_pentest_action_registry_and_builders():
    from agents.pentest.actions import (
        build_pentest_registry,
        HostDiscoveryParams,
        ServiceScanParams,
        VulnScanParams,
        WebEnumParams,
        SmbEnumParams,
        ExploitProbeParams,
        CredentialCheckParams,
        PrivescCheckParams,
        PostExploitationParams,
        GenerateReportParams,
    )
    import agents.pentest.command_builders as pt_cb
    from agents.pentest.completion import PENTEST_COMPLETION, verify_completion
    from core.investigation import EvidenceStore

    registry = build_pentest_registry()
    actions = registry.list_actions()
    assert len(actions) >= 10  # core 10 + shared primitives (http_probe, browser_probe; self_script when opted in)
    action_ids = [a.action_id for a in actions]
    assert "host_discovery" in action_ids
    assert "service_scan" in action_ids
    assert "exploit_probe" in action_ids

    builders_and_params = [
        (pt_cb.build_host_discovery, HostDiscoveryParams(target="10.10.10.0/24")),
        (pt_cb.build_service_scan, ServiceScanParams(target="10.10.10.50")),
        (pt_cb.build_vuln_scan, VulnScanParams(target="10.10.10.50")),
        (pt_cb.build_web_enum, WebEnumParams(target="http://10.10.10.50")),
        (pt_cb.build_smb_enum, SmbEnumParams(target="10.10.10.50")),
        (pt_cb.build_exploit_probe, ExploitProbeParams(target="http://10.10.10.50", exploit_type="rce", test_parameter="cmd=id")),
        (pt_cb.build_credential_check, CredentialCheckParams(target="10.10.10.50", service="ftp", user="anonymous")),
        (pt_cb.build_privesc_check, PrivescCheckParams(target="10.10.10.50", check_type="sudo")),
        (pt_cb.build_post_exploitation, PostExploitationParams(target="10.10.10.50")),
        (pt_cb.build_generate_report, GenerateReportParams(target="10.10.10.50")),
    ]

    for builder, params in builders_and_params:
        cmds = builder(params)
        assert isinstance(cmds, list), f"{builder} must return a list"
        assert len(cmds) >= 1
        for argv in cmds:
            assert isinstance(argv, list)
            assert all(isinstance(arg, str) for arg in argv)
            for arg in argv:
                assert not any(op in arg for op in [" | ", " ; ", " && ", " || "])

    # Verify completion verification rejects when no phases run
    empty_store = EvidenceStore()
    ok, reason = verify_completion(PENTEST_COMPLETION, {}, empty_store)
    assert not ok


# ---------------------------------------------------------------------------
# Test 18: Active Directory Action Registry & Command Builders
# ---------------------------------------------------------------------------
def test_active_directory_action_registry_and_builders():
    from agents.active_directory.actions import (
        build_ad_registry,
        DcDiscoveryParams,
        AdEnumParams,
        KerberoastParams,
        AsrepRoastParams,
        DcsyncAuditParams,
        AclAuditParams,
        TrustAuditParams,
        GpoAuditParams,
        Tier0ImpactParams,
        AdReportParams,
    )
    import agents.active_directory.command_builders as ad_cb
    from agents.active_directory.completion import AD_COMPLETION, verify_completion
    from core.investigation import EvidenceStore

    registry = build_ad_registry()
    actions = registry.list_actions()
    assert len(actions) >= 10  # core 10 + shared primitives (http_probe, browser_probe; self_script when opted in)
    action_ids = [a.action_id for a in actions]
    assert "dc_discovery" in action_ids
    assert "kerberoast" in action_ids
    assert "dcsync_audit" in action_ids

    builders_and_params = [
        (ad_cb.build_dc_discovery, DcDiscoveryParams(target="corp.local", dns_server="10.10.10.1")),
        (ad_cb.build_ad_enum, AdEnumParams(target="10.10.10.1", domain="corp.local")),
        (ad_cb.build_kerberoast, KerberoastParams(target="10.10.10.1", domain="corp.local")),
        (ad_cb.build_asrep_roast, AsrepRoastParams(target="10.10.10.1", domain="corp.local")),
        (ad_cb.build_dcsync_audit, DcsyncAuditParams(target="10.10.10.1", domain="corp.local")),
        (ad_cb.build_acl_audit, AclAuditParams(target="10.10.10.1", domain="corp.local")),
        (ad_cb.build_trust_audit, TrustAuditParams(target="10.10.10.1", domain="corp.local")),
        (ad_cb.build_gpo_audit, GpoAuditParams(target="10.10.10.1", domain="corp.local")),
        (ad_cb.build_tier0_impact, Tier0ImpactParams(target="corp.local")),
        (ad_cb.build_ad_report, AdReportParams(target="corp.local")),
    ]

    for builder, params in builders_and_params:
        cmds = builder(params)
        assert isinstance(cmds, list)
        assert len(cmds) >= 1
        for argv in cmds:
            assert isinstance(argv, list)
            assert all(isinstance(arg, str) for arg in argv)
            for arg in argv:
                assert not any(op in arg for op in [" | ", " ; ", " && ", " || "])

    # Completion gate check
    empty_store = EvidenceStore()
    ok, reason = verify_completion(AD_COMPLETION, {}, empty_store)
    assert not ok


# ---------------------------------------------------------------------------
# Test 19: Cloud Action Registry & Command Builders
# ---------------------------------------------------------------------------
def test_cloud_action_registry_and_builders():
    from agents.cloud.actions import (
        build_cloud_registry,
        IamAuditParams,
        ResourceEnumParams,
        MisconfigValidationParams,
        S3BucketAuditParams,
        AssumeRoleAuditParams,
        SecurityGroupAuditParams,
        CloudTrailAuditParams,
        MetadataServiceAuditParams,
        PrivescPathAuditParams,
        CloudReportParams,
    )
    import agents.cloud.command_builders as cloud_cb
    from agents.cloud.completion import CLOUD_COMPLETION, verify_completion
    from core.investigation import EvidenceStore

    registry = build_cloud_registry()
    actions = registry.list_actions()
    assert len(actions) >= 10  # core 10 + shared primitives (http_probe, browser_probe; self_script when opted in)
    action_ids = [a.action_id for a in actions]
    assert "iam_audit" in action_ids
    assert "s3_bucket_audit" in action_ids
    assert "metadata_service_audit" in action_ids

    builders_and_params = [
        (cloud_cb.build_iam_audit, IamAuditParams(target="123456789012")),
        (cloud_cb.build_resource_enum, ResourceEnumParams(target="123456789012")),
        (cloud_cb.build_misconfig_validation, MisconfigValidationParams(target="123456789012", resource_id="arn:aws:s3:::corp-data")),
        (cloud_cb.build_s3_bucket_audit, S3BucketAuditParams(target="123456789012", bucket_name="corp-data")),
        (cloud_cb.build_assume_role_audit, AssumeRoleAuditParams(target="123456789012", role_arn="arn:aws:iam::123456789012:role/Admin")),
        (cloud_cb.build_security_group_audit, SecurityGroupAuditParams(target="123456789012")),
        (cloud_cb.build_cloudtrail_audit, CloudTrailAuditParams(target="123456789012")),
        (cloud_cb.build_metadata_service_audit, MetadataServiceAuditParams(target="169.254.169.254")),
        (cloud_cb.build_privesc_path_audit, PrivescPathAuditParams(target="123456789012")),
        (cloud_cb.build_cloud_report, CloudReportParams(target="123456789012")),
    ]

    for builder, params in builders_and_params:
        cmds = builder(params)
        assert isinstance(cmds, list)
        assert len(cmds) >= 1
        for argv in cmds:
            assert isinstance(argv, list)
            assert all(isinstance(arg, str) for arg in argv)
            for arg in argv:
                assert not any(op in arg for op in [" | ", " ; ", " && ", " || "])

    # Completion gate check
    empty_store = EvidenceStore()
    ok, reason = verify_completion(CLOUD_COMPLETION, {}, empty_store)
    assert not ok


# ---------------------------------------------------------------------------
# Test 20: Red Team Action Registry & Command Builders
# ---------------------------------------------------------------------------
def test_red_team_action_registry_and_builders():
    from agents.red_team.actions import (
        build_red_team_registry,
        ExternalReconParams,
        PerimeterScanParams,
        InitialAccessProbeParams,
        CredentialStuffingParams,
        PrivescCheckParams,
        DefenseEvasionParams,
        LateralMovementParams,
        DataDiscoveryParams,
        ExfiltrationTestParams,
        ObjectiveImpactParams,
    )
    import agents.red_team.command_builders as rt_cb
    from agents.red_team.completion import RED_TEAM_COMPLETION, verify_completion
    from core.investigation import EvidenceStore

    registry = build_red_team_registry()
    actions = registry.list_actions()
    assert len(actions) >= 10  # core 10 + shared primitives (http_probe, browser_probe; self_script when opted in)
    action_ids = [a.action_id for a in actions]
    assert "external_recon" in action_ids
    assert "initial_access_probe" in action_ids
    assert "lateral_movement" in action_ids

    builders_and_params = [
        (rt_cb.build_external_recon, ExternalReconParams(target="target.corp")),
        (rt_cb.build_perimeter_scan, PerimeterScanParams(target="10.10.10.0/24")),
        (rt_cb.build_initial_access_probe, InitialAccessProbeParams(target="https://target.corp")),
        (rt_cb.build_credential_stuffing, CredentialStuffingParams(target="https://target.corp/login", test_user="audit_user")),
        (rt_cb.build_privesc_check, PrivescCheckParams(target="10.10.10.50")),
        (rt_cb.build_defense_evasion, DefenseEvasionParams(target="10.10.10.50")),
        (rt_cb.build_lateral_movement, LateralMovementParams(target="10.10.10.50", dest_ip="10.10.10.51")),
        (rt_cb.build_data_discovery, DataDiscoveryParams(target="10.10.10.50")),
        (rt_cb.build_exfiltration_test, ExfiltrationTestParams(target="egress.target.corp")),
        (rt_cb.build_objective_impact, ObjectiveImpactParams(target="target.corp")),
    ]

    for builder, params in builders_and_params:
        cmds = builder(params)
        assert isinstance(cmds, list)
        assert len(cmds) >= 1
        for argv in cmds:
            assert isinstance(argv, list)
            assert all(isinstance(arg, str) for arg in argv)
            for arg in argv:
                assert not any(op in arg for op in [" | ", " ; ", " && ", " || "])

    empty_store = EvidenceStore()
    ok, reason = verify_completion(RED_TEAM_COMPLETION, {}, empty_store)
    assert not ok


# ---------------------------------------------------------------------------
# Test 21: Mobile App Action Registry & Command Builders
# ---------------------------------------------------------------------------
def test_mobile_action_registry_and_builders():
    from agents.mobile.actions import (
        build_mobile_registry,
        StaticAnalysisParams,
        DecompileAppParams,
        PermAuditParams,
        HardcodedSecretsParams,
        InsecureStorageParams,
        TlsPinningCheckParams,
        AuthTokenTestParams,
        DynamicInstrumentationParams,
        WebViewAuditParams,
        MobileReportParams,
    )
    import agents.mobile.command_builders as mob_cb
    from agents.mobile.completion import MOBILE_COMPLETION, verify_completion
    from core.investigation import EvidenceStore

    registry = build_mobile_registry()
    actions = registry.list_actions()
    assert len(actions) >= 10  # core 10 + shared primitives (http_probe, browser_probe; self_script when opted in)
    action_ids = [a.action_id for a in actions]
    assert "static_analysis" in action_ids
    assert "decompile_app" in action_ids
    assert "hardcoded_secrets" in action_ids

    builders_and_params = [
        (mob_cb.build_static_analysis, StaticAnalysisParams(target="app.apk")),
        (mob_cb.build_decompile_app, DecompileAppParams(target="app.apk")),
        (mob_cb.build_perm_audit, PermAuditParams(target="AndroidManifest.xml")),
        (mob_cb.build_hardcoded_secrets, HardcodedSecretsParams(target="src/")),
        (mob_cb.build_insecure_storage, InsecureStorageParams(target="com.example.app")),
        (mob_cb.build_tls_pinning_check, TlsPinningCheckParams(target="api.example.com")),
        (mob_cb.build_auth_token_test, AuthTokenTestParams(target="https://api.example.com")),
        (mob_cb.build_dynamic_instrumentation, DynamicInstrumentationParams(target="com.example.app")),
        (mob_cb.build_webview_audit, WebViewAuditParams(target="src/")),
        (mob_cb.build_mobile_report, MobileReportParams(target="com.example.app")),
    ]

    for builder, params in builders_and_params:
        cmds = builder(params)
        assert isinstance(cmds, list)
        assert len(cmds) >= 1
        for argv in cmds:
            assert isinstance(argv, list)
            assert all(isinstance(arg, str) for arg in argv)
            for arg in argv:
                assert not any(op in arg for op in [" | ", " ; ", " && ", " || "])

    empty_store = EvidenceStore()
    ok, reason = verify_completion(MOBILE_COMPLETION, {}, empty_store)
    assert not ok


# ---------------------------------------------------------------------------
# Test 22: Firewall Bypass Action Registry & Command Builders
# ---------------------------------------------------------------------------
def test_firewall_bypass_action_registry_and_builders():
    from agents.firewall_bypass.actions import (
        build_firewall_registry,
        WafFingerprintParams,
        RuleEnumParams,
        ChunkedEncodingParams,
        UnicodeNormalizationParams,
        HeaderMutationParams,
        ContentTypeConfusionParams,
        PathNormalizationParams,
        ParameterPollutionParams,
        PayloadMutationParams,
        WafBypassReportParams,
    )
    import agents.firewall_bypass.command_builders as waf_cb
    from agents.firewall_bypass.completion import FIREWALL_BYPASS_COMPLETION, verify_completion
    from core.investigation import EvidenceStore

    registry = build_firewall_registry()
    actions = registry.list_actions()
    assert len(actions) >= 10  # core 10 + shared primitives (http_probe, browser_probe; self_script when opted in)
    action_ids = [a.action_id for a in actions]
    assert "waf_fingerprint" in action_ids
    assert "chunked_encoding" in action_ids
    assert "payload_mutation" in action_ids

    builders_and_params = [
        (waf_cb.build_waf_fingerprint, WafFingerprintParams(target="https://protected.corp")),
        (waf_cb.build_rule_enum, RuleEnumParams(target="https://protected.corp")),
        (waf_cb.build_chunked_encoding, ChunkedEncodingParams(target="https://protected.corp")),
        (waf_cb.build_unicode_normalization, UnicodeNormalizationParams(target="https://protected.corp")),
        (waf_cb.build_header_mutation, HeaderMutationParams(target="https://protected.corp")),
        (waf_cb.build_content_type_confusion, ContentTypeConfusionParams(target="https://protected.corp")),
        (waf_cb.build_path_normalization, PathNormalizationParams(target="https://protected.corp")),
        (waf_cb.build_parameter_pollution, ParameterPollutionParams(target="https://protected.corp")),
        (waf_cb.build_payload_mutation, PayloadMutationParams(target="https://protected.corp")),
        (waf_cb.build_waf_bypass_report, WafBypassReportParams(target="https://protected.corp")),
    ]

    for builder, params in builders_and_params:
        cmds = builder(params)
        assert isinstance(cmds, list)
        assert len(cmds) >= 1
        for argv in cmds:
            assert isinstance(argv, list)
            assert all(isinstance(arg, str) for arg in argv)
            for arg in argv:
                assert not any(op in arg for op in [" | ", " ; ", " && ", " || "])

    empty_store = EvidenceStore()
    ok, reason = verify_completion(FIREWALL_BYPASS_COMPLETION, {}, empty_store)
    assert not ok


# ---------------------------------------------------------------------------
# Test 23: Security Research Action Registry & Command Builders
# ---------------------------------------------------------------------------
def test_security_research_action_registry_and_builders():
    from agents.security_research.actions import (
        build_research_registry,
        CveLookupParams,
        SourceDiffParams,
        RootCauseAnalysisParams,
        HarnessGenerationParams,
        BenignPocDevParams,
        ExploitabilityAuditParams,
        VariationFuzzingParams,
        RemediationGuideParams,
        ResearchWriteupParams,
        CvssCalculatorParams,
    )
    import agents.security_research.command_builders as res_cb
    from agents.security_research.completion import SECURITY_RESEARCH_COMPLETION, verify_completion
    from core.investigation import EvidenceStore

    registry = build_research_registry()
    actions = registry.list_actions()
    assert len(actions) >= 10  # core 10 + shared primitives (http_probe, browser_probe; self_script when opted in)
    action_ids = [a.action_id for a in actions]
    assert "cve_lookup" in action_ids
    assert "benign_poc_dev" in action_ids
    assert "cvss_calculator" in action_ids

    builders_and_params = [
        (res_cb.build_cve_lookup, CveLookupParams(target="CVE-2023-38606")),
        (res_cb.build_source_diff, SourceDiffParams(target="repo/")),
        (res_cb.build_root_cause_analysis, RootCauseAnalysisParams(target="CVE-2023-38606")),
        (res_cb.build_harness_generation, HarnessGenerationParams(target="kernel_driver")),
        (res_cb.build_benign_poc_dev, BenignPocDevParams(target="http://localhost:8080")),
        (res_cb.build_exploitability_audit, ExploitabilityAuditParams(target="target_binary")),
        (res_cb.build_variation_fuzzing, VariationFuzzingParams(target="http://localhost:8080")),
        (res_cb.build_remediation_guide, RemediationGuideParams(target="CVE-2023-38606")),
        (res_cb.build_research_writeup, ResearchWriteupParams(target="CVE-2023-38606")),
        (res_cb.build_cvss_calculator, CvssCalculatorParams(target="CVE-2023-38606")),
    ]

    for builder, params in builders_and_params:
        cmds = builder(params)
        assert isinstance(cmds, list)
        assert len(cmds) >= 1
        for argv in cmds:
            assert isinstance(argv, list)
            assert all(isinstance(arg, str) for arg in argv)
            for arg in argv:
                assert not any(op in arg for op in [" | ", " ; ", " && ", " || "])

    empty_store = EvidenceStore()
    ok, reason = verify_completion(SECURITY_RESEARCH_COMPLETION, {}, empty_store)
    assert not ok


# ---------------------------------------------------------------------------
# Test 24: All 9 Specialized Agents Autonomous-Ready
# ---------------------------------------------------------------------------
def test_all_9_agents_autonomous_ready():
    from agents import (
        CTFAgent,
        BugBountyAgent,
        PentestAgent,
        ActiveDirectoryAgent,
        CloudAgent,
        RedTeamAgent,
        MobileAppAgent,
        FirewallBypassAgent,
        SecurityResearchAgent,
    )
    from core.scope import Scope

    scope = Scope(target="10.10.10.5")
    agents = [
        CTFAgent(scope=scope, mode="automated"),
        BugBountyAgent(scope=scope, mode="automated"),
        PentestAgent(scope=scope, mode="automated"),
        ActiveDirectoryAgent(scope=scope, mode="automated"),
        CloudAgent(scope=scope, mode="automated"),
        RedTeamAgent(scope=scope, mode="automated"),
        MobileAppAgent(scope=scope, mode="automated"),
        FirewallBypassAgent(scope=scope, mode="automated"),
        SecurityResearchAgent(scope=scope, mode="automated"),
    ]

    assert len(agents) == 9
    for agent in agents:
        assert hasattr(agent, "get_action_registry"), f"{agent.agent_type} missing get_action_registry"
        assert hasattr(agent, "get_completion_criteria"), f"{agent.agent_type} missing get_completion_criteria"
        assert hasattr(agent, "run_autonomous_default"), f"{agent.agent_type} missing run_autonomous_default"
        registry = agent.get_action_registry()
        # >= 10: every agent has its 10 core actions; bug_bounty also carries the
        # detection engines + shared primitives added 2026-11, so it has more.
        assert len(registry.list_actions()) >= 10, f"{agent.agent_type} expected >=10 actions, got {len(registry.list_actions())}"
        criteria = agent.get_completion_criteria()
        assert criteria is not None, f"{agent.agent_type} criteria is None"
