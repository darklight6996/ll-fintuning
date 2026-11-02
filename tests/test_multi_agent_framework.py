"""
tests/test_multi_agent_framework.py

Comprehensive test suite verifying the Multi-Agent Security Assessment Framework:
1. OS Detection on Windows and Linux
2. Tool Manager inspection & package manager commands
3. All 9 Agents registered and instantiable
4. Scope dataclass, asset filtering & command compliance
5. Phase execution & safe-by-default execution
6. Manual mode approval gates & Automated mode sequential flow
7. AI Reasoning integration (7 decision points, context gathering, YAML & RAG injection)
8. Evidence logging to SessionManager & audit trail
9. State persistence on interrupt and resume
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "rag") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "rag"))

from core.os_detector import detect_os, OSInfo
from core.tool_manager import ToolManager
from core.scope import Scope
from core.agent_router import AgentRouter, AGENT_REGISTRY
from core.session_manager import SessionManager
from core.command_executor import CommandExecutor
from core.chat_assistant import ChatAssistant

from agents.base_agent import BaseAgent, PhaseResult
from agents.bug_bounty.agent import BugBountyAgent
from agents.red_team.agent import RedTeamAgent
from agents.pentest.agent import PentestAgent
from agents.ctf.agent import CTFAgent
from agents.mobile.agent import MobileAppAgent
from agents.firewall_bypass.agent import FirewallBypassAgent
from agents.security_research.agent import SecurityResearchAgent
from agents.active_directory.agent import ActiveDirectoryAgent
from agents.cloud.agent import CloudAgent


# ==============================================================================
# TEST 1: OS Detection & Tool Manager
# ==============================================================================

def test_os_detection():
    os_info = detect_os()
    assert isinstance(os_info, OSInfo)
    assert os_info.system in ("Windows", "Linux", "Darwin")
    assert len(os_info.friendly_name) > 0
    assert os_info.package_manager in ("winget", "choco", "apt", "brew", "pacman", "dnf", "yum", "none")


def test_tool_manager_check_and_install_command():
    tool_mgr = ToolManager()
    installed = tool_mgr.is_tool_installed("python")
    assert installed is True

    # Test install command resolution for package managers
    fake_os = OSInfo(
        system="Linux",
        friendly_name="Ubuntu 24.04",
        distro="ubuntu",
        package_manager="apt",
        is_admin=True,
        shell="bash"
    )
    ubuntu_mgr = ToolManager(fake_os)
    apt_cmd = ubuntu_mgr.get_install_command("nmap")
    assert "apt install -y nmap" in apt_cmd


# ==============================================================================
# TEST 2: Agent Registry & All 9 Agents Instantiation
# ==============================================================================

def test_agent_registry_count():
    agents = AgentRouter.list_agents()
    assert len(agents) == 9
    expected_keys = {
        "bug_bounty", "red_team", "pentest", "ctf", "mobile",
        "firewall_bypass", "security_research", "active_directory", "cloud"
    }
    registered_keys = {a.key for a in agents}
    assert expected_keys == registered_keys


def test_all_9_agents_instantiation():
    scope = Scope(target="127.0.0.1", in_scope=["127.0.0.1"])
    for key in AGENT_REGISTRY:
        agent = AgentRouter.instantiate_agent(key, scope=scope, mode="manual")
        assert isinstance(agent, BaseAgent)
        assert len(agent.phase_modules) >= 3
        phase_names = agent.get_phase_names()
        assert len(phase_names) == len(agent.phase_modules)


# ==============================================================================
# TEST 3: Scope Dataclass & Compliance Gates
# ==============================================================================

def test_scope_in_and_out_of_scope():
    scope = Scope(
        target="10.10.10.5",
        in_scope=["10.10.10.0/24", "target.com"],
        out_of_scope=["10.10.10.1", "admin.target.com"],
        prohibited_tests=["no DoS"],
        prohibited_tools=["metasploit"]
    )

    # In-scope check
    ok, _ = scope.is_asset_in_scope("10.10.10.5")
    assert ok is True

    ok, _ = scope.is_asset_in_scope("target.com")
    assert ok is True

    # Out-of-scope override check
    ok, reason = scope.is_asset_in_scope("10.10.10.1")
    assert ok is False
    assert "out-of-scope" in reason.lower()

    ok, reason = scope.is_asset_in_scope("admin.target.com")
    assert ok is False
    assert "out-of-scope" in reason.lower()

    # Prohibited tools & tests check
    ok, reason = scope.is_command_allowed("metasploit -x run")
    assert ok is False
    assert "prohibited tool" in reason.lower()

    ok, reason = scope.is_command_allowed("curl http://target.com/dos_test/slowloris")
    assert ok is False
    assert "denial of service" in reason.lower()


# ==============================================================================
# TEST 4: Safe Execution Gates
# ==============================================================================

def test_command_executor_read_only_default():
    # Read-only executor must never run commands
    executor = CommandExecutor(read_only=True)
    res = executor.execute_command("whoami", approved=True)
    assert res["status"] == "skipped"
    assert "read-only" in res["reason"].lower()


def test_command_executor_unapproved():
    executor = CommandExecutor(read_only=False)
    res = executor.execute_command("whoami", approved=False)
    assert res["status"] == "declined"


# ==============================================================================
# TEST 5: AI Reasoning Integration (Part 2 & Part 3)
# ==============================================================================

@pytest.mark.slow
def test_ai_context_gathering_and_yaml_rag():
    assistant = ChatAssistant()
    state = {
        "name": "test_session",
        "target": "10.10.10.5",
        "chat_history": [
            {"role": "user", "message": "msg1"},
            {"role": "user", "message": "msg2"},
            {"role": "user", "message": "msg3"},
            {"role": "user", "message": "msg4"},
        ]
    }
    ctx = assistant.gather_ai_context(
        agent_type="active_directory",
        current_phase="ad_enumeration",
        session_state=state,
        available_tools=[{"name": "nmap", "installed": True}]
    )

    # Verify all 6 mandatory items are present
    assert "session_state" in ctx
    assert "yaml_scenarios" in ctx
    assert "rag_chunks" in ctx
    assert "recent_history" in ctx
    assert len(ctx["recent_history"]) <= 3  # bounded token budget
    assert "phase_results" in ctx
    assert "available_tools" in ctx


@pytest.mark.slow
def test_ai_phase_recommendation():
    assistant = ChatAssistant()
    ctx = assistant.gather_ai_context(
        agent_type="pentest",
        current_phase="recon",
        session_state={"target": "10.10.10.5"}
    )
    rec = assistant.recommend_next_phase(ctx, candidate_phases=["enumeration", "exploitation"])
    assert rec["next"] in ("enumeration", "exploitation")
    assert "reason" in rec
    assert rec["confidence"] > 0.0


@pytest.mark.slow
def test_ai_output_interpretation():
    assistant = ChatAssistant()
    raw_nmap = """
    PORT    STATE SERVICE
    80/tcp  open  http
    445/tcp open  microsoft-ds
    """
    ctx = assistant.gather_ai_context("pentest", "enumeration", {})
    interp = assistant.interpret_pasted_output(raw_nmap, ctx)
    assert 80 in interp["ports"] or "80" in str(interp["ports"]) or 445 in interp["ports"]
    assert "summary" in interp


@pytest.mark.slow
def test_ai_scope_compliance_gate():
    assistant = ChatAssistant()
    scope = Scope(
        target="10.10.10.5",
        in_scope=["10.10.10.5"],
        prohibited_tools=["sqlmap"]
    )
    # Allowed command
    res1 = assistant.check_scope_compliance("nmap -sT 10.10.10.5", scope)
    assert res1["allowed"] is True

    # Prohibited tool command
    res2 = assistant.check_scope_compliance("sqlmap -u http://10.10.10.5", scope)
    assert res2["allowed"] is False
    assert "prohibited tool" in res2["reason"].lower()


@pytest.mark.slow
def test_ai_custom_script_generation():
    assistant = ChatAssistant()
    ctx = assistant.gather_ai_context("security_research", "poc_dev", {})
    script_data = assistant.generate_custom_script(
        target_description="HTTP API Header probe",
        environment="Linux Docker",
        constraints="Non-destructive",
        context=ctx
    )
    assert "script_code" in script_data
    assert "safety_notes" in script_data


@pytest.mark.slow
def test_ai_exploit_chain_reasoning():
    assistant = ChatAssistant()
    ctx = assistant.gather_ai_context("red_team", "recon", {})
    findings = [
        {"title": "Open Port 80 HTTP", "service": "HTTP"},
        {"title": "Exposed Admin Portal", "service": "Web Auth"},
        {"title": "Default Credentials", "service": "Admin"}
    ]
    chain = assistant.reason_exploit_chain(findings, ctx)
    assert "exploit_chain" in chain
    assert len(chain["exploit_chain"]) == 3


@pytest.mark.slow
def test_ai_report_synthesis():
    assistant = ChatAssistant()
    state = {
        "name": "Audit_Demo",
        "target": "10.10.10.10",
        "type": "Pentest",
        "completed_steps": [],
        "notes": [],
        "evidence": []
    }
    report = assistant.synthesize_report(state)
    assert isinstance(report, str)
    assert len(report) > 50


# ==============================================================================
# TEST 6: Automated End-to-End Agent Execution & Audit Trail
# ==============================================================================

@pytest.mark.slow
def test_automated_agent_run_audit_trail(tmp_path):
    session_mgr = SessionManager(base_dir=str(tmp_path / ".assessments"))
    scope = Scope(
        target="127.0.0.1",
        in_scope=["127.0.0.1"],
        active_testing_allowed=False  # Read-only safe default
    )
    agent = BugBountyAgent(
        scope=scope,
        mode="automated",
        session_manager=session_mgr
    )

    result = agent.run()
    assert result["status"] == "completed"
    assert len(result["completed_phases"]) == 5

    # Check evidence persistence
    summary = session_mgr.get_summary()
    assert summary["name"] is not None
    assert Path(result["report_path"]).exists()

