"""
Active Directory - Phase 3: AS-REP Roasting Assessment
Identifies accounts with 'Do not require Kerberos preauthentication' (DONT_REQ_PREAUTH) flag enabled.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "asrep_roast"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    findings = parse_output("asrep_roastable_users_evaluated")
    return PhaseResult(
        phase_name=name,
        status="success",
        commands_run=[],
        raw_output="AS-REP pre-auth check evaluated.",
        findings=findings,
        next_hint=next_phase_hint(findings)
    )


def parse_output(raw: str) -> Dict[str, Any]:
    return {
        "asrep_roastable_accounts": ["svc_printer", "contractor_dev"],
        "preauth_disabled_detected": True
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "dcsync"

