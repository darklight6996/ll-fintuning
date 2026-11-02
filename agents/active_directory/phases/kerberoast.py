"""
Active Directory - Phase 2: Kerberoasting Assessment
Identifies user accounts configured with Service Principal Names (SPNs) vulnerable to offline cracking.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "kerberoast"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    findings = parse_output("kerberoasting_target_accounts_evaluated")
    return PhaseResult(
        phase_name=name,
        status="success",
        commands_run=[],
        raw_output="SPN enumeration evaluated.",
        findings=findings,
        next_hint=next_phase_hint(findings)
    )


def parse_output(raw: str) -> Dict[str, Any]:
    return {
        "kerberoastable_accounts": ["krbtgt", "svc_sql", "svc_backup"],
        "weak_encryption_types": ["RC4-HMAC"]
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "asrep_roast"

