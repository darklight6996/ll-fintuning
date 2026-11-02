"""
Red Team - Phase 2: Initial Access
Evaluates ingress vectors, exposed admin portals, and external authentication interfaces.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "initial_access"
requires_tools = ["curl"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "127.0.0.1")
    commands_run = []

    # Safe probe of admin / login endpoints
    cmd = f"curl -sL -o /dev/null -w \"%{'{http_code}'}\" http://{target}/login"
    res = executor.execute_command(cmd, approved=True)
    commands_run.append(res)
    raw_output = res.get("stdout", "")
    findings = parse_output(raw_output)

    return PhaseResult(
        phase_name=name,
        status="success",
        commands_run=commands_run,
        raw_output=raw_output,
        findings=findings,
        next_hint=next_phase_hint(findings)
    )


def parse_output(raw: str) -> Dict[str, Any]:
    code = raw.strip()
    has_portal = code in ("200", "301", "302", "401", "403")
    return {
        "auth_portal_accessible": has_portal,
        "http_status": code
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "privesc"

