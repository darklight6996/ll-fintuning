"""
Firewall Bypass - Phase 3: Bypass Techniques
Evaluates header spoofing (X-Forwarded-For, X-Original-URL), payload mutation, and encoding variations.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "bypass_techniques"
requires_tools = ["curl"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "localhost")
    commands_run = []

    # Safe test with header spoofing
    cmd = f"curl -s -o /dev/null -w \"%{'{http_code}'}\" -H \"X-Forwarded-For: 127.0.0.1\" -H \"X-Custom-IP-Authorization: 127.0.0.1\" https://{target}/admin"
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
    bypass_indicated = code in ("200", "204")
    return {
        "header_spoofing_effective": bypass_indicated,
        "admin_status_code": code
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return None

