"""
Firewall Bypass - Phase 2: Rule Enumeration
Sends benign probe patterns (HTML tags, SQL keywords, path traversal tokens) to map WAF blocking triggers.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "rule_enumeration"
requires_tools = ["curl"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "localhost")
    commands_run = []

    # Safe rule trigger probes
    cmd = f"curl -s -o /dev/null -w \"%{'{http_code}'}\" \"https://{target}/?test=%3Cscript%3E\""
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
    blocked = code in ("403", "406", "429", "503")
    return {
        "xss_rule_active": blocked,
        "trigger_response_code": code
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "bypass_techniques"

