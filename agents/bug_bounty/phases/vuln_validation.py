"""
Bug Bounty - Phase 5: Vulnerability Validation
Validates identified security misconfigurations, missing headers (HSTS, CSP), and CORS policies.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "vuln_validation"
requires_tools = ["curl"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "localhost")
    commands_run = []

    # Check security headers and CORS
    cmd = f"curl -sI -k -H \"Origin: https://evil.com\" https://{target}"
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
    headers = raw.lower()
    missing_headers = []
    if "strict-transport-security" not in headers:
        missing_headers.append("Strict-Transport-Security (HSTS)")
    if "content-security-policy" not in headers:
        missing_headers.append("Content-Security-Policy (CSP)")
    if "x-frame-options" not in headers:
        missing_headers.append("X-Frame-Options (Clickjacking)")

    cors_misconfig = "access-control-allow-origin: https://evil.com" in headers or "access-control-allow-origin: *" in headers

    return {
        "missing_security_headers": missing_headers,
        "cors_misconfiguration": cors_misconfig,
        "vulnerabilities": [f"Missing {h}" for h in missing_headers] + (["Permissive CORS Origin"] if cors_misconfig else [])
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return None

