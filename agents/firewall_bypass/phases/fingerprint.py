"""
Firewall Bypass - Phase 1: WAF Fingerprinting
Detects Web Application Firewalls (Cloudflare, AWS WAF, ModSecurity, Imperva, Akamai) via signature headers.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "fingerprint"
requires_tools = ["curl"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "localhost")
    commands_run = []

    cmd = f"curl -sI -k https://{target}"
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
    raw_lower = raw.lower()
    waf_detected = "None"
    if "cf-ray" in raw_lower or "__cfduid" in raw_lower or "cloudflare" in raw_lower:
        waf_detected = "Cloudflare"
    elif "awselb" in raw_lower or "x-amz-" in raw_lower:
        waf_detected = "AWS WAF / ALB"
    elif "mod_security" in raw_lower or "modsecurity" in raw_lower:
        waf_detected = "ModSecurity"
    elif "x-akamai" in raw_lower:
        waf_detected = "Akamai"
    return {
        "detected_waf": waf_detected,
        "is_behind_waf": waf_detected != "None"
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "rule_enumeration"

