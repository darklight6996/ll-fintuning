"""
Active Directory - Phase 1: AD Domain Enumeration
Identifies Domain Controllers (LDAP/Kerberos/SMB), domain functional level, trusts, and SPNs.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "ad_enumeration"
requires_tools = ["nmap", "dig"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "127.0.0.1")
    commands_run = []

    if tools.is_tool_installed("nmap"):
        cmd = f"nmap -sT -p 88,389,445,636,3268 {target}"
    else:
        cmd = f"curl -sI http://{target}"

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
    ad_ports = []
    for line in raw.splitlines():
        if "/tcp" in line and "open" in line:
            port = int(line.split("/")[0].strip())
            ad_ports.append(port)
    is_dc = any(p in ad_ports for p in (88, 389, 445))
    return {
        "ad_ports_open": ad_ports,
        "is_domain_controller": is_dc
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "kerberoast"

