"""
Bug Bounty - Phase 1: Reconnaissance
Passive OSINT, domain resolution, whois, and TLS/banner inspection.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "recon"
requires_tools = ["curl", "whois"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "localhost")
    commands_run = []
    
    # 1. WHOIS / Domain registration
    cmd1 = f"whois {target}" if tools.is_tool_installed("whois") else f"curl -sI https://{target}"
    res1 = executor.execute_command(cmd1, approved=True)
    commands_run.append(res1)

    # 2. HTTP header & TLS banner grab
    cmd2 = f"curl -sI -k https://{target}"
    res2 = executor.execute_command(cmd2, approved=True)
    commands_run.append(res2)

    raw_output = (res1.get("stdout", "") + "\n" + res2.get("stdout", "")).strip()
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
    findings = {"servers": [], "technologies": [], "domains": []}
    for line in raw.splitlines():
        if line.lower().startswith("server:"):
            srv = line.split(":", 1)[1].strip()
            if srv not in findings["servers"]:
                findings["servers"].append(srv)
        elif line.lower().startswith("x-powered-by:"):
            tech = line.split(":", 1)[1].strip()
            if tech not in findings["technologies"]:
                findings["technologies"].append(tech)
    return findings


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "subdomain_enum"

