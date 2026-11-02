"""
Bug Bounty - Phase 2: Subdomain Enumeration
Enumerates authorized target subdomains via subfinder or DNS lookup.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "subdomain_enum"
requires_tools = ["subfinder", "dig"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "localhost")
    commands_run = []

    if tools.is_tool_installed("subfinder"):
        cmd = f"subfinder -d {target} -silent"
    elif tools.is_tool_installed("dig"):
        cmd = f"dig +short {target} NS"
    else:
        cmd = f"nslookup {target}"

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
    subdomains = []
    for line in raw.splitlines():
        line = line.strip()
        if "." in line and not line.startswith(";"):
            subdomains.append(line)
    return {"subdomains": list(set(subdomains))}


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "endpoint_discovery"

