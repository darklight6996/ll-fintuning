"""
Red Team - Phase 4: Lateral Movement
Discovers adjacent internal subnets, active sessions, and reachable network neighbors.
"""

from typing import Dict, Any, List, Optional
import platform
from agents.base_agent import PhaseResult

name = "lateral"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    commands_run = []

    cmd = "arp -a" if platform.system() == "Windows" else "ip neighbor show || arp -a"
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
    neighbors = []
    for line in raw.splitlines():
        parts = line.split()
        if parts and any(c.isdigit() for c in parts[0]) and "." in parts[0]:
            neighbors.append(parts[0])
    return {"network_neighbors": neighbors}


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "impact"

