"""
Red Team - Phase 1: Reconnaissance
Perimeter port discovery and service identification.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "recon"
requires_tools = ["nmap", "curl"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "127.0.0.1")
    commands_run = []

    if tools.is_tool_installed("nmap"):
        cmd = f"nmap -sT -T4 --top-ports 20 {target}"
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
    open_ports = []
    for line in raw.splitlines():
        if "/tcp" in line and "open" in line:
            port_part = line.split("/")[0].strip()
            if port_part.isdigit():
                open_ports.append(int(port_part))
    return {"open_ports": open_ports}


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "initial_access"

