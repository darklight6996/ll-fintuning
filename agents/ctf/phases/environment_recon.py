"""
CTF - Phase 1: Environment Reconnaissance
Determines current shell user, groups, filesystem layout, and environment variables.
"""

from typing import Dict, Any, List, Optional
import platform
from agents.base_agent import PhaseResult

name = "environment_recon"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    commands_run = []

    cmd = "whoami && cd && dir" if platform.system() == "Windows" else "whoami && pwd && ls -la"
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
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    user = lines[0] if lines else "unknown"
    files = [l for l in lines[1:] if not l.startswith("total")]
    return {
        "current_user": user,
        "directory_entries": files[:15]
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "enumeration"

