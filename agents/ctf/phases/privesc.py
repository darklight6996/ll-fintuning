"""
CTF - Phase 3: Privilege Escalation
Identifies sudo permissions, SUID root binaries, and writable system scripts.
"""

from typing import Dict, Any, List, Optional
import platform
from agents.base_agent import PhaseResult

name = "privesc"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    commands_run = []

    if platform.system() == "Windows":
        cmd = "whoami /groups"
    else:
        cmd = "sudo -l -n 2>/dev/null || find / -perm -4000 -type f 2>/dev/null | head -n 15"

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
    binaries = [l.strip() for l in raw.splitlines() if l.strip()]
    return {
        "suid_or_sudo_vectors": binaries[:10]
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "flag_capture"

