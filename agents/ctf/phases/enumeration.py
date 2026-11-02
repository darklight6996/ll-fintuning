"""
CTF - Phase 2: Local Enumeration
Searches for challenge artifacts, readable flags, SUID binaries, and cron jobs.
"""

from typing import Dict, Any, List, Optional
import platform
from agents.base_agent import PhaseResult

name = "enumeration"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    commands_run = []

    if platform.system() == "Windows":
        cmd = "dir /s /b *flag* *pass* *secret*"
    else:
        cmd = "find / -name '*flag*' -type f 2>/dev/null | head -n 10"

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
    flags = [l.strip() for l in raw.splitlines() if l.strip()]
    return {
        "candidate_flag_files": flags
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "privesc"

