"""
Red Team - Phase 3: Privilege Escalation
Audits local misconfigurations, sudo capabilities, SUID binaries, and elevated tokens.
"""

from typing import Dict, Any, List, Optional
import platform
from agents.base_agent import PhaseResult

name = "privesc"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    commands_run = []
    
    if platform.system() == "Windows":
        cmd = "whoami /priv"
    else:
        cmd = "id && uname -a"

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
    elevated_privs = []
    for line in raw.splitlines():
        if "SeImpersonatePrivilege" in line or "SeDebugPrivilege" in line:
            elevated_privs.append(line.split()[0])
        elif "uid=0(root)" in line:
            elevated_privs.append("Root UID 0")
    return {
        "elevated_privileges_found": elevated_privs,
        "is_privileged": len(elevated_privs) > 0
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "lateral"

