"""
CTF - Phase 4: Flag Capture Verification
Reads and parses target flag strings (e.g. HTB{...}, flag{...}, pwn.college{...}).
"""

import re
from typing import Dict, Any, List, Optional
import platform
from agents.base_agent import PhaseResult

name = "flag_capture"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    commands_run = []

    if platform.system() == "Windows":
        cmd = "type flag.txt 2>nul || type C:\\flag.txt 2>nul"
    else:
        cmd = "cat flag.txt 2>/dev/null || cat /flag 2>/dev/null || cat /challenge/.flag 2>/dev/null"

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
    flags = re.findall(r"(?:flag|pwn\.college|htb|thm|ctf)\{[^\}\s]+\}", raw, re.IGNORECASE)
    # Also detect 32-character bandit hashes
    hashes = re.findall(r"\b[a-zA-Z0-9]{32,33}\b", raw)
    detected_flags = flags + hashes

    return {
        "flags_captured": detected_flags,
        "challenge_complete": len(detected_flags) > 0
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return None

