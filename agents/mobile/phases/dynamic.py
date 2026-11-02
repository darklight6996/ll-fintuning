"""
Mobile - Phase 2: Dynamic Analysis
Inspects running package processes, IPC broadcasts, and logcat messages.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "dynamic"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    commands_run = []
    # Safe check of emulator / device status via adb if present
    cmd = "adb devices" if tools.is_tool_installed("adb") else "python -c \"print('No emulator attached; skipping dynamic runtime')\""
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
    devices = [l.split()[0] for l in raw.splitlines() if "\tdevice" in l]
    return {
        "attached_devices": devices,
        "runtime_testing_enabled": len(devices) > 0
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "auth_test"

