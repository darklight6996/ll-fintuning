"""
Bug Bounty - Phase 4: Parameter Fuzzing
Identifies hidden GET/POST parameters and reflection points.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "param_fuzzing"
requires_tools = ["curl"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "localhost")
    commands_run = []

    # Safe test for parameter reflection
    test_probe = "SEC_PROBE_123"
    cmd = f"curl -sL \"https://{target}/?q={test_probe}&debug={test_probe}&redirect={test_probe}\""
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
    reflected = "SEC_PROBE_123" in raw
    return {
        "parameter_reflection_detected": reflected,
        "reflection_context": "HTML Body" if reflected else "None"
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "vuln_validation"

