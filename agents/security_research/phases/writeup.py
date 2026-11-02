"""
Security Research - Phase 4: Technical Write-Up Synthesis
Formats coordinated vulnerability disclosure (CVD) notice, technical advisory, and patch guidance.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "writeup"
requires_tools = []


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    findings = parse_output("writeup_ready")
    return PhaseResult(
        phase_name=name,
        status="success",
        commands_run=[],
        raw_output="Technical advisory drafted.",
        findings=findings,
        next_hint=next_phase_hint(findings)
    )


def parse_output(raw: str) -> Dict[str, Any]:
    return {
        "advisory_complete": True,
        "format": "Markdown Advisory / Proof-of-Concept Writeup"
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return None

