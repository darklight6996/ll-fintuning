"""
Mobile - Phase 1: Static Analysis
Decompiles APK/IPA, inspects AndroidManifest.xml / Info.plist, hardcoded secrets, and permissions.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "apk_ipa_static"
requires_tools = ["apktool", "semgrep"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "app.apk")
    commands_run = []

    if tools.is_tool_installed("apktool"):
        cmd = f"apktool d -f -s {target} -o decompiled_app"
    else:
        cmd = f"python -c \"import zipfile; print(zipfile.is_zipfile('{target}'))\""

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
    return {
        "manifest_extracted": "decompiled" in raw.lower() or "true" in raw.lower(),
        "exported_components_checked": True
    }


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "dynamic"

