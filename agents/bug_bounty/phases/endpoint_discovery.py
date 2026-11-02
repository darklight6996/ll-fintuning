"""
Bug Bounty - Phase 3: Endpoint Discovery
Crawls and identifies API paths, robots.txt, sitemaps, and common web endpoints.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import PhaseResult

name = "endpoint_discovery"
requires_tools = ["curl"]


def run(session: Dict[str, Any], tools: Any, executor: Any, scope: Any) -> PhaseResult:
    target = scope.target or session.get("target", "localhost")
    commands_run = []

    # Check robots.txt and sitemap.xml
    cmd1 = f"curl -sL https://{target}/robots.txt"
    res1 = executor.execute_command(cmd1, approved=True)
    commands_run.append(res1)

    cmd2 = f"curl -sL https://{target}/sitemap.xml"
    res2 = executor.execute_command(cmd2, approved=True)
    commands_run.append(res2)

    raw_output = (res1.get("stdout", "") + "\n" + res2.get("stdout", "")).strip()
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
    endpoints = []
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("disallow:") or line.lower().startswith("allow:"):
            path = line.split(":", 1)[1].strip()
            if path and path not in endpoints:
                endpoints.append(path)
        elif "<loc>" in line:
            url = line.replace("<loc>", "").replace("</loc>", "").strip()
            endpoints.append(url)
    return {"endpoints": endpoints}


def next_phase_hint(result: Dict[str, Any]) -> Optional[str]:
    return "param_fuzzing"

