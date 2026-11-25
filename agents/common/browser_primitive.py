"""
agents/common/browser_primitive.py

Framework-level headless-browser COVERAGE action wrapping
core/tools/browser_probe.py.

Shared across all agents (same pattern as http_primitive):

    from agents.common.browser_primitive import BROWSER_PROBE, build_browser_probe
    registry.register(BROWSER_PROBE, build_browser_probe)

and in the agent's fact_extractors.extract_facts():

    from agents.common.browser_primitive import extract_browser_facts
    if action_id == "browser_probe":
        return extract_browser_facts(stdout, stderr)

The engine emits a coverage map (discovered_endpoints / forms / network_endpoints
/ js_endpoints / parameters) plus a coverage `vuln_signal` (confirmed=False, so
it never manufactures a finding). The coverage map lands in the observation's
parsed_facts and flows into the planner's "discovered facts" -- which is the
point: it expands the surface the planner reasons over, so later probes
(http_probe, sqli_deep_scan, dalfox...) can target endpoints and params that a
curl crawl never saw.
"""

import json
import os
import sys
from typing import List

from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BROWSER_PROBE_SCRIPT = os.path.join(_REPO_ROOT, "core", "tools", "browser_probe.py")


class BrowserProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="URL to render in a headless browser and harvest coverage from.")


BROWSER_PROBE = ActionDefinition(
    action_id="browser_probe",
    phase="endpoint_discovery",
    description=("Render the target in a headless browser and harvest full coverage: links, forms, "
                 "parameters, network (XHR/fetch) endpoints, and endpoints mined from JavaScript."),
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=BrowserProbeParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_browser_probe",
    timeout_seconds=150,  # shallow crawl visits several pages
    max_retries=2,
    satisfies_phase=True,
)


def build_browser_probe(params: BrowserProbeParams) -> List[List[str]]:
    return [[sys.executable, BROWSER_PROBE_SCRIPT, "--url", params.target]]


def extract_browser_facts(stdout: str, stderr: str) -> dict:
    """Parse the engine's JSON coverage object. Returns {} if nothing parseable."""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
    return {}
