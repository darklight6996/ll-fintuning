"""
agents/common/http_primitive.py

Framework-level HTTP-primitive ACTION wrapping core/tools/http_probe.py.

This is the shared action layer for the rich HTTP primitive: the ActionDefinition,
its typed params, the command builder (which invokes the engine script with the
CURRENT interpreter so it runs under the venv), and the fact extractor that reads
the engine's JSON verdict. Any agent lights it up with two lines:

    from agents.common.http_primitive import HTTP_PROBE, build_http_probe
    registry.register(HTTP_PROBE, build_http_probe)

and, in that agent's fact_extractors.extract_facts():

    from agents.common.http_primitive import extract_http_probe_facts
    if action_id == "http_probe":
        return extract_http_probe_facts(stdout, stderr)

The engine emits the same `vuln_signal` convention sqlmap/nuclei use, so the
verdict-aware evidence promotion in base_agent handles it with no extra code: a
confirmed oracle hit -> a real Finding; a clean probe or a mere "lead" (e.g. an
unencoded reflection, which is an XSS *surface*, not a confirmed bug) -> no
finding.
"""

import json
import os
import sys
from typing import List, Optional

from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition

# core/tools/http_probe.py lives at the repo root's core/tools/. abspath (not
# realpath) keeps the repo's rag/junction view and resolves agents/common/ ->
# repo root correctly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HTTP_PROBE_SCRIPT = os.path.join(_REPO_ROOT, "core", "tools", "http_probe.py")


class HttpProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(
        ...,
        description="Target URL to probe. Include a query string (e.g. "
                    "https://site/filter?category=Gifts) so there are parameters to tamper with.",
    )
    focus_param: Optional[str] = Field(
        None, description="Optional: tamper ONE named parameter (targeted mode) instead of the auto battery.")
    payload: Optional[str] = Field(
        None, description="Optional: the value to set focus_param to (targeted mode). Benign probes only.")


HTTP_PROBE = ActionDefinition(
    action_id="http_probe",
    phase="vuln_validation",
    description=("Surgical HTTP differential prober: baseline vs mutated requests with a SQL-error / "
                 "boolean-divergence / LFI-signature / open-redirect / reflection oracle. "
                 "Low-noise, benign, detection-only."),
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="low",
    params_model=HttpProbeParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_http_probe",
    timeout_seconds=90,
    max_retries=2,
    satisfies_phase=True,
)


def build_http_probe(params: HttpProbeParams) -> List[List[str]]:
    """Invoke the engine with the SAME interpreter running the agent (so it uses
    the venv), in auto mode by default or targeted mode when a focus param +
    payload are supplied."""
    argv = [sys.executable, HTTP_PROBE_SCRIPT, "--url", params.target]
    if params.focus_param and params.payload is not None:
        argv += ["--mode", "targeted", "--param", params.focus_param, "--payload", params.payload]
    else:
        argv += ["--mode", "auto"]
    return [argv]


def extract_http_probe_facts(stdout: str, stderr: str) -> dict:
    """Parse the engine's single-line JSON vuln_signal. Returns {} if the engine
    produced nothing parseable (which never promotes a finding)."""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "vuln_signal" in obj:
                return obj
    return {}
