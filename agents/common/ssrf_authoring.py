"""
agents/common/ssrf_authoring.py

Deterministic SSRF engine wrapper (core/tools/ssrf_probe_runner.py). The model
names a safe control value and the internal/metadata attack value plus a
content marker unique to the attack target; the engine computes the verdict,
never the model.

OPT-IN (shares self_script's switch): export PENTEST_ENABLE_SELF_SCRIPTING=1
Suspicion-gated: offered only once a hypothesis is open.
"""

import json
import os
import sys
from typing import List, Optional

from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SSRF_RUNNER = os.path.join(_REPO_ROOT, "core", "tools", "ssrf_probe_runner.py")


class SSRFProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="The authorized endpoint URL (or template, if body_template is given).")
    hypothesis: str = Field(..., description="One line: which param fetches a server-side URL and why.")
    param: Optional[str] = Field(None, description="The query/form param to substitute the control/attack value "
                                                   "into (mutually exclusive with body_template).")
    body_template: Optional[str] = Field(None, description="A raw request body with a literal {VALUE} placeholder, "
                                                           "for JSON/XML-bodied SSRF (overrides param).")
    control_value: str = Field(..., description="A SAFE, real external URL/host the param should legitimately be "
                                                 "able to fetch (e.g. https://example.com/).")
    attack_value: str = Field(..., description="The internal/metadata target under test (e.g. "
                                               "http://169.254.169.254/latest/meta-data/, http://127.0.0.1:PORT/...).")
    expect_marker: Optional[str] = Field(None, description="A content signature unique to the attack target "
                                                           "(e.g. 'ami-id' for cloud metadata, or text unique to "
                                                           "an internal admin page). Required for a PROVEN "
                                                           "verdict -- without it the engine can only report a lead.")
    content_type: Optional[str] = Field("application/json", description="Content-Type when using body_template.")
    method: Optional[str] = Field("GET", description="HTTP method (default GET).")


SSRF_PROBE = ActionDefinition(
    action_id="ssrf_probe",
    phase="vuln_validation",
    description=("TEST SSRF reliably: name a safe control URL vs an internal/metadata attack URL for a "
                 "URL-fetching param, plus a content marker unique to the attack target. Confirms ONLY if "
                 "the control request succeeds AND the marker appears in the attack response but not the "
                 "control's -- a bare status/body difference or a connection timeout is explicitly NOT proof "
                 "by itself. Use for params that are URL/host-shaped or named like a fetch/webhook/callback."),
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="high",
    params_model=SSRFProbeParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_ssrf_probe",
    requires_suspicion=True,
    vuln_class="ssrf",
    request_volume="low",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)


def build_ssrf_probe(params: SSRFProbeParams) -> List[List[str]]:
    argv = [sys.executable, SSRF_RUNNER, "--target", params.target,
            "--method", params.method or "GET",
            "--control-value", params.control_value, "--attack-value", params.attack_value,
            "--content-type", params.content_type or "application/json"]
    if params.body_template:
        argv += ["--body-template", params.body_template]
    elif params.param:
        argv += ["--param", params.param]
    if params.expect_marker:
        argv += ["--expect-marker", params.expect_marker]
    return [argv]


def extract_ssrf_probe_facts(stdout: str, stderr: str) -> dict:
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
