"""
agents/common/xxe_authoring.py

Deterministic XXE engine wrapper (core/tools/xxe_probe_runner.py). The model
supplies the control/attack XML bodies it has observed the app actually wants
(from browser_probe/recon_scan's own evidence -- e.g. a `*.js` filename
containing "xml"/"Xml" is the clue that should drive this, not a form-encoded
guess) and a content marker to look for; the engine computes the verdict.

OPT-IN (shares self_script's switch): export PENTEST_ENABLE_SELF_SCRIPTING=1
Suspicion-gated: offered only once a hypothesis is open.
"""

import base64
import json
import os
import sys
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XXE_RUNNER = os.path.join(_REPO_ROOT, "core", "tools", "xxe_probe_runner.py")


class XXEProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="The authorized XML-accepting endpoint URL.")
    hypothesis: str = Field(..., description="One line: what the entity should disclose/reach if vulnerable.")
    control_body: str = Field(..., description="A WELL-FORMED XML request body matching the app's real schema "
                                               "(no DOCTYPE/entity) -- must be a body this endpoint ACTUALLY "
                                               "accepts, or the verdict can never be proven.")
    attack_body: str = Field(..., description="The SAME XML shape, with a DOCTYPE/external or internal entity "
                                              "injected (e.g. <!DOCTYPE foo [<!ENTITY xxe SYSTEM "
                                              "\"file:///etc/hostname\">]>) whose value is referenced in a field "
                                              "that gets echoed back or otherwise disclosed.")
    expect_marker: Optional[str] = Field(None, description="A content signature that ONLY appears if the entity "
                                                           "resolved (e.g. the known content of the file/URL "
                                                           "referenced). Required for a PROVEN verdict -- without "
                                                           "it the engine can only report a lead.")
    content_type: Optional[str] = Field("application/xml", description="Content-Type header (default application/xml).")
    method: Optional[str] = Field("POST", description="HTTP method (default POST).")
    headers: Optional[Dict[str, str]] = Field(None, description="Extra headers (e.g. a required Accept header).")


XXE_PROBE = ActionDefinition(
    action_id="xxe_probe",
    phase="vuln_validation",
    description=("TEST XXE reliably: supply the app's real control XML body plus an attack body carrying an "
                 "entity, and a content marker that would only appear if the entity resolved. Confirms ONLY if "
                 "the control succeeds (proves the body format is right) AND the marker appears in the attack "
                 "response but not the control's. Use for XML-accepting endpoints (Content-Type: application/xml, "
                 "or a *.js resource literally named with 'xml'/'Xml')."),
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="high",
    params_model=XXEProbeParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_xxe_probe",
    requires_suspicion=True,
    vuln_class="xxe",
    request_volume="low",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)


def build_xxe_probe(params: XXEProbeParams) -> List[List[str]]:
    argv = [sys.executable, XXE_RUNNER, "--target", params.target,
            "--method", params.method or "POST",
            "--control-body-b64", base64.b64encode(params.control_body.encode("utf-8")).decode("ascii"),
            "--attack-body-b64", base64.b64encode(params.attack_body.encode("utf-8")).decode("ascii"),
            "--content-type", params.content_type or "application/xml"]
    if params.expect_marker:
        argv += ["--expect-marker", params.expect_marker]
    if params.headers:
        argv += ["--headers-b64", base64.b64encode(json.dumps(params.headers).encode("utf-8")).decode("ascii")]
    return [argv]


def extract_xxe_probe_facts(stdout: str, stderr: str) -> dict:
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
