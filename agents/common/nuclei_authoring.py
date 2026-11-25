"""
agents/common/nuclei_authoring.py

Framework-level NUCLEI-TEMPLATE AUTHORING action wrapping
core/tools/nuclei_template_runner.py: the agent WRITES ITS OWN nuclei YAML
detection template for a specific suspicion and the runner executes it
(detection-only) against one authorized target.

This is the second half of "the hunter writes its own tests" (self_script is the
first): where self_script is arbitrary Python, this lets the model express a
test in nuclei's declarative request/matcher language -- the format most web
detections are already written in, and the one the RAG corpus is full of.

OPT-IN ONLY (shares self_script's switch, since both run model-authored tests):

    export PENTEST_ENABLE_SELF_SCRIPTING=1

Suspicion-gated like self_script: offered only once a hypothesis is open.
"""

import base64
import json
import os
import sys
from typing import List

from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition
from agents.common.self_scripting import self_scripting_enabled  # shared opt-in switch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUCLEI_RUNNER = os.path.join(_REPO_ROOT, "core", "tools", "nuclei_template_runner.py")


class NucleiTemplateParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="The authorized in-scope URL/host the template runs against.")
    hypothesis: str = Field(..., description="One line: the suspicion this template is meant to confirm.")
    template: str = Field(
        ...,
        description=(
            "A complete, self-contained nuclei YAML template (detection-only: "
            "id, info, and an http request block with matchers). Request/matcher "
            "only -- no code/javascript protocol, no interactsh/OOB. It should "
            "match ONLY when the suspected vulnerability is actually present (a "
            "distinctive matcher, not a generic status/word match -- a loose matcher "
            "yields a LEAD, not a proven finding)."
        ),
    )


NUCLEI_TEMPLATE = ActionDefinition(
    action_id="nuclei_template",
    phase="vuln_validation",
    description=("WRITE YOUR OWN nuclei template: author a detection-only nuclei YAML (request + "
                 "matchers) for ONE specific suspicion the built-in templates don't cover, and run "
                 "it against the target. The test is the template YOU write."),
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="high",
    params_model=NucleiTemplateParams,
    allowed_target_kinds=["url", "domain", "ip"],
    command_builder="build_nuclei_template",
    # Suspicion-gated generic weapon: unlocks once any hypothesis is open.
    requires_suspicion=True,
    vuln_class=None,
    request_volume="high",
    timeout_seconds=180,
    max_retries=1,
    satisfies_phase=True,
)


def build_nuclei_template(params: NucleiTemplateParams) -> List[List[str]]:
    b64 = base64.b64encode((params.template or "").encode("utf-8")).decode("ascii")
    return [[sys.executable, NUCLEI_RUNNER, "--target", params.target,
             "--template-b64", b64, "--timeout", "150"]]


def extract_nuclei_template_facts(stdout: str, stderr: str) -> dict:
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
