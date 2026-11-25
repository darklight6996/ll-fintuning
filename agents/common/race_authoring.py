"""
agents/common/race_authoring.py

Shape B reliable engine: RACE / LIMIT-OVERRUN probe wrapping
core/tools/race_probe_runner.py. The model supplies the state-changing endpoint
and its form params; the engine handles robust CSRF/session priming and runs a
SEQUENTIAL control vs a CONCURRENT test, confirming an overrun objectively
(concurrency beats the limit the control respected). This replaces the fragile
hand-rolled CSRF+threading the model kept failing on in self_script.

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
RACE_RUNNER = os.path.join(_REPO_ROOT, "core", "tools", "race_probe_runner.py")


class RaceProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="The authorized state-changing endpoint URL (e.g. .../cart, .../cart/coupon).")
    hypothesis: str = Field(..., description="One line: the limited resource / invariant you think concurrency can break.")
    form_params: Dict[str, Any] = Field(..., description="The POST body params to send, WITHOUT csrf (the engine primes and "
                                                         "injects the CSRF token itself), e.g. {\"productId\":1,\"quantity\":1,\"redir\":\"/cart\"} "
                                                         "or {\"coupon\":\"PROMO20\"}.")
    prime_url: Optional[str] = Field(None, description="Optional page to GET first to establish the session and read the CSRF "
                                                       "token (defaults to target). Use the page that renders the form.")
    success_marker: Optional[str] = Field(None, description="Optional string whose presence in the response means the state-change "
                                                            "SUCCEEDED (e.g. a discount/total string). Used to count successes; "
                                                            "falls back to HTTP status. Strongly recommended for coupon/redeem actions.")
    count: Optional[int] = Field(20, description="How many requests to fire sequentially and then concurrently (2-40).")
    method: Optional[str] = Field("POST", description="HTTP method (default POST).")


RACE_PROBE = ActionDefinition(
    action_id="race_probe",
    phase="vuln_validation",
    description=("TEST A RACE / LIMIT-OVERRUN reliably: name the state-changing endpoint + its form params and the engine "
                 "primes the session, auto-extracts the CSRF token, then fires a SEQUENTIAL control vs a CONCURRENT burst. "
                 "It confirms ONLY if concurrency produces more successes than the sequential control (the overrun). Use for "
                 "single-use coupons/gift cards, purchase/stock limits, balance overdraft, double-spend."),
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="high",
    params_model=RaceProbeParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_race_probe",
    requires_suspicion=True,
    vuln_class="race-condition",
    request_volume="high",
    timeout_seconds=90,
    max_retries=2,
    satisfies_phase=True,
)


def build_race_probe(params: RaceProbeParams) -> List[List[str]]:
    argv = [sys.executable, RACE_RUNNER, "--target", params.target,
            "--params-b64", base64.b64encode(json.dumps(params.form_params or {}).encode("utf-8")).decode("ascii"),
            "--count", str(params.count or 20), "--method", (params.method or "POST")]
    if params.prime_url:
        argv += ["--prime-url", params.prime_url]
    if params.success_marker:
        argv += ["--success-marker", params.success_marker]
    return [argv]


def extract_race_probe_facts(stdout: str, stderr: str) -> dict:
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
