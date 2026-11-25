"""
agents/common/raw_http_authoring.py

Framework-level RAW-HTTP action (Shape C: protocol / parser desync) wrapping
core/tools/raw_http_runner.py. The agent WRITES THE EXACT REQUEST BYTES; the
runner sends them over a raw socket to one authorized target and reports a
timing/marker verdict. This is the capability a normal HTTP client cannot give:
byte-level control for HTTP request smuggling, cache poisoning, and Host-header
routing -- classes that need no injectable parameter at all.

OPT-IN (shares self_script's switch): export PENTEST_ENABLE_SELF_SCRIPTING=1
Suspicion-gated: offered only once a hypothesis is open.
"""

import base64
import json
import os
import sys
from typing import List, Optional

from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_RUNNER = os.path.join(_REPO_ROOT, "core", "tools", "raw_http_runner.py")


class RawHttpParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="The authorized in-scope URL/host. The socket connects ONLY here.")
    hypothesis: str = Field(..., description="One line: the desync/smuggling/routing suspicion this confirms.")
    payload_raw: str = Field(..., description=(
        "The EXACT raw HTTP request bytes to send, with CRLF line endings, including the request "
        "line, headers and body. For smuggling, craft the Content-Length / Transfer-Encoding "
        "disagreement here. Sent verbatim -- this is the test."))
    control_raw: Optional[str] = Field(None, description=(
        "Optional well-formed baseline request (CRLF endings). The runner confirms a TIMING DESYNC "
        "only when this control returns fast AND the payload stalls -- the strongest detection-only "
        "smuggling signal. Always provide one for a timing test."))
    expect_marker: Optional[str] = Field(None, description=(
        "Optional distinctive string. If it appears in the response, the crafted/smuggled bytes were "
        "processed -> strong confirmation."))


RAW_HTTP = ActionDefinition(
    action_id="raw_http",
    phase="vuln_validation",
    description=("SEND YOUR OWN RAW HTTP BYTES over a socket: author the exact request (CRLF) to detect a "
                 "front-end/back-end desync (request smuggling), cache / Host-header routing, or anything "
                 "below the request-line abstraction. Pair a well-formed control with the crafted payload; "
                 "a timing stall on the payload but not the control is the proof."),
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="high",
    params_model=RawHttpParams,
    allowed_target_kinds=["url", "domain", "ip"],
    command_builder="build_raw_http",
    requires_suspicion=True,
    vuln_class=None,
    request_volume="low",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)


def build_raw_http(params: RawHttpParams) -> List[List[str]]:
    argv = [sys.executable, RAW_RUNNER, "--target", params.target,
            "--payload-b64", base64.b64encode((params.payload_raw or "").encode("utf-8", "replace")).decode("ascii"),
            "--timeout", "10"]
    if params.control_raw:
        argv += ["--control-b64", base64.b64encode(params.control_raw.encode("utf-8", "replace")).decode("ascii")]
    if params.expect_marker:
        argv += ["--marker", params.expect_marker]
    return [argv]


def extract_raw_http_facts(stdout: str, stderr: str) -> dict:
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
