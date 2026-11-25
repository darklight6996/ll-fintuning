"""
agents/common/self_scripting.py

Framework-level SELF-SCRIPTING action wrapping core/tools/script_runner.py: the
agent writes a short Python probe/PoC and the sandboxed runner executes it.

OPT-IN ONLY. Because this runs model-written code, it is the highest-risk
action, so it is registered ONLY when the operator explicitly enables it:

    export PENTEST_ENABLE_SELF_SCRIPTING=1

Use `self_scripting_enabled()` to gate registration (see
build_bug_bounty_registry). When the flag is unset the action does not exist, so
the planner can never propose it.

Benign-PoC framing: the probe should CONFIRM exploitability non-destructively
(timing, out-of-band callback, unique-token echo) and print one JSON line
{"vuln_signal": {...}}. The runner statically screens the code and runs it under
tight resource limits with no filesystem-write; see script_runner.py for the
full boundary.
"""

import base64
import json
import os
import sys
from typing import List, Optional

from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_RUNNER = os.path.join(_REPO_ROOT, "core", "tools", "script_runner.py")

_ENABLE_ENV = "PENTEST_ENABLE_SELF_SCRIPTING"


def self_scripting_enabled() -> bool:
    """True only when the operator has explicitly opted in via env var."""
    return os.environ.get(_ENABLE_ENV) == "1"


class SelfScriptParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="The authorized in-scope URL/host the probe addresses.")
    hypothesis: str = Field(..., description="One line: what this probe is trying to confirm or refute.")
    code: str = Field(
        ...,
        description=(
            "A short, self-contained Python 3 probe using the stdlib "
            "(urllib, http.client, socket, ssl, json, time, threading, concurrent.futures, asyncio) "
            "OR the requests library (all are available). It must be "
            "non-destructive: no file writes, no process spawning, no data "
            "exfiltration. CONCURRENCY / RACE CONDITIONS: to test timing windows or "
            "state invariants (e.g. single-use coupon, inventory, checkout, balance), use "
            "threading (threading.Thread or concurrent.futures.ThreadPoolExecutor) or asyncio "
            "to burst N parallel requests simultaneously and verify if the invariant is violated. "
            "CONFIRMATION RIGOR: send a CONTROL that should fail plus "
            "the real payload / parallel burst, and only set confirmed=true if the control fails AND "
            "the payload yields a DISTINCTIVE positive or an OBSERVABLE INVARIANT VIOLATION "
            "(e.g. single-use resource consumed twice, limit exceeded, negative total; never 'responses differ', a "
            "bare status code, or generic page text). Print exactly one JSON line: a "
            "vuln_signal object with keys confirmed (bool), vuln_class, severity, "
            "summary, and evidence (the distinctive proof you captured, quoted). "
            "A confirmed signal WITHOUT evidence is recorded only as a LEAD, not a "
            "finding. The authorized target is also in the TARGET env variable. "
            "REQUEST FORMAT MATTERS: if your control request ALSO errors identically to "
            "your attack request, that is not a refutation -- it usually means the request "
            "FORMAT is wrong, not the hypothesis. A backend XML parser error on a form-encoded "
            "or JSON POST (signatures: 'SAXParseException', 'Content is not allowed in prolog', "
            "'Entities are not allowed', 'DOCTYPE is disallowed') means the endpoint expects a "
            "raw XML body (Content-Type: application/xml or text/xml) -- rewrite the control AND "
            "attack to send well-formed XML matching the field names you observed (e.g. "
            "<stockCheck><productId>1</productId><storeId>1</storeId></stockCheck>), with the "
            "attack payload injected via an XML entity/DOCTYPE in the suspect field for an "
            "XXE-driven test, before concluding the hypothesis is refuted."
        ),
    )


SELF_SCRIPT = ActionDefinition(
    action_id="self_script",
    phase="vuln_validation",
    description=("WRITE YOUR OWN TEST: author a short, sandboxed, non-destructive Python probe that "
                 "confirms or refutes ONE specific suspicion you have already formed as a hypothesis "
                 "(e.g. substitute an internal URL into a URL-shaped parameter to test SSRF; replay an "
                 "object id to test IDOR; launch concurrent parallel requests with threading to test race "
                 "conditions and single-use/limit invariants). This is the universal weapon for any bug "
                 "class the fixed tools don't cover -- the test is the payload/logic YOU supply, not a canned scanner."),
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="high",
    params_model=SelfScriptParams,
    allowed_target_kinds=["url", "domain", "ip"],
    command_builder="build_self_script",
    # Suspicion-gated (2026-11-16): this is the generic weapon -- it unlocks the
    # moment ANY hypothesis is open (vuln_class=None), i.e. once the model has
    # actually formed a suspicion to test. It is never offered on a bare target
    # with no suspicion, which is exactly "suspect the vuln before firing".
    requires_suspicion=True,
    vuln_class=None,
    timeout_seconds=40,
    max_retries=4,  # the model often needs 2-3 tries to get a probe right
    satisfies_phase=True,
)


def build_self_script(params: SelfScriptParams) -> List[List[str]]:
    b64 = base64.b64encode((params.code or "").encode("utf-8")).decode("ascii")
    return [[sys.executable, SCRIPT_RUNNER, "--target", params.target, "--code-b64", b64, "--timeout", "20"]]


def extract_self_script_facts(stdout: str, stderr: str) -> dict:
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
