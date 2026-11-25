"""
agents/common/browser_exec_authoring.py

Shape D (client / browser-context) execution probe wrapping
core/tools/browser_exec_runner.py. The model authors a payload URL + a unique
MARKER canary (and optional JS detector / click trigger); the runner loads it in
headless Chromium, executes the page JS, and confirms XSS only if the canary
ACTUALLY RUNS in the target origin -- real DOM / reflected / stored XSS execution,
not a reflection-diff guess.

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
BROWSER_EXEC_RUNNER = os.path.join(_REPO_ROOT, "core", "tools", "browser_exec_runner.py")


class BrowserExecParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="The authorized in-scope URL to load, WITH your payload embedded "
                                         "(e.g. a reflected query param). The browser loads only this URL.")
    hypothesis: str = Field(..., description="One line: the client-side suspicion this confirms (DOM/reflected/stored XSS).")
    marker: str = Field(..., description="A unique canary string your BENIGN payload surfaces on execution, e.g. "
                                         "alert('MARKER') or console.log('MARKER'). Confirmed only if it actually runs.")
    detector_js: Optional[str] = Field(None, description="Optional JS expression evaluated AFTER load; a truthy result "
                                                         "confirms (e.g. reading a DOM property/sink you polluted).")
    click_selector: Optional[str] = Field(None, description="Optional CSS selector to click after load, to trigger an "
                                                            "event-driven sink (stored / DOM XSS).")


BROWSER_EXEC = ActionDefinition(
    action_id="browser_exec",
    phase="vuln_validation",
    description=("LOAD YOUR PAYLOAD IN A REAL BROWSER: author a URL carrying your payload plus a unique marker "
                 "canary (alert/console.log). Headless Chromium executes the page JS and confirms XSS ONLY if the "
                 "canary actually RUNS in the target origin -- real execution, not reflection. Use for DOM, "
                 "reflected and stored XSS and client-side sinks."),
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="high",
    params_model=BrowserExecParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_browser_exec",
    requires_suspicion=True,
    vuln_class=None,
    request_volume="low",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)


def build_browser_exec(params: BrowserExecParams) -> List[List[str]]:
    argv = [sys.executable, BROWSER_EXEC_RUNNER, "--url", params.target,
            "--marker", params.marker, "--timeout-ms", "15000"]
    if params.detector_js:
        argv += ["--detector-b64", base64.b64encode(params.detector_js.encode("utf-8", "replace")).decode("ascii")]
    if params.click_selector:
        argv += ["--click-selector", params.click_selector]
    return [argv]


def extract_browser_exec_facts(stdout: str, stderr: str) -> dict:
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
