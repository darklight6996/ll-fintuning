"""
agents/common/bounty_platform_primitive.py

Framework-level ACTION wrapping core/tools/bounty_platform.py (2026-11-25
4-repo harvest, asset 7 -- re-implemented from pentest-agents' mcp-bounty-
server, see work.md).

OPT-IN, like self_scripting.py: this is the ONLY action in the framework that
calls a THIRD-PARTY SaaS platform using the operator's own real account
credentials rather than testing the engagement target, so it is registered
ONLY when the operator has explicitly set PENTEST_ENABLE_BOUNTY_PLATFORM=1 --
use `bounty_platform_enabled()` to gate registration (mirrors
self_scripting_enabled()'s pattern exactly). Even then, the underlying tool
refuses to do anything unless HACKERONE_USERNAME/HACKERONE_TOKEN are ALSO set
-- two separate opt-ins, not one.

allowed_target_kinds=["local"]: the "target" here is a HackerOne program
handle, not an asset to test, so this deliberately bypasses the in-scope
check the same way agents/ctf/actions.py's filesystem-only actions do --
there is nothing to check it against (fetching scope IS the point).
network_effect="read_only" (not "active"): the network call is to the
platform's own API, never to the engagement target, so it is correctly
exempt from the active_network_allowed gate.
"""

import json
import os
import sys
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOUNTY_PLATFORM_SCRIPT = os.path.join(_REPO_ROOT, "core", "tools", "bounty_platform.py")

_ENABLE_ENV = "PENTEST_ENABLE_BOUNTY_PLATFORM"


def bounty_platform_enabled() -> bool:
    """True only when the operator has explicitly opted in via env var."""
    return os.environ.get(_ENABLE_ENV) == "1"


class BountyPlatformParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Program handle/slug as it appears in the platform URL (not a URL to test).")
    platform: str = Field("hackerone", description="Bug-bounty platform ID. Only 'hackerone' is supported today.")
    mode: Literal["scope", "policy", "sync"] = Field(
        "sync",
        description="'scope' fetches in/out-of-scope assets, 'policy' fetches policy text + safe-harbor status, "
                    "'sync' fetches both and writes local bounty_scope.yaml/bounty_policy.md files.",
    )
    target_dir: Optional[str] = Field(None, description="Directory to write files to in 'sync' mode. Defaults to cwd.")


BOUNTY_PLATFORM_RECON = ActionDefinition(
    action_id="bounty_platform_recon",
    phase="recon",
    description=(
        "Fetch a bug-bounty program's in-scope/out-of-scope assets and policy text from the platform's "
        "own API (read-only). Use BEFORE testing to confirm what is actually in scope and what the "
        "program's rules of engagement are."
    ),
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="medium",
    params_model=BountyPlatformParams,
    allowed_target_kinds=["local"],
    command_builder="build_bounty_platform_recon",
    timeout_seconds=60,
    satisfies_phase=True,
)


def build_bounty_platform_recon(params: BountyPlatformParams) -> List[List[str]]:
    argv = [
        sys.executable, BOUNTY_PLATFORM_SCRIPT,
        "--mode", params.mode,
        "--platform", params.platform,
        "--program", params.target,
    ]
    if params.mode == "sync":
        argv += ["--target-dir", params.target_dir or "."]
    return [argv]


def extract_bounty_platform_facts(stdout: str, stderr: str) -> dict:
    """Parse the engine's single-line JSON result. Returns {} if the engine
    produced nothing parseable -- this is a pure informational fact (like
    recon_scan's dns/ports/tls), never a Finding source."""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "bounty" in obj:
                return obj
    return {}
