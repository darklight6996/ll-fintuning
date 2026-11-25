"""
agents/common/recon_primitive.py

Framework-level RECON ACTION wrapping core/tools/recon_scan_runner.py.

Same shape as agents/common/http_primitive.py / browser_primitive.py: the
ActionDefinition, its typed params, the command builder (invokes the engine
with the CURRENT interpreter so it runs under the venv), and the fact
extractor that reads the engine's JSON facts bundle. Any agent lights it up
with two lines:

    from agents.common.recon_primitive import RECON_SCAN, build_recon_scan
    registry.register(RECON_SCAN, build_recon_scan)

and, in that agent's fact_extractors.extract_facts():

    from agents.common.recon_primitive import extract_recon_scan_facts
    if action_id == "recon_scan":
        return extract_recon_scan_facts(stdout, stderr)

This is a COVERAGE action (vuln_class=None, requires_suspicion=False, same
posture as header_analysis/endpoint_discovery): it surveys surface -- DNS,
open ports, TLS posture, lightweight tech fingerprint (with version capture),
common auth paths, leaked-secret patterns, exposed sensitive files,
real-IP-behind-CDN discovery, and (only when explicitly asked) subdomains +
takeover detection. `dns`/`ports`/`tls`/`tech`/`real_ip` never promote a
Finding at all (pure facts); `secrets`/`exposed_files`/`takeovers` hits DO get
promoted, but ONLY ever to strength="lead" (never "proven" -- see
agents/base_agent.py::_promote_lead_findings_from_recon). See work.md's
2026-11-23 and 2026-11-24 ARTEX gap analysis / merge plan entries for the
rationale and what was deliberately excluded.
"""

import json
import os
import sys
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from core.action_contract import ActionDefinition

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECON_SCAN_SCRIPT = os.path.join(_REPO_ROOT, "core", "tools", "recon_scan_runner.py")

# Modes available when the caller is NOT restricted to a single pinned host
# (domain-wide engagement). "subdomains" is its own opt-in on top of this --
# see RECON_MODES_SINGLE_HOST below for the single-host subset.
#
# secrets/exposed_files/real_ip (2026-11-24 ARTEX merge round 2) are all
# target-only -- real_ip's MX/header checks run against the ONE pinned host,
# same as tls/tech/authsurface, so it's safe for single-host too. Only
# "ports" (whole-host port sweep) and "subdomains" (whole-domain, plus the
# takeover check that depends on it) stay excluded from single-host -- same
# reasoning as the existing SUBDOMAIN_ENUM/PORT_SCAN exclusion.
#
# js_recon/auth_sweep (2026-11-25 4-repo harvest, asset 5, Claude-BugHunter-
# derived) are ALSO target-only: js_recon mines the homepage + same-origin JS
# bundles of the ONE pinned host, and auth_sweep discovers/probes an OpenAPI
# spec served by that same host -- neither depends on domain-wide enumeration,
# so both are safe for single-host.
RECON_MODES_DEFAULT = "dns,ports,tls,tech,authsurface,secrets,exposed_files,real_ip,js_recon,auth_sweep"
RECON_MODES_SINGLE_HOST = "dns,tls,tech,authsurface,secrets,exposed_files,real_ip,js_recon,auth_sweep"


class ReconScanParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target URL or bare host to survey.")
    modes: Optional[Union[str, List[str]]] = Field(
        None,
        description="Optional override: comma-separated subset of "
                    "dns,ports,tls,tech,authsurface,subdomains,secrets,exposed_files,real_ip. "
                    "Leave unset for the default sweep. Ignored entirely on a single-host "
                    "engagement -- the framework always uses its own fixed, narrower mode set "
                    "there regardless of what you pass here.",
    )

    @field_validator("modes", mode="before")
    @classmethod
    def _join_list_modes(cls, v):
        # A model will sometimes naturally propose this as a JSON array
        # (observed live, 2026-11-23 -- rejected at Gate 2, self-corrected to
        # a string on retry) even though the field is documented as
        # comma-separated. Accept both instead of burning a retry on it.
        if isinstance(v, list):
            return ",".join(str(x) for x in v)
        return v


RECON_SCAN = ActionDefinition(
    action_id="recon_scan",
    phase="recon",
    description=(
        "Breadth-first parallel recon sweep: DNS records, open ports (risk-annotated), "
        "TLS/certificate posture, versioned tech fingerprint, common auth-surface paths, "
        "leaked-secret patterns (masked), exposed sensitive files, and real-IP-behind-CDN "
        "discovery -- all in one pass. Read-only, detection-only, coverage action (never "
        "confirms a vulnerability itself -- secrets/exposed_files hits surface only as "
        "unverified leads). Run this EARLY to map the whole visible surface before "
        "committing to one hypothesis."
    ),
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=ReconScanParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_recon_scan",
    request_volume="high",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
    # Coverage action: always offered, never gated behind a prior suspicion --
    # same posture as header_analysis / endpoint_discovery.
    vuln_class=None,
    requires_suspicion=False,
)


def build_recon_scan(params: ReconScanParams) -> List[List[str]]:
    """Invoke the engine with the SAME interpreter running the agent (so it
    uses the venv). Mode selection: an explicit params.modes override from the
    AI wins; otherwise default to the full set. Use ONLY for domain-wide
    engagements (the full BugBountyAgent registry) -- a single pinned host
    must use build_recon_scan_single_host below instead."""
    modes = params.modes or RECON_MODES_DEFAULT
    argv = [sys.executable, RECON_SCAN_SCRIPT, "--url", params.target, "--modes", modes]
    return [argv]


def build_recon_scan_single_host(params: ReconScanParams) -> List[List[str]]:
    """Same engine, but the mode set is CLAMPED here, deterministically, to
    RECON_MODES_SINGLE_HOST -- never to whatever the AI asked for in
    params.modes. 'ports' and 'subdomains' are whole-domain/whole-host probes
    outside the one URL instance the operator actually authorized (same
    reasoning as the existing SUBDOMAIN_ENUM/PORT_SCAN exclusion in
    agents/bug_bounty/single_host_mode.py); the framework decides this, not
    the model. register() this builder instead of build_recon_scan() wherever
    Scope.single_host_only=True."""
    argv = [sys.executable, RECON_SCAN_SCRIPT, "--url", params.target, "--modes", RECON_MODES_SINGLE_HOST]
    return [argv]


def extract_recon_scan_facts(stdout: str, stderr: str) -> dict:
    """Parse the engine's single-line JSON facts bundle. Returns {} if the
    engine produced nothing parseable."""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "recon" in obj:
                return obj
    return {}
