"""
agents/bug_bounty/actions.py

Action definitions and registry factory for BugBountyAgent.
Covers web reconnaissance, subdomain discovery, parameter fuzzing, and vulnerability triage.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition, ActionRegistry


# ---------------------------------------------------------------------------
# Typed Action Parameter Models (Pydantic v2, extra="forbid")
# ---------------------------------------------------------------------------

class SubdomainEnumParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target domain or root domain (e.g. example.com)")
    timeout_seconds: int = Field(60, description="Enumeration timeout in seconds")


class PortScanParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target domain or host")
    ports: str = Field("80,443,8080,8443,8000,8888", description="Comma-separated port list")


class EndpointDiscoveryParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Base target URL (e.g. https://example.com)")
    depth: int = Field(2, description="Crawl depth")


class DirBruteParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Base target URL")
    wordlist: str = Field("common.txt", description="Wordlist identifier or file path")
    extensions: str = Field("php,html,js,json,txt", description="File extensions to fuzz")


class ParamFuzzParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target host or base URL")
    endpoint: str = Field(..., description="Target URI endpoint (e.g. /api/v1/user)")
    param_names: str = Field("id,user,file,redirect,url,page", description="Comma-separated parameter names")


class HeaderAnalysisParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target URL for header and CORS inspection")


class IdorProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Base target URL")
    endpoint: str = Field(..., description="Endpoint with object identifier (e.g. /api/user)")
    user_a_id: str = Field(..., description="Authorized identifier for User A")
    user_b_id: str = Field(..., description="Target identifier for User B to test cross-tenant access")


class SqliProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Base target URL")
    endpoint: str = Field(..., description="Endpoint taking the query parameter")
    test_param: str = Field(..., description="Parameter name to probe safely")


class XssProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Base target URL")
    endpoint: str = Field(..., description="Endpoint taking input")
    param_name: str = Field(..., description="Parameter name to test for reflection")


class VulnValidationParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Base target URL")
    endpoint: str = Field(..., description="Endpoint being validated")
    vuln_type: str = Field(..., description="Identified vulnerability class (e.g. idor, sqli, cors)")
    payload: str = Field(..., description="Non-destructive benign test probe")


class SqliDeepScanParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(
        ...,
        description=(
            "Full target URL to test for SQL injection, ideally including a query "
            "string with the parameter(s) to test, e.g. "
            "https://site/filter?category=Gifts. sqlmap tests the parameters in "
            "this URL."
        ),
    )
    test_parameter: Optional[str] = Field(
        None,
        description=(
            "Optional: focus the scan on a single named parameter (sqlmap -p). "
            "Leave unset to let sqlmap test every parameter it finds in the URL."
        ),
    )
    # NOTE: deliberately NO knobs for --level/--risk/--technique/--dump/etc. The
    # command builder hardcodes a safe, detection-only, non-destructive flag set
    # so the planner can never request a data-dumping or data-modifying sqlmap
    # run. See agents/bug_bounty/command_builders.py::build_sqli_deep_scan.


class NucleiScanParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(
        ...,
        description="Target URL or host to scan with nuclei's templated checks (CVEs, misconfigurations, exposures).",
    )
    # No template/severity/fuzzing knobs exposed: the builder hardcodes a safe,
    # detection-only template posture (no -dast/fuzzing) and a severity floor.


class DalfoxScanParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(
        ...,
        description="Target URL (ideally with a query string) to test for XSS. dalfox mines parameters itself.",
    )


# ---------------------------------------------------------------------------
# Action Definitions
# ---------------------------------------------------------------------------

SUBDOMAIN_ENUM = ActionDefinition(
    action_id="subdomain_enum",
    phase="subdomain_enum",
    description="Enumerate subdomains via DNS queries and passive sources",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=SubdomainEnumParams,
    allowed_target_kinds=["domain", "url"],
    command_builder="build_subdomain_enum",
    request_volume="high",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)

PORT_SCAN = ActionDefinition(
    action_id="port_scan",
    phase="recon",
    description="Scan target host for open web and API ports",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=PortScanParams,
    allowed_target_kinds=["ip", "domain"],
    command_builder="build_port_scan",
    request_volume="high",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)

ENDPOINT_DISCOVERY = ActionDefinition(
    action_id="endpoint_discovery",
    phase="endpoint_discovery",
    description="Discover web endpoints, routes, and links on the target",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=EndpointDiscoveryParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_endpoint_discovery",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

DIR_BRUTE = ActionDefinition(
    action_id="dir_brute",
    phase="endpoint_discovery",
    description="Brute-force directories and endpoints using a wordlist",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=DirBruteParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_dir_brute",
    request_volume="high",
    timeout_seconds=120,
    max_retries=1,
    satisfies_phase=True,
)

PARAM_FUZZ = ActionDefinition(
    action_id="param_fuzz",
    phase="param_fuzzing",
    description="Fuzz HTTP parameters on an endpoint to find hidden or unlinked inputs",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=ParamFuzzParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_param_fuzz",
    timeout_seconds=60,
    max_retries=1,
    satisfies_phase=True,
)

HEADER_ANALYSIS = ActionDefinition(
    action_id="header_analysis",
    phase="recon",
    description="Analyze response headers, CSP, CORS origins, and security flags",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=HeaderAnalysisParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_header_analysis",
    timeout_seconds=20,
    max_retries=2,
    satisfies_phase=True,
)

IDOR_PROBE = ActionDefinition(
    action_id="idor_probe",
    requires_suspicion=True,  # suspicion-gated: only offered once a matching hypothesis exists
    vuln_class="idor",
    phase="vuln_validation",
    description="Test cross-tenant access to object identifiers on API endpoints",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=IdorProbeParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_idor_probe",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

SQLI_PROBE = ActionDefinition(
    action_id="sqli_probe",
    requires_suspicion=True,  # suspicion-gated: only offered once a matching hypothesis exists
    vuln_class="sqli",
    phase="vuln_validation",
    description="Send benign syntax probes to detect SQL syntax distortion or errors",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=SqliProbeParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_sqli_probe",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

XSS_PROBE = ActionDefinition(
    action_id="xss_probe",
    requires_suspicion=True,  # suspicion-gated: only offered once a matching hypothesis exists
    vuln_class="xss",
    phase="vuln_validation",
    description="Send benign alphanumeric canary to test reflection and contextual encoding",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=XssProbeParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_xss_probe",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

VULN_VALIDATION = ActionDefinition(
    action_id="vuln_validation",
    requires_suspicion=True,  # suspicion-gated: only offered once a matching hypothesis exists
    phase="vuln_validation",
    description="Execute a targeted, non-destructive probe to confirm or refute a vulnerability hypothesis",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=VulnValidationParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_vuln_validation",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

# First specialist DETECTION ENGINE wired in as an action (2026-11-14): sqlmap
# does deep, multi-technique SQLi detection (boolean/error/union/time-based
# blind) that the shallow single-payload sqli_probe curl cannot -- this is the
# "use a real technique the model can't reproduce by hand, read its structured
# output, reason confirmed/refuted" capability. state_change="none" and the
# hardcoded safe flag set in build_sqli_deep_scan make it strictly
# detection-only and non-destructive; the verdict-aware evidence promotion in
# base_agent._promote_evidence_if_warranted turns a CONFIRMED injection into a
# real high-severity Finding and a CLEAN scan into a recorded non-finding.
SQLI_DEEP_SCAN = ActionDefinition(
    action_id="sqli_deep_scan",
    requires_suspicion=True,  # suspicion-gated: only offered once a matching hypothesis exists
    vuln_class="sqli",
    phase="vuln_validation",
    description="Run sqlmap for deep, multi-technique SQL injection detection on a URL's parameters",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=SqliDeepScanParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_sqli_deep_scan",
    request_volume="high",
    timeout_seconds=300,
    max_retries=1,
    satisfies_phase=True,
)

# Second detection engine (2026-11-14): nuclei brings templated breadth --
# thousands of community checks for known CVEs, misconfigurations, default
# creds, and exposures -- the breadth layer complementing sqlmap's SQLi depth.
# Detection-only (no fuzzing/-dast templates); the verdict-aware promotion maps
# a matched template's severity straight onto the Finding, and a no-match scan
# becomes a recorded non-finding.
NUCLEI_SCAN = ActionDefinition(
    action_id="nuclei_scan",
    requires_suspicion=True,  # suspicion-gated: only offered once a matching hypothesis exists
    phase="vuln_validation",
    description="Run nuclei templated detection for known CVEs, misconfigurations, and exposures",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=NucleiScanParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_nuclei_scan",
    request_volume="high",
    timeout_seconds=300,
    max_retries=1,
    satisfies_phase=True,
)

# Third detection engine (2026-11-16): dalfox -- real XSS detection/confirmation
# (reflected, DOM, verified) that the shallow canary xss_probe cannot do. Sends
# many payloads (request_volume="high", gated under stealth). The verdict-aware
# promotion turns a VERIFIED XSS into a Finding and a clean scan into a
# non-finding; reflected-only hits are reported as leads.
DALFOX_SCAN = ActionDefinition(
    action_id="dalfox_scan",
    requires_suspicion=True,  # suspicion-gated: only offered once a matching hypothesis exists
    vuln_class="xss",
    phase="vuln_validation",
    description="Run dalfox for deep XSS detection (reflected / DOM / verified) on a URL's parameters",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=DalfoxScanParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_dalfox_scan",
    request_volume="high",
    timeout_seconds=300,
    max_retries=1,
    satisfies_phase=True,
)


def build_bug_bounty_registry() -> ActionRegistry:
    """Build and return the ActionRegistry for BugBountyAgent."""
    registry = ActionRegistry()
    from agents.bug_bounty.command_builders import (
        build_dir_brute,
        build_endpoint_discovery,
        build_header_analysis,
        build_idor_probe,
        build_param_fuzz,
        build_port_scan,
        build_nuclei_scan,
        build_dalfox_scan,
        build_sqli_deep_scan,
        build_sqli_probe,
        build_subdomain_enum,
        build_vuln_validation,
        build_xss_probe,
    )

    registry.register(SUBDOMAIN_ENUM, build_subdomain_enum)
    registry.register(PORT_SCAN, build_port_scan)
    registry.register(ENDPOINT_DISCOVERY, build_endpoint_discovery)
    registry.register(DIR_BRUTE, build_dir_brute)
    registry.register(PARAM_FUZZ, build_param_fuzz)
    registry.register(HEADER_ANALYSIS, build_header_analysis)
    registry.register(IDOR_PROBE, build_idor_probe)
    registry.register(SQLI_PROBE, build_sqli_probe)
    registry.register(XSS_PROBE, build_xss_probe)
    registry.register(VULN_VALIDATION, build_vuln_validation)
    registry.register(SQLI_DEEP_SCAN, build_sqli_deep_scan)
    registry.register(NUCLEI_SCAN, build_nuclei_scan)
    registry.register(DALFOX_SCAN, build_dalfox_scan)
    # Framework-level shared primitives (cross-agent).
    from agents.common.http_primitive import HTTP_PROBE, build_http_probe
    registry.register(HTTP_PROBE, build_http_probe)
    from agents.common.browser_primitive import BROWSER_PROBE, build_browser_probe
    registry.register(BROWSER_PROBE, build_browser_probe)
    # Breadth-first recon sweep (DNS/ports/TLS/tech/authsurface, parallel,
    # read-only coverage). Domain-wide engagement -> the full mode set.
    from agents.common.recon_primitive import RECON_SCAN, build_recon_scan
    registry.register(RECON_SCAN, build_recon_scan)
    # Self-scripting is OPT-IN (runs model-written code): registered only when
    # the operator sets PENTEST_ENABLE_SELF_SCRIPTING=1.
    from agents.common.self_scripting import SELF_SCRIPT, build_self_script, self_scripting_enabled
    if self_scripting_enabled():
        registry.register(SELF_SCRIPT, build_self_script)
    # bounty_platform_recon is OPT-IN (calls a third-party platform with the
    # operator's own credentials): registered only when the operator sets
    # PENTEST_ENABLE_BOUNTY_PLATFORM=1.
    from agents.common.bounty_platform_primitive import (
        BOUNTY_PLATFORM_RECON, build_bounty_platform_recon, bounty_platform_enabled,
    )
    if bounty_platform_enabled():
        registry.register(BOUNTY_PLATFORM_RECON, build_bounty_platform_recon)
    return registry
