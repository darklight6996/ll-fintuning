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


def build_bug_bounty_registry() -> ActionRegistry:
    """Build and return the ActionRegistry for BugBountyAgent."""
    registry = ActionRegistry()
    registry.register(SUBDOMAIN_ENUM)
    registry.register(PORT_SCAN)
    registry.register(ENDPOINT_DISCOVERY)
    registry.register(DIR_BRUTE)
    registry.register(PARAM_FUZZ)
    registry.register(HEADER_ANALYSIS)
    registry.register(IDOR_PROBE)
    registry.register(SQLI_PROBE)
    registry.register(XSS_PROBE)
    registry.register(VULN_VALIDATION)
    return registry
