"""
agents/bug_bounty/command_builders.py

Pure Python command translators for BugBountyAgent actions.
Translates typed action parameter models into executable argument vectors.

SAFETY INVARIANTS:
  - Returns strictly List[List[str]]
  - Never shell strings
  - No shell=True
  - Cross-platform compatible
"""

import sys
from typing import List

from agents.bug_bounty.actions import (
    SubdomainEnumParams,
    PortScanParams,
    EndpointDiscoveryParams,
    DirBruteParams,
    ParamFuzzParams,
    HeaderAnalysisParams,
    IdorProbeParams,
    SqliProbeParams,
    XssProbeParams,
    VulnValidationParams,
)


def build_subdomain_enum(params: SubdomainEnumParams) -> List[List[str]]:
    """Build command vector for subdomain discovery."""
    return [
        ["subfinder", "-d", params.target, "-silent", "-timeout", str(params.timeout_seconds)]
    ]


def build_port_scan(params: PortScanParams) -> List[List[str]]:
    """Build command vector for port scanning web services."""
    return [
        ["nmap", "-sS", "-sV", "-p", params.ports, "--open", params.target]
    ]


def build_endpoint_discovery(params: EndpointDiscoveryParams) -> List[List[str]]:
    """Build command vector to crawl links and endpoints."""
    return [
        ["curl", "-s", "-L", "-k", "--max-time", "15", params.target]
    ]


def build_dir_brute(params: DirBruteParams) -> List[List[str]]:
    """Build command vector for directory/endpoint fuzzing."""
    return [
        ["ffuf", "-u", f"{params.target.rstrip('/')}/FUZZ", "-w", params.wordlist, "-e", params.extensions, "-mc", "200,204,301,302,307,401,403", "-silent"]
    ]


def build_param_fuzz(params: ParamFuzzParams) -> List[List[str]]:
    """Build command vector for parameter fuzzing."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-G", url, "-d", "test=probe"]
    ]


def build_header_analysis(params: HeaderAnalysisParams) -> List[List[str]]:
    """Build command vector for inspecting response headers and CORS."""
    return [
        ["curl", "-s", "-I", "-k", "-H", "Origin: https://evil.corp", params.target]
    ]


def build_idor_probe(params: IdorProbeParams) -> List[List[str]]:
    """Build command vector for testing unauthorized cross-tenant object access."""
    url_b = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}/{params.user_b_id}"
    return [
        ["curl", "-s", "-k", "-H", f"X-User-ID: {params.user_a_id}", url_b]
    ]


def build_sqli_probe(params: SqliProbeParams) -> List[List[str]]:
    """Build command vector for benign SQL syntax distortion probe."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-G", url, "--data-urlencode", f"{params.test_param}=' OR '1'='1"]
    ]


def build_xss_probe(params: XssProbeParams) -> List[List[str]]:
    """Build command vector for benign canary reflection probe."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-G", url, "--data-urlencode", f"{params.param_name}=canary_test_probe_xyz"]
    ]


def build_vuln_validation(params: VulnValidationParams) -> List[List[str]]:
    """Build command vector to confirm/refute vulnerability hypothesis."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-v", url, "-d", params.payload]
    ]

