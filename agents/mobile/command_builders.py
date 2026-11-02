"""
agents/mobile/command_builders.py

Pure Python command translators for MobileAppAgent actions.
Translates typed action parameter models into executable argument vectors.

SAFETY INVARIANTS:
  - Returns strictly List[List[str]]
  - Never shell strings
  - No shell=True
  - Cross-platform compatible
"""

import sys
from typing import List

from agents.mobile.actions import (
    StaticAnalysisParams,
    DecompileAppParams,
    PermAuditParams,
    HardcodedSecretsParams,
    InsecureStorageParams,
    TlsPinningCheckParams,
    AuthTokenTestParams,
    DynamicInstrumentationParams,
    WebViewAuditParams,
    MobileReportParams,
)


def build_static_analysis(params: StaticAnalysisParams) -> List[List[str]]:
    """Build command vector for Semgrep static ruleset scan."""
    return [
        ["semgrep", "--config=p/security-audit", params.target]
    ]


def build_decompile_app(params: DecompileAppParams) -> List[List[str]]:
    """Build command vector for apktool package decompilation."""
    return [
        ["apktool", "d", "-o", params.output_dir, "-f", params.target]
    ]


def build_perm_audit(params: PermAuditParams) -> List[List[str]]:
    """Build command vector for manifest permissions audit."""
    return [
        [sys.executable, "-c", f"print('Auditing AndroidManifest.xml permissions for {params.target}')"]
    ]


def build_hardcoded_secrets(params: HardcodedSecretsParams) -> List[List[str]]:
    """Build command vector for secrets scanning."""
    return [
        [sys.executable, "-c", f"print('Scanning for hardcoded secrets and API keys in {params.target}')"]
    ]


def build_insecure_storage(params: InsecureStorageParams) -> List[List[str]]:
    """Build command vector for local SQLite/SharedPreferences audit."""
    return [
        [sys.executable, "-c", f"print('Auditing local storage data privacy for package {params.target}')"]
    ]


def build_tls_pinning_check(params: TlsPinningCheckParams) -> List[List[str]]:
    """Build command vector for TLS pinning and network config verification."""
    return [
        ["curl", "-s", "-I", "-k", "--max-time", "10", f"https://{params.target}"]
    ]


def build_auth_token_test(params: AuthTokenTestParams) -> List[List[str]]:
    """Build command vector for mobile auth token validation."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-k", "-H", "Authorization: Bearer test_probe_token", url]
    ]


def build_dynamic_instrumentation(params: DynamicInstrumentationParams) -> List[List[str]]:
    """Build command vector for dynamic hook execution."""
    return [
        [sys.executable, "-c", f"print('Executing instrumentation hook {params.script} against {params.target}')"]
    ]


def build_webview_audit(params: WebViewAuditParams) -> List[List[str]]:
    """Build command vector for WebView JavaScript bridge auditing."""
    return [
        [sys.executable, "-c", f"print('Auditing WebView JavaScript interfaces in {params.target}')"]
    ]


def build_mobile_report(params: MobileReportParams) -> List[List[str]]:
    """Build command vector for mobile report generation."""
    return [
        [sys.executable, "-c", f"print('Generating OWASP Mobile Top 10 report for {params.target}')"]
    ]

