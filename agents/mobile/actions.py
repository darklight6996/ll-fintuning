"""
agents/mobile/actions.py

Action definitions and registry factory for MobileAppAgent.
Covers mobile application security: APK/IPA static decompilation, manifest auditing,
hardcoded secrets discovery, local data storage, TLS pinning, and dynamic instrumentation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition, ActionRegistry


# ---------------------------------------------------------------------------
# Typed Action Parameter Models (Pydantic v2, extra="forbid")
# ---------------------------------------------------------------------------

class StaticAnalysisParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Path to APK or IPA package file")


class DecompileAppParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Path to package file")
    output_dir: str = Field("decompiled", description="Destination directory for decompiled source")


class PermAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Path to decompiled package or manifest")


class HardcodedSecretsParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Path to source directory or decompiled package")


class InsecureStorageParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target app package identifier (e.g. com.example.app)")


class TlsPinningCheckParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target app package or endpoint domain")


class AuthTokenTestParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target mobile backend API URL")
    endpoint: str = Field("/api/v1/auth", description="Auth endpoint to test")


class DynamicInstrumentationParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target package name or process")
    script: str = Field("bypass_pinning.js", description="Instrumentation script name")


class WebViewAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Path to decompiled app source")


class MobileReportParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target app package identifier")
    output_format: str = Field("markdown", description="Report format (markdown, json)")


# ---------------------------------------------------------------------------
# Action Definitions
# ---------------------------------------------------------------------------

STATIC_ANALYSIS = ActionDefinition(
    action_id="static_analysis",
    phase="apk_ipa_static",
    description="Run static security analyzer (Semgrep / AST audit) across mobile app codebase",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=StaticAnalysisParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_static_analysis",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)

DECOMPILE_APP = ActionDefinition(
    action_id="decompile_app",
    phase="apk_ipa_static",
    description="Decompile APK/IPA package into Smali/Java resources using apktool/jadx",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=DecompileAppParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_decompile_app",
    timeout_seconds=90,
    max_retries=1,
    satisfies_phase=True,
)

PERM_AUDIT = ActionDefinition(
    action_id="perm_audit",
    phase="apk_ipa_static",
    description="Audit AndroidManifest.xml / Info.plist for dangerous permissions and exported components",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=PermAuditParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_perm_audit",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

HARDCODED_SECRETS = ActionDefinition(
    action_id="hardcoded_secrets",
    phase="apk_ipa_static",
    description="Scan decompiled code and assets for embedded API tokens, private keys, and cloud credentials",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=HardcodedSecretsParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_hardcoded_secrets",
    timeout_seconds=45,
    max_retries=2,
    satisfies_phase=True,
)

INSECURE_STORAGE = ActionDefinition(
    action_id="insecure_storage",
    phase="data_storage",
    description="Inspect local SQLite databases, SharedPreferences, and cache files for unencrypted data",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=InsecureStorageParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_insecure_storage",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

TLS_PINNING_CHECK = ActionDefinition(
    action_id="tls_pinning_check",
    phase="network_traffic",
    description="Evaluate network security configuration and verify certificate pinning implementations",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=TlsPinningCheckParams,
    allowed_target_kinds=["domain", "url", "local"],
    command_builder="build_tls_pinning_check",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

AUTH_TOKEN_TEST = ActionDefinition(
    action_id="auth_token_test",
    phase="auth_test",
    description="Test backend API authentication token validation, expiration, and session revocation",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=AuthTokenTestParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_auth_token_test",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

DYNAMIC_INSTRUMENTATION = ActionDefinition(
    action_id="dynamic_instrumentation",
    phase="dynamic",
    description="Deploy dynamic instrumentation hooks (Frida) to trace sensitive method execution",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="medium",
    params_model=DynamicInstrumentationParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_dynamic_instrumentation",
    timeout_seconds=45,
    max_retries=1,
    satisfies_phase=True,
)

WEBVIEW_AUDIT = ActionDefinition(
    action_id="webview_audit",
    phase="apk_ipa_static",
    description="Audit WebView implementations for enabled JavaScript interfaces and URL scheme handling",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=WebViewAuditParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_webview_audit",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

MOBILE_REPORT = ActionDefinition(
    action_id="mobile_report",
    phase="data_storage",
    description="Generate formal OWASP Mobile Top 10 security assessment audit report",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=MobileReportParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_mobile_report",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)


def build_mobile_registry() -> ActionRegistry:
    """Build and return the ActionRegistry for MobileAppAgent."""
    registry = ActionRegistry()
    registry.register(STATIC_ANALYSIS)
    registry.register(DECOMPILE_APP)
    registry.register(PERM_AUDIT)
    registry.register(HARDCODED_SECRETS)
    registry.register(INSECURE_STORAGE)
    registry.register(TLS_PINNING_CHECK)
    registry.register(AUTH_TOKEN_TEST)
    registry.register(DYNAMIC_INSTRUMENTATION)
    registry.register(WEBVIEW_AUDIT)
    registry.register(MOBILE_REPORT)
    return registry

