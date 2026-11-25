"""
agents/firewall_bypass/actions.py

Action definitions and registry factory for FirewallBypassAgent.
Covers Web Application Firewall (WAF) & IDS/IPS evasion: fingerprinting, rule mapping,
chunked encoding, header mutation, path normalization, and payload obfuscation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition, ActionRegistry


# ---------------------------------------------------------------------------
# Typed Action Parameter Models (Pydantic v2, extra="forbid")
# ---------------------------------------------------------------------------

class WafFingerprintParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target domain or base URL")


class RuleEnumParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target base URL")
    endpoint: str = Field("/", description="Endpoint to probe for rule triggers")


class ChunkedEncodingParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target base URL")
    endpoint: str = Field("/", description="Endpoint tested with chunked transfer")


class UnicodeNormalizationParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target base URL")
    endpoint: str = Field("/", description="Endpoint tested with overlong or full-width characters")


class HeaderMutationParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target base URL")
    endpoint: str = Field("/", description="Endpoint tested with origin/forwarding headers")


class ContentTypeConfusionParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target base URL")
    endpoint: str = Field("/", description="Endpoint tested with MIME confusion")


class PathNormalizationParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target base URL")
    endpoint: str = Field("/admin", description="Target path tested with dot/slash/semicolon mutations")


class ParameterPollutionParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target base URL")
    endpoint: str = Field("/search", description="Endpoint tested with duplicate parameter keys")


class PayloadMutationParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target base URL")
    endpoint: str = Field("/", description="Endpoint tested with obfuscated benign probe")
    mutation_type: str = Field("sql_comment_inline", description="Obfuscation type (sql_comment_inline, case_toggle, null_byte)")


class WafBypassReportParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target base URL or domain")
    output_format: str = Field("markdown", description="Report format (markdown, json)")


# ---------------------------------------------------------------------------
# Action Definitions
# ---------------------------------------------------------------------------

WAF_FINGERPRINT = ActionDefinition(
    action_id="waf_fingerprint",
    phase="fingerprint",
    description="Identify active Web Application Firewall (Cloudflare, AWS WAF, ModSecurity, Imperva)",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=WafFingerprintParams,
    allowed_target_kinds=["domain", "url", "ip"],
    command_builder="build_waf_fingerprint",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

RULE_ENUM = ActionDefinition(
    action_id="rule_enum",
    phase="rule_enumeration",
    description="Map WAF blocking thresholds, HTTP verbs, and banned header structures",
    requires_active_testing=True,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=RuleEnumParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_rule_enum",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

CHUNKED_ENCODING = ActionDefinition(
    action_id="chunked_encoding",
    phase="bypass_techniques",
    description="Test HTTP chunked transfer encoding fragmentation to bypass WAF inspection engines",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=ChunkedEncodingParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_chunked_encoding",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

UNICODE_NORMALIZATION = ActionDefinition(
    action_id="unicode_normalization",
    phase="bypass_techniques",
    description="Test Unicode best-fit and overlong UTF-8 normalization parsing discrepancies",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=UnicodeNormalizationParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_unicode_normalization",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

HEADER_MUTATION = ActionDefinition(
    action_id="header_mutation",
    phase="bypass_techniques",
    description="Test IP spoofing headers (X-Forwarded-For, X-Real-IP) and hop-by-hop headers",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="low",
    params_model=HeaderMutationParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_header_mutation",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

CONTENT_TYPE_CONFUSION = ActionDefinition(
    action_id="content_type_confusion",
    phase="bypass_techniques",
    description="Evaluate MIME type parsing differences between WAF and backend application server",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="low",
    params_model=ContentTypeConfusionParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_content_type_confusion",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

PATH_NORMALIZATION = ActionDefinition(
    action_id="path_normalization",
    phase="bypass_techniques",
    description="Test URL path matrix parameters, trailing dots, and semicolon path confusion",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="low",
    params_model=PathNormalizationParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_path_normalization",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

PARAMETER_POLLUTION = ActionDefinition(
    action_id="parameter_pollution",
    phase="bypass_techniques",
    description="Test HTTP Parameter Pollution (HPP) precedence differences between proxy and server",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="low",
    params_model=ParameterPollutionParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_parameter_pollution",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

PAYLOAD_MUTATION = ActionDefinition(
    action_id="payload_mutation",
    phase="bypass_techniques",
    description="Send obfuscated benign syntax tokens (inline comments, case alterations) to test regex evasion",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=PayloadMutationParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_payload_mutation",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

WAF_BYPASS_REPORT = ActionDefinition(
    action_id="waf_bypass_report",
    phase="bypass_techniques",
    description="Generate formal Web Application Firewall evasion and rule efficacy audit report",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=WafBypassReportParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_waf_bypass_report",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)


def build_firewall_registry() -> ActionRegistry:
    """Build and return the ActionRegistry for FirewallBypassAgent."""
    registry = ActionRegistry()
    from agents.firewall_bypass.command_builders import (
        build_chunked_encoding,
        build_content_type_confusion,
        build_header_mutation,
        build_parameter_pollution,
        build_path_normalization,
        build_payload_mutation,
        build_rule_enum,
        build_unicode_normalization,
        build_waf_bypass_report,
        build_waf_fingerprint,
    )

    registry.register(WAF_FINGERPRINT, build_waf_fingerprint)
    registry.register(RULE_ENUM, build_rule_enum)
    registry.register(CHUNKED_ENCODING, build_chunked_encoding)
    registry.register(UNICODE_NORMALIZATION, build_unicode_normalization)
    registry.register(HEADER_MUTATION, build_header_mutation)
    registry.register(CONTENT_TYPE_CONFUSION, build_content_type_confusion)
    registry.register(PATH_NORMALIZATION, build_path_normalization)
    registry.register(PARAMETER_POLLUTION, build_parameter_pollution)
    registry.register(PAYLOAD_MUTATION, build_payload_mutation)
    registry.register(WAF_BYPASS_REPORT, build_waf_bypass_report)
    # Framework-level shared primitives (cross-agent): HTTP primitive,
    # headless-browser coverage, and opt-in self-scripting -- the same
    # capabilities every agent gets, wired identically.
    from agents.common.http_primitive import HTTP_PROBE, build_http_probe
    registry.register(HTTP_PROBE, build_http_probe)
    from agents.common.browser_primitive import BROWSER_PROBE, build_browser_probe
    registry.register(BROWSER_PROBE, build_browser_probe)
    from agents.common.self_scripting import SELF_SCRIPT, build_self_script, self_scripting_enabled
    if self_scripting_enabled():
        registry.register(SELF_SCRIPT, build_self_script)
    return registry

