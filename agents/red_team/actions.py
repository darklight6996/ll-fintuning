"""
agents/red_team/actions.py

Action definitions and registry factory for RedTeamAgent.
Covers adversary emulation workflows: reconnaissance, initial access, privilege escalation,
defense evasion, lateral movement, and objective impact.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition, ActionRegistry


# ---------------------------------------------------------------------------
# Typed Action Parameter Models (Pydantic v2, extra="forbid")
# ---------------------------------------------------------------------------

class ExternalReconParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target domain, IP, or organization name")


class PerimeterScanParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target host, IP, or CIDR")
    ports: str = Field("80,443,22,21,3389,8080,8443", description="Perimeter ports to assess")


class InitialAccessProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target external service URL or endpoint")
    vector: str = Field("web_foothold", description="Foothold vector (web_foothold, api_token, vpn_portal)")


class CredentialStuffingParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target authentication portal URL")
    test_user: str = Field(..., description="Account username to test")


class PrivescCheckParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Compromised target host")
    vector: str = Field("token_privileges", description="Privilege escalation check (token_privileges, service_binary, suid)")


class DefenseEvasionParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target host")
    mechanism: str = Field("process_injection_audit", description="Defense mechanism to audit (amsi_check, process_injection_audit)")


class LateralMovementParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Pivoting source host")
    dest_ip: str = Field(..., description="Destination internal host IP")
    protocol: str = Field("smb", description="Pivot protocol (smb, wmi, winrm, ssh)")


class DataDiscoveryParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target host")
    search_pattern: str = Field("*.kdbx,*.pfx,password*,id_rsa*", description="Target file patterns")


class ExfiltrationTestParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target host")
    channel: str = Field("https", description="Egress channel (https, dns, icmp)")


class ObjectiveImpactParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target organization or crown jewel asset")
    objective: str = Field("customer_db_access", description="Engagement objective description")


# ---------------------------------------------------------------------------
# Action Definitions
# ---------------------------------------------------------------------------

EXTERNAL_RECON = ActionDefinition(
    action_id="external_recon",
    phase="recon",
    description="Perform passive external reconnaissance, IP ownership, and ASN mapping",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=ExternalReconParams,
    allowed_target_kinds=["domain", "ip"],
    command_builder="build_external_recon",
    timeout_seconds=45,
    max_retries=2,
    satisfies_phase=True,
)

PERIMETER_SCAN = ActionDefinition(
    action_id="perimeter_scan",
    phase="recon",
    description="Scan perimeter network assets for exposed services and entry points",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=PerimeterScanParams,
    allowed_target_kinds=["ip", "cidr", "domain"],
    command_builder="build_perimeter_scan",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)

INITIAL_ACCESS_PROBE = ActionDefinition(
    action_id="initial_access_probe",
    phase="initial_access",
    description="Validate initial access entry vectors without payload deployment",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=InitialAccessProbeParams,
    allowed_target_kinds=["url", "domain", "ip"],
    command_builder="build_initial_access_probe",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

CREDENTIAL_STUFFING = ActionDefinition(
    action_id="credential_stuffing",
    phase="initial_access",
    description="Test authentication endpoint response against single credential pair to verify lockout controls",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=CredentialStuffingParams,
    allowed_target_kinds=["url", "domain"],
    command_builder="build_credential_stuffing",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

PRIVESC_CHECK = ActionDefinition(
    action_id="privesc_check",
    phase="privesc",
    description="Audit local escalation paths and misconfigured service permissions on compromised host",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=PrivescCheckParams,
    allowed_target_kinds=["local", "ip"],
    command_builder="build_privesc_check",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

DEFENSE_EVASION = ActionDefinition(
    action_id="defense_evasion",
    phase="privesc",
    description="Audit telemetry and detection controls (EDR sensors, event logging, AMSI posture)",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=DefenseEvasionParams,
    allowed_target_kinds=["local", "ip"],
    command_builder="build_defense_evasion",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

LATERAL_MOVEMENT = ActionDefinition(
    action_id="lateral_movement",
    phase="lateral",
    description="Probe internal network pivot routes and host reachability from compromised foothold",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=LateralMovementParams,
    allowed_target_kinds=["ip", "local"],
    command_builder="build_lateral_movement",
    timeout_seconds=45,
    max_retries=1,
    satisfies_phase=True,
)

DATA_DISCOVERY = ActionDefinition(
    action_id="data_discovery",
    phase="lateral",
    description="Discover high-value sensitive data, API keys, credentials, and configuration files",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=DataDiscoveryParams,
    allowed_target_kinds=["local", "ip"],
    command_builder="build_data_discovery",
    timeout_seconds=45,
    max_retries=1,
    satisfies_phase=True,
)

EXFILTRATION_TEST = ActionDefinition(
    action_id="exfiltration_test",
    phase="impact",
    description="Test egress channels (DNS tunneling, HTTPS outbound) for data exfiltration detection",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="low",
    params_model=ExfiltrationTestParams,
    allowed_target_kinds=["local", "domain"],
    command_builder="build_exfiltration_test",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

OBJECTIVE_IMPACT = ActionDefinition(
    action_id="objective_impact",
    phase="impact",
    description="Synthesize demonstrated breach consequences and quantify business impact",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=ObjectiveImpactParams,
    allowed_target_kinds=["domain", "local", "ip"],
    command_builder="build_objective_impact",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)


def build_red_team_registry() -> ActionRegistry:
    """Build and return the ActionRegistry for RedTeamAgent."""
    registry = ActionRegistry()
    registry.register(EXTERNAL_RECON)
    registry.register(PERIMETER_SCAN)
    registry.register(INITIAL_ACCESS_PROBE)
    registry.register(CREDENTIAL_STUFFING)
    registry.register(PRIVESC_CHECK)
    registry.register(DEFENSE_EVASION)
    registry.register(LATERAL_MOVEMENT)
    registry.register(DATA_DISCOVERY)
    registry.register(EXFILTRATION_TEST)
    registry.register(OBJECTIVE_IMPACT)
    return registry

