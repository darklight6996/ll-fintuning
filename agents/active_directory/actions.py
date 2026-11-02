"""
agents/active_directory/actions.py

Action definitions and registry factory for ActiveDirectoryAgent.
Covers Domain Controller discovery, Kerberoasting, AS-REP Roasting, DCSync rights,
ACL relationships, and Tier-0 trust auditing.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition, ActionRegistry


# ---------------------------------------------------------------------------
# Typed Action Parameter Models (Pydantic v2, extra="forbid")
# ---------------------------------------------------------------------------

class DcDiscoveryParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target domain or root domain (e.g. corp.local)")
    dns_server: Optional[str] = Field(None, description="Optional DNS server IP for SRV lookup")


class AdEnumParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target Domain Controller IP or hostname")
    domain: str = Field(..., description="Active Directory domain name")
    enum_scope: str = Field("users,groups,computers", description="AD object classes to query")


class KerberoastParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Domain Controller IP or hostname")
    domain: str = Field(..., description="Active Directory domain name")
    spn_filter: str = Field("*", description="SPN filter for target service accounts")


class AsrepRoastParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Domain Controller IP or hostname")
    domain: str = Field(..., description="Active Directory domain name")
    user_list: Optional[str] = Field(None, description="Comma-separated username list or file path")


class DcsyncAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Domain Controller IP or hostname")
    domain: str = Field(..., description="Active Directory domain name")
    target_account: str = Field("krbtgt", description="Target account for replication rights check")


class AclAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Domain Controller IP or hostname")
    domain: str = Field(..., description="Active Directory domain name")
    principal: str = Field("Domain Users", description="Security principal whose write/reset ACLs are audited")


class TrustAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Domain Controller IP or hostname")
    domain: str = Field(..., description="Active Directory domain name")


class GpoAuditParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Domain Controller IP or hostname")
    domain: str = Field(..., description="Active Directory domain name")


class Tier0ImpactParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target domain")
    objective: str = Field("domain_admin_path", description="Attack path evaluation objective")


class AdReportParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(..., description="Target domain")
    output_format: str = Field("markdown", description="Report format (markdown, json)")


# ---------------------------------------------------------------------------
# Action Definitions
# ---------------------------------------------------------------------------

DC_DISCOVERY = ActionDefinition(
    action_id="dc_discovery",
    phase="ad_enumeration",
    description="Locate Domain Controllers via DNS SRV records and port probing (88, 389, 445)",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=DcDiscoveryParams,
    allowed_target_kinds=["domain", "ip"],
    command_builder="build_dc_discovery",
    timeout_seconds=45,
    max_retries=2,
    satisfies_phase=True,
)

AD_ENUM = ActionDefinition(
    action_id="ad_enum",
    phase="ad_enumeration",
    description="Query domain users, security groups, computers, and password policy via LDAP/RPC",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=AdEnumParams,
    allowed_target_kinds=["ip", "domain"],
    command_builder="build_ad_enum",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)

KERBEROAST = ActionDefinition(
    action_id="kerberoast",
    phase="kerberoast",
    description="Request TGS service tickets for user accounts configured with SPNs to test crackability",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=KerberoastParams,
    allowed_target_kinds=["ip", "domain"],
    command_builder="build_kerberoast",
    timeout_seconds=60,
    max_retries=1,
    satisfies_phase=True,
)

ASREP_ROAST = ActionDefinition(
    action_id="asrep_roast",
    phase="asrep_roast",
    description="Request AS-REQ for accounts with DONT_REQ_PREAUTH set to capture offline hash",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="medium",
    params_model=AsrepRoastParams,
    allowed_target_kinds=["ip", "domain"],
    command_builder="build_asrep_roast",
    timeout_seconds=60,
    max_retries=1,
    satisfies_phase=True,
)

DCSYNC_AUDIT = ActionDefinition(
    action_id="dcsync_audit",
    phase="dcsync",
    description="Audit replication rights (DS-Replication-Get-Changes-All) for directory sync vulnerabilities",
    requires_active_testing=True,
    network_effect="active",
    state_change="none",
    risk_level="high",
    params_model=DcsyncAuditParams,
    allowed_target_kinds=["ip", "domain"],
    command_builder="build_dcsync_audit",
    timeout_seconds=60,
    max_retries=1,
    satisfies_phase=True,
)

ACL_AUDIT = ActionDefinition(
    action_id="acl_audit",
    phase="ad_enumeration",
    description="Audit dangerous ACE permissions (GenericAll, WriteDacl, WriteOwner) on high-value targets",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=AclAuditParams,
    allowed_target_kinds=["ip", "domain"],
    command_builder="build_acl_audit",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)

TRUST_AUDIT = ActionDefinition(
    action_id="trust_audit",
    phase="ad_enumeration",
    description="Enumerate forest and external domain trusts and sid-filtering configurations",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=TrustAuditParams,
    allowed_target_kinds=["ip", "domain"],
    command_builder="build_trust_audit",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

GPO_AUDIT = ActionDefinition(
    action_id="gpo_audit",
    phase="ad_enumeration",
    description="Audit Group Policy Objects for sensitive scripts, local admin additions, or weak file permissions",
    requires_active_testing=False,
    network_effect="read_only",
    state_change="none",
    risk_level="low",
    params_model=GpoAuditParams,
    allowed_target_kinds=["ip", "domain"],
    command_builder="build_gpo_audit",
    timeout_seconds=45,
    max_retries=2,
    satisfies_phase=True,
)

TIER0_IMPACT = ActionDefinition(
    action_id="tier0_impact",
    phase="impact",
    description="Correlate discovered AD escalation vectors into an impact assessment on Tier-0 assets",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=Tier0ImpactParams,
    allowed_target_kinds=["domain", "ip"],
    command_builder="build_tier0_impact",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

AD_REPORT = ActionDefinition(
    action_id="ad_report",
    phase="impact",
    description="Generate formal enterprise Active Directory security assessment audit report",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=AdReportParams,
    allowed_target_kinds=["domain", "ip"],
    command_builder="build_ad_report",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)


def build_ad_registry() -> ActionRegistry:
    """Build and return the ActionRegistry for ActiveDirectoryAgent."""
    registry = ActionRegistry()
    registry.register(DC_DISCOVERY)
    registry.register(AD_ENUM)
    registry.register(KERBEROAST)
    registry.register(ASREP_ROAST)
    registry.register(DCSYNC_AUDIT)
    registry.register(ACL_AUDIT)
    registry.register(TRUST_AUDIT)
    registry.register(GPO_AUDIT)
    registry.register(TIER0_IMPACT)
    registry.register(AD_REPORT)
    return registry

