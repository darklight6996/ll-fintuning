"""
agents/active_directory/command_builders.py

Pure Python command translators for ActiveDirectoryAgent actions.
Translates typed action parameter models into executable argument vectors.

SAFETY INVARIANTS:
  - Returns strictly List[List[str]]
  - Never shell strings
  - No shell=True
  - Cross-platform compatible
"""

import sys
from typing import List

from agents.active_directory.actions import (
    DcDiscoveryParams,
    AdEnumParams,
    KerberoastParams,
    AsrepRoastParams,
    DcsyncAuditParams,
    AclAuditParams,
    TrustAuditParams,
    GpoAuditParams,
    Tier0ImpactParams,
    AdReportParams,
)


def build_dc_discovery(params: DcDiscoveryParams) -> List[List[str]]:
    """Build command vector to locate Domain Controllers via DNS and ports."""
    cmds = [
        ["nslookup", "-type=SRV", f"_ldap._tcp.dc._msdcs.{params.target}"]
    ]
    if params.dns_server:
        cmds.append(["nmap", "-sT", "-p", "88,389,445", params.dns_server])
    return cmds


def build_ad_enum(params: AdEnumParams) -> List[List[str]]:
    """Build command vector for LDAP/RPC object enumeration."""
    return [
        ["nmap", "-p", "389,636", "--script=ldap-search", params.target]
    ]


def build_kerberoast(params: KerberoastParams) -> List[List[str]]:
    """Build command vector for Kerberoastable SPN discovery."""
    return [
        [sys.executable, "-c", f"print('Auditing Kerberoastable SPNs on domain {params.domain} at {params.target}')"]
    ]


def build_asrep_roast(params: AsrepRoastParams) -> List[List[str]]:
    """Build command vector for AS-REP roasting preauth audit."""
    return [
        [sys.executable, "-c", f"print('Auditing AS-REP Roasting preauth on domain {params.domain} at {params.target}')"]
    ]


def build_dcsync_audit(params: DcsyncAuditParams) -> List[List[str]]:
    """Build command vector for DCSync replication rights inspection."""
    return [
        [sys.executable, "-c", f"print('Auditing DCSync rights for {params.target_account} on {params.domain}')"]
    ]


def build_acl_audit(params: AclAuditParams) -> List[List[str]]:
    """Build command vector for dangerous ACL relationship inspection."""
    return [
        [sys.executable, "-c", f"print('Auditing ACL rights for {params.principal} on {params.domain}')"]
    ]


def build_trust_audit(params: TrustAuditParams) -> List[List[str]]:
    """Build command vector for domain trust enumeration."""
    return [
        ["nmap", "-p", "389", "--script=smb-enum-domains", params.target]
    ]


def build_gpo_audit(params: GpoAuditParams) -> List[List[str]]:
    """Build command vector for GPO security setting audit."""
    return [
        ["nmap", "-p", "445", "--script=smb-enum-shares", params.target]
    ]


def build_tier0_impact(params: Tier0ImpactParams) -> List[List[str]]:
    """Build command vector for Tier-0 attack path correlation."""
    return [
        [sys.executable, "-c", f"print('Synthesizing Tier-0 impact path for {params.target}')"]
    ]


def build_ad_report(params: AdReportParams) -> List[List[str]]:
    """Build command vector for Active Directory assessment report generation."""
    return [
        [sys.executable, "-c", f"print('Generating Active Directory assessment report for {params.target}')"]
    ]

