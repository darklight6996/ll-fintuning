"""
agents/red_team/command_builders.py

Pure Python command translators for RedTeamAgent actions.
Translates typed action parameter models into executable argument vectors.

SAFETY INVARIANTS:
  - Returns strictly List[List[str]]
  - Never shell strings
  - No shell=True
  - Cross-platform compatible
"""

import sys
from typing import List

from agents.red_team.actions import (
    ExternalReconParams,
    PerimeterScanParams,
    InitialAccessProbeParams,
    CredentialStuffingParams,
    PrivescCheckParams,
    DefenseEvasionParams,
    LateralMovementParams,
    DataDiscoveryParams,
    ExfiltrationTestParams,
    ObjectiveImpactParams,
)


def build_external_recon(params: ExternalReconParams) -> List[List[str]]:
    """Build command vector for external DNS and WHOIS recon."""
    return [
        ["nslookup", params.target]
    ]


def build_perimeter_scan(params: PerimeterScanParams) -> List[List[str]]:
    """Build command vector for stealth perimeter port scanning."""
    return [
        ["nmap", "-sS", "-Pn", "-p", params.ports, "--open", params.target]
    ]


def build_initial_access_probe(params: InitialAccessProbeParams) -> List[List[str]]:
    """Build command vector for non-destructive foothold validation."""
    return [
        ["curl", "-s", "-I", "-k", "--max-time", "10", params.target]
    ]


def build_credential_stuffing(params: CredentialStuffingParams) -> List[List[str]]:
    """Build command vector for single-probe authentication endpoint validation."""
    return [
        ["curl", "-s", "-k", "-X", "POST", params.target, "-d", f"user={params.test_user}&pass=BenignAuditProbe123!"]
    ]


def build_privesc_check(params: PrivescCheckParams) -> List[List[str]]:
    """Build command vector for privilege escalation checks."""
    if sys.platform == "win32":
        return [["whoami", "/priv"]]
    return [["id"], ["sudo", "-n", "-l"]]  # -n: never prompt for a password; fail fast instead of hanging an unattended run


def build_defense_evasion(params: DefenseEvasionParams) -> List[List[str]]:
    """Build command vector for defense and EDR auditing."""
    if sys.platform == "win32":
        return [["sc", "query", "windefend"]]
    return [["ps", "aux"]]


def build_lateral_movement(params: LateralMovementParams) -> List[List[str]]:
    """Build command vector for testing internal host pivot reachability."""
    port_map = {"smb": "445", "wmi": "135", "winrm": "5985", "ssh": "22"}
    port = port_map.get(params.protocol.lower(), "445")
    return [
        ["nmap", "-sT", "-p", port, params.dest_ip]
    ]


def build_data_discovery(params: DataDiscoveryParams) -> List[List[str]]:
    """Build command vector for sensitive file artifact discovery."""
    if sys.platform == "win32":
        return [["dir", "/s", "/b", params.search_pattern]]
    return [["find", ".", "-name", "*.kdbx"]]


def build_exfiltration_test(params: ExfiltrationTestParams) -> List[List[str]]:
    """Build command vector for outbound egress channel check."""
    return [
        ["curl", "-s", "-k", "--max-time", "5", f"https://{params.target}"]
    ]


def build_objective_impact(params: ObjectiveImpactParams) -> List[List[str]]:
    """Build command vector for objective impact demonstration."""
    return [
        [sys.executable, "-c", f"print('Synthesizing adversary impact on {params.target}: {params.objective}')"]
    ]

