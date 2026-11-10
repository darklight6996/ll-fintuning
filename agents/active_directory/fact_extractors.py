"""
agents/active_directory/fact_extractors.py

C5: best-effort structured fact extraction from active_directory action output.
kerberoast/asrep_roast/dcsync_audit/acl_audit/tier0_impact/ad_report are currently
placeholder print() stubs in command_builders.py (no real LDAP/Kerberos tool call
yet) -- those intentionally return {} rather than fabricating facts.
"""

from typing import Any, Dict

from core.fact_parsing import parse_dns_addresses, parse_nmap_open_ports


def extract_facts(action_id: str, stdout: str, stderr: str) -> Dict[str, Any]:
    try:
        if action_id == "dc_discovery":
            ips = parse_dns_addresses(stdout)
            facts: Dict[str, Any] = {}
            if ips:
                facts["resolved_ips"] = ips
            ports = parse_nmap_open_ports(stdout)
            if ports:
                facts["open_ports"] = ports
            return facts

        if action_id in ("ad_enum", "trust_audit", "gpo_audit"):
            ports = parse_nmap_open_ports(stdout)
            return {"open_ports": ports} if ports else {}
    except Exception:
        pass
    return {}
