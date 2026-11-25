"""
agents/red_team/fact_extractors.py

C5: best-effort structured fact extraction from red_team action output.
"""

from typing import Any, Dict

from core.fact_parsing import (
    parse_dns_addresses,
    parse_find_paths,
    parse_http_headers,
    parse_line_list,
    parse_nmap_open_ports,
)


def extract_facts(action_id: str, stdout: str, stderr: str) -> Dict[str, Any]:
    try:
        if action_id == "http_probe":
            from agents.common.http_primitive import extract_http_probe_facts
            return extract_http_probe_facts(stdout, stderr)
        if action_id == "browser_probe":
            from agents.common.browser_primitive import extract_browser_facts
            return extract_browser_facts(stdout, stderr)
        if action_id == "self_script":
            from agents.common.self_scripting import extract_self_script_facts
            return extract_self_script_facts(stdout, stderr)
        if action_id == "external_recon":
            ips = parse_dns_addresses(stdout)
            return {"resolved_ips": ips} if ips else {}

        if action_id in ("perimeter_scan", "lateral_movement"):
            ports = parse_nmap_open_ports(stdout)
            return {"open_ports": ports} if ports else {}

        if action_id in ("initial_access_probe", "credential_stuffing", "exfiltration_test"):
            return parse_http_headers(stdout)

        if action_id in ("privesc_check", "defense_evasion"):
            lines = parse_line_list(stdout)
            return {"raw_lines": lines} if lines else {}

        if action_id == "data_discovery":
            # Unix `find` output is bare paths; Windows `dir /s /b` output is also
            # one path per line, so the generic path extractor covers both forms.
            paths = parse_find_paths(stdout) or parse_line_list(stdout, max_items=30)
            return {"discovered_files": paths} if paths else {}
    except Exception:
        pass
    return {}
