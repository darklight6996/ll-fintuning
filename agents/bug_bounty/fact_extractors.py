"""
agents/bug_bounty/fact_extractors.py

C5: best-effort structured fact extraction from bug_bounty action output, so the
AI planner sees "subdomains: [...]" / "open_ports: [...]" instead of 120 raw
characters of tool stdout it has to re-derive from scratch every step.

extract_facts() never raises -- a parsing failure or unrecognized output format
yields {} for that observation, never an exception that would break the loop.
"""

from typing import Any, Dict

from core.fact_parsing import parse_http_headers, parse_line_list, parse_nmap_open_ports


def extract_facts(action_id: str, stdout: str, stderr: str) -> Dict[str, Any]:
    try:
        if action_id == "subdomain_enum":
            subs = parse_line_list(stdout)
            return {"subdomains": subs} if subs else {}

        if action_id == "port_scan":
            ports = parse_nmap_open_ports(stdout)
            return {"open_ports": ports} if ports else {}

        if action_id == "dir_brute":
            # ffuf -silent prints one matching URL/line per hit; exact format
            # varies by ffuf version, so this is a best-effort line capture.
            hits = parse_line_list(stdout)
            return {"discovered_paths": hits} if hits else {}

        if action_id in (
            "endpoint_discovery", "param_fuzz", "header_analysis",
            "idor_probe", "sqli_probe", "xss_probe", "vuln_validation",
        ):
            return parse_http_headers(stdout)
    except Exception:
        pass
    return {}
