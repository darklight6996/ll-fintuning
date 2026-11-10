"""
core/fact_parsing.py

Shared, defensive parsers that turn raw tool stdout into small structured facts
(C5). These populate Observation.parsed_facts so the AI planner can see "open
ports: [...]" instead of 120 raw characters of nmap/curl output it has to
re-derive from scratch every step.

Every parser is best-effort: unexpected or malformed tool output yields an
empty or partial result, never an exception. parsed_facts is a summarization
convenience for the planner prompt, not a safety gate -- nothing downstream of
it (Gate 1-10, completion criteria) depends on it being correct or complete.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_NMAP_PORT_LINE = re.compile(
    r"^(?P<port>\d{1,5})/(?P<proto>tcp|udp)\s+(?P<state>open|closed|filtered)\s+"
    r"(?P<service>\S+)(?:\s+(?P<version>.*))?$",
    re.MULTILINE,
)

_NMAP_HOST_LINE = re.compile(
    r"Nmap scan report for (?:(?P<name>\S+) \()?(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\)?"
)

_HTTP_STATUS_LINE = re.compile(r"^\s*(?:<\s*)?HTTP/[\d.]+\s+(\d{3})", re.MULTILINE)
_HTTP_HEADER_LINE = re.compile(r"^\s*(?:[<>]\s*)?([A-Za-z][A-Za-z0-9-]*):\s*(.+?)\s*$")

_WAF_HEADER_SIGNATURES = {
    "cf-ray": "Cloudflare",
    "cf-cache-status": "Cloudflare",
    "x-sucuri-id": "Sucuri",
    "x-sucuri-cache": "Sucuri",
    "x-akamai-transformed": "Akamai",
    "x-iinfo": "Incapsula/Imperva",
    "x-cdn": "Generic CDN",
    "x-waf-event-info": "Generic WAF",
}


def parse_nmap_open_ports(stdout: str) -> List[Dict[str, Any]]:
    """Extract {port, proto, service, version} for each 'open' line in nmap -sV output."""
    results: List[Dict[str, Any]] = []
    try:
        for m in _NMAP_PORT_LINE.finditer(stdout or ""):
            if m.group("state") != "open":
                continue
            results.append({
                "port": int(m.group("port")),
                "proto": m.group("proto"),
                "service": m.group("service"),
                "version": (m.group("version") or "").strip(),
            })
    except Exception:
        return results
    return results


def parse_nmap_hosts_up(stdout: str) -> List[str]:
    """Extract host IPs/hostnames reported by nmap -sn (host discovery)."""
    hosts: List[str] = []
    try:
        for m in _NMAP_HOST_LINE.finditer(stdout or ""):
            host = m.group("name") or m.group("ip")
            if host and host not in hosts:
                hosts.append(host)
    except Exception:
        return hosts
    return hosts


def parse_http_headers(stdout: str) -> Dict[str, Any]:
    """
    Extract status code, a flattened header dict, and a best-effort WAF/CDN
    signature from curl -I / -v output. Handles both plain `curl -I` (headers
    at column 0) and `curl -v` (request/response lines prefixed '> '/'< ').
    """
    facts: Dict[str, Any] = {}
    try:
        status_match = _HTTP_STATUS_LINE.search(stdout or "")
        if status_match:
            facts["status_code"] = int(status_match.group(1))

        headers: Dict[str, str] = {}
        for raw_line in (stdout or "").splitlines():
            line = raw_line.strip()
            if line.startswith(("< ", "> ")):
                line = line[2:].strip()
            m = _HTTP_HEADER_LINE.match(line)
            if m:
                headers[m.group(1).lower()] = m.group(2).strip()

        if headers:
            facts["headers"] = headers
            if "server" in headers:
                facts["server"] = headers["server"]

        detected_waf = None
        for h_name, waf_name in _WAF_HEADER_SIGNATURES.items():
            if h_name in headers:
                detected_waf = waf_name
                break
        if not detected_waf:
            server_val = headers.get("server", "").lower()
            if "cloudflare" in server_val:
                detected_waf = "Cloudflare"
            elif "awselb" in server_val:
                detected_waf = "AWS ALB (possibly fronted by AWS WAF)"
        if detected_waf:
            facts["waf_signature"] = detected_waf
    except Exception:
        return facts
    return facts


def parse_find_paths(stdout: str, max_items: int = 50) -> List[str]:
    """
    Extract absolute filesystem paths from `find` output -- both the bare
    `find ... -type f` form (whole line is the path) and the `find ... -ls`
    form (path is the last whitespace-separated column).
    """
    paths: List[str] = []
    try:
        for raw_line in (stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            token = tokens[-1] if tokens else line
            if token.startswith("/") and token not in paths:
                paths.append(token)
            if len(paths) >= max_items:
                break
    except Exception:
        return paths
    return paths


_UID_LINE = re.compile(r"uid=(\d+)\(([^)]+)\)\s+gid=\d+\(([^)]+)\)(?:.*groups=(.*))?")


def parse_unix_identity(stdout: str) -> Dict[str, Any]:
    """Extract {user, group, groups} from `id` command output, if present."""
    facts: Dict[str, Any] = {}
    try:
        m = _UID_LINE.search(stdout or "")
        if m:
            facts["user"] = m.group(2)
            facts["group"] = m.group(3)
            if m.group(4):
                groups = re.findall(r"\(([^)]+)\)", m.group(4))
                if groups:
                    facts["groups"] = groups
    except Exception:
        return facts
    return facts


_DNS_ADDRESS_LINE = re.compile(r"^Address:\s*(\d{1,3}(?:\.\d{1,3}){3})\s*$", re.MULTILINE)


def parse_dns_addresses(stdout: str) -> List[str]:
    """Extract resolved IPv4 addresses from nslookup output."""
    try:
        return list(dict.fromkeys(_DNS_ADDRESS_LINE.findall(stdout or "")))
    except Exception:
        return []


def parse_line_list(stdout: str, max_items: int = 50) -> List[str]:
    """
    Generic fallback for list-producing tools (subfinder, ffuf -silent, etc.):
    non-empty, non-comment lines, deduplicated and order-preserved, capped at
    max_items so a single noisy observation can't blow out the planner prompt.
    """
    seen: List[str] = []
    try:
        for raw_line in (stdout or "").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and line not in seen:
                seen.append(line)
            if len(seen) >= max_items:
                break
    except Exception:
        return seen
    return seen
