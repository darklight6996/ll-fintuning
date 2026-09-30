"""
core/intelligence/parsers/ports.py

Parser plugin for discovered network ports from nmap, ss, netstat, and lsof.
"""

import re
from typing import Dict, Any, List
from core.intelligence.base import BaseParser
from core.intelligence.models import PortIntel


class PortParser(BaseParser):
    """
    Parses open ports, states, and port numbers from network reconnaissance tools.
    """

    @property
    def name(self) -> str:
        return "ports"

    def parse(self, text: str, session_state: Dict[str, Any]) -> PortIntel:
        open_ports: List[str] = []

        # Pattern 1: Nmap tabular output: '80/tcp open http' or '445/tcp filtered microsoft-ds'
        nmap_matches = re.findall(
            r'(\d+/(?:tcp|udp))\s+(\w+)\s+([^\r\n]+)',
            text
        )
        for port_proto, state_str, svc_desc in nmap_matches:
            entry = f"{port_proto} ({state_str}) - {svc_desc.strip()}"
            if entry not in open_ports:
                open_ports.append(entry)

        # Pattern 2: Generic port mentions e.g. 'port 8080' or ':8080'
        port_mentions = re.findall(r'\b(?:port\s+|:)(\d{2,5})\b', text, re.IGNORECASE)
        for p in port_mentions:
            p_str = f"Port {p}"
            # Avoid adding duplicate port if already represented in detailed nmap entry
            if not any(f"{p}/" in op for op in open_ports) and p_str not in open_ports:
                open_ports.append(p_str)

        return PortIntel(open_ports=open_ports)
