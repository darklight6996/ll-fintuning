"""
core/intelligence/parsers/banners.py

Parser plugin for service banners, HTTP Server headers, and network daemon signatures.
"""

import re
from typing import Dict, Any, List
from core.intelligence.base import BaseParser
from core.intelligence.models import BannerIntel


class BannerParser(BaseParser):
    """
    Parses network service banners and daemon version strings.
    """

    @property
    def name(self) -> str:
        return "banners"

    def parse(self, text: str, session_state: Dict[str, Any]) -> BannerIntel:
        network_services: List[str] = []

        # Pattern 1: Explicit banner/server labels
        banner_matches = re.findall(
            r'(?:banner|server|service)[:\s]+[\'"]?([^\r\n\'"]{3,60})[\'"]?',
            text,
            re.IGNORECASE
        )
        for b in banner_matches:
            b_clean = b.strip()
            if b_clean and b_clean not in network_services:
                network_services.append(b_clean)

        # Pattern 2: Common daemon fingerprints in output (Apache, nginx, OpenSSH, etc.)
        fingerprints = re.findall(
            r'\b((?:Apache|nginx|OpenSSH|vsftpd|ProFTPD|MySQL|PostgreSQL|Werkzeug|Microsoft-IIS)[/_v\s][0-9a-zA-Z.-]+)',
            text,
            re.IGNORECASE
        )
        for fp in fingerprints:
            fp_clean = fp.strip()
            if fp_clean and fp_clean not in network_services:
                network_services.append(fp_clean)

        return BannerIntel(network_services=network_services)
