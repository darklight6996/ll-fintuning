"""
core/intelligence/parsers/binaries.py

Parser plugin for challenge executables and custom non-standard binaries.
"""

import re
from typing import Dict, Any, List
from core.intelligence.base import BaseParser
from core.intelligence.models import BinaryIntel


class BinaryParser(BaseParser):
    """
    Identifies custom challenge binaries, compiled scripts, and local executables.
    """

    @property
    def name(self) -> str:
        return "binaries"

    def parse(self, text: str, session_state: Dict[str, Any]) -> BinaryIntel:
        custom_binaries: List[str] = []
        standard_binaries: List[str] = []

        # Find absolute executable file paths
        paths = re.findall(
            r'(/(?:challenge|opt|var|tmp|home|nix|root|usr/local)[^\s\n\(\)]+)',
            text
        )

        for p in paths:
            clean_p = p.rstrip(",.:;'\"")
            if "/challenge/" in clean_p or clean_p.endswith("/run") or clean_p.endswith(".bin"):
                if clean_p not in custom_binaries:
                    custom_binaries.append(clean_p)
            elif clean_p.startswith("/usr/local/bin") or "/opt/" in clean_p:
                if clean_p not in custom_binaries:
                    custom_binaries.append(clean_p)
            else:
                if clean_p not in standard_binaries:
                    standard_binaries.append(clean_p)

        # Detect relative execution patterns like './run' or './vuln'
        rel_matches = re.findall(r'(?:^|[\s"\'`])(\./[a-zA-Z0-9_-]+)', text)
        for rel in rel_matches:
            if rel not in custom_binaries:
                custom_binaries.append(rel)

        return BinaryIntel(
            custom_binaries=custom_binaries,
            standard_binaries=standard_binaries
        )
