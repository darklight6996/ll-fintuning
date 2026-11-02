"""
core/intelligence/parsers/kernel.py

Parser plugin for operating system, kernel release, and CPU architecture.
"""

import re
from typing import Dict, Any, Optional
from core.intelligence.base import BaseParser
from core.intelligence.models import KernelIntel


class KernelParser(BaseParser):
    """
    Parses uname, /proc/version, and OS identification lines.
    """

    @property
    def name(self) -> str:
        return "kernel"

    def parse(self, text: str, session_state: Dict[str, Any]) -> KernelIntel:
        kernel: Optional[str] = None
        os_name: Optional[str] = None
        arch: Optional[str] = None

        # Linux uname match
        uname_match = re.search(r'Linux\s+([^\s]+)\s+([^\s]+)(?:\s+.*)?', text)
        if uname_match:
            kernel = uname_match.group(0).strip()
            os_name = "Linux"

        # Windows NT match
        win_match = re.search(r'Microsoft\s+Windows\s+\[Version\s+([^\]]+)\]', text, re.IGNORECASE)
        if win_match:
            kernel = f"Windows Version {win_match.group(1)}"
            os_name = "Windows"

        # Architecture detection
        if re.search(r'\b(x86_64|amd64)\b', text, re.IGNORECASE):
            arch = "x86_64"
        elif re.search(r'\b(aarch64|arm64)\b', text, re.IGNORECASE):
            arch = "aarch64"
        elif re.search(r'\b(i386|i686)\b', text, re.IGNORECASE):
            arch = "x86_32"

        return KernelIntel(
            kernel=kernel,
            os_name=os_name,
            architecture=arch
        )
