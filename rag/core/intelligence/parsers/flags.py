"""
core/intelligence/parsers/flags.py

Parser plugin for CTF flags, target tokens, and flag file permission inspection.
"""

import re
from typing import Dict, Any, Optional
from core.intelligence.base import BaseParser
from core.intelligence.models import FlagIntel


class FlagParser(BaseParser):
    """
    Detects CTF flags and target flag file filesystem permissions.
    """

    @property
    def name(self) -> str:
        return "flags"

    def parse(self, text: str, session_state: Dict[str, Any]) -> FlagIntel:
        flag_permissions: Optional[str] = None
        flag_content: Optional[str] = None

        # Pattern 1: Target flag file permissions:
        # e.g. -r-------- 1 root root 53 Sep 30 12:00 /flag
        flag_perm_match = re.search(
            r'([-d][-rwx]{9})\s+\d+\s+root\s+root\s+\d+\s+.*(/flag)',
            text
        )
        if flag_perm_match:
            flag_permissions = flag_perm_match.group(1)

        # Pattern 2: CTF Flag & Password formats
        # Matches picoCTF{...}, pwn.college{...}, HTB{...}, flag{...}, OverTheWire Bandit passwords, etc.
        flag_match = re.search(
            r'(?:(?:pwn\.college|flag|htb|ctf|picoctf|hackthebox)\{[^}\s]+\}'
            r'|\b(?:FLAG|Flag):\s*[a-zA-Z0-9_\-\{\}]{8,}'
            r'|(?:The password you are looking for is:\s*|NEXT_PASSWORD:\s*|password\s+is:\s*)([a-zA-Z0-9_\-\.]{10,}))',
            text,
            re.IGNORECASE
        )
        if flag_match:
            # Capture specific password group if present, otherwise full match
            flag_content = flag_match.group(1) if flag_match.lastindex and flag_match.group(1) else flag_match.group(0)

        return FlagIntel(
            flag_permissions=flag_permissions,
            flag_content=flag_content
        )
