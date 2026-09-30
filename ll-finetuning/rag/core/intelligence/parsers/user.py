"""
core/intelligence/parsers/user.py

Parser plugin for user identity, privilege level, and group discovery.
"""

import re
from typing import Dict, Any, List
from core.intelligence.base import BaseParser
from core.intelligence.models import UserIntel


class UserParser(BaseParser):
    """
    Parses user accounts, groups, UIDs, GIDs, and sudo privileges from shell output.
    """

    @property
    def name(self) -> str:
        return "user"

    def parse(self, text: str, session_state: Dict[str, Any]) -> UserIntel:
        user = None
        uid = None
        gid = None
        groups: List[str] = []
        has_sudo = False

        # Pattern 1: POSIX 'id' command output:
        # uid=1000(hacker) gid=1000(hacker) groups=1000(hacker),27(sudo)
        id_match = re.search(
            r'uid=(\d+)\(([^)]+)\)\s+gid=(\d+)\(([^)]+)\)\s+groups=([^\r\n]+)',
            text
        )
        if id_match:
            uid = int(id_match.group(1))
            user = id_match.group(2)
            gid = int(id_match.group(3))
            groups = [g.strip() for g in id_match.group(5).split(',')]

        # Pattern 2: Fallback whoami
        if not user:
            whoami_match = re.search(r'\bwhoami\s*\n\s*([a-zA-Z0-9_.-]+)', text)
            if whoami_match:
                user = whoami_match.group(1).strip()

        # Check sudo status
        if any("sudo" in g.lower() or "wheel" in g.lower() or "admin" in g.lower() for g in groups):
            has_sudo = True
        elif user == "root":
            has_sudo = True
        elif re.search(r'(?:NOPASSWD|\(ALL\s*:\s*ALL\)\s*ALL)', text):
            has_sudo = True

        return UserIntel(
            user=user,
            uid=uid,
            gid=gid,
            groups=groups,
            has_sudo=has_sudo
        )
