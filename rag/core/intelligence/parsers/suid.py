"""
core/intelligence/parsers/suid.py

Parser plugin for SUID/SGID privileged binaries (GTFOBins attack surfaces).
"""

import re
from typing import Dict, Any, List, Set
from core.intelligence.base import BaseParser
from core.intelligence.models import SUIDIntel


# Known standard/benign POSIX SUID binaries
STANDARD_POSIX_SUID: Set[str] = {
    "/usr/bin/passwd", "/usr/bin/su", "/usr/bin/chfn", "/usr/bin/chsh",
    "/usr/bin/gpasswd", "/usr/bin/newgrp", "/usr/bin/mount", "/usr/bin/umount",
    "/bin/mount", "/bin/umount", "/bin/su", "/usr/lib/openssh/ssh-keysign"
}

# Known GTFOBins / high-risk privilege escalation binaries
INTERESTING_SUID_NAMES: Set[str] = {
    "run", "cat", "fortune", "bash-suid", "sh-suid", "exec-suid", "python-suid",
    "bash", "sh", "python", "python3", "perl", "ruby", "find", "vim", "nano",
    "nmap", "awk", "gawk", "sed", "env", "pkexec", "sudo", "docker", "tar", "zip"
}


class SUIDParser(BaseParser):
    """
    Parses SUID/SGID executable findings from 'find / -perm -4000' or 'ls -l' listings.
    """

    @property
    def name(self) -> str:
        return "suid"

    def parse(self, text: str, session_state: Dict[str, Any]) -> SUIDIntel:
        suid_binaries: List[str] = []

        # Find paths typically associated with SUID sweeps
        paths = re.findall(
            r'(/(?:challenge|usr|bin|sbin|opt|tmp|home|nix|var)[^\s\n\(\)]+)',
            text
        )

        for raw_path in paths:
            path = raw_path.rstrip(",.:;'\"")
            basename = path.split("/")[-1]

            # 1. Obvious challenge or elevated wrappers
            if "/challenge/" in path or "/opt/" in path:
                if path not in suid_binaries:
                    suid_binaries.append(path)
            # 2. Match known GTFOBins or interesting executable names
            elif basename in INTERESTING_SUID_NAMES or any(path.endswith(f"/{b}") for b in ("bash-suid", "sh-suid", "exec-suid", "python-suid")):
                if path not in suid_binaries:
                    suid_binaries.append(path)
            # 3. Non-standard /usr/bin or /bin paths
            elif (path.startswith("/usr/bin/") or path.startswith("/usr/sbin/") or path.startswith("/bin/")) and path not in STANDARD_POSIX_SUID:
                # If command context indicates SUID search:
                if "4000" in text or "perm" in text or "suid" in text.lower():
                    if path not in suid_binaries:
                        suid_binaries.append(path)

        return SUIDIntel(suid_binaries=suid_binaries)
