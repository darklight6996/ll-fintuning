"""
core/intelligence/parsers/filesystem.py

Parser plugin for current working directories, writable temp paths, and mount points.
"""

import re
from typing import Dict, Any, Optional, List
from core.intelligence.base import BaseParser
from core.intelligence.models import FilesystemIntel


class FilesystemParser(BaseParser):
    """
    Parses current working directory (pwd) and identifies writable staging paths.
    """

    @property
    def name(self) -> str:
        return "filesystem"

    def parse(self, text: str, session_state: Dict[str, Any]) -> FilesystemIntel:
        pwd: Optional[str] = None
        writable_paths: List[str] = []

        # 1. PWD detection
        pwd_match = re.search(r'\bpwd\s*\n\s*(/[^\s\r\n]+)', text)
        if pwd_match:
            pwd = pwd_match.group(1).strip()
        else:
            # Common prompt format e.g. 'hacker@dojo:/challenge$ ' or 'root@kali:~#'
            prompt_match = re.search(r':([/~][^\$#\s]*)\s*[\$#]', text)
            if prompt_match:
                pwd = prompt_match.group(1).strip()

        # 2. Writable staging directories mentioned in output
        common_writable = ["/tmp", "/dev/shm", "/var/tmp", "/tmp/"]
        for p in common_writable:
            clean_p = p.rstrip("/")
            if clean_p in text and clean_p not in writable_paths:
                writable_paths.append(clean_p)

        # 3. Discovered files from ls / directory listings
        discovered_files: List[str] = []
        # Pattern 3a: Long listing (-rw-r--r-- 1 bandit0 bandit0 33 May 7 readme)
        for match in re.finditer(r'[-d][-rwx]{9}\+?\s+\d+\s+\S+\s+\S+\s+\d+\s+.*?\s+([^\s/]+)', text):
            fname = match.group(1).strip()
            if fname and fname not in (".", "..") and fname not in discovered_files:
                discovered_files.append(fname)

        # Pattern 3b: Simple ls output (e.g. prompt followed by ls and file names like readme)
        ls_match = re.search(r'\bls\s*\n\s*([a-zA-Z0-9_\-\.\s]+)', text)
        if ls_match:
            candidate_tokens = ls_match.group(1).strip().split()
            for token in candidate_tokens:
                token = token.strip()
                if token and not token.startswith("$") and not token.startswith("#") and token not in discovered_files:
                    discovered_files.append(token)

        return FilesystemIntel(
            pwd=pwd,
            writable_paths=writable_paths,
            discovered_files=discovered_files
        )
