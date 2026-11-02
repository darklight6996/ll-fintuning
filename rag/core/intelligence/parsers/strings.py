"""
core/intelligence/parsers/strings.py

Parser plugin for extracted binary strings, decompiled tokens, and vulnerable calls.
"""

import re
from typing import Dict, Any, List
from core.intelligence.base import BaseParser
from core.intelligence.models import StringsIntel


class StringsParser(BaseParser):
    """
    Identifies high-value tokens, hardcoded paths, and insecure API calls
    (system(), /bin/sh -p, unquoted commands, tokens) from string outputs.
    """

    @property
    def name(self) -> str:
        return "strings"

    def parse(self, text: str, session_state: Dict[str, Any]) -> StringsIntel:
        strings_findings: List[str] = []

        # 1. Unquoted or relative binary invocation (e.g. 'fortune')
        if "fortune" in text:
            strings_findings.append("fortune (called without absolute path)")

        # 2. Privileged shell invocation
        if "/bin/sh -p" in text or "/bin/bash -p" in text:
            strings_findings.append("/bin/sh -p (preserves elevated SUID privileges)")

        # 3. Exec wrapper patterns
        if "exec-suid" in text:
            strings_findings.append("exec-suid wrapper")

        # 4. Insecure C library function calls
        insecure_calls = ["system(", "popen(", "strcpy(", "strcat(", "gets(", "execve("]
        for call in insecure_calls:
            if call in text:
                finding = f"Vulnerable function reference: {call}"
                if finding not in strings_findings:
                    strings_findings.append(finding)

        # 5. Potential credentials/tokens
        cred_matches = re.findall(
            r'(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*[\'"]?([^\s\'"]{6,40})',
            text,
            re.IGNORECASE
        )
        for cred in cred_matches:
            strings_findings.append(f"Hardcoded credential/token candidate: {cred}")

        return StringsIntel(strings_findings=strings_findings)
