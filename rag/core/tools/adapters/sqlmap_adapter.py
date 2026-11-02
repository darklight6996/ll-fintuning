"""
core/tools/adapters/sqlmap_adapter.py

SQLMap Vulnerability Assessment Adapter — Phase 11

Features:
  - Automated SQL injection verification for authorized targets
  - Safety constraints: forces --batch (non-interactive), caps risk and level
  - Parsing of backend DBMS detection and injectable parameters
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from core.tool_registry import ToolParameter
from core.tools.adapters.base import BaseToolAdapter


class SqlmapAdapter(BaseToolAdapter):
    """Adapter for authorized SQL injection testing via sqlmap."""

    name = "sqlmap_scan"
    description = (
        "Execute automated SQL injection testing against an authorized endpoint. "
        "Verifies injectable parameters and detects backend DBMS architecture."
    )
    category = "audit"
    safe_preview = True

    def extract_target(self, **kwargs) -> str:
        url = str(kwargs.get("url", "")).strip()
        if "://" in url:
            parsed = urlparse(url)
            return parsed.netloc or parsed.path
        return url

    def build_command(self, **kwargs) -> List[str]:
        url = str(kwargs.get("url", "")).strip()
        if not url:
            raise ValueError("Target URL is required for sqlmap.")

        data = str(kwargs.get("data", "")).strip()
        cookie = str(kwargs.get("cookie", "")).strip()
        technique = str(kwargs.get("technique", "BEUSTQ")).strip().upper()
        # Cap risk at 2 and level at 3 to prevent destructive payloads
        risk = min(max(int(kwargs.get("risk", 1)), 1), 2)
        level = min(max(int(kwargs.get("level", 1)), 1), 3)
        dbs = kwargs.get("dbs", False)
        extra_args = str(kwargs.get("extra_args", "")).strip()

        cmd = [
            "sqlmap",
            "-u", url,
            "--batch",       # Non-interactive mode (mandatory)
            "--technique", technique,
            f"--risk={risk}",
            f"--level={level}",
        ]

        if data:
            cmd.extend(["--data", data])

        if cookie:
            cmd.extend(["--cookie", cookie])

        if dbs:
            cmd.append("--dbs")

        if extra_args:
            cmd.extend(shlex.split(extra_args))

        return cmd

    def parse_output(self, stdout: str, stderr: str, returncode: Optional[int]) -> Dict[str, Any]:
        """Parse sqlmap output for vulnerability confirmation and DBMS details."""
        is_vulnerable = False
        dbms = "unknown"
        injectable_parameters: List[str] = []

        if "is vulnerable. Do you want to keep testing" in stdout or "sqlmap identified the following injection point" in stdout:
            is_vulnerable = True

        # Extract DBMS
        dbms_match = re.search(r"back-end DBMS(?:\s+is|\:)\s*(.+)$", stdout, re.MULTILINE | re.IGNORECASE)
        if dbms_match:
            dbms = dbms_match.group(1).strip()

        # Extract Parameter: id (GET)
        param_pattern = re.compile(r"Parameter:\s*([^\s\(]+)\s*\(([^\)]+)\)", re.IGNORECASE)
        for m in param_pattern.finditer(stdout):
            injectable_parameters.append(f"{m.group(1)} ({m.group(2)})")

        # Check for retrieved databases
        databases: List[str] = []
        dbs_section = re.search(r"available databases \[\d+\]:\s*\n((?:\[\*\]\s*.+\n)+)", stdout)
        if dbs_section:
            for line in dbs_section.group(1).splitlines():
                clean_db = line.replace("[*]", "").strip()
                if clean_db:
                    databases.append(clean_db)

        return {
            "is_vulnerable": is_vulnerable,
            "dbms": dbms,
            "injectable_parameters": list(set(injectable_parameters)),
            "databases": databases,
            "raw_output": stdout,
        }

    def format_summary(self, parsed: Dict[str, Any], raw: Any) -> str:
        if raw.status == "dry_run":
            return super().format_summary(parsed, raw)

        if not raw.is_success and not parsed.get("is_vulnerable"):
            return f"❌ SQLMap scan failed (code {raw.returncode}):\n{raw.stderr or raw.stdout}"

        if not parsed.get("is_vulnerable"):
            return f"SQLMap assessment completed for {raw.target}: Target does not appear vulnerable under current parameters."

        lines = [
            f"=== ⚠️ SQLMap Vulnerability Confirmed for {raw.target} ===",
            f"Back-End DBMS: {parsed.get('dbms', 'Detected')}",
            f"Injectable Parameter(s): {', '.join(parsed.get('injectable_parameters', ['Unspecified']))}",
        ]
        if parsed.get("databases"):
            lines.append(f"Enumerated Database(s): {', '.join(parsed['databases'])}")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("url", "string", "Target URL to test for SQL injection.", required=True),
            ToolParameter("data", "string", "POST data string (e.g. 'username=admin&pass=123').", required=False, default=""),
            ToolParameter("cookie", "string", "HTTP Cookie header value.", required=False, default=""),
            ToolParameter("level", "integer", "Level of tests to perform (1-3).", required=False, default=1),
            ToolParameter("risk", "integer", "Risk of tests to perform (1-2).", required=False, default=1),
            ToolParameter("dbs", "boolean", "Enumerate DBMS databases if vulnerable.", required=False, default=False),
            ToolParameter("extra_args", "string", "Additional sqlmap command flags.", required=False, default=""),
        ]
