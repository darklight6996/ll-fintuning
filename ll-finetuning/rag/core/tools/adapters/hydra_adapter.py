"""
core/tools/adapters/hydra_adapter.py

Hydra Authentication & Credential Auditing Adapter — Phase 11

Features:
  - Authorized credential auditing for network services (SSH, FTP, HTTP, etc.)
  - Single credential testing or dictionary testing
  - Task concurrency throttling to prevent service denial-of-service
  - Regex parsing of validated login credentials
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from core.tool_registry import ToolParameter
from core.tools.adapters.base import BaseToolAdapter


class HydraAdapter(BaseToolAdapter):
    """Adapter for authorized credential auditing via THC-Hydra."""

    name = "hydra_audit"
    description = (
        "Execute credential verification against an authorized network service. "
        "Tests for default or weak credentials in controlled environments."
    )
    category = "audit"
    safe_preview = True

    def extract_target(self, **kwargs) -> str:
        target = str(kwargs.get("target", "")).strip()
        if "://" in target:
            parsed = urlparse(target)
            return parsed.hostname or parsed.netloc
        if ":" in target and not target.count(":") > 1:
            return target.split(":")[0]
        return target

    def build_command(self, **kwargs) -> List[str]:
        target = self.extract_target(**kwargs)
        if not target:
            raise ValueError("Target host or IP is required for Hydra.")

        service = str(kwargs.get("service", "ssh")).strip().lower()
        username = str(kwargs.get("username", "")).strip()
        user_file = str(kwargs.get("user_file", "")).strip()
        password = str(kwargs.get("password", "")).strip()
        pass_file = str(kwargs.get("pass_file", "")).strip()
        port = kwargs.get("port", 0)
        tasks = min(max(int(kwargs.get("tasks", 4)), 1), 16)
        extra_args = str(kwargs.get("extra_args", "")).strip()

        cmd = ["hydra", "-t", str(tasks), "-f"]  # -f: exit on first valid pair

        if username:
            cmd.extend(["-l", username])
        elif user_file:
            cmd.extend(["-L", user_file])
        else:
            cmd.extend(["-l", "admin"])

        if password:
            cmd.extend(["-p", password])
        elif pass_file:
            cmd.extend(["-P", pass_file])
        else:
            raise ValueError("Either 'password' or 'pass_file' must be provided.")

        if port and int(port) > 0:
            cmd.extend(["-s", str(port)])

        if extra_args:
            cmd.extend(shlex.split(extra_args))

        cmd.extend([target, service])
        return cmd

    def parse_output(self, stdout: str, stderr: str, returncode: Optional[int]) -> Dict[str, Any]:
        """Parse Hydra valid login lines."""
        valid_credentials: List[Dict[str, str]] = []

        # Example: [22][ssh] host: 10.10.10.5   login: root   password: toor
        cred_pattern = re.compile(
            r"\[(\d+)\]\[([a-zA-Z0-9\-]+)\]\s+host:\s*([^\s]+)\s+login:\s*([^\s]+)\s+password:\s*(.*)$",
            re.MULTILINE,
        )

        for m in cred_pattern.finditer(stdout):
            valid_credentials.append({
                "port": m.group(1),
                "service": m.group(2),
                "host": m.group(3),
                "login": m.group(4),
                "password": m.group(5).strip(),
            })

        return {
            "valid_credentials_count": len(valid_credentials),
            "valid_credentials": valid_credentials,
            "raw_output": stdout,
        }

    def format_summary(self, parsed: Dict[str, Any], raw: Any) -> str:
        if raw.status == "dry_run":
            return super().format_summary(parsed, raw)

        if not raw.is_success and not parsed.get("valid_credentials"):
            return f"❌ Hydra execution failed (code {raw.returncode}):\n{raw.stderr or raw.stdout}"

        creds = parsed.get("valid_credentials", [])
        if not creds:
            return f"Hydra credential audit completed for {raw.target}: No valid credentials discovered."

        lines = [
            f"=== ⚠️ Hydra Valid Credentials Discovered for {raw.target} ===",
            f"Total Valid Accounts Found: {len(creds)}",
        ]
        for c in creds:
            lines.append(f"  • Service: {c['service']} ({c['port']}) | Username: {c['login']} | Password: {c['password']}")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("target", "string", "Target host or IP address.", required=True),
            ToolParameter("service", "string", "Target network protocol/service (ssh, ftp, telnet, http-get, etc.).", required=False, default="ssh"),
            ToolParameter("username", "string", "Single username to test.", required=False, default=""),
            ToolParameter("user_file", "string", "Path to usernames wordlist.", required=False, default=""),
            ToolParameter("password", "string", "Single password to test.", required=False, default=""),
            ToolParameter("pass_file", "string", "Path to passwords wordlist.", required=False, default=""),
            ToolParameter("port", "integer", "Custom port override (0 for default).", required=False, default=0),
            ToolParameter("tasks", "integer", "Concurrent task workers (1-16).", required=False, default=4),
            ToolParameter("extra_args", "string", "Additional hydra command flags.", required=False, default=""),
        ]
