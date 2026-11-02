"""
core/tools/adapters/gobuster_adapter.py

Gobuster Directory & DNS Enumeration Adapter — Phase 11

Features:
  - Directory enumeration (dir mode)
  - Subdomain brute-forcing (dns mode)
  - Virtual host discovery (vhost mode)
  - Regex parsing of discovery streams
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from core.tool_registry import ToolParameter
from core.tools.adapters.base import BaseToolAdapter


class GobusterAdapter(BaseToolAdapter):
    """Adapter for directory and DNS enumeration via Gobuster."""

    name = "gobuster_scan"
    description = (
        "Execute directory, subdomain, or virtual host enumeration via Gobuster. "
        "Finds exposed web paths and DNS records."
    )
    category = "recon"
    safe_preview = True

    def extract_target(self, **kwargs) -> str:
        target = str(kwargs.get("target", "")).strip()
        if "://" in target:
            parsed = urlparse(target)
            return parsed.netloc or parsed.path
        return target

    def build_command(self, **kwargs) -> List[str]:
        target = str(kwargs.get("target", "")).strip()
        if not target:
            raise ValueError("Target URL or domain is required for Gobuster.")

        mode = str(kwargs.get("mode", "dir")).strip().lower()
        if mode not in ("dir", "dns", "vhost"):
            mode = "dir"

        wordlist = str(kwargs.get("wordlist", "/usr/share/wordlists/dirb/common.txt")).strip()
        status_codes = str(kwargs.get("status_codes", "")).strip()
        threads = kwargs.get("threads", 20)
        extra_args = str(kwargs.get("extra_args", "")).strip()

        cmd = ["gobuster", mode, "-w", wordlist, "-q", "--no-color", "-t", str(threads)]

        if mode == "dns":
            cmd.extend(["-d", target])
        else:
            cmd.extend(["-u", target])

        if status_codes and mode == "dir":
            cmd.extend(["-s", status_codes])

        if extra_args:
            cmd.extend(shlex.split(extra_args))

        return cmd

    def parse_output(self, stdout: str, stderr: str, returncode: Optional[int]) -> Dict[str, Any]:
        """Parse Gobuster output lines into structured findings."""
        findings: List[Dict[str, Any]] = []

        # Example dir output: /admin (Status: 301) [Size: 194] [--> http://target/admin/]
        dir_pattern = re.compile(
            r"^(/\S+)\s+\(Status:\s*(\d+)\)(?:\s+\[Size:\s*(\d+)\])?(?:\s+\[-->\s*(\S+)\])?",
            re.MULTILINE,
        )
        # Example dns output: Found: mail.example.com
        dns_pattern = re.compile(r"^Found:\s*(\S+)", re.MULTILINE)

        for m in dir_pattern.finditer(stdout):
            findings.append({
                "path": m.group(1),
                "status": int(m.group(2)),
                "size": int(m.group(3)) if m.group(3) else None,
                "redirect": m.group(4) if m.group(4) else None,
            })

        for m in dns_pattern.finditer(stdout):
            findings.append({
                "subdomain": m.group(1),
            })

        return {
            "total_findings": len(findings),
            "findings": findings,
            "raw_output": stdout,
        }

    def format_summary(self, parsed: Dict[str, Any], raw: Any) -> str:
        if raw.status == "dry_run":
            return super().format_summary(parsed, raw)

        if not raw.is_success and not parsed.get("findings"):
            return f"❌ Gobuster failed (code {raw.returncode}):\n{raw.stderr or raw.stdout}"

        findings = parsed.get("findings", [])
        if not findings:
            return f"Gobuster scan completed for {raw.target}: No records found matching criteria."

        lines = [
            f"=== Gobuster Enumeration for {raw.target} ===",
            f"Discovered {len(findings)} Record(s):",
        ]
        for f in findings[:20]:
            if "path" in f:
                redir = f" -> {f['redirect']}" if f.get("redirect") else ""
                lines.append(f"  • {f['path']} (Status: {f['status']}){redir}")
            elif "subdomain" in f:
                lines.append(f"  • {f['subdomain']}")

        if len(findings) > 20:
            lines.append(f"  ... and {len(findings) - 20} more entries.")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("target", "string", "Target URL (for dir/vhost) or domain (for dns).", required=True),
            ToolParameter("mode", "string", "Enumeration mode: 'dir', 'dns', or 'vhost'.", required=False, default="dir", enum=["dir", "dns", "vhost"]),
            ToolParameter("wordlist", "string", "Path to wordlist file.", required=False, default="/usr/share/wordlists/dirb/common.txt"),
            ToolParameter("status_codes", "string", "Positive status codes to include (dir mode only).", required=False, default=""),
            ToolParameter("threads", "integer", "Number of concurrent threads.", required=False, default=20),
            ToolParameter("extra_args", "string", "Additional gobuster command flags.", required=False, default=""),
        ]
