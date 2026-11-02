"""
core/tools/adapters/nmap_adapter.py

Nmap Port & Service Discovery Adapter — Phase 11

Features:
  - Configurable port ranges (top100, common, all, specific ports)
  - Service version detection (-sV), timing controls (-T1 through -T5)
  - Script scanning (-sC or custom scripts)
  - Parsing of tabular output into structured open ports and service models
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional

from core.tool_registry import ToolParameter
from core.tools.adapters.base import BaseToolAdapter


class NmapAdapter(BaseToolAdapter):
    """Adapter for network reconnaissance via Nmap."""

    name = "nmap_scan"
    description = (
        "Execute Nmap network reconnaissance against an authorized target. "
        "Discovers open TCP/UDP ports, listening services, and daemon versions."
    )
    category = "recon"
    safe_preview = True

    def extract_target(self, **kwargs) -> str:
        return str(kwargs.get("target", "")).strip()

    def build_command(self, **kwargs) -> List[str]:
        target = self.extract_target(**kwargs)
        if not target:
            raise ValueError("Target host or IP address is required for Nmap.")

        ports = str(kwargs.get("ports", "top100")).strip().lower()
        scan_type = str(kwargs.get("scan_type", "syn")).strip().lower()
        timing = str(kwargs.get("timing", "T4")).strip().upper()
        service_version = kwargs.get("service_version", True)
        scripts = str(kwargs.get("scripts", "")).strip()
        extra_args = str(kwargs.get("extra_args", "")).strip()

        cmd = ["nmap"]

        # Timing
        if timing in ("T0", "T1", "T2", "T3", "T4", "T5"):
            cmd.append(f"-{timing}")
        else:
            cmd.append("-T4")

        # Scan technique
        if scan_type == "connect":
            cmd.append("-sT")
        elif scan_type == "udp":
            cmd.append("-sU")
        else:
            # Default SYN scan
            cmd.append("-sS")

        # Service versioning
        if service_version:
            cmd.append("-sV")

        # Port selection
        if ports in ("top100", "top-100"):
            cmd.extend(["--top-ports", "100"])
        elif ports in ("top1000", "top-1000"):
            cmd.extend(["--top-ports", "1000"])
        elif ports in ("all", "full", "p-"):
            cmd.append("-p-")
        elif ports:
            cmd.extend(["-p", ports])

        # Scripts
        if scripts in ("default", "sC"):
            cmd.append("-sC")
        elif scripts:
            cmd.extend(["--script", scripts])

        # Treat hosts as online by default for faster local/CTF scans
        if kwargs.get("pn", True):
            cmd.append("-Pn")

        # Extra flags safely split
        if extra_args:
            cmd.extend(shlex.split(extra_args))

        cmd.append(target)
        return cmd

    def parse_output(self, stdout: str, stderr: str, returncode: Optional[int]) -> Dict[str, Any]:
        """Parse Nmap tabular text output into structured port list."""
        open_ports: List[Dict[str, Any]] = []
        os_matches: List[str] = []
        host_status = "unknown"

        # Check host status
        if "Host is up" in stdout:
            host_status = "up"
        elif "Host seems down" in stdout:
            host_status = "down"

        # Regex for open ports: 22/tcp open ssh OpenSSH 8.2p1
        port_pattern = re.compile(
            r"^(\d+)/(tcp|udp)\s+([a-zA-Z0-9\|\-]+)\s+([^\s]+)\s*(.*)$",
            re.MULTILINE,
        )

        for match in port_pattern.finditer(stdout):
            port_num = int(match.group(1))
            proto = match.group(2)
            state = match.group(3)
            service = match.group(4)
            version = match.group(5).strip()

            if "open" in state.lower():
                open_ports.append({
                    "port": port_num,
                    "protocol": proto,
                    "state": state,
                    "service": service,
                    "version": version,
                })

        # Regex for OS detection if present
        os_pattern = re.compile(r"OS details:\s*(.+)$", re.MULTILINE)
        for m in os_pattern.finditer(stdout):
            os_matches.append(m.group(1).strip())

        return {
            "host_status": host_status,
            "open_ports_count": len(open_ports),
            "open_ports": open_ports,
            "os_matches": os_matches,
            "raw_output": stdout,
        }

    def format_summary(self, parsed: Dict[str, Any], raw: Any) -> str:
        if raw.status == "dry_run":
            return super().format_summary(parsed, raw)

        if not raw.is_success:
            return f"❌ Nmap scan failed (code {raw.returncode}):\n{raw.stderr or raw.stdout}"

        ports = parsed.get("open_ports", [])
        if not ports:
            return f"Nmap scan completed for {raw.target}: No open ports detected."

        lines = [
            f"=== Nmap Discovery Results for {raw.target} ===",
            f"Host Status: {parsed.get('host_status', 'up').upper()}",
            f"Discovered {len(ports)} Open Port(s):",
        ]
        for p in ports:
            v_str = f" - {p['version']}" if p.get("version") else ""
            lines.append(f"  • {p['port']}/{p['protocol']}: {p['service']}{v_str}")

        if parsed.get("os_matches"):
            lines.append(f"OS Guesses: {', '.join(parsed['os_matches'])}")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("target", "string", "Target IP address, hostname, or CIDR to scan.", required=True),
            ToolParameter("ports", "string", "Port specification: 'top100', 'top1000', 'all', or comma-separated list like '22,80,443'.", required=False, default="top100"),
            ToolParameter("scan_type", "string", "Scan technique: 'syn', 'connect', or 'udp'.", required=False, default="syn", enum=["syn", "connect", "udp"]),
            ToolParameter("timing", "string", "Timing template from T0 (slow) to T5 (insane).", required=False, default="T4", enum=["T0", "T1", "T2", "T3", "T4", "T5"]),
            ToolParameter("service_version", "boolean", "Perform service version detection (-sV).", required=False, default=True),
            ToolParameter("scripts", "string", "NSE scripts to run ('default', 'vuln', or specific script name).", required=False, default=""),
            ToolParameter("pn", "boolean", "Skip host discovery and assume target is online (-Pn).", required=False, default=True),
            ToolParameter("extra_args", "string", "Additional nmap command flags.", required=False, default=""),
        ]
