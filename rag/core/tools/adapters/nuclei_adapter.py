"""
core/tools/adapters/nuclei_adapter.py

Nuclei Vulnerability & Misconfiguration Scanner Adapter — Phase 11

Features:
  - Template-based vulnerability auditing
  - Severity filtering (critical, high, medium, low, info)
  - Tag selection (cve, rce, sqli, lfi, exposure, config)
  - JSONL parsing of matched templates and extracted artifacts
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from core.tool_registry import ToolParameter
from core.tools.adapters.base import BaseToolAdapter


class NucleiAdapter(BaseToolAdapter):
    """Adapter for vulnerability scanning via ProjectDiscovery Nuclei."""

    name = "nuclei_scan"
    description = (
        "Execute template-based vulnerability assessment against an authorized target. "
        "Audits for known CVEs, security misconfigurations, and sensitive data exposures."
    )
    category = "audit"
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
            raise ValueError("Target URL or host is required for Nuclei.")

        severity = str(kwargs.get("severity", "critical,high,medium")).strip().lower()
        tags = str(kwargs.get("tags", "")).strip().lower()
        templates = str(kwargs.get("templates", "")).strip()
        rate_limit = kwargs.get("rate_limit", 150)
        extra_args = str(kwargs.get("extra_args", "")).strip()

        cmd = ["nuclei", "-u", target, "-silent", "-no-color", "-jsonl"]

        if severity:
            cmd.extend(["-severity", severity])

        if tags:
            cmd.extend(["-tags", tags])

        if templates:
            cmd.extend(["-t", templates])

        if rate_limit:
            cmd.extend(["-rate-limit", str(rate_limit)])

        if extra_args:
            cmd.extend(shlex.split(extra_args))

        return cmd

    def parse_output(self, stdout: str, stderr: str, returncode: Optional[int]) -> Dict[str, Any]:
        """Parse Nuclei JSONL stream into structured findings."""
        findings: List[Dict[str, Any]] = []
        severity_counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                info = item.get("info", {})
                sev = info.get("severity", "unknown").lower()
                template_id = item.get("template-id", "unknown")
                name = info.get("name", template_id)
                matched_at = item.get("matched-at", item.get("host", ""))
                extracted = item.get("extracted-results", [])

                if sev in severity_counts:
                    severity_counts[sev] += 1

                findings.append({
                    "template_id": template_id,
                    "name": name,
                    "severity": sev,
                    "matched_at": matched_at,
                    "extracted": extracted,
                    "description": info.get("description", ""),
                    "cve_id": info.get("classification", {}).get("cve-id", []),
                })
            except json.JSONDecodeError:
                # Handle non-JSON lines (e.g. error banners)
                continue

        return {
            "total_findings": len(findings),
            "severity_counts": severity_counts,
            "findings": findings,
            "raw_output": stdout,
        }

    def format_summary(self, parsed: Dict[str, Any], raw: Any) -> str:
        if raw.status == "dry_run":
            return super().format_summary(parsed, raw)

        if not raw.is_success and not parsed.get("findings"):
            return f"❌ Nuclei scan failed (code {raw.returncode}):\n{raw.stderr or raw.stdout}"

        findings = parsed.get("findings", [])
        if not findings:
            return f"Nuclei scan completed for {raw.target}: No vulnerabilities matching criteria were found."

        counts = parsed.get("severity_counts", {})
        lines = [
            f"=== Nuclei Vulnerability Findings for {raw.target} ===",
            f"Summary: Critical: {counts.get('critical', 0)}, High: {counts.get('high', 0)}, Medium: {counts.get('medium', 0)}, Low: {counts.get('low', 0)}",
            "Matched Findings:",
        ]
        for f in findings[:15]:  # Limit top 15 to stay within token budget
            sev_badge = f"[{f['severity'].upper()}]"
            cve_str = f" ({', '.join(f['cve_id'])})" if f.get("cve_id") else ""
            lines.append(f"  • {sev_badge} {f['name']}{cve_str} -> {f['matched_at']}")

        if len(findings) > 15:
            lines.append(f"  ... and {len(findings) - 15} additional findings.")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("target", "string", "Target URL or hostname to scan.", required=True),
            ToolParameter("severity", "string", "Severity filter: comma-separated list of critical,high,medium,low,info.", required=False, default="critical,high,medium"),
            ToolParameter("tags", "string", "Tags to filter templates (e.g. 'cve', 'exposure', 'misconfig', 'rce').", required=False, default=""),
            ToolParameter("templates", "string", "Path or identifier of specific templates/directories to run.", required=False, default=""),
            ToolParameter("rate_limit", "integer", "Maximum HTTP requests per second.", required=False, default=150),
            ToolParameter("extra_args", "string", "Additional nuclei command flags.", required=False, default=""),
        ]
