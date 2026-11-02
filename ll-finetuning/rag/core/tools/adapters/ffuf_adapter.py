"""
core/tools/adapters/ffuf_adapter.py

FFUF Web Fuzzing & Directory Discovery Adapter — Phase 11

Features:
  - High-performance web endpoint and path discovery
  - Status code filtering (-fc) and matching (-mc)
  - Wordlist parameterization and extension appending
  - JSON output parsing with content length and redirect tracking
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from core.tool_registry import ToolParameter
from core.tools.adapters.base import BaseToolAdapter


class FfufAdapter(BaseToolAdapter):
    """Adapter for web endpoint fuzzing via ffuf."""

    name = "ffuf_fuzz"
    description = (
        "Execute web directory and parameter fuzzing against an authorized web application. "
        "Discovers hidden endpoints, backup files, and API routes."
    )
    category = "recon"
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
            raise ValueError("Target URL is required for ffuf.")

        # Ensure FUZZ keyword exists in URL
        if "FUZZ" not in url:
            if url.endswith("/"):
                url = f"{url}FUZZ"
            else:
                url = f"{url}/FUZZ"

        wordlist = str(kwargs.get("wordlist", "/usr/share/wordlists/dirb/common.txt")).strip()
        extensions = str(kwargs.get("extensions", "")).strip()
        filter_status = str(kwargs.get("filter_status", "404")).strip()
        match_status = str(kwargs.get("match_status", "")).strip()
        threads = kwargs.get("threads", 40)
        extra_args = str(kwargs.get("extra_args", "")).strip()

        cmd = [
            "ffuf",
            "-u", url,
            "-w", wordlist,
            "-s",            # Silent / clean
            "-o", "-",       # Output to stdout
            "-of", "json",   # JSON format
            "-t", str(threads),
        ]

        if filter_status:
            cmd.extend(["-fc", filter_status])

        if match_status:
            cmd.extend(["-mc", match_status])

        if extensions:
            cmd.extend(["-e", extensions])

        if extra_args:
            cmd.extend(shlex.split(extra_args))

        return cmd

    def parse_output(self, stdout: str, stderr: str, returncode: Optional[int]) -> Dict[str, Any]:
        """Parse ffuf JSON stdout into structured endpoint list."""
        endpoints: List[Dict[str, Any]] = []

        try:
            # ffuf output can be pure json or have trailing newline
            data = json.loads(stdout.strip())
            results = data.get("results", [])
            for r in results:
                input_data = r.get("input", {})
                fuzz_val = input_data.get("FUZZ", "")
                status = r.get("status", 0)
                length = r.get("length", 0)
                words = r.get("words", 0)
                lines = r.get("lines", 0)
                redirect = r.get("redirectlocation", "")
                target_url = r.get("url", "")

                endpoints.append({
                    "path": fuzz_val,
                    "url": target_url,
                    "status": status,
                    "length": length,
                    "words": words,
                    "lines": lines,
                    "redirect": redirect,
                })
        except Exception:
            # Fallback text parsing if JSON formatting fails
            for line in stdout.splitlines():
                if line.strip():
                    endpoints.append({"raw": line.strip()})

        return {
            "total_discovered": len(endpoints),
            "endpoints": endpoints,
            "raw_output": stdout,
        }

    def format_summary(self, parsed: Dict[str, Any], raw: Any) -> str:
        if raw.status == "dry_run":
            return super().format_summary(parsed, raw)

        if not raw.is_success and not parsed.get("endpoints"):
            return f"❌ ffuf failed (code {raw.returncode}):\n{raw.stderr or raw.stdout}"

        endpoints = parsed.get("endpoints", [])
        if not endpoints:
            return f"ffuf fuzzing completed for {raw.target}: No active endpoints matched criteria."

        lines = [
            f"=== ffuf Endpoint Discovery for {raw.target} ===",
            f"Discovered {len(endpoints)} Endpoint(s):",
        ]
        for ep in endpoints[:20]:
            if "status" in ep:
                redir = f" -> {ep['redirect']}" if ep.get("redirect") else ""
                lines.append(f"  • [HTTP {ep['status']}] {ep['path']} (Size: {ep['length']}b){redir}")
            else:
                lines.append(f"  • {ep.get('raw', '')}")

        if len(endpoints) > 20:
            lines.append(f"  ... and {len(endpoints) - 20} more endpoints.")

        return "\n".join(lines)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("url", "string", "Target URL with optional FUZZ placeholder (e.g. http://10.10.10.5/FUZZ).", required=True),
            ToolParameter("wordlist", "string", "Path to dictionary wordlist.", required=False, default="/usr/share/wordlists/dirb/common.txt"),
            ToolParameter("extensions", "string", "Comma-separated extensions to append (e.g. '.php,.txt,.html').", required=False, default=""),
            ToolParameter("filter_status", "string", "HTTP status codes to filter out (e.g. '404,403').", required=False, default="404"),
            ToolParameter("match_status", "string", "HTTP status codes to match (e.g. '200,301,302').", required=False, default=""),
            ToolParameter("threads", "integer", "Number of concurrent worker threads.", required=False, default=40),
            ToolParameter("extra_args", "string", "Additional ffuf command flags.", required=False, default=""),
        ]
