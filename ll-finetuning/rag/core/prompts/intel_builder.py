"""
core/prompts/intel_builder.py

IntelBuilder formats extracted session intelligence and forensic artifacts
into a secure, sandboxed evidence block with deduplication and token bounding.
"""

from typing import Dict, Any, Optional, List, Union
from core.intelligence.models import SessionIntelligence


class IntelBuilder:
    """
    Constructs the UNTRUSTED EVIDENCE section of the prompt.
    Enforces strict prompt-injection sandboxing and deduplication across
    ports, binaries, banners, and strings.
    """

    def __init__(self, max_items_per_field: int = 8):
        self.max_items_per_field = max_items_per_field

    def _dedupe_list(self, items: List[Any]) -> List[str]:
        """Deduplicate list preserving insertion order."""
        seen = set()
        deduped = []
        for it in items:
            s = str(it).strip()
            if s and s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    def build(
        self,
        intel: Optional[Union[Dict[str, Any], SessionIntelligence]] = None,
        max_chars: int = 1200
    ) -> str:
        """
        Build formatted and sandboxed evidence string.
        """
        if intel is None:
            return "(no live intel yet)"

        data = intel.to_dict() if isinstance(intel, SessionIntelligence) else intel

        lines = []

        # User and Privileges
        if data.get("user"):
            sudo_str = " (sudo enabled)" if data.get("has_sudo") else ""
            lines.append(f"- User: {data['user']}{sudo_str}")

        if data.get("groups"):
            groups = self._dedupe_list(data["groups"])[:self.max_items_per_field]
            lines.append(f"- Groups: {', '.join(groups)}")

        # OS & Kernel
        if data.get("kernel"):
            lines.append(f"- Kernel: {str(data['kernel'])[:80]}")

        # Current working directory
        if data.get("pwd"):
            lines.append(f"- PWD: {data['pwd']}")

        # Network Recon
        if data.get("open_ports"):
            ports = self._dedupe_list(data["open_ports"])[:self.max_items_per_field]
            lines.append(f"- Open Ports: {', '.join(ports)}")

        if data.get("network_services"):
            services = self._dedupe_list(data["network_services"])[:5]
            lines.append(f"- Network Banners: {', '.join(services)}")

        # Privilege Escalation Attack Surfaces
        if data.get("suid_binaries"):
            suid = self._dedupe_list(data["suid_binaries"])[:6]
            lines.append(f"- SUID Targets: {', '.join(suid)}")

        if data.get("custom_binaries"):
            bins = self._dedupe_list(data["custom_binaries"])[:4]
            lines.append(f"- Custom Binaries: {', '.join(bins)}")

        if data.get("strings_findings"):
            strings = self._dedupe_list(data["strings_findings"])[:4]
            lines.append(f"- Binary Strings: {', '.join(strings)}")

        # Flags & Targets
        if data.get("flag_permissions"):
            lines.append(f"- /flag Permissions: {data['flag_permissions']}")

        if data.get("flag_content"):
            lines.append(f"- Flag Discovered: {data['flag_content']}")

        body = "\n".join(lines).strip()
        if not body:
            return "(no live intel yet)"

        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "... [intel truncated]"

        sandboxed = (
            "<<<BEGIN_EVIDENCE>>>\n"
            f"{body}\n"
            "<<<END_EVIDENCE>>>"
        )
        return sandboxed
