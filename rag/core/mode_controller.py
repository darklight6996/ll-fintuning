"""
core/mode_controller.py

Assignment Mode and Scope Enforcement Controller — Phase 11

Enforces operational boundaries for cybersecurity assessments:
  1. AssignmentMode:
     - MANUAL: AI proposes commands, dry-run only, safe preview enabled.
       Operator manually runs commands on their terminal. Required for real
       client engagements, bug bounty programs, and environments where automated
       scanning is strictly prohibited.
     - AUTOMATED: Autonomous command execution enabled, dry-run disabled.
       For use ONLY in authorized local lab environments, CTF challenges,
       HackTheBox, TryHackMe, or explicitly approved red team automation ranges.
  2. ScopeEnforcer:
     - Strict target validation against approved IP addresses, CIDR blocks,
       domains, and hostnames.
     - Blocks commands targeting unauthorized hosts, localhost/127.0.0.1 (unless
       whitelisted), or out-of-scope subnets.
     - Load scope definitions from files (JSON, YAML, text) or programmatic lists.
"""

from __future__ import annotations

import enum
import ipaddress
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse


class AssignmentMode(str, enum.Enum):
    """Execution mode for cybersecurity engagements."""
    MANUAL = "MANUAL"          # AI proposes commands; operator runs them manually
    AUTOMATED = "AUTOMATED"    # AI executes commands directly (labs / CTF / authorized only)


@dataclass
class ScopeDefinition:
    """Holds parsed scope rules for targets."""
    ips: Set[ipaddress.IPv4Address | ipaddress.IPv6Address] = field(default_factory=set)
    cidrs: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = field(default_factory=list)
    domains: Set[str] = field(default_factory=set)            # Exact domains (e.g. target.com)
    wildcard_domains: List[str] = field(default_factory=list) # e.g. *.target.com -> .target.com
    excluded_ips: Set[ipaddress.IPv4Address | ipaddress.IPv6Address] = field(default_factory=set)
    excluded_cidrs: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = field(default_factory=list)
    excluded_domains: Set[str] = field(default_factory=set)
    allow_localhost: bool = False


class ScopeEnforcer:
    """
    Validates whether potential targets (IPs, CIDRs, hostnames, URLs) are in scope.
    Guarantees no execution occurs against unauthorized assets.
    """

    def __init__(self, allow_localhost: bool = False):
        self.scope = ScopeDefinition(allow_localhost=allow_localhost)
        self._raw_entries: List[str] = []

    def add_target(self, target_str: str) -> bool:
        """
        Add a single target specification to the scope.
        Supports:
          - IPv4 / IPv6 addresses: 10.10.10.5
          - CIDR blocks: 10.10.10.0/24, 192.168.1.0/24
          - Domains: example.com, test.local
          - Wildcards: *.example.com
          - Exclusions prefixed with '!': !10.10.10.1, !*.internal.example.com
        """
        target_str = target_str.strip()
        if not target_str or target_str.startswith("#"):
            return False

        self._raw_entries.append(target_str)
        is_exclude = target_str.startswith("!")
        clean_target = target_str[1:].strip() if is_exclude else target_str

        # 1. Try CIDR network
        if "/" in clean_target:
            try:
                network = ipaddress.ip_network(clean_target, strict=False)
                if is_exclude:
                    self.scope.excluded_cidrs.append(network)
                else:
                    self.scope.cidrs.append(network)
                return True
            except ValueError:
                pass

        # 2. Try single IP
        try:
            ip = ipaddress.ip_address(clean_target)
            if is_exclude:
                self.scope.excluded_ips.add(ip)
            else:
                self.scope.ips.add(ip)
            return True
        except ValueError:
            pass

        # 3. Domain or Wildcard
        norm_domain = clean_target.lower()
        if norm_domain.startswith("*."):
            suffix = norm_domain[1:]  # Keep leading dot: .example.com
            if is_exclude:
                self.scope.excluded_domains.add(suffix)
            else:
                self.scope.wildcard_domains.append(suffix)
            return True
        else:
            if is_exclude:
                self.scope.excluded_domains.add(norm_domain)
            else:
                self.scope.domains.add(norm_domain)
            return True

    def load_from_list(self, targets: List[str]) -> int:
        """Load multiple target strings."""
        count = 0
        for t in targets:
            if self.add_target(t):
                count += 1
        return count

    def load_from_file(self, file_path: str) -> int:
        """
        Load scope targets from a text or JSON file.
        Lines starting with '#' are treated as comments.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Scope file not found: {file_path}")

        count = 0
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                targets = []
                if isinstance(data, list):
                    targets = data
                elif isinstance(data, dict):
                    targets = data.get("scope", []) + data.get("targets", [])
                    exclusions = [f"!{e}" for e in data.get("excluded", [])]
                    targets.extend(exclusions)
                for t in targets:
                    if isinstance(t, str) and self.add_target(t):
                        count += 1
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if self.add_target(line):
                        count += 1
        return count

    def extract_host(self, target_str: str) -> str:
        """Extract plain hostname or IP from target string, URL, or host:port."""
        target_str = target_str.strip()
        # Handle URLs
        if "://" in target_str:
            parsed = urlparse(target_str)
            target_str = parsed.hostname or target_str
        elif "/" in target_str and not re.search(r"/\d{1,2}$", target_str):
            # Path without protocol e.g. target.com/admin
            target_str = target_str.split("/")[0]

        # Strip port if present (and not a CIDR)
        if ":" in target_str and not target_str.count(":") > 1: # not IPv6
            target_str = target_str.split(":")[0]

        return target_str

    def validate_target(self, target_input: str) -> Tuple[bool, str]:
        """
        Validate whether a target is permitted under the active scope.
        Returns:
            (True, "In scope") or (False, "Reason for rejection")
        """
        if not target_input or not target_input.strip():
            return False, "Empty target specification."

        host = self.extract_host(target_input).strip()
        if not host:
            return False, f"Could not extract valid host from '{target_input}'."

        # Check localhost / loopback
        if host in ("localhost", "127.0.0.1", "::1"):
            if self.scope.allow_localhost:
                return True, "Localhost allowed by scope configuration."
            return False, "Localhost (127.0.0.1) target is blocked. Enable allow_localhost explicitly if intended."

        # Try evaluating as IP address
        try:
            ip = ipaddress.ip_address(host)

            # Check explicit exclusions
            if ip in self.scope.excluded_ips:
                return False, f"Target IP {ip} is explicitly excluded in scope."
            for exc_cidr in self.scope.excluded_cidrs:
                if ip in exc_cidr:
                    return False, f"Target IP {ip} belongs to excluded CIDR {exc_cidr}."

            # Check exact IP match
            if ip in self.scope.ips:
                return True, f"Target IP {ip} is in scope (exact match)."

            # Check CIDR match
            for cidr in self.scope.cidrs:
                if ip in cidr:
                    return True, f"Target IP {ip} is in scope (matched CIDR {cidr})."

            return False, f"Target IP {ip} is not in authorized scope."
        except ValueError:
            # Not an IP address, evaluate as domain / hostname
            pass

        norm_domain = host.lower()

        # Check explicit domain exclusions
        if norm_domain in self.scope.excluded_domains:
            return False, f"Target domain '{norm_domain}' is explicitly excluded in scope."
        for exc_wild in self.scope.excluded_domains:
            if exc_wild.startswith(".") and (norm_domain == exc_wild.lstrip(".") or norm_domain.endswith(exc_wild)):
                return False, f"Target domain '{norm_domain}' matches excluded wildcard '{exc_wild}'."

        # Check exact domain match
        if norm_domain in self.scope.domains:
            return True, f"Target domain '{norm_domain}' is in scope (exact match)."

        # Check wildcard domain match (.example.com matches sub.example.com and example.com)
        for wild in self.scope.wildcard_domains:
            if norm_domain == wild.lstrip(".") or norm_domain.endswith(wild):
                return True, f"Target domain '{norm_domain}' is in scope (matched wildcard '*{wild}')."

        return False, f"Target domain '{norm_domain}' is not in authorized scope."

    def is_in_scope(self, target_input: str) -> bool:
        """Convenience boolean check for scope membership."""
        ok, _ = self.validate_target(target_input)
        return ok

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of active scope rules."""
        return {
            "total_entries": len(self._raw_entries),
            "ips_count": len(self.scope.ips),
            "cidrs_count": len(self.scope.cidrs),
            "domains_count": len(self.scope.domains),
            "wildcard_domains_count": len(self.scope.wildcard_domains),
            "exclusions_count": (
                len(self.scope.excluded_ips)
                + len(self.scope.excluded_cidrs)
                + len(self.scope.excluded_domains)
            ),
            "allow_localhost": self.scope.allow_localhost,
        }


class ModeController:
    """
    Central controller tracking operation mode (MANUAL vs AUTOMATED) and scope.
    Guarantees that automated execution is gated behind scope approval and explicit mode.
    """

    def __init__(
        self,
        default_mode: AssignmentMode = AssignmentMode.MANUAL,
        scope_enforcer: Optional[ScopeEnforcer] = None,
    ):
        self._mode = default_mode
        self._scope = scope_enforcer or ScopeEnforcer()
        self._session_modes: Dict[str, AssignmentMode] = {}

    @property
    def mode(self) -> AssignmentMode:
        return self._mode

    @mode.setter
    def mode(self, new_mode: AssignmentMode) -> None:
        self._mode = new_mode

    @property
    def scope(self) -> ScopeEnforcer:
        return self._scope

    def set_session_mode(self, session_id: str, mode: AssignmentMode) -> None:
        """Set execution mode for a specific session."""
        self._session_modes[session_id] = mode

    def get_session_mode(self, session_id: Optional[str] = None) -> AssignmentMode:
        """Retrieve execution mode for session, falling back to global mode."""
        if session_id and session_id in self._session_modes:
            return self._session_modes[session_id]
        return self._mode

    def is_automated(self, session_id: Optional[str] = None) -> bool:
        return self.get_session_mode(session_id) == AssignmentMode.AUTOMATED

    def is_manual(self, session_id: Optional[str] = None) -> bool:
        return self.get_session_mode(session_id) == AssignmentMode.MANUAL

    def authorize_execution(
        self,
        target: str,
        session_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate if a command can be executed against a target.
        Rules:
          1. If mode is MANUAL -> REJECT (commands must be manually run by operator).
          2. If target is empty or out-of-scope -> REJECT.
          3. If mode is AUTOMATED and target is in scope -> AUTHORIZED.
        """
        active_mode = self.get_session_mode(session_id)
        if active_mode == AssignmentMode.MANUAL:
            return False, (
                "Execution rejected: System is in MANUAL mode. "
                "Automated scanning is prohibited on this engagement. "
                "Review the previewed command and execute manually."
            )

        if not target or not target.strip():
            return False, "Execution rejected: No target specified for scope validation."

        in_scope, reason = self._scope.validate_target(target)
        if not in_scope:
            return False, f"Execution rejected: Target violation: {reason}"

        return True, f"Execution authorized: {reason}"


# ===========================================================================
# SINGLETON ACCESSOR
# ===========================================================================

_MODE_CONTROLLER: Optional[ModeController] = None


def get_mode_controller() -> ModeController:
    """Return process-wide ModeController singleton."""
    global _MODE_CONTROLLER
    if _MODE_CONTROLLER is None:
        _MODE_CONTROLLER = ModeController()
    return _MODE_CONTROLLER
