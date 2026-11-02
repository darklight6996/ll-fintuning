"""
core/scope.py

First-class Scope dataclass and validator for the Security Assessment Framework.
Every phase checks the scope before executing any active action.
If a phase would exceed scope, it must halt and require explicit confirmation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import ipaddress
import re
from datetime import datetime


@dataclass
class Scope:
    """
    Engagement scope boundaries and compliance constraints.
    """
    target: str = ""
    in_scope: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    prohibited_tests: List[str] = field(default_factory=list)
    prohibited_tools: List[str] = field(default_factory=list)
    time_window: str = "Anytime"
    authorization_ref: str = "N/A"
    active_testing_allowed: bool = False
    recon_only_authorized: bool = False
    active_network_allowed: bool = False
    state_changes_allowed: bool = False

    def is_asset_in_scope(self, asset: str) -> Tuple[bool, str]:
        """
        Determine whether a specific asset (IP, domain, CIDR, URL, path) is authorized.
        Out-of-scope ALWAYS overrides in-scope.
        """
        clean_asset = asset.strip().lower()
        if not clean_asset:
            return False, "Asset identifier is empty."

        # Extract host if URL/URI or host:port format
        extracted_host = ""
        if "://" in clean_asset:
            from urllib.parse import urlparse
            try:
                parsed = urlparse(clean_asset)
                extracted_host = (parsed.hostname or "").lower()
            except Exception:
                pass
        elif "/" in clean_asset and not clean_asset.startswith("/"):
            # e.g. domain.com/path
            extracted_host = clean_asset.split("/")[0].split(":")[0]

        candidates = [clean_asset]
        if extracted_host and extracted_host != clean_asset:
            candidates.append(extracted_host)

        # 1. Check out-of-scope first (explicit prohibition)
        for oos in self.out_of_scope:
            oos_clean = oos.strip().lower()
            if not oos_clean:
                continue
            for cand in candidates:
                if cand == oos_clean or oos_clean in cand or cand.endswith("." + oos_clean):
                    return False, f"Asset '{asset}' is explicitly prohibited by out-of-scope entry: '{oos}'"
                # IP / CIDR check
                try:
                    asset_ip = ipaddress.ip_address(cand.split(":")[0].split("/")[0])
                    if "/" in oos_clean:
                        if asset_ip in ipaddress.ip_network(oos_clean, strict=False):
                            return False, f"IP '{asset}' falls into out-of-scope network: '{oos}'"
                    elif asset_ip == ipaddress.ip_address(oos_clean):
                        return False, f"IP '{asset}' matches out-of-scope IP: '{oos}'"
                except ValueError:
                    pass

        # 2. If in_scope is specified, verify membership
        if self.in_scope:
            matched = False
            for ins in self.in_scope:
                ins_clean = ins.strip().lower()
                if not ins_clean:
                    continue
                if clean_asset == ins_clean or ins_clean in clean_asset or clean_asset.endswith("." + ins_clean):
                    matched = True
                    break
                # IP / CIDR check
                try:
                    asset_ip = ipaddress.ip_address(clean_asset.split(":")[0].split("/")[0])
                    if "/" in ins_clean:
                        if asset_ip in ipaddress.ip_network(ins_clean, strict=False):
                            matched = True
                            break
                    elif asset_ip == ipaddress.ip_address(ins_clean):
                        matched = True
                        break
                except ValueError:
                    pass

            if not matched:
                return False, f"Asset '{asset}' is not listed in authorized in-scope boundaries."

        return True, "Asset is authorized."

    def is_command_allowed(self, command: str) -> Tuple[bool, str]:
        """
        Check if an intended command uses prohibited tools or tests.
        """
        clean_cmd = command.strip().lower()

        # Check prohibited tools
        for tool in self.prohibited_tools:
            t_clean = tool.strip().lower()
            if not t_clean:
                continue
            pattern = rf"(^|[/\s\\]){re.escape(t_clean)}(\.exe)?(\s|$)"
            if re.search(pattern, clean_cmd):
                return False, f"Command uses prohibited tool: '{tool}'"

        # Check prohibited tests
        dos_keywords = ["forkbomb", "dos", "ddos", "slowloris", "flood", ":(){ :|:& };:"]
        if any("dos" in pt.lower() for pt in self.prohibited_tests):
            for kw in dos_keywords:
                if kw in clean_cmd:
                    return False, f"Command contains potential Denial of Service indicator: '{kw}'"

        cred_keywords = ["mimikatz", "sekurlsa", "lsass", "procdump"]
        if any("credential" in pt.lower() or "cred" in pt.lower() for pt in self.prohibited_tests):
            for kw in cred_keywords:
                if kw in clean_cmd:
                    return False, f"Command contains credential dumping indicator: '{kw}'"

        return True, "Command compliance verified."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "in_scope": self.in_scope,
            "out_of_scope": self.out_of_scope,
            "prohibited_tests": self.prohibited_tests,
            "prohibited_tools": self.prohibited_tools,
            "time_window": self.time_window,
            "authorization_ref": self.authorization_ref,
            "active_testing_allowed": self.active_testing_allowed,
            "recon_only_authorized": self.recon_only_authorized,
            "active_network_allowed": self.active_network_allowed,
            "state_changes_allowed": self.state_changes_allowed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scope":
        return cls(
            target=data.get("target", ""),
            in_scope=list(data.get("in_scope", [])),
            out_of_scope=list(data.get("out_of_scope", [])),
            prohibited_tests=list(data.get("prohibited_tests", [])),
            prohibited_tools=list(data.get("prohibited_tools", [])),
            time_window=data.get("time_window", "Anytime"),
            authorization_ref=data.get("authorization_ref", "N/A"),
            active_testing_allowed=bool(data.get("active_testing_allowed", False)),
            recon_only_authorized=bool(data.get("recon_only_authorized", False)),
            active_network_allowed=bool(data.get("active_network_allowed", False)),
            state_changes_allowed=bool(data.get("state_changes_allowed", False)),
        )

