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
    # Explicit, operator-set declaration that this engagement is authorized
    # against exactly one pinned host/URL instance (in_scope should name just
    # that one host), not a whole domain with subdomains of its own. Read by
    # BugBountyAgent.get_action_registry()/get_completion_criteria() to decide
    # -- deterministically, never by AI judgment -- whether whole-domain
    # actions (subdomain_enum, port_scan) belong in the registry at all.
    # Default False preserves every existing Scope/session's behavior exactly.
    single_host_only: bool = False
    # Stealth profile for REAL (non-lab) targets: when True, the stealth layer
    # (core/stealth.py) paces requests with jitter, backs off adaptively on
    # WAF/rate-limit signals, and gates loud high-volume scanners off in favor of
    # surgical actions -- so the agent can be pointed at an authorized bug-bounty
    # target without getting rate-limited or banned. Default False = lab speed,
    # byte-identical behavior to before. Env PENTEST_STEALTH=1 forces it on.
    stealth: bool = False
    # Real-target safety switch: many bug bounty programs PROHIBIT automated
    # scanning. False drops the loud automated scanners (wordlist dir-brute,
    # sqlmap, nuclei, dalfox) from the registry, leaving the surgical primitives
    # (coverage, http_probe, targeted probes, self_script). Default True =
    # every action available, so labs/existing scopes are unchanged.
    allow_scanners: bool = True

    def is_asset_in_scope(self, asset: str) -> Tuple[bool, str]:
        """
        Determine whether a specific asset (IP, domain, CIDR, URL, path) is authorized.
        Out-of-scope ALWAYS overrides in-scope.
        """
        clean_asset = asset.strip().lower()
        if not clean_asset:
            return False, "Asset identifier is empty."

        # BUG FIX (found via live GPU smoke test, pentest agent, 2026-11-14): the
        # model's next action after host_discovery echoed the hostname nmap
        # itself printed ("Nmap scan report for localhost (127.0.0.1)") and
        # proposed target="localhost" -- rejected as "not listed in authorized
        # in-scope boundaries" even though Scope.in_scope=["127.0.0.1"] already
        # authorizes this exact machine; it just didn't recognize the other
        # spelling of the same address. Per RFC 6761, "localhost" MUST always
        # resolve to the loopback interface and can never legitimately refer to
        # any other host -- so "localhost"/"127.0.0.1"/"::1" denote one single
        # target, not three different ones. This is a deterministic networking
        # fact applied here in the framework's own scope check, never an AI
        # judgment call about what's in scope -- handled as a small, fully
        # self-contained early check (rather than folded into the general
        # substring/CIDR matching below, none of which is meaningful for a bare
        # loopback literal anyway) so out_of_scope exclusions are checked with
        # the same alias-awareness as in_scope authorizations, and
        # out-of-scope still unconditionally overrides in-scope.
        _LOOPBACK_ALIASES = ("localhost", "127.0.0.1", "::1")
        if clean_asset in _LOOPBACK_ALIASES:
            oos_clean_set = {s.strip().lower() for s in self.out_of_scope}
            if oos_clean_set & set(_LOOPBACK_ALIASES):
                return False, (
                    f"Asset '{asset}' is explicitly prohibited by an out-of-scope "
                    f"entry for the loopback interface."
                )
            if self.in_scope:
                ins_clean_set = {s.strip().lower() for s in self.in_scope}
                if ins_clean_set & set(_LOOPBACK_ALIASES):
                    return True, "Asset is authorized (loopback alias of an authorized target)."
                # self.in_scope is non-empty but names something else entirely
                # (no loopback spelling at all) -- fall through to the general
                # logic below exactly as before this fix, so an engagement that
                # never authorized loopback at all still rejects it normally.
            else:
                # No in_scope restriction declared at all: fall through, same as
                # any other asset would when self.in_scope is empty.
                pass

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
                # BUG FIX (2026-11-25, found live: every action in a real
                # autonomous run was rejected as out_of_scope -- see work.md).
                # `candidates` (built above for the out-of-scope check) already
                # includes `extracted_host`, the hostname pulled out of a full
                # URL/host:port asset -- but this loop was only ever checking
                # the raw `clean_asset` string, so an in_scope entry naming
                # just the bare host ("x.web-security-academy.net") never
                # matched an asset that is a full URL with a path/query
                # ("https://x.web-security-academy.net/filter?category=Gifts"),
                # since that string neither equals nor ends with "." + the
                # bare host. Checking every candidate (not a loose
                # `ins_clean in clean_asset` substring match, which would
                # wrongly let "evil-x.web-security-academy.net.attacker.com"
                # match too) restores exact/suffix matching against the
                # EXTRACTED HOST, not just the untouched raw string.
                if any(cand == ins_clean or cand.endswith("." + ins_clean) for cand in candidates):
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
            "allow_scanners": self.allow_scanners,
            "prohibited_tests": self.prohibited_tests,
            "prohibited_tools": self.prohibited_tools,
            "time_window": self.time_window,
            "authorization_ref": self.authorization_ref,
            "active_testing_allowed": self.active_testing_allowed,
            "recon_only_authorized": self.recon_only_authorized,
            "active_network_allowed": self.active_network_allowed,
            "state_changes_allowed": self.state_changes_allowed,
            "single_host_only": self.single_host_only,
            "stealth": self.stealth,
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
            single_host_only=bool(data.get("single_host_only", False)),
            stealth=bool(data.get("stealth", False)),
        )

