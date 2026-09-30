"""
core/intelligence/models.py

Strongly-typed dataclass representations for cybersecurity forensic artifacts
and session intelligence extracted from live shell/terminal output.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class UserIntel:
    """Discovered user identity, privileges, and group memberships."""
    user: Optional[str] = None
    uid: Optional[int] = None
    gid: Optional[int] = None
    groups: List[str] = field(default_factory=list)
    has_sudo: bool = False


@dataclass
class KernelIntel:
    """Operating system and kernel version details."""
    kernel: Optional[str] = None
    os_name: Optional[str] = None
    architecture: Optional[str] = None


@dataclass
class PortIntel:
    """Discovered network ports."""
    open_ports: List[str] = field(default_factory=list)


@dataclass
class BannerIntel:
    """Network service banners and protocol fingerprints."""
    network_services: List[str] = field(default_factory=list)


@dataclass
class BinaryIntel:
    """Discovered custom or challenge binaries."""
    custom_binaries: List[str] = field(default_factory=list)
    standard_binaries: List[str] = field(default_factory=list)


@dataclass
class SUIDIntel:
    """SUID and SGID binaries discovered on system."""
    suid_binaries: List[str] = field(default_factory=list)


@dataclass
class StringsIntel:
    """Critical strings and disassembled tokens from binary analysis."""
    strings_findings: List[str] = field(default_factory=list)


@dataclass
class FlagIntel:
    """CTF flag content, target flag file permissions, and tokens."""
    flag_permissions: Optional[str] = None
    flag_content: Optional[str] = None


@dataclass
class FilesystemIntel:
    """Current working directory and writeable filesystem paths."""
    pwd: Optional[str] = None
    writable_paths: List[str] = field(default_factory=list)


@dataclass
class SessionIntelligence:
    """
    Unified intelligence container combining all parser outputs.
    Provides complete backward compatibility with existing dictionary consumers.
    """
    user_intel: UserIntel = field(default_factory=UserIntel)
    kernel_intel: KernelIntel = field(default_factory=KernelIntel)
    port_intel: PortIntel = field(default_factory=PortIntel)
    banner_intel: BannerIntel = field(default_factory=BannerIntel)
    binary_intel: BinaryIntel = field(default_factory=BinaryIntel)
    suid_intel: SUIDIntel = field(default_factory=SUIDIntel)
    strings_intel: StringsIntel = field(default_factory=StringsIntel)
    flag_intel: FlagIntel = field(default_factory=FlagIntel)
    filesystem_intel: FilesystemIntel = field(default_factory=FilesystemIntel)
    raw_notes_text: List[str] = field(default_factory=list)
    custom_facts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dictionary format used across prompt and context builders."""
        return {
            "user": self.user_intel.user,
            "groups": list(self.user_intel.groups),
            "has_sudo": self.user_intel.has_sudo,
            "kernel": self.kernel_intel.kernel,
            "pwd": self.filesystem_intel.pwd,
            "suid_binaries": list(self.suid_intel.suid_binaries),
            "custom_binaries": list(self.binary_intel.custom_binaries),
            "strings_findings": list(self.strings_intel.strings_findings),
            "flag_permissions": self.flag_intel.flag_permissions,
            "flag_content": self.flag_intel.flag_content,
            "open_ports": list(self.port_intel.open_ports),
            "network_services": list(self.banner_intel.network_services),
            "raw_notes_text": list(self.raw_notes_text),
            "custom_facts": dict(self.custom_facts)
        }
