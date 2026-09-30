"""
test_intelligence_parsers.py

Unit test suite verifying Phase 5 Intelligence Extraction Refactor:
1. Individual tests for each of the 9 domain-specific parsers.
2. Extensible plugin registry in IntelligenceExtractor.
3. Custom parser registration and error isolation.
4. Backward-compatible dictionary schema generation.
"""

import sys
from pathlib import Path
import pytest

CORE_DIR = Path(__file__).resolve().parent
RAG_DIR = CORE_DIR.parent
ROOT_DIR = RAG_DIR.parent

for p in (ROOT_DIR, RAG_DIR, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.intelligence.base import BaseParser
from core.intelligence.models import (
    UserIntel,
    KernelIntel,
    PortIntel,
    BannerIntel,
    BinaryIntel,
    SUIDIntel,
    StringsIntel,
    FlagIntel,
    FilesystemIntel,
    SessionIntelligence
)
from core.intelligence.parsers import (
    UserParser,
    KernelParser,
    PortParser,
    BannerParser,
    BinaryParser,
    SUIDParser,
    StringsParser,
    FlagParser,
    FilesystemParser
)
from core.intelligence.extractor import IntelligenceExtractor


# ==========================================================
# 1. INDIVIDUAL PARSER UNIT TESTS
# ==========================================================

def test_user_parser():
    parser = UserParser()
    assert parser.name == "user"

    sample = "uid=1000(hacker) gid=1000(hacker) groups=1000(hacker),27(sudo),100(users)"
    intel = parser.parse(sample, {})
    assert isinstance(intel, UserIntel)
    assert intel.user == "hacker"
    assert intel.uid == 1000
    assert intel.gid == 1000
    assert "27(sudo)" in intel.groups
    assert intel.has_sudo is True

    # Fallback whoami
    sample_whoami = "whoami\nroot"
    intel_root = parser.parse(sample_whoami, {})
    assert intel_root.user == "root"
    assert intel_root.has_sudo is True


def test_kernel_parser():
    parser = KernelParser()
    assert parser.name == "kernel"

    sample = "Linux ctf-box 5.15.0-generic #1 SMP x86_64 GNU/Linux"
    intel = parser.parse(sample, {})
    assert isinstance(intel, KernelIntel)
    assert "Linux ctf-box 5.15.0-generic" in intel.kernel
    assert intel.os_name == "Linux"
    assert intel.architecture == "x86_64"


def test_port_parser():
    parser = PortParser()
    assert parser.name == "ports"

    sample = """
    PORT     STATE SERVICE
    22/tcp   open  ssh
    80/tcp   open  http
    8080/tcp open  http-proxy
    Connecting to port 443
    """
    intel = parser.parse(sample, {})
    assert isinstance(intel, PortIntel)
    assert any("22/tcp (open) - ssh" in p for p in intel.open_ports)
    assert any("80/tcp (open) - http" in p for p in intel.open_ports)
    assert any("Port 443" in p for p in intel.open_ports)


def test_banner_parser():
    parser = BannerParser()
    assert parser.name == "banners"

    sample = "Server: Apache/2.4.41 (Ubuntu)\nOpenSSH_8.2p1 Ubuntu-4ubuntu0.5"
    intel = parser.parse(sample, {})
    assert isinstance(intel, BannerIntel)
    assert any("Apache/2.4.41" in b for b in intel.network_services)
    assert any("OpenSSH_8.2p1" in b for b in intel.network_services)


def test_binary_parser():
    parser = BinaryParser()
    assert parser.name == "binaries"

    sample = "Located challenge artifact at /challenge/run and executed ./solve"
    intel = parser.parse(sample, {})
    assert isinstance(intel, BinaryIntel)
    assert "/challenge/run" in intel.custom_binaries
    assert "./solve" in intel.custom_binaries


def test_suid_parser():
    parser = SUIDParser()
    assert parser.name == "suid"

    sample = """
    find / -perm -4000 2>/dev/null
    /usr/bin/passwd
    /usr/bin/python-suid
    /challenge/run
    /usr/bin/find
    """
    intel = parser.parse(sample, {})
    assert isinstance(intel, SUIDIntel)
    assert "/challenge/run" in intel.suid_binaries
    assert "/usr/bin/python-suid" in intel.suid_binaries
    assert "/usr/bin/find" in intel.suid_binaries
    # Standard passwd should not be flagged as a privileged escalation target
    assert "/usr/bin/passwd" not in intel.suid_binaries


def test_strings_parser():
    parser = StringsParser()
    assert parser.name == "strings"

    sample = "Strings output reveals call to fortune and /bin/sh -p with system(buf)"
    intel = parser.parse(sample, {})
    assert isinstance(intel, StringsIntel)
    assert any("fortune" in f for f in intel.strings_findings)
    assert any("/bin/sh -p" in f for f in intel.strings_findings)
    assert any("system(" in f for f in intel.strings_findings)


def test_flag_parser():
    parser = FlagParser()
    assert parser.name == "flags"

    sample = """
    ls -la /flag
    -r-------- 1 root root 53 Sep 30 12:00 /flag
    Success! Captured token: pwn.college{test_token_12345}
    """
    intel = parser.parse(sample, {})
    assert isinstance(intel, FlagIntel)
    assert intel.flag_permissions == "-r--------"
    assert intel.flag_content == "pwn.college{test_token_12345}"


def test_filesystem_parser():
    parser = FilesystemParser()
    assert parser.name == "filesystem"

    sample = "hacker@dojo:/challenge$ pwd\n/challenge\nTemp files can be stored in /tmp"
    intel = parser.parse(sample, {})
    assert isinstance(intel, FilesystemIntel)
    assert intel.pwd == "/challenge"
    assert "/tmp" in intel.writable_paths


# ==========================================================
# 2. EXTRACTOR AGGREGATION & EXTENSION TESTS
# ==========================================================

def test_intelligence_extractor_e2e():
    """Verify IntelligenceExtractor aggregates all parsers into SessionIntelligence."""
    extractor = IntelligenceExtractor()

    session_state = {
        "notes": [
            {"content": "uid=1000(ctfuser) gid=1000(ctfuser) groups=1000(ctfuser)"},
            {"content": "Open port 80 detected"}
        ],
        "evidence": [
            {"output": "-r-------- 1 root root 40 /flag\nLinux box 5.4.0 amd64"}
        ],
        "completed_steps": [
            {"stdout": "Discovered SUID: /challenge/run"}
        ]
    }

    intel_obj = extractor.extract(session_state)
    assert isinstance(intel_obj, SessionIntelligence)
    assert intel_obj.user_intel.user == "ctfuser"
    assert "/challenge/run" in intel_obj.suid_intel.suid_binaries
    assert intel_obj.flag_intel.flag_permissions == "-r--------"
    assert any("80" in p for p in intel_obj.port_intel.open_ports)

    # Verify dictionary export matches legacy schema
    intel_dict = intel_obj.to_dict()
    assert intel_dict["user"] == "ctfuser"
    assert "/challenge/run" in intel_dict["suid_binaries"]
    assert intel_dict["flag_permissions"] == "-r--------"
    assert isinstance(intel_dict["raw_notes_text"], list)
    assert len(intel_dict["raw_notes_text"]) == 2


def test_custom_parser_plugin_registration():
    """Verify third-party or custom parsers can be registered dynamically."""
    class CustomCVEParser(BaseParser):
        @property
        def name(self) -> str:
            return "cve_parser"

        def parse(self, text: str, session_state):
            import re
            return re.findall(r'CVE-\d{4}-\d{4,7}', text)

    extractor = IntelligenceExtractor()
    custom_parser = CustomCVEParser()
    extractor.register_parser(custom_parser)

    assert "cve_parser" in extractor.registered_parsers
    assert extractor.get_parser("cve_parser") is custom_parser

    session_state = {
        "notes": [{"content": "Target vulnerable to CVE-2021-4034 (PwnKit)"}]
    }
    intel_obj = extractor.extract(session_state)
    assert "cve_parser" in intel_obj.custom_facts
    assert "CVE-2021-4034" in intel_obj.custom_facts["cve_parser"]

    # Clean unregister
    unregistered = extractor.unregister_parser("cve_parser")
    assert unregistered is custom_parser
    assert "cve_parser" not in extractor.registered_parsers
