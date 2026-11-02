"""
core/os_detector.py

Operating System and package manager detection for Multi-Agent Security Assessment Framework.
Detects:
- Windows (winget, choco)
- Linux Debian / Ubuntu (apt)
- macOS (brew)
- Other Linux distributions (pacman, dnf, yum)
"""

import sys
import platform
import shutil
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OSInfo:
    system: str              # 'Windows', 'Linux', 'Darwin'
    friendly_name: str       # 'Windows 11', 'Ubuntu 24.04', 'macOS Sonoma', etc.
    distro: str              # 'debian', 'ubuntu', 'windows', 'macos', 'arch', 'fedora', 'unknown'
    package_manager: str     # 'winget', 'choco', 'apt', 'brew', 'pacman', 'none'
    is_admin: bool           # True if running with elevated privileges
    shell: str               # 'powershell.exe' / 'cmd.exe' / 'bash' / 'zsh'


def is_admin_or_root() -> bool:
    """Check if current process has administrative or root privileges."""
    if platform.system() == "Windows":
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    else:
        return os.geteuid() == 0 if hasattr(os, "geteuid") else False


def detect_os() -> OSInfo:
    """Detect current operating system and preferred package manager."""
    sys_name = platform.system()
    distro = "unknown"
    pkg_mgr = "none"
    shell = os.environ.get("SHELL", "bash")

    if sys_name == "Windows":
        distro = "windows"
        release = platform.release()
        friendly = f"Windows {release}"
        shell = "powershell.exe" if shutil.which("powershell.exe") else "cmd.exe"

        # Detect package manager
        if shutil.which("winget"):
            pkg_mgr = "winget"
        elif shutil.which("choco"):
            pkg_mgr = "choco"

    elif sys_name == "Darwin":
        distro = "macos"
        mac_ver = platform.mac_ver()[0]
        friendly = f"macOS {mac_ver}"
        shell = os.environ.get("SHELL", "/bin/zsh")
        if shutil.which("brew"):
            pkg_mgr = "brew"

    elif sys_name == "Linux":
        # Inspect /etc/os-release
        release_file = "/etc/os-release"
        pretty_name = "Linux"
        id_like = ""
        distro_id = ""

        if os.path.exists(release_file):
            try:
                with open(release_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("PRETTY_NAME="):
                            pretty_name = line.split("=", 1)[1].strip().strip('"')
                        elif line.startswith("ID="):
                            distro_id = line.split("=", 1)[1].strip().strip('"').lower()
                        elif line.startswith("ID_LIKE="):
                            id_like = line.split("=", 1)[1].strip().strip('"').lower()
            except Exception:
                pass

        friendly = pretty_name
        if "ubuntu" in distro_id or "ubuntu" in id_like:
            distro = "ubuntu"
            pkg_mgr = "apt"
        elif "debian" in distro_id or "debian" in id_like:
            distro = "debian"
            pkg_mgr = "apt"
        elif "arch" in distro_id or "arch" in id_like:
            distro = "arch"
            pkg_mgr = "pacman"
        elif "fedora" in distro_id or "rhel" in id_like or "centos" in id_like:
            distro = "fedora"
            pkg_mgr = "dnf" if shutil.which("dnf") else "yum"
        else:
            distro = distro_id or "linux"
            if shutil.which("apt"):
                pkg_mgr = "apt"
            elif shutil.which("pacman"):
                pkg_mgr = "pacman"
            elif shutil.which("dnf"):
                pkg_mgr = "dnf"

    else:
        friendly = f"Unknown ({sys_name})"

    return OSInfo(
        system=sys_name,
        friendly_name=friendly,
        distro=distro,
        package_manager=pkg_mgr,
        is_admin=is_admin_or_root(),
        shell=shell
    )

