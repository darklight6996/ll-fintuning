"""
core/tool_manager.py

Tool management and deterministic verification across operating systems.
Supports checking availability on PATH and installing missing tools via
OS package managers (winget/choco on Windows, apt on Debian/Ubuntu, brew on macOS).
"""

import shutil
import subprocess
import os
from typing import Dict, List, Optional, Tuple, Any
from core.os_detector import OSInfo, detect_os


# Mapping of common security assessment tools to package names per OS package manager
TOOL_PACKAGE_MAP: Dict[str, Dict[str, str]] = {
    "nmap": {
        "winget": "Insecure.Nmap",
        "choco": "nmap",
        "apt": "nmap",
        "brew": "nmap",
        "binary": "nmap"
    },
    "curl": {
        "winget": "cURL.cURL",
        "choco": "curl",
        "apt": "curl",
        "brew": "curl",
        "binary": "curl"
    },
    "nikto": {
        "choco": "nikto",
        "apt": "nikto",
        "brew": "nikto",
        "binary": "nikto"
    },
    "sqlmap": {
        "choco": "sqlmap",
        "apt": "sqlmap",
        "brew": "sqlmap",
        "binary": "sqlmap"
    },
    "gobuster": {
        "apt": "gobuster",
        "brew": "gobuster",
        "binary": "gobuster"
    },
    "ffuf": {
        "apt": "ffuf",
        "brew": "ffuf",
        "binary": "ffuf"
    },
    "whois": {
        "apt": "whois",
        "brew": "whois",
        "binary": "whois"
    },
    "dig": {
        "apt": "dnsutils",
        "brew": "bind",
        "binary": "dig"
    },
    "git": {
        "winget": "Git.Git",
        "choco": "git",
        "apt": "git",
        "brew": "git",
        "binary": "git"
    },
    "python": {
        "winget": "Python.Python.3.11",
        "choco": "python",
        "apt": "python3",
        "brew": "python",
        "binary": "python"
    },
    "subfinder": {
        "brew": "subfinder",
        "binary": "subfinder"
    },
    "nuclei": {
        "brew": "nuclei",
        "binary": "nuclei"
    },
    "apktool": {
        "apt": "apktool",
        "brew": "apktool",
        "binary": "apktool"
    },
    "trufflehog": {
        "brew": "trufflehog",
        "binary": "trufflehog"
    },
    "semgrep": {
        "apt": "semgrep",
        "brew": "semgrep",
        "binary": "semgrep"
    }
}


class ToolManager:
    """
    Manages deterministic tool verification and OS package installations.
    """

    def __init__(self, os_info: Optional[OSInfo] = None):
        self.os_info = os_info or detect_os()

    def is_tool_installed(self, tool_name: str) -> bool:
        """Check if a tool executable exists on the system PATH."""
        clean_name = tool_name.strip().lower()
        mapping = TOOL_PACKAGE_MAP.get(clean_name, {})
        bin_name = mapping.get("binary", clean_name)

        if shutil.which(bin_name):
            return True

        # Check common Windows extensions
        if self.os_info.system == "Windows":
            for ext in (".exe", ".bat", ".cmd", ".ps1"):
                if shutil.which(f"{bin_name}{ext}"):
                    return True

        return False

    def get_tool_path(self, tool_name: str) -> Optional[str]:
        """Return the absolute path of the tool executable if present."""
        clean_name = tool_name.strip().lower()
        mapping = TOOL_PACKAGE_MAP.get(clean_name, {})
        bin_name = mapping.get("binary", clean_name)

        path = shutil.which(bin_name)
        if path:
            return path

        if self.os_info.system == "Windows":
            for ext in (".exe", ".bat", ".cmd", ".ps1"):
                p = shutil.which(f"{bin_name}{ext}")
                if p:
                    return p
        return None

    def get_tool_version(self, tool_name: str) -> str:
        """Attempt to extract tool version output."""
        path = self.get_tool_path(tool_name)
        if not path:
            return "Not Installed"
        try:
            res = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
            out = res.stdout.strip() or res.stderr.strip()
            first_line = out.splitlines()[0] if out else "Installed (version unknown)"
            return first_line[:60]
        except Exception:
            return "Installed"

    def check_tools(self, tool_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Check multiple tools and return status report.
        """
        results = {}
        for tool in tool_list:
            installed = self.is_tool_installed(tool)
            path = self.get_tool_path(tool) if installed else None
            version = self.get_tool_version(tool) if installed else "Not Installed"
            results[tool] = {
                "installed": installed,
                "path": path,
                "version": version
            }
        return results

    def get_install_command(self, tool_name: str) -> Optional[str]:
        """Construct the package manager installation command for the current OS."""
        pkg_mgr = self.os_info.package_manager
        if not pkg_mgr or pkg_mgr == "none":
            return None

        clean_name = tool_name.strip().lower()
        pkg_info = TOOL_PACKAGE_MAP.get(clean_name, {})
        pkg = pkg_info.get(pkg_mgr, clean_name)

        if pkg_mgr == "winget":
            return f"winget install --id {pkg} -e --accept-source-agreements --accept-package-agreements"
        elif pkg_mgr == "choco":
            return f"choco install {pkg} -y"
        elif pkg_mgr == "apt":
            return f"sudo apt update && sudo apt install -y {pkg}"
        elif pkg_mgr == "brew":
            return f"brew install {pkg}"
        elif pkg_mgr == "pacman":
            return f"sudo pacman -S --noconfirm {pkg}"
        return None

    def install_tool(self, tool_name: str) -> Tuple[bool, str]:
        """
        Execute deterministic tool installation using system package manager.
        Requires explicit operator approval.
        """
        cmd = self.get_install_command(tool_name)
        if not cmd:
            return False, f"No automated package installer known for '{tool_name}' via {self.os_info.package_manager}."

        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
            if res.returncode == 0:
                return True, f"Successfully installed {tool_name} via {self.os_info.package_manager}."
            else:
                return False, f"Installation failed (exit {res.returncode}):\n{res.stderr or res.stdout}"
        except Exception as e:
            return False, f"Installation error: {e}"

