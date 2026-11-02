"""
test_tool_adapters.py

Phase 11 — Tool Adapters unit tests.

Tests:
1. NmapAdapter: command synthesis, output parsing, dry-run
2. NucleiAdapter: command synthesis, JSONL parsing, severity counts
3. FfufAdapter: command synthesis, FUZZ placement, JSON output parsing
4. GobusterAdapter: dir/dns modes, regex parsing
5. SqlmapAdapter: safety limits, --batch enforcement, output parsing
6. HydraAdapter: concurrency limits, credential parsing
7. register_offensive_tools: tool registry integration and schema validation
"""

import json
import os
import sys
import pytest

_HERE = os.path.dirname(__file__)
_CORE_ROOT = os.path.dirname(_HERE)
_RAG_ROOT = os.path.dirname(_CORE_ROOT)
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)

from core.tool_registry import ToolRegistry
from core.tools import (
    FfufAdapter,
    GobusterAdapter,
    HydraAdapter,
    NmapAdapter,
    NucleiAdapter,
    SqlmapAdapter,
    register_offensive_tools,
)


# ===========================================================================
# 1. NmapAdapter Tests
# ===========================================================================

def test_nmap_adapter_build_command():
    adapter = NmapAdapter()

    # Default build
    cmd = adapter.build_command(target="10.10.10.5")
    assert cmd[0] == "nmap"
    assert "-sS" in cmd
    assert "-sV" in cmd
    assert "--top-ports" in cmd
    assert "100" in cmd
    assert cmd[-1] == "10.10.10.5"

    # Custom all ports & connect scan
    cmd2 = adapter.build_command(target="10.10.10.5", ports="all", scan_type="connect", timing="T2")
    assert "-sT" in cmd2
    assert "-p-" in cmd2
    assert "-T2" in cmd2


def test_nmap_adapter_parse_output():
    adapter = NmapAdapter()
    sample_stdout = """\
Starting Nmap 7.92 at 2026-10-02 12:00
Nmap scan report for target.local (10.10.10.5)
Host is up (0.0021s latency).
PORT     STATE SERVICE VERSION
22/tcp   open  ssh     OpenSSH 8.2p1 Ubuntu
80/tcp   open  http    Apache httpd 2.4.41 ((Ubuntu))
3306/tcp open  mysql   MySQL 8.0.28
OS details: Linux 5.4.0-42-generic (Ubuntu)
"""
    parsed = adapter.parse_output(sample_stdout, "", 0)
    assert parsed["host_status"] == "up"
    assert parsed["open_ports_count"] == 3

    ports = parsed["open_ports"]
    assert ports[0]["port"] == 22
    assert ports[0]["service"] == "ssh"
    assert "OpenSSH" in ports[0]["version"]

    assert ports[1]["port"] == 80
    assert ports[1]["service"] == "http"

    assert len(parsed["os_matches"]) == 1
    assert "Ubuntu" in parsed["os_matches"][0]


def test_nmap_adapter_dry_run():
    adapter = NmapAdapter()
    res = adapter.run(target="10.10.10.5", dry_run=True)
    assert res.success is True
    assert "MANUAL MODE PREVIEW" in res.output or "nmap" in res.output


# ===========================================================================
# 2. NucleiAdapter Tests
# ===========================================================================

def test_nuclei_adapter_build_command():
    adapter = NucleiAdapter()
    cmd = adapter.build_command(target="http://10.10.10.5:8080", severity="critical,high", tags="cve")
    assert cmd[0] == "nuclei"
    assert "-u" in cmd
    assert "http://10.10.10.5:8080" in cmd
    assert "-severity" in cmd
    assert "critical,high" in cmd
    assert "-tags" in cmd
    assert "cve" in cmd
    assert "-jsonl" in cmd


def test_nuclei_adapter_parse_output():
    adapter = NucleiAdapter()
    jsonl_output = "\n".join([
        json.dumps({
            "template-id": "CVE-2021-41773",
            "info": {
                "name": "Apache 2.4.49 - Path Traversal",
                "severity": "critical",
                "classification": {"cve-id": ["CVE-2021-41773"]}
            },
            "matched-at": "http://10.10.10.5:80/icons/.%2e/%2e%2e/%2e%2e/etc/passwd"
        }),
        json.dumps({
            "template-id": "git-config-exposure",
            "info": {
                "name": "Git Config File Exposure",
                "severity": "medium",
            },
            "matched-at": "http://10.10.10.5:80/.git/config"
        })
    ])

    parsed = adapter.parse_output(jsonl_output, "", 0)
    assert parsed["total_findings"] == 2
    assert parsed["severity_counts"]["critical"] == 1
    assert parsed["severity_counts"]["medium"] == 1

    findings = parsed["findings"]
    assert findings[0]["template_id"] == "CVE-2021-41773"
    assert findings[0]["severity"] == "critical"
    assert "CVE-2021-41773" in findings[0]["cve_id"]


# ===========================================================================
# 3. FfufAdapter Tests
# ===========================================================================

def test_ffuf_adapter_build_command():
    adapter = FfufAdapter()
    cmd = adapter.build_command(url="http://target.local", extensions=".php,.txt", filter_status="404,403")
    assert cmd[0] == "ffuf"
    assert "-u" in cmd
    # Should append /FUZZ if not present
    assert any("FUZZ" in arg for arg in cmd)
    assert "-e" in cmd
    assert ".php,.txt" in cmd
    assert "-fc" in cmd
    assert "404,403" in cmd


def test_ffuf_adapter_parse_output():
    adapter = FfufAdapter()
    sample_json = json.dumps({
        "results": [
            {
                "input": {"FUZZ": "admin"},
                "url": "http://target.local/admin",
                "status": 301,
                "length": 194,
                "words": 15,
                "lines": 8,
                "redirectlocation": "http://target.local/admin/",
            },
            {
                "input": {"FUZZ": "robots.txt"},
                "url": "http://target.local/robots.txt",
                "status": 200,
                "length": 45,
                "words": 4,
                "lines": 2,
                "redirectlocation": "",
            }
        ]
    })

    parsed = adapter.parse_output(sample_json, "", 0)
    assert parsed["total_discovered"] == 2
    ep = parsed["endpoints"][0]
    assert ep["path"] == "admin"
    assert ep["status"] == 301
    assert ep["redirect"] == "http://target.local/admin/"


# ===========================================================================
# 4. GobusterAdapter Tests
# ===========================================================================

def test_gobuster_adapter_build_command():
    adapter = GobusterAdapter()

    # Dir mode
    cmd_dir = adapter.build_command(target="http://target.local", mode="dir")
    assert cmd_dir[0] == "gobuster"
    assert cmd_dir[1] == "dir"
    assert "-u" in cmd_dir

    # DNS mode
    cmd_dns = adapter.build_command(target="example.com", mode="dns")
    assert cmd_dns[1] == "dns"
    assert "-d" in cmd_dns
    assert "example.com" in cmd_dns


def test_gobuster_adapter_parse_output():
    adapter = GobusterAdapter()
    sample_out = """\
/login (Status: 200) [Size: 1045]
/dashboard (Status: 302) [Size: 0] [--> /login]
"""
    parsed = adapter.parse_output(sample_out, "", 0)
    assert parsed["total_findings"] == 2
    assert parsed["findings"][0]["path"] == "/login"
    assert parsed["findings"][0]["status"] == 200
    assert parsed["findings"][1]["redirect"] == "/login"


# ===========================================================================
# 5. SqlmapAdapter Tests
# ===========================================================================

def test_sqlmap_adapter_build_command():
    adapter = SqlmapAdapter()
    cmd = adapter.build_command(url="http://target.local/vuln.php?id=1", risk=5, level=5)
    assert cmd[0] == "sqlmap"
    assert "--batch" in cmd  # Enforced non-interactive
    # Capped at risk=2 and level=3
    assert "--risk=2" in cmd
    assert "--level=3" in cmd


def test_sqlmap_adapter_parse_output():
    adapter = SqlmapAdapter()
    sample_out = """\
[INFO] testing 'AND boolean-based blind - WHERE or HAVING clause'
[INFO] GET parameter 'id' is vulnerable. Do you want to keep testing the others? [y/N] N
sqlmap identified the following injection point(s) with a total of 45 HTTP(s) requests:
---
Parameter: id (GET)
    Type: boolean-based blind
    Title: AND boolean-based blind - WHERE or HAVING clause
---
[INFO] the back-end DBMS is MySQL
web server operating system: Linux Ubuntu 20.04
available databases [2]:
[*] information_schema
[*] app_db
"""
    parsed = adapter.parse_output(sample_out, "", 0)
    assert parsed["is_vulnerable"] is True
    assert "MySQL" in parsed["dbms"]
    assert any("id" in p for p in parsed["injectable_parameters"])
    assert "app_db" in parsed["databases"]


# ===========================================================================
# 6. HydraAdapter Tests
# ===========================================================================

def test_hydra_adapter_build_command():
    adapter = HydraAdapter()
    cmd = adapter.build_command(target="10.10.10.5", service="ssh", username="root", password="toor", tasks=20)
    assert cmd[0] == "hydra"
    assert "-t" in cmd
    # Capped at 16
    idx = cmd.index("-t")
    assert cmd[idx + 1] == "16"
    assert "-l" in cmd
    assert "root" in cmd
    assert "-p" in cmd
    assert "toor" in cmd
    assert "10.10.10.5" in cmd
    assert "ssh" in cmd


def test_hydra_adapter_parse_output():
    adapter = HydraAdapter()
    sample_out = """\
Hydra v9.2 (c) 2021 by van Hauser / THC & David Maciejak - Please do not use in military or secret service organizations, or for illegal purposes.
[DATA] max 4 tasks per 1 server, overall 4 tasks, 1 login tries (l:1/p:1), ~1 tries per task
[22][ssh] host: 10.10.10.5   login: root   password: toor
1 of 1 target successfully completed, 1 valid password found
"""
    parsed = adapter.parse_output(sample_out, "", 0)
    assert parsed["valid_credentials_count"] == 1
    cred = parsed["valid_credentials"][0]
    assert cred["service"] == "ssh"
    assert cred["login"] == "root"
    assert cred["password"] == "toor"


# ===========================================================================
# 7. Integration & Registration Tests
# ===========================================================================

def test_register_offensive_tools():
    fresh_reg = ToolRegistry()
    registered = register_offensive_tools(fresh_reg)

    expected_tools = {
        "nmap_scan",
        "nuclei_scan",
        "ffuf_fuzz",
        "gobuster_scan",
        "sqlmap_scan",
        "hydra_audit",
    }
    assert expected_tools.issubset(set(registered))
    assert expected_tools.issubset(set(fresh_reg.list_tools()))

    # Verify schema generation
    schemas = fresh_reg.get_schema()
    names = {s["function"]["name"] for s in schemas}
    assert expected_tools.issubset(names)
