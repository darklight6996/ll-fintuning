"""
test_cli_architecture.py

Unit test suite verifying:
1. SessionManager creation, persistence, and loading.
2. FileIngestionEngine with Nmap XML, Burp XML, Cloud JSON, and generic text.
3. Planner dynamic step evaluation (get_next_step) and backward compatibility (create_plan).
4. ReportGenerator compiling clean Markdown reports.
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

# Path setup
CORE_DIR = Path(__file__).resolve().parent
RAG_DIR = CORE_DIR.parent
ROOT_DIR = RAG_DIR.parent.parent

for p in (ROOT_DIR, RAG_DIR, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.session_manager import SessionManager
from core.ingestion import FileIngestionEngine
from core.planner import Planner, AssessmentPlan
from core.report_generator import ReportGenerator


def test_session_manager():
    print("[*] Testing SessionManager...")
    temp_dir = tempfile.mkdtemp()
    try:
        sm = SessionManager(base_dir=temp_dir)
        session = sm.create_session("test-corp-audit", "10.0.5.0/24, api.corp.internal")
        assert session["name"] == "test-corp-audit"
        assert "10.0.5.0/24" in session["targets"]["ips"]
        assert "api.corp.internal" in session["targets"]["domains"]

        # Add notes and evidence
        note = sm.add_note("Discovered potential SQLi parameter on /search")
        assert note["id"] == "note_1"
        assert len(sm.state["notes"]) == 1

        ev = sm.add_evidence("whoami", "corp\\auditor", step_id="1")
        assert ev["output"] == "corp\\auditor"

        # Verify state.json was written
        state_file = Path(temp_dir) / "test-corp-audit" / "state.json"
        assert state_file.exists()

        # Reload session in a new manager instance
        sm2 = SessionManager(base_dir=temp_dir)
        loaded = sm2.load_session("test-corp-audit")
        assert loaded["name"] == "test-corp-audit"
        assert len(loaded["notes"]) == 1
        assert len(loaded["evidence"]) == 1
        print("    [PASS] SessionManager creation, persistence, and reload verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_file_ingestion():
    print("[*] Testing FileIngestionEngine...")
    engine = FileIngestionEngine()
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Test Nmap XML
        nmap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE nmaprun>
<nmaprun scanner="nmap" version="7.92">
  <host>
    <status state="up"/>
    <address addr="10.0.5.25" addrtype="ipv4"/>
    <hostnames><hostname name="dc01.corp.internal" type="PTR"/></hostnames>
    <ports>
      <port protocol="tcp" portid="445"><state state="open"/><service name="microsoft-ds"/></port>
      <port protocol="tcp" portid="88"><state state="open"/><service name="kerberos-sec"/></port>
      <port protocol="tcp" portid="80"><state state="open"/><service name="http" product="Apache"/></port>
    </ports>
  </host>
</nmaprun>"""
        nmap_file = Path(temp_dir) / "scan.xml"
        nmap_file.write_text(nmap_xml, encoding="utf-8")

        res_nmap = engine.ingest_file(str(nmap_file))
        assert res_nmap.parser_type == "nmap"
        assert 445 in res_nmap.entities["ports"]
        assert 88 in res_nmap.entities["ports"]
        assert 80 in res_nmap.entities["ports"]
        assert "10.0.5.25" in res_nmap.entities["ips"]
        assert "dc01.corp.internal" in res_nmap.entities["domains"]
        print("    [PASS] Nmap XML parser verified.")

        # 2. Test Burp XML
        burp_xml = """<?xml version="1.0"?>
<issues burpVersion="2022.1">
  <issue>
    <name>SQL injection</name>
    <host ip="10.0.5.30">https://api.corp.internal</host>
    <path>/v1/users</path>
    <location>/v1/users?id=1</location>
    <severity>High</severity>
    <confidence>Certain</confidence>
    <issueDetail>Boolean-based blind SQLi discovered on parameter id</issueDetail>
  </issue>
</issues>"""
        burp_file = Path(temp_dir) / "burp.xml"
        burp_file.write_text(burp_xml, encoding="utf-8")

        res_burp = engine.ingest_file(str(burp_file))
        assert res_burp.parser_type == "burp"
        assert len(res_burp.entities["vulnerabilities"]) > 0
        assert "api.corp.internal" in res_burp.entities["domains"]
        print("    [PASS] Burp XML parser verified.")

        # 3. Test Cloud JSON
        cloud_json = json.dumps([
            {
                "roleDefinitionName": "Owner",
                "principalName": "admin@acme.onmicrosoft.com",
                "scope": "/subscriptions/sub-123"
            },
            {
                "name": "kv-prod-vault",
                "vaultUri": "https://kv-prod-vault.vault.azure.net/"
            }
        ])
        cloud_file = Path(temp_dir) / "roles.json"
        cloud_file.write_text(cloud_json, encoding="utf-8")

        res_cloud = engine.ingest_file(str(cloud_file))
        assert res_cloud.parser_type == "cloud"
        assert any("RoleAssignment:Owner" in r for r in res_cloud.entities["cloud_resources"])
        assert any("KeyVault:kv-prod-vault" in r for r in res_cloud.entities["cloud_resources"])
        print("    [PASS] Cloud CLI JSON parser verified.")

        # 4. Test Generic Text
        txt_content = "Discovered internal server at 192.168.1.100 and database host db.corp.lan with password = SuperSecretPassword123"
        txt_file = Path(temp_dir) / "notes.txt"
        txt_file.write_text(txt_content, encoding="utf-8")

        res_txt = engine.ingest_file(str(txt_file))
        assert res_txt.parser_type == "generic"
        assert "192.168.1.100" in res_txt.entities["ips"]
        assert "db.corp.lan" in res_txt.entities["domains"]
        assert len(res_txt.entities["vulnerabilities"]) > 0
        print("    [PASS] Generic Text parser verified.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_planner_dynamic_step():
    print("[*] Testing Planner dynamic step evaluation...")
    planner = Planner()

    # Case 1: Initial state with IP in scope -> Recommend Recon/Nmap
    state_initial = {
        "scope": "10.0.5.10",
        "targets": {"ips": ["10.0.5.10"], "ports": [], "domains": []},
        "completed_steps": [],
        "ingested_findings": {}
    }
    step1 = planner.get_next_step(state_initial)
    assert step1["step_number"] == 1
    assert step1["phase"] == "reconnaissance"
    assert any("nmap" in cmd for cmd in step1["commands"])
    print("    [PASS] Initial state correctly recommended port discovery.")

    # Case 2: Ingested finding with Port 445 & 88 (Active Directory)
    state_ad = {
        "scope": "10.0.5.10",
        "targets": {"ips": ["10.0.5.10"], "ports": [445, 88], "domains": ["corp.local"]},
        "completed_steps": [step1],
        "ingested_findings": {"nmap": [{"summary": "SMB and Kerberos open"}]}
    }
    step2 = planner.get_next_step(state_ad)
    assert step2["step_number"] == 2
    assert "Active Directory" in step2["title"]
    assert any("netexec" in cmd for cmd in step2["commands"])
    print("    [PASS] Active Directory open ports dynamically prioritized SMB/Kerberos enumeration.")

    # Case 3: Backward compatibility with create_plan
    plan = planner.create_plan([
        {
            "id": "T1",
            "name": "Test Template",
            "steps": [
                {"phase": "reconnaissance", "action": "Recon step"},
                {"phase": "privilege_escalation", "action": "Privesc step"}
            ]
        }
    ])
    assert isinstance(plan, AssessmentPlan)
    assert len(plan.sequential_steps) == 2
    assert plan.sequential_steps[0]["phase"] == "reconnaissance"
    print("    [PASS] Planner backward compatibility verified.")


def test_report_generator():
    print("[*] Testing ReportGenerator...")
    rg = ReportGenerator()
    temp_dir = tempfile.mkdtemp()
    try:
        sample_state = {
            "name": "audit-q3",
            "scope": "10.0.4.0/24, azure-prod",
            "status": "completed",
            "created_at": "2026-09-03T19:00:00Z",
            "updated_at": "2026-09-03T19:30:00Z",
            "targets": {
                "ips": ["10.0.4.15"],
                "domains": ["corp.acme.com"],
                "ports": [80, 443, 445],
                "services": ["http", "https", "smb"],
                "cloud_resources": ["RoleAssignment:Owner"],
                "vulnerabilities": ["[HIGH] SQL injection"]
            },
            "ingested_findings": {
                "nmap": [{"source_file": "scan.xml", "ingested_at": "2026-09-03T19:05:00Z", "data": {"summary": "Live hosts found"}}]
            },
            "completed_steps": [
                {"step_number": 1, "phase": "reconnaissance", "title": "Network Port Discovery", "completed_at": "2026-09-03T19:10:00Z", "execution": {"status": "success"}}
            ],
            "notes": [
                {"id": "note_1", "content": "Confirmed perimeter firewall active.", "author": "auditor", "timestamp": "2026-09-03T19:12:00Z"}
            ],
            "evidence": [
                {"id": "ev_1", "step_id": "1", "command": "nmap 10.0.4.15", "output": "Port 80 open", "status": "success"}
            ]
        }

        rep_path = Path(temp_dir) / "report.md"
        report_md = rg.generate(sample_state, output_path=str(rep_path))
        assert rep_path.exists()
        assert "# Security Assessment & Compliance Report" in report_md
        assert "audit-q3" in report_md
        assert "10.0.4.15" in report_md
        assert "SQL injection" in report_md
        assert "Network Port Discovery" in report_md
        print("    [PASS] ReportGenerator successfully generated valid Markdown audit report.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING CLI ARCHITECTURE UNIT TESTS")
    print("=" * 60)
    test_session_manager()
    test_file_ingestion()
    test_planner_dynamic_step()
    test_report_generator()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
