"""
core/report_generator.py

Purpose
-------
Synthesize all session data, scope, notes, ingested findings, and execution evidence
into a clean, professional Markdown report.

Responsibilities
----------------
1. Read session state from SessionManager.
2. Structure sections: Executive Summary, Scope, Ingested Findings Matrix, Action Timeline, Notes, and Evidence Proof.
3. Automatically export report to `.assessments/<session_name>/report.md`.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List


class ReportGenerator:
    """
    Compiles assessment session data into an executive-ready Markdown audit report.
    """

    def generate(
        self,
        session_state: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate Markdown report from session state.

        Args:
            session_state: Dictionary containing full session data.
            output_path: Optional custom destination file path.

        Returns:
            The complete Markdown report string.
        """
        name = session_state.get("name", "Unnamed Assessment")
        scope = session_state.get("scope", "General Scope")
        status = session_state.get("status", "in_progress").upper()
        created_at = session_state.get("created_at", "N/A")
        updated_at = session_state.get("updated_at", "N/A")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        completed_steps = session_state.get("completed_steps", [])
        evidence_list = session_state.get("evidence", [])
        notes = session_state.get("notes", [])
        targets = session_state.get("targets", {})
        ingested = session_state.get("ingested_findings", {})

        lines = [
            f"# Security Assessment & Compliance Report",
            f"## Engagement: {name}",
            f"",
            f"> **Generated:** {now} | **Status:** `{status}` | **Framework:** Automated Tactical Planner",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary",
            f"",
            f"This document provides the formal record of assessment activities, technical observations, scanner ingestion findings, and verified execution proof for **{name}**.",
            f"",
            f"| Metric | Value |",
            f"| :--- | :--- |",
            f"| **Session Identifier** | `{name}` |",
            f"| **Engagement Scope** | `{scope or 'Broad assessment'}` |",
            f"| **Initiated At** | `{created_at}` |",
            f"| **Last Active At** | `{updated_at}` |",
            f"| **Completed Action Items** | **{len(completed_steps)}** |",
            f"| **Ingested Scan Sources** | **{sum(len(v) for v in ingested.values())}** |",
            f"| **Recorded Evidence Proofs** | **{len(evidence_list)}** |",
            f"| **Manual Observations/Notes** | **{len(notes)}** |",
            f"",
            f"---",
            f"",
            f"## 2. Engagement Scope & Target Inventory",
            f"",
            f"**Defined Scope Description:**",
            f"> {scope if scope else 'No formal scope boundary provided.'}",
            f"",
            f"### Identified Attack Surface & Targets",
            f""
        ]

        # Target entities breakdown
        ips = targets.get("ips", [])
        domains = targets.get("domains", [])
        ports = targets.get("ports", [])
        services = targets.get("services", [])
        cloud = targets.get("cloud_resources", [])
        vulns = targets.get("vulnerabilities", [])

        lines.append(f"* **Target IP Addresses ({len(ips)}):** {', '.join(f'`{ip}`' for ip in ips) if ips else 'None extracted'}")
        lines.append(f"* **Target Domain Names ({len(domains)}):** {', '.join(f'`{d}`' for d in domains) if domains else 'None extracted'}")
        lines.append(f"* **Discovered Open Ports ({len(ports)}):** {', '.join(str(p) for p in ports) if ports else 'None recorded'}")
        lines.append(f"* **Services Identified ({len(services)}):** {', '.join(services[:15]) if services else 'None recorded'}")
        if cloud:
            lines.append(f"* **Cloud Resources ({len(cloud)}):** {', '.join(cloud[:10])}")
        if vulns:
            lines.append(f"* **Identified Vulnerability Indicators ({len(vulns)}):**")
            for v in vulns:
                lines.append(f"  - ⚠️ {v}")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Ingested Scanner Findings & Artifacts",
            f""
        ])

        # Nmap findings
        nmap_entries = ingested.get("nmap", [])
        if nmap_entries:
            lines.append(f"### 3.1 Network Discovery (Nmap)")
            for idx, entry in enumerate(nmap_entries, 1):
                src = Path(entry.get("source_file", "nmap")).name
                data = entry.get("data", {})
                lines.append(f"#### Scan {idx}: `{src}` ({entry.get('ingested_at', '')})")
                lines.append(f"**Summary:** {data.get('summary', 'Nmap scan ingested.')}")
                hosts = data.get("details", {}).get("hosts", [])
                if hosts:
                    lines.append(f"| Host IP | Hostnames | Open Ports & Services |")
                    lines.append(f"| :--- | :--- | :--- |")
                    for h in hosts:
                        h_ips = ", ".join(h.get("ips", []))
                        h_names = ", ".join(h.get("hostnames", [])) or "-"
                        p_list = ", ".join(f"{p['port']}/{p['service']}" for p in h.get("open_ports", [])) or "None"
                        lines.append(f"| `{h_ips}` | {h_names} | {p_list} |")
                lines.append("")
        else:
            lines.append(f"*No Nmap scans ingested for this session.*")
            lines.append("")

        # Burp findings
        burp_entries = ingested.get("burp", [])
        if burp_entries:
            lines.append(f"### 3.2 Web Application Security (Burp Suite)")
            for idx, entry in enumerate(burp_entries, 1):
                src = Path(entry.get("source_file", "burp")).name
                data = entry.get("data", {})
                lines.append(f"#### Export {idx}: `{src}`")
                lines.append(f"**Summary:** {data.get('summary', 'Burp export ingested.')}")
                issues = data.get("details", {}).get("issues", [])
                if issues:
                    lines.append(f"| Severity | Finding Name | Host / Path |")
                    lines.append(f"| :--- | :--- | :--- |")
                    for iss in issues[:20]:
                        sev = iss.get("severity", "Info")
                        icon = "🔴" if sev.lower() == "high" else ("🟠" if sev.lower() == "medium" else "🟡")
                        lines.append(f"| {icon} **{sev}** | {iss.get('name')} | `{iss.get('host')}{iss.get('path')}` |")
                lines.append("")

        # Cloud findings
        cloud_entries = ingested.get("cloud", [])
        if cloud_entries:
            lines.append(f"### 3.3 Cloud Infrastructure & IAM Findings")
            for idx, entry in enumerate(cloud_entries, 1):
                src = Path(entry.get("source_file", "cloud")).name
                data = entry.get("data", {})
                lines.append(f"#### Configuration {idx}: `{src}`")
                lines.append(f"**Summary:** {data.get('summary', 'Cloud CLI output ingested.')}")
                roles = data.get("details", {}).get("roles", [])
                if roles:
                    lines.append(f"*Discovered Roles:* {', '.join(f'`{r}`' for r in roles)}")
                lines.append("")

        # Action Timeline
        lines.extend([
            f"---",
            f"",
            f"## 4. Chronological Action Timeline",
            f"",
            f"Step-by-step record of planner-recommended actions executed during this engagement.",
            f""
        ])

        if completed_steps:
            lines.append(f"| Step | Phase | Action / Title | Completed At | Execution Status |")
            lines.append(f"| :--- | :--- | :--- | :--- | :--- |")
            for step in completed_steps:
                s_num = step.get("step_number", "-")
                phase = step.get("phase", "General").title()
                title = step.get("title") or step.get("action", "Action Item")
                comp_at = step.get("completed_at", "N/A")
                exec_stat = step.get("execution", {}).get("status", "completed").upper()
                lines.append(f"| **{s_num}** | `{phase}` | {title} | `{comp_at}` | `{exec_stat}` |")
            lines.append("")
        else:
            lines.append(f"*No planner steps completed yet.*")
            lines.append("")

        # Operator Notes
        lines.extend([
            f"---",
            f"",
            f"## 5. Operator Notes & Observations",
            f""
        ])

        if notes:
            for note in notes:
                author = note.get("author", "operator")
                ts = note.get("timestamp", "")
                content = note.get("content", "").replace("\n", "\n> ")
                lines.append(f"#### Note {note.get('id')} — *{author} ({ts})*")
                lines.append(f"> {content}")
                lines.append("")
        else:
            lines.append(f"*No manual notes recorded for this session.*")
            lines.append("")

        # Evidence Proof
        lines.extend([
            f"---",
            f"",
            f"## 6. Technical Evidence Proof & Execution Logs",
            f""
        ])

        if evidence_list:
            for ev in evidence_list:
                ev_id = ev.get("id")
                step_id = ev.get("step_id", "N/A")
                cmd = ev.get("command", "")
                out = ev.get("output", "")
                ts = ev.get("timestamp", "")
                stat = ev.get("status", "success")

                lines.append(f"### Evidence #{ev_id} (Step: `{step_id}`) - `{stat.upper()}`")
                if cmd:
                    lines.append(f"**Command Dispatched:**")
                    lines.append(f"```bash\n{cmd}\n```")
                if out:
                    lines.append(f"**Captured Output:**")
                    lines.append(f"```text\n{out[:1500]}\n```")
                lines.append("")
        else:
            lines.append(f"*No command execution evidence records captured.*")
            lines.append("")

        # Strategic Recommendations
        lines.extend([
            f"---",
            f"",
            f"## 7. Strategic Recommendations & Remediation Plan",
            f"",
            f"Based on the observed attack surface and findings in this engagement:",
            f"",
            f"1. **Attack Surface Reduction:** Close or firewall unneeded services discovered during network enumeration.",
            f"2. **Identity & Access Governance:** Enforce Least Privilege on all cloud and domain accounts. Restrict wildcard IAM assignments.",
            f"3. **Web Application Defense:** Sanitize and validate all user inputs at application boundaries to prevent injection vulnerabilities.",
            f"4. **Continuous Telemetry:** Ensure endpoint and cloud audit logging (CloudTrail, Sysmon) monitors and alerts on privilege elevation attempts.",
            f"",
            f"---",
            f"*End of Assessment Report — Generated by Security Assessment & Planning CLI.*"
        ])

        report_content = "\n".join(lines)

        # Write to file if path provided
        if output_path:
            out_file = Path(output_path).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(report_content)

        return report_content
