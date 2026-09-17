"""
main.py

Purpose
-------
Structured, interactive assessment CLI tool that uses a dynamic step-by-step planner,
manages session state, and supports multi-format file ingestion (Nmap, Burp, Cloud, Text).

Commands
--------
  new <name>     Initialize a new assessment folder and set scope
  open <name>    Resume an existing assessment folder and load state
  next           Fetch and display exactly ONE recommended step from the planner
  run            Execute the current recommended step (or mark it as completed)
  paste          Prompt for a multi-line text block to record notes or evidence
  upload <path>  Trigger the file ingestion pipeline for a local scanner file
  scope [new]    View or update engagement scope boundaries
  type [new]     View or change assessment methodology type (e.g. ctf, linux, web, ad)
  target [...]   Manage targets: view, set primary, add/remove entities (ip, domain, port)
  back / undo    Roll back the last completed step so it can be re-evaluated
  reset <steps>  Reset completed steps to re-evaluate engagement from current state
  report         Compile accumulated session notes and evidence into Markdown
  status         Display overview of the active assessment session and targets
  list           List all existing assessment sessions
  quit           Safely save state and exit the CLI
"""

import os
import sys
import cmd
from pathlib import Path
from typing import Optional

# Ensure console output is UTF-8 safe on Windows to avoid UnicodeEncodeError (CP1252 charmap)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass  # Python < 3.7 or output already redirected/piped

# Setup import paths so 'core...' modules resolve cleanly
CURRENT_DIR = Path(__file__).resolve().parent
POSSIBLE_RAG_DIRS = [
    CURRENT_DIR / "ll-finetuning" / "rag",
    CURRENT_DIR / "rag",
    CURRENT_DIR
]

for d in POSSIBLE_RAG_DIRS:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

POSSIBLE_CORE_DIRS = [
    CURRENT_DIR / "ll-finetuning" / "rag" / "core",
    CURRENT_DIR / "rag" / "core",
    CURRENT_DIR / "core"
]

for d in POSSIBLE_CORE_DIRS:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

# Import core components
try:
    from core.session_manager import SessionManager
    from core.ingestion import FileIngestionEngine
    from core.planner import Planner
    from core.report_generator import ReportGenerator
    from core.command_executor import CommandExecutor
    from core.chat_assistant import ChatAssistant
except ImportError:
    from session_manager import SessionManager
    from ingestion import FileIngestionEngine
    from planner import Planner
    from report_generator import ReportGenerator
    from command_executor import CommandExecutor
    try:
        from chat_assistant import ChatAssistant
    except ImportError:
        ChatAssistant = None


BANNER = r"""
================================================================================
   ______      __               ____                          __
  / ____/_  __/ /_  ___  _____ / __/___ _  ____ _____  ____  / /_____  _____
 / /   / / / / __ \/ _ \/ ___// /_/ __ `/ / __ `/ __ \/ __ \/ //_/ _ \/ ___/
/ /___/ /_/ / /_/ /  __/ /   / __/ /_/ / / /_/ / /_/ / /_/ / ,< /  __/ /
\____/\__, /_.___/\___/_/   /_/  \__,_/  \__,_/ .___/ .___/_/|_|\___/_/
     /____/                                  /_/   /_/
              CYBER AI CO-PILOT & HACKER CHAT ASSISTANT
================================================================================
Talk to the AI freely at any time! Ask questions, explain outputs, or steer the
strategy. Type your thoughts directly, or use session keywords:
  • 'new <name>'   - Start a new assessment / CTF session
  • 'open <name>'  - Load existing assessment session (supports partial names)
  • 'next'         - Fetch the next planned tactical step
  • 'paste'        - Paste terminal output, scan logs, or strings for AI analysis
  • 'run'          - Execute or mark the pending step completed
  • 'status'       - Review targets, scope, and discovered entities
  • 'report'       - Generate comprehensive audit Markdown report
================================================================================
"""


class AssessmentCLI(cmd.Cmd):
    """Interactive command-line shell for assessment workflows and AI Co-Pilot chat."""

    intro = BANNER
    prompt = "Co-Pilot [no session] > "

    def __init__(self):
        super().__init__()
        self.session_mgr = SessionManager()
        self.ingestion_engine = FileIngestionEngine()
        self.planner = Planner()
        self.report_gen = ReportGenerator()
        self.executor = CommandExecutor(read_only=False)
        self.chat_assistant = ChatAssistant() if ChatAssistant else None
        self._update_prompt()

    def _update_prompt(self):
        if self.session_mgr.active_session_name:
            self.prompt = f"Co-Pilot [{self.session_mgr.active_session_name}] > "
        else:
            self.prompt = "Co-Pilot [no session] > "

    def _require_session(self) -> bool:
        if not self.session_mgr.active_session_name:
            print("\n[!] No active assessment session.")
            print("    Use 'new <name>' to create one or 'open <name>' to load an existing session.\n")
            return False
        return True

    # ------------------------------------------------------------------
    # Command: new <name>
    # ------------------------------------------------------------------
    def do_new(self, arg):
        """
        Initialize a new assessment session, target, and scope.
        Usage: new <session_name>
        """
        name = arg.strip()
        if not name:
            name = input("Enter assessment session name (e.g., HTB_Forest, pwn_college): ").strip()

        if not name:
            print("[!] Session name cannot be empty.")
            return

        print(f"\n[*] Configuring assessment session '{name}'...")
        target = input("Target (IP, Domain, or localhost):\n> ").strip()

        print("\nSelect Environment / Assessment Type:")
        print("  [1] Local CTF / Interactive Shell (pwn.college, Docker, local box) [DEFAULT]")
        print("  [2] Internal Network / VPN / Active Directory (HackTheBox, OSCP)")
        print("  [3] Web Application / API / Bug Bounty")
        print("  [4] Cloud Infrastructure (AWS / Azure / GCP)")
        choice = input("Enter option (1-4) or custom type [default: 1]: ").strip()
        if choice == "1" or not choice:
            target_type = "ctf"
        elif choice == "2":
            target_type = "active_directory"
        elif choice == "3":
            target_type = "web"
        elif choice == "4":
            target_type = "cloud"
        else:
            target_type = choice

        scope = input("\nScope boundaries (e.g., local shell /challenge /flag, or IP CIDR):\n> ").strip()
        if not scope:
            scope = "local shell /challenge /flag" if target_type == "ctf" else (target or "General Assessment")

        conn = input("\nConnectivity: Is target via VPN, local container, or direct? [VPN/Local/Direct, default: Local]: ").strip() or "Local"

        try:
            state = self.session_mgr.create_session(
                name=name,
                scope=scope,
                target=target,
                target_type=target_type
            )
            state["connectivity"] = conn
            self.session_mgr.save_current_state()
            self._update_prompt()
            print(f"\n[+] Assessment session '{name}' successfully initialized!")
            print(f"    Folder:       {self.session_mgr.session_dir}")
            print(f"    Target:       {target or 'N/A'}")
            print(f"    Type:         {target_type.upper()}")
            print(f"    Connectivity: {conn}")
            print(f"    Scope:        {scope}")
            print("\nType 'next' to get recommended action items, or simply ask the AI questions directly!\n")
        except Exception as e:
            print(f"[!] Error creating session: {e}")

    # ------------------------------------------------------------------
    # Command: open <name>
    # ------------------------------------------------------------------
    def do_open(self, arg):
        """
        Resume an existing assessment session and load state.
        Usage: open <session_name>
        """
        name = arg.strip()
        if not name:
            sessions = self.session_mgr.list_sessions()
            if not sessions:
                print("[!] No existing sessions found in .assessments/. Use 'new <name>' to create one.")
                return
            print("\nAvailable Assessment Sessions:")
            for s in sessions:
                print(f"  - {s}")
            name = input("\nEnter session name to open: ").strip()

        if not name:
            return

        # Fuzzy / prefix name matching — let the user type a partial session name
        available = self.session_mgr.list_sessions()
        resolved_name = name
        if name not in available:
            # Try case-insensitive prefix match
            matches = [s for s in available if s.lower().startswith(name.lower())]
            if not matches:
                # Fall back to substring match
                matches = [s for s in available if name.lower() in s.lower()]
            if len(matches) == 1:
                resolved_name = matches[0]
                print(f"[*] Resolved '{name}' → '{resolved_name}'")
            elif len(matches) > 1:
                print(f"\n[!] Ambiguous session name '{name}'. Did you mean one of:")
                for m in matches:
                    print(f"  - {m}")
                resolved_name = input("\nEnter exact session name: ").strip()
                if not resolved_name:
                    return
            # If no match at all, pass original name through to produce a clear error

        try:
            state = self.session_mgr.load_session(resolved_name)
            self._update_prompt()
            print(f"\n[+] Session '{resolved_name}' loaded successfully.")
            print(f"    Scope:           {state.get('scope') or 'None'}")
            print(f"    Completed Steps: {len(state.get('completed_steps', []))}")
            print(f"    Evidence Count:  {len(state.get('evidence', []))}")
            print(f"    Notes Count:     {len(state.get('notes', []))}")
            curr = state.get("current_step")
            if curr:
                print(f"    Pending Step:    #{curr.get('step_number')} [{curr.get('phase')}] {curr.get('title')}")
            print("\nType 'next' for next step, 'status' for details, or 'run' to execute pending step.\n")
        except Exception as e:
            print(f"[!] Could not load session '{resolved_name}': {e}")

    # ------------------------------------------------------------------
    # Command: list
    # ------------------------------------------------------------------
    def do_list(self, arg):
        """
        List all saved assessment sessions.
        Usage: list
        """
        sessions = self.session_mgr.list_sessions()
        if not sessions:
            print("\n[!] No assessment sessions found in .assessments/.\n")
            return

        print("\n=== Saved Assessment Sessions ===")
        for s in sessions:
            active_marker = " (ACTIVE)" if s == self.session_mgr.active_session_name else ""
            print(f"  * {s}{active_marker}")
        print()

    # ------------------------------------------------------------------
    # Command: status
    # ------------------------------------------------------------------
    def do_status(self, arg):
        """
        Display current session status, targets, and progress overview.
        Usage: status
        """
        if not self._require_session():
            return

        summary = self.session_mgr.get_summary()
        targets = summary.get("targets", {})

        print("\n" + "=" * 65)
        print(f"  ASSESSMENT STATUS: {summary.get('name')}")
        print("=" * 65)
        print(f"Target:              {self.session_mgr.state.get('target') or 'N/A'}")
        print(f"Type:                {self.session_mgr.state.get('type') or 'General'}")
        print(f"Scope:               {summary.get('scope') or 'N/A'}")
        print(f"Created At:          {summary.get('created_at')}")
        print(f"Completed Steps:     {summary.get('completed_steps_count')}")
        print(f"Notes Recorded:      {summary.get('notes_count')}")
        print(f"Evidence Proofs:     {summary.get('evidence_count')}")
        print("-" * 65)
        print("Discovered Targets & Findings:")
        print(f"  IPs:               {', '.join(targets.get('ips', [])) or 'None'}")
        print(f"  Domains:           {', '.join(targets.get('domains', [])) or 'None'}")
        print(f"  Open Ports:        {', '.join(str(p) for p in targets.get('ports', [])) or 'None'}")
        print(f"  Services:          {', '.join(targets.get('services', [])[:8]) or 'None'}")
        print(f"  Cloud Resources:   {', '.join(targets.get('cloud_resources', [])[:5]) or 'None'}")
        print(f"  Vulnerabilities:   {len(targets.get('vulnerabilities', []))} logged")
        print("-" * 65)

        curr = summary.get("current_step")
        if curr:
            print(f"Active Step:         #{curr.get('step_number')} [{curr.get('phase').upper()}]")
            print(f"Action:              {curr.get('title') or curr.get('action')}")
        else:
            print("Active Step:         None pending. Type 'next' to get recommended step.")
        print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # Command: scope [new_scope]
    # ------------------------------------------------------------------
    def do_scope(self, arg):
        """
        View or modify engagement scope boundaries.
        Usage:
          scope             - View current scope
          scope <new_scope> - Update scope (e.g., 'scope local shell /challenge /flag')
        """
        if not self._require_session():
            return
        new_scope = arg.strip()
        if not new_scope:
            print(f"\nCurrent Scope: {self.session_mgr.state.get('scope') or 'N/A'}\n")
            return
        updated = self.session_mgr.update_scope(new_scope)
        print(f"\n[+] Scope updated: {updated}\n")

    # ------------------------------------------------------------------
    # Command: type [new_type]
    # ------------------------------------------------------------------
    def do_type(self, arg):
        """
        View or modify assessment target type (e.g., ctf, linux, web, active_directory, cloud).
        Usage:
          type          - View current type
          type <type>   - Set type (e.g., 'type ctf' or 'type linux')
        """
        if not self._require_session():
            return
        new_type = arg.strip()
        if not new_type:
            print(f"\nCurrent Target Type: {self.session_mgr.state.get('type') or 'General'}\n")
            return
        updated = self.session_mgr.update_type(new_type)
        print(f"\n[+] Assessment type updated to: {updated}")
        print("    Type 'next' to refresh planner recommendations for this methodology.\n")

    # ------------------------------------------------------------------
    # Command: target [subcommand]
    # ------------------------------------------------------------------
    def do_target(self, arg):
        """
        View, update, or manage targets and target entities.
        Usage:
          target                         - View all targets and entities
          target set <value>             - Update primary target (e.g. 'target set localhost')
          target add <ip|domain|port> <val> - Add an entity (e.g. 'target add port 80')
          target rm <ip|domain|port> <val>  - Remove an entity (e.g. 'target rm domain pwn.college')
          target clear [domains|ips|ports] - Clear target category
        """
        if not self._require_session():
            return

        parts = arg.strip().split()
        if not parts:
            targets = self.session_mgr.state.get("targets", {})
            print("\n=== Current Target Configuration ===")
            print(f"Primary Target: {self.session_mgr.state.get('target') or 'N/A'}")
            print(f"Target Type:    {self.session_mgr.state.get('type') or 'General'}")
            print(f"Scope:          {self.session_mgr.state.get('scope') or 'N/A'}")
            print(f"IPs:            {', '.join(targets.get('ips', [])) or 'None'}")
            print(f"Domains:        {', '.join(targets.get('domains', [])) or 'None'}")
            print(f"Ports:          {', '.join(str(p) for p in targets.get('ports', [])) or 'None'}")
            print(f"Services:       {', '.join(targets.get('services', [])) or 'None'}")
            print("====================================\n")
            return

        subcmd = parts[0].lower()
        if subcmd == "set":
            if len(parts) < 2:
                print("[!] Usage: target set <value>")
                return
            new_target = " ".join(parts[1:])
            self.session_mgr.update_target(new_target)
            print(f"[+] Primary target set to: {new_target}")

        elif subcmd == "add":
            if len(parts) < 3:
                print("[!] Usage: target add <category> <value> (e.g., target add ip 10.10.10.5)")
                return
            cat, val = parts[1], parts[2]
            if self.session_mgr.add_target_entity(cat, val):
                print(f"[+] Added {cat} '{val}' to targets.")
            else:
                print(f"[*] {cat} '{val}' already exists or is invalid.")

        elif subcmd in ("rm", "del", "delete", "remove"):
            if len(parts) < 3:
                print("[!] Usage: target rm <category> <value> (e.g., target rm domain pwn.college)")
                return
            cat, val = parts[1], parts[2]
            if self.session_mgr.remove_target_entity(cat, val):
                print(f"[+] Removed {cat} '{val}' from targets.")
            else:
                print(f"[!] {cat} '{val}' not found in active targets.")

        elif subcmd == "clear":
            cat = parts[1] if len(parts) > 1 else None
            self.session_mgr.clear_target_category(cat)
            print(f"[+] Cleared {cat or 'all'} targets.")
        else:
            print(f"[!] Unknown target action '{subcmd}'. Use 'help target'.")

    # ------------------------------------------------------------------
    # Command: back / undo
    # ------------------------------------------------------------------
    def do_back(self, arg):
        """
        Undo the last completed step and restore it as the pending step.
        Usage: back
        """
        if not self._require_session():
            return
        restored = self.session_mgr.undo_last_step()
        if not restored:
            print("\n[!] No completed steps to roll back.\n")
            return
        print(f"\n[+] Rolled back step #{restored.get('step_number')}: {restored.get('title')}")
        print("    This step is now set as the active pending step. Type 'next' or 'run'.\n")

    def do_undo(self, arg):
        """Alias for 'back'."""
        return self.do_back(arg)

    # ------------------------------------------------------------------
    # Command: reset
    # ------------------------------------------------------------------
    def do_reset(self, arg):
        """
        Reset completed steps or session findings.
        Usage:
          reset steps   - Clear completed steps so planner restarts
          reset targets - Clear all identified target entities
        """
        if not self._require_session():
            return
        sub = arg.strip().lower()
        if sub in ("steps", "step"):
            count = self.session_mgr.reset_steps()
            print(f"\n[+] Reset {count} completed step(s). Type 'next' to receive fresh recommendation.\n")
        elif sub in ("targets", "target"):
            self.session_mgr.clear_target_category(None)
            print("\n[+] Cleared all targets from session.\n")
        else:
            print("[!] Usage: reset steps | reset targets")

    # ------------------------------------------------------------------
    # Command: next
    # ------------------------------------------------------------------
    def do_next(self, arg):
        """
        Fetch and display exactly ONE recommended step from the planner based on state.
        Usage: next
        """
        if not self._require_session():
            return

        step = self.planner.get_next_step(self.session_mgr.state)
        self.session_mgr.set_current_step(step)

        if step.get("id") == "flag_captured":
            print("\n" + "=" * 65)
            print("  >>> FLAG CAPTURED -- CHALLENGE COMPLETE! <<<")
            print("=" * 65)
            print(f"Flag Detected: {step.get('action')}")
            print(f"Purpose:       {step.get('purpose')}")
            print(f"Rationale:     {step.get('rationale')}")
            print("-" * 65)
            print("Type 'report' to generate the final assessment audit report.")
            print("=" * 65 + "\n")
            return

        if step.get("id") == "engagement_complete":
            print("\n" + "=" * 65)
            print("  [+] ASSESSMENT METHODOLOGY COMPLETE")
            print("=" * 65)
            print(f"Action:    {step.get('action')}")
            print(f"Rationale: {step.get('rationale')}")
            print("-" * 65)
            print("All methodology steps finished. Type 'report' to export Markdown report.")
            print("=" * 65 + "\n")
            return

        step_num = step.get("step_number", 1)
        phase = step.get("phase", "general").upper()
        title = step.get("title") or step.get("action")
        purpose = step.get("purpose") or step.get("action")
        commands = step.get("commands", [])

        print(f"\nStep {step_num}\n")
        print("Run:\n")
        for cmd_str in commands:
            print(f"  {cmd_str}")
        print("\nPurpose\n")
        print(f"  {purpose}\n")
        print("-" * 60)
        print("Run it externally and paste the output ('paste'), or type 'run' to execute.")
        print("-" * 60 + "\n")

    # ------------------------------------------------------------------
    # Command: run
    # ------------------------------------------------------------------
    def do_run(self, arg):
        """
        Execute the current recommended step (or mark it as completed).
        Usage: run
        """
        if not self._require_session():
            return

        current_step = self.session_mgr.state.get("current_step")
        if not current_step:
            print("\n[!] No active step to run. Type 'next' to request the next action item first.\n")
            return

        commands = current_step.get("commands", [])
        print("\n" + "-" * 60)
        print(f"Execute Step #{current_step.get('step_number')}: {current_step.get('title')}")
        print("-" * 60)
        for c in commands:
            print(f"  >> {c}")
        print("-" * 60)

        choice = input("\nDo you want to (E)xecute commands now, (M)ark step done manually, or (C)ancel? [E/M/C]: ").strip().lower()

        if choice in ("e", "execute"):
            execution_results = []
            for cmd_str in commands:
                print(f"\n[*] Executing: {cmd_str}")
                confirm = input("    Confirm launch? (Y/N): ").strip().lower()
                if confirm == "y":
                    res = self.executor.execute_command(cmd_str, approved=True)
                    execution_results.append(res)
                    print(f"    Status: {res.get('status')}")
                    if res.get("stdout"):
                        print(f"    Output Preview:\n{res['stdout'][:500]}")
                    if res.get("stderr"):
                        print(f"    Stderr:\n{res['stderr'][:300]}")
                else:
                    print("    Execution skipped by operator.")
                    execution_results.append({"command": cmd_str, "status": "declined", "stdout": ""})

            # Save primary result
            main_res = execution_results[0] if execution_results else {"status": "executed"}
            self.session_mgr.complete_current_step(main_res)
            print("\n[+] Step completed and execution evidence recorded.")

        elif choice in ("m", "mark"):
            print("Optional note/output to attach for this step (type/paste text, finish with 'EOF' or press Enter on empty line):")
            note_lines = []
            while True:
                try:
                    nl = input()
                    if nl.strip().upper() in ("EOF", "END"):
                        break
                    if not nl.strip() and (not note_lines or len(note_lines) == 1):
                        break
                    note_lines.append(nl)
                except (KeyboardInterrupt, EOFError):
                    break
            note = "\n".join(note_lines).strip()
            res = {"command": commands[0] if commands else "manual", "status": "manual_completed", "stdout": note}
            self.session_mgr.complete_current_step(res)
            if note:
                self.session_mgr.add_note(f"Step #{current_step.get('step_number')} Output:\n{note}")
            print("\n[+] Step marked as completed.")
        else:
            print("\n[*] Execution cancelled. Step remains pending.")

    # ------------------------------------------------------------------
    # Command: paste
    # ------------------------------------------------------------------
    def do_paste(self, arg):
        """
        Prompt for a multi-line text block to manually add notes or scanner/tool output.
        Usage: paste
        """
        if not self._require_session():
            return

        print("\n--- PASTE SCAN / TOOL OUTPUT ---")
        print("Enter or paste text below. When finished, type 'EOF' or 'END' on a new line:")
        lines = []
        while True:
            try:
                line = input()
            except (KeyboardInterrupt, EOFError):
                break
            if line.strip().upper() in ("EOF", "END"):
                break
            lines.append(line)

        content = "\n".join(lines).strip()
        if not content:
            print("[!] Empty content; nothing saved.")
            return

        note = self.session_mgr.add_note(content)

        # Intelligent terminal scan output analysis
        analysis = self.ingestion_engine.analyze_terminal_text(content)
        detected_services = analysis.get("detected_services", [])
        entities = analysis.get("entities", {})
        has_findings = (
            detected_services
            or entities.get("ports")
            or analysis.get("web_endpoints")
            or entities.get("domains")
            or entities.get("vulnerabilities")
        )

        if has_findings:
            # Merge findings into session state
            self.session_mgr.add_ingested_findings(
                finding_type="terminal_paste",
                findings=analysis,
                raw_source=f"pasted_text_{note['id']}"
            )

        # Complete the active pending step if one exists, or auto-infer one from planner
        current_step = self.session_mgr.state.get("current_step")
        if not current_step or current_step.get("id") == "engagement_complete":
            # Auto-infer current step from planner if none is set
            pending = self.planner.get_next_step(self.session_mgr.state)
            if pending:
                current_step = pending
                self.session_mgr.set_current_step(current_step)

        if current_step and current_step.get("id") not in ("engagement_complete", "flag_captured"):
            step_cmd = (current_step.get("commands") or ["manual_step"])[0]
            self.session_mgr.complete_current_step({
                "command": step_cmd,
                "status": "completed_via_paste",
                "stdout": content[:2000]
            })
            print(f"\n[+] Completed Step #{current_step.get('step_number')}: {current_step.get('title')}")
            print(f"    Evidence recorded (Note ID: {note['id']}).")

            if detected_services:
                print("\nDiscovered Services:")
                for srv in detected_services:
                    print(f"  - {srv}")

            # Query planner dynamically for the adapted next step
            print("\nUpdating assessment planner...\n")
            next_step = self.planner.get_next_step(self.session_mgr.state)
            self.session_mgr.set_current_step(next_step)

            if next_step.get("id") == "flag_captured":
                print("=" * 65)
                print("  >>> FLAG CAPTURED -- CHALLENGE COMPLETE! <<<")
                print("=" * 65)
                print(f"  {next_step.get('action')}")
                print(f"  {next_step.get('rationale')}")
                print("=" * 65 + "\n")
                print("Type 'report' to generate your final assessment report.\n")
            elif next_step.get("id") == "engagement_complete":
                print("=" * 65)
                print("  [+] ASSESSMENT METHODOLOGY COMPLETE")
                print("=" * 65)
                print(f"  {next_step.get('action')}")
                print(f"  {next_step.get('rationale')}")
                print("=" * 65 + "\n")
                print("Type 'report' to export final Markdown report.\n")
            else:
                print(f"Next Recommended Step #{next_step.get('step_number')}:")
                print(f"  {next_step.get('title')}\n")
                if next_step.get("commands"):
                    print("Run:")
                    for c in next_step.get("commands", []):
                        print(f"  {c}")
                    print()
                print("-" * 60)
                print("Run externally and paste output ('paste'), or type 'next' for details.")
                print("-" * 60 + "\n")
        else:
            next_step = self.planner.get_next_step(self.session_mgr.state)
            self.session_mgr.set_current_step(next_step)
            if next_step.get("id") == "flag_captured":
                print("\n" + "=" * 65)
                print("  >>> FLAG CAPTURED -- CHALLENGE COMPLETE! <<<")
                print("=" * 65)
                print(f"  {next_step.get('action')}")
                print(f"  {next_step.get('rationale')}")
                print("=" * 65 + "\n")
            elif next_step.get("id") == "engagement_complete":
                print("\n" + "=" * 65)
                print("  [+] ASSESSMENT METHODOLOGY COMPLETE")
                print("=" * 65)
                print(f"  {next_step.get('action')}")
                print(f"  {next_step.get('rationale')}")
                print("=" * 65 + "\n")
            else:
                if has_findings and detected_services:
                    print("\nDiscovered Services:")
                    for srv in detected_services:
                        print(f"  - {srv}")
                print(f"\n[+] Note saved successfully (ID: {note['id']}).")
                print(f"    Next step #{next_step.get('step_number')}: {next_step.get('title')}")
                print("    Type 'next' to receive the next recommended action item.\n")

    # ------------------------------------------------------------------
    # Command: upload <path>
    # ------------------------------------------------------------------
    def do_upload(self, arg):
        """
        Trigger the file ingestion pipeline for a local scanner file (Nmap XML, Burp XML/TXT, Cloud JSON, Text).
        Usage: upload <path_to_file>
        """
        if not self._require_session():
            return

        file_path = arg.strip()
        if not file_path:
            file_path = input("Enter path to file to ingest: ").strip()

        if not file_path:
            print("[!] File path required.")
            return

        file_path = file_path.strip("\"'")

        print(f"\n[*] Ingesting and analyzing '{file_path}'...")
        try:
            result = self.ingestion_engine.ingest_file(
                file_path,
                artifacts_dir=self.session_mgr.artifacts_dir
            )

            # Store in session state
            self.session_mgr.add_ingested_findings(
                finding_type=result.parser_type,
                findings=result.to_dict(),
                raw_source=result.source_path
            )

            # Check if this scan contains ports / services
            ports = result.entities.get("ports", [])
            friendly_services = []
            if result.parser_type == "nmap":
                for host in result.details.get("hosts", []):
                    for p in host.get("open_ports", []):
                        srv_name = p.get("service", "").upper()
                        if srv_name and srv_name not in friendly_services:
                            friendly_services.append(srv_name)

            # Advance previous step if pending
            current_step = self.session_mgr.state.get("current_step")
            if current_step:
                self.session_mgr.complete_current_step({
                    "command": f"upload {file_path}",
                    "status": "completed_via_upload",
                    "stdout": result.summary
                })

            print("\nDetected\n")
            if friendly_services:
                for srv in friendly_services:
                    print(f"{srv}")
            elif ports:
                print(f"Open Ports: {', '.join(str(p) for p in ports)}")
            else:
                print(result.summary)

            lower_summary = result.summary.lower()
            if any(p in ports for p in (88, 389, 445)) or "kerberos" in lower_summary:
                profile = "Likely Active Directory."
            elif any(p in ports for p in (80, 443, 8080)):
                profile = "Web Application Server."
            else:
                profile = "Identified Host."

            print(f"\n{profile}")
            print("\nUpdating assessment...\n")

            # Query planner dynamically for the adapted next step
            next_step = self.planner.get_next_step(self.session_mgr.state)
            self.session_mgr.set_current_step(next_step)

            print("Suggested next step:\n")
            print(f"{next_step.get('title')}.\n")
            if next_step.get("commands"):
                print(f"Run: {next_step['commands'][0]}\n")
            print("-" * 60)
            print("Type 'next' to inspect the full step details or 'run' to execute.")
            print("-" * 60 + "\n")

        except Exception as e:
            print(f"[!] Failed to ingest file: {e}")

    # ------------------------------------------------------------------
    # Command: report
    # ------------------------------------------------------------------
    def do_report(self, arg):
        """
        Compile accumulated session notes, findings, and evidence into Markdown report.
        Usage: report
        """
        if not self._require_session():
            return

        report_filename = self.session_mgr.session_dir / "report.md"
        print(f"\n[*] Generating structured assessment report for '{self.session_mgr.active_session_name}'...")

        try:
            report_text = self.report_gen.generate(
                self.session_mgr.state,
                output_path=str(report_filename)
            )

            print("\n" + "=" * 65)
            print("  ASSESSMENT REPORT COMPILED SUCCESSFULLY")
            print("=" * 65)
            print(f"Saved Path:     {report_filename}")
            print(f"Report Size:    {len(report_text)} characters")
            print(f"Completed Steps:{len(self.session_mgr.state.get('completed_steps', []))}")
            print(f"Evidence Proofs:{len(self.session_mgr.state.get('evidence', []))}")
            print("=" * 65 + "\n")
        except Exception as e:
            print(f"[!] Report compilation failed: {e}")

    # ------------------------------------------------------------------
    # Command: quit / exit
    # ------------------------------------------------------------------
    def do_quit(self, arg):
        """
        Safely save session state and exit the CLI.
        Usage: quit
        """
        if self.session_mgr.active_session_name:
            print(f"\n[*] Saving session state for '{self.session_mgr.active_session_name}'...")
            try:
                self.session_mgr.save_current_state()
                print("[+] State successfully persisted.")
            except Exception as e:
                print(f"[!] Warning during save: {e}")

        print("\nExiting Assessment CLI. Goodbye!\n")
        return True

    def do_exit(self, arg):
        """Exit the CLI."""
        return self.do_quit(arg)

    # ------------------------------------------------------------------
    # Command: chat <message>
    # ------------------------------------------------------------------
    def do_chat(self, arg):
        """
        Ask the AI Security Co-Pilot questions or discuss next moves.
        Usage: chat <your question or thought>
        Note: You can also just type any question directly without 'chat'!
        """
        query = arg.strip()
        if not query:
            print("[!] Please provide a question or topic to discuss with the AI.")
            return

        if not self.chat_assistant:
            print("[!] ChatAssistant is not available.")
            return

        state = self.session_mgr.state if self.session_mgr.active_session_name else {}
        print("\n[*] Consulting AI Security Co-Pilot...")
        answer = self.chat_assistant.respond(query, state)
        print(f"\n{answer}\n")

        # If in an active session, record note and check if operator steered towards exploitation
        if self.session_mgr.active_session_name:
            # Only append '...' if the answer was actually truncated
            note_text = answer[:400] + ("..." if len(answer) > 400 else "")
            self.session_mgr.add_note(f"Operator Question: {query}\nAI Response: {note_text}")
            if state.get("operator_directive") == "exploit":
                # Align active step with exploit if applicable
                pending = self.planner.get_next_step(self.session_mgr.state)
                if pending and pending.get("id") == "ctf_path_privesc_exploit":
                    self.session_mgr.set_current_step(pending)
        else:
            print("[Tip: Type 'new <name>' or 'open <name>' to start tracking an active assessment session.]\n")

    def default(self, line):
        """Route natural language questions, conversational steering, or commands to the AI Co-Pilot."""
        clean_line = line.strip()
        if not clean_line:
            return

        lowered = clean_line.lower()

        # Handle natural conversational shortcuts for key commands
        if lowered in ("what's next?", "what's next", "whats next", "what is next", "next step", "what should i do next", "what should i do now"):
            if self.session_mgr.active_session_name:
                return self.do_next("")
        elif lowered in ("paste output", "i want to paste", "paste results", "paste evidence"):
            if self.session_mgr.active_session_name:
                return self.do_paste("")
        elif lowered in ("show status", "what is the status?", "status overview"):
            if self.session_mgr.active_session_name:
                return self.do_status("")
        elif lowered in ("show report", "generate report", "export report"):
            if self.session_mgr.active_session_name:
                return self.do_report("")

        # Route directly to AI conversational chat
        self.do_chat(clean_line)


def main():
    try:
        cli = AssessmentCLI()
        cli.cmdloop()
    except (KeyboardInterrupt, EOFError):
        print("\n\nSession interrupted. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
