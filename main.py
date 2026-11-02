"""
main.py

Multi-Agent Security Assessment Framework — System Orchestrator (Layer 1)

Responsibilities:
1. Detect Operating System (Windows / Linux Debian / Linux Ubuntu / macOS)
2. Check installed tools; prompt to install missing ones via OS package manager (winget/choco, apt, brew)
3. Display Agent Menu (all 9 specialized assessment agents)
4. Gather Scope from operator (Target, In-scope, Out-of-scope, Prohibited tests/tools, Time window, Auth reference)
5. Select Execution Mode (Manual Co-Pilot vs Automated Sequential)
6. Instantiate the selected Agent with the Scope object
7. Hand over complete control to the Agent

Architectural Rule:
The Orchestrator does NOT run tests, make tactical decisions, or call the AI directly.
All reasoning occurs inside Layer 2 Agents & Layer 4 ChatAssistant.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, List

# Ensure UTF-8 output on Windows console
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# Ensure project directories are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
for p in (CURRENT_DIR, CURRENT_DIR / "rag", CURRENT_DIR / "ll-finetuning" / "rag"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.os_detector import detect_os, OSInfo
from core.tool_manager import ToolManager
from core.scope import Scope
from core.agent_router import AgentRouter
from core.session_manager import SessionManager


BANNER = r"""
================================================================================
   __  ___      __  __  _      ___                    __ 
  /  |/  /_  __/ / / /_(_)    /   | ____ ____  ____  / /_
 / /|_/ / / / / / / __/ /    / /| |/ __ `/ _ \/ __ \/ __/
/ /  / / /_/ / / / /_/ /    / ___ / /_/ /  __/ / / / /_  
/_/  /_/\__,_/_/_/\__/_/    /_/  |_\__, /\___/_/ /_/\__/  
                                  /____/                  
      MULTI-AGENT SECURITY ASSESSMENT & ADVERSARY EMULATION FRAMEWORK
================================================================================
"""


def prompt_string(prompt_text: str, default: str = "") -> str:
    """Prompt operator for single line input with default fallback."""
    suffix = f" [default: {default}]: " if default else ": "
    try:
        val = input(prompt_text + suffix).strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print("\n\nOperation cancelled by user.")
        sys.exit(0)


def prompt_list(prompt_text: str, default_list: Optional[List[str]] = None) -> List[str]:
    """Prompt operator for comma-separated or space-separated items."""
    def_str = ", ".join(default_list) if default_list else "None"
    suffix = f" (comma-separated) [default: {def_str}]: "
    try:
        raw = input(prompt_text + suffix).strip()
        if not raw:
            return default_list or []
        items = [i.strip() for i in raw.replace(";", ",").split(",") if i.strip()]
        return items
    except (KeyboardInterrupt, EOFError):
        print("\n\nOperation cancelled by user.")
        sys.exit(0)


def prompt_yes_no(prompt_text: str, default: bool = False) -> bool:
    """Prompt operator for boolean confirmation."""
    def_str = "Y/n" if default else "y/N"
    try:
        val = input(f"{prompt_text} [{def_str}]: ").strip().lower()
        if not val:
            return default
        return val in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        print("\n\nOperation cancelled by user.")
        sys.exit(0)


def check_and_install_prerequisites(tool_mgr: ToolManager) -> None:
    """Pre-flight tool check and operator-approved package manager installation."""
    os_info = tool_mgr.os_info
    print("\n[*] Pre-Flight Environmental Tool Check:")
    common_tools = ["curl", "nmap", "python", "git"]
    results = tool_mgr.check_tools(common_tools)

    missing = []
    for tool, info in results.items():
        status = "[INSTALLED]" if info["installed"] else "[MISSING]"
        print(f"    {status:<12} {tool:<12} ({info['version']})")
        if not info["installed"]:
            missing.append(tool)

    if missing:
        print(f"\n[!] Missing core tools: {', '.join(missing)}")
        if os_info.package_manager != "none":
            if prompt_yes_no(f"Install missing tools now via {os_info.package_manager}?", default=True):
                for t in missing:
                    print(f"[*] Installing {t}...")
                    ok, msg = tool_mgr.install_tool(t)
                    print(f"    {msg}")
        else:
            print("[*] Note: No automated package manager detected. Please ensure assessment tools are on PATH.")


def gather_scope(os_info: OSInfo) -> Scope:
    """
    Interactively gather strict engagement boundaries from operator.
    """
    print("\n" + "=" * 70)
    print("  ENGAGEMENT SCOPE & BOUNDARY SPECIFICATION")
    print("=" * 70)
    print("Provide engagement parameters. Out-of-scope assets and prohibited tests")
    print("are strictly enforced by the Agent & AI Scope Validator before execution.\n")

    target = prompt_string("Primary Target (IP, Domain, CIDR, URL, APK, or Tenant ID)", default="127.0.0.1")

    in_scope_default = [target]
    in_scope = prompt_list("In-Scope Assets", default_list=in_scope_default)
    out_of_scope = prompt_list("Out-of-Scope Assets (EXPLICITLY FORBIDDEN)", default_list=[])

    print("\nSafety Constraints:")
    prohibited_tests = prompt_list(
        "Prohibited Tests (e.g. no DoS, no credential dumping, no social engineering)",
        default_list=["no DoS", "no social engineering"]
    )
    prohibited_tools = prompt_list(
        "Prohibited Tools (e.g. no Metasploit, no sqlmap)",
        default_list=[]
    )

    time_window = prompt_string("Authorized Time Window", default="Business Hours / Anytime")
    auth_ref = prompt_string("Authorization Reference (Contract / Ticket / JIRA ID)", default="DEMO-AUTH-2026")

    print("\nExecution Authorization:")
    print("  Safe/Passive Mode:  Runs non-destructive queries and scans (read_only=True).")
    print("  Active Testing:     Authorizes simulated exploit validations and active probes.")
    active_allowed = prompt_yes_no("Authorize Active Testing?", default=False)

    return Scope(
        target=target,
        in_scope=in_scope,
        out_of_scope=out_of_scope,
        prohibited_tests=prohibited_tests,
        prohibited_tools=prohibited_tools,
        time_window=time_window,
        authorization_ref=auth_ref,
        active_testing_allowed=active_allowed
    )


def select_agent() -> str:
    """Display all 9 specialized assessment agents and return operator choice."""
    agents = AgentRouter.list_agents()

    print("\n" + "=" * 70)
    print("  SELECT ASSESSMENT AGENT")
    print("=" * 70)
    for idx, a in enumerate(agents, 1):
        print(f"  [{idx}] {a.name:<40} ({a.key})")
        print(f"      {a.description}")

    while True:
        choice = prompt_string("\nSelect Agent (1-9)", default="3")
        meta = AgentRouter.get_agent_meta(choice)
        if meta:
            print(f"\n[+] Selected Agent: {meta.name} ({meta.key})")
            return choice
        print("[!] Invalid selection. Please enter a number between 1 and 9.")


def select_mode() -> str:
    """Prompt operator for execution mode."""
    print("\n" + "=" * 70)
    print("  SELECT OPERATION MODE")
    print("=" * 70)
    print("  [1] Manual Mode     - Per-phase operator approval gate (Y/N/Skip/Paste)")
    print("  [2] Automated Mode  - Autonomous sequential execution with scope-compliant safety gates")
    choice = prompt_string("\nEnter Mode [1-2]", default="1")
    return "automated" if choice.strip() == "2" else "manual"


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Security Assessment Framework")
    parser.add_argument("--resume", type=str, default=None, help="Resume an existing assessment session")
    parser.add_argument("--agent", type=str, default=None, help="Directly select agent by key or number (1-9)")
    parser.add_argument("--mode", type=str, choices=["manual", "automated"], default=None, help="Execution mode")
    parser.add_argument("--autohunt", action="store_true", help="Launch agent in autonomous hunter mode")
    parser.add_argument("--target", type=str, default=None, help="Directly specify target identifier")
    args = parser.parse_args()

    print(BANNER)

    # 1. OS Detection (Layer 1 Responsibility)
    os_info = detect_os()
    print(f"[*] Operating System:  {os_info.friendly_name} ({os_info.system})")
    print(f"[*] Distro / Family:   {os_info.distro.upper()}")
    print(f"[*] Package Manager:   {os_info.package_manager}")
    print(f"[*] Elevated Privs:    {'YES' if os_info.is_admin else 'NO (Standard User)'}")

    # 2. Tool Manager & Pre-Flight Checks
    tool_mgr = ToolManager(os_info)
    check_and_install_prerequisites(tool_mgr)

    # 3. Handle Resume Existing Session
    session_mgr = SessionManager()
    if args.resume:
        try:
            state = session_mgr.load_session(args.resume)
            print(f"\n[+] Resumed existing session '{args.resume}'.")
            target = state.get("target", "127.0.0.1")
            agent_type = state.get("type", "pentest")
            scope = Scope(target=target, in_scope=[target])
            agent = AgentRouter.instantiate_agent(
                identifier=agent_type,
                scope=scope,
                mode=state.get("mode", "manual"),
                session_manager=session_mgr
            )
            print(f"[*] Handing control to resumed agent: {agent.agent_name}...")
            agent.run()
            return
        except Exception as e:
            print(f"[!] Could not resume session '{args.resume}': {e}")
            sys.exit(1)

    # 4. Agent Selection Menu
    agent_id = args.agent or select_agent()

    # 5. Scope Gathering
    if args.target:
        scope = Scope(target=args.target, in_scope=[args.target])
    elif args.autohunt:
        # --autohunt without --target: prompt minimally (no full scope wizard)
        target = prompt_string("Target for autonomous hunt (IP, Domain, URL, CIDR)", default="127.0.0.1")
        scope = Scope(target=target, in_scope=[target])
    else:
        scope = gather_scope(os_info)

    # 6. Mode Selection
    if args.autohunt:
        mode = "automated"
    else:
        mode = args.mode or select_mode()

    # 7. Instantiate Agent & Hand Over Control
    agent_meta = AgentRouter.get_agent_meta(agent_id)
    print(f"\n[*] Initializing {agent_meta.name} in {mode.upper()} mode...")
    agent = AgentRouter.instantiate_agent(
        identifier=agent_id,
        scope=scope,
        mode=mode,
        session_manager=session_mgr
    )

    if args.autohunt and hasattr(agent, "get_action_registry"):
        registry = agent.get_action_registry()
        action_count = len(registry.list_actions())
        print(f"[*] Agent:   {agent.agent_name}")
        print(f"[*] Target:  {scope.target}")
        print(f"[*] Actions: {action_count} registered (autonomous hunter loop)")

    print(f"[+] Orchestrator handing complete control to {agent.agent_name}...\n")
    if mode == "automated" and hasattr(agent, "run_autonomous_default"):
        print("[*] Launching in Autonomous Hunter Loop...")
        agent.run_autonomous_default()
    else:
        agent.run()


if __name__ == "__main__":
    main()
