#!/usr/bin/env python3
"""
run_target.py

Run the autonomous BugBountyAgent against ONE *real* (non-lab) in-scope host,
with the safe real-target profile:

  * STEALTH ON by default (paced + jittered requests, adaptive WAF/429 backoff).
  * SCANNERS OFF by default (no wordlist dir-brute, sqlmap, nuclei, dalfox) --
    turn on with --allow-scanners ONLY if the program's policy permits automated
    scanning.
  * HUMAN APPROVAL STAYS ON -- every model-written self_script hits the [y/N]
    gate before it touches the target. This harness REFUSES to run if the lab
    auto-approve env (PENTEST_LAB_AUTO_APPROVE=1) is set.
  * SCOPE LOCKED to the in-scope asset(s) you pass; out-of-scope entries are hard
    exclusions; an --authorization-ref is REQUIRED (you assert you're authorized).

AUTHORIZATION: only point this at assets inside a bug bounty program you are
operating within, or that you have explicit written permission to test. That is
what makes this hunting and not a crime. This tool enforces scope, not law --
the authorization is on you.

Usage:
    python3 run_target.py https://app.example.com/ \\
        --authorization-ref "HACKERONE: example / handle: yourname" \\
        --in-scope app.example.com --out-of-scope admin.example.com \\
        [--allow-scanners] [--no-stealth] [--max-steps 40]

Outputs (kept SEPARATE from lab data):
    data/real_target_trajectories_raw.jsonl
    data/real_target_trajectories_chatml.jsonl
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.scope import Scope
from core.session_manager import SessionManager
from agents.bug_bounty.agent import BugBountyAgent

# Reuse the lab harness's helpers (hostname parse, ChatML builder, confirmed-vuln scan).
from run_portswigger_lab import _lab_hostname, _build_chatml_record, _confirmed_vuln_from_history

ROOT_DIR = Path(__file__).resolve().parent
RAW_LOG_PATH = ROOT_DIR / "data" / "real_target_trajectories_raw.jsonl"
CHATML_LOG_PATH = ROOT_DIR / "data" / "real_target_trajectories_chatml.jsonl"


def _write_raw(target, name, scope_dict, result, history):
    RAW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "target": target, "name": name, "scope": scope_dict,
        "result": result, "execution_history": history,
    }
    with open(RAW_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="The in-scope URL/host to assess (one host per run).")
    ap.add_argument("--authorization-ref", required=True,
                    help="REQUIRED: the program/permission you are operating under "
                         "(e.g. 'HACKERONE: acme / researcher: you').")
    ap.add_argument("--name", default="", help="Human-readable label for logging.")
    ap.add_argument("--in-scope", action="append", default=[],
                    help="In-scope asset(s). Defaults to the target's hostname if omitted.")
    ap.add_argument("--out-of-scope", action="append", default=[],
                    help="Hard-excluded asset(s) the agent must never touch.")
    ap.add_argument("--prohibit-tool", action="append", default=[], help="Prohibited tool name(s).")
    ap.add_argument("--prohibit-test", action="append", default=[],
                    help="Prohibited test type(s), e.g. 'dos', 'credential'.")
    ap.add_argument("--allow-scanners", action="store_true",
                    help="Enable loud automated scanners (dir-brute/sqlmap/nuclei/dalfox). "
                         "Set ONLY if the program policy permits automated scanning.")
    ap.add_argument("--no-stealth", action="store_true",
                    help="Disable the stealth profile (NOT recommended for real targets).")
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()

    # Hard safety interlock: never auto-approve model-written code against a real target.
    if os.environ.get("PENTEST_LAB_AUTO_APPROVE") == "1":
        print("[REFUSING] PENTEST_LAB_AUTO_APPROVE=1 is set. That lab-only switch auto-runs "
              "model-written code with no human review -- never against a real target. "
              "Unset it and re-run:  unset PENTEST_LAB_AUTO_APPROVE")
        sys.exit(2)

    host = _lab_hostname(args.target)
    in_scope = args.in_scope or [host]
    name = args.name or args.target

    scope = Scope(
        target=args.target,
        in_scope=in_scope,
        out_of_scope=args.out_of_scope,
        prohibited_tools=args.prohibit_tool,
        prohibited_tests=args.prohibit_test,
        active_testing_allowed=True,
        active_network_allowed=True,
        state_changes_allowed=False,
        authorization_ref=args.authorization_ref,
        single_host_only=True,
        stealth=not args.no_stealth,
        allow_scanners=args.allow_scanners,
    )

    print("=" * 74)
    print(f"REAL-TARGET RUN: {name}")
    print(f"  Target:        {args.target}")
    print(f"  In scope:      {in_scope}")
    print(f"  Out of scope:  {args.out_of_scope or '(none)'}")
    print(f"  Authorization: {args.authorization_ref}")
    print(f"  Stealth:       {'ON' if scope.stealth else 'OFF (!)'}")
    print(f"  Scanners:      {'ALLOWED (policy permits automation)' if scope.allow_scanners else 'OFF (primitives only)'}")
    print(f"  Approval:      INTERACTIVE (every self_script needs your [y/N])")
    print(f"  Max steps:     {args.max_steps}")
    print("=" * 74)
    print("Reminder: verify any lead by hand before you report it. Leads != confirmed bugs.\n")

    session_mgr = SessionManager()
    agent = BugBountyAgent(scope=scope, mode="automated", session_manager=session_mgr)
    registry = agent.get_action_registry()

    result = agent.run_autonomous(
        registry=registry,
        completion_criteria=agent.get_completion_criteria(),
        max_steps=args.max_steps,
    )

    history = session_mgr.state.get("autonomous_execution_history", [])
    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(json.dumps(result, indent=2, default=str))
    print(f"\nConfirmed a vuln signal: {_confirmed_vuln_from_history(history)}")

    _write_raw(args.target, name, scope.to_dict() if hasattr(scope, "to_dict") else {}, result, history)
    chatml = _build_chatml_record(args.target, name, result, history, registry)
    CHATML_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHATML_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(chatml) + "\n")
    print(f"\nRaw trajectory:  {RAW_LOG_PATH}")
    print(f"ChatML record:   {CHATML_LOG_PATH}")


if __name__ == "__main__":
    main()
