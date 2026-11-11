#!/usr/bin/env python
"""
verify_autohunt_smoke_test.py

A small, bounded, LOCAL-ONLY smoke test of the autonomous hunter loop
(agents.base_agent.BaseAgent.run_autonomous) after this session's remediation pass
(C1 full extension, C2, C3, C5, MED-1, MED-2, MED-3, MED-4, MED-5, --resume, M1, M2).

Runs the CTF agent against the machine THIS SCRIPT runs on (target="127.0.0.1").
CTF's phases are local-shell checks (id/whoami/find SUID/strings/privesc validation,
etc.) -- nothing here touches the network, and --authorize-network is deliberately
left OFF. active_testing_allowed / state_changes_allowed are turned on so the
planner has the full CTF action set available, including path_hijack_validation
(the one MED-5 fixed) -- it only ever writes a throwaway script into work_dir below
and does not actually need a vulnerable SUID binary to exist for the test to be
informative; it just needs to run without crashing or being rejected unsafely.

This prints a full JSON result, a human-readable evidence-chain summary (real
observation/hypothesis/validation/finding counts, not just status strings), and a
plain PASS/FAIL/PARTIAL verdict explaining what that combination actually means.

Usage:
    python verify_autohunt_smoke_test.py [max_steps]

Default max_steps = 8 -- small on purpose so this finishes in a few minutes even on
CPU. Raise it (e.g. 20) for a more thorough run once the small one looks sane.
"""
import sys
import json
import tempfile

from core.scope import Scope
from agents.ctf.agent import CTFAgent
from core.session_manager import SessionManager


def main():
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    print("=" * 70)
    print(f"AUTONOMOUS LOOP SMOKE TEST -- CTF agent, local target, max_steps={max_steps}")
    print("=" * 70)

    work_dir = tempfile.mkdtemp(prefix="ctf_smoke_")
    print(f"[*] Using scratch work_dir for path_hijack_validation: {work_dir}")

    scope = Scope(
        target="127.0.0.1",
        in_scope=["127.0.0.1", "local"],
        active_testing_allowed=True,
        active_network_allowed=False,
        state_changes_allowed=True,
        authorization_ref="SMOKE-TEST-LOCAL-ONLY",
    )

    session_mgr = SessionManager()
    agent = CTFAgent(scope=scope, mode="automated", session_manager=session_mgr)

    result = agent.run_autonomous_default(max_steps=max_steps)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str))

    store = session_mgr.load_investigation_store()
    print("\n" + "=" * 70)
    print("EVIDENCE CHAIN DETAIL")
    print("=" * 70)
    if store is None:
        print("No evidence store was persisted.")
    else:
        confirmed = sum(1 for h in store.hypotheses.values() if h.status == "confirmed")
        print(f"Observations : {len(store.observations)}")
        print(f"Hypotheses   : {len(store.hypotheses)}  (confirmed: {confirmed})")
        print(f"Validations  : {len(store.validations)}")
        print(f"Findings     : {len(store.findings)}")
        for fid, finding in store.findings.items():
            print(f"  - [{fid}] {finding.severity.upper()}: {finding.title}")
            print(f"        evidence: {finding.evidence_summary[:150]!r}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    status = result.get("status")
    findings_count = result.get("findings_count", 0)
    if status == "completed" and findings_count >= 1:
        print("PASS: framework-verified completion WITH at least one real finding.")
    elif status == "max_steps_reached" and findings_count >= 1:
        print(
            "PARTIAL PASS: at least one finding WAS created (the evidence chain is "
            "working), but max_steps ran out before verify_completion() accepted "
            "completion. Try a higher max_steps, or this just reflects the local "
            "model needing more steps / being inconsistent about declaring done."
        )
    elif status in ("aborted_planner_failures", "stalled_no_progress"):
        print(
            f"FAIL (caught by a new guard): status='{status}'. MED-1/MED-2 did their "
            f"job of ending the run early with a clear reason instead of silently "
            f"burning max_steps, but the underlying cause (planner JSON failures or "
            f"zero progress) still needs a look -- check the printed RESULT above and "
            f"the .assessments/<session>/state.json 'autonomous_execution_history'."
        )
    elif findings_count == 0:
        print(
            "FAIL: no finding was ever created. This is the exact failure mode the "
            "whole C1 fix targeted -- worth investigating with the RESULT/history above."
        )
    else:
        print(f"UNCLEAR: status='{status}', findings_count={findings_count} -- inspect above.")


if __name__ == "__main__":
    main()
