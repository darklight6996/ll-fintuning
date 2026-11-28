#!/usr/bin/env python
"""
run_portswigger_lab.py

Run the plain BugBountyAgent, with Scope.single_host_only=True, autonomously
against ONE PortSwigger Web Security Academy lab instance, then convert the
real trajectory it produced into training-data records matching the schema
the existing qwen-qlora pipeline already trains on (data/qwen_chatml.jsonl,
single-field ChatML {"text": "<|im_start|>..."}).

Why single_host_only=True: BUG_BOUNTY_COMPLETION normally requires the
subdomain_enum phase (subfinder against a whole domain) and the registry
includes port_scan (nmap against a whole host) -- both scope-inappropriate
for a single-lab-instance target. Setting Scope.single_host_only=True makes
BugBountyAgent.get_action_registry()/get_completion_criteria() swap in
agents/bug_bounty/single_host_mode.py's registry/criteria instead (see that
module's docstring) -- a deterministic, scope-driven choice made by the
framework, never something the AI planner decides for itself at runtime. No
subclass is needed: this is the SAME BugBountyAgent used for real,
whole-domain bug-bounty engagements -- it just reads an explicit flag on the
Scope you hand it. Any other single-URL/single-IP target (not just
PortSwigger) gets the same treatment by setting the same flag.

Scope lock: in_scope is set to EXACTLY the lab's own hostname (the random
per-instance hash PortSwigger assigns, e.g. 0a1b...web-security-academy.net),
never the shared parent domain -- see core/scope.py Scope.is_asset_in_scope().
state_changes_allowed is left False; every bug_bounty action's
ActionDefinition.state_change is already "none" so this has no practical
effect on what the agent can do, it just keeps the same conservative default
used by verify_pentest_smoke_test.py / verify_autohunt_smoke_test.py.

Usage:
    python3 run_portswigger_lab.py <lab_url> [--name "Lab: SQLi in WHERE clause"] [--max-steps 25]

    <lab_url>   The lab's own unique instance URL PortSwigger gave you after
                clicking "Access the lab", e.g.:
                https://0a1b00c2034d.web-security-academy.net/

Outputs (per run, appended -- never overwritten):
    data/portswigger_trajectories_raw.jsonl     Full-fidelity structured log of
                                                 every step (for re-deriving
                                                 training data later in a
                                                 different style without
                                                 re-running the lab).
    data/portswigger_trajectories_chatml.jsonl  One ChatML {"text": ...}
                                                 record per completed/attempted
                                                 lab run, in the exact schema
                                                 data/qwen_chatml.jsonl uses.
                                                 Kept SEPARATE from
                                                 qwen_chatml.jsonl itself so
                                                 nothing is fed into a future
                                                 fine-tune without a look first
                                                 -- merge the two files
                                                 yourself once you're happy
                                                 with a batch of lab runs.

Go through labs one at a time: run this once per lab URL (PortSwigger gives a
fresh URL each time you click "Access the lab", even for the same lab name).
There's no PortSwigger API for listing/launching labs from here, so you
supply each lab_url yourself as you go through the Web Security Academy site.
"""
import argparse
import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from core.scope import Scope
from core.session_manager import SessionManager
from agents.bug_bounty.agent import BugBountyAgent

ROOT_DIR = Path(__file__).resolve().parent
RAW_LOG_PATH = ROOT_DIR / "data" / "portswigger_trajectories_raw.jsonl"
CHATML_LOG_PATH = ROOT_DIR / "data" / "portswigger_trajectories_chatml.jsonl"

CHATML_SYSTEM = (
    "You are a cybersecurity and penetration testing expert inside a live "
    "operator console."
)


def _lab_hostname(lab_url: str) -> str:
    parsed = urlparse(lab_url if "://" in lab_url else f"https://{lab_url}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"Could not parse a hostname out of lab_url={lab_url!r}")
    return host


def _truncate(s: str, n: int = 400) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 3] + "..."


def _write_raw_trajectory(lab_url: str, lab_name: str, result: dict, history: list) -> None:
    RAW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "lab_url": lab_url,
        "lab_name": lab_name,
        "result": result,
        "execution_history": history,
    }
    with open(RAW_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _build_chatml_record(lab_url: str, lab_name: str, result: dict, history: list, registry) -> dict:
    """
    Build ONE ChatML training record summarizing this real lab run, in the
    same {"text": "<|im_start|>system...<|im_end|>\\n<|im_start|>user...
    <|im_end|>\\n<|im_start|>assistant...<|im_end|>\\n"} shape as
    data/qwen_chatml.jsonl, so it can be concatenated straight into that file
    once reviewed.

    The assistant turn is built ONLY from what actually happened in this run
    (real argv, real exit codes, real finding text) -- never invented -- so a
    failed/incomplete run still produces an honest record rather than being
    silently dropped.
    """
    user_msg = (
        f"Walk through how you would approach the PortSwigger Web Security "
        f"Academy lab \"{lab_name}\" at {lab_url}, and report what you find.\n\n"
        f"Context:\nYou have been given direct, authorized access to this single "
        f"lab instance. Stay scoped to this one host only."
    )

    steps_text = []
    for entry in history:
        action_id = entry.get("action_id")
        status = entry.get("status")
        if not action_id or action_id == "planner_decision":
            continue
        defn = registry.get(action_id) if hasattr(registry, "get") else None
        phase = getattr(defn, "phase", None) or entry.get("action_identity") or "recon"
        results = entry.get("results") or []
        argv_strs = []
        output_snips = []
        for res in results:
            argv = res.get("argv")
            if argv:
                argv_strs.append(" ".join(str(a) for a in argv))
            out = res.get("stdout") or res.get("stderr") or ""
            if out:
                output_snips.append(_truncate(out, 300))
        step_line = f"[{phase}] {action_id} -> {status}"
        if argv_strs:
            step_line += f"\n   Command: `{argv_strs[0]}`"
        if output_snips:
            step_line += f"\n   Observed: {output_snips[0]}"
        if entry.get("rejection_reason"):
            step_line += f"\n   Rejected: {entry['rejection_reason']}"
        steps_text.append(step_line)

    findings_count = result.get("findings_count", 0)
    status = result.get("status", "unknown")

    assistant_parts = [
        f"Methodology actually executed against this lab instance "
        f"({result.get('steps_run', len(history))} step(s), status={status}):",
        "",
        *[f"{i+1}. {s}" for i, s in enumerate(steps_text)],
        "",
    ]
    if findings_count > 0:
        assistant_parts.append(
            f"Result: {findings_count} confirmed finding(s) were recorded for this lab, "
            f"meeting the framework's evidence-chain completion criteria."
        )
    else:
        assistant_parts.append(
            f"Result: run ended with status='{status}' and no confirmed finding yet. "
            f"Further probing (additional param/endpoint coverage) would be needed."
        )
    assistant_msg = "\n".join(assistant_parts)

    text = (
        f"<|im_start|>system\n{CHATML_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
    )
    return {"text": text}


def _confirmed_vuln_from_history(history: list) -> bool:
    """
    HONEST success signal: did any probe/engine emit a CONFIRMED verdict during
    this run? Scans every observation stdout for a vuln_signal with
    confirmed==true -- independent of findings_count (which also counts
    info-severity phase-milestone promotions), so it reflects whether the agent
    actually confirmed a real vulnerability, not just that it ran steps.
    """
    for entry in history:
        for res in (entry.get("results") or []):
            out = res.get("stdout") or ""
            if "vuln_signal" not in out or "confirmed" not in out:
                continue
            for line in out.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sig = obj.get("vuln_signal") if isinstance(obj, dict) else None
                if isinstance(sig, dict) and sig.get("confirmed") is True:
                    return True
    return False


def run_lab(lab_url: str, lab_name: str = "", max_steps: int = 25, verbose: bool = True,
            manual_payload: bool = False, sweep: bool = False) -> dict:
    """
    Run ONE lab autonomously, log raw + ChatML trajectory, return a compact
    summary. Importable so a batch runner can loop over many labs in ONE process
    -- the GPU model is a module-level singleton, so it loads once and is reused
    across every call instead of paying the ~20s load per lab.
    """
    lab_name = lab_name or lab_url
    hostname = _lab_hostname(lab_url)

    # Declare this as a LAB engagement for the risk-tiered approval gate
    # (BUILD SPEC v1 workstream B). This marks the target "lab" so Tier 0/1
    # actions auto-proceed on fake-data lab instances; Tier 2 (dangerous) still
    # requires approval unless PENTEST_LAB_AUTO_APPROVE[_DANGEROUS]=1 is set.
    # run_target.py sets this to "real" and refuses the lab flags, so real
    # targets are never mis-tiered.
    os.environ["PENTEST_TARGET_KIND"] = "lab"

    # NO SILENT FALLBACK (2026-11-27): verify which brain is ACTUALLY going to
    # serve this run before doing anything else. llm_service.py's CPU path is a
    # real, working code path (useful for quick dev smoke tests), which is
    # exactly why it can silently produce a run that LOOKS like it executed but
    # is reasoning with google/flan-t5-large -- too small to be a meaningful
    # test of anything. Fail loudly and refuse to start rather than let that
    # happen quietly; PENTEST_ALLOW_CPU_FALLBACK=1 is the explicit, named
    # escape hatch for someone who really does want a CPU smoke test.
    from core.services.llm_service import get_llm_service
    _svc = get_llm_service(verbose=True, role="planner")
    _backend_mode = getattr(_svc, "mode", None) if _svc else None
    # Require an explicit real backend (ollama/openrouter) -- NOT just "mode is
    # not cpu", because a misconfigured OpenRouter (e.g. key set, model unset)
    # fails to LOAD at all (_svc is None / mode defaults to "cpu"'s init value),
    # which must refuse exactly the same way a bare unset-everything case does.
    if _backend_mode not in ("ollama", "openrouter") and os.environ.get("PENTEST_ALLOW_CPU_FALLBACK") != "1":
        print("\n[REFUSING] No real backend is configured (or it failed to load) -- this would "
              "silently run on the CPU fallback model (google/flan-t5-large), which is "
              "meaningless for a lab assessment. Set one real brain before running:")
        print("  OPENROUTER_API_KEY + OPENROUTER_MODEL   (a frontier model via OpenRouter), or")
        print("  OLLAMA_BASE_URL + OLLAMA_MODEL           (the local Ollama server)")
        print("If you genuinely want a CPU smoke test, set PENTEST_ALLOW_CPU_FALLBACK=1.")
        sys.exit(2)
    if verbose:
        print("=" * 70)
        print(f"PORTSWIGGER LAB RUN: {lab_name}")
        print(f"Target URL: {lab_url}")
        print(f"Locked scope hostname: {hostname}")
        print(f"Max steps: {max_steps}")
        print(f"Brain backend: {_backend_mode or '(unavailable)'}"
              f"{' -- ' + _svc.model_name if _svc and getattr(_svc, 'model_name', None) else ''}")
        print("=" * 70)

    scope = Scope(
        target=lab_url,
        in_scope=[hostname],
        active_testing_allowed=True,
        active_network_allowed=True,
        state_changes_allowed=False,
        authorization_ref=f"PORTSWIGGER-ACADEMY-OWN-ACCOUNT:{hostname}",
        single_host_only=True,
    )

    session_mgr = SessionManager()
    agent = BugBountyAgent(scope=scope, mode="automated", session_manager=session_mgr)
    registry = agent.get_action_registry()

    result = agent.run_autonomous(
        registry=registry,
        completion_criteria=agent.get_completion_criteria(),
        max_steps=max_steps,
        manual_payload=manual_payload,
        sweep=sweep,
    )

    history = session_mgr.state.get("autonomous_execution_history", [])

    if verbose:
        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(json.dumps(result, indent=2, default=str))

    _write_raw_trajectory(lab_url, lab_name, result, history)
    chatml_record = _build_chatml_record(lab_url, lab_name, result, history, registry)
    CHATML_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHATML_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(chatml_record) + "\n")

    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING DATA LOGGED")
        print("=" * 70)
        print(f"Raw trajectory appended to:    {RAW_LOG_PATH}")
        print(f"ChatML record appended to:     {CHATML_LOG_PATH}")

    return {
        "lab_url": lab_url,
        "lab_name": lab_name,
        "status": result.get("status"),
        "steps_run": result.get("steps_run"),
        "findings_count": result.get("findings_count", 0),
        "confirmed_vuln": _confirmed_vuln_from_history(history),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lab_url", help="The lab's own unique instance URL from PortSwigger.")
    parser.add_argument("--name", default="", help="Human-readable lab name/title for logging.")
    parser.add_argument("--max-steps", type=int, default=25, help="Max autonomous steps (default 25).")
    parser.add_argument("--blind", action="store_true",
                        help="Strip PortSwigger lab identifiers (title, banner, solution link) "
                             "from what the agent sees, so it must find the bug from the real app "
                             "surface -- a fair proxy for a live target.")
    parser.add_argument("--manual-payload", action="store_true",
                        help="Human-in-the-loop payload handoff: whenever the planner would run its "
                             "own self_script probe, pause and print a payload brief (vuln_class, "
                             "endpoint, exact parameter, observed response, what would confirm it) "
                             "instead, then read a human-pasted PAYLOAD:/CURL:/SCRIPT: from stdin and "
                             "run IT through the same gated executor + confirmation oracle. Requires "
                             "an interactive terminal (the HIGH-RISK approval gate still applies too).")
    parser.add_argument("--sweep", action="store_true",
                        help="METHODICAL CLASS-SWEEP hunter mode: enumerate the full surface, "
                             "characterize it (cheap name-based tags), run the context gate to decide "
                             "which vuln classes are even plausible per endpoint/param, and track a "
                             "coverage ledger (surfaced to the planner) instead of tunnel-visioning on "
                             "the first interesting endpoint. Raises the default --max-steps to 80 "
                             "unless --max-steps is given explicitly (it's local; the only cost is "
                             "wall-time). Does not change any gate -- same scope/approval/confirmation-"
                             "rigor path as every other mode.")
    args = parser.parse_args()
    if args.blind:
        os.environ["PENTEST_BLIND_LAB"] = "1"
    max_steps = args.max_steps
    if args.sweep and max_steps == 25:  # user didn't override --max-steps
        max_steps = 80
    run_lab(args.lab_url, args.name, max_steps, verbose=True, manual_payload=args.manual_payload,
            sweep=args.sweep)


if __name__ == "__main__":
    main()
