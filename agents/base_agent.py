
def datetime_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
"""
agents/base_agent.py

BaseAgent class that all specialized assessment agents inherit from.
Coordinates:
- Deterministic phase sequence execution
- AI reasoning integration (AI as consultant, not driver)
- Safety gating via CommandExecutor (read_only unless scope authorizes active testing)
- Scope compliance checking before active testing
- Operator approval gates (Manual: prompt per phase; Automated: sequential with safety checks)
- Audit trail persistence via SessionManager and EvidenceManager
- Ctrl+C state saving and resuming
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path

from core.scope import Scope
from core.session_manager import SessionManager
from core.command_executor import CommandExecutor
from core.chat_assistant import ChatAssistant
from core.tool_manager import ToolManager


@dataclass
class PhaseResult:
    """Standardized result returned by every deterministic phase script."""
    phase_name: str
    status: str                              # 'success', 'failed', 'skipped', 'blocked'
    commands_run: List[Dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""
    findings: Dict[str, Any] = field(default_factory=dict)
    next_hint: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "status": self.status,
            "commands_run": self.commands_run,
            "raw_output": self.raw_output[:500] if self.raw_output else "",
            "findings": self.findings,
            "next_hint": self.next_hint,
            "error": self.error
        }


class BaseAgent:
    """
    Abstract BaseAgent providing the unified four-layer assessment lifecycle.
    """

    agent_type: str = "base"
    agent_name: str = "Base Security Agent"
    description: str = "Base security assessment agent."

    def __init__(
        self,
        scope: Scope,
        mode: str = "manual",
        session_manager: Optional[SessionManager] = None,
        chat_assistant: Optional[ChatAssistant] = None,
        command_executor: Optional[CommandExecutor] = None,
    ):
        self.scope = scope
        self.mode = mode.lower()  # "manual" or "automated"
        self.session_mgr = session_manager or SessionManager()
        self.chat_assistant = chat_assistant or ChatAssistant()
        
        # Read-only executor by default unless active testing is explicitly authorized
        read_only = not self.scope.active_testing_allowed
        self.executor = command_executor or CommandExecutor(read_only=read_only)
        self.tool_manager = ToolManager()
        self.phase_modules: List[Any] = []
        self._load_phases()

    def _load_phases(self) -> None:
        """Must be implemented by subclasses to register their deterministic phase modules."""
        raise NotImplementedError("Subclasses must implement _load_phases()")

    def get_phase_names(self) -> List[str]:
        return [getattr(p, "name", str(p)) for p in self.phase_modules]

    def run(self) -> Dict[str, Any]:
        """
        Execute the assessment workflow across all registered phases.
        Handles:
        - Manual vs Automated execution flow
        - Pre-flight tool verification
        - AI scope compliance verification
        - Adaptive phase recommendations from YAML + RAG
        - Real-time audit recording in SessionManager & EvidenceManager
        - Graceful Ctrl+C interruption and state persistence
        """
        print("\n" + "=" * 70)
        print(f"  LAUNCHING: {self.agent_name.upper()}")
        print(f"  Target:    {self.scope.target}")
        print(f"  Mode:      {self.mode.upper()}")
        print(f"  Active:    {'AUTHORIZED' if self.scope.active_testing_allowed else 'READ-ONLY (Passive)'}")
        print("=" * 70 + "\n")

        # Ensure session exists
        if not self.session_mgr.active_session_name:
            session_name = f"{self.agent_type}_{int(time.time())}"
            self.session_mgr.create_session(
                name=session_name,
                scope=str(self.scope.in_scope),
                target=self.scope.target,
                target_type=self.agent_type
            )

        completed_phases: List[PhaseResult] = []
        phase_queue = list(self.phase_modules)
        current_idx = 0

        try:
            while current_idx < len(phase_queue):
                phase = phase_queue[current_idx]
                phase_name = getattr(phase, "name", f"phase_{current_idx}")
                req_tools = getattr(phase, "requires_tools", [])

                print(f"\n[*] Entering Phase [{current_idx + 1}/{len(phase_queue)}]: {phase_name.upper()}")

                # 1. Check required tools
                missing_tools = [t for t in req_tools if not self.tool_manager.is_tool_installed(t)]
                if missing_tools:
                    print(f"    [!] Missing required tool(s): {', '.join(missing_tools)}")
                    if self.mode == "manual":
                        install_choice = input(f"    Attempt installation via {self.tool_manager.os_info.package_manager}? [Y/n]: ").strip().lower()
                        if install_choice != "n":
                            for t in missing_tools:
                                print(f"    [*] Installing {t}...")
                                ok, msg = self.tool_manager.install_tool(t)
                                print(f"        {msg}")
                    else:
                        print("    [*] Automated mode: proceeding with fallback mechanisms if tool missing.")

                # 2. Gather AI unified context (Session + YAML + RAG + History + Tools)
                tools_info = [
                    {"name": t, "installed": self.tool_manager.is_tool_installed(t)}
                    for t in req_tools
                ]
                ai_context = self.chat_assistant.gather_ai_context(
                    agent_type=self.agent_type,
                    current_phase=phase_name,
                    session_state=self.session_mgr.state,
                    phase_results={p.phase_name: p.to_dict() for p in completed_phases},
                    available_tools=tools_info
                )

                # 3. AI Scope Compliance Gate (Part 2, Item 6)
                if self.scope.active_testing_allowed:
                    comp_check = self.chat_assistant.check_scope_compliance(
                        intended_command=f"Phase {phase_name} on {self.scope.target}",
                        scope=self.scope,
                        prohibited_items=self.scope.prohibited_tests + self.scope.prohibited_tools
                    )
                    if not comp_check.get("allowed", True):
                        print(f"    [!] SCOPE VIOLATION DETECTED BY AI: {comp_check.get('reason')}")
                        print(f"    [!] Halting phase '{phase_name}'.")
                        res = PhaseResult(phase_name=phase_name, status="blocked", error=comp_check.get("reason"))
                        completed_phases.append(res)
                        current_idx += 1
                        continue

                # 4. Manual Approval Gate
                if self.mode == "manual":
                    print(f"    Ready to execute phase: {phase_name}")
                    print(f"    Required Tools: {', '.join(req_tools) or 'None (Built-in)'}")
                    user_action = input("    Proceed with phase? [(Y)es / (N)o / (S)kip / (P)aste external]: ").strip().lower()
                    
                    if user_action in ("s", "skip"):
                        print(f"    [*] Skipped phase {phase_name}.")
                        completed_phases.append(PhaseResult(phase_name=phase_name, status="skipped"))
                        current_idx += 1
                        continue
                    elif user_action in ("p", "paste"):
                        print("    Enter/paste external terminal output below (finish with EOF or empty line):")
                        pasted_lines = []
                        while True:
                            try:
                                line = input()
                                if line.strip().upper() in ("EOF", "END") or not line.strip():
                                    break
                                pasted_lines.append(line)
                            except (KeyboardInterrupt, EOFError):
                                break
                        raw_paste = "\n".join(pasted_lines).strip()
                        interp = self.chat_assistant.interpret_pasted_output(raw_paste, ai_context)
                        print(f"\n    [+] AI Interpretation of pasted output:\n        {interp.get('summary')}")
                        self.session_mgr.add_note(f"Pasted in phase {phase_name}:\n{raw_paste}")
                        res = PhaseResult(
                            phase_name=phase_name,
                            status="success",
                            raw_output=raw_paste,
                            findings=interp
                        )
                        completed_phases.append(res)
                        current_idx += 1
                        continue
                    elif user_action == "n":
                        print("    [*] Phase execution declined by operator.")
                        completed_phases.append(PhaseResult(phase_name=phase_name, status="declined"))
                        current_idx += 1
                        continue

                # 5. Execute Phase Script
                try:
                    result: PhaseResult = phase.run(
                        session=self.session_mgr.state,
                        tools=self.tool_manager,
                        executor=self.executor,
                        scope=self.scope
                    )
                except Exception as exc:
                    print(f"    [!] Error during {phase_name}: {exc}")
                    # Novel situation handling via AI (Part 2, Item 3)
                    recovery = self.chat_assistant.suggest_recovery(str(exc), ai_context)
                    print(f"    [*] AI Diagnostic: {recovery.get('diagnostic')}")
                    print(f"    [*] AI Suggested Recovery: {recovery.get('ai_reasoning')}")
                    result = PhaseResult(phase_name=phase_name, status="failed", error=str(exc))

                completed_phases.append(result)

                # Record evidence for each command executed in this phase
                for cmd_entry in result.commands_run:
                    self.session_mgr.add_evidence(
                        command=cmd_entry.get("command", phase_name),
                        output=cmd_entry.get("stdout", "") or cmd_entry.get("output", ""),
                        step_id=phase_name,
                        status=cmd_entry.get("status", "completed")
                    )

                # Update session findings if phase produced any
                if result.findings:
                    self.session_mgr.add_ingested_findings(
                        finding_type=f"{self.agent_type}_{phase_name}",
                        findings=result.findings,
                        raw_source=phase_name
                    )

                # 6. Adaptive Planning via AI (Part 2, Item 1)
                remaining_candidates = [
                    getattr(p, "name", str(p))
                    for p in phase_queue[current_idx + 1:]
                ]
                if remaining_candidates:
                    ai_recommendation = self.chat_assistant.recommend_next_phase(
                        context=ai_context,
                        candidate_phases=remaining_candidates
                    )
                    recommended_next = ai_recommendation.get("next")
                    print(f"    [AI Consultant] Recommended next phase: {recommended_next}")
                    print(f"    [AI Consultant] Rationale: {ai_recommendation.get('reason')}")
                    if ai_recommendation.get("scenario_id"):
                        print(f"    [AI Consultant] Grounded in YAML scenario: {ai_recommendation.get('scenario_id')}")

                current_idx += 1

            # 7. End of assessment: AI Report Synthesis (Part 2, Item 7)
            print("\n" + "=" * 70)
            print("  ALL METHODOLOGY PHASES COMPLETED. SYNTHESIZING REPORT...")
            print("=" * 70)
            report_text = self.chat_assistant.synthesize_report(self.session_mgr.state)
            report_file = self.session_mgr.session_dir / "report.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"[+] Final Audit Report successfully compiled to: {report_file}\n")

            return {
                "status": "completed",
                "completed_phases": [p.to_dict() for p in completed_phases],
                "report_path": str(report_file)
            }

        except KeyboardInterrupt:
            print("\n\n[!] Execution paused by operator (Ctrl+C).")
            print("[*] Persisting session state to disk...")
            self.session_mgr.save_current_state()
            print(f"[+] State saved to session '{self.session_mgr.active_session_name}'.")
            print(f"    Resume anytime with: python main.py --resume {self.session_mgr.active_session_name}\n")
            return {
                "status": "interrupted",
                "session_name": self.session_mgr.active_session_name,
                "completed_phases": [p.to_dict() for p in completed_phases]
            }


    def run_autonomous(
        self,
        registry,
        completion_criteria=None,
        evidence_store=None,
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Execute an autonomous, hypothesis-driven security assessment.

        Enforces:
        - AI proposes structured actions via ActionRegistry (never raw shell).
        - 10-gate safety pipeline executed before every action.
        - Observations, hypotheses, validations, and findings recorded in EvidenceStore.
        - Framework-verified completion (AI cannot declare completion prematurely).
        - Atomic state persistence on each step and on Ctrl+C.
        """
        from core.action_contract import ActionRegistry, AIActionProposal
        from core.investigation import EvidenceStore, Observation, Hypothesis, Validation, Finding
        from core.ai_planner import AIPlanner

        print("\n" + "=" * 70)
        print(f"  LAUNCHING AUTONOMOUS MODE: {self.agent_name.upper()}")
        print(f"  Target:    {self.scope.target}")
        print(f"  Max Steps: {max_steps}")
        print(f"  Active Net Allowed: {getattr(self.scope, 'active_network_allowed', False)}")
        print(f"  State Changes Allowed: {getattr(self.scope, 'state_changes_allowed', False)}")
        print("=" * 70 + "\n")

        # Ensure session exists
        if not self.session_mgr.active_session_name:
            session_name = f"{self.agent_type}_autohunt_{int(time.time())}"
            self.session_mgr.create_session(
                name=session_name,
                scope=str(self.scope.in_scope),
                target=self.scope.target,
                target_type=self.agent_type
            )

        # Initialize or load EvidenceStore
        if evidence_store is None:
            loaded_store = self.session_mgr.load_investigation_store()
            evidence_store = loaded_store if loaded_store is not None else EvidenceStore()

        planner = AIPlanner(
            registry=registry,
            chat_assistant=self.chat_assistant,
            scope=self.scope,
        )

        step_count = 0
        status = "running"

        try:
            while step_count < max_steps:
                step_count += 1
                print(f"\n[*] --- Step {step_count}/{max_steps} ---")

                # 1. Compute retry budget status from execution history
                execution_history = self.session_mgr.state.get("autonomous_execution_history", [])
                retry_counts: Dict[str, int] = {}
                for entry in execution_history:
                    act_id = entry.get("action_id")
                    if act_id:
                        retry_counts[act_id] = retry_counts.get(act_id, 0) + 1

                # 2. Ask AI planner for next action or completion
                try:
                    proposal: AIActionProposal = planner.decide_next(
                        session_state=self.session_mgr.state,
                        evidence_store=evidence_store,
                        completion_criteria=completion_criteria,
                        execution_history=execution_history,
                        retry_budget_status=retry_counts,
                    )
                except Exception as exc:
                    print(f"    [!] Planner error: {exc}")
                    # Record planner failure and try next step
                    self.session_mgr.append_autonomous_history({
                        "step": step_count,
                        "action_id": "planner_decision",
                        "status": "failed",
                        "rejection_reason": str(exc),
                        "timestamp": datetime_iso(),
                    })
                    continue

                print(f"    AI Decision: {proposal.decision.upper()}")
                print(f"    Action:      {proposal.action_id}")
                print(f"    Rationale:   {proposal.rationale}")
                print(f"    Confidence:  {proposal.confidence}")

                # 3. Handle Completion Proposal
                if proposal.decision == "propose_completion":
                    if completion_criteria is None:
                        print("    [+] No completion criteria registered. Accepting completion.")
                        status = "completed"
                        break

                    from agents.ctf.completion import verify_completion
                    can_complete, reason = verify_completion(
                        criteria=completion_criteria,
                        session_state=self.session_mgr.state,
                        evidence_store=evidence_store,
                    )

                    if can_complete:
                        print(f"    [+] Framework VERIFIED completion: {reason}")
                        self.session_mgr.append_autonomous_history({
                            "step": step_count,
                            "decision": "propose_completion",
                            "status": "success",
                            "reason": reason,
                            "timestamp": datetime_iso(),
                        })
                        status = "completed"
                        break
                    else:
                        print(f"    [!] Framework REJECTED completion: {reason}")
                        self.session_mgr.append_autonomous_history({
                            "step": step_count,
                            "decision": "propose_completion",
                            "status": "rejected",
                            "rejection_reason": reason,
                            "timestamp": datetime_iso(),
                        })
                        continue

                # 4. Handle Action Proposal through 10-gate safety pipeline
                exec_result = self.executor.execute_action(
                    proposal=proposal,
                    registry=registry,
                    scope=self.scope,
                    session_state=self.session_mgr.state,
                    mode=self.mode,
                )

                exec_status = exec_result.get("status", "unknown")
                print(f"    Execution Status: {exec_status.upper()}")

                history_entry = {
                    "step": step_count,
                    "action_id": proposal.action_id,
                    "action_identity": exec_result.get("action_identity"),
                    "status": exec_status,
                    "rejection_reason": exec_result.get("rejection_reason"),
                    "rejection_category": exec_result.get("rejection_category"),
                    "results": exec_result.get("results", []),
                    "timestamp": datetime_iso(),
                }
                self.session_mgr.append_autonomous_history(history_entry)

                # 5. On successful execution: Record Observation in EvidenceStore
                if exec_status == "success":
                    for res in exec_result.get("results", []):
                        argv = res.get("argv", [])
                        stdout = res.get("stdout", "")
                        stderr = res.get("stderr", "")
                        returncode = res.get("returncode", 0)

                        obs = Observation(
                            action_id=proposal.action_id or "unknown",
                            argv=argv,
                            raw_stdout=stdout,
                            raw_stderr=stderr,
                            exit_code=returncode,
                        )
                        evidence_store.add_observation(obs)

                # 6. Save investigation store to disk atomically
                self.session_mgr.save_investigation_store(evidence_store)

            if step_count >= max_steps and status != "completed":
                print(f"\n[!] Reached max_steps limit ({max_steps}). Assessment stopped.")
                status = "max_steps_reached"

            # Synthesize final report
            print("\n" + "=" * 70)
            print("  AUTONOMOUS ASSESSMENT FINISHED. GENERATING AUDIT REPORT...")
            print("=" * 70)
            report_text = self.chat_assistant.synthesize_report(self.session_mgr.state)
            report_file = self.session_mgr.session_dir / "report.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"[+] Final Audit Report compiled to: {report_file}\n")

            return {
                "status": status,
                "steps_run": step_count,
                "findings_count": len(evidence_store.findings),
                "observations_count": len(evidence_store.observations),
                "session_name": self.session_mgr.active_session_name,
                "report_path": str(report_file),
            }

        except KeyboardInterrupt:
            print("\n\n[!] Autonomous execution interrupted by operator (Ctrl+C).")
            print("[*] Persisting session state and evidence store to disk...")
            self.session_mgr.save_current_state()
            self.session_mgr.save_investigation_store(evidence_store)
            print(f"[+] State saved to session '{self.session_mgr.active_session_name}'.")
            return {
                "status": "interrupted",
                "steps_run": step_count,
                "session_name": self.session_mgr.active_session_name,
                "findings_count": len(evidence_store.findings),
            }
