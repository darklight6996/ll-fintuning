"""
core/chat_assistant.py

Conversational AI Security & Hacker Co-Pilot ("AI for Hackers").
Orchestrates:
1. Session intelligence extraction from live artifacts/evidence
2. Context building (RAG + YAML scenario tactics) via ContextBuilder
3. Deterministic intent classification via IntentRouter
4. Neural LLM reasoning via PromptBuilder and LLMService
5. Deterministic rule-based fallback via IntentRouter
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re
import os
import sys

CORE_DIR = Path(__file__).resolve().parent
RAG_DIR = CORE_DIR.parent
ROOT_DIR = RAG_DIR.parent

for p in (ROOT_DIR, RAG_DIR, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Services layer — singleton accessors
from core.services.llm_service import get_llm_service
from core.services.rag_service import get_rag_service
from core.services.yaml_service import get_yaml_service

# Routers, Builders, Intelligence, and Agent
from core.routers.intent_router import IntentRouter, Intent
from core.builders.context_builder import ContextBuilder, ContextPackage
from core.builders.prompt_builder import PromptBuilder
from core.intelligence import get_intelligence_extractor, IntelligenceExtractor
from core.agent import CyberAgent, AgentRequest, AgentResponse


class ChatAssistant:
    """
    Conversational Hacker Co-Pilot & Security Mentor.

    Orchestrates session intelligence, context building, intent classification,
    and generation across the services layer.
    """

    def __init__(
        self,
        intent_router: Optional[IntentRouter] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        intelligence_extractor: Optional[IntelligenceExtractor] = None,
        role: str = "default",
    ):
        # M2: "default" (chat, scope-compliance, report synthesis) or "planner"
        # (AIPlanner.decide_next()). Only changes behavior if
        # core.config.PLANNER_MODEL_NAME is explicitly set -- see
        # core/services/llm_service.py::get_llm_service() for the fallback.
        self.role = role
        # Stored as instance references so tests can inject mocks
        self._llm_svc = None
        self._rag_svc = None
        self._yaml_svc = None
        self.global_chat_history: List[Dict[str, str]] = []

        # Subcomponents
        self.intelligence_extractor = intelligence_extractor or get_intelligence_extractor()
        self.intent_router = intent_router or IntentRouter()
        self.context_builder = context_builder or ContextBuilder(
            intelligence_extractor=self.extract_session_intelligence
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self._agent = None

    # ------------------------------------------------------------------
    # Service accessors (cached after first resolution)
    # ------------------------------------------------------------------

    def _get_llm_router(self):
        """Return the LLMService singleton for this assistant's role (or None if unavailable)."""
        if self._llm_svc is None:
            self._llm_svc = get_llm_service(verbose=True, role=self.role) or False
        return self._llm_svc if self._llm_svc is not False else None

    def _get_rag_engine(self):
        """Return the RAGService singleton (or None if index is missing)."""
        if self._rag_svc is None:
            self._rag_svc = get_rag_service() or False
        return self._rag_svc if self._rag_svc is not False else None

    def _get_yaml_search(self):
        """Return the YAMLService singleton (or None if scenarios are missing)."""
        if self._yaml_svc is None:
            self._yaml_svc = get_yaml_service() or False
        return self._yaml_svc if self._yaml_svc is not False else None

    # ==================================================================
    # ARTIFACT & TERMINAL OUTPUT INTELLIGENCE EXTRACTION
    # ==================================================================
    def extract_session_intelligence(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deeply inspect notes, evidence, and completed steps to extract technical facts:
        - Current user ID, name, and groups
        - Discovered binaries and SUID executables (especially custom challenge binaries)
        - Strings and disassembled instructions (e.g. calls to 'fortune', '/bin/sh -p')
        - File permissions (e.g. /flag ownership and access modes)
        - Discovered network ports, services, and web banners

        Delegates to the decoupled, plugin-based IntelligenceExtractor.
        """
        if not session_state:
            return {
                "user": None,
                "groups": [],
                "kernel": None,
                "pwd": None,
                "suid_binaries": [],
                "custom_binaries": [],
                "strings_findings": [],
                "flag_permissions": None,
                "flag_content": None,
                "has_sudo": False,
                "open_ports": [],
                "network_services": [],
                "raw_notes_text": []
            }

        return self.intelligence_extractor.extract_dict(session_state)

    # ==================================================================
    # MAIN RESPOND METHOD (Delegated to CyberAgent)
    # ==================================================================
    def respond(
        self,
        user_message: str,
        session_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a reasoned, context-aware cybersecurity response.
        Works seamlessly both within an active session and as a standalone conversational advisor.
        Delegates execution to CyberAgent (Phase 8 Agent Framework) while keeping state synchronized.
        """
        state = session_state or {}
        clean_msg = user_message.strip()

        # Update chat history in session_state if active session
        if session_state is not None:
            chat_history = state.setdefault("chat_history", [])
            chat_history.append({"role": "operator", "message": clean_msg})
        else:
            chat_history = []

        session_id = state.get("name") or "default_session"

        # Instantiate or reuse CyberAgent configured with our components
        agent = self._agent or CyberAgent(
            session_id=session_id,
            router=self.intent_router,
            extractor=self.intelligence_extractor,
            prompt_builder=self.prompt_builder,
            llm=self._get_llm_router(),
        )

        req = AgentRequest(
            query=clean_msg,
            state=state,
            session_id=session_id
        )

        res = agent.run(req)
        self.last_response = res
        response = res.text

        # Record assistant reply into session_state history
        if session_state is not None:
            chat_history.append({"role": "assistant", "message": response})

        return response

    # ==================================================================
    # BACKWARD COMPATIBILITY DELEGATES
    # ==================================================================
    def _detect_operator_steering(self, query: str) -> bool:
        return self.intent_router.detect_operator_steering(query)

    def _detect_reset_or_different_challenge(self, query: str) -> bool:
        return self.intent_router.detect_reset_or_different_challenge(query)

    def _detect_recon_request(self, query: str) -> bool:
        return self.intent_router.detect_recon_request(query)

    def _execute_reset(self, state: Dict[str, Any]) -> str:
        return self.intent_router.execute_reset(state)

    def _build_llm_prompt(
        self,
        query: str,
        state: Dict[str, Any],
        intel: Dict[str, Any],
        rag_context: str,
        yaml_tactics: List[str],
        tokenizer: Optional[Any] = None
    ) -> str:
        return self.prompt_builder.build(
            query=query,
            state=state,
            intel=intel,
            rag_context=rag_context,
            yaml_tactics=yaml_tactics,
            tokenizer=tokenizer
        )

    def _reason_reset_challenge(self, query: str, state: Dict[str, Any]) -> str:
        return self.intent_router.reason_reset_challenge(query, state)

    def _reason_linux_recon_methodology(self, query: str, state: Dict[str, Any]) -> str:
        return self.intent_router.reason_linux_recon_methodology(query, state)

    def _reason_steering_response(self, query: str, state: Dict[str, Any], intel: Dict[str, Any]) -> str:
        return self.intent_router.reason_steering_response(query, state, intel)

    def _explain_pasted_output_and_binaries(self, query: str, state: Dict[str, Any], intel: Dict[str, Any]) -> str:
        return self.intent_router.explain_pasted_output_and_binaries(query, state, intel)

    def _reason_ctf_query(self, query: str, state: Dict[str, Any], intel: Dict[str, Any], rag_context: str, yaml_tactics: List[str]) -> str:
        return self.intent_router.reason_ctf_query(query, state, intel, rag_context, yaml_tactics)

    def _reason_ad_query(self, query: str, state: Dict[str, Any], rag_context: str, yaml_tactics: List[str]) -> str:
        return self.intent_router.reason_ad_query(query, state, rag_context, yaml_tactics)

    def _reason_web_query(self, query: str, state: Dict[str, Any], rag_context: str, yaml_tactics: List[str]) -> str:
        return self.intent_router.reason_web_query(query, state, rag_context, yaml_tactics)

    def _reason_general_query(self, query: str, state: Dict[str, Any], rag_context: str, yaml_tactics: List[str]) -> str:
        return self.intent_router.reason_general_query(query, state, rag_context, yaml_tactics)

    # ==================================================================
    # MULTI-AGENT FRAMEWORK AI INTEGRATION POINTS (Part 2 & Part 3)
    # ==================================================================

    def gather_ai_context(
        self,
        agent_type: str,
        current_phase: str,
        session_state: Dict[str, Any],
        phase_results: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Assemble the 6 mandatory context items required for reliable AI reasoning:
        1. Session State (target, scope, mode, findings, completed phases)
        2. YAML Scenarios (loaded & searched via yaml_search)
        3. RAG Corpus (top-k relevant chunks)
        4. Recent History (last 3 turns)
        5. Phase Results (structured output of completed phases)
        6. Available Tools (installed tools with version and path)
        """
        state = session_state or {}
        yaml_svc = self._get_yaml_search()
        yaml_scenarios = []
        if yaml_svc:
            try:
                yaml_scenarios = yaml_svc.search(query=f"{agent_type} {current_phase}", top_k=3)
            except Exception:
                yaml_scenarios = []

        rag_svc = self._get_rag_engine()
        rag_chunks = []
        rag_text = ""
        if rag_svc:
            try:
                rag_chunks = rag_svc.retrieve(query=f"{agent_type} {current_phase}", min_score=-2.5)
                rag_text = rag_svc.build_context(rag_chunks)
            except Exception:
                rag_text = ""

        all_hist = state.get("chat_history", [])
        recent_history = all_hist[-3:] if len(all_hist) > 3 else list(all_hist)
        results = phase_results if phase_results is not None else state.get("phase_results", {})
        tools = available_tools or []

        return {
            "session_state": state,
            "yaml_scenarios": yaml_scenarios,
            "rag_chunks": rag_chunks,
            "rag_text": rag_text,
            "recent_history": recent_history,
            "phase_results": results,
            "available_tools": tools,
            "agent_type": agent_type,
            "current_phase": current_phase
        }

    def recommend_next_phase(
        self,
        context: Dict[str, Any],
        candidate_phases: List[str]
    ) -> Dict[str, Any]:
        """
        Decision Point 1: Adaptive Planning / Phase Selection.
        Input: session state, current phase results, matching YAML scenarios.
        Output: recommended next phase + reasoning + confidence + scenario citation.
        """
        current_phase = context.get("current_phase", "recon")
        agent_type = context.get("agent_type", "general")
        yaml_scenarios = context.get("yaml_scenarios", [])
        phase_results = context.get("phase_results", {})
        findings = context.get("session_state", {}).get("targets", {})

        # Pick matching YAML scenario ID if available
        matched_yaml = yaml_scenarios[0].get("id") if yaml_scenarios else None

        # Build prompt for LLM
        prompt = (
            f"You are the Adaptive Planning AI inside the {agent_type} assessment agent.\n"
            f"Current phase: {current_phase}\n"
            f"Candidate next phases: {candidate_phases}\n"
            f"Current findings: {findings}\n"
            f"YAML Scenarios available: {[s.get('id', 'scenario') for s in yaml_scenarios]}\n"
            f"Recommend exactly ONE phase from the candidate list. State your reasoning and confidence."
        )

        llm = self._get_llm_router()
        llm_response = None
        if llm and hasattr(llm, "generate"):
            try:
                llm_response = llm.generate(prompt, max_new_tokens=25)
            except Exception:
                pass

        # Parse or default next phase
        selected_phase = candidate_phases[0] if candidate_phases else None
        if llm_response:
            for p in candidate_phases:
                if p.lower() in llm_response.lower():
                    selected_phase = p
                    break

        return {
            "next": selected_phase,
            "reason": f"Selected {selected_phase} based on {agent_type} workflow and YAML scenario {matched_yaml or 'baseline'}.",
            "confidence": 0.90 if matched_yaml else 0.80,
            "scenario_id": matched_yaml,
            "ai_raw": llm_response
        }

    def interpret_pasted_output(
        self,
        raw_text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decision Point 2: Output Interpretation (Paste Analysis).
        Input: raw text from operator terminal, current session state.
        Output: structured facts (ports, users, versions, vulnerabilities, indicators).
        """
        from core.ingestion import FileIngestionEngine
        engine = FileIngestionEngine()
        analysis = engine.analyze_terminal_text(raw_text)

        intel = self.extract_session_intelligence({"notes": [{"content": raw_text}]})
        entities = analysis.get("entities", {})

        # Merge extracted ports and services
        detected_ports = list(set(entities.get("ports", []) + intel.get("open_ports", [])))
        detected_services = list(set(analysis.get("detected_services", []) + intel.get("network_services", [])))

        return {
            "summary": analysis.get("summary", "Analyzed terminal output"),
            "ports": detected_ports,
            "services": detected_services,
            "domains": entities.get("domains", []),
            "vulnerabilities": entities.get("vulnerabilities", []),
            "users": intel.get("user") or entities.get("users", []),
            "analysis_details": analysis
        }

    def suggest_recovery(
        self,
        error_message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decision Point 3: Novel Situation Handling / Error Recovery.
        Input: error message, phase context, prior attempts.
        Output: diagnostic reasoning + alternative approach + safe commands to try.
        """
        phase = context.get("current_phase", "unknown")
        tools = [t.get("name") for t in context.get("available_tools", [])]

        prompt = (
            f"Phase '{phase}' encountered an error:\n{error_message}\n"
            f"Available system tools: {tools}\n"
            f"Diagnose the failure and suggest an alternative non-destructive approach or recovery command."
        )

        llm = self._get_llm_router()
        llm_resp = None
        if llm and hasattr(llm, "generate"):
            try:
                llm_resp = llm.generate(prompt)
            except Exception:
                pass

        return {
            "diagnostic": f"Execution failed in phase '{phase}': {error_message[:200]}",
            "alternative_approach": "Fallback to passive enumeration or alternate flag syntax.",
            "recovery_commands": ["netstat -tuln", "id", "whoami", "ip a"],
            "ai_reasoning": llm_resp or "Inspect logs, verify target reachability, and check tool flags."
        }

    def generate_custom_script(
        self,
        target_description: str,
        environment: str,
        constraints: str,
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Decision Point 4: Custom Tool/Script Generation.
        Input: target description, environment, constraints.
        Output: complete script as text + explanation + safety notes.
        NOTE: Operator MUST approve before saving or running; AI does NOT write to disk.
        """
        prompt = (
            f"Author a custom Python or Bash script for testing:\n"
            f"Target: {target_description}\n"
            f"Environment: {environment}\n"
            f"Constraints: {constraints}\n\n"
            f"Include code, safety notes, and audit justification. Never execute destructive operations."
        )

        llm = self._get_llm_router()
        script_text = ""
        if llm and hasattr(llm, "generate"):
            try:
                script_text = llm.generate(prompt)
            except Exception:
                pass

        if not script_text:
            script_text = (
                "```python\n"
                "# Custom Verification Script\n"
                "import requests, sys\n\n"
                "target = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost'\n"
                "try:\n"
                "    r = requests.get(target, timeout=5)\n"
                "    print(f'[+] Status: {r.status_code}, Server: {r.headers.get(\"Server\", \"Unknown\")}')\n"
                "except Exception as e:\n"
                "    print(f'[-] Error: {e}')\n"
                "```"
            )

        return {
            "script_code": script_text,
            "explanation": "Custom assessment script generated based on target environment parameters.",
            "safety_notes": "Read-only socket probe; strictly adheres to non-destructive constraints."
        }

    def reason_exploit_chain(
        self,
        findings: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decision Point 5: Exploit Path Reasoning.
        Input: list of findings, YAML scenarios, ATT&CK mappings.
        Output: ordered exploit chain with justification per step.
        """
        yaml_scenarios = context.get("yaml_scenarios", [])
        scenario_ids = [s.get("id") for s in yaml_scenarios if s.get("id")]

        chain = []
        for i, f in enumerate(findings):
            chain.append({
                "step": i + 1,
                "tactic": f.get("service", f.get("name", "Enumeration")),
                "justification": f"Finding '{f.get('title', 'Entity')}' enables progression.",
                "referenced_scenario": scenario_ids[i % len(scenario_ids)] if scenario_ids else "generic-chain"
            })

        return {
            "exploit_chain": chain,
            "summary": f"Identified {len(chain)}-step assessment attack path based on active evidence."
        }

    def check_scope_compliance(
        self,
        intended_command: str,
        scope: Any,
        prohibited_items: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Decision Point 6: Scope Compliance Check.
        Runs before every active phase to verify command parameters against scope boundaries.
        Input: intended command, scope object, prohibited items list.
        Output: {allowed: bool, reason: str, suggested_alternative: str | None}
        """
        if hasattr(scope, "is_command_allowed"):
            allowed, reason = scope.is_command_allowed(intended_command)
        else:
            allowed, reason = True, "No strict prohibitions registered."

        if not allowed:
            return {
                "allowed": False,
                "reason": reason,
                "suggested_alternative": f"curl -sI {getattr(scope, 'target', 'TARGET')}"
            }

        return {
            "allowed": True,
            "reason": "Command conforms to in-scope assets and authorized testing methods.",
            "suggested_alternative": None
        }

    def generate_raw(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        min_new_tokens: int = 40,
        system_prompt: Optional[str] = None,
        action_id_choices: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a raw text response from the LLM without any intent routing,
        RAG enrichment, or post-processing.

        Used by AIPlanner to obtain structured JSON proposals.
        Falls back to a safe default JSON if the LLM is unavailable.

        Args:
            prompt: The user-facing prompt text.
            max_new_tokens: Maximum tokens to generate.
            min_new_tokens: Floor below which eos/<|im_end|> is ineligible (see
                ModelRouter.generate()'s "found via live GPU smoke test" note).
                40 is enough room for the shortest real decision -- e.g.
                propose_completion's summary -- without being so large it forces
                padding past a legitimately short, valid completion.
            system_prompt: Optional system-level instruction override.

        Returns:
            Raw string output from the LLM.
        """
        llm = self._get_llm_router()

        if llm is None:
            # LLM unavailable — return a safe fallback that the planner can parse
            return json.dumps({
                "decision": "propose_action",
                "action_id": "environment_recon",
                "target": None,
                "parameters": {},
                "rationale": "LLM unavailable; defaulting to initial recon.",
                "confidence": 0.5,
            })

        try:
            # H3 fix: pass system_prompt through as the model's real system role instead
            # of concatenating it into the user turn. ModelRouter._format_prompt applies
            # its own default "pentest mentor" persona system message whenever no
            # system_prompt kwarg is supplied — concatenation meant the planner's strict
            # "return exactly one JSON object" contract was being demoted into user text
            # and overridden by that generic persona.
            # H4 fix: do_sample=False (greedy decoding) — generate_raw() exists solely to
            # produce parseable structured JSON for AIPlanner, where sampling variance is
            # a pure liability, not a feature.
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "system_prompt": system_prompt,
                "do_sample": False,
            }
            # M1: forwarded to ModelRouter.generate(), which only acts on it if
            # non-empty -- see core/constrained_decoding.py for scope.
            if action_id_choices:
                gen_kwargs["action_id_choices"] = action_id_choices

            # Use the LLM service directly — bypass intent routing
            if hasattr(llm, "generate"):
                return llm.generate(prompt, **gen_kwargs)
            elif hasattr(llm, "model_router") and hasattr(llm.model_router, "generate"):
                return llm.model_router.generate(prompt, **gen_kwargs)
            else:
                # Last resort: use the respond() path
                return self.respond(prompt)
        except Exception as exc:
            raise RuntimeError(
                f"generate_raw LLM failure: {type(exc).__name__}: {exc}"
            ) from exc

    def synthesize_report(
        self,
        session_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Decision Point 7: Report Synthesis.
        Input: session evidence, findings, YAML compliance mappings.
        Output: executive summary + technical breakdown + remediation priorities.
        """
        from core.report_generator import ReportGenerator
        generator = ReportGenerator()
        return generator.generate(session_state, context=context)

