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

# Routers, Builders, and Intelligence
from core.routers.intent_router import IntentRouter, Intent
from core.builders.context_builder import ContextBuilder, ContextPackage
from core.builders.prompt_builder import PromptBuilder
from core.intelligence import get_intelligence_extractor, IntelligenceExtractor


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
        intelligence_extractor: Optional[IntelligenceExtractor] = None
    ):
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

    # ------------------------------------------------------------------
    # Service accessors (cached after first resolution)
    # ------------------------------------------------------------------

    def _get_llm_router(self):
        """Return the LLMService singleton (or None if unavailable)."""
        if self._llm_svc is None:
            self._llm_svc = get_llm_service(verbose=True) or False
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
    # MAIN RESPOND METHOD (Orchestration)
    # ==================================================================
    def respond(
        self,
        user_message: str,
        session_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a reasoned, context-aware cybersecurity response.
        Works seamlessly both within an active session and as a standalone conversational advisor.
        Utilizes local Neural LLM reasoning with grounded RAG context, falling back gracefully
        to deterministic rule-based modules if offline or uninitialized.
        """
        state = session_state or {}
        clean_msg = user_message.strip()

        # Update chat history — only persist to state if a real session dict was passed
        if session_state is not None:
            chat_history = state.setdefault("chat_history", [])
            chat_history.append({"role": "operator", "message": clean_msg})
        else:
            chat_history = []

        # 1. Build context package (live intel + RAG + YAML tactics)
        ctx = self.context_builder.build(clean_msg, state)

        # 2. Classify intent
        intent = self.intent_router.classify(clean_msg, state, ctx.intel)

        # 3. Primary reasoning: Neural LLM Generation
        response = None
        llm = self._get_llm_router()

        if llm:
            tokenizer = getattr(llm, "tokenizer", None)
            model_id = getattr(llm, 'model_name', 'unknown')
            print(f"[LLM] Using model: {model_id}")

            prompt = self.prompt_builder.build(clean_msg, state, context=ctx, tokenizer=tokenizer)
            try:
                raw_out = llm.generate(prompt, temperature=0.3)
                if raw_out and len(raw_out.strip()) > 10:
                    response = raw_out.strip()
                    print(f"[LLM] Response generated ({len(response)} chars).")
                else:
                    print("[LLM] Empty/short output — falling back to rules.")
            except Exception as e:
                print(f"[LLM] Generation error ({type(e).__name__}: {e}) — falling back to rules.")
                response = None
        else:
            print("[FALLBACK] LLM unavailable — using rule-based engine.")

        # 4. Fallback reasoning: Deterministic Rules Engine
        if not response:
            response = self.intent_router.dispatch_fallback(
                intent=intent,
                query=clean_msg,
                state=state,
                intel=ctx.intel,
                rag_context=ctx.rag_context,
                yaml_tactics=ctx.yaml_tactics
            )

        # Record assistant reply into history (only if we have a live list)
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
