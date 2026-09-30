"""
core/agent.py

Local Agent Framework — Phase 8

A lightweight, offline-first cybersecurity agent orchestrator that:
  1. Classifies query intent via IntentRouter
  2. Extracts session intelligence from accumulated evidence
  3. Assembles a token-budgeted prompt via PromptBuilder
  4. Routes to LLM (primary) or deterministic fallback (secondary)
  5. Persists conversation turns to SQLite memory
  6. Exposes tool_call() for structured tool dispatch
  7. Returns an AgentResponse contract consumed by UI/CLI layers

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │                     AgentLoop                       │
  │                                                     │
  │  Query ──► IntentRouter.classify()                  │
  │                │                                    │
  │       ┌────────┴──────────┐                         │
  │   RESET intent        all other intents             │
  │   (gate + wipe)       ┌──────────────┐              │
  │                       │ IntelExtract  │              │
  │                       │ PromptBuild   │              │
  │                       │ LLM.generate  │              │
  │                       │    │ (fail?)  │              │
  │                       │ Fallback      │              │
  │                       └──────────────┘              │
  │                                 │                   │
  │                       MemoryService.add_turn()      │
  │                              │                      │
  │                        AgentResponse                │
  └─────────────────────────────────────────────────────┘

Usage:
    from core.agent import CyberAgent, AgentRequest

    agent = CyberAgent(session_id="htb-forest")
    response = agent.run(AgentRequest(query="where do I start?", state=state))
    print(response.text)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ===========================================================================
# REQUEST / RESPONSE CONTRACTS
# ===========================================================================

@dataclass
class AgentRequest:
    """Input contract for a single agent turn."""
    query: str
    state: Dict[str, Any] = field(default_factory=dict)
    session_id: str = "default_session"
    # Override model family for this turn (e.g., "qwen", "llama")
    model_family: Optional[str] = None
    # Force deterministic fallback (skip LLM)
    force_fallback: bool = False


@dataclass
class AgentResponse:
    """Output contract for a single agent turn."""
    text: str
    intent: str
    source: str        # "llm" | "fallback" | "reset"
    latency_ms: float = 0.0
    tool_calls_made: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_llm(self) -> bool:
        return self.source == "llm"

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"


# ===========================================================================
# CYBER AGENT
# ===========================================================================

class CyberAgent:
    """
    Autonomous cybersecurity co-pilot agent.

    Responsibilities
    ----------------
    * Intent classification (stateless IntentRouter)
    * Session intelligence extraction (IntelligenceExtractor)
    * Prompt assembly (PromptBuilder)
    * LLM generation with deterministic fallback
    * Persistent conversation memory (MemoryService)
    * Tool dispatch (ToolRegistry)

    Parameters
    ----------
    session_id : str
        Unique identifier for the chat session (maps to SQLite record group).
    db_path : str, optional
        Override the SQLite database path from config.
    verbose : bool
        Print diagnostic info on first load.
    """

    def __init__(
        self,
        session_id: str = "default_session",
        db_path: Optional[str] = None,
        verbose: bool = False,
        router: Optional[Any] = None,
        extractor: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        llm: Optional[Any] = None,
        memory: Optional[Any] = None,
        tools: Optional[Any] = None,
    ):
        self.session_id = session_id
        self._verbose = verbose
        self._router = router
        self._extractor = extractor
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._memory = memory
        self._tools = tools
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Lazy loaders — all wrapped in try/except so import failures don't
    # crash the agent; fallback engine still works without heavy deps.
    # ------------------------------------------------------------------

    @property
    def router(self):
        if self._router is None:
            from core.routers.intent_router import IntentRouter
            self._router = IntentRouter()
        return self._router

    @property
    def extractor(self):
        if self._extractor is None:
            from core.intelligence.extractor import IntelligenceExtractor
            self._extractor = IntelligenceExtractor()
        return self._extractor

    @property
    def prompt_builder(self):
        if self._prompt_builder is None:
            from core.prompts.prompt_builder import PromptBuilder
            self._prompt_builder = PromptBuilder()
        return self._prompt_builder

    @property
    def llm(self):
        if self._llm is None:
            try:
                from core.services.llm_service import get_llm_service
                self._llm = get_llm_service(verbose=self._verbose)
            except Exception:
                self._llm = None
        return self._llm

    @property
    def memory(self):
        if self._memory is None:
            from core.services.memory_service import get_memory
            self._memory = get_memory(self.session_id, db_path=self._db_path)
        return self._memory

    @property
    def tools(self):
        if self._tools is None:
            from core.tool_registry import get_tool_registry
            self._tools = get_tool_registry()
        return self._tools

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentResponse:
        """
        Execute a single agent turn.

        Parameters
        ----------
        request : AgentRequest
            Encapsulates query, session state, and control flags.

        Returns
        -------
        AgentResponse
            Always returns a valid response — never raises.
        """
        t_start = time.monotonic()

        query = request.query.strip()
        state = request.state or {}

        # 1. Intent Classification
        intel_dict = self._extract_intel(state)
        intent = self.router.classify(query, state=state, intel=intel_dict)

        # 2. Handle control-gate intents (reset, steering, recon) deterministically
        if intent.value == "reset":
            text = self.router.dispatch_fallback(
                intent=intent,
                query=query,
                state=state,
                intel=intel_dict,
                rag_context="",
                yaml_tactics=[],
            )
            self._save_turn(query, text)
            return AgentResponse(
                text=text,
                intent=intent.value,
                source="reset",
                latency_ms=_elapsed_ms(t_start),
            )
        elif intent.value in ("steering", "recon"):
            text = self.router.dispatch_fallback(
                intent=intent,
                query=query,
                state=state,
                intel=intel_dict,
                rag_context="",
                yaml_tactics=[],
            )
            self._save_turn(query, text)
            return AgentResponse(
                text=text,
                intent=intent.value,
                source="fallback",
                latency_ms=_elapsed_ms(t_start),
            )

        # 3. Retrieve RAG context + YAML tactics
        rag_context, yaml_tactics = self._retrieve_context(query)

        # 4. Attempt LLM generation
        text = None
        source = "fallback"

        if not request.force_fallback and self.llm and self.llm.is_available:
            try:
                prompt = self.prompt_builder.build(
                    query=query,
                    state=state,
                    intel=intel_dict,
                    rag_context=rag_context,
                    yaml_tactics=yaml_tactics,
                    model_family=request.model_family,
                    tokenizer=getattr(self.llm, "tokenizer", None),
                )
                text = self.llm.generate(prompt)
                if text and text.strip():
                    source = "llm"
                else:
                    text = None
            except Exception as exc:
                if self._verbose:
                    print(f"[!] LLM generation failed: {exc}")
                text = None

        # 5. Fallback to deterministic engine
        if text is None:
            text = self.router.dispatch_fallback(
                intent=intent,
                query=query,
                state=state,
                intel=intel_dict,
                rag_context=rag_context,
                yaml_tactics=yaml_tactics,
            )
            source = "fallback"

        # 6. Persist turn to session memory
        self._save_turn(query, text)

        return AgentResponse(
            text=text,
            intent=intent.value,
            source=source,
            latency_ms=_elapsed_ms(t_start),
            metadata={
                "rag_chars": len(rag_context),
                "tactic_count": len(yaml_tactics),
            },
        )

    def tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """
        Execute a named tool and return a ToolResult.

        Parameters
        ----------
        tool_name : str
            Tool name (must be registered in ToolRegistry).
        arguments : dict
            Tool arguments.
        """
        from core.tool_registry import ToolResult
        return self.tools.dispatch(tool_name, arguments)

    def reset_session(self) -> None:
        """Wipe persistent session memory for this session_id."""
        if self._memory:
            self._memory.reset_session()

    def get_history(self, limit: int = 10) -> List[tuple]:
        """Return recent (role, content) turns from session memory."""
        return self.memory.get_recent(limit=limit)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_intel(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract intelligence dict from session state."""
        try:
            intel = self.extractor.extract(state)
            if hasattr(intel, "to_dict"):
                return intel.to_dict()
            elif isinstance(intel, dict):
                return intel
            return {}
        except Exception:
            return {}

    def _retrieve_context(self, query: str):
        """Retrieve RAG context string and YAML tactics list."""
        rag_context = ""
        yaml_tactics: List[str] = []

        try:
            from core.services.rag_service import get_rag_service
            svc = get_rag_service()
            rag_context = svc.build_context(query) or ""
        except Exception:
            pass

        try:
            from core.services.yaml_service import get_yaml_service
            svc = get_yaml_service()
            results = svc.search(query, top_k=3) or []
            for r in results:
                steps = r.get("steps") or r.get("commands") or []
                if isinstance(steps, list):
                    yaml_tactics.extend(str(s) for s in steps[:5])
                elif isinstance(steps, str):
                    yaml_tactics.append(steps)
        except Exception:
            pass

        return rag_context, yaml_tactics

    def _save_turn(self, query: str, response: str) -> None:
        """Persist (user, assistant) turn to session memory."""
        try:
            self.memory.add_turn(query, response)
        except Exception:
            pass


# ===========================================================================
# CONVENIENCE FUNCTION
# ===========================================================================

def _elapsed_ms(t_start: float) -> float:
    return round((time.monotonic() - t_start) * 1000, 1)


_AGENTS: Dict[str, CyberAgent] = {}


def get_agent(session_id: str = "default_session", verbose: bool = False) -> CyberAgent:
    """
    Return a per-session CyberAgent singleton.
    One agent per session_id per process.
    """
    global _AGENTS
    if session_id not in _AGENTS:
        _AGENTS[session_id] = CyberAgent(session_id=session_id, verbose=verbose)
    return _AGENTS[session_id]
