"""
core/tool_registry.py

Tool Calling System — Phase 7

Provides a production-quality, type-safe tool registry for the AI agent:
  - JSON Schema tool definitions consumable by any OpenAI-compatible LLM
  - Decorator-based tool registration (@register_tool)
  - ToolResult contract: success/error, output, metadata
  - ToolDispatcher: validates, executes, and wraps results
  - Built-in security tools pre-registered (command preview, RAG search, YAML lookup)

Usage:
    from core.tool_registry import get_tool_registry, ToolResult

    registry = get_tool_registry()
    result = registry.dispatch("run_command_preview", {"command": "id"})
    print(result.output)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ===========================================================================
# DATA CONTRACTS
# ===========================================================================

@dataclass
class ToolParameter:
    """Describes a single named parameter for a tool."""
    name: str
    type: str                    # JSON Schema type: string, integer, boolean, array, object
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


@dataclass
class ToolDefinition:
    """Full JSON Schema–compatible tool definition."""
    name: str
    description: str
    parameters: List[ToolParameter]
    handler: Callable
    category: str = "general"    # "recon", "exploit", "rag", "memory", "general"
    safe_preview: bool = True    # True = always preview before executing

    def to_schema(self) -> Dict[str, Any]:
        """
        Export as OpenAI function-calling-compatible schema:
        {
          "type": "function",
          "function": {
            "name": ...,
            "description": ...,
            "parameters": {
              "type": "object",
              "properties": {...},
              "required": [...]
            }
          }
        }
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for p in self.parameters:
            prop: Dict[str, Any] = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolResult:
    """Execution result contract returned by all tool handlers."""
    tool_name: str
    success: bool
    output: str                              # Human-readable output for LLM prompt injection
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        if self.success:
            return f"[Tool: {self.tool_name}]\n{self.output}"
        return f"[Tool: {self.tool_name}] ERROR: {self.error}\n{self.output}"


# ===========================================================================
# TOOL REGISTRY
# ===========================================================================

class ToolRegistry:
    """
    Central registry mapping tool names to ToolDefinition objects.
    Thread-safe for read operations (register once at startup).
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, definition: ToolDefinition) -> None:
        """Register a ToolDefinition. Raises ValueError on name collision."""
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' is already registered.")
        self._tools[definition.name] = definition

    def register_function(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        handler: Callable,
        category: str = "general",
        safe_preview: bool = True,
    ) -> ToolDefinition:
        """Convenience method to register a handler directly."""
        defn = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            category=category,
            safe_preview=safe_preview,
        )
        self.register(defn)
        return defn

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Validate arguments and execute the named tool.

        Parameters
        ----------
        tool_name : str
            Registered tool name.
        arguments : dict
            Key/value arguments from the LLM tool call.

        Returns
        -------
        ToolResult
            Always returns a ToolResult — never raises.
        """
        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unknown tool '{tool_name}'. Available: {list(self._tools.keys())}",
            )

        defn = self._tools[tool_name]

        # Validate required parameters
        for param in defn.parameters:
            if param.required and param.name not in arguments:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output="",
                    error=f"Missing required parameter '{param.name}' for tool '{tool_name}'.",
                )

        # Apply defaults for optional params
        for param in defn.parameters:
            if not param.required and param.name not in arguments and param.default is not None:
                arguments[param.name] = param.default

        # Execute
        try:
            result = defn.handler(**arguments)
            if isinstance(result, ToolResult):
                return result
            # Auto-wrap plain string returns
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=str(result),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"{type(exc).__name__}: {exc}",
                metadata={"traceback": traceback.format_exc()},
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_schema(self) -> List[Dict[str, Any]]:
        """Return all tool schemas in OpenAI function-calling format."""
        return [defn.to_schema() for defn in self._tools.values()]

    def get_schema_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Return schemas filtered by category."""
        return [
            defn.to_schema()
            for defn in self._tools.values()
            if defn.category == category
        ]

    def list_tools(self) -> List[str]:
        """Return registered tool names."""
        return list(self._tools.keys())

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# ===========================================================================
# BUILT-IN SECURITY TOOLS
# ===========================================================================

def _tool_rag_search(query: str, top_k: int = 3) -> ToolResult:
    """
    Perform semantic RAG search over the cybersecurity knowledge base.
    Returns the most relevant document chunks for the given query.
    """
    try:
        from core.services.rag_service import get_rag_service
        svc = get_rag_service()
        context = svc.build_context(query, top_k=top_k)
        if not context:
            return ToolResult(
                tool_name="rag_search",
                success=True,
                output="No relevant documents found for this query.",
                metadata={"query": query, "top_k": top_k},
            )
        return ToolResult(
            tool_name="rag_search",
            success=True,
            output=context,
            metadata={"query": query, "top_k": top_k},
        )
    except Exception as exc:
        return ToolResult(
            tool_name="rag_search",
            success=False,
            output="",
            error=str(exc),
        )


def _tool_yaml_lookup(query: str, top_k: int = 3) -> ToolResult:
    """
    Search the YAML scenario playbook registry for attack/defense tactics.
    Returns the top matching scenario steps.
    """
    try:
        from core.services.yaml_service import get_yaml_service
        svc = get_yaml_service()
        results = svc.search(query, top_k=top_k)
        if not results:
            return ToolResult(
                tool_name="yaml_lookup",
                success=True,
                output="No matching scenarios found.",
                metadata={"query": query},
            )
        lines = []
        for r in results:
            name = r.get("name", r.get("title", "Unknown"))
            score = r.get("score", 0.0)
            lines.append(f"• {name} (score: {score:.3f})")
        return ToolResult(
            tool_name="yaml_lookup",
            success=True,
            output="\n".join(lines),
            metadata={"query": query, "count": len(results)},
        )
    except Exception as exc:
        return ToolResult(
            tool_name="yaml_lookup",
            success=False,
            output="",
            error=str(exc),
        )


def _tool_command_preview(command: str, target: str = "") -> ToolResult:
    """
    Preview a shell command without executing it.
    Returns the command with any variable substitutions resolved.
    Useful for LLM to confirm the command before live execution.
    """
    preview = f"```bash\n{command}\n```"
    note = (
        "\n⚠️  Preview mode — not executed. "
        "Operator must confirm before live execution."
    )
    return ToolResult(
        tool_name="command_preview",
        success=True,
        output=preview + note,
        metadata={"command": command, "target": target},
    )


def _tool_session_memory_read(session_id: str, limit: int = 6) -> ToolResult:
    """
    Read recent conversation turns from persistent session memory.
    Returns the last N user/assistant exchanges.
    """
    try:
        from core.services.memory_service import get_memory
        mem = get_memory(session_id)
        turns = mem.get_recent(limit=limit)
        if not turns:
            return ToolResult(
                tool_name="session_memory_read",
                success=True,
                output="No conversation history for this session.",
                metadata={"session_id": session_id},
            )
        lines = [f"[{role}]: {content}" for role, content in turns]
        return ToolResult(
            tool_name="session_memory_read",
            success=True,
            output="\n".join(lines),
            metadata={"session_id": session_id, "turns": len(turns)},
        )
    except Exception as exc:
        return ToolResult(
            tool_name="session_memory_read",
            success=False,
            output="",
            error=str(exc),
        )


def _tool_intelligence_extract(terminal_output: str) -> ToolResult:
    """
    Extract structured intelligence (users, ports, kernel, SUID binaries, flags)
    from raw terminal output using the modular parser registry.
    """
    try:
        from core.intelligence.extractor import IntelligenceExtractor
        extractor = IntelligenceExtractor()

        # IntelligenceExtractor.extract() expects a session_state dict
        # with optional keys: notes, evidence, completed_steps.
        # We wrap the raw string as a single evidence entry.
        session_state = {
            "notes": [{"content": terminal_output}],
            "evidence": [],
            "completed_steps": [],
        }
        intel = extractor.extract(session_state)
        d = intel.to_dict()

        # Build human-readable summary — skip empty values
        lines = ["=== Extracted Intelligence ==="]
        for key, val in d.items():
            if val:
                lines.append(f"{key}: {val}")
        if len(lines) == 1:
            lines.append("No specific intelligence patterns detected.")

        return ToolResult(
            tool_name="intelligence_extract",
            success=True,
            output="\n".join(lines),
            metadata=d,
        )
    except Exception as exc:
        return ToolResult(
            tool_name="intelligence_extract",
            success=False,
            output="",
            error=str(exc),
        )


# ===========================================================================
# REGISTRY SINGLETON
# ===========================================================================

_REGISTRY: Optional[ToolRegistry] = None


def _build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register_function(
        name="rag_search",
        description=(
            "Semantic search over the cybersecurity RAG knowledge base. "
            "Use when you need technical reference documentation, CVE details, "
            "attack methodology steps, or compliance standards."
        ),
        parameters=[
            ToolParameter("query", "string", "The search query.", required=True),
            ToolParameter("top_k", "integer", "Number of results to return.", required=False, default=3),
        ],
        handler=_tool_rag_search,
        category="rag",
    )

    registry.register_function(
        name="yaml_lookup",
        description=(
            "Search the YAML scenario playbook registry for offensive/defensive tactics. "
            "Use when you need structured attack playbook steps, tool commands, or "
            "enumeration checklists for a specific technique."
        ),
        parameters=[
            ToolParameter("query", "string", "The tactic or technique to look up.", required=True),
            ToolParameter("top_k", "integer", "Number of scenarios to return.", required=False, default=3),
        ],
        handler=_tool_yaml_lookup,
        category="rag",
    )

    registry.register_function(
        name="command_preview",
        description=(
            "Preview a shell command before live execution. "
            "Always call this before suggesting any command to the operator "
            "so they can review the exact syntax."
        ),
        parameters=[
            ToolParameter("command", "string", "The shell command to preview.", required=True),
            ToolParameter("target", "string", "Optional target IP or hostname.", required=False, default=""),
        ],
        handler=_tool_command_preview,
        category="recon",
        safe_preview=True,
    )

    registry.register_function(
        name="session_memory_read",
        description=(
            "Read recent conversation history from persistent session memory. "
            "Use when context is needed about previous exchanges in this engagement."
        ),
        parameters=[
            ToolParameter("session_id", "string", "The unique session identifier.", required=True),
            ToolParameter("limit", "integer", "Number of recent turns to retrieve.", required=False, default=6),
        ],
        handler=_tool_session_memory_read,
        category="memory",
    )

    registry.register_function(
        name="intelligence_extract",
        description=(
            "Extract structured session intelligence from raw terminal output. "
            "Parses usernames, UIDs, open ports, kernel version, SUID binaries, "
            "service banners, flags, and filesystem paths."
        ),
        parameters=[
            ToolParameter(
                "terminal_output", "string",
                "Raw terminal output to analyze (nmap, id, uname, netstat, etc.).",
                required=True,
            ),
        ],
        handler=_tool_intelligence_extract,
        category="recon",
    )

    return registry


def get_tool_registry() -> ToolRegistry:
    """
    Return the process-wide tool registry singleton.
    Registers all built-in security tools on first call.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_default_registry()
    return _REGISTRY
