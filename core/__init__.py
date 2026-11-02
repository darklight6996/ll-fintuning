"""
core package: merges root core with rag/core to support both legacy and new multi-agent imports.
"""

from pathlib import Path
import sys

# Extend package search path to include rag/core
_RAG_CORE = Path(__file__).resolve().parent.parent / "rag" / "core"
if _RAG_CORE.exists() and str(_RAG_CORE) not in __path__:
    __path__.append(str(_RAG_CORE))

from rag.core.scope import Scope
from rag.core.os_detector import OSInfo, detect_os, is_admin_or_root
from rag.core.tool_manager import ToolManager, TOOL_PACKAGE_MAP
from rag.core.agent_router import AgentRouter, AgentMetadata, AGENT_REGISTRY

__all__ = [
    "Scope",
    "OSInfo",
    "detect_os",
    "is_admin_or_root",
    "ToolManager",
    "TOOL_PACKAGE_MAP",
    "AgentRouter",
    "AgentMetadata",
    "AGENT_REGISTRY",
]
