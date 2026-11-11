"""
core package: merges root core/ with ll-finetuning/rag/core/ so that a single
`core.X` import resolves regardless of which directory a given module physically
lives in. Most modules (config, model_router, chat_assistant, llm_service,
constrained_decoding, fact_parsing, session_manager, command_executor, ...) exist
ONLY under ll-finetuning/rag/core/ -- the __path__.append() below is what makes
`from core.config import ...` etc. find them at all, and must stay.

CLEANUP (2026-11-11): a handful of modules (scope, os_detector, tool_manager,
agent_router, ai_planner, action_contract, investigation) used to exist as
DIVERGED DUPLICATES in both directories. Since this package's own __path__ lists
its own directory (repo-root core/) before the appended rag/core one, every
ordinary `from core.X import Y` elsewhere in the codebase (agents/base_agent.py,
main.py, every test) was ALREADY resolving to the repo-root copy -- confirmed by
grepping every such import site. This file used to reach around that and pull
Scope/OSInfo/ToolManager/AgentRouter from the OTHER (rag/core) copies instead via
absolute `from rag.core.X import Y` imports, assigning them as core-package-level
attributes (core.Scope, etc.) -- but nothing in the codebase ever consumed them
that way (confirmed: no `from core import Scope`-style import exists anywhere),
so those lines only had the effect of loading a second, stale copy of each class
for no reason, and risked crashing this package's own import if the `rag` symlink
ever failed to resolve (as it does on at least one Linux environment this project
has been developed on). Switched to relative imports of the same repo-root files
every other caller already uses, and the now-fully-unreferenced duplicate files
under ll-finetuning/rag/core/ were deleted.
"""

from pathlib import Path

# Extend package search path to include ll-finetuning/rag/core -- required for
# every module that exists ONLY there (see docstring above).
_RAG_CORE = Path(__file__).resolve().parent.parent / "rag" / "core"
if _RAG_CORE.exists() and str(_RAG_CORE) not in __path__:
    __path__.append(str(_RAG_CORE))

from .scope import Scope
from .os_detector import OSInfo, detect_os, is_admin_or_root
from .tool_manager import ToolManager, TOOL_PACKAGE_MAP
from .agent_router import AgentRouter, AgentMetadata, AGENT_REGISTRY

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
