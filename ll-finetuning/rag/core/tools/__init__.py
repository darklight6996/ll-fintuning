"""
core/tools/__init__.py

Offensive Tool Execution Engine — Phase 11

Exposes execution machinery, adapters, and registration helpers.
"""

from __future__ import annotations

from typing import List, Optional

from core.tool_registry import ToolRegistry, get_tool_registry
from core.tools.adapters import (
    BaseToolAdapter,
    FfufAdapter,
    GobusterAdapter,
    HydraAdapter,
    NmapAdapter,
    NucleiAdapter,
    SqlmapAdapter,
)
from core.tools.executor import ExecutionResult, ToolExecutor


def register_offensive_tools(
    registry: Optional[ToolRegistry] = None,
    executor: Optional[ToolExecutor] = None,
) -> List[str]:
    """
    Register all Phase 11 offensive security adapters into the tool registry.
    Returns list of newly registered tool names.
    """
    target_registry = registry if registry is not None else get_tool_registry()
    shared_executor = executor or ToolExecutor()

    adapters: List[BaseToolAdapter] = [
        NmapAdapter(executor=shared_executor),
        NucleiAdapter(executor=shared_executor),
        FfufAdapter(executor=shared_executor),
        GobusterAdapter(executor=shared_executor),
        SqlmapAdapter(executor=shared_executor),
        HydraAdapter(executor=shared_executor),
    ]

    registered_names: List[str] = []
    for adapter in adapters:
        # If tool not already registered, register it
        if adapter.name not in target_registry:
            target_registry.register_function(
                name=adapter.name,
                description=adapter.description,
                parameters=adapter.get_parameters(),
                handler=adapter.run,
                category=adapter.category,
                safe_preview=adapter.safe_preview,
            )
            registered_names.append(adapter.name)

    return registered_names


__all__ = [
    "ToolExecutor",
    "ExecutionResult",
    "BaseToolAdapter",
    "NmapAdapter",
    "NucleiAdapter",
    "FfufAdapter",
    "GobusterAdapter",
    "SqlmapAdapter",
    "HydraAdapter",
    "register_offensive_tools",
]
