"""
core/builders/prompt_builder.py

Alias redirecting to the refactored core/prompts subsystem.
Maintained for 100% backward compatibility with existing imports.
"""

from core.prompts.prompt_builder import (
    PromptBuilder,
    DEFAULT_SYSTEM_PERSONA
)
from core.prompts.budget import PromptBudget
from core.prompts.history_builder import HistoryBuilder
from core.prompts.intel_builder import IntelBuilder
from core.prompts.knowledge_builder import KnowledgeBuilder
from core.prompts.scenario_builder import ScenarioBuilder
from core.prompts.templates import PromptTemplate, ModelFamily

__all__ = [
    "PromptBuilder",
    "DEFAULT_SYSTEM_PERSONA",
    "PromptBudget",
    "HistoryBuilder",
    "IntelBuilder",
    "KnowledgeBuilder",
    "ScenarioBuilder",
    "PromptTemplate",
    "ModelFamily",
]
