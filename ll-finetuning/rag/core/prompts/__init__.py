"""
core/prompts package

Comprehensive prompt engineering and token budgeting subsystem:
- PromptBudget: Dynamic context token partitioning
- HistoryBuilder: Rolling conversation summaries and semantic deduplication
- IntelBuilder: Sandboxed session evidence serialization
- KnowledgeBuilder: Deduplicated RAG document context compilation
- ScenarioBuilder: YAML attack playbook step formatting
- PromptTemplate: Multi-model token framing (Qwen ChatML, Llama 3, Mistral, Gemma, DeepSeek)
- PromptBuilder: Unified generation prompt coordinator
- PromptManager: Backward compatibility wrapper
"""

from core.prompts.budget import PromptBudget
from core.prompts.history_builder import HistoryBuilder
from core.prompts.intel_builder import IntelBuilder
from core.prompts.knowledge_builder import KnowledgeBuilder
from core.prompts.scenario_builder import ScenarioBuilder
from core.prompts.templates import (
    PromptTemplate,
    ModelFamily,
    detect_model_family
)
from core.prompts.prompt_builder import (
    PromptBuilder,
    DEFAULT_SYSTEM_PERSONA
)
from core.prompts.manager import PromptManager

__all__ = [
    "PromptBudget",
    "HistoryBuilder",
    "IntelBuilder",
    "KnowledgeBuilder",
    "ScenarioBuilder",
    "PromptTemplate",
    "ModelFamily",
    "detect_model_family",
    "PromptBuilder",
    "DEFAULT_SYSTEM_PERSONA",
    "PromptManager",
]
