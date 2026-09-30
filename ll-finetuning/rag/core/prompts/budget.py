"""
core/prompts/budget.py

PromptBudget dynamically calculates token and character limits across prompt sections:
- System instruction
- Live session intelligence
- Conversation history & rolling summaries
- RAG knowledge documents
- Scenario tactics
- Generation reservation
"""

from typing import Dict, Any


class PromptBudget:
    """
    Dynamically partitions total available model context across prompt components.
    Ensures that prompts never overflow maximum input sequence length while
    reallocating unused budget from small sections to larger ones.
    """

    # Average estimation: 1 token ≈ 4 characters in English technical text
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        total_tokens: int = 2048,
        reserve_generation: int = 250,
        reserve_system: int = 200,
        reserve_query: int = 150
    ):
        self.total_tokens = total_tokens
        self.reserve_generation = reserve_generation
        self.reserve_system = reserve_system
        self.reserve_query = reserve_query

    @property
    def available_context_tokens(self) -> int:
        """Remaining tokens available for dynamic context blocks."""
        avail = self.total_tokens - self.reserve_generation - self.reserve_system - self.reserve_query
        return max(200, avail)

    def tokens_to_chars(self, tokens: int) -> int:
        """Convert token count to approximate character limit."""
        return max(0, int(tokens * self.CHARS_PER_TOKEN))

    def chars_to_tokens(self, chars: int) -> int:
        """Convert character count to approximate token count."""
        return max(0, int(chars // self.CHARS_PER_TOKEN))

    def allocate(
        self,
        intel_chars: int = 0,
        history_chars: int = 0,
        rag_chars: int = 0,
        tactics_chars: int = 0
    ) -> Dict[str, int]:
        """
        Dynamically allocate character budgets across the 4 context components.
        Target base shares:
        - Intelligence: 25%
        - History: 20%
        - Scenario Tactics: 20%
        - Knowledge (RAG): 35%

        If a component needs fewer characters than its allocated base share,
        its surplus is dynamically shifted to components that exceed their base share.
        """
        total_avail_chars = self.tokens_to_chars(self.available_context_tokens)

        # Baseline partitions
        base_shares = {
            "intel": int(total_avail_chars * 0.25),
            "history": int(total_avail_chars * 0.20),
            "tactics": int(total_avail_chars * 0.20),
            "rag": int(total_avail_chars * 0.35),
        }

        requested = {
            "intel": intel_chars,
            "history": history_chars,
            "tactics": tactics_chars,
            "rag": rag_chars,
        }

        # Calculate surplus from components that need less than their base share
        surplus = 0
        final_budget = {}
        needy_components = []

        for key, base in base_shares.items():
            req = requested[key]
            if req <= base:
                final_budget[key] = max(100, req)
                surplus += (base - req)
            else:
                final_budget[key] = base
                needy_components.append(key)

        # Distribute surplus among needy components
        if needy_components and surplus > 0:
            extra_per_needy = surplus // len(needy_components)
            for key in needy_components:
                final_budget[key] += extra_per_needy

        return final_budget
