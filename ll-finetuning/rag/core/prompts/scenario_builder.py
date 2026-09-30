"""
core/prompts/scenario_builder.py

ScenarioBuilder formats procedural YAML attack playbooks and assessment tactics
into actionable, numbered tactical steps adhering to token budgets.
"""

from typing import List, Dict, Any, Optional, Union


class ScenarioBuilder:
    """
    Constructs the SCENARIO TACTICS section of the prompt.
    Formats playbook steps and commands while eliminating duplicate tactics.
    """

    def build(
        self,
        tactics: Optional[Union[List[str], List[Dict[str, Any]]]] = None,
        max_chars: int = 1000
    ) -> str:
        """
        Build formatted tactics block within max_chars.
        """
        if not tactics:
            return "(no specific scenario tactics matched)"

        formatted_lines = []
        seen = set()

        for item in tactics:
            if isinstance(item, str):
                line = item.strip()
            elif isinstance(item, dict):
                phase = item.get("phase", "action").upper()
                name = item.get("name", "Step")
                cmd = item.get("command", "")
                line = f"[{phase}] {name}: {cmd}" if cmd else f"[{phase}] {name}"
            else:
                line = str(item).strip()

            if line and line not in seen:
                seen.add(line)
                formatted_lines.append(line)

        result = "\n".join(formatted_lines).strip()
        if not result:
            return "(no specific scenario tactics matched)"

        if len(result) > max_chars:
            result = result[:max_chars].rstrip() + "\n... [tactics truncated to budget]"

        return result
