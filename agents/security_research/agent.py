"""
SecurityResearchAgent definition.
Workflow: cve_research -> poc_dev -> analysis -> writeup
Supports both deterministic sequential phase execution (run()) and autonomous hunter loop (run_autonomous()).
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.security_research.phases import (
    cve_research,
    poc_dev,
    analysis,
    writeup,
)
from agents.security_research.actions import build_research_registry
from agents.security_research.completion import SECURITY_RESEARCH_COMPLETION


class SecurityResearchAgent(BaseAgent):
    agent_type = "security_research"
    agent_name = "Vulnerability Research & PoC Development Agent"
    description = "Specialized agent for 0-day / N-day vulnerability analysis, patch diffing, root cause isolation, PoC harness creation, and technical writeups."

    def _load_phases(self) -> None:
        self.phase_modules = [
            cve_research,
            poc_dev,
            analysis,
            writeup,
        ]

    def get_action_registry(self):
        """Return the pre-configured ActionRegistry for Security Research autonomous mode."""
        return build_research_registry()

    def get_completion_criteria(self):
        """Return the completion criteria for Security Research autonomous mode."""
        return SECURITY_RESEARCH_COMPLETION

    def run_autonomous_default(self, max_steps: int = 50) -> Dict[str, Any]:
        """Convenience runner for autonomous mode using registered actions and criteria."""
        return self.run_autonomous(
            registry=self.get_action_registry(),
            completion_criteria=self.get_completion_criteria(),
            max_steps=max_steps,
        )
