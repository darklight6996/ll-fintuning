"""
RedTeamAgent definition.
Workflow: scope -> recon -> initial_access -> privesc -> lateral -> impact
Supports both deterministic sequential phase execution (run()) and autonomous hunter loop (run_autonomous()).
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.red_team.phases import (
    recon,
    initial_access,
    privesc,
    lateral,
    impact,
)
from agents.red_team.actions import build_red_team_registry
from agents.red_team.completion import RED_TEAM_COMPLETION


class RedTeamAgent(BaseAgent):
    agent_type = "red_team"
    agent_name = "Red Team Adversary Emulation Agent"
    description = "Full MITRE ATT&CK adversary emulation: stealth recon, initial access, privilege escalation, defense evasion, lateral movement, and impact."

    def _load_phases(self) -> None:
        self.phase_modules = [
            recon,
            initial_access,
            privesc,
            lateral,
            impact,
        ]

    def get_action_registry(self):
        """Return the pre-configured ActionRegistry for Red Team autonomous mode."""
        return build_red_team_registry()

    def get_completion_criteria(self):
        """Return the completion criteria for Red Team autonomous mode."""
        return RED_TEAM_COMPLETION

    def run_autonomous_default(self, max_steps: int = 50) -> Dict[str, Any]:
        """Convenience runner for autonomous mode using registered actions and criteria."""
        return self.run_autonomous(
            registry=self.get_action_registry(),
            completion_criteria=self.get_completion_criteria(),
            max_steps=max_steps,
        )
