"""
ActiveDirectoryAgent definition.
Workflow: ad_enumeration -> kerberoast -> asrep_roast -> dcsync -> impact
Supports both deterministic sequential phase execution (run()) and autonomous hunter loop (run_autonomous()).
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.active_directory.phases import (
    ad_enumeration,
    kerberoast,
    asrep_roast,
    dcsync,
    impact,
)
from agents.active_directory.actions import build_ad_registry
from agents.active_directory.completion import AD_COMPLETION


class ActiveDirectoryAgent(BaseAgent):
    agent_type = "active_directory"
    agent_name = "Active Directory & Enterprise Identity Agent"
    description = "Specialized agent for Active Directory enterprise environments: Domain Controller discovery, Kerberoasting, AS-REP roasting, DCSync rights, and Tier-0 trust auditing."

    def _load_phases(self) -> None:
        self.phase_modules = [
            ad_enumeration,
            kerberoast,
            asrep_roast,
            dcsync,
            impact,
        ]

    def get_action_registry(self):
        """Return the pre-configured ActionRegistry for Active Directory autonomous mode."""
        return build_ad_registry()

    def get_completion_criteria(self):
        """Return the completion criteria for Active Directory autonomous mode."""
        return AD_COMPLETION

    def run_autonomous_default(self, max_steps: int = 50) -> Dict[str, Any]:
        """Convenience runner for autonomous mode using registered actions and criteria."""
        return self.run_autonomous(
            registry=self.get_action_registry(),
            completion_criteria=self.get_completion_criteria(),
            max_steps=max_steps,
        )
