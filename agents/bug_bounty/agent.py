"""
BugBountyAgent definition.
Workflow: scope -> recon -> subdomain_enum -> endpoint_discovery -> param_fuzzing -> vuln_validation
Supports both deterministic sequential phase execution (run()) and autonomous hunter loop (run_autonomous()).
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.bug_bounty.phases import (
    recon,
    subdomain_enum,
    endpoint_discovery,
    param_fuzzing,
    vuln_validation,
)
from agents.bug_bounty.actions import build_bug_bounty_registry
from agents.bug_bounty.completion import BUG_BOUNTY_COMPLETION


class BugBountyAgent(BaseAgent):
    agent_type = "bug_bounty"
    agent_name = "Bug Bounty Assessment Agent"
    description = "Specialized agent for web application discovery, subdomain mapping, API analysis, and vulnerability triage."

    def _load_phases(self) -> None:
        self.phase_modules = [
            recon,
            subdomain_enum,
            endpoint_discovery,
            param_fuzzing,
            vuln_validation,
        ]

    def get_action_registry(self):
        """Return the pre-configured ActionRegistry for Bug Bounty autonomous mode."""
        return build_bug_bounty_registry()

    def get_completion_criteria(self):
        """Return the completion criteria for Bug Bounty autonomous mode."""
        return BUG_BOUNTY_COMPLETION

    def run_autonomous_default(self, max_steps: int = 50) -> Dict[str, Any]:
        """Convenience runner for autonomous mode using registered actions and criteria."""
        return self.run_autonomous(
            registry=self.get_action_registry(),
            completion_criteria=self.get_completion_criteria(),
            max_steps=max_steps,
        )
