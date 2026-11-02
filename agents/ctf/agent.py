"""
CTFAgent definition.
Workflow: environment_recon -> enumeration -> privesc -> flag_capture
Supports both deterministic phase execution (run()) and autonomous hunter loop (run_autonomous()).
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.ctf.phases import environment_recon, enumeration, privesc, flag_capture
from agents.ctf.actions import build_ctf_registry
from agents.ctf.completion import CTF_COMPLETION


class CTFAgent(BaseAgent):
    agent_type = "ctf"
    agent_name = "CTF & Wargame Autonomous Co-Pilot"
    description = "Specialized agent for CTF challenges, Linux shell wargames, privilege escalation puzzles, and flag extraction."

    def _load_phases(self) -> None:
        self.phase_modules = [
            environment_recon,
            enumeration,
            privesc,
            flag_capture
        ]

    def get_action_registry(self):
        """Return the pre-configured ActionRegistry for autonomous mode."""
        return build_ctf_registry()

    def get_completion_criteria(self):
        """Return the completion criteria for autonomous mode."""
        return CTF_COMPLETION

    def run_autonomous_default(self, max_steps: int = 50) -> Dict[str, Any]:
        """Convenience runner for autonomous mode using registered actions and criteria."""
        return self.run_autonomous(
            registry=self.get_action_registry(),
            completion_criteria=self.get_completion_criteria(),
            max_steps=max_steps
        )
