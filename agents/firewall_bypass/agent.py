"""
FirewallBypassAgent definition.
Workflow: fingerprint -> rule_enumeration -> bypass_techniques
Supports both deterministic sequential phase execution (run()) and autonomous hunter loop (run_autonomous()).
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.firewall_bypass.phases import (
    fingerprint,
    rule_enumeration,
    bypass_techniques,
)
from agents.firewall_bypass.actions import build_firewall_registry
from agents.firewall_bypass.completion import FIREWALL_BYPASS_COMPLETION


class FirewallBypassAgent(BaseAgent):
    agent_type = "firewall_bypass"
    agent_name = "Firewall & WAF Bypass Agent"
    description = "Specialized agent for Web Application Firewall detection, rule mapping, payload mutation, and evasion techniques against Cloudflare, AWS WAF, and ModSecurity."

    def _load_phases(self) -> None:
        self.phase_modules = [
            fingerprint,
            rule_enumeration,
            bypass_techniques,
        ]

    def get_action_registry(self):
        """Return the pre-configured ActionRegistry for Firewall Bypass autonomous mode."""
        return build_firewall_registry()

    def get_completion_criteria(self):
        """Return the completion criteria for Firewall Bypass autonomous mode."""
        return FIREWALL_BYPASS_COMPLETION

    def run_autonomous_default(self, max_steps: int = 50) -> Dict[str, Any]:
        """Convenience runner for autonomous mode using registered actions and criteria."""
        return self.run_autonomous(
            registry=self.get_action_registry(),
            completion_criteria=self.get_completion_criteria(),
            max_steps=max_steps,
        )
