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
from agents.bug_bounty.single_host_mode import (
    build_single_host_bug_bounty_registry,
    SINGLE_HOST_BUG_BOUNTY_COMPLETION,
)


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
        """
        Return the ActionRegistry for Bug Bounty autonomous mode.

        This is scope-derived, not AI-decided: when Scope.single_host_only is
        True -- the operator has declared up front that this engagement is
        authorized against exactly one pinned host/URL instance, not a whole
        domain (e.g. a PortSwigger Web Security Academy lab, a single staging
        endpoint, one client-provided test URL) -- the two whole-domain
        actions (subdomain_enum, port_scan) are left out of the registry
        entirely, so the AI planner can never even propose them, and never
        gets to "judge" on its own whether enumerating subdomains is in scope.
        That judgment call is the framework's, made deterministically from an
        explicit Scope flag the operator set, which is the same posture as
        every other scope/safety gate in this codebase: the AI proposes,
        the framework decides.

        Default (single_host_only=False, unset on every existing Scope/
        session) is byte-identical to before this flag existed.
        """
        if getattr(self.scope, "single_host_only", False):
            return build_single_host_bug_bounty_registry(
                allow_scanners=getattr(self.scope, "allow_scanners", True)
            )
        return build_bug_bounty_registry()

    def get_completion_criteria(self):
        """Return the completion criteria matching get_action_registry()'s choice above."""
        if getattr(self.scope, "single_host_only", False):
            return SINGLE_HOST_BUG_BOUNTY_COMPLETION
        return BUG_BOUNTY_COMPLETION

    def run_autonomous_default(self, max_steps: int = 50) -> Dict[str, Any]:
        """Convenience runner for autonomous mode using registered actions and criteria."""
        return self.run_autonomous(
            registry=self.get_action_registry(),
            completion_criteria=self.get_completion_criteria(),
            max_steps=max_steps,
        )
