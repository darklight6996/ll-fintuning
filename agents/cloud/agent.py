"""
CloudAgent definition.
Workflow: iam_audit -> resource_enum -> misconfig_validation
Supports both deterministic sequential phase execution (run()) and autonomous hunter loop (run_autonomous()).
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.cloud.phases import (
    iam_audit,
    resource_enum,
    misconfig_validation,
)
from agents.cloud.actions import build_cloud_registry
from agents.cloud.completion import CLOUD_COMPLETION


class CloudAgent(BaseAgent):
    agent_type = "cloud"
    agent_name = "Cloud Infrastructure Security Agent"
    description = "Specialized agent for AWS, Azure, and GCP security assessments: IAM permission analysis, cloud asset inventory, and security posture validation."

    def _load_phases(self) -> None:
        self.phase_modules = [
            iam_audit,
            resource_enum,
            misconfig_validation,
        ]

    def get_action_registry(self):
        """Return the pre-configured ActionRegistry for Cloud autonomous mode."""
        return build_cloud_registry()

    def get_completion_criteria(self):
        """Return the completion criteria for Cloud autonomous mode."""
        return CLOUD_COMPLETION

    def run_autonomous_default(self, max_steps: int = 50) -> Dict[str, Any]:
        """Convenience runner for autonomous mode using registered actions and criteria."""
        return self.run_autonomous(
            registry=self.get_action_registry(),
            completion_criteria=self.get_completion_criteria(),
            max_steps=max_steps,
        )
