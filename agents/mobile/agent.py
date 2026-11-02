"""
MobileAppAgent definition.
Workflow: apk_ipa_static -> dynamic -> auth_test -> data_storage -> network_traffic
Supports both deterministic sequential phase execution (run()) and autonomous hunter loop (run_autonomous()).
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from agents.mobile.phases import (
    apk_ipa_static,
    dynamic,
    auth_test,
    data_storage,
    network_traffic,
)
from agents.mobile.actions import build_mobile_registry
from agents.mobile.completion import MOBILE_COMPLETION


class MobileAppAgent(BaseAgent):
    agent_type = "mobile"
    agent_name = "Mobile Application (iOS / Android) Security Agent"
    description = "Specialized agent for APK and IPA mobile application security: static decompilation, manifest auditing, hardcoded secrets, local data storage, and network interception."

    def _load_phases(self) -> None:
        self.phase_modules = [
            apk_ipa_static,
            dynamic,
            auth_test,
            data_storage,
            network_traffic,
        ]

    def get_action_registry(self):
        """Return the pre-configured ActionRegistry for Mobile autonomous mode."""
        return build_mobile_registry()

    def get_completion_criteria(self):
        """Return the completion criteria for Mobile autonomous mode."""
        return MOBILE_COMPLETION

    def run_autonomous_default(self, max_steps: int = 50) -> Dict[str, Any]:
        """Convenience runner for autonomous mode using registered actions and criteria."""
        return self.run_autonomous(
            registry=self.get_action_registry(),
            completion_criteria=self.get_completion_criteria(),
            max_steps=max_steps,
        )
