"""
agents/bug_bounty/completion.py

Completion criteria verification for BugBountyAgent.
Framework-enforced completion gate: the AI proposes completion; the framework
determines whether required reconnaissance, enumeration, and finding thresholds are satisfied.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BugBountyCompletionCriteria:
    required_phases: List[str] = field(default_factory=list)
    evidence_required_for_phases: List[str] = field(default_factory=list)
    minimum_findings: int = 1
    allow_unresolved_errors: bool = False


BUG_BOUNTY_COMPLETION = BugBountyCompletionCriteria(
    required_phases=["recon", "subdomain_enum", "endpoint_discovery"],
    evidence_required_for_phases=["recon", "endpoint_discovery"],
    minimum_findings=1,
    allow_unresolved_errors=False,
)


def verify_completion(
    criteria: BugBountyCompletionCriteria,
    session_state: Dict[str, Any],
    evidence_store,
) -> Tuple[bool, str]:
    """
    Framework-enforced completion gate for Bug Bounty assessments.
    """
    execution_history = session_state.get("autonomous_execution_history", [])
    completed_actions = {
        entry.get("action_id")
        for entry in execution_history
        if entry.get("status") == "success"
    }

    action_to_phase = {
        "subdomain_enum": "subdomain_enum",
        "port_scan": "recon",
        "endpoint_discovery": "endpoint_discovery",
        "dir_brute": "endpoint_discovery",
        "param_fuzz": "param_fuzzing",
        "header_analysis": "recon",
        "idor_probe": "vuln_validation",
        "sqli_probe": "vuln_validation",
        "xss_probe": "vuln_validation",
        "vuln_validation": "vuln_validation",
    }

    completed_phases = set()
    for action_id in completed_actions:
        phase = action_to_phase.get(action_id)
        if phase:
            completed_phases.add(phase)

    for required_phase in criteria.required_phases:
        if required_phase not in completed_phases:
            return False, (
                f"Required phase '{required_phase}' has not been completed. "
                f"Completed phases: {sorted(completed_phases)}"
            )

    phase_to_action_ids = {}
    for action_id, phase in action_to_phase.items():
        phase_to_action_ids.setdefault(phase, []).append(action_id)

    for evidence_phase in criteria.evidence_required_for_phases:
        phase_action_ids = phase_to_action_ids.get(evidence_phase, [])
        has_observation = any(
            obs.action_id in phase_action_ids
            for obs in evidence_store.observations.values()
        )
        if not has_observation:
            return False, (
                f"Phase '{evidence_phase}' requires at least one recorded Observation "
                f"in the EvidenceStore, but none found."
            )

    confirmed_findings = len(evidence_store.findings)
    if confirmed_findings < criteria.minimum_findings:
        return False, (
            f"Minimum {criteria.minimum_findings} confirmed Finding(s) required, "
            f"but found {confirmed_findings}. "
            f"Validate at least one vulnerability before declaring completion."
        )

    if not criteria.allow_unresolved_errors:
        failed_actions = [
            entry.get("action_id", "unknown")
            for entry in execution_history
            if entry.get("status") == "failed" and not entry.get("recovered", False)
        ]
        if failed_actions:
            return False, (
                f"Unresolved failed action(s) block completion: {failed_actions}. "
                f"Recover from failures or set allow_unresolved_errors=True."
            )

    return True, "Completion criteria satisfied."

