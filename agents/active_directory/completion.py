"""
agents/active_directory/completion.py

Completion criteria verification for ActiveDirectoryAgent.
Framework-enforced completion gate: the AI proposes completion; the framework
determines whether required AD enumeration, roasting phases, and finding thresholds are satisfied.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AdCompletionCriteria:
    required_phases: List[str] = field(default_factory=list)
    evidence_required_for_phases: List[str] = field(default_factory=list)
    minimum_findings: int = 1
    allow_unresolved_errors: bool = False


AD_COMPLETION = AdCompletionCriteria(
    required_phases=["ad_enumeration", "kerberoast"],
    evidence_required_for_phases=["ad_enumeration"],
    minimum_findings=1,
    allow_unresolved_errors=False,
)


def verify_completion(
    criteria: AdCompletionCriteria,
    session_state: Dict[str, Any],
    evidence_store,
) -> Tuple[bool, str]:
    """
    Framework-enforced completion gate for Active Directory assessments.
    """
    execution_history = session_state.get("autonomous_execution_history", [])
    completed_actions = {
        entry.get("action_id")
        for entry in execution_history
        if entry.get("status") == "success"
    }

    action_to_phase = {
        "dc_discovery": "ad_enumeration",
        "ad_enum": "ad_enumeration",
        "kerberoast": "kerberoast",
        "asrep_roast": "asrep_roast",
        "dcsync_audit": "dcsync",
        "acl_audit": "ad_enumeration",
        "trust_audit": "ad_enumeration",
        "gpo_audit": "ad_enumeration",
        "tier0_impact": "impact",
        "ad_report": "impact",
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

    confirmed_findings = evidence_store.proven_finding_count  # proven only -- leads/milestones don't count
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

