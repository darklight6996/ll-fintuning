"""
agents/mobile/completion.py

Completion criteria verification for MobileAppAgent.
Framework-enforced completion gate: the AI proposes completion; the framework
determines whether required static, permission, and secret discovery thresholds are satisfied.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MobileCompletionCriteria:
    required_phases: List[str] = field(default_factory=list)
    evidence_required_for_phases: List[str] = field(default_factory=list)
    minimum_findings: int = 1
    allow_unresolved_errors: bool = False


MOBILE_COMPLETION = MobileCompletionCriteria(
    required_phases=["apk_ipa_static", "data_storage"],
    evidence_required_for_phases=["apk_ipa_static"],
    minimum_findings=1,
    allow_unresolved_errors=False,
)


def verify_completion(
    criteria: MobileCompletionCriteria,
    session_state: Dict[str, Any],
    evidence_store,
) -> Tuple[bool, str]:
    """
    Framework-enforced completion gate for Mobile Application assessments.
    """
    execution_history = session_state.get("autonomous_execution_history", [])
    completed_actions = {
        entry.get("action_id")
        for entry in execution_history
        if entry.get("status") == "success"
    }

    action_to_phase = {
        "static_analysis": "apk_ipa_static",
        "decompile_app": "apk_ipa_static",
        "perm_audit": "apk_ipa_static",
        "hardcoded_secrets": "apk_ipa_static",
        "insecure_storage": "data_storage",
        "tls_pinning_check": "network_traffic",
        "auth_token_test": "auth_test",
        "dynamic_instrumentation": "dynamic",
        "webview_audit": "apk_ipa_static",
        "mobile_report": "data_storage",
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
        # BUG FIX (2026-11-25): a "recovered" key was checked here but nothing
        # anywhere in the codebase ever SET it, so a failed action permanently
        # blocked completion even after a later retry of the SAME action_id
        # succeeded. Compute recovery from the history itself instead of
        # depending on a flag nothing writes: a failed entry is recovered if
        # any LATER entry for the same action_id succeeded (an explicit
        # recovered=True on the entry itself still short-circuits too, for any
        # caller that does set it directly).
        failed_actions = []
        for idx, entry in enumerate(execution_history):
            if entry.get("status") != "failed" or entry.get("recovered", False):
                continue
            action_id = entry.get("action_id")
            later_success = any(
                e.get("action_id") == action_id and e.get("status") == "success"
                for e in execution_history[idx + 1:]
            )
            if not later_success:
                failed_actions.append(action_id or "unknown")
        if failed_actions:
            return False, (
                f"Unresolved failed action(s) block completion: {failed_actions}. "
                f"Recover from failures or set allow_unresolved_errors=True."
            )

    return True, "Completion criteria satisfied."

