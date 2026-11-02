"""
agents/ctf/completion.py

Completion criteria verification for the CTF agent.

The framework enforces completion — the AI cannot declare the assessment
complete. Each check is deterministic and verifiable from the EvidenceStore
and session state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CompletionCriteria:
    """
    Formal completion requirements for an agent run.
    All fields are evaluated by verify_completion().
    """
    required_phases: List[str] = field(default_factory=list)
    """Phase names that MUST appear as completed in the session state."""

    evidence_required_for_phases: List[str] = field(default_factory=list)
    """Phases that must have at least one Observation in the EvidenceStore."""

    minimum_findings: int = 0
    """Minimum number of confirmed Findings required before completion."""

    allow_unresolved_errors: bool = False
    """If False, any phase with status='failed' blocks completion."""

    require_flag_pattern: Optional[str] = None
    """If set, at least one Finding must contain this pattern in its description."""


# CTF-specific completion criteria
CTF_COMPLETION = CompletionCriteria(
    required_phases=["environment_recon", "suid_audit", "flag_capture"],
    evidence_required_for_phases=["environment_recon", "suid_audit"],
    minimum_findings=1,
    allow_unresolved_errors=False,
    require_flag_pattern=None,  # Flag content is target-specific
)


def verify_completion(
    criteria: CompletionCriteria,
    session_state: Dict[str, Any],
    evidence_store,  # EvidenceStore from core.investigation
) -> Tuple[bool, str]:
    """
    Framework-enforced completion gate.

    The AI proposes completion with decision='propose_completion'.
    This function accepts or rejects the proposal with a reason.

    Returns:
        (True, "Completion criteria satisfied.") if all checks pass.
        (False, "<reason>") listing the first unmet criterion.
    """
    execution_history = session_state.get("autonomous_execution_history", [])
    completed_actions = {
        entry.get("action_id")
        for entry in execution_history
        if entry.get("status") == "success"
    }

    # Map action_id to phases for phase coverage check
    action_to_phase = {
        "environment_recon": "environment_recon",
        "suid_audit": "suid_audit",
        "strings_analysis": "enumeration",
        "find_writable": "enumeration",
        "kernel_exploit_check": "enumeration",
        "process_list": "enumeration",
        "cron_job_audit": "enumeration",
        "path_hijack_validation": "privesc",
        "flag_search": "flag_capture",
        "read_file": "flag_capture",
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

    # Check evidence requirements — observations must exist for these phases
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

    # Check minimum findings
    confirmed_findings = len(evidence_store.findings)
    if confirmed_findings < criteria.minimum_findings:
        return False, (
            f"Minimum {criteria.minimum_findings} confirmed Finding(s) required, "
            f"but only {confirmed_findings} found. "
            f"Validate at least one hypothesis before declaring completion."
        )

    # Check for unresolved errors
    if not criteria.allow_unresolved_errors:
        failed_actions = [
            entry.get("action_id", "unknown")
            for entry in execution_history
            if entry.get("status") == "failed" and not entry.get("recovered", False)
        ]
        if failed_actions:
            return False, (
                f"Unresolved failed action(s) block completion: {failed_actions}. "
                f"Either recover from failures or set allow_unresolved_errors=True."
            )

    # Check flag pattern if required
    if criteria.require_flag_pattern:
        pattern = criteria.require_flag_pattern.lower()
        flag_found = any(
            pattern in (f.description or "").lower() or pattern in (f.title or "").lower()
            for f in evidence_store.findings.values()
        )
        if not flag_found:
            return False, (
                f"No Finding contains the required flag pattern '{criteria.require_flag_pattern}'."
            )

    return True, "Completion criteria satisfied."
