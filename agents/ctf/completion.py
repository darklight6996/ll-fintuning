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

    phase_actions: Dict[str, List[str]] = field(default_factory=dict)
    """
    Reverse mapping of milestone phase name -> action_ids that satisfy it.
    FIX (live WSL smoke test, 2026-11-13): the AI planner's prompt showed
    `required_phases` and `Successful actions so far`, but never told the
    (weak, local 7B) model WHICH action_id would satisfy an unmet phase --
    e.g. "flag_capture" is only satisfied by running flag_search or read_file,
    but nothing in the prompt said so, and the model kept re-running
    suid_audit/find_writable instead of ever proposing flag_search. Exposing
    this mapping lets AIPlanner._format_completion_status() (generic, shared
    across every agent) name the specific actions still needed, without
    hardcoding any CTF-specific knowledge into the shared planner itself.
    """


# Milestone phase name -> the action(s) that satisfy it. NOTE: these milestone
# names are a CTF-domain completion policy, independent of each action's own
# ActionDefinition.phase label (used for Observation.phase / MED-7) -- e.g.
# suid_audit's ActionDefinition.phase is "environment_recon" (it belongs to
# the recon phase of execution), but it is ALSO, separately, its own required
# completion milestone here ("you must specifically have audited for SUID
# binaries before this assessment can be considered done"). Single source of
# truth for both verify_completion() and the planner-facing phase_actions
# mapping, so the two can no longer drift apart from each other.
ACTION_TO_MILESTONE_PHASE: Dict[str, str] = {
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

_MILESTONE_PHASE_TO_ACTIONS: Dict[str, List[str]] = {}
for _action_id, _phase in ACTION_TO_MILESTONE_PHASE.items():
    _MILESTONE_PHASE_TO_ACTIONS.setdefault(_phase, []).append(_action_id)


# CTF-specific completion criteria
CTF_COMPLETION = CompletionCriteria(
    required_phases=["environment_recon", "suid_audit", "flag_capture"],
    evidence_required_for_phases=["environment_recon", "suid_audit"],
    minimum_findings=1,
    allow_unresolved_errors=False,
    require_flag_pattern=None,  # Flag content is target-specific
    phase_actions=_MILESTONE_PHASE_TO_ACTIONS,
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

    # Map action_id to phases for phase coverage check (module-level constant,
    # also exposed via CompletionCriteria.phase_actions for the planner prompt --
    # see the comment on ACTION_TO_MILESTONE_PHASE above).
    action_to_phase = ACTION_TO_MILESTONE_PHASE

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
    phase_to_action_ids = _MILESTONE_PHASE_TO_ACTIONS

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
    confirmed_findings = evidence_store.proven_finding_count  # proven only -- leads/milestones don't count
    if confirmed_findings < criteria.minimum_findings:
        return False, (
            f"Minimum {criteria.minimum_findings} confirmed Finding(s) required, "
            f"but only {confirmed_findings} found. "
            f"Validate at least one hypothesis before declaring completion."
        )

    # Check for unresolved errors
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
