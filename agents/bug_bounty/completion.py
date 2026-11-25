"""
agents/bug_bounty/completion.py

Completion criteria verification for BugBountyAgent.
Framework-enforced completion gate: the AI proposes completion; the framework
determines whether required reconnaissance, enumeration, and finding thresholds are satisfied.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Single source of truth for action_id -> phase, mirroring the pattern already
# applied to agents/ctf/completion.py and agents/pentest/completion.py: one
# module-level dict instead of a copy living inline inside verify_completion(),
# plus a reverse mapping so AIPlanner._format_completion_status() can tell the
# planner exactly which action_id(s) satisfy an unmet required phase (the same
# planner-guidance gap fixed for CTF's flag_capture and pentest's phases).
ACTION_TO_PHASE: Dict[str, str] = {
    "subdomain_enum": "subdomain_enum",
    "port_scan": "recon",
    "endpoint_discovery": "endpoint_discovery",
    "browser_probe": "endpoint_discovery",
    "dir_brute": "endpoint_discovery",
    "param_fuzz": "param_fuzzing",
    "header_analysis": "recon",
    "recon_scan": "recon",
    "idor_probe": "vuln_validation",
    "sqli_probe": "vuln_validation",
    "sqli_deep_scan": "vuln_validation",
    "nuclei_scan": "vuln_validation",
    "dalfox_scan": "vuln_validation",
    "http_probe": "vuln_validation",
    "self_script": "vuln_validation",
    "xss_probe": "vuln_validation",
    "vuln_validation": "vuln_validation",
}

_PHASE_TO_ACTIONS: Dict[str, List[str]] = {}
for _action_id, _phase in ACTION_TO_PHASE.items():
    _PHASE_TO_ACTIONS.setdefault(_phase, []).append(_action_id)


@dataclass
class BugBountyCompletionCriteria:
    required_phases: List[str] = field(default_factory=list)
    evidence_required_for_phases: List[str] = field(default_factory=list)
    minimum_findings: int = 1
    allow_unresolved_errors: bool = False
    # Opt-in: reverse mapping (milestone phase name -> action_id(s) that satisfy
    # it), consumed by AIPlanner._format_completion_status() via getattr() if
    # present. Left as {} on any criteria instance that doesn't set it, which
    # is a byte-identical no-op for callers that predate this field.
    phase_actions: Dict[str, List[str]] = field(default_factory=dict)


BUG_BOUNTY_COMPLETION = BugBountyCompletionCriteria(
    required_phases=["recon", "subdomain_enum", "endpoint_discovery"],
    evidence_required_for_phases=["recon", "endpoint_discovery"],
    minimum_findings=1,
    allow_unresolved_errors=False,
    phase_actions=_PHASE_TO_ACTIONS,
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

    completed_phases = set()
    for action_id in completed_actions:
        phase = ACTION_TO_PHASE.get(action_id)
        if phase:
            completed_phases.add(phase)

    for required_phase in criteria.required_phases:
        if required_phase not in completed_phases:
            return False, (
                f"Required phase '{required_phase}' has not been completed. "
                f"Completed phases: {sorted(completed_phases)}"
            )

    for evidence_phase in criteria.evidence_required_for_phases:
        phase_action_ids = _PHASE_TO_ACTIONS.get(evidence_phase, [])
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
