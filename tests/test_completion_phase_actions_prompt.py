"""
tests/test_completion_phase_actions_prompt.py

Regression tests for the "never pursues flag_search" gap found via the live
WSL smoke test on 2026-11-13: the planner's prompt showed the required
completion phases and which actions had already succeeded, but never told the
(weak, local) model WHICH action_id would satisfy a phase it hadn't completed
yet. On the live 8-step run the model kept re-proposing suid_audit/
find_writable (already satisfied) and never once proposed flag_search or
read_file, so the "flag_capture" phase -- and therefore completion -- was
never reached, even with real findings in hand.

Two independent pieces, both covered here:
  1. agents/ctf/completion.py: CTF_COMPLETION.phase_actions is a reverse
     mapping (milestone phase -> action_ids) built from the SAME module-level
     dict verify_completion() itself now uses, so the two can't drift apart.
  2. core/ai_planner.py: AIPlanner._format_completion_status() surfaces, for
     every unmet required phase, exactly which action_id(s) would satisfy it
     -- and stays a no-op (unchanged old behavior) for any agent whose
     CompletionCriteria doesn't set phase_actions at all.
"""

from pydantic import BaseModel

from agents.ctf.completion import CTF_COMPLETION, CompletionCriteria, verify_completion
from core.action_contract import ActionRegistry
from core.ai_planner import AIPlanner
from core.investigation import EvidenceStore
from core.scope import Scope


class _NoParams(BaseModel):
    model_config = {"extra": "forbid"}


def _make_planner():
    scope = Scope(target="local", active_testing_allowed=True)
    registry = ActionRegistry(agent_name="test")
    return AIPlanner(registry=registry, chat_assistant=None, scope=scope)


def test_ctf_completion_phase_actions_matches_verify_completions_own_mapping():
    # flag_capture must be satisfied by exactly flag_search / read_file --
    # the same two actions verify_completion() itself checks internally.
    assert set(CTF_COMPLETION.phase_actions["flag_capture"]) == {"flag_search", "read_file"}
    assert CTF_COMPLETION.phase_actions["suid_audit"] == ["suid_audit"]
    assert "environment_recon" in CTF_COMPLETION.phase_actions["environment_recon"]


def test_format_completion_status_names_the_action_needed_for_an_unmet_phase():
    planner = _make_planner()
    execution_history = [
        {"action_id": "environment_recon", "status": "success"},
        {"action_id": "suid_audit", "status": "success"},
        # flag_capture (flag_search / read_file) never run.
    ]
    status = planner._format_completion_status(
        criteria=CTF_COMPLETION,
        evidence_store=EvidenceStore(),
        execution_history=execution_history,
    )
    assert "Unmet required phases" in status
    assert "flag_capture" in status
    assert "flag_search" in status
    assert "read_file" in status
    # The two already-satisfied phases must NOT be listed as unmet.
    assert "'environment_recon': try" not in status
    assert "'suid_audit': try" not in status


def test_format_completion_status_reports_all_satisfied_when_every_phase_is_done():
    planner = _make_planner()
    execution_history = [
        {"action_id": "environment_recon", "status": "success"},
        {"action_id": "suid_audit", "status": "success"},
        {"action_id": "flag_search", "status": "success"},
    ]
    status = planner._format_completion_status(
        criteria=CTF_COMPLETION,
        evidence_store=EvidenceStore(),
        execution_history=execution_history,
    )
    assert "All required phases are satisfied" in status
    assert "Unmet required phases" not in status


def test_format_completion_status_is_unchanged_for_criteria_without_phase_actions():
    """
    Any agent whose CompletionCriteria doesn't set phase_actions (the default,
    unless a completion module explicitly opts in like agents/ctf/completion.py
    now does) must see the exact same prompt section as before this fix --
    no new lines, no regression for the other ~8 agents sharing this planner.
    """
    planner = _make_planner()
    bare_criteria = CompletionCriteria(
        required_phases=["recon"],
        evidence_required_for_phases=[],
        minimum_findings=0,
    )
    status = planner._format_completion_status(
        criteria=bare_criteria,
        evidence_store=EvidenceStore(),
        execution_history=[],
    )
    assert "Unmet required phases" not in status
    assert "All required phases are satisfied" not in status
    assert "Required phases: ['recon']" in status


def test_verify_completion_still_rejects_when_flag_capture_phase_is_unmet():
    evidence_store = EvidenceStore()
    session_state = {
        "autonomous_execution_history": [
            {"action_id": "environment_recon", "status": "success"},
            {"action_id": "suid_audit", "status": "success"},
        ]
    }
    can_complete, reason = verify_completion(CTF_COMPLETION, session_state, evidence_store)
    assert can_complete is False
    assert "flag_capture" in reason
