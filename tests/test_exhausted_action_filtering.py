"""
tests/test_exhausted_action_filtering.py

Regression tests for the "stuck re-proposing an exhausted action" bug found via
the live GPU smoke test on 2026-11-11: AIPlanner._build_prompt must exclude any
action whose retry budget is exhausted from the REGISTERED ACTIONS catalog shown
to the model, rather than merely instructing the model (via RULE 4 text) not to
propose it. Before this fix, the planner kept re-proposing "environment_recon"
every single step after its budget was exhausted on step 2, got rejected 8/8
times, and the run ended with 0 findings.
"""

from pydantic import BaseModel

from core.action_contract import ActionDefinition, ActionRegistry
from core.ai_planner import AIPlanner
from core.investigation import EvidenceStore
from core.scope import Scope


class _NoParams(BaseModel):
    """Empty params model shared by the fake actions registered below."""
    model_config = {"extra": "forbid"}


def _make_planner_with_actions():
    scope = Scope(target="local", active_testing_allowed=True)
    registry = ActionRegistry(agent_name="test")
    registry.register(ActionDefinition(
        action_id="environment_recon",
        phase="recon",
        description="recon",
        params_model=_NoParams,
        command_builder="noop",
        max_retries=1,
    ))
    registry.register(ActionDefinition(
        action_id="suid_audit",
        phase="enum",
        description="suid audit",
        params_model=_NoParams,
        command_builder="noop",
        max_retries=2,
    ))
    planner = AIPlanner(registry=registry, chat_assistant=None, scope=scope)
    return planner


def _catalog_section(prompt: str) -> str:
    start = prompt.find("=== REGISTERED ACTIONS")
    end = prompt.find("=== COMPLETION CRITERIA")
    assert start != -1 and end != -1, "prompt section headers changed -- update this test"
    return prompt[start:end]


def _retry_section(prompt: str) -> str:
    start = prompt.find("=== RETRY BUDGET STATUS")
    end = prompt.find("=== RECENT EXECUTION HISTORY")
    assert start != -1 and end != -1, "prompt section headers changed -- update this test"
    return prompt[start:end]


def test_exhausted_action_excluded_from_catalog_but_others_remain():
    planner = _make_planner_with_actions()
    prompt = planner._build_prompt(
        session_state={},
        evidence_store=EvidenceStore(),
        completion_criteria=None,
        execution_history=[],
        retry_budget_status={"environment_recon": 1},  # 1/1 -> exhausted
    )
    catalog = _catalog_section(prompt)
    assert "environment_recon" not in catalog
    assert "suid_audit" in catalog


def test_retry_status_tags_exhausted_action():
    planner = _make_planner_with_actions()
    prompt = planner._build_prompt(
        session_state={},
        evidence_store=EvidenceStore(),
        completion_criteria=None,
        execution_history=[],
        retry_budget_status={"environment_recon": 1},
    )
    retry_section = _retry_section(prompt)
    assert "environment_recon: 1/1 attempts used" in retry_section
    assert "[EXHAUSTED -- no longer offered in REGISTERED ACTIONS above]" in retry_section


def test_non_exhausted_action_not_tagged():
    planner = _make_planner_with_actions()
    prompt = planner._build_prompt(
        session_state={},
        evidence_store=EvidenceStore(),
        completion_criteria=None,
        execution_history=[],
        retry_budget_status={"environment_recon": 1, "suid_audit": 1},  # 1/2 -> not exhausted
    )
    catalog = _catalog_section(prompt)
    retry_section = _retry_section(prompt)
    assert "suid_audit" in catalog
    assert "suid_audit: 1/2 attempts used" in retry_section
    # Only environment_recon's line should carry the EXHAUSTED tag.
    suid_line = next(l for l in retry_section.splitlines() if "suid_audit" in l)
    assert "EXHAUSTED" not in suid_line


def test_empty_catalog_fallback_message_when_all_actions_exhausted():
    scope = Scope(target="local", active_testing_allowed=True)
    registry = ActionRegistry(agent_name="test")
    registry.register(ActionDefinition(
        action_id="environment_recon",
        phase="recon",
        description="recon",
        params_model=_NoParams,
        command_builder="noop",
        max_retries=1,
    ))
    planner = AIPlanner(registry=registry, chat_assistant=None, scope=scope)
    prompt = planner._build_prompt(
        session_state={},
        evidence_store=EvidenceStore(),
        completion_criteria=None,
        execution_history=[],
        retry_budget_status={"environment_recon": 1},
    )
    catalog = _catalog_section(prompt)
    assert "(none available" in catalog
    assert "propose_completion" in catalog
    assert "environment_recon" not in catalog
