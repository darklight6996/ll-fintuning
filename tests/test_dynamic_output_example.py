"""
tests/test_dynamic_output_example.py

Regression tests for the "stuck echoing the system prompt's example verbatim" bug
found via the live GPU smoke test on 2026-11-11: the OUTPUT FORMAT example in
AIPlanner's system prompt used to hard-code a literal action_id
("environment_recon") and rationale sentence. This 7B model reliably copied that
exact literal as its *answer* turn after turn instead of reasoning about current
state -- confirmed live: 5 consecutive REJECTED steps with byte-identical
action_id AND rationale text, even after the action's retry budget was already
exhausted and excluded from REGISTERED ACTIONS.

Fix: _build_prompt() now records the first currently-available (non-exhausted)
action as self._example_action_id / self._example_rationale, and
decide_next() rebuilds the system prompt's OUTPUT FORMAT example from that each
call via _render_system_prompt(), instead of reusing one frozen constant.
"""

import json

from core.action_contract import ActionDefinition, ActionRegistry
from core.ai_planner import AIPlanner, _render_system_prompt
from core.investigation import EvidenceStore
from core.scope import Scope
from pydantic import BaseModel


class _NoParams(BaseModel):
    model_config = {"extra": "forbid"}


def _registry(*, environment_recon_max_retries=1, include_suid_audit=True):
    reg = ActionRegistry(agent_name="test")
    reg.register(ActionDefinition(
        action_id="environment_recon", phase="recon", description="recon",
        params_model=_NoParams, command_builder="noop",
        max_retries=environment_recon_max_retries,
    ))
    if include_suid_audit:
        reg.register(ActionDefinition(
            action_id="suid_audit", phase="enum", description="suid audit",
            params_model=_NoParams, command_builder="noop", max_retries=2,
        ))
    return reg


def _planner(registry):
    scope = Scope(target="local", active_testing_allowed=True)
    return AIPlanner(registry=registry, chat_assistant=None, scope=scope)


def _example_json_from(system_prompt: str):
    idx = system_prompt.find("OUTPUT FORMAT")
    assert idx != -1
    brace = system_prompt.find("{", idx)
    # The example JSON now lives inside a ```action fenced block (workstream C
    # output convention), so there is trailing text (the closing fence) after the
    # object -- use raw_decode to parse just the object and ignore the rest.
    obj, _end = json.JSONDecoder().raw_decode(system_prompt[brace:])
    return obj


def test_example_action_id_shifts_once_first_action_is_exhausted():
    planner = _planner(_registry())
    # Nothing exhausted yet -> example should be the first catalog action.
    planner._build_prompt(
        session_state={}, evidence_store=EvidenceStore(), completion_criteria=None,
        execution_history=[], retry_budget_status={},
    )
    assert planner._example_action_id == "environment_recon"

    # Now environment_recon is exhausted (1/1) -> example must shift to the
    # remaining non-exhausted action, never the exhausted one.
    planner._build_prompt(
        session_state={}, evidence_store=EvidenceStore(), completion_criteria=None,
        execution_history=[], retry_budget_status={"environment_recon": 1},
    )
    assert planner._example_action_id == "suid_audit"


def test_render_system_prompt_example_matches_given_action_and_has_no_leftover_tokens():
    system_prompt = _render_system_prompt("suid_audit", "suid_audit is available.")
    assert "__EXAMPLE_DECISION__" not in system_prompt
    assert "__EXAMPLE_JSON__" not in system_prompt
    example = _example_json_from(system_prompt)
    assert example["decision"] == "propose_action"
    assert example["action_id"] == "suid_audit"
    assert example["rationale"] == "suid_audit is available."


def test_render_system_prompt_falls_back_to_propose_completion_when_nothing_available():
    system_prompt = _render_system_prompt(None, "nothing left")
    example = _example_json_from(system_prompt)
    assert example["decision"] == "propose_completion"
    assert example["action_id"] is None


def test_example_never_names_an_exhausted_action_even_when_it_is_the_only_one_registered():
    planner = _planner(_registry(include_suid_audit=False))
    planner._build_prompt(
        session_state={}, evidence_store=EvidenceStore(), completion_criteria=None,
        execution_history=[], retry_budget_status={"environment_recon": 1},
    )
    assert planner._example_action_id is None
    system_prompt = _render_system_prompt(planner._example_action_id, planner._example_rationale)
    example = _example_json_from(system_prompt)
    assert example["action_id"] != "environment_recon"
    assert example["decision"] == "propose_completion"
