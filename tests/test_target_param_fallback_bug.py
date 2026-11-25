"""
tests/test_target_param_fallback_bug.py

Regression test for the Gate 1 / "target" param fallback fix in
core/action_contract.py::ActionRegistry.validate_proposal().

Found live (pentest agent, Qwen2.5-7B-Instruct + qwen-qlora smoke test,
2026-11-14): the model proposed action_id + partial parameters but omitted
"target" from the nested "parameters" dict on 10/10 steps across 5 different
actions (host_discovery, service_scan, vuln_scan, web_enum, smb_enum), even
though AIPlanner._parse_proposal() already resolves a safe top-level
proposal.target (falling back to scope.target). Every one of those 10 steps
was rejected by Gate 1's pydantic validation before Gate 2's actual scope
check ever ran, stalling the run via MAX_STEPS_WITHOUT_PROGRESS with zero
observations ever recorded.
"""
from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition, AIActionProposal, ActionRegistry


class _FakeTargetParams(BaseModel):
    model_config = {"extra": "forbid"}
    target: str = Field(...)
    ports: str = Field("80,443")


class _FakeNoTargetParams(BaseModel):
    model_config = {"extra": "forbid"}
    ports: str = Field("80,443")


def _registry_with(defn: ActionDefinition) -> ActionRegistry:
    reg = ActionRegistry(agent_name="test")
    reg.register(defn)
    return reg


def test_missing_target_in_parameters_is_filled_from_proposal_target():
    """The exact bug: parameters omits 'target', but proposal.target (already
    resolved by AIPlanner against scope) is present -- validation must now
    succeed instead of rejecting on a missing-field error."""
    defn = ActionDefinition(
        action_id="service_scan",
        phase="recon",
        description="test",
        params_model=_FakeTargetParams,
        allowed_target_kinds=["ip", "domain"],
        command_builder="build_service_scan",
    )
    registry = _registry_with(defn)

    proposal = AIActionProposal(
        decision="propose_action",
        action_id="service_scan",
        target="127.0.0.1",
        parameters={"ports": "21,22,25,80"},  # "target" deliberately omitted
    )

    is_valid, typed_params, err = registry.validate_proposal(proposal)
    assert is_valid, f"Expected validation to succeed via fallback, got error: {err}"
    assert typed_params.target == "127.0.0.1"
    assert typed_params.ports == "21,22,25,80"


def test_explicit_target_in_parameters_is_never_overridden():
    """If the model DID supply a target inside parameters, the fallback must
    never clobber it, even if it differs from proposal.target."""
    defn = ActionDefinition(
        action_id="service_scan",
        phase="recon",
        description="test",
        params_model=_FakeTargetParams,
        allowed_target_kinds=["ip", "domain"],
        command_builder="build_service_scan",
    )
    registry = _registry_with(defn)

    proposal = AIActionProposal(
        decision="propose_action",
        action_id="service_scan",
        target="127.0.0.1",
        parameters={"target": "10.0.0.5", "ports": "80"},
    )

    is_valid, typed_params, err = registry.validate_proposal(proposal)
    assert is_valid
    assert typed_params.target == "10.0.0.5"  # explicit value wins, unchanged


def test_missing_target_with_no_fallback_still_fails_cleanly():
    """If proposal.target is ALSO empty (no fallback available at all),
    validation must still fail with a clear pydantic error -- this fix only
    fills a known value, it never invents one or weakens the gate."""
    defn = ActionDefinition(
        action_id="service_scan",
        phase="recon",
        description="test",
        params_model=_FakeTargetParams,
        allowed_target_kinds=["ip", "domain"],
        command_builder="build_service_scan",
    )
    registry = _registry_with(defn)

    proposal = AIActionProposal(
        decision="propose_action",
        action_id="service_scan",
        target=None,
        parameters={"ports": "80"},
    )

    is_valid, typed_params, err = registry.validate_proposal(proposal)
    assert not is_valid
    assert typed_params is None
    assert "Parameter validation failed" in err


def test_action_without_a_target_field_is_unaffected():
    """An action whose params_model has no 'target' field at all (e.g. a pure
    local-recon action) must behave exactly as before -- the fallback only
    ever touches a field the schema actually declares."""
    defn = ActionDefinition(
        action_id="local_check",
        phase="recon",
        description="test",
        params_model=_FakeNoTargetParams,
        allowed_target_kinds=["local"],
        command_builder="build_local_check",
    )
    registry = _registry_with(defn)

    proposal = AIActionProposal(
        decision="propose_action",
        action_id="local_check",
        target="127.0.0.1",
        parameters={"ports": "80"},
    )

    is_valid, typed_params, err = registry.validate_proposal(proposal)
    assert is_valid
    assert not hasattr(typed_params, "target")
