"""
tests/test_med6_med7.py

Regression tests for two MEDIUM-severity items from Claude outputs/AUDIT.md:

MED-6: AIActionProposal.compute_identity() used json.dumps() directly on the
params dict with no fallback. A non-JSON-serializable parameter value (bytes, a
set, a custom object, ...) would raise TypeError and crash the retry-budget
identity computation -- a far worse failure than a slightly less precise hash.
Fixed with a repr()-based fallback that can encode any Python object.

MED-7: Observation never recorded which phase (recon/enum/privesc/...) the
producing action belonged to, making it impossible to group or filter evidence
by phase later. Fixed by adding an optional `phase` field to Observation and
populating it in agents/base_agent.py::run_autonomous from the action's own
ActionDefinition.phase.
"""

from core.action_contract import AIActionProposal
from core.investigation import Observation


def test_compute_identity_succeeds_with_ordinary_json_serializable_params():
    proposal = AIActionProposal(
        decision="propose_action",
        action_id="port_scan",
        target="10.0.0.1",
        parameters={"ports": [80, 443], "timeout": 5},
    )
    identity = proposal.compute_identity()
    assert isinstance(identity, str)
    assert len(identity) == 16


def test_compute_identity_does_not_raise_on_non_json_serializable_param_value():
    """A set is not JSON-serializable -- json.dumps() alone would raise TypeError
    here. compute_identity() must fall back to a repr()-based encoding instead of
    crashing, since it backs the retry-budget gate."""
    proposal = AIActionProposal(
        decision="propose_action",
        action_id="weird_action",
        target="x",
        parameters={"weird_value": {1, 2, 3}},
    )
    identity = proposal.compute_identity()
    assert isinstance(identity, str)
    assert len(identity) == 16


def test_compute_identity_is_still_deterministic_for_identical_proposals():
    p1 = AIActionProposal(
        decision="propose_action", action_id="a", target="t", parameters={"x": 1},
    )
    p2 = AIActionProposal(
        decision="propose_action", action_id="a", target="t", parameters={"x": 1},
    )
    assert p1.compute_identity() == p2.compute_identity()


def test_observation_phase_field_defaults_to_none():
    obs = Observation(action_id="recon_action", argv=["id"], raw_stdout="uid=0")
    assert obs.phase is None


def test_observation_phase_field_can_be_set():
    obs = Observation(
        action_id="suid_audit", argv=["find", "/", "-perm", "-4000"],
        raw_stdout="", phase="privesc",
    )
    assert obs.phase == "privesc"
