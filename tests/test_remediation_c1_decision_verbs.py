"""
tests/test_remediation_c1_decision_verbs.py

Regression tests for the full C1 extension: the AI planner's three evidence-chain
decision verbs (form_hypothesis, record_validation, record_finding) on top of the
original propose_action / propose_completion pair.

Locks in:
  - A fabricated/unknown obs_id or hyp_id is REJECTED, never silently accepted.
  - A legitimate chain (real obs -> hypothesis -> supporting validation -> finding)
    succeeds end-to-end through BaseAgent's handler methods.
  - record_finding is refused by the framework (not just by convention) when the
    hypothesis has not actually been confirmed by a supporting validation.
  - AIPlanner._parse_proposal accepts the three new decision values and enforces
    each verb's required parameters, raising ValueError (the same exception type
    the retry loop in decide_next() already catches) when they are missing.
"""

import pytest

from core.scope import Scope
from core.action_contract import AIActionProposal
from core.investigation import EvidenceStore, Observation
from agents.base_agent import BaseAgent


class _MinimalAgent(BaseAgent):
    """A BaseAgent subclass with no real phases -- same pattern used in
    test_remediation_c1_c2_h1.py to exercise shared BaseAgent logic in isolation
    without constructing a specific agent's real phase modules."""
    agent_type = "ctf"

    def _load_phases(self):
        self.phase_modules = []


def _make_agent():
    scope = Scope(target="local", active_testing_allowed=True, state_changes_allowed=True)
    return _MinimalAgent(scope=scope, mode="automated")


def _make_store_with_observation():
    store = EvidenceStore()
    obs = Observation(
        action_id="environment_recon",
        argv=["id"],
        raw_stdout="uid=0(root) gid=0(root) groups=0(root)",
        raw_stderr="",
        exit_code=0,
    )
    store.add_observation(obs)
    return store, obs


# ---------------------------------------------------------------------------
# form_hypothesis
# ---------------------------------------------------------------------------

def test_form_hypothesis_rejects_fabricated_observation_id():
    agent = _make_agent()
    store, _real_obs = _make_store_with_observation()

    proposal = AIActionProposal(
        decision="form_hypothesis",
        parameters={
            "observation_ids": ["obs_doesnotexist"],
            "description": "made up evidence",
            "validation_test": "environment_recon",
            "confidence": "plausible",
        },
    )
    result = agent._handle_form_hypothesis(proposal, store)
    assert result["status"] == "rejected"
    assert "obs_doesnotexist" in result["rejection_reason"]
    assert len(store.hypotheses) == 0


def test_form_hypothesis_succeeds_on_real_observation():
    agent = _make_agent()
    store, obs = _make_store_with_observation()

    proposal = AIActionProposal(
        decision="form_hypothesis",
        parameters={
            "observation_ids": [obs.obs_id],
            "description": "Running as root based on id output.",
            "validation_test": "environment_recon",
            "confidence": "likely",
        },
    )
    result = agent._handle_form_hypothesis(proposal, store)
    assert result["status"] == "success"
    assert result["hypothesis_id"] in store.hypotheses
    assert store.hypotheses[result["hypothesis_id"]].status == "open"


# ---------------------------------------------------------------------------
# record_validation
# ---------------------------------------------------------------------------

def test_record_validation_rejects_unknown_hypothesis_id():
    agent = _make_agent()
    store, obs = _make_store_with_observation()

    proposal = AIActionProposal(
        decision="record_validation",
        parameters={
            "hypothesis_id": "hyp_doesnotexist",
            "observation_id": obs.obs_id,
            "verdict": "supports",
            "reasoning": "fabricated",
        },
    )
    result = agent._handle_record_validation(proposal, store)
    assert result["status"] == "rejected"
    assert "hyp_doesnotexist" in result["rejection_reason"]


def test_record_validation_rejects_unknown_observation_id():
    agent = _make_agent()
    store, obs = _make_store_with_observation()
    hyp_result = agent._handle_form_hypothesis(
        AIActionProposal(
            decision="form_hypothesis",
            parameters={
                "observation_ids": [obs.obs_id],
                "description": "root access",
                "validation_test": "environment_recon",
            },
        ),
        store,
    )
    hyp_id = hyp_result["hypothesis_id"]

    proposal = AIActionProposal(
        decision="record_validation",
        parameters={
            "hypothesis_id": hyp_id,
            "observation_id": "obs_fabricated",
            "verdict": "supports",
            "reasoning": "fabricated",
        },
    )
    result = agent._handle_record_validation(proposal, store)
    assert result["status"] == "rejected"
    assert "obs_fabricated" in result["rejection_reason"]
    assert store.hypotheses[hyp_id].status == "open"  # unchanged


def test_record_validation_confirms_hypothesis_on_supports_verdict():
    agent = _make_agent()
    store, obs = _make_store_with_observation()
    hyp_result = agent._handle_form_hypothesis(
        AIActionProposal(
            decision="form_hypothesis",
            parameters={
                "observation_ids": [obs.obs_id],
                "description": "root access",
                "validation_test": "environment_recon",
            },
        ),
        store,
    )
    hyp_id = hyp_result["hypothesis_id"]

    proposal = AIActionProposal(
        decision="record_validation",
        parameters={
            "hypothesis_id": hyp_id,
            "observation_id": obs.obs_id,
            "verdict": "supports",
            "reasoning": "id output literally shows uid=0(root).",
        },
    )
    result = agent._handle_record_validation(proposal, store)
    assert result["status"] == "success"
    assert store.hypotheses[hyp_id].status == "confirmed"


# ---------------------------------------------------------------------------
# record_finding
# ---------------------------------------------------------------------------

def test_record_finding_refused_without_confirmed_hypothesis():
    agent = _make_agent()
    store, obs = _make_store_with_observation()
    hyp_result = agent._handle_form_hypothesis(
        AIActionProposal(
            decision="form_hypothesis",
            parameters={
                "observation_ids": [obs.obs_id],
                "description": "root access",
                "validation_test": "environment_recon",
            },
        ),
        store,
    )
    hyp_id = hyp_result["hypothesis_id"]
    # No record_validation call -- hypothesis is still 'open', not 'confirmed'.

    proposal = AIActionProposal(
        decision="record_finding",
        parameters={
            "hypothesis_id": hyp_id,
            "title": "Running as root",
            "description": "The recon action confirms uid 0.",
            "severity": "high",
            "affected_target": "local",
            "evidence_summary": "uid=0(root)",
        },
    )
    result = agent._handle_record_finding(proposal, store)
    assert result["status"] == "rejected"
    assert len(store.findings) == 0


def test_record_finding_succeeds_after_full_chain():
    agent = _make_agent()
    store, obs = _make_store_with_observation()
    hyp_id = agent._handle_form_hypothesis(
        AIActionProposal(
            decision="form_hypothesis",
            parameters={
                "observation_ids": [obs.obs_id],
                "description": "root access",
                "validation_test": "environment_recon",
            },
        ),
        store,
    )["hypothesis_id"]
    agent._handle_record_validation(
        AIActionProposal(
            decision="record_validation",
            parameters={
                "hypothesis_id": hyp_id,
                "observation_id": obs.obs_id,
                "verdict": "supports",
                "reasoning": "uid=0(root) in raw output",
            },
        ),
        store,
    )

    proposal = AIActionProposal(
        decision="record_finding",
        parameters={
            "hypothesis_id": hyp_id,
            "title": "Running as root",
            "description": "The recon action confirms uid 0.",
            "severity": "high",
            "affected_target": "local",
            "evidence_summary": "uid=0(root)",
            "reproducible_steps": ["id"],
            "impact": "Full host compromise if reachable remotely.",
            "confidence": 0.9,
        },
    )
    result = agent._handle_record_finding(proposal, store)
    assert result["status"] == "success"
    assert len(store.findings) == 1
    finding = store.findings[result["finding_id"]]
    assert finding.severity == "high"
    assert finding.hypothesis_id == hyp_id


# ---------------------------------------------------------------------------
# AIPlanner._parse_proposal: new decisions + per-verb required-parameter guards
# ---------------------------------------------------------------------------

def _make_planner():
    from core.ai_planner import AIPlanner
    from core.action_contract import ActionRegistry

    scope = Scope(target="local", active_testing_allowed=True)
    registry = ActionRegistry(agent_name="test")
    return AIPlanner(registry=registry, chat_assistant=None, scope=scope)


def test_parse_proposal_accepts_form_hypothesis():
    planner = _make_planner()
    raw = (
        '{"decision": "form_hypothesis", "parameters": '
        '{"observation_ids": ["obs_1"], "description": "x", "validation_test": "y"}, '
        '"rationale": "r"}'
    )
    proposal = planner._parse_proposal(raw)
    assert proposal.decision == "form_hypothesis"
    assert proposal.parameters["observation_ids"] == ["obs_1"]


def test_parse_proposal_rejects_form_hypothesis_missing_params():
    planner = _make_planner()
    raw = '{"decision": "form_hypothesis", "parameters": {"description": "x"}}'
    with pytest.raises(ValueError):
        planner._parse_proposal(raw)


def test_parse_proposal_rejects_record_finding_missing_params():
    planner = _make_planner()
    raw = '{"decision": "record_finding", "parameters": {"title": "x"}}'
    with pytest.raises(ValueError):
        planner._parse_proposal(raw)


def test_parse_proposal_accepts_record_validation_full_params():
    planner = _make_planner()
    raw = (
        '{"decision": "record_validation", "parameters": '
        '{"hypothesis_id": "hyp_1", "observation_id": "obs_1", '
        '"verdict": "supports", "reasoning": "because"}}'
    )
    proposal = planner._parse_proposal(raw)
    assert proposal.decision == "record_validation"
    assert proposal.parameters["verdict"] == "supports"
