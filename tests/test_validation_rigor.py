"""
tests/test_validation_rigor.py

Structural rigor gate for record_validation's verdict='supports' (2026-11-26).

BACKGROUND: a live run report found that EvidenceStore.add_validation() flipped
a Hypothesis to 'confirmed' on the AI's bare say-so -- there was NO check that
the cited observation's content actually supports the claim. A dead `pass`-only
self_script (no real request, no real output) combined with the planner's own
hardcoded `"verdict": "supports"` worked example to produce a false 'proven'
SSRF finding that reached propose_completion and got written into the
training-trajectory JSONL files. agents/base_agent.py::_validate_supports_claim
+ its wiring into _handle_record_validation closes this gap structurally.
"""
import os
os.environ["PENTEST_ENABLE_SELF_SCRIPTING"] = "1"

from core.scope import Scope
from core.investigation import EvidenceStore, Observation, Hypothesis
from agents.bug_bounty.agent import BugBountyAgent


def _agent():
    return BugBountyAgent(scope=Scope(target="h.example", in_scope=["h.example"],
                                      active_testing_allowed=True, single_host_only=True))


def _proposal(hypothesis_id, observation_id, verdict, reasoning):
    class _P:
        pass
    p = _P()
    p.parameters = {
        "hypothesis_id": hypothesis_id, "observation_id": observation_id,
        "verdict": verdict, "reasoning": reasoning,
    }
    return p


def _store_with_hyp_and_obs(obs: Observation):
    store = EvidenceStore()
    store.add_observation(obs)
    hyp = Hypothesis(observation_ids=[obs.obs_id], description="d", validation_test="self_script")
    store.add_hypothesis(hyp)
    return store, hyp.hyp_id


# --- _validate_supports_claim unit tests ----------------------------------

def test_supports_rejected_on_empty_observation_output():
    a = _agent()
    obs = Observation(action_id="self_script", argv=[], raw_stdout="", exit_code=0)
    ok, why = a._validate_supports_claim(obs, "this clearly shows SSRF is real")
    assert ok is False
    assert "no output" in why.lower() or "empty" in why.lower()


def test_supports_rejected_when_vuln_signal_says_not_confirmed():
    a = _agent()
    obs = Observation(action_id="self_script", argv=[], raw_stdout='{"vuln_signal": {"confirmed": false}}',
                       exit_code=0, parsed_facts={"vuln_signal": {"confirmed": False, "summary": "no evidence"}})
    ok, why = a._validate_supports_claim(obs, "the probe confirms SSRF")
    assert ok is False
    assert "confirmed=false" in why.lower() or "confirmed" in why.lower()


def test_supports_accepted_when_vuln_signal_says_confirmed():
    a = _agent()
    obs = Observation(action_id="self_script", argv=[], raw_stdout='{"vuln_signal": {"confirmed": true}}',
                       exit_code=0,
                       parsed_facts={"vuln_signal": {"confirmed": True, "evidence": "root:x:0:0:root:/root:/bin/bash"}})
    ok, why = a._validate_supports_claim(obs, "engine confirmed it")
    assert ok is True


def test_supports_rejected_on_weak_reasoning_keyword():
    a = _agent()
    obs = Observation(action_id="self_script", argv=[], raw_stdout="some real output here", exit_code=0)
    ok, why = a._validate_supports_claim(obs, "the response status code suggests this might be vulnerable")
    assert ok is False


def test_supports_rejected_on_xml_parser_error_in_reasoning():
    a = _agent()
    obs = Observation(action_id="self_script", argv=[], raw_stdout="some real output here", exit_code=0)
    ok, why = a._validate_supports_claim(
        obs, "SAXParseException: Content is not allowed in prolog -- this confirms SSRF")
    assert ok is False


def test_supports_rejected_on_too_short_reasoning():
    a = _agent()
    obs = Observation(action_id="self_script", argv=[], raw_stdout="some real output here", exit_code=0)
    ok, why = a._validate_supports_claim(obs, "yes")
    assert ok is False


def test_supports_accepted_with_concrete_quoted_evidence():
    a = _agent()
    obs = Observation(action_id="self_script", argv=[],
                       raw_stdout='Internal admin page returned "ami-id: i-0abc123"', exit_code=0)
    ok, why = a._validate_supports_claim(
        obs, 'Response body literally contains the internal admin marker "ami-id: i-0abc123"')
    assert ok is True


# --- end-to-end: _handle_record_validation rejects the exact live bug scenario --

def test_record_validation_rejects_dead_script_claiming_supports():
    """Reproduces the reported live false positive: a self_script that produced
    NO real output (the `pass`-only dead-code case) cannot be cited as support."""
    a = _agent()
    obs = Observation(action_id="self_script", argv=["python", "probe.py"], raw_stdout="", exit_code=0)
    store, hyp_id = _store_with_hyp_and_obs(obs)
    proposal = _proposal(hyp_id, obs.obs_id, "supports",
                          "obs provides concrete evidence of SSRF via internal IP timeout")
    result = a._handle_record_validation(proposal, store)
    assert result["status"] == "rejected"
    assert store.hypotheses[hyp_id].status == "open"  # never flipped to confirmed


def test_record_validation_accepts_real_evidence_and_confirms_hypothesis():
    a = _agent()
    obs = Observation(action_id="self_script", argv=["python", "probe.py"],
                       raw_stdout='Metadata endpoint returned "ami-id: i-0abc123def"', exit_code=0)
    store, hyp_id = _store_with_hyp_and_obs(obs)
    proposal = _proposal(hyp_id, obs.obs_id, "supports",
                          'Response literally contains the cloud metadata marker "ami-id: i-0abc123def"')
    result = a._handle_record_validation(proposal, store)
    assert result["status"] == "success"
    assert store.hypotheses[hyp_id].status == "confirmed"


def test_record_validation_refutes_is_never_gated_even_with_empty_observation():
    a = _agent()
    obs = Observation(action_id="self_script", argv=[], raw_stdout="", exit_code=0)
    store, hyp_id = _store_with_hyp_and_obs(obs)
    proposal = _proposal(hyp_id, obs.obs_id, "refutes", "no evidence of anything, the probe found nothing")
    result = a._handle_record_validation(proposal, store)
    assert result["status"] == "success"
    assert store.hypotheses[hyp_id].status == "refuted"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
