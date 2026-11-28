"""
tests/test_record_finding_strength.py

CRITICAL FIX (2026-11-28): _handle_record_finding() never passed a `strength`
to EvidenceStore.record_finding(), so every AI-driven record_finding call
defaulted to strength="proven" with NO rigor check. Live incident (session
bug_bounty_autohunt_1795852409, OpenRouter-brain run): a self_script's own
naive "response length differs" logic set vuln_signal.confirmed=True;
record_validation's verdict='supports' was correctly accepted (it trusts a
vuln_signal's own confirmed flag); but record_finding then recorded a
"proven" IDOR finding with zero re-scrutiny of that self_script's weak
evidence. agents/base_agent.py::_compute_finding_strength() closes this by
re-running every supporting Validation's cited Observation through the same
_confirmation_strength() gate before a Finding can be "proven".
"""
import os
os.environ["PENTEST_ENABLE_SELF_SCRIPTING"] = "1"

from core.scope import Scope
from core.investigation import EvidenceStore, Observation, Hypothesis, Validation
from agents.bug_bounty.agent import BugBountyAgent


def _agent():
    return BugBountyAgent(scope=Scope(target="h.example", in_scope=["h.example"],
                                      active_testing_allowed=True, single_host_only=True))


def _proposal(hypothesis_id, **overrides):
    class _P:
        pass
    p = _P()
    p.parameters = {
        "hypothesis_id": hypothesis_id, "title": "t", "description": "d",
        "severity": "medium", "affected_target": "h.example",
        "evidence_summary": "e", "confidence": 0.9,
    }
    p.parameters.update(overrides)
    return p


def _confirmed_hypothesis_with_observation(action_id, parsed_facts, reasoning="concrete evidence cited here"):
    """Builds a store with one CONFIRMED hypothesis (via the real
    add_validation() status-flip), backed by one Observation/Validation pair."""
    store = EvidenceStore()
    obs = Observation(action_id=action_id, argv=["x"], raw_stdout="some real output",
                       exit_code=0, parsed_facts=parsed_facts)
    store.add_observation(obs)
    hyp = Hypothesis(observation_ids=[obs.obs_id], description="d", validation_test=action_id)
    store.add_hypothesis(hyp)
    store.add_validation(Validation(hypothesis_id=hyp.hyp_id, action_id=action_id,
                                    observation=obs, verdict="supports", reasoning=reasoning))
    assert store.hypotheses[hyp.hyp_id].status == "confirmed"
    return store, hyp.hyp_id


def test_live_incident_self_script_weak_len_diff_yields_lead_not_proven():
    """Exact reproduction: self_script sets confirmed=True from a bare
    'response length differs' heuristic -- must be demoted to lead."""
    a = _agent()
    store, hyp_id = _confirmed_hypothesis_with_observation(
        "self_script",
        {"vuln_signal": {
            "confirmed": True, "vuln_class": "idor", "severity": "medium",
            "summary": "IDOR on productId parameter accesses hidden products",
            "evidence": "Found 2 productIds with 200 OK and different content: 16 (len=1234), 17 (len=1240)",
        }},
    )
    result = a._handle_record_finding(_proposal(hyp_id), store)
    assert result["status"] == "success"
    finding = store.findings[result["finding_id"]]
    assert finding.strength == "lead", finding.strength
    assert store.proven_finding_count == 0


def test_framework_engine_confirmed_signal_yields_proven():
    a = _agent()
    store, hyp_id = _confirmed_hypothesis_with_observation(
        "sqli_deep_scan",
        {"vuln_signal": {"confirmed": True, "summary": "sqlmap confirmed injection"}},
    )
    result = a._handle_record_finding(_proposal(hyp_id), store)
    assert result["status"] == "success"
    finding = store.findings[result["finding_id"]]
    assert finding.strength == "proven"
    assert store.proven_finding_count == 1


def test_model_authored_real_evidence_still_yields_proven():
    a = _agent()
    store, hyp_id = _confirmed_hypothesis_with_observation(
        "self_script",
        {"vuln_signal": {
            "confirmed": True, "vuln_class": "lfi", "summary": "file disclosure",
            "evidence": "root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin",
        }},
    )
    result = a._handle_record_finding(_proposal(hyp_id), store)
    assert store.findings[result["finding_id"]].strength == "proven"


def test_no_vuln_signal_at_all_yields_lead():
    """Pure narrative-reasoning support (no engine/probe oracle at all) must
    never default to proven -- this is the ai_planner hardcoded-'supports'
    class of incident, one layer further downstream."""
    a = _agent()
    store, hyp_id = _confirmed_hypothesis_with_observation("browser_probe", {})
    result = a._handle_record_finding(_proposal(hyp_id), store)
    assert result["status"] == "success"
    assert store.findings[result["finding_id"]].strength == "lead"


def test_ai_severity_param_cannot_override_computed_strength():
    """The AI's own severity/confidence must never force a proven verdict."""
    a = _agent()
    store, hyp_id = _confirmed_hypothesis_with_observation(
        "self_script",
        {"vuln_signal": {"confirmed": True, "summary": "x", "evidence": "the response status code suggests this"}},
    )
    result = a._handle_record_finding(_proposal(hyp_id, severity="critical", confidence=1.0), store)
    assert store.findings[result["finding_id"]].strength == "lead"


def test_no_supporting_validation_yields_lead_reason():
    a = _agent()
    store = EvidenceStore()
    hyp = Hypothesis(observation_ids=[], description="d", validation_test="self_script")
    store.add_hypothesis(hyp)
    strength, reason = a._compute_finding_strength(hyp.hyp_id, store)
    assert strength == "lead"
    assert "no supporting validation" in reason.lower()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
