"""
tests/test_confirmation_rigor.py

Confirmation-rigor layer (2026-11-20). General across vuln classes:
  - framework detection engines are trusted (confirmed => proven);
  - model-authored probes (self_script/nuclei_template) must carry concrete
    `evidence` and avoid weak/heuristic bases, else they are downgraded to a LEAD;
  - proven_finding_count counts only PROVEN findings, so completion can't be
    satisfied by a lead or a phase milestone.
"""
import os
os.environ["PENTEST_ENABLE_SELF_SCRIPTING"] = "1"

from core.scope import Scope
from core.investigation import EvidenceStore, Observation, Hypothesis, Validation
from agents.bug_bounty.agent import BugBountyAgent


def _agent():
    return BugBountyAgent(scope=Scope(target="h.example", in_scope=["h.example"],
                                      active_testing_allowed=True, single_host_only=True))


def test_engine_confirmations_are_proven():
    a = _agent()
    for eng in ("sqli_deep_scan", "nuclei_scan", "dalfox_scan"):
        strength, _ = a._confirmation_strength(eng, {"confirmed": True, "summary": "x"})
        assert strength == "proven", eng


def test_self_script_without_evidence_is_a_lead():
    a = _agent()
    strength, why = a._confirmation_strength("self_script", {"confirmed": True, "summary": "SSRF detected"})
    assert strength == "lead", why


def test_self_script_weak_basis_is_a_lead():
    a = _agent()
    # the exact false-positive we observed: "responses differ" / status-based
    for summ in ("Response divergence detected", "status != 200 so SSRF", "different response suggests SSRF"):
        strength, why = a._confirmation_strength("self_script",
            {"confirmed": True, "summary": summ, "evidence": "x"})
        assert strength == "lead", (summ, why)


def test_self_script_with_real_evidence_is_proven():
    a = _agent()
    strength, _ = a._confirmation_strength("self_script", {
        "confirmed": True, "vuln_class": "lfi",
        "summary": "file disclosure",
        "evidence": "root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin",
    })
    assert strength == "proven"


def _chain(store, action_id="self_script"):
    obs = Observation(action_id=action_id, argv=[], raw_stdout="x", exit_code=0)
    store.add_observation(obs)
    hyp = Hypothesis(observation_ids=[obs.obs_id], description="d", validation_test=action_id)
    store.add_hypothesis(hyp)
    store.add_validation(Validation(hypothesis_id=hyp.hyp_id, action_id=action_id,
                                    observation=obs, verdict="supports", reasoning="r"))
    return hyp


def test_proven_finding_count_ignores_leads_and_milestones():
    store = EvidenceStore()
    for strength, sev in (("lead", "low"), ("milestone", "info"), ("proven", "high")):
        hyp = _chain(store)
        f, msg = store.record_finding(hypothesis_id=hyp.hyp_id, title=f"{strength}",
                                      description="d", severity=sev, affected_target="h",
                                      evidence_summary="e", strength=strength)
        assert f is not None, msg
    assert len(store.findings) == 3
    assert store.proven_finding_count == 1   # only the proven one counts
