"""
tests/test_premise_crosscheck.py

BUILD SPEC v1 workstream D-1 (2026-11-28): the false-IDOR fix (audit Defect #6 +
Section 1's second root cause) + EvidenceStore dedup (#25).

CORE GUARD under test: a model-authored claim is DEMOTED only when the store's
recorded facts ACTUALLY CONTRADICT its premise. Absence of corroboration is NOT
contradiction -- a genuine finding whose premise is not contradicted must still
reach "proven" (no new false negatives). Deterministic-engine signals are never
premise-doubted. Dedup is on (overlapping observation_ids AND same claim), so
distinct vulns on a shared observation both still count.
"""
import os
os.environ.setdefault("PENTEST_ENABLE_SELF_SCRIPTING", "1")

from core.scope import Scope
from core.investigation import EvidenceStore, Observation, Hypothesis, Validation
from agents.bug_bounty.agent import BugBountyAgent


def _agent():
    return BugBountyAgent(scope=Scope(target="h.example", in_scope=["h.example"],
                                      active_testing_allowed=True, single_host_only=True))


def _endpoint_obs(n=20):
    """An endpoint_discovery observation listing productId=1..n as PUBLIC links."""
    return Observation(
        action_id="endpoint_discovery", argv=[], raw_stdout="<html>...</html>", exit_code=0,
        parsed_facts={"endpoints": [f"https://h.example/product?productId={i}" for i in range(1, n + 1)]},
    )


def _selfscript_obs(evidence, confirmed=True, action_id="self_script", vuln_class="idor"):
    sig = {"confirmed": confirmed, "severity": "medium", "vuln_class": vuln_class,
           "summary": "IDOR", "evidence": evidence}
    return Observation(
        action_id=action_id, argv=["python", "probe.py"], raw_stdout="{}", exit_code=0,
        parsed_facts={"vuln_signal": sig},
    )


def _confirm(store, hyp, obs):
    store.add_validation(Validation(hypothesis_id=hyp.hyp_id, action_id=obs.action_id,
                                    observation=obs, verdict="supports", reasoning="evidence cited"))


# --- Part 1/2: the false-IDOR demotion --------------------------------------

def test_false_idor_premise_contradicted_demoted_to_lead():
    a = _agent()
    store = EvidenceStore()
    ep = _endpoint_obs(20)
    ss = _selfscript_obs(
        "productId=16 returned HTTP 200 exposing gift-card note 'ADMIN-GC-9910', absent for productId=1")
    store.add_observation(ep)
    store.add_observation(ss)
    hyp = Hypothesis(
        observation_ids=[ss.obs_id, ep.obs_id],
        description="IDOR on productId",
        validation_test="self_script",
        vuln_class="idor",
        premise="productIds 16 to 20 are hidden, beyond the public range 1-15, so admin-only",
    )
    store.add_hypothesis(hyp)
    _confirm(store, hyp, ss)

    strength, reason = a._compute_finding_strength(hyp.hyp_id, store)
    assert strength == "lead", (strength, reason)
    assert "contradict" in reason.lower()
    assert ep.obs_id in reason   # names the contradicting observation


def test_false_idor_validation_rejected_on_premise_contradiction():
    """The record_validation gate also rejects a model-authored 'supports' whose
    premise the store contradicts."""
    a = _agent()
    store = EvidenceStore()
    ep = _endpoint_obs(20)
    ss = _selfscript_obs("productId=17 leaked another user's record 'ADMIN-GC-9911'")
    store.add_observation(ep)
    store.add_observation(ss)
    hyp = Hypothesis(observation_ids=[ss.obs_id], description="x", validation_test="self_script",
                     vuln_class="idor", premise="productId 17 is hidden / beyond public range")
    store.add_hypothesis(hyp)
    ok, why = a._validate_supports_claim(ss, "productId=17 leaked a record", evidence_store=store, hypothesis=hyp)
    assert ok is False
    assert "premise contradiction" in why.lower()


# --- Guard against OVER-correction ------------------------------------------

def test_uncontradicted_premise_still_proven():
    """A model-authored confirmed claim whose premise is NOT contradicted by the
    store must STILL be proven -- no false negative."""
    a = _agent()
    store = EvidenceStore()
    ep = _endpoint_obs(20)   # public values exist, but the premise makes no hidden-ness claim
    ss = _selfscript_obs(
        "stockApi fetched http://169.254.169.254/ and returned the cloud metadata document body",
        vuln_class="ssrf")
    store.add_observation(ep)
    store.add_observation(ss)
    hyp = Hypothesis(
        observation_ids=[ss.obs_id],
        description="SSRF via stockApi",
        validation_test="self_script",
        vuln_class="ssrf",
        premise="the stockApi parameter is forwarded to a server-side HTTP fetch",
    )
    store.add_hypothesis(hyp)
    _confirm(store, hyp, ss)
    strength, reason = a._compute_finding_strength(hyp.hyp_id, store)
    assert strength == "proven", (strength, reason)


def test_premise_naming_unknown_values_is_not_contradiction():
    """Premise frames values as hidden, but those values are NOT in the store ->
    absence of corroboration is not contradiction -> still proven."""
    a = _agent()
    store = EvidenceStore()
    store.add_observation(_endpoint_obs(5))   # store only knows productId 1..5
    ss = _selfscript_obs("productId=777 leaked another user's private order 'ORD-777'")
    store.add_observation(ss)
    hyp = Hypothesis(observation_ids=[ss.obs_id], description="x", validation_test="self_script",
                     vuln_class="idor", premise="productId 777 is hidden / beyond the public range")
    store.add_hypothesis(hyp)
    _confirm(store, hyp, ss)
    strength, _ = a._compute_finding_strength(hyp.hyp_id, store)
    assert strength == "proven"   # 777 never seen publicly -> premise not contradicted


# --- Deterministic engine signals are never premise-doubted -----------------

def test_engine_signal_not_subjected_to_premise_doubt():
    a = _agent()
    store = EvidenceStore()
    ep = _endpoint_obs(20)
    # Engine action (NOT in _MODEL_AUTHORED_ACTIONS), with a premise that WOULD be
    # contradicted if it were a model claim -- must stay proven regardless.
    eng = _selfscript_obs("productId=16 dumped another row", action_id="sqli_deep_scan", vuln_class="idor")
    store.add_observation(ep)
    store.add_observation(eng)
    hyp = Hypothesis(observation_ids=[eng.obs_id], description="x", validation_test="sqli_deep_scan",
                     vuln_class="idor", premise="productIds 16-20 are hidden beyond public range")
    store.add_hypothesis(hyp)
    _confirm(store, hyp, eng)
    strength, reason = a._compute_finding_strength(hyp.hyp_id, store)
    assert strength == "proven", (strength, reason)


def test_premise_contradiction_helper_direct():
    a = _agent()
    store = EvidenceStore()
    store.add_observation(_endpoint_obs(20))
    # Hidden-marker + overlapping value -> contradiction.
    c, msg, obs = a._premise_contradiction("ids 16-18 are hidden beyond the public range", store)
    assert c is True and obs is not None and "16" in msg
    # No hidden-marker -> not a contradiction.
    c2, _, _ = a._premise_contradiction("stockApi is forwarded to an internal fetch", store)
    assert c2 is False
    # Hidden-marker but value not in store -> not a contradiction.
    c3, _, _ = a._premise_contradiction("id 999 is hidden beyond range", store)
    assert c3 is False


# --- Part 3: dedup at the EvidenceStore -------------------------------------

def _proven_finding(store, obs, vuln_class, title):
    hyp = Hypothesis(observation_ids=[obs.obs_id], description=title, validation_test="self_script",
                     vuln_class=vuln_class)
    store.add_hypothesis(hyp)
    store.add_validation(Validation(hypothesis_id=hyp.hyp_id, action_id=obs.action_id,
                                    observation=obs, verdict="supports", reasoning="r"))
    return store.record_finding(
        hypothesis_id=hyp.hyp_id, title=title, description="d", severity="high",
        affected_target="h.example", evidence_summary="e", strength="proven")


def test_dedup_same_claim_same_obs_counts_once():
    """Both the automatic and manual promotion paths ultimately call
    EvidenceStore.record_finding; two calls for the SAME claim + overlapping
    observation must yield exactly ONE Finding, proven_finding_count == 1."""
    store = EvidenceStore()
    obs = _selfscript_obs("ssrf confirmed", vuln_class="ssrf")
    store.add_observation(obs)
    f1, _m1 = _proven_finding(store, obs, "ssrf", "SSRF via stockApi")
    f2, m2 = _proven_finding(store, obs, "ssrf", "SSRF via stockApi (again)")
    assert f1 is not None and f2 is not None
    assert f2.finding_id == f1.finding_id      # merged into the existing one
    assert "merged" in m2.lower()
    assert len(store.findings) == 1
    assert store.proven_finding_count == 1


def test_dedup_does_not_collapse_distinct_vulns_on_shared_obs():
    """Two GENUINELY DISTINCT vulns (different vuln_class) evidenced by the SAME
    observation must BOTH count -- dedup is on (obs overlap AND same claim)."""
    store = EvidenceStore()
    obs = _selfscript_obs("two issues here", vuln_class="ssrf")
    store.add_observation(obs)
    f1, _ = _proven_finding(store, obs, "ssrf", "SSRF via stockApi")
    f2, _ = _proven_finding(store, obs, "xss", "Stored XSS in comment")
    assert f1.finding_id != f2.finding_id
    assert len(store.findings) == 2
    assert store.proven_finding_count == 2


def test_dedup_merge_upgrades_strength():
    """A later 'proven' for the same claim upgrades a prior 'lead' (merge keeps the
    stronger), still as ONE finding."""
    store = EvidenceStore()
    obs = _selfscript_obs("ssrf", vuln_class="ssrf")
    store.add_observation(obs)
    hyp = Hypothesis(observation_ids=[obs.obs_id], description="SSRF", validation_test="self_script",
                     vuln_class="ssrf")
    store.add_hypothesis(hyp)
    store.add_validation(Validation(hypothesis_id=hyp.hyp_id, action_id=obs.action_id,
                                    observation=obs, verdict="supports", reasoning="r"))
    lead, _ = store.record_finding(hypothesis_id=hyp.hyp_id, title="SSRF", description="d",
                                   severity="low", affected_target="h.example",
                                   evidence_summary="e", strength="lead")
    assert store.proven_finding_count == 0
    proven, msg = store.record_finding(hypothesis_id=hyp.hyp_id, title="SSRF", description="d",
                                       severity="high", affected_target="h.example",
                                       evidence_summary="e", strength="proven")
    assert proven.finding_id == lead.finding_id
    assert len(store.findings) == 1
    assert store.proven_finding_count == 1


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
