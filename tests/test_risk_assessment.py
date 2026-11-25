"""
tests/test_risk_assessment.py

Tests for the deterministic, rule-based risk scoring + dual-perspective
(Owner/Hunter) analysis in core/risk_assessment.py, against synthetic
core.investigation.EvidenceStore fixtures. No cloud/LLM calls -- everything
here is plain Python logic.
"""

import pytest

from core.investigation import EvidenceStore, Observation, Hypothesis, Finding
from core.risk_assessment import (
    collect_recon_facts, assess_risk, owner_perspective, hunter_perspective,
    build_risk_report,
)


def _recon_observation(recon_dict):
    return Observation(
        action_id="recon_scan",
        argv=["python", "recon_scan_runner.py"],
        raw_stdout="",
        parsed_facts={"recon": recon_dict},
    )


def test_empty_store_is_low_risk():
    store = EvidenceStore()
    recon_facts = collect_recon_facts(store)
    risk = assess_risk(store, recon_facts)
    assert risk["level"] == "Low"
    assert risk["score"] == 0
    assert risk["factors"] == []


def test_collect_recon_facts_merges_multiple_observations():
    store = EvidenceStore()
    store.add_observation(_recon_observation({"ports": [{"port": 22, "risk_level": "medium", "service": "ssh"}]}))
    store.add_observation(_recon_observation({"ports": [{"port": 445, "risk_level": "critical", "service": "smb"}],
                                               "subdomains": ["admin.example.com"]}))
    facts = collect_recon_facts(store)
    ports = {p["port"] for p in facts["ports"]}
    assert ports == {22, 445}
    assert facts["subdomains"] == ["admin.example.com"]


def test_proven_finding_raises_risk_and_appears_in_owner_key_findings():
    store = EvidenceStore()
    hyp = Hypothesis(description="SQLi in search param", validation_test="boolean pair", vuln_class="sqli")
    store.add_hypothesis(hyp)
    hyp.status = "confirmed"
    finding = Finding(
        hypothesis_id=hyp.hyp_id, observation_ids=[], validation_ids=[],
        title="SQL injection in /search", description="Confirmed boolean-based SQLi",
        severity="critical", affected_target="https://x.example/search",
        evidence_summary="boolean divergence observed", strength="proven",
    )
    store.findings[finding.finding_id] = finding

    recon_facts = collect_recon_facts(store)
    risk = assess_risk(store, recon_facts)
    assert risk["level"] in ("High", "Critical")
    assert risk["score"] > 0
    assert any("proven finding" in f for f in risk["factors"])

    owner = owner_perspective(store, recon_facts, risk)
    assert any("SQL injection in /search" in kf for kf in owner["key_findings"])
    assert owner["immediate_actions"]


def test_lead_findings_do_not_count_as_proven():
    store = EvidenceStore()
    hyp = Hypothesis(description="maybe XSS", validation_test="reflection check")
    store.add_hypothesis(hyp)
    lead = Finding(
        hypothesis_id=hyp.hyp_id, observation_ids=[], validation_ids=[],
        title="Possible reflected XSS", description="unencoded reflection, unverified",
        severity="medium", affected_target="https://x.example/", evidence_summary="reflection only",
        strength="lead",
    )
    store.findings[lead.finding_id] = lead
    recon_facts = collect_recon_facts(store)
    risk = assess_risk(store, recon_facts)
    assert risk["score"] == 0
    assert risk["level"] == "Low"


def test_critical_pattern_subdomains_bucketed_and_scored():
    store = EvidenceStore()
    store.add_observation(_recon_observation({"subdomains": ["admin.example.com", "phpmyadmin.example.com",
                                                               "www.example.com"]}))
    recon_facts = collect_recon_facts(store)
    risk = assess_risk(store, recon_facts)
    assert risk["critical_assets"] >= 1
    assert risk["score"] > 0
    buckets = risk["subdomain_buckets"]
    assert buckets["critical"]["admin"]  # admin.example.com caught
    assert "www.example.com" not in sum(buckets["critical"].values(), [])


def test_hunter_perspective_surfaces_open_hypotheses_and_risky_ports():
    store = EvidenceStore()
    store.add_hypothesis(Hypothesis(description="Possible SSRF via image fetch",
                                     validation_test="OOB callback", vuln_class="ssrf"))
    store.add_observation(_recon_observation({
        "ports": [{"port": 6379, "risk_level": "high", "service": "redis", "note": "Unauthenticated by default"}],
        "auth_pages": [{"url": "https://x.example/login", "type": "login", "status": 200}],
    }))
    recon_facts = collect_recon_facts(store)
    risk = assess_risk(store, recon_facts)
    hunter = hunter_perspective(store, recon_facts, risk)
    targets = [t["target"] for t in hunter["promising_targets"]]
    assert any("6379" in t for t in targets)
    assert any("login" in t for t in targets)
    assert any("SSRF" in t or "ssrf" in str(t).lower() or "Possible SSRF" in t for t in targets)


def test_build_risk_report_shape():
    store = EvidenceStore()
    report = build_risk_report(store)
    assert set(report.keys()) == {"risk_assessment", "recon_facts", "owner_analysis", "hunter_analysis"}
    assert "summary" in report["owner_analysis"]
    assert "summary" in report["hunter_analysis"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
