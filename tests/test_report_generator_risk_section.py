"""
tests/test_report_generator_risk_section.py

Tests that ll-finetuning/rag/core/report_generator.py's new "Risk Assessment &
Dual-Perspective Analysis" section only appears when an EvidenceStore is
supplied via context, and that the report renders unchanged (the original
4-bullet Strategic Recommendations as "## 7.") when it is not -- preserving
old behavior for any caller that doesn't pass a context.
"""

import pytest

from core.report_generator import ReportGenerator
from core.investigation import EvidenceStore, Observation, Finding, Hypothesis

MINIMAL_STATE = {
    "name": "test_session",
    "scope": "https://x.example/",
    "status": "completed",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:05:00Z",
    "completed_steps": [],
    "evidence": [],
    "notes": [],
    "targets": {},
    "ingested_findings": {},
}


def test_no_context_renders_original_boilerplate_section_as_7():
    generator = ReportGenerator()
    report = generator.generate(MINIMAL_STATE)
    assert "## 7. Strategic Recommendations & Remediation Plan" in report
    assert "Risk Assessment & Dual-Perspective Analysis" not in report


def test_context_without_evidence_store_also_falls_back():
    generator = ReportGenerator()
    report = generator.generate(MINIMAL_STATE, context={})
    assert "## 7. Strategic Recommendations & Remediation Plan" in report
    assert "Risk Assessment & Dual-Perspective Analysis" not in report


def test_context_with_evidence_store_inserts_risk_section_and_renumbers():
    store = EvidenceStore()
    hyp = Hypothesis(description="Reflected XSS in q param", validation_test="marker reflection")
    store.add_hypothesis(hyp)
    hyp.status = "confirmed"
    finding = Finding(
        hypothesis_id=hyp.hyp_id, observation_ids=[], validation_ids=[],
        title="Reflected XSS in /search?q=", description="Confirmed via console marker",
        severity="high", affected_target="https://x.example/search",
        evidence_summary="marker fired in headless browser", strength="proven",
    )
    store.findings[finding.finding_id] = finding
    store.add_observation(Observation(
        action_id="recon_scan", argv=["python", "recon_scan_runner.py"], raw_stdout="",
        parsed_facts={"recon": {"ports": [{"port": 22, "risk_level": "medium", "service": "ssh"}]}},
    ))

    generator = ReportGenerator()
    report = generator.generate(MINIMAL_STATE, context={"evidence_store": store})

    assert "## 7. Risk Assessment & Dual-Perspective Analysis" in report
    assert "## 8. Strategic Recommendations & Remediation Plan" in report
    assert "Owner Perspective" in report
    assert "Hunter Perspective" in report
    assert "Reflected XSS in /search?q=" in report


def test_risk_assessment_exception_degrades_gracefully(monkeypatch):
    import core.report_generator as rg_module

    def boom(evidence_store):
        raise RuntimeError("simulated failure")

    # Patch the import target so generate()'s try/except path is exercised.
    import core.risk_assessment as risk_module
    monkeypatch.setattr(risk_module, "build_risk_report", boom)

    store = EvidenceStore()
    generator = ReportGenerator()
    report = generator.generate(MINIMAL_STATE, context={"evidence_store": store})
    assert "Risk assessment unavailable" in report
    # Falls back to section "7" for recommendations since the risk section failed.
    assert "## 7. Strategic Recommendations & Remediation Plan" in report


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
