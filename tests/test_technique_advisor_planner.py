"""
tests/test_technique_advisor_planner.py

Tests for the framework-level RAG/YAML technique advisor (core/technique_advisor.py)
and its wiring into the shared planner (core/ai_planner.py).

The advisor is an ENHANCEMENT, never a dependency: with no FAISS index built (or
the sources otherwise unavailable) it must degrade silently to an empty string
and the planner prompt must build exactly as before, just with a placeholder in
the technique section. When knowledge IS available, it must appear in the prompt.

Because the planner is shared by every agent, this wiring lights up RAG for all
of them at once -- these tests use the bug_bounty registry as a representative.
"""

import os

import pytest

from core.technique_advisor import TechniqueAdvisor, get_technique_advisor
from core.ai_planner import AIPlanner
from core.scope import Scope
from core.investigation import EvidenceStore, Hypothesis, Observation, Validation
from agents.bug_bounty.actions import build_bug_bounty_registry
from agents.bug_bounty.completion import BUG_BOUNTY_COMPLETION


# --- advisor graceful degradation -----------------------------------------

def test_advisor_disabled_via_env_returns_empty(monkeypatch):
    monkeypatch.setenv("PENTEST_DISABLE_TECHNIQUE_ADVISOR", "1")
    adv = TechniqueAdvisor()
    assert adv.advise("sql injection on category parameter") == ""


def test_advisor_empty_query_returns_empty():
    assert TechniqueAdvisor().advise("   ") == ""


def test_advisor_degrades_when_both_sources_unavailable(monkeypatch):
    # Simulate no index / import failure for BOTH sources -> empty, never raises.
    adv = TechniqueAdvisor()
    monkeypatch.setattr(adv, "_get_rag", lambda: None)
    monkeypatch.setattr(adv, "_get_yaml", lambda: None)
    assert adv.advise("request smuggling CL.TE desync detection") == ""


def test_advisor_formats_both_sources(monkeypatch):
    adv = TechniqueAdvisor()
    monkeypatch.setattr(adv, "_rag_snippets", lambda q, k: ["Use sqlmap --crawl for param discovery"])
    monkeypatch.setattr(adv, "_yaml_snippets", lambda q, k: ["Web SQLi scenario [sqli, web]: test filter params"])
    out = adv.advise("sqli")
    assert "sqlmap --crawl" in out
    assert "Web SQLi scenario" in out
    assert "Methodology templates:" in out and "Reference technique notes:" in out


def test_advisor_truncates_to_budget(monkeypatch):
    adv = TechniqueAdvisor()
    monkeypatch.setattr(adv, "_rag_snippets", lambda q, k: ["x" * 5000])
    monkeypatch.setattr(adv, "_yaml_snippets", lambda q, k: [])
    out = adv.advise("anything", max_chars=200)
    assert len(out) <= 200


# --- planner wiring --------------------------------------------------------

def _planner():
    return AIPlanner(
        registry=build_bug_bounty_registry(),
        chat_assistant=None,  # _build_prompt never calls the model
        scope=Scope(target="https://x.web-security-academy.net/",
                    in_scope=["x.web-security-academy.net"],
                    active_testing_allowed=True, active_network_allowed=True),
    )


def test_prompt_builds_and_has_section_when_advisor_empty(monkeypatch):
    monkeypatch.setenv("PENTEST_DISABLE_TECHNIQUE_ADVISOR", "1")
    prompt = _planner()._build_prompt(
        session_state={}, evidence_store=EvidenceStore(),
        completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={},
    )
    assert "RELEVANT TECHNIQUE KNOWLEDGE" in prompt
    assert "no matching technique knowledge retrieved" in prompt


def test_prompt_injects_advice_when_available(monkeypatch):
    class _StubAdvisor:
        def advise(self, query, **kw):
            return "CANNED_ADVICE_MARKER: prefer sqlmap --crawl=2 for the category param"

    import core.technique_advisor as ta
    monkeypatch.setattr(ta, "get_technique_advisor", lambda: _StubAdvisor())

    prompt = _planner()._build_prompt(
        session_state={}, evidence_store=EvidenceStore(),
        completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={},
    )
    assert "CANNED_ADVICE_MARKER" in prompt


# --- payload pack wiring (2026-11-25 4-repo harvest, asset 8) --------------

def test_retrieve_technique_knowledge_appends_payload_pack_for_candidate_class(monkeypatch):
    monkeypatch.setenv("PENTEST_DISABLE_TECHNIQUE_ADVISOR", "1")  # isolate: no RAG/YAML noise
    planner = _planner()
    result = planner._retrieve_technique_knowledge(
        evidence_store=EvidenceStore(),
        available_action_ids=[],
        candidate_classes={"ssrf"},
    )
    assert "Payload pack (ssrf):" in result


def test_retrieve_technique_knowledge_no_candidate_classes_has_no_payload_pack(monkeypatch):
    monkeypatch.setenv("PENTEST_DISABLE_TECHNIQUE_ADVISOR", "1")
    planner = _planner()
    result = planner._retrieve_technique_knowledge(
        evidence_store=EvidenceStore(),
        available_action_ids=[],
        candidate_classes=None,
    )
    assert "Payload pack" not in result


def test_retrieve_technique_knowledge_payload_pack_failure_never_breaks_planning(monkeypatch):
    monkeypatch.setenv("PENTEST_DISABLE_TECHNIQUE_ADVISOR", "1")
    import core.payload_pack as pp
    monkeypatch.setattr(pp, "build_payload_pack", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    planner = _planner()
    result = planner._retrieve_technique_knowledge(
        evidence_store=EvidenceStore(),
        available_action_ids=[],
        candidate_classes={"ssrf"},
    )
    assert "no matching technique knowledge retrieved" in result


# --- confirmed-but-unpromoted hypothesis steering (2026-11-25 fix) ---------
#
# Diagnosed from a live PortSwigger lab run: two hypotheses reached
# status="confirmed" via record_validation, but findings_count stayed 0 for
# the whole run -- the model spent its remaining steps on form_hypothesis
# instead of ever calling record_finding. Root cause: a confirmed hypothesis
# was silently dropped from BOTH the "ACTIVE HYPOTHESES" prompt section and
# the steering ladder the instant record_validation confirmed it, so the
# model lost all visibility into it.

def _confirm_a_hypothesis(evidence_store):
    """Build one hypothesis + a supporting observation + validation so the
    hypothesis ends up status='confirmed' with no Finding yet."""
    obs = Observation(action_id="self_script", argv=["python", "x.py"], raw_stdout="ok")
    evidence_store.add_observation(obs)
    hyp = Hypothesis(
        observation_ids=[obs.obs_id],
        description="shopId on /product is an SSRF vector into the internal admin panel.",
        validation_test="self_script",
        vuln_class="ssrf",
        confidence="likely",
    )
    evidence_store.add_hypothesis(hyp)
    evidence_store.add_validation(Validation(
        hypothesis_id=hyp.hyp_id, action_id="self_script", observation=obs,
        verdict="supports", reasoning="Internal admin login page returned for shopId=127.0.0.1:8080.",
    ))
    return hyp


def test_confirmed_unpromoted_hypotheses_detects_hyp_with_no_finding():
    store = EvidenceStore()
    hyp = _confirm_a_hypothesis(store)
    planner = _planner()
    unpromoted = planner._confirmed_unpromoted_hypotheses(store)
    assert [h.hyp_id for h in unpromoted] == [hyp.hyp_id]


def test_confirmed_unpromoted_hypotheses_excludes_already_promoted():
    store = EvidenceStore()
    hyp = _confirm_a_hypothesis(store)
    store.record_finding(
        hypothesis_id=hyp.hyp_id, title="SSRF via shopId", description="x",
        severity="high", affected_target="/product", evidence_summary="x",
        reproducible_steps=["x"], impact="x", confidence=0.8,
    )
    planner = _planner()
    assert planner._confirmed_unpromoted_hypotheses(store) == []


def test_build_prompt_steers_toward_record_finding_when_hypothesis_confirmed():
    store = EvidenceStore()
    hyp = _confirm_a_hypothesis(store)
    prompt = _planner()._build_prompt(
        session_state={}, evidence_store=store,
        completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={},
    )
    assert "record_finding" in prompt
    assert hyp.hyp_id in prompt
    assert "CONFIRMED (no Finding yet" in prompt
    assert "do NOT form a new hypothesis" in prompt


def test_build_prompt_lists_confirmed_hyp_id_in_active_hypotheses_section():
    # Regression for the exact visibility bug: a confirmed hyp_id must still
    # appear somewhere the model can copy it verbatim from.
    store = EvidenceStore()
    hyp = _confirm_a_hypothesis(store)
    prompt = _planner()._build_prompt(
        session_state={}, evidence_store=store,
        completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={},
    )
    assert f"[{hyp.hyp_id}]" in prompt


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
