"""
tests/test_lead_promotion.py

Tests for agents/base_agent.py::BaseAgent._promote_lead_findings_from_recon()
(2026-11-24 ARTEX merge round 2). The GRADING RULE this function exists to
enforce: recon_scan's secrets/exposed_files/takeovers hits are regex/pattern/
content-signature matches, never an actual confirming probe, so they must
land as Finding(strength="lead") and NEVER "proven" -- structurally, not by
convention. The adversarial test below plants a fabricated "confirmed": true
inside the recon facts to prove the function has no code path that can be
tricked into emitting "proven".
"""

import pytest

from core.scope import Scope
from core.investigation import Observation


class _DummyChatAssistant:
    def __init__(self, *args, **kwargs):
        pass

    def synthesize_report(self, state, context=None):
        return "stub report"


def _make_agent(monkeypatch):
    import agents.base_agent as base_agent_mod
    from agents.pentest.agent import PentestAgent
    from core.session_manager import SessionManager

    monkeypatch.setattr(base_agent_mod, "ChatAssistant", _DummyChatAssistant)
    scope = Scope(target="https://x.example/", in_scope=["x.example"],
                  active_testing_allowed=True, active_network_allowed=True)
    agent = PentestAgent(scope=scope, mode="automated", session_manager=SessionManager())
    agent.chat_assistant = _DummyChatAssistant()
    return agent


def _recon_obs(recon_dict):
    return Observation(
        action_id="recon_scan",
        argv=["python", "recon_scan_runner.py"],
        raw_stdout="",
        parsed_facts={"recon": recon_dict},
    )


def test_secrets_hit_promotes_exactly_one_lead_finding(monkeypatch):
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = _recon_obs({"secrets": [
        {"name": "AWS Access Key ID", "severity": "high", "masked_value": "AKIA****NOP"},
    ]})
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 1
    finding = next(iter(store.findings.values()))
    assert finding.strength == "lead"
    assert "AKIA****NOP" in finding.description
    assert "AKIAABCD" not in finding.description  # the real value was never passed in


def test_adversarial_confirmed_true_in_recon_facts_never_becomes_proven(monkeypatch):
    """A fabricated 'confirmed': true inside the secrets hit must not change
    anything -- this function has no branch that reads a 'confirmed' key at
    all, so the only possible strength it can ever emit is 'lead'."""
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = _recon_obs({"secrets": [
        {"name": "AWS Secret Access Key", "severity": "critical",
         "masked_value": "xxxx****xxxx", "confirmed": True, "proven": True},
    ]})
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 1
    finding = next(iter(store.findings.values()))
    assert finding.strength == "lead"


def test_exposed_file_without_signature_match_is_not_promoted(monkeypatch):
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = _recon_obs({"exposed_files": [
        {"path": "/.env", "url": "https://x.example/.env", "status": 200,
         "severity": "critical", "description": "env file", "signature_matched": False},
    ]})
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 0


def test_exposed_file_with_signature_match_promotes_a_lead(monkeypatch):
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = _recon_obs({"exposed_files": [
        {"path": "/.git/HEAD", "url": "https://x.example/.git/HEAD", "status": 200,
         "severity": "high", "description": "Git repository metadata", "signature_matched": True},
    ]})
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 1
    finding = next(iter(store.findings.values()))
    assert finding.strength == "lead"
    assert finding.severity in ("info", "low")


def test_takeover_hit_promotes_a_lead(monkeypatch):
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = _recon_obs({"takeovers": [
        {"subdomain": "sub.x.example", "cname": "sub.x.example.github.io.", "service": "github_pages"},
    ]})
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 1
    finding = next(iter(store.findings.values()))
    assert finding.strength == "lead"


def test_dns_ports_tls_real_ip_facts_never_promote_any_finding(monkeypatch):
    """Pure-fact modes (dns/ports/tls/tech/real_ip/js_recon) must never get a
    Finding from this function -- only secrets/exposed_files/takeovers/
    auth_sweep do."""
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = _recon_obs({
        "dns": {"A": ["1.2.3.4"]},
        "ports": [{"port": 22, "risk_level": "medium", "service": "ssh"}],
        "tls": {"self_signed": True, "expired": True},
        "tech": [{"name": "nginx", "version": "1.18.0"}],
        "real_ip": {"candidates": [{"ip": "45.33.32.156", "score": 10}]},
        "js_recon": {"routes": [{"url": "https://x.example/api/users", "param": "", "source": "js_html"}],
                     "forms": [], "total_mined": 1, "pruned_inert": 0},
    })
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 0


def test_auth_sweep_hit_promotes_a_lead(monkeypatch):
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = _recon_obs({"auth_sweep": [
        {"url": "https://x.example/admin/users", "method": "GET", "status": 200,
         "note": "GET /admin/users -> 200 unauthenticated (declared-secured, NOT enforced)"},
    ]})
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 1
    finding = next(iter(store.findings.values()))
    assert finding.strength == "lead"
    assert "x.example/admin/users" in finding.description


def test_auth_sweep_error_entry_is_not_promoted(monkeypatch):
    """A failed sweep reports [{"error": ...}] with no 'url' key -- must never
    be mistaken for a finding-shaped hit."""
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = _recon_obs({"auth_sweep": [{"error": "connection refused"}]})
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 0


def test_adversarial_confirmed_true_in_auth_sweep_never_becomes_proven(monkeypatch):
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = _recon_obs({"auth_sweep": [
        {"url": "https://x.example/admin/users", "method": "GET", "status": 200,
         "confirmed": True, "proven": True},
    ]})
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 1
    finding = next(iter(store.findings.values()))
    assert finding.strength == "lead"


def test_non_recon_scan_observations_are_ignored(monkeypatch):
    from core.investigation import EvidenceStore
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    obs = Observation(
        action_id="header_analysis", argv=["curl"], raw_stdout="",
        parsed_facts={"secrets": [{"name": "fake", "severity": "high", "masked_value": "x"}]},
    )
    store.add_observation(obs)

    agent._promote_lead_findings_from_recon(observations=[obs], evidence_store=store)

    assert len(store.findings) == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
