"""
tests/test_stealth.py

Tests for the stealth layer (core/stealth.py) and its executor integration.
Default (lab) profile must be a total no-op; stealth profile must pace, back off
adaptively, and gate loud high-volume scanners.
"""

import types

import pytest

from core.stealth import StealthController
from core.scope import Scope
from core.action_contract import AIActionProposal
from core.command_executor import CommandExecutor
from agents.bug_bounty.actions import build_bug_bounty_registry


def _defn(action_id, volume):
    return types.SimpleNamespace(action_id=action_id, request_volume=volume,
                                 requires_active_testing=True, network_effect="active")


# --- default (lab) profile is a no-op -------------------------------------

def test_disabled_controller_is_noop(monkeypatch):
    monkeypatch.delenv("PENTEST_STEALTH", raising=False)
    c = StealthController.from_scope(Scope(target="127.0.0.1"))
    assert c.enabled is False
    assert c.is_action_allowed(_defn("sqli_deep_scan", "high")) == (True, "")
    # pace never sleeps when disabled
    slept = []
    monkeypatch.setattr("core.stealth.time.sleep", lambda s: slept.append(s))
    assert c.pace(is_active=True) == 0.0 and slept == []


def test_env_forces_on_and_off(monkeypatch):
    monkeypatch.setenv("PENTEST_STEALTH", "1")
    assert StealthController.from_scope(Scope(target="x")).enabled is True
    monkeypatch.setenv("PENTEST_STEALTH", "0")
    assert StealthController.from_scope(Scope(target="x", stealth=True)).enabled is False


def test_scope_flag_enables(monkeypatch):
    monkeypatch.delenv("PENTEST_STEALTH", raising=False)
    assert StealthController.from_scope(Scope(target="x", stealth=True)).enabled is True


# --- scanner gating --------------------------------------------------------

def test_high_volume_gated_under_stealth(monkeypatch):
    monkeypatch.delenv("PENTEST_STEALTH_ALLOW_SCANNERS", raising=False)
    c = StealthController(enabled=True, aggressive_allowed=False)
    ok, reason = c.is_action_allowed(_defn("nuclei_scan", "high"))
    assert ok is False and "gated off" in reason
    assert c.is_action_allowed(_defn("http_probe", "low"))[0] is True


def test_scanners_allowed_when_opted_in():
    c = StealthController(enabled=True, aggressive_allowed=True)
    assert c.is_action_allowed(_defn("sqli_deep_scan", "high"))[0] is True


# --- pacing + adaptive backoff --------------------------------------------

def test_pace_sleeps_only_for_active(monkeypatch):
    slept = []
    monkeypatch.setattr("core.stealth.time.sleep", lambda s: slept.append(s))
    c = StealthController(enabled=True, min_delay=1.0, max_delay=1.0)
    c.pace(is_active=False)
    assert slept == []
    c.pace(is_active=True)
    assert slept and slept[0] >= 1.0


def test_backoff_grows_on_waf_and_recovers():
    c = StealthController(enabled=True, min_delay=0.0, max_delay=0.0)
    base = c.backoff
    c.note_result("HTTP/1.1 429 Too Many Requests")
    assert c.backoff > base
    high = c.backoff
    for _ in range(20):
        c.note_result("normal response body")
    assert c.backoff <= high and c.backoff >= 1.0


# --- action marking + executor integration --------------------------------

def test_loud_scanners_marked_high_volume():
    reg = build_bug_bounty_registry()
    for aid in ("subdomain_enum", "port_scan", "dir_brute", "sqli_deep_scan", "nuclei_scan"):
        assert reg.get(aid).request_volume == "high", aid
    for aid in ("http_probe", "browser_probe", "header_analysis", "sqli_probe"):
        assert reg.get(aid).request_volume == "low", aid


def test_executor_gates_scanner_under_stealth(monkeypatch):
    monkeypatch.delenv("PENTEST_STEALTH", raising=False)
    monkeypatch.delenv("PENTEST_STEALTH_ALLOW_SCANNERS", raising=False)
    ex = CommandExecutor(read_only=False)
    scope = Scope(target="https://x.web-security-academy.net/",
                  in_scope=["x.web-security-academy.net"],
                  active_testing_allowed=True, active_network_allowed=True, stealth=True)
    reg = build_bug_bounty_registry()
    proposal = AIActionProposal(
        decision="propose_action", action_id="sqli_deep_scan",
        target="https://x.web-security-academy.net/filter?category=Gifts",
        parameters={"target": "https://x.web-security-academy.net/filter?category=Gifts"},
    )
    res = ex.execute_action(proposal=proposal, registry=reg, scope=scope,
                            execution_history=[], mode="automated")
    # Gated at GATE 6.5, BEFORE any command runs -- no network touched.
    assert res["status"] == "rejected" and res["rejection_category"] == "stealth_gated"


def test_executor_does_not_gate_in_lab_mode(monkeypatch):
    # Without stealth, the same scanner is NOT gated (it proceeds past 6.5).
    monkeypatch.delenv("PENTEST_STEALTH", raising=False)
    ex = CommandExecutor(read_only=False)
    scope = Scope(target="127.0.0.1", in_scope=["127.0.0.1"],
                  active_testing_allowed=True, active_network_allowed=True)  # stealth defaults False
    c = ex._stealth(scope)
    assert c.enabled is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
