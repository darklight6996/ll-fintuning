"""
test_mode_controller.py

Phase 11 — Scope Enforcer & Mode Controller unit tests.

Tests:
1. AssignmentMode enum values
2. ScopeEnforcer exact IP matching
3. ScopeEnforcer CIDR network matching
4. ScopeEnforcer domain and wildcard matching
5. ScopeEnforcer URL host extraction
6. ScopeEnforcer explicit exclusions (!IP, !domain)
7. ScopeEnforcer localhost blocking by default and allow_localhost
8. ScopeEnforcer load from text and JSON files
9. ModeController manual mode blocks execution with guidance
10. ModeController automated mode authorizes in-scope targets
11. ModeController per-session mode override
12. Singleton accessor get_mode_controller()
"""

import json
import os
import sys
import tempfile
import pytest

_HERE = os.path.dirname(__file__)
_CORE_ROOT = os.path.dirname(_HERE)
_RAG_ROOT = os.path.dirname(_CORE_ROOT)
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)

from core.mode_controller import (
    AssignmentMode,
    ModeController,
    ScopeEnforcer,
    get_mode_controller,
)


def test_assignment_mode_values():
    assert AssignmentMode.MANUAL.value == "MANUAL"
    assert AssignmentMode.AUTOMATED.value == "AUTOMATED"


def test_scope_enforcer_exact_ip():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    enforcer.add_target("192.168.1.100")

    assert enforcer.is_in_scope("10.10.10.5") is True
    assert enforcer.is_in_scope("192.168.1.100") is True
    assert enforcer.is_in_scope("10.10.10.6") is False
    assert enforcer.is_in_scope("8.8.8.8") is False


def test_scope_enforcer_cidr_network():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.0/24")

    assert enforcer.is_in_scope("10.10.10.1") is True
    assert enforcer.is_in_scope("10.10.10.254") is True
    assert enforcer.is_in_scope("10.10.11.1") is False
    assert enforcer.is_in_scope("192.168.1.1") is False


def test_scope_enforcer_domain_and_wildcard():
    enforcer = ScopeEnforcer()
    enforcer.add_target("target.local")
    enforcer.add_target("*.corp.example.com")

    # Exact domain
    assert enforcer.is_in_scope("target.local") is True
    assert enforcer.is_in_scope("other.local") is False

    # Wildcard
    assert enforcer.is_in_scope("auth.corp.example.com") is True
    assert enforcer.is_in_scope("api.corp.example.com") is True
    assert enforcer.is_in_scope("corp.example.com") is True
    assert enforcer.is_in_scope("evil.com") is False


def test_scope_enforcer_url_host_extraction():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    enforcer.add_target("api.target.com")

    assert enforcer.is_in_scope("http://10.10.10.5:8080/admin/login") is True
    assert enforcer.is_in_scope("https://api.target.com/v1/users") is True
    assert enforcer.is_in_scope("https://out-of-scope.com/test") is False


def test_scope_enforcer_explicit_exclusions():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.0/24")
    enforcer.add_target("!10.10.10.1")  # Exclude gateway
    enforcer.add_target("*.target.com")
    enforcer.add_target("!internal.target.com")

    assert enforcer.is_in_scope("10.10.10.5") is True
    assert enforcer.is_in_scope("10.10.10.1") is False  # Excluded IP

    assert enforcer.is_in_scope("api.target.com") is True
    assert enforcer.is_in_scope("internal.target.com") is False  # Excluded domain


def test_scope_enforcer_localhost_protection():
    strict_enforcer = ScopeEnforcer(allow_localhost=False)
    strict_enforcer.add_target("127.0.0.1")
    strict_enforcer.add_target("localhost")

    ok, reason = strict_enforcer.validate_target("127.0.0.1")
    assert ok is False
    assert "blocked" in reason.lower()

    permissive_enforcer = ScopeEnforcer(allow_localhost=True)
    ok2, _ = permissive_enforcer.validate_target("127.0.0.1")
    assert ok2 is True


def test_scope_enforcer_load_from_files():
    enforcer = ScopeEnforcer()

    # 1. Text file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tf:
        tf.write("# Scope comments\n10.10.10.10\n10.10.20.0/24\n*.lab.htb\n")
        tf_name = tf.name

    try:
        count = enforcer.load_from_file(tf_name)
        assert count == 3
        assert enforcer.is_in_scope("10.10.10.10") is True
        assert enforcer.is_in_scope("10.10.20.15") is True
        assert enforcer.is_in_scope("dc.lab.htb") is True
    finally:
        os.remove(tf_name)

    # 2. JSON file
    enforcer_json = ScopeEnforcer()
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as jf:
        json.dump({
            "targets": ["172.16.1.5", "app.local"],
            "excluded": ["172.16.1.1"]
        }, jf)
        jf_name = jf.name

    try:
        count_json = enforcer_json.load_from_file(jf_name)
        assert count_json == 3
        assert enforcer_json.is_in_scope("172.16.1.5") is True
        assert enforcer_json.is_in_scope("app.local") is True
        assert enforcer_json.is_in_scope("172.16.1.1") is False
    finally:
        os.remove(jf_name)


def test_mode_controller_manual_mode_blocks_execution():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")

    mc = ModeController(default_mode=AssignmentMode.MANUAL, scope_enforcer=enforcer)
    authorized, reason = mc.authorize_execution("10.10.10.5")

    assert authorized is False
    assert "MANUAL mode" in reason
    assert "prohibited" in reason.lower()


def test_mode_controller_automated_mode_authorizes():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")

    mc = ModeController(default_mode=AssignmentMode.AUTOMATED, scope_enforcer=enforcer)

    # In scope
    authorized, reason = mc.authorize_execution("10.10.10.5")
    assert authorized is True
    assert "authorized" in reason.lower()

    # Out of scope
    authorized_out, reason_out = mc.authorize_execution("192.168.1.50")
    assert authorized_out is False
    assert "Target violation" in reason_out


def test_mode_controller_per_session_mode():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")

    mc = ModeController(default_mode=AssignmentMode.MANUAL, scope_enforcer=enforcer)

    # Global is MANUAL
    assert mc.is_manual() is True
    assert mc.authorize_execution("10.10.10.5")[0] is False

    # Session A set to AUTOMATED
    mc.set_session_mode("session_lab_1", AssignmentMode.AUTOMATED)
    assert mc.is_automated("session_lab_1") is True
    assert mc.authorize_execution("10.10.10.5", session_id="session_lab_1")[0] is True

    # Session B still falls back to MANUAL
    assert mc.is_manual("session_client_live") is True
    assert mc.authorize_execution("10.10.10.5", session_id="session_client_live")[0] is False


def test_singleton_accessor():
    mc1 = get_mode_controller()
    mc2 = get_mode_controller()
    assert mc1 is mc2
