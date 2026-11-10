"""
tests/test_resume_scope_persistence.py

Regression tests for --resume scope persistence:
  - SessionManager.save_scope() / load_scope() round-trip every Scope field,
    including the C3 independent authorization flags.
  - create_session() now persists "mode" so a resumed autonomous session can be
    dispatched back to run_autonomous_default() instead of silently falling back
    to the manual phase loop.
"""

import dataclasses
import tempfile

from core.scope import Scope
from core.session_manager import SessionManager


def _make_session_manager():
    tmp = tempfile.mkdtemp()
    return SessionManager(base_dir=tmp)


def test_save_and_load_scope_round_trips_all_fields():
    mgr = _make_session_manager()
    mgr.create_session(name="test_session", target="10.0.0.5", scope="['10.0.0.5']", target_type="pentest", mode="automated")

    scope = Scope(
        target="10.0.0.5",
        in_scope=["10.0.0.5", "10.0.0.0/24"],
        out_of_scope=["10.0.0.1"],
        prohibited_tests=["dos"],
        prohibited_tools=["metasploit"],
        time_window="9am-5pm",
        authorization_ref="SOW-123",
        active_testing_allowed=True,
        active_network_allowed=True,
        state_changes_allowed=False,
        recon_only_authorized=False,
    )
    mgr.save_scope(scope)

    persisted = mgr.load_scope()
    assert persisted is not None
    restored = Scope(**persisted)
    assert restored == scope


def test_load_scope_returns_none_for_session_without_saved_scope():
    mgr = _make_session_manager()
    mgr.create_session(name="legacy_session", target="10.0.0.5", scope="[]", target_type="pentest")
    assert mgr.load_scope() is None


def test_create_session_persists_mode():
    mgr = _make_session_manager()
    mgr.create_session(name="auto_session", target="10.0.0.5", scope="[]", target_type="ctf", mode="automated")
    assert mgr.state.get("mode") == "automated"


def test_create_session_defaults_mode_to_manual():
    mgr = _make_session_manager()
    mgr.create_session(name="manual_session", target="10.0.0.5", scope="[]", target_type="ctf")
    assert mgr.state.get("mode") == "manual"


def test_save_scope_persists_across_reload():
    mgr = _make_session_manager()
    mgr.create_session(name="persist_session", target="10.0.0.5", scope="[]", target_type="pentest", mode="automated")
    scope = Scope(target="10.0.0.5", active_testing_allowed=True, state_changes_allowed=True)
    mgr.save_scope(scope)

    # Simulate a fresh --resume: a brand-new SessionManager loading from disk.
    mgr2 = SessionManager(base_dir=mgr.base_dir)
    mgr2.load_session("persist_session")
    persisted = mgr2.load_scope()
    assert persisted is not None
    assert persisted["active_testing_allowed"] is True
    assert persisted["state_changes_allowed"] is True
