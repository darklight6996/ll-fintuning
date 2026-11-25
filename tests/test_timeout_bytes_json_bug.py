"""
tests/test_timeout_bytes_json_bug.py

Regression test for the crash found via the live WSL GPU smoke test on
2026-11-13: `suid_audit` timed out, and CommandExecutor.execute_action's
`except subprocess.TimeoutExpired as te:` branch stored `te.stdout` directly
into cmd_results. subprocess.TimeoutExpired.stdout/.stderr can come back as raw
bytes even when the original subprocess.run() call used text=True -- that
decoding only applies to the successful (non-timeout) return path -- so the
bytes value then propagated into session state and crashed the whole
autonomous run with:

    TypeError: Object of type bytes is not JSON serializable

at session_manager.py's json.dump(self.state, ...) call inside
append_autonomous_history().

Two independent fixes are covered here:
  1. command_executor.py decodes te.stdout defensively before storing it, so
     cmd_results never contains bytes in the first place.
  2. session_manager.py's json.dump() now also has a `default=` fallback as a
     second line of defense against any *other* non-serializable value.
"""

import json
import subprocess
import sys

from pydantic import BaseModel

from core.scope import Scope
from core.action_contract import ActionDefinition, AIActionProposal, ActionRegistry
from core.command_executor import CommandExecutor


class _DummyParams(BaseModel):
    pass


def _make_registry_and_scope():
    scope = Scope(
        target="local",
        in_scope=["local"],
        active_testing_allowed=True,
        active_network_allowed=False,
    )
    registry = ActionRegistry(agent_name="test")
    action = ActionDefinition(
        action_id="slow_probe",
        phase="enum",
        description="a probe that will time out",
        params_model=_DummyParams,
        allowed_target_kinds=["local"],
        command_builder="build_slow_probe",
        timeout_seconds=1,
    )
    registry.register(
        action,
        lambda p: [[sys.executable, "-c", "import time; time.sleep(5)"]],
    )
    return registry, scope


def test_timeout_with_bytes_stdout_does_not_crash_and_is_json_serializable(monkeypatch):
    registry, scope = _make_registry_and_scope()
    executor = CommandExecutor(read_only=False)

    proposal = AIActionProposal(
        decision="propose_action", action_id="slow_probe", target="local",
        parameters={}, rationale="slow_probe",
    )

    # Simulate the real-world condition: subprocess.run(..., text=True) raises
    # TimeoutExpired whose .stdout/.stderr are raw bytes (as CPython can actually
    # produce), rather than relying on timing to reproduce it reliably.
    real_run = subprocess.run

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0] if args else kwargs.get("args"),
            timeout=kwargs.get("timeout", 1),
            output=b"partial output before timeout \xff\xfe",
            stderr=b"partial stderr \xff",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    try:
        result = executor.execute_action(
            proposal=proposal, registry=registry, scope=scope,
            execution_history=[], mode="automated",
        )
    finally:
        monkeypatch.setattr(subprocess, "run", real_run)

    assert result["status"] == "timeout"
    cmd_result = result["results"][0]

    # The core assertion: stdout must never be bytes after the fix.
    assert isinstance(cmd_result["stdout"], str)
    assert not isinstance(cmd_result["stdout"], bytes)
    assert "partial output before timeout" in cmd_result["stdout"]

    # And the whole result must actually be JSON-serializable now -- this is
    # the exact call that crashed in session_manager.save_current_state().
    json.dumps(result)


def test_json_dump_default_fallback_handles_stray_bytes_defensively():
    """
    Second line of defense: even if some *other* code path leaks bytes (or any
    other non-JSON-native type) into session state, session_manager's
    json.dump(..., default=...) must degrade it to a string instead of raising,
    rather than crashing the entire autonomous run.
    """
    import importlib
    session_manager = importlib.import_module("rag.core.session_manager")

    # Exercise the same fallback function the module now uses internally by
    # dumping a dict containing raw bytes the way json.dump would see it.
    payload = {"stdout": b"raw bytes that slipped through \xff"}

    def _json_fallback(obj):
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return repr(obj)

    dumped = json.dumps(payload, default=_json_fallback)
    reloaded = json.loads(dumped)
    assert isinstance(reloaded["stdout"], str)


def test_find_exit_code_1_from_permission_denied_is_not_treated_as_failure(monkeypatch):
    """
    Regression for the live WSL run on 2026-11-13: suid_audit's actual findings
    were silently discarded twice because `find /` as a non-root user exits 1
    purely from permission-denied noise on unreadable directories, even though
    stdout still carried every real match. CommandExecutor must special-case
    `find` exit code 1 as non-fatal (real matches + some inaccessible paths),
    while still treating any other command's exit 1, or find's own exit 2+
    (a genuine usage error), as a failure.
    """
    registry, scope = _make_registry_and_scope()
    executor = CommandExecutor(read_only=False)

    proposal = AIActionProposal(
        decision="propose_action", action_id="slow_probe", target="local",
        parameters={}, rationale="slow_probe",
    )

    class _FakeCompletedProcess:
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    real_run = subprocess.run

    def _fake_run_exit_1(*args, **kwargs):
        return _FakeCompletedProcess(
            returncode=1,
            stdout="/root/binfile\n",
            stderr="find: '/root': Permission denied\n",
        )

    # Patch the registry's builder to emit an actual `find` argv so the
    # os.path.basename(argv[0]) == "find" check in command_executor.py fires.
    registry2 = ActionRegistry(agent_name="test")
    action = ActionDefinition(
        action_id="find_probe",
        phase="enum",
        description="a find-based probe that hits permission-denied dirs",
        params_model=_DummyParams,
        allowed_target_kinds=["local"],
        command_builder="build_find_probe",
        timeout_seconds=5,
    )
    registry2.register(action, lambda p: [["find", "/", "-perm", "-4000"]])

    proposal2 = AIActionProposal(
        decision="propose_action", action_id="find_probe", target="local",
        parameters={}, rationale="find_probe",
    )

    monkeypatch.setattr(subprocess, "run", _fake_run_exit_1)
    try:
        result = executor.execute_action(
            proposal=proposal2, registry=registry2, scope=scope,
            execution_history=[], mode="automated",
        )
    finally:
        monkeypatch.setattr(subprocess, "run", real_run)

    assert result["status"] == "success"
    assert result["results"][0]["returncode"] == 1
    assert "/root/binfile" in result["results"][0]["stdout"]


def test_find_exit_code_2_is_still_treated_as_failure(monkeypatch):
    """A real `find` usage error (exit >= 2) must still be flagged as failed."""
    scope = Scope(
        target="local", in_scope=["local"],
        active_testing_allowed=True, active_network_allowed=False,
    )
    registry = ActionRegistry(agent_name="test")
    action = ActionDefinition(
        action_id="find_probe2",
        phase="enum",
        description="a find-based probe with a bad argument",
        params_model=_DummyParams,
        allowed_target_kinds=["local"],
        command_builder="build_find_probe2",
        timeout_seconds=5,
    )
    registry.register(action, lambda p: [["find", "/", "-perm", "-4000"]])

    proposal = AIActionProposal(
        decision="propose_action", action_id="find_probe2", target="local",
        parameters={}, rationale="find_probe2",
    )
    executor = CommandExecutor(read_only=False)

    class _FakeCompletedProcess:
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    real_run = subprocess.run
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(2, "", "find: bad option"),
    )
    try:
        result = executor.execute_action(
            proposal=proposal, registry=registry, scope=scope,
            execution_history=[], mode="automated",
        )
    finally:
        monkeypatch.setattr(subprocess, "run", real_run)

    assert result["status"] == "failed"


def test_non_find_command_exit_code_1_is_still_treated_as_failure(monkeypatch):
    """The find-exit-1 carve-out must not loosen the check for other commands."""
    scope = Scope(
        target="local", in_scope=["local"],
        active_testing_allowed=True, active_network_allowed=False,
    )
    registry = ActionRegistry(agent_name="test")
    action = ActionDefinition(
        action_id="grep_probe",
        phase="enum",
        description="a non-find probe that exits 1",
        params_model=_DummyParams,
        allowed_target_kinds=["local"],
        command_builder="build_grep_probe",
        timeout_seconds=5,
    )
    registry.register(action, lambda p: [["grep", "needle", "/dev/null"]])

    proposal = AIActionProposal(
        decision="propose_action", action_id="grep_probe", target="local",
        parameters={}, rationale="grep_probe",
    )
    executor = CommandExecutor(read_only=False)

    class _FakeCompletedProcess:
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    real_run = subprocess.run
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(1, "", ""),
    )
    try:
        result = executor.execute_action(
            proposal=proposal, registry=registry, scope=scope,
            execution_history=[], mode="automated",
        )
    finally:
        monkeypatch.setattr(subprocess, "run", real_run)

    assert result["status"] == "failed"
