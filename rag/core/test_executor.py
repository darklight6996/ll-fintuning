"""
test_executor.py

Phase 11 — Offensive Tool Execution Engine unit tests.

Tests:
1. ExecutionResult dataclass properties and ToolResult conversions
2. ToolExecutor dry_run in MANUAL mode (no subprocess spawned)
3. ToolExecutor scope-blocked rejection
4. ToolExecutor live execution with shell=False
5. ToolExecutor non-zero returncode and stderr capture
6. ToolExecutor timeout handling
7. ToolExecutor output truncation limit
8. ToolExecutor missing executable error
9. ToolExecutor async execution
10. ToolExecutor kill switch and abort_all
"""

import asyncio
import os
import sys
import pytest

_HERE = os.path.dirname(__file__)
_CORE_ROOT = os.path.dirname(_HERE)
_RAG_ROOT = os.path.dirname(_CORE_ROOT)
if _RAG_ROOT not in sys.path:
    sys.path.insert(0, _RAG_ROOT)

from core.mode_controller import AssignmentMode, ModeController, ScopeEnforcer
from core.tools.executor import ExecutionResult, ToolExecutor


def test_execution_result_to_tool_result():
    # 1. Blocked
    res_blocked = ExecutionResult(
        command="nmap 192.168.1.1",
        command_args=["nmap", "192.168.1.1"],
        target="192.168.1.1",
        status="blocked",
        error="Target out of scope",
    )
    tr_blocked = res_blocked.to_tool_result("nmap_scan")
    assert tr_blocked.success is False
    assert "out of scope" in tr_blocked.error.lower()

    # 2. Dry run
    res_dry = ExecutionResult(
        command="nmap 10.10.10.5",
        command_args=["nmap", "10.10.10.5"],
        target="10.10.10.5",
        status="dry_run",
        dry_run=True,
    )
    tr_dry = res_dry.to_tool_result("nmap_scan")
    assert tr_dry.success is True
    assert "DRY-RUN" in tr_dry.output or "MANUAL" in tr_dry.output

    # 3. Success
    res_ok = ExecutionResult(
        command="echo hello",
        command_args=["echo", "hello"],
        target="",
        status="success",
        returncode=0,
        stdout="hello\n",
    )
    tr_ok = res_ok.to_tool_result("echo_tool")
    assert tr_ok.success is True
    assert "hello" in tr_ok.output


def test_executor_manual_mode_defaults_to_dry_run():
    # Setup MANUAL mode with in-scope target
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    mc = ModeController(default_mode=AssignmentMode.MANUAL, scope_enforcer=enforcer)

    executor = ToolExecutor(mode_controller=mc)
    res = executor.execute(["nmap", "10.10.10.5"], target="10.10.10.5")

    # In manual mode, it must return dry_run status
    assert res.status == "dry_run"
    assert res.dry_run is True
    assert "[SIMULATION]" in res.stdout


def test_executor_scope_blocked():
    # Setup AUTOMATED mode but target is out of scope
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    mc = ModeController(default_mode=AssignmentMode.AUTOMATED, scope_enforcer=enforcer)

    executor = ToolExecutor(mode_controller=mc)
    res = executor.execute(["nmap", "192.168.1.99"], target="192.168.1.99")

    assert res.status == "blocked"
    assert "Target violation" in res.error


def test_executor_live_execution():
    # Setup AUTOMATED mode with in-scope target
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    mc = ModeController(default_mode=AssignmentMode.AUTOMATED, scope_enforcer=enforcer)

    executor = ToolExecutor(mode_controller=mc)

    # Use python executable to run a safe diagnostic print
    py_cmd = [sys.executable, "-c", "print('EXEC_SUCCESS_1234')"]
    res = executor.execute(py_cmd, target="10.10.10.5")

    assert res.status == "success"
    assert res.returncode == 0
    assert "EXEC_SUCCESS_1234" in res.stdout
    assert res.duration_ms > 0


def test_executor_nonzero_returncode():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    mc = ModeController(default_mode=AssignmentMode.AUTOMATED, scope_enforcer=enforcer)

    executor = ToolExecutor(mode_controller=mc)
    py_cmd = [sys.executable, "-c", "import sys; sys.stderr.write('fatal error'); sys.exit(3)"]
    res = executor.execute(py_cmd, target="10.10.10.5")

    assert res.status == "failed"
    assert res.returncode == 3
    assert "fatal error" in res.stderr


def test_executor_timeout_handling():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    mc = ModeController(default_mode=AssignmentMode.AUTOMATED, scope_enforcer=enforcer)

    executor = ToolExecutor(mode_controller=mc, default_timeout=1)
    py_cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
    res = executor.execute(py_cmd, target="10.10.10.5", timeout=1)

    assert res.status == "timeout"
    assert res.returncode == -1
    assert "timed out" in res.error.lower()


def test_executor_output_capping():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    mc = ModeController(default_mode=AssignmentMode.AUTOMATED, scope_enforcer=enforcer)

    executor = ToolExecutor(mode_controller=mc, max_output_bytes=100)
    # Generate 500 characters
    py_cmd = [sys.executable, "-c", "print('A' * 500)"]
    res = executor.execute(py_cmd, target="10.10.10.5")

    assert res.status == "success"
    assert len(res.stdout) <= 100


def test_executor_missing_binary():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    mc = ModeController(default_mode=AssignmentMode.AUTOMATED, scope_enforcer=enforcer)

    executor = ToolExecutor(mode_controller=mc)
    res = executor.execute(["non_existent_binary_xyz_123"], target="10.10.10.5")

    assert res.status == "error"
    assert res.returncode == -1
    assert "not found" in res.error.lower()


def test_executor_async_execution():
    enforcer = ScopeEnforcer()
    enforcer.add_target("10.10.10.5")
    mc = ModeController(default_mode=AssignmentMode.AUTOMATED, scope_enforcer=enforcer)

    executor = ToolExecutor(mode_controller=mc)
    py_cmd = [sys.executable, "-c", "print('ASYNC_OK')"]

    res = asyncio.run(executor.execute_async(py_cmd, target="10.10.10.5"))

    assert res.status == "success"
    assert "ASYNC_OK" in res.stdout
