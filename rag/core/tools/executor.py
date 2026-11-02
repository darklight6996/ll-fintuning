"""
core/tools/executor.py

Offensive Tool Execution Engine — Subprocess Runner & Safety Gate

Provides a robust, sandboxed execution wrapper for offensive security tooling:
  - Command execution via subprocess with shell=False (prevents injection)
  - Strict scope enforcement and mode checks via ModeController
  - Dry-run simulation mode (default for manual engagements)
  - Process timeout management and kill-switch termination
  - Standardized ExecutionResult contract convertible to ToolResult
"""

from __future__ import annotations

import asyncio
import datetime
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from core.mode_controller import AssignmentMode, ModeController, get_mode_controller
from core.tool_registry import ToolResult


@dataclass
class ExecutionResult:
    """Standardized result returned by all tool executions."""
    command: str
    command_args: List[str]
    target: str
    status: str                         # "success", "failed", "timeout", "blocked", "dry_run", "error"
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    dry_run: bool = False
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status in ("success", "dry_run") and (self.returncode is None or self.returncode == 0)

    def to_tool_result(self, tool_name: str) -> ToolResult:
        """Convert to ToolResult contract for ToolRegistry integration."""
        if self.status == "blocked":
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=self.error or "Execution blocked by policy.",
                metadata=self.to_dict(),
            )
        elif self.status == "dry_run":
            output_msg = (
                f"### [DRY-RUN / MANUAL MODE PREVIEW]\n"
                f"**Command:** `{self.command}`\n"
                f"**Target:** `{self.target or 'N/A'}`\n"
                f"**Status:** Command prepared for manual execution.\n"
                f"**Operator Instruction:** Automated execution is disabled in this mode. "
                f"Copy and run the command in your authorized terminal."
            )
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output_msg,
                metadata=self.to_dict(),
            )
        elif self.status == "timeout":
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=self.stdout,
                error=f"Command timed out: {self.error}",
                metadata=self.to_dict(),
            )
        elif self.status == "error":
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=self.stdout,
                error=self.error or "Execution error occurred.",
                metadata=self.to_dict(),
            )
        else:
            success = self.returncode == 0
            output_text = self.stdout if self.stdout else (self.stderr if not success else "Command executed successfully.")
            return ToolResult(
                tool_name=tool_name,
                success=success,
                output=output_text,
                error=self.stderr if not success else None,
                metadata=self.to_dict(),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "command_args": self.command_args,
            "target": self.target,
            "status": self.status,
            "returncode": self.returncode,
            "stdout_length": len(self.stdout),
            "stderr_length": len(self.stderr),
            "duration_ms": round(self.duration_ms, 2),
            "dry_run": self.dry_run,
            "timestamp": self.timestamp,
            "error": self.error,
            "metadata": self.metadata,
        }


class ToolExecutor:
    """
    Subprocess execution manager enforcing security policies and timeouts.
    """

    def __init__(
        self,
        mode_controller: Optional[ModeController] = None,
        default_timeout: int = 60,
        max_output_bytes: int = 200_000,
    ):
        self.mode_controller = mode_controller or get_mode_controller()
        self.default_timeout = default_timeout
        self.max_output_bytes = max_output_bytes
        self._active_pids: Dict[int, subprocess.Popen] = {}

    def _split_command(self, cmd: Union[str, List[str]]) -> Tuple[str, List[str]]:
        """Normalize command string or list to (command_str, args_list)."""
        if isinstance(cmd, list):
            cmd_args = [str(x) for x in cmd]
            cmd_str = " ".join(shlex.quote(x) for x in cmd_args)
            return cmd_str, cmd_args
        else:
            cmd_str = cmd.strip()
            # Parse safely into args list for subprocess (POSIX compliant)
            cmd_args = shlex.split(cmd_str, posix=(os.name != "nt"))
            return cmd_str, cmd_args

    def execute(
        self,
        command: Union[str, List[str]],
        target: str = "",
        timeout: Optional[int] = None,
        dry_run: Optional[bool] = None,
        session_id: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute command synchronously with scope and mode enforcement.
        """
        start_time = time.perf_counter()
        cmd_str, cmd_args = self._split_command(command)
        timeout_val = timeout or self.default_timeout

        # Determine effective dry-run setting
        # If dry_run is not explicitly requested, check if mode is MANUAL
        is_manual_mode = self.mode_controller.is_manual(session_id)
        effective_dry_run = dry_run if dry_run is not None else is_manual_mode

        # 1. If not dry_run, check authorization against Scope & Mode
        if not effective_dry_run:
            authorized, reason = self.mode_controller.authorize_execution(target, session_id=session_id)
            if not authorized:
                duration_ms = (time.perf_counter() - start_time) * 1000
                return ExecutionResult(
                    command=cmd_str,
                    command_args=cmd_args,
                    target=target,
                    status="blocked",
                    duration_ms=duration_ms,
                    error=reason,
                    dry_run=False,
                )

        # 2. Dry run preview return
        if effective_dry_run:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                command=cmd_str,
                command_args=cmd_args,
                target=target,
                status="dry_run",
                stdout=f"[SIMULATION] Prepared command: {cmd_str}",
                duration_ms=duration_ms,
                dry_run=True,
            )

        # 3. Live Execution
        proc = None
        try:
            full_env = os.environ.copy()
            if env:
                full_env.update(env)

            proc = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                cwd=cwd,
                env=full_env,
            )
            self._active_pids[proc.pid] = proc

            try:
                stdout_data, stderr_data = proc.communicate(timeout=timeout_val)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_data, stderr_data = proc.communicate()
                duration_ms = (time.perf_counter() - start_time) * 1000
                return ExecutionResult(
                    command=cmd_str,
                    command_args=cmd_args,
                    target=target,
                    status="timeout",
                    returncode=-1,
                    stdout=stdout_data[: self.max_output_bytes],
                    stderr=stderr_data[: self.max_output_bytes],
                    duration_ms=duration_ms,
                    error=f"Execution timed out after {timeout_val} seconds.",
                    dry_run=False,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000
            # Truncate if exceeds bounds
            truncated_stdout = stdout_data[: self.max_output_bytes]
            truncated_stderr = stderr_data[: self.max_output_bytes]

            return ExecutionResult(
                command=cmd_str,
                command_args=cmd_args,
                target=target,
                status="success" if proc.returncode == 0 else "failed",
                returncode=proc.returncode,
                stdout=truncated_stdout,
                stderr=truncated_stderr,
                duration_ms=duration_ms,
                dry_run=False,
            )

        except FileNotFoundError as fnf:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                command=cmd_str,
                command_args=cmd_args,
                target=target,
                status="error",
                returncode=-1,
                error=f"Executable '{cmd_args[0]}' not found on system PATH: {fnf}",
                duration_ms=duration_ms,
                dry_run=False,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                command=cmd_str,
                command_args=cmd_args,
                target=target,
                status="error",
                returncode=-1,
                error=f"Execution error: {type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
                dry_run=False,
            )
        finally:
            if proc and proc.pid in self._active_pids:
                self._active_pids.pop(proc.pid, None)

    async def execute_async(
        self,
        command: Union[str, List[str]],
        target: str = "",
        timeout: Optional[int] = None,
        dry_run: Optional[bool] = None,
        session_id: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute command asynchronously via asyncio.create_subprocess_exec.
        """
        start_time = time.perf_counter()
        cmd_str, cmd_args = self._split_command(command)
        timeout_val = timeout or self.default_timeout

        is_manual_mode = self.mode_controller.is_manual(session_id)
        effective_dry_run = dry_run if dry_run is not None else is_manual_mode

        if not effective_dry_run:
            authorized, reason = self.mode_controller.authorize_execution(target, session_id=session_id)
            if not authorized:
                duration_ms = (time.perf_counter() - start_time) * 1000
                return ExecutionResult(
                    command=cmd_str,
                    command_args=cmd_args,
                    target=target,
                    status="blocked",
                    duration_ms=duration_ms,
                    error=reason,
                    dry_run=False,
                )

        if effective_dry_run:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                command=cmd_str,
                command_args=cmd_args,
                target=target,
                status="dry_run",
                stdout=f"[SIMULATION] Prepared command: {cmd_str}",
                duration_ms=duration_ms,
                dry_run=True,
            )

        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=full_env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_val,
                )
                stdout_str = stdout_bytes.decode(errors="replace")[: self.max_output_bytes]
                stderr_str = stderr_bytes.decode(errors="replace")[: self.max_output_bytes]
                duration_ms = (time.perf_counter() - start_time) * 1000

                return ExecutionResult(
                    command=cmd_str,
                    command_args=cmd_args,
                    target=target,
                    status="success" if proc.returncode == 0 else "failed",
                    returncode=proc.returncode,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    duration_ms=duration_ms,
                    dry_run=False,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                duration_ms = (time.perf_counter() - start_time) * 1000
                return ExecutionResult(
                    command=cmd_str,
                    command_args=cmd_args,
                    target=target,
                    status="timeout",
                    returncode=-1,
                    duration_ms=duration_ms,
                    error=f"Async execution timed out after {timeout_val} seconds.",
                    dry_run=False,
                )
        except FileNotFoundError as fnf:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                command=cmd_str,
                command_args=cmd_args,
                target=target,
                status="error",
                returncode=-1,
                error=f"Executable '{cmd_args[0]}' not found: {fnf}",
                duration_ms=duration_ms,
                dry_run=False,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                command=cmd_str,
                command_args=cmd_args,
                target=target,
                status="error",
                returncode=-1,
                error=f"Async execution error: {type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
                dry_run=False,
            )

    def kill_process(self, pid: int) -> bool:
        """Terminate a tracked active process."""
        proc = self._active_pids.get(pid)
        if proc:
            try:
                proc.kill()
                return True
            except Exception:
                return False
        return False

    def abort_all(self) -> int:
        """Kill all active processes (emergency kill switch)."""
        killed = 0
        for pid, proc in list(self._active_pids.items()):
            try:
                proc.kill()
                killed += 1
            except Exception:
                pass
        self._active_pids.clear()
        return killed
