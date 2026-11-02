"""
core/tools/adapters/base.py

Base Tool Adapter Contract — Phase 11

Provides an abstract foundation for offensive and security assessment tool adapters:
  - Command synthesis with validation
  - Structured output parsing
  - Scope and execution lifecycle integration
  - Automatic OpenAI function parameter schema generation
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

from core.tool_registry import ToolParameter, ToolResult
from core.tools.executor import ExecutionResult, ToolExecutor


class BaseToolAdapter(abc.ABC):
    """Abstract contract for tool integrations."""

    name: str = "base_tool"
    description: str = "Base security tool adapter"
    category: str = "recon"
    safe_preview: bool = True

    def __init__(self, executor: Optional[ToolExecutor] = None):
        self.executor = executor or ToolExecutor()

    @abc.abstractmethod
    def build_command(self, **kwargs) -> List[str]:
        """Construct the CLI command argument list."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_target(self, **kwargs) -> str:
        """Extract primary target host/IP/domain from parameters for scope validation."""
        raise NotImplementedError

    def parse_output(self, stdout: str, stderr: str, returncode: Optional[int]) -> Dict[str, Any]:
        """Parse raw process output into structured dictionary."""
        return {
            "raw_output": stdout or stderr,
            "returncode": returncode,
        }

    def format_summary(self, parsed: Dict[str, Any], raw: ExecutionResult) -> str:
        """Format parsed findings into human-readable summary for LLM prompt."""
        if raw.status == "dry_run":
            return (
                f"### [MANUAL MODE PREVIEW: {self.name.upper()}]\n"
                f"**Command:** `{raw.command}`\n"
                f"**Target:** `{raw.target}`\n"
                f"**Action:** Please execute this command manually in your authorized terminal."
            )
        if raw.status == "blocked":
            return f"❌ Execution Blocked: {raw.error}"

        if not raw.is_success:
            return f"⚠️ {self.name} failed with code {raw.returncode}:\n{raw.stderr or raw.stdout}"

        return f"=== {self.name.upper()} Execution Results ===\n{raw.stdout}"

    def run(
        self,
        dry_run: Optional[bool] = None,
        session_id: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute the tool through the safety-enforced executor."""
        target = self.extract_target(**kwargs)
        cmd_args = self.build_command(**kwargs)

        exec_res = self.executor.execute(
            command=cmd_args,
            target=target,
            timeout=timeout,
            dry_run=dry_run,
            session_id=session_id,
        )

        if exec_res.status == "blocked":
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=exec_res.error or "Execution blocked by scope policy.",
                metadata=exec_res.to_dict(),
            )

        if exec_res.status == "dry_run":
            summary = self.format_summary({}, exec_res)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=summary,
                metadata=exec_res.to_dict(),
            )

        parsed = self.parse_output(exec_res.stdout, exec_res.stderr, exec_res.returncode)
        summary = self.format_summary(parsed, exec_res)

        return ToolResult(
            tool_name=self.name,
            success=exec_res.is_success,
            output=summary,
            error=exec_res.error,
            metadata={**exec_res.to_dict(), "findings": parsed},
        )

    def preview(self, **kwargs) -> str:
        """Generate a dry-run command preview."""
        cmd_args = self.build_command(**kwargs)
        target = self.extract_target(**kwargs)
        cmd_str = " ".join(cmd_args)
        return f"```bash\n{cmd_str}\n```\nTarget: {target}"

    @abc.abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Return parameters definition for ToolRegistry."""
        raise NotImplementedError
