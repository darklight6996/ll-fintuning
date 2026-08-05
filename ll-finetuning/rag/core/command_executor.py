"""
core/command_executor.py

Purpose
-------
Interacts with the environment safely by preparing, explaining, and executing commands after explicit approval.

Responsibilities
----------------
1. Read-only mode by default: display commands, explain purpose, expected output, and risks.
2. Require approval before executing any command.
3. Safe execution wrapper returning standardized execution results.
"""

from typing import Dict, Any, List, Optional
import subprocess
import shlex


class CommandPreview:
    """
    Structured representation of a command ready for inspection/approval.
    """

    def __init__(
        self,
        command: str,
        purpose: str,
        expected_output: str,
        possible_risks: str,
        step_id: Optional[str] = None
    ):
        self.command = command
        self.purpose = purpose
        self.expected_output = expected_output
        self.possible_risks = possible_risks
        self.step_id = step_id

    def display(self) -> str:
        """Format preview block for user display."""
        lines = [
            "--------------------------------------------------",
            f"Step ID:         {self.step_id or 'N/A'}",
            f"Command:         {self.command}",
            f"Purpose:         {self.purpose}",
            f"Expected Output: {self.expected_output}",
            f"Possible Risks:  {self.possible_risks}",
            "--------------------------------------------------",
            "Approve execution? (Y/N)"
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "command": self.command,
            "purpose": self.purpose,
            "expected_output": self.expected_output,
            "possible_risks": self.possible_risks
        }


class CommandExecutor:
    """
    Executor for safety-gated assessment commands.
    """

    def __init__(self, read_only: bool = True):
        """
        Initialize CommandExecutor.

        Args:
            read_only: If True, commands are only formatted and explained, never run automatically.
        """
        self.read_only = read_only

    def prepare_step_commands(
        self,
        step: Dict[str, Any],
        variable_engine=None
    ) -> List[CommandPreview]:
        """
        Prepare a list of CommandPreview objects from a plan/YAML step dict.

        Args:
            step: Step dictionary containing commands, action, expected_output, etc.
            variable_engine: Optional VariableEngine to perform string substitution.

        Returns:
            List of CommandPreview objects.
        """
        commands = step.get("commands", [])
        if isinstance(commands, str):
            commands = [commands]

        action = step.get("action", step.get("name", "Assessment Command"))
        expected_output = step.get("expected_output", "N/A")
        detection_or_risk = step.get("detection", step.get("risks", "Standard execution traffic"))
        if isinstance(detection_or_risk, dict):
            detection_or_risk = str(detection_or_risk)

        step_id = step.get("id", f"step_{step.get('sequence_number', '0')}")

        previews = []
        for cmd in commands:
            # Substitute variables if engine provided
            final_cmd = cmd
            if variable_engine:
                final_cmd = variable_engine.substitute(cmd)

            preview = CommandPreview(
                command=final_cmd,
                purpose=action,
                expected_output=expected_output,
                possible_risks=detection_or_risk,
                step_id=step_id
            )
            previews.append(preview)

        return previews

    def execute_command(self, command_str: str, approved: bool = False) -> Dict[str, Any]:
        """
        Execute a command if read_only is False and explicit approval is given.

        Args:
            command_str: Shell command to run
            approved: Explicit user authorization flag

        Returns:
            Dict containing execution status, stdout, stderr, and returncode.
        """
        if self.read_only:
            return {
                "status": "skipped",
                "reason": "Executor is in Read-Only Mode.",
                "command": command_str,
                "stdout": "",
                "stderr": "",
                "returncode": None
            }

        if not approved:
            return {
                "status": "declined",
                "reason": "Execution was not approved by user.",
                "command": command_str,
                "stdout": "",
                "stderr": "",
                "returncode": None
            }

        try:
            # Run command shell safely
            result = subprocess.run(
                command_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "command": command_str,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "status": "error",
                "command": command_str,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
