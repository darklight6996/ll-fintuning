"""
agents/ctf/actions.py

CTF agent action registry: all approved, safety-governed actions for
the CTFAgent autonomous loop. Actions are low-risk, local-only commands
unless the scope explicitly authorizes active testing.

Each ActionDefinition:
- Has a Pydantic params_model (extra='forbid') for strict validation
- Maps to a named builder function in command_builders.py
- Declares risk_level, network_effect, state_change, requires_active_testing
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.action_contract import ActionDefinition, ActionRegistry


# ---------------------------------------------------------------------------
# Pydantic parameter models — strict, no extra fields allowed
# ---------------------------------------------------------------------------

class EnvironmentReconParams(BaseModel):
    """No parameters needed — runs a fixed battery of recon commands."""
    model_config = {"extra": "forbid"}


class SuidAuditParams(BaseModel):
    """Search path for SUID binaries."""
    model_config = {"extra": "forbid"}
    search_path: str = Field(default="/", description="Root path to search for SUID binaries.")


class StringsAnalysisParams(BaseModel):
    """Run strings on a target binary."""
    model_config = {"extra": "forbid"}
    binary_path: str = Field(description="Absolute path to the binary to analyze.")
    min_length: int = Field(default=8, description="Minimum string length to extract.")


class ReadFileParams(BaseModel):
    """Read contents of a file (typically a flag file)."""
    model_config = {"extra": "forbid"}
    file_path: str = Field(description="Absolute path of the file to read.")


class PathHijackValidationParams(BaseModel):
    """Validate a PATH hijack vector by creating a fake binary."""
    model_config = {"extra": "forbid"}
    fake_binary_name: str = Field(description="Name of the binary to shadow (e.g. 'fortune').")
    target_suid_binary: str = Field(description="Absolute path of the SUID binary that calls the shadowed command.")
    work_dir: str = Field(default="/tmp", description="Directory to place the fake binary.")


class FindWritableParams(BaseModel):
    """Find world-writable directories or files."""
    model_config = {"extra": "forbid"}
    search_path: str = Field(default="/", description="Root path to search.")
    find_type: str = Field(default="d", description="'d' for directories, 'f' for files.")


class KernelExploitCheckParams(BaseModel):
    """Check kernel version for known exploitable conditions."""
    model_config = {"extra": "forbid"}


class ProcessListParams(BaseModel):
    """List running processes for privilege analysis."""
    model_config = {"extra": "forbid"}


class CronJobAuditParams(BaseModel):
    """Audit cron jobs for writeable scripts or misconfigurations."""
    model_config = {"extra": "forbid"}


class FlagSearchParams(BaseModel):
    """Search filesystem for flag files by name pattern."""
    model_config = {"extra": "forbid"}
    pattern: str = Field(default="flag*", description="Glob/find name pattern for flag files.")
    search_path: str = Field(default="/", description="Root path to search.")


# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------

ENVIRONMENT_RECON = ActionDefinition(
    action_id="environment_recon",
    phase="environment_recon",
    description="Run initial recon battery: id, uname -a, pwd, ls -la, hostname, env. Builds baseline situational awareness.",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=EnvironmentReconParams,
    allowed_target_kinds=["local"],
    command_builder="build_environment_recon_argv",
    timeout_seconds=30,
    max_retries=1,
    satisfies_phase=True,
)

SUID_AUDIT = ActionDefinition(
    action_id="suid_audit",
    phase="environment_recon",
    description="Find all SUID-set binaries on the filesystem. Identifies privesc candidates.",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=SuidAuditParams,
    allowed_target_kinds=["local"],
    command_builder="build_suid_audit_argv",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=True,
)

STRINGS_ANALYSIS = ActionDefinition(
    action_id="strings_analysis",
    phase="enumeration",
    description="Run strings on a binary to extract embedded commands, paths, or credentials.",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=StringsAnalysisParams,
    allowed_target_kinds=["local"],
    command_builder="build_strings_analysis_argv",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=False,
)

READ_FILE = ActionDefinition(
    action_id="read_file",
    phase="flag_capture",
    description="Read the contents of a file. Used to retrieve captured flags.",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=ReadFileParams,
    allowed_target_kinds=["local"],
    command_builder="build_read_file_argv",
    timeout_seconds=15,
    max_retries=1,
    satisfies_phase=True,
)

PATH_HIJACK_VALIDATION = ActionDefinition(
    action_id="path_hijack_validation",
    phase="privesc",
    description="Validate a PATH hijack vector: create fake binary in /tmp, prepend /tmp to PATH, execute SUID target.",
    requires_active_testing=True,
    network_effect="none",
    state_change="target_modify",
    risk_level="medium",
    params_model=PathHijackValidationParams,
    allowed_target_kinds=["local"],
    command_builder="build_path_hijack_argv",
    timeout_seconds=30,
    max_retries=2,
    satisfies_phase=True,
)

FIND_WRITABLE = ActionDefinition(
    action_id="find_writable",
    phase="enumeration",
    description="Find world-writable directories or files for potential exploitation.",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=FindWritableParams,
    allowed_target_kinds=["local"],
    command_builder="build_find_writable_argv",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=False,
)

KERNEL_EXPLOIT_CHECK = ActionDefinition(
    action_id="kernel_exploit_check",
    phase="enumeration",
    description="Check kernel version and OS release for known exploitable conditions.",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=KernelExploitCheckParams,
    allowed_target_kinds=["local"],
    command_builder="build_kernel_exploit_check_argv",
    timeout_seconds=20,
    max_retries=1,
    satisfies_phase=False,
)

PROCESS_LIST = ActionDefinition(
    action_id="process_list",
    phase="enumeration",
    description="List running processes to identify services running as root or other users.",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=ProcessListParams,
    allowed_target_kinds=["local"],
    command_builder="build_process_list_argv",
    timeout_seconds=15,
    max_retries=1,
    satisfies_phase=False,
)

CRON_JOB_AUDIT = ActionDefinition(
    action_id="cron_job_audit",
    phase="enumeration",
    description="Audit cron jobs for writeable scripts or privilege escalation misconfigurations.",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=CronJobAuditParams,
    allowed_target_kinds=["local"],
    command_builder="build_cron_job_audit_argv",
    timeout_seconds=20,
    max_retries=1,
    satisfies_phase=False,
)

FLAG_SEARCH = ActionDefinition(
    action_id="flag_search",
    phase="flag_capture",
    description="Search the filesystem for flag files matching common CTF naming patterns.",
    requires_active_testing=False,
    network_effect="none",
    state_change="none",
    risk_level="low",
    params_model=FlagSearchParams,
    allowed_target_kinds=["local"],
    command_builder="build_flag_search_argv",
    timeout_seconds=60,
    max_retries=2,
    satisfies_phase=False,
)


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def build_ctf_registry() -> ActionRegistry:
    """
    Build and return the CTF agent's ActionRegistry pre-loaded with all
    authorized actions and their command builders.
    """
    from agents.ctf.command_builders import (
        build_environment_recon_argv,
        build_suid_audit_argv,
        build_strings_analysis_argv,
        build_read_file_argv,
        build_path_hijack_argv,
        build_find_writable_argv,
        build_kernel_exploit_check_argv,
        build_process_list_argv,
        build_cron_job_audit_argv,
        build_flag_search_argv,
    )

    registry = ActionRegistry(agent_name="ctf")

    registry.register(ENVIRONMENT_RECON, build_environment_recon_argv)
    registry.register(SUID_AUDIT, build_suid_audit_argv)
    registry.register(STRINGS_ANALYSIS, build_strings_analysis_argv)
    registry.register(READ_FILE, build_read_file_argv)
    registry.register(PATH_HIJACK_VALIDATION, build_path_hijack_argv)
    registry.register(FIND_WRITABLE, build_find_writable_argv)
    registry.register(KERNEL_EXPLOIT_CHECK, build_kernel_exploit_check_argv)
    registry.register(PROCESS_LIST, build_process_list_argv)
    registry.register(CRON_JOB_AUDIT, build_cron_job_audit_argv)
    registry.register(FLAG_SEARCH, build_flag_search_argv)

    return registry
