"""
tests/test_suid_audit_mount_pruning.py

Regression test for the suid_audit/find_writable/flag_search TIMEOUTs observed
on the live WSL smoke test on 2026-11-13: these builders' default
search_path="/" ran a plain `find /`, which on WSL also sweeps the slow
DrvFs/9p-mounted Windows drives under /mnt/* (and the /proc, /sys
pseudo-filesystems), pushing the commands past their timeout_seconds. Fixed by
a shared _find_prefix_with_mount_pruning() helper that adds -xdev plus
explicit -prune clauses for /proc, /sys, /mnt when search_path is the default
"/", used by all three full-filesystem-sweep builders.
"""

from agents.ctf.command_builders import (
    build_suid_audit_argv,
    build_find_writable_argv,
    build_flag_search_argv,
)


def _assert_pruned_root_sweep(argv):
    assert argv[0] == "find"
    assert argv[1] == "/"
    assert "-xdev" in argv
    assert "/proc" in argv
    assert "/sys" in argv
    assert "/mnt" in argv
    assert "-prune" in argv
    assert all(isinstance(a, str) for a in argv)


def test_suid_audit_default_search_path_prunes_mnt_proc_sys_and_uses_xdev():
    argv_list = build_suid_audit_argv(None)
    assert len(argv_list) == 1
    argv = argv_list[0]
    _assert_pruned_root_sweep(argv)
    assert "-perm" in argv and "-4000" in argv
    assert "-type" in argv and "f" in argv
    assert "-ls" in argv


def test_suid_audit_custom_non_root_search_path_is_not_pruned():
    params = type("P", (), {"search_path": "/opt/app"})()
    argv_list = build_suid_audit_argv(params)
    argv = argv_list[0]
    assert argv[0] == "find"
    assert argv[1] == "/opt/app"
    assert "-xdev" in argv
    assert "-prune" not in argv
    assert "-perm" in argv and "-4000" in argv


def test_find_writable_default_search_path_prunes_mnt_proc_sys():
    argv_list = build_find_writable_argv(None)
    assert len(argv_list) == 1
    argv = argv_list[0]
    _assert_pruned_root_sweep(argv)
    assert "-type" in argv and "d" in argv
    assert "-perm" in argv and "-0002" in argv


def test_find_writable_custom_search_path_and_type_is_not_pruned():
    params = type("P", (), {"search_path": "/tmp", "find_type": "f"})()
    argv_list = build_find_writable_argv(params)
    argv = argv_list[0]
    assert argv[1] == "/tmp"
    assert "-prune" not in argv
    assert "-type" in argv and "f" in argv


def test_flag_search_default_search_path_prunes_mnt_proc_sys_for_both_finds():
    argv_list = build_flag_search_argv(None)
    assert len(argv_list) == 2
    for argv in argv_list:
        _assert_pruned_root_sweep(argv)
    # First find targets the flag-name pattern, second targets *.txt; both
    # should still carry their own search criteria unmodified.
    assert "-name" in argv_list[0] and "flag*" in argv_list[0]
    assert "-name" in argv_list[1] and "*.txt" in argv_list[1]


def test_flag_search_custom_search_path_is_not_pruned():
    params = type("P", (), {"search_path": "/home/user", "pattern": "ctf_flag*"})()
    argv_list = build_flag_search_argv(params)
    for argv in argv_list:
        assert argv[1] == "/home/user"
        assert "-prune" not in argv
    assert "ctf_flag*" in argv_list[0]
