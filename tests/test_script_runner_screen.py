"""
tests/test_script_runner_screen.py

Tests for core/tools/script_runner.py::screen() -- the static pre-execution
denylist that gates every self_script (model- or human-authored) before it
ever runs in the sandboxed subprocess.
"""

import pytest

from core.tools.script_runner import screen


def test_re_compile_is_allowed():
    # BUG FIX (2026-11-25): re.compile(...) is one of the most common things a
    # probe would legitimately do (building a signature regex) and was
    # previously rejected as a false-positive "dynamic compile" escape vector.
    ok, reason = screen('import re\npattern = re.compile(r"foo")\n')
    assert ok, reason


def test_bare_builtin_compile_is_still_rejected():
    ok, reason = screen('code_obj = compile("1+1", "<string>", "eval")\n')
    assert not ok
    assert "compile" in reason


def test_eval_is_rejected():
    ok, reason = screen('eval("1+1")\n')
    assert not ok


def test_exec_is_rejected():
    ok, reason = screen('exec("1+1")\n')
    assert not ok


def test_subprocess_import_is_rejected():
    ok, reason = screen('import subprocess\nsubprocess.run(["ls"])\n')
    assert not ok


def test_file_write_is_rejected():
    ok, reason = screen('open("/tmp/x", "w").write("data")\n')
    assert not ok


def test_benign_requests_probe_is_allowed():
    ok, reason = screen(
        'import requests, json\n'
        'r = requests.get("https://example.com")\n'
        'print(json.dumps({"vuln_signal": {"confirmed": False}}))\n'
    )
    assert ok, reason


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
