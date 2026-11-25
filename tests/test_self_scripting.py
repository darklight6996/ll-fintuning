"""
tests/test_self_scripting.py

Tests for the sandboxed self-scripting primitive. The static safety SCREEN is
the most important surface here -- it must reject dangerous code before it ever
runs -- so it gets the most coverage. The runner's benign-execution path and the
opt-in registration gating are also checked.
"""

import io
import json
import sys
from contextlib import redirect_stdout

import pytest

import core.tools.script_runner as sr
from agents.common.self_scripting import (
    SelfScriptParams, build_self_script, extract_self_script_facts, self_scripting_enabled,
)
from agents.bug_bounty.actions import build_bug_bounty_registry


# --- the static safety screen ---------------------------------------------

@pytest.mark.parametrize("bad", [
    "import subprocess; subprocess.run(['ls'])",
    "import os; os.system('id')",
    "import os; os.popen('whoami')",
    "import os; os.remove('/etc/passwd')",
    "import shutil; shutil.rmtree('/tmp/x')",
    "open('/tmp/evil','w').write('x')",
    "f = open('out.txt', 'a')",
    "__import__('os').system('id')",
    "eval('2+2')",
    "exec('print(1)')",
    "import ctypes",
    "import pty; pty.spawn('/bin/sh')",
    "from pathlib import Path; Path('x').write_text('y')",
])
def test_screen_rejects_dangerous(bad):
    ok, reason = sr.screen(bad)
    assert ok is False and reason


@pytest.mark.parametrize("good", [
    "import urllib.request, json\nr=urllib.request.urlopen('https://x')\nprint(json.dumps({'vuln_signal':{'confirmed':False}}))",
    "import socket, ssl, time, json\nprint('{}')",
    "import http.client\nprint('ok')",
    "data = open('/etc/hostname').read()",  # READ is allowed
])
def test_screen_allows_benign(good):
    ok, reason = sr.screen(good)
    assert ok is True, reason


# --- sandboxed execution ---------------------------------------------------

def _run_capture(code, target="https://x", timeout=10):
    buf = io.StringIO()
    with redirect_stdout(buf):
        sr.run(code, target, timeout)
    return json.loads([l for l in buf.getvalue().splitlines() if l.strip().startswith("{")][-1])["vuln_signal"]


def test_benign_script_emits_its_signal():
    code = ('import json\n'
            'print(json.dumps({"vuln_signal": {"confirmed": False, '
            '"vuln_class": "self_script", "summary": "probe ran"}}))')
    sig = _run_capture(code)
    assert sig["vuln_class"] == "self_script"
    assert sig["summary"] == "probe ran"


def test_dangerous_script_is_rejected_not_run():
    sig = _run_capture("import subprocess; subprocess.run(['id'])")
    assert sig["confirmed"] is False and sig.get("rejected") is True


def test_script_without_signal_is_nonfinding():
    sig = _run_capture("print('just some output, no verdict')")
    assert sig["confirmed"] is False and sig["vuln_class"] == "self_script"


# --- wrapper + opt-in gating ----------------------------------------------

def test_builder_base64_encodes_code():
    import base64
    argv = build_self_script(SelfScriptParams(target="https://x", hypothesis="h", code="print('hi')"))[0]
    i = argv.index("--code-b64")
    assert base64.b64decode(argv[i + 1]).decode() == "print('hi')"


def test_extractor_parses_signal():
    out = '{"vuln_signal": {"confirmed": false, "vuln_class": "self_script"}}'
    assert extract_self_script_facts(out, "")["vuln_signal"]["vuln_class"] == "self_script"


def test_self_script_not_registered_without_optin(monkeypatch):
    monkeypatch.delenv("PENTEST_ENABLE_SELF_SCRIPTING", raising=False)
    assert build_bug_bounty_registry().get("self_script") is None


def test_self_script_registered_with_optin(monkeypatch):
    monkeypatch.setenv("PENTEST_ENABLE_SELF_SCRIPTING", "1")
    assert self_scripting_enabled()
    reg = build_bug_bounty_registry()
    defn = reg.get("self_script")
    assert defn is not None and callable(reg.get_builder(defn.command_builder))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
