"""
tests/test_med5_path_hijack.py

Regression tests for MED-5: agents/ctf/command_builders.py::build_path_hijack_argv
must never interpolate AI-supplied strings into a shell command, and must reject
unsafe values outright rather than building a command from them.
"""

import sys
import tempfile
import shutil

import pytest

from agents.ctf.command_builders import build_path_hijack_argv


class _Params:
    def __init__(self, fake_binary_name, target_suid_binary, work_dir="/tmp"):
        self.fake_binary_name = fake_binary_name
        self.target_suid_binary = target_suid_binary
        self.work_dir = work_dir


def test_rejects_shell_injection_in_fake_binary_name():
    params = _Params(fake_binary_name="x; rm -rf /tmp/pwned", target_suid_binary="/usr/bin/sudo")
    with pytest.raises(ValueError):
        build_path_hijack_argv(params)


def test_rejects_command_substitution_in_target():
    params = _Params(fake_binary_name="fortune", target_suid_binary="/usr/bin/$(whoami)")
    with pytest.raises(ValueError):
        build_path_hijack_argv(params)


def test_rejects_path_traversal_in_fake_binary_name():
    params = _Params(fake_binary_name="../../etc/cron.d/evil", target_suid_binary="/usr/bin/sudo")
    with pytest.raises(ValueError):
        build_path_hijack_argv(params)


def test_no_shell_string_in_returned_argv_for_valid_input():
    params = _Params(fake_binary_name="fortune", target_suid_binary="/usr/bin/sudo", work_dir="/tmp")
    argv_list = build_path_hijack_argv(params)

    assert isinstance(argv_list, list) and all(isinstance(a, list) for a in argv_list)
    # None of the returned argv vectors should invoke a shell interpreter at all --
    # that was the actual vulnerability, not merely "injection didn't happen to fire".
    for argv in argv_list:
        assert argv[0] not in ("bash", "sh", "/bin/bash", "/bin/sh")

    # Step 1 writes the fake script via python -c, receiving the path only as argv,
    # never concatenated into the -c source string itself.
    assert argv_list[0][0] == sys.executable
    assert argv_list[0][-1] == "/tmp/fortune"
    assert "/tmp/fortune" not in argv_list[0][2]  # not embedded in the script source

    # Step 3 uses `env VAR=value cmd` argv syntax, not `bash -c "VAR=value cmd"`.
    assert argv_list[2][0] == "env"
    assert argv_list[2][1].startswith("PATH=/tmp:")
    assert argv_list[2][2] == "/usr/bin/sudo"


def test_fake_script_source_contains_real_newline_escapes():
    """The generated `python -c` source, once actually executed, must produce a
    working '#!/bin/sh\\n/bin/sh -p\\n' file -- not just avoid injection. Verified by
    compiling+exec'ing the script string with sys.argv / pathlib.Path monkeypatched
    for the duration (the script's own `import sys, pathlib` would otherwise shadow
    anything passed via exec()'s globals dict). Deliberately not a subprocess + real
    filesystem write test: work_dir/target_suid_binary are inherently Unix paths (this
    action targets SUID privilege escalation on a Linux target), which would make a
    real-path test fail spuriously when pytest itself runs on a Windows host."""
    import sys
    import pathlib

    params = _Params(fake_binary_name="fortune", target_suid_binary="/usr/bin/sudo", work_dir="/tmp")
    argv_list = build_path_hijack_argv(params)
    script = argv_list[0][2]
    fake_path_arg = argv_list[0][3]
    assert fake_path_arg == "/tmp/fortune"

    written = {}

    class _FakePath:
        def __init__(self, p):
            self.p = p

        def write_text(self, content):
            written[self.p] = content

    old_argv, old_path_cls = sys.argv, pathlib.Path
    sys.argv = ["prog", fake_path_arg]
    pathlib.Path = _FakePath
    try:
        exec(compile(script, "<path_hijack_script>", "exec"), {})
    finally:
        sys.argv = old_argv
        pathlib.Path = old_path_cls

    assert written == {"/tmp/fortune": "#!/bin/sh\n/bin/sh -p\n"}
