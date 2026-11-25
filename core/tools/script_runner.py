#!/usr/bin/env python3
"""
core/tools/script_runner.py

THE SANDBOXED SELF-SCRIPTING PRIMITIVE (engine).

Runs a short Python probe/PoC that the AI planner WROTE ITSELF, so the agent can
craft a bespoke confirmation step no fixed tool covers -- the "write its own
scripts to check a parameter / chain a request / prove a bug" capability. This
is the most powerful primitive and, deliberately, the most constrained.

HONEST SECURITY BOUNDARY
------------------------
This is NOT a bulletproof OS sandbox. Real containment needs containers /
namespaces, which can't be assumed on the operator's own box, and in-process
Python sandboxing is escapable. Instead this is DEFENSE-IN-DEPTH appropriate for
code run on the operator's OWN machine against their OWN authorized targets:

  1. STATIC PRE-SCREEN: the code is rejected BEFORE execution if it trips a
     denylist of dangerous operations (process spawning, filesystem writes,
     destructive os/shutil calls, __import__/eval/exec escape vectors, ctypes).
  2. RESOURCE-LIMITED SUBPROCESS: python -I (isolated env; site-packages like
     requests allowed), with RLIMIT_CPU, RLIMIT_AS, and
     RLIMIT_FSIZE=0 (cannot write ANY file), an isolated temp cwd, a minimal
     env, its own session (killed as a group on timeout), and a hard wall-clock
     timeout.
  3. BENIGN-POC FRAMING: it exists to CONFIRM exploitability non-destructively
     (timing, out-of-band callback, unique-token echo), never to weaponize,
     drop a shell, pivot, or exfiltrate.
  4. OPT-IN: the action that invokes this is not even registered unless the
     operator sets PENTEST_ENABLE_SELF_SCRIPTING=1 (see
     agents/common/self_scripting.py), so model-written code never runs casually.

The script is expected to print ONE JSON line: {"vuln_signal": {...}} (confirmed
True only when it actually proved something). Anything else is captured as a
non-finding observation.
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile

# Patterns that cause OUTRIGHT REJECTION before the code ever runs. Conservative
# by design: a rejected benign script is a cheap retry; an executed destructive
# one is not. Network via urllib/http.client/socket IS allowed (probes need it);
# filesystem-write, process-spawn, and dynamic-import/eval escapes are not.
_DENY = [
    (r"\bimport\s+subprocess\b", "spawns processes (subprocess)"),
    (r"\bfrom\s+subprocess\b", "spawns processes (subprocess)"),
    (r"\bimport\s+pty\b", "spawns a terminal (pty)"),
    (r"\bimport\s+ctypes\b", "raw memory/FFI (ctypes)"),
    (r"\bimport\s+cffi\b", "raw FFI (cffi)"),
    (r"\bos\.system\b", "shell exec (os.system)"),
    (r"\bos\.popen\b", "shell exec (os.popen)"),
    (r"\bos\.exec[lv]\w*\b", "process replace (os.exec*)"),
    (r"\bos\.fork\b", "fork (os.fork)"),
    (r"\bos\.spawn\w*\b", "process spawn (os.spawn*)"),
    (r"\bos\.remove\b|\bos\.unlink\b|\bos\.rmdir\b|\bos\.removedirs\b", "deletes files (os.remove/unlink/rmdir)"),
    (r"\bos\.rename\b|\bos\.replace\b", "moves/overwrites files (os.rename/replace)"),
    (r"\bos\.chmod\b|\bos\.chown\b", "changes permissions/ownership"),
    (r"\bos\.truncate\b", "truncates files"),
    (r"\bshutil\.(rmtree|move|copy\w*|make_archive)\b", "filesystem mutation (shutil)"),
    (r"\b__import__\b", "dynamic import (sandbox-escape vector)"),
    (r"\beval\s*\(", "dynamic eval (escape vector)"),
    (r"\bexec\s*\(", "dynamic exec (escape vector)"),
    (r"\bcompile\s*\(", "dynamic compile (escape vector)"),
    # open(...) in any write/append/update mode.
    (r"""open\s*\([^)]*,[^)]*['"][rwax+bt]*[wax+][rwax+bt]*['"]""", "writes files (open in write mode)"),
    (r"\bPath\([^)]*\)\.(write_text|write_bytes|unlink|rmdir|mkdir|rename|replace|chmod)\b", "filesystem mutation (pathlib)"),
]


def screen(code: str):
    """Return (ok: bool, reason: str). ok=False means do NOT execute."""
    for pat, why in _DENY:
        if re.search(pat, code):
            return False, why
    return True, ""


def _emit(signal: dict):
    print(json.dumps({"vuln_signal": signal}))


def _limits():
    """preexec_fn: cap CPU, address space, and file size (0 = no writes)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))               # 10s CPU
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)  # 512MB mem
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))              # cannot write files
        os.setsid()                                                    # own group for clean kill
    except Exception:
        pass


def run(code: str, target: str, timeout: int):
    ok, reason = screen(code)
    if not ok:
        _emit({"confirmed": False, "vuln_class": "self_script", "rejected": True,
               "summary": f"Self-script REJECTED by static safety screen: {reason}. "
                          f"Rewrite the probe without that operation."})
        return

    tmpdir = tempfile.mkdtemp(prefix="llhunt_")
    script_path = os.path.join(tmpdir, "probe.py")
    # Write the script ourselves (the child can't -- RLIMIT_FSIZE=0).
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TARGET": target or "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": tmpdir,
    }
    if os.name == "nt":
        # Windows' resolver (ws2_32.dll) needs SystemRoot to find its DNS/hosts
        # config; without it getaddrinfo() fails for every hostname. This adds
        # no attack surface -- preexec_fn/_limits (the actual resource-limit
        # sandboxing) is POSIX-only and already a no-op on Windows, so this env
        # var is purely a networking compatibility fix, not a sandbox weakening.
        system_root = os.environ.get("SYSTEMROOT") or os.environ.get("SystemRoot")
        if system_root:
            env["SYSTEMROOT"] = system_root
        comspec = os.environ.get("COMSPEC")
        if comspec:
            env["COMSPEC"] = comspec
    try:
        proc = subprocess.run(
            [sys.executable, "-I", script_path],  # -I isolates env; NO -S so the venv's requests/urllib3 import works
            cwd=tmpdir, env=env, capture_output=True, text=True,
            timeout=timeout, preexec_fn=_limits if os.name == "posix" else None,
        )
        out = (proc.stdout or "")
        # Prefer a vuln_signal the script printed itself.
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and "vuln_signal" in obj:
                    print(json.dumps(obj))
                    return
                # Forgiving contract: the model often prints the signal object directly
                # (e.g. {"confirmed": true, "vuln_class": "ssrf", ...}) without the
                # {"vuln_signal": {...}} wrapper. Accept that and wrap it ourselves so a
                # correct verdict is never lost to a nesting slip.
                if isinstance(obj, dict) and "confirmed" in obj:
                    print(json.dumps({"vuln_signal": obj}))
                    return
        # Otherwise capture the run as a non-finding observation.
        tail = (out[-400:] if out else (proc.stderr or "")[-400:]).replace("\n", " ")
        _emit({"confirmed": False, "vuln_class": "self_script",
               "summary": f"Self-script ran against {target!r} (exit {proc.returncode}); it did not emit a "
                          f"confirmed verdict. Output tail: {tail!r}"})
    except subprocess.TimeoutExpired:
        _emit({"confirmed": False, "vuln_class": "self_script",
               "summary": f"Self-script exceeded the {timeout}s wall-clock limit and was killed."})
    except Exception as e:
        _emit({"confirmed": False, "vuln_class": "self_script",
               "summary": f"Self-script harness error: {e}"})
    finally:
        try:
            os.remove(script_path)
            os.rmdir(tmpdir)
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sandboxed runner for model-written Python probes.")
    ap.add_argument("--code-b64", required=True, help="base64-encoded Python source.")
    ap.add_argument("--target", default="", help="Authorized in-scope target the probe addresses.")
    ap.add_argument("--timeout", type=int, default=20)
    args = ap.parse_args(argv)
    try:
        code = base64.b64decode(args.code_b64).decode("utf-8", "replace")
    except Exception as e:
        _emit({"confirmed": False, "vuln_class": "self_script", "summary": f"Could not decode script: {e}"})
        return
    run(code, args.target, max(1, min(args.timeout, 60)))


if __name__ == "__main__":
    main()
