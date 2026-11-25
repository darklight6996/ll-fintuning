#!/usr/bin/env python3
"""
core/tools/nuclei_template_runner.py

MODEL-AUTHORED NUCLEI TEMPLATE engine. Lets the hunter WRITE ITS OWN detection
template (a nuclei YAML) for a specific suspicion the fixed catalog can't express,
then run it against one authorized target. This is the "write its own test scripts
like a nuclei template" capability: the test logic is the YAML the model supplies.

argv in, one JSON line out (a vuln_signal), like the other detection engines, so
it flows through the executor's gates + scope lock unchanged.

SAFETY
  * Detection-only: runs nuclei WITHOUT -code (nuclei does not execute code-protocol
    templates unless -code is passed and the template is signed) and WITHOUT -dast
    fuzzing. Templates are statically screened and REJECTED if they reference the
    code/javascript execution protocols or any interactsh/OOB exfil.
  * Bounded: single target, hard timeout, rate-limited.
  * GRACEFUL: if nuclei isn't installed, emits a clean non-finding note and exits 0.
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Static denylist: reject a model-authored template that tries to execute code or
# phone out of band, regardless of intent. Detection templates never need these.
_DENY = re.compile(r"(?im)^\s*-?\s*(code|javascript)\s*:|\bprotocol\s*:\s*code\b|interactsh|oast|\bexec\b")

_SEV = {"info": "info", "low": "low", "medium": "medium", "high": "high", "critical": "critical"}


def _emit(sig):
    print(json.dumps({"vuln_signal": sig}))


def run(target, template, timeout):
    if not shutil.which("nuclei"):
        _emit({"confirmed": False, "vuln_class": "custom-template",
               "summary": "nuclei is not installed yet; cannot run the authored template. "
                          "Install it at setup time (go install nuclei / apt)."})
        return

    if not template or "id:" not in template or "info:" not in template:
        _emit({"confirmed": False, "vuln_class": "custom-template",
               "summary": "Authored template is not a valid nuclei template (missing id:/info:)."})
        return

    if _DENY.search(template):
        _emit({"confirmed": False, "vuln_class": "custom-template",
               "summary": "Authored template was REJECTED: it references a code-execution or "
                          "out-of-band protocol. Detection templates must be request/matcher only."})
        return

    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="authored_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(template)

        argv = ["nuclei", "-t", tmp, "-u", target, "-jsonl", "-silent", "-no-color",
                "-timeout", "10", "-retries", "1", "-rate-limit", "40",
                "-disable-update-check"]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            _emit({"confirmed": False, "vuln_class": "custom-template",
                   "summary": f"Authored nuclei template timed out after {timeout}s (no match within budget)."})
            return

        matches = []
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            matches.append(obj)

        if not matches:
            _emit({"confirmed": False, "vuln_class": "custom-template",
                   "summary": "Authored template ran cleanly: no match (hypothesis not confirmed by this template)."})
            return

        # Highest-severity match drives the verdict.
        def sev_rank(m):
            s = ((m.get("info") or {}).get("severity") or "info").lower()
            return ["info", "low", "medium", "high", "critical"].index(s) if s in _SEV else 0
        top = sorted(matches, key=sev_rank, reverse=True)[0]
        info = top.get("info") or {}
        sev = (info.get("severity") or "medium").lower()
        if sev not in _SEV:
            sev = "medium"
        tid = top.get("template-id") or info.get("name") or "authored-template"
        matched_at = top.get("matched-at") or top.get("host") or target
        _emit({"confirmed": True, "vuln_class": "custom-template", "severity": sev,
               "summary": f"Authored nuclei template '{tid}' MATCHED at {matched_at} "
                          f"({len(matches)} match(es), severity {sev})."})
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a model-authored nuclei template (detection-only).")
    ap.add_argument("--target", required=True)
    ap.add_argument("--template-b64", required=True)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args(argv)
    try:
        template = base64.b64decode(args.template_b64).decode("utf-8", "replace")
    except Exception as e:
        _emit({"confirmed": False, "vuln_class": "custom-template",
               "summary": f"Could not decode authored template: {str(e)[:120]}"})
        return
    try:
        run(args.target, template, args.timeout)
    except Exception as e:
        _emit({"confirmed": False, "vuln_class": "custom-template",
               "summary": f"nuclei_template_runner internal error: {str(e).splitlines()[0][:150]}"})


if __name__ == "__main__":
    main()
