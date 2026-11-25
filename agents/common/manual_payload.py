"""
agents/common/manual_payload.py

Helpers for MANUAL PAYLOAD MODE (--manual-payload on run_portswigger_lab.py):
a human-in-the-loop alternative to the planner's own self_script authoring.
When enabled, BaseAgent.run_autonomous() pauses right before a model-proposed
self_script would run, prints a payload-request brief (vuln_class, endpoint,
exact parameter, observed response, what would confirm it), and reads a
human-pasted PAYLOAD:/CURL:/SCRIPT: from stdin instead -- see
BaseAgent._handle_manual_payload() in agents/base_agent.py.

Design: the human's input is converted into a REPLACEMENT AIActionProposal
(http_probe for PAYLOAD:, self_script for CURL:/SCRIPT:) that then flows
through the EXACT SAME 10-gate execution path, scope checks, and
confirmation-rigor grading as any model-authored proposal -- no new grading
or gate logic, this module only builds proposals. A human-supplied PAYLOAD:
is graded by http_probe's existing typed oracles (same ones a model-authored
proposal would hit); a human-supplied CURL:/SCRIPT: is explicitly tagged
strength="lead" in its own vuln_signal (CURL: has no independent oracle to
verify an arbitrary replayed response against) or, for SCRIPT:, is scrutinized
by the EXISTING _confirmation_strength() self_script path exactly like a
model-authored probe -- "human payload = untrusted candidate, graded lead
until the oracle proves it" is enforced by infrastructure that already
exists, not duplicated here.
"""

import re
import shlex
from typing import Dict, Optional, Tuple

_PARAM_IN_URL_RE = re.compile(r"[?&]([A-Za-z0-9_\-]+)=")
_PARAM_IN_TEXT_RE = re.compile(r"parameter\s+['\"]?([A-Za-z0-9_\-]+)['\"]?", re.I)


def infer_param_name(target: str, hypothesis_text: str) -> Optional[str]:
    """Best-effort: the parameter name a hypothesis is actually about, from
    the target URL's own query string first, then the hypothesis's own
    wording ("parameter 'shopId'"). Returns None if neither matches -- the
    caller must then ask the human to be explicit."""
    m = _PARAM_IN_URL_RE.search(target or "")
    if m:
        return m.group(1)
    m = _PARAM_IN_TEXT_RE.search(hypothesis_text or "")
    if m:
        return m.group(1)
    return None


def parse_curl(curl_line: str) -> Tuple[str, str, Dict[str, str], Optional[str]]:
    """
    Parse a single-line curl command into (method, url, headers, body).
    Supports the common flags: -X/--request, -H/--header (repeatable),
    -d/--data/--data-raw/--data-binary, -b/--cookie. Raises ValueError on
    anything it can't make sense of (no url found) -- the caller treats that
    as "could not parse", never a fabricated best guess.
    """
    tokens = shlex.split(curl_line.strip())
    if tokens and tokens[0] == "curl":
        tokens = tokens[1:]

    method: Optional[str] = None
    headers: Dict[str, str] = {}
    body: Optional[str] = None
    url: Optional[str] = None

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-X", "--request") and i + 1 < len(tokens):
            method = tokens[i + 1].upper()
            i += 2
            continue
        if tok in ("-H", "--header") and i + 1 < len(tokens):
            raw_header = tokens[i + 1]
            if ":" in raw_header:
                name, _, value = raw_header.partition(":")
                headers[name.strip()] = value.strip()
            i += 2
            continue
        if tok in ("-d", "--data", "--data-raw", "--data-binary", "--data-urlencode") and i + 1 < len(tokens):
            body = tokens[i + 1]
            i += 2
            continue
        if tok in ("-b", "--cookie") and i + 1 < len(tokens):
            headers["Cookie"] = tokens[i + 1]
            i += 2
            continue
        if tok in ("-A", "--user-agent") and i + 1 < len(tokens):
            headers["User-Agent"] = tokens[i + 1]
            i += 2
            continue
        if tok.startswith("-"):
            # Unknown flag -- skip it (and a following value-looking token is
            # ambiguous, so only skip the flag itself; most curl flags this
            # parser doesn't special-case are boolean, e.g. -s/-k/-i/-L).
            i += 1
            continue
        if url is None and ("://" in tok or tok.startswith("//")):
            url = tok
        i += 1

    if url is None:
        raise ValueError("no URL found in curl command")
    if method is None:
        method = "POST" if body is not None else "GET"
    return method, url, headers, body


def build_curl_replay_script(method: str, url: str, headers: Dict[str, str], body: Optional[str]) -> str:
    """
    A minimal, self-contained self_script probe that replays EXACTLY the
    human-supplied curl request once and captures the evidence -- no control
    request, no oracle judgment (there is no independent signature to check
    an arbitrary human-chosen request against), so this always reports
    strength="lead": the framework's own confirmation-rigor check
    (_confirmation_strength) demotes anything tagged "lead" regardless of
    what the request's own response looks like.
    """
    headers_repr = repr(headers)
    body_repr = repr(body)
    return (
        "import requests, json, re\n"
        f"resp = requests.request({method!r}, {url!r}, headers={headers_repr}, "
        f"data={body_repr}, timeout=15, allow_redirects=False)\n"
        "body = resp.text[:800]\n"
        "_secret_re = re.compile(r'(?:AKIA|ASIA)[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{10,}\\.[A-Za-z0-9_-]{10,}')\n"
        "evidence = _secret_re.sub('[REDACTED]', body)\n"
        "print(json.dumps({'vuln_signal': {\n"
        "    'confirmed': False, 'strength': 'lead', 'vuln_class': 'manual-curl-replay',\n"
        "    'severity': 'low',\n"
        "    'summary': 'Human-supplied curl command replayed through the gated sandbox; "
        "evidence captured, not independently confirmed by a framework oracle.',\n"
        "    'evidence': f'HTTP {resp.status_code}, Location={resp.headers.get(\"Location\")!r}: {evidence!r}',\n"
        "}}))\n"
    )


def build_solving_curl(method: str, url: str, headers: Optional[Dict[str, str]] = None,
                        body: Optional[str] = None) -> str:
    """Render a copy-pasteable curl command for a confirmed finding's exact
    confirming request."""
    parts = ["curl", "-s"]
    if method and method.upper() != "GET":
        parts += ["-X", method.upper()]
    for name, value in (headers or {}).items():
        parts += ["-H", f"'{name}: {value}'"]
    if body is not None:
        parts += ["-d", f"'{body}'"]
    parts.append(f"'{url}'")
    return " ".join(parts)
