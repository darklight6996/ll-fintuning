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

REBUILT 2026-11-25 after operator feedback on the first version: that version
asked the human to blindly invent a test value against a bare parameter name,
with no explanation of WHY it's exploitable or HOW to actually craft a
payload for it -- backwards from what was asked. build_payload_brief() now
leads with the CONFIRMED exploitation technique (plain-language, from
TECHNIQUE_DESCRIPTIONS, keyed on the matching open Hypothesis's vuln_class),
the exact request to send with the tamper point marked, and payload-crafting
guidance (reusing core/payload_pack.py's existing per-class checklist/seed
payloads) -- all BEFORE asking the operator for anything.
"""

import re
import shlex
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit

_PARAM_IN_URL_RE = re.compile(r"[?&]([A-Za-z0-9_\-]+)=")
# Quotes required (not optional) -- without them "parameter named here" would
# falsely capture "named" as the parameter name.
_PARAM_IN_TEXT_RE = re.compile(r"parameter\s+['\"]([A-Za-z0-9_\-]+)['\"]", re.I)

# Plain-language exploitation-technique explanations per canonical vuln_class
# (2026-11-25, rebuilt after operator feedback: the first version of manual-
# payload mode asked the human to blindly invent test values with no guidance
# on WHY a parameter is exploitable or HOW to actually weaponize it -- this is
# the "confirm the technique, then guide the human" half the operator asked
# for). Deliberately plain-language, not a payload itself -- the point is to
# teach the mechanism so the human can craft (or recognize when to ask for) a
# real payload, not to hand them one blind either.
TECHNIQUE_DESCRIPTIONS: Dict[str, str] = {
    "ssrf": (
        "SSRF: the SERVER itself makes an outbound request using (all or part of) this "
        "parameter's value. Substituting an internal-network address, a cloud metadata "
        "IP, or another protocol/port makes the server fetch something on your behalf "
        "that you could never reach directly. If the request body this parameter lives "
        "in is XML, also consider SSRF-VIA-XXE: define an external entity in a DOCTYPE "
        "and reference it from a field, so the XML PARSER (not your own injected value) "
        "makes the outbound fetch -- this bypasses URL-shaped-value filters that only "
        "look at an obviously-a-URL field."
    ),
    "xxe": (
        "XXE: the request body is XML and the parser resolves external entities. "
        "Define `<!DOCTYPE x [<!ENTITY e SYSTEM \"...\">]>` and reference `&e;` inside a "
        "field value -- the parser fetches/reads whatever the entity's SYSTEM URI points "
        "to (a local file via file://, or an internal/external URL via http://, "
        "which is itself an SSRF primitive mediated through the parser)."
    ),
    "sqli": (
        "SQL injection: this parameter's value reaches a SQL query without being "
        "properly parameterized. A single quote that breaks the query syntax (causing a "
        "DB error) confirms the primitive; from there, boolean/time-based blind "
        "techniques or a UNION-based extraction are both viable depending on whether "
        "the response reflects query output."
    ),
    "idor": (
        "IDOR / broken access control: this parameter is a raw object reference (an id, "
        "a UUID, a filename) with no server-side ownership check. Swapping it for "
        "another user's/tenant's identifier should return data that does not belong to "
        "the current session -- the server trusts the CLIENT-supplied id instead of "
        "re-deriving it from the authenticated session."
    ),
    "ssti": (
        "Server-side template injection: this value is rendered through a template "
        "engine rather than inserted as plain text. A non-destructive arithmetic probe "
        "(e.g. a multiplication the template engine itself would evaluate) confirms the "
        "primitive before escalating toward the engine's known RCE gadget chain."
    ),
    "rce": (
        "Command/code injection: this value reaches a shell command or a deserialization "
        "sink. A benign, observable side effect (a distinctive timing delay, an "
        "out-of-band callback) confirms execution without needing a destructive payload."
    ),
    "redirect": (
        "Open redirect: this parameter controls a Location-header or client-side "
        "redirect target with no allowlist check. Pointing it at an external host and "
        "confirming the server/page actually redirects THERE (not back to its own "
        "domain) is the whole primitive -- useful chained into an OAuth/SSO flow."
    ),
    "lfi": (
        "Local file inclusion / path traversal: this value is used to build a filesystem "
        "path server-side with insufficient sanitization. Traversal sequences "
        "(`../`) walking up to a well-known file (e.g. /etc/passwd) confirm the "
        "primitive via a recognizable signature in the response."
    ),
    "auth": (
        "Authentication/authorization bypass: a check that should gate this "
        "request is missing, trusting a client-controlled value (a header, a cookie "
        "flag, a role field) instead of independently verifying the session server-side."
    ),
    "oauth": (
        "OAuth/OIDC flow flaw: a redirect_uri, state, or token-type check is "
        "missing or too permissive -- controlling the redirect target or forging/"
        "reusing a token across its intended scope is the usual confirming step."
    ),
}

_DEFAULT_TECHNIQUE = (
    "No specific technique template for this vuln_class yet -- reason from the "
    "hypothesis text itself: what does the server DO with this parameter's value, "
    "and what would a value need to achieve to prove that assumption is fragile?"
)


def describe_technique(vuln_class: Optional[str]) -> str:
    """Plain-language exploitation-technique explanation for a vuln_class, or
    a generic reasoning prompt if there's no template for it yet."""
    if not vuln_class:
        return _DEFAULT_TECHNIQUE
    return TECHNIQUE_DESCRIPTIONS.get(vuln_class.strip().lower(), _DEFAULT_TECHNIQUE)


def render_request_template(target: str, param: Optional[str]) -> str:
    """A minimal, honest reconstruction of the confirming request: method/URL
    with the exploitable parameter marked. GET-only (this framework's own
    request-shape facts -- forms/content-type -- aren't threaded through to
    this helper yet; if the real request is POST/body-based, say so and let
    the human correct it via CURL:/SCRIPT: instead)."""
    parts = urlsplit(target)
    path = parts.path or "/"
    query = f"?{parts.query}" if parts.query else ""
    lines = [f"GET {path}{query} HTTP/1.1", f"Host: {parts.netloc}"]
    if param:
        lines.append(f"Tamper point: the '{param}' value above -- replace it with your crafted payload.")
    else:
        lines.append("Tamper point: (parameter not yet identified -- name it explicitly in your PAYLOAD:)")
    lines.append(
        "If the real request is actually POST / a different content-type (form, JSON, XML) -- "
        "check the most recent observation above for the real shape, then use CURL: or SCRIPT: "
        "instead of PAYLOAD: to get the method/body/headers right."
    )
    return "\n".join(lines)


def build_payload_brief(target: str, param: Optional[str], vuln_class: Optional[str],
                         hypothesis_text: str, recent_excerpt: str) -> str:
    """
    Assemble the full manual-payload brief: confirmed technique, the exact
    request to send with the tamper point marked, and payload-crafting
    guidance (reusing core/payload_pack.py's existing per-class checklist +
    seed payloads -- same methodology a model-authored self_script would be
    steered toward, not duplicated here). This is what BaseAgent.
    _handle_manual_payload() prints before reading the operator's input.
    """
    from core.payload_pack import build_attack_checklist, normalize_vuln_class, SEED_PAYLOADS

    canonical = normalize_vuln_class(vuln_class) if vuln_class else None
    lines = [
        f"Vulnerability class:   {canonical or '(not yet classified -- see hypothesis below)'}",
        f"Exploitable parameter:  {param or '(could not infer -- name it explicitly in your PAYLOAD:)'}",
        f"Hypothesis:             {hypothesis_text or '(none given)'}",
        "",
        "Exploitation technique:",
        f"  {describe_technique(canonical)}",
        "",
        "Request to send (the confirming probe):",
    ]
    lines.extend(f"  {line}" for line in render_request_template(target, param).splitlines())
    lines.append("")
    lines.append("Payload-crafting guidance:")
    if canonical:
        for item in build_attack_checklist(canonical)[:5]:
            lines.append(f"  - {item}")
        seeds = SEED_PAYLOADS.get(canonical, [])
        if seeds:
            lines.append(f"  Example seed payloads for this class (starting points, not a final answer): "
                          f"{', '.join(seeds[:4])}")
    else:
        lines.append("  (no vuln_class yet -- craft a payload that would distinguish your hypothesis "
                      "being TRUE from it being FALSE, same as the checklist above.)")
    lines.append("")
    lines.append(f"Observed response (most recent observation, truncated):\n  {recent_excerpt}")
    return "\n".join(lines)


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
