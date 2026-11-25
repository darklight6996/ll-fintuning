#!/usr/bin/env python3
"""
core/tools/ssrf_probe_runner.py

DETERMINISTIC SSRF confirmation engine (--sweep class-sweep spec; also usable
standalone). Same discipline as xxe_probe_runner.py / race_probe_runner.py:
the verdict is computed HERE from objective evidence, never asserted by the
model, and a body/status DIFFERENCE between control and attack is explicitly
NOT sufficient (that is exactly the false-positive pattern already caught and
fixed once in agents/base_agent.py::_confirmation_strength -- this engine is
the proactive version for SSRF specifically).

Contract: the model supplies a CONTROL value (a safe, known-external URL/host
the param should legitimately be able to fetch, e.g. a real public page) and
an ATTACK value (the internal/metadata target under test, e.g.
http://169.254.169.254/latest/meta-data/ or http://127.0.0.1:<port>/...),
either substituted into a URL query param or into a raw request body template
(for JSON/XML-bodied endpoints) via `{VALUE}`.

CONFIRM only if:
  1. The CONTROL request gets a normal response (no transport error) --
     proves the request SHAPE is one the endpoint accepts, so a failure on
     the attack side can't be blamed on malformed input.
  2. The ATTACK response contains `expect_marker` (a content signature that
     can ONLY appear if the server itself fetched the attack target --
     e.g. "ami-id"/"instance-id" for a cloud metadata endpoint, or a known
     string unique to an internal admin page) AND the control response does
     NOT contain it.
A connection-refused/timeout difference with no in-band marker is a `lead`,
never `proven` -- it is suggestive, not evidence that content was fetched.
"""

import argparse
import base64
import json
import sys


def _emit(sig):
    print(json.dumps({"vuln_signal": sig}))


def _substitute(url_or_template, param, value, body_template):
    if body_template:
        return None, body_template.replace("{VALUE}", value)
    import urllib.parse
    parts = urllib.parse.urlsplit(url_or_template)
    q = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    q[param] = value
    new_q = urllib.parse.urlencode(q)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, new_q, parts.fragment)), None


def run(target, method, param, control_value, attack_value, body_template, content_type, expect_marker, timeout):
    try:
        import requests
    except Exception:
        _emit({"confirmed": False, "vuln_class": "ssrf",
               "summary": "requests is not installed; cannot run ssrf_probe."})
        return

    def fire(value):
        url, body = _substitute(target, param, value, body_template)
        kwargs = {"timeout": timeout}
        if body is not None:
            kwargs["data"] = body.encode("utf-8")
            kwargs["headers"] = {"Content-Type": content_type or "application/json"}
            url = target
        try:
            return requests.request(method, url, **kwargs), None
        except Exception as exc:
            return None, str(exc)

    control_resp, control_err = fire(control_value)
    if control_resp is None:
        _emit({"confirmed": False, "vuln_class": "ssrf", "severity": "low",
               "summary": ("Control request (the safe/external value) failed at the transport level -- "
                           "cannot establish a baseline, so an attack-side failure proves nothing."),
               "evidence": f"control exception: {control_err}"})
        return

    attack_resp, attack_err = fire(attack_value)
    if attack_resp is None:
        if not expect_marker:
            _emit({"confirmed": False, "vuln_class": "ssrf", "severity": "low",
                   "summary": ("Attack request failed at the transport level (e.g. connection refused/"
                               "timeout to the internal target) while control succeeded -- SUGGESTIVE of "
                               "server-side fetch behavior differing by destination, but a timing/"
                               "connection difference alone is not in-band evidence of content being "
                               "fetched. Treat as a lead; supply expect_marker for an in-band check, or "
                               "escalate to an OOB/collaborator channel for objective proof."),
                   "evidence": f"control_status={control_resp.status_code}, attack exception: {attack_err}"})
        else:
            _emit({"confirmed": False, "vuln_class": "ssrf", "severity": "low",
                   "summary": "Attack request failed at the transport level; no marker to check.",
                   "evidence": f"attack exception: {attack_err}"})
        return

    if not expect_marker:
        _emit({"confirmed": False, "vuln_class": "ssrf", "severity": "low",
               "summary": ("No expect_marker given -- this engine only proves SSRF via an IN-BAND content "
                           "signature unique to the internal target (e.g. a cloud metadata field, or "
                           "distinctive text from an internal admin page). A bare status/body difference "
                           "is not proof. Supply expect_marker, or escalate to an OOB/collaborator channel."),
               "evidence": f"control_status={control_resp.status_code}, attack_status={attack_resp.status_code}"})
        return

    marker_in_attack = expect_marker in attack_resp.text
    marker_in_control = expect_marker in control_resp.text
    if marker_in_attack and not marker_in_control:
        _emit({"confirmed": True, "vuln_class": "ssrf", "severity": "high",
               "summary": "SSRF CONFIRMED: the attack target's content (expect_marker) was returned by "
                           "the server; the control (safe external target) did not produce it.",
               "evidence": f"expect_marker {expect_marker!r} present in attack response, absent from "
                           f"control response. attack excerpt: "
                           f"{attack_resp.text[max(0, attack_resp.text.find(expect_marker) - 40):][:200]!r}"})
        return

    _emit({"confirmed": False, "vuln_class": "ssrf", "severity": "low",
           "summary": "No fetch-content evidence: expect_marker did not appear in the attack response "
                       "(or appeared in both, which proves nothing).",
           "evidence": f"marker_in_attack={marker_in_attack}, marker_in_control={marker_in_control}"})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deterministic SSRF confirmation engine (control vs attack target).")
    ap.add_argument("--target", required=True)
    ap.add_argument("--method", default="GET")
    ap.add_argument("--param", default=None)
    ap.add_argument("--control-value", required=True)
    ap.add_argument("--attack-value", required=True)
    ap.add_argument("--body-template", default=None,
                     help="Raw body with a {VALUE} placeholder, for JSON/XML-bodied SSRF (overrides --param).")
    ap.add_argument("--content-type", default="application/json")
    ap.add_argument("--expect-marker", default=None)
    ap.add_argument("--timeout", type=int, default=10)
    a = ap.parse_args(argv)
    if not a.body_template and not a.param:
        _emit({"confirmed": False, "vuln_class": "ssrf", "summary": "need either --param or --body-template"})
        return
    run(a.target, a.method, a.param, a.control_value, a.attack_value, a.body_template,
        a.content_type, a.expect_marker, a.timeout)


if __name__ == "__main__":
    main(sys.argv[1:])
