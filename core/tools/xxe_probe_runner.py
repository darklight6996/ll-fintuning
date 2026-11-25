#!/usr/bin/env python3
"""
core/tools/xxe_probe_runner.py

DETERMINISTIC XXE confirmation engine (--sweep class-sweep spec; also usable
standalone). Built directly in response to a live, confirmed false positive
(2026-11-25 blind A/B run 1, logged in work.md): a self_script sent a
form-encoded POST to an XML-only endpoint, got two DIFFERENT XML-parser error
messages for control vs attack (NEITHER reached real app logic), and its
"Response body changed" claim slipped through confirmation-rigor as a PROVEN
SSRF finding. The structural fix for that was reactive (demote on the parser-
error signature strings); this engine is the PROACTIVE fix -- it REFUSES to
call anything confirmed unless:
  1. The CONTROL request itself succeeds (no XML-parser-error signature, a
     real HTTP response) -- i.e. the body format the model sent is one the
     endpoint actually accepts. If the control itself errors, the verdict is
     a `lead` naming the exact fix needed (rewrite as well-formed XML), never
     a diff-based guess.
  2. The ATTACK request's response contains an `expect_marker` the CONTROL's
     response does NOT contain -- an in-band, content-based confirmation
     (file disclosure via an external/internal entity), not a status/body
     "changed" claim. Blind XXE (no in-band marker possible) is explicitly
     NOT provable by this engine and always returns `lead`.

The model supplies the control/attack XML bodies (it has observed the app's
real request shape) and the marker to look for; the engine computes the
verdict OBJECTIVELY from both responses, same pattern as race_probe_runner.py.
"""

import argparse
import base64
import json
import sys

_XML_PARSER_ERROR_SIGNATURES = (
    "saxparseexception", "content is not allowed in prolog",
    "entities are not allowed", "doctype is disallowed",
    "xml parse error", "unexpected end of file",
)


def _emit(sig):
    print(json.dumps({"vuln_signal": sig}))


def _looks_like_parser_error(body):
    low = (body or "").lower()
    return next((sig for sig in _XML_PARSER_ERROR_SIGNATURES if sig in low), None)


def run(target, method, control_body, attack_body, content_type, expect_marker, headers, timeout):
    try:
        import requests
    except Exception:
        _emit({"confirmed": False, "vuln_class": "xxe",
               "summary": "requests is not installed; cannot run xxe_probe."})
        return

    hdrs = dict(headers or {})
    hdrs.setdefault("Content-Type", content_type or "application/xml")
    try:
        control_resp = requests.request(method, target, data=control_body.encode("utf-8"),
                                         headers=hdrs, timeout=timeout)
    except Exception as exc:
        _emit({"confirmed": False, "vuln_class": "xxe", "severity": "low",
               "summary": "Control request errored at the transport level (not a parser/app response).",
               "evidence": f"control request exception: {exc}"})
        return

    control_err = _looks_like_parser_error(control_resp.text)
    if control_err:
        _emit({"confirmed": False, "vuln_class": "xxe", "severity": "low",
               "summary": ("Control request itself hit an XML-parser error -- the request body format is "
                           "wrong for this endpoint, not evidence of anything. Rewrite control_body as "
                           "well-formed XML this endpoint actually accepts (check the real sample request "
                           "from browser_probe) and retry."),
               "evidence": f"control response contained XML-parser-error signature {control_err!r}: "
                           f"{control_resp.text[:300]!r}"})
        return

    try:
        attack_resp = requests.request(method, target, data=attack_body.encode("utf-8"),
                                        headers=hdrs, timeout=timeout)
    except Exception as exc:
        _emit({"confirmed": False, "vuln_class": "xxe", "severity": "low",
               "summary": "Attack request errored at the transport level.",
               "evidence": f"attack request exception: {exc}"})
        return

    if not expect_marker:
        _emit({"confirmed": False, "vuln_class": "xxe", "severity": "low",
               "summary": ("No expect_marker given -- this engine only proves XXE via an IN-BAND content "
                           "signature (e.g. a known file's distinctive content, or a reflected entity "
                           "value). Blind/OOB XXE cannot be objectively confirmed by this engine; supply "
                           "expect_marker, or treat this as a lead pending an out-of-band channel."),
               "evidence": f"control_status={control_resp.status_code}, attack_status={attack_resp.status_code}"})
        return

    marker_in_attack = expect_marker in attack_resp.text
    marker_in_control = expect_marker in control_resp.text
    if marker_in_attack and not marker_in_control:
        _emit({"confirmed": True, "vuln_class": "xxe", "severity": "high",
               "summary": "XXE CONFIRMED: the attack body's external/internal entity resolved and its "
                           "content (expect_marker) appeared in the response; the control (well-formed, "
                           "no entity) did not produce it.",
               "evidence": f"expect_marker {expect_marker!r} present in attack response, absent from "
                           f"control response. attack excerpt: "
                           f"{attack_resp.text[max(0, attack_resp.text.find(expect_marker) - 40):][:200]!r}"})
        return

    _emit({"confirmed": False, "vuln_class": "xxe", "severity": "low",
           "summary": "No entity resolution evidence: expect_marker did not appear in the attack response "
                       "(or appeared in both, which proves nothing).",
           "evidence": f"marker_in_attack={marker_in_attack}, marker_in_control={marker_in_control}"})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deterministic XXE confirmation engine (control vs attack XML body).")
    ap.add_argument("--target", required=True)
    ap.add_argument("--method", default="POST")
    ap.add_argument("--control-body-b64", required=True)
    ap.add_argument("--attack-body-b64", required=True)
    ap.add_argument("--content-type", default="application/xml")
    ap.add_argument("--expect-marker", default=None)
    ap.add_argument("--headers-b64", default=None)
    ap.add_argument("--timeout", type=int, default=10)
    a = ap.parse_args(argv)
    try:
        control_body = base64.b64decode(a.control_body_b64).decode("utf-8", "replace")
        attack_body = base64.b64decode(a.attack_body_b64).decode("utf-8", "replace")
        headers = json.loads(base64.b64decode(a.headers_b64).decode("utf-8", "replace")) if a.headers_b64 else {}
    except Exception as exc:
        _emit({"confirmed": False, "vuln_class": "xxe", "summary": f"bad input: {exc}"})
        return
    run(a.target, a.method, control_body, attack_body, a.content_type, a.expect_marker, headers, a.timeout)


if __name__ == "__main__":
    main(sys.argv[1:])
