"""
tests/test_xxe_ssrf_engines.py

Deterministic XXE/SSRF confirmation engines (core/tools/xxe_probe_runner.py,
core/tools/ssrf_probe_runner.py). Same discipline as race_probe_runner: the
verdict is computed from objective control-vs-attack evidence, never a bare
diff. These tests mock `requests` (no real network) and assert the engines
REFUSE to confirm on exactly the false-positive shapes already caught once
(XML-parser-error diffs, SSRF timing/status diffs with no in-band marker).
"""
import json
import sys
import types

import pytest


class _FakeResp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


def _install_fake_requests(monkeypatch, responses):
    """responses: a list consumed in call order, or a callable(method,url,**kw)->_FakeResp."""
    fake = types.ModuleType("requests")
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if callable(responses):
            return responses(method, url, **kwargs)
        return responses[len(calls) - 1]

    fake.request = request
    monkeypatch.setitem(sys.modules, "requests", fake)
    return calls


# --- xxe_probe_runner --------------------------------------------------------

import core.tools.xxe_probe_runner as xxe


def test_xxe_control_parser_error_demotes_to_lead_never_proven(monkeypatch, capsys):
    _install_fake_requests(monkeypatch, [
        _FakeResp("SAXParseException: Content is not allowed in prolog"),
    ])
    xxe.run("https://x/stock", "POST", "<bad", "<bad-attack", "application/xml", "root:", {}, 10)
    out = json.loads(capsys.readouterr().out.strip())
    sig = out["vuln_signal"]
    assert sig["confirmed"] is False
    assert "parser error" in sig["summary"].lower()


def test_xxe_no_expect_marker_is_never_proven(monkeypatch, capsys):
    _install_fake_requests(monkeypatch, [_FakeResp("ok"), _FakeResp("ok")])
    xxe.run("https://x/stock", "POST", "<ok/>", "<ok-attack/>", "application/xml", None, {}, 10)
    sig = json.loads(capsys.readouterr().out.strip())["vuln_signal"]
    assert sig["confirmed"] is False


def test_xxe_marker_in_both_responses_is_not_confirmed(monkeypatch, capsys):
    _install_fake_requests(monkeypatch, [_FakeResp("root:x:0:0"), _FakeResp("root:x:0:0")])
    xxe.run("https://x/stock", "POST", "<ok/>", "<attack/>", "application/xml", "root:x:0:0", {}, 10)
    sig = json.loads(capsys.readouterr().out.strip())["vuln_signal"]
    assert sig["confirmed"] is False


def test_xxe_confirmed_when_control_clean_and_marker_only_in_attack(monkeypatch, capsys):
    _install_fake_requests(monkeypatch, [_FakeResp("no entity here"), _FakeResp("root:x:0:0:root:/root")])
    xxe.run("https://x/stock", "POST", "<ok/>", "<attack/>", "application/xml", "root:x:0:0", {}, 10)
    sig = json.loads(capsys.readouterr().out.strip())["vuln_signal"]
    assert sig["confirmed"] is True
    assert sig["vuln_class"] == "xxe"


# --- ssrf_probe_runner -------------------------------------------------------

import core.tools.ssrf_probe_runner as ssrf


def test_ssrf_connection_error_with_no_marker_is_a_lead_never_proven(monkeypatch, capsys):
    def responder(method, url, **kw):
        if "example.com" in url:
            return _FakeResp("fine")
        raise ConnectionError("refused")
    _install_fake_requests(monkeypatch, responder)
    ssrf.run("https://x/fetch", "GET", "target", "https://example.com/", "http://169.254.169.254/", None,
             "application/json", None, 5)
    sig = json.loads(capsys.readouterr().out.strip())["vuln_signal"]
    assert sig["confirmed"] is False
    assert "lead" not in sig  # no 'strength' key asserting proven -- confirmed flag is what completion reads
    assert sig["confirmed"] is False


def test_ssrf_confirmed_only_with_in_band_marker_not_in_control(monkeypatch, capsys):
    def responder(method, url, **kw):
        if "example.com" in url:
            return _FakeResp("normal external page")
        return _FakeResp("ami-id: i-0abc123")
    _install_fake_requests(monkeypatch, responder)
    ssrf.run("https://x/fetch", "GET", "target", "https://example.com/", "http://169.254.169.254/latest/meta-data/",
              None, "application/json", "ami-id", 5)
    sig = json.loads(capsys.readouterr().out.strip())["vuln_signal"]
    assert sig["confirmed"] is True
    assert sig["vuln_class"] == "ssrf"


def test_ssrf_bare_status_difference_with_no_marker_is_never_proven(monkeypatch, capsys):
    def responder(method, url, **kw):
        return _FakeResp("different body here", status_code=500 if "169.254" in url else 200)
    _install_fake_requests(monkeypatch, responder)
    ssrf.run("https://x/fetch", "GET", "target", "https://example.com/", "http://169.254.169.254/", None,
             "application/json", None, 5)
    sig = json.loads(capsys.readouterr().out.strip())["vuln_signal"]
    assert sig["confirmed"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
