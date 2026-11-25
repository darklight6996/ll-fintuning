"""
tests/test_http_probe.py

Tests for core/tools/http_probe.py's typed oracles. The LFI and open-redirect
oracles (2026-11-25 4-repo harvest, asset 6) are re-implemented from
Claude-BugHunter's engine/verifier.py "fact-is-proof" pattern -- a single
deterministic signature match in the replayed response IS the confirming
evidence. `_fetch`/`_fetch_no_redirect` are monkeypatched so no real network
call ever happens in this suite.
"""

import json

import pytest

import core.tools.http_probe as hp


# --- LFI oracle ---------------------------------------------------------------

def test_probe_param_lfi_oracle_confirms_on_passwd_signature(monkeypatch):
    def fake_fetch(url, timeout):
        if "etc%2Fpasswd" in url or "etc/passwd" in url:
            return 200, "root:x:0:0:root:/root:/bin/sh\ndaemon:x:1:1::/usr/sbin:/usr/sbin/nologin", 0.1
        return 200, "<html>benign</html>", 0.1
    monkeypatch.setattr(hp, "_fetch", fake_fetch)
    monkeypatch.setattr(hp, "_fetch_no_redirect", lambda *a, **k: (None, None))

    budget = [20]
    result = hp._probe_param("https://example.com/view?file=report.txt", "file",
                              200, "<html>benign</html>", 5, budget)
    assert result is not None
    assert result["confirmed"] is True
    assert result["vuln_class"] == "lfi"
    assert "root:x:0:0:" in result["evidence"]


def test_probe_param_lfi_oracle_not_triggered_when_baseline_already_has_signature(monkeypatch):
    # If the baseline itself already contains the passwd signature (e.g. a
    # page that legitimately prints it), the LFI check must not fire --
    # otherwise every response would look like a hit.
    base_body = "root:x:0:0:root:/root:/bin/sh"
    calls = []

    def fake_fetch(url, timeout):
        calls.append(url)
        return 200, base_body, 0.1
    monkeypatch.setattr(hp, "_fetch", fake_fetch)
    monkeypatch.setattr(hp, "_fetch_no_redirect", lambda *a, **k: (None, None))

    budget = [20]
    hp._probe_param("https://example.com/view?file=report.txt", "file", 200, base_body, 5, budget)
    assert not any("etc" in c and "passwd" in c for c in calls)


def test_probe_param_lfi_oracle_clean_target_finds_nothing(monkeypatch):
    monkeypatch.setattr(hp, "_fetch", lambda url, timeout: (200, "<html>nothing here</html>", 0.1))
    monkeypatch.setattr(hp, "_fetch_no_redirect", lambda *a, **k: (None, None))

    budget = [20]
    result = hp._probe_param("https://example.com/view?file=report.txt", "file",
                              200, "<html>nothing here</html>", 5, budget)
    assert result is None


# --- open-redirect oracle ------------------------------------------------------

def test_probe_param_open_redirect_oracle_confirms_on_location_host_match(monkeypatch):
    monkeypatch.setattr(hp, "_fetch", lambda url, timeout: (200, "<html>benign</html>", 0.1))

    def fake_no_redirect(url, timeout):
        if hp._REDIRECT_CANARY_HOST in url:
            return 302, f"https://{hp._REDIRECT_CANARY_HOST}/"
        return 200, None
    monkeypatch.setattr(hp, "_fetch_no_redirect", fake_no_redirect)

    budget = [20]
    result = hp._probe_param("https://example.com/go?next=/home", "next",
                              200, "<html>benign</html>", 5, budget)
    assert result is not None
    assert result["confirmed"] is True
    assert result["vuln_class"] == "open-redirect"
    assert hp._REDIRECT_CANARY_HOST in result["evidence"]


def test_probe_param_open_redirect_oracle_not_flagged_when_location_differs(monkeypatch):
    monkeypatch.setattr(hp, "_fetch", lambda url, timeout: (200, "<html>benign</html>", 0.1))
    # App normalizes the redirect back to its own domain -- not vulnerable.
    monkeypatch.setattr(hp, "_fetch_no_redirect", lambda url, timeout: (302, "https://example.com/home"))

    budget = [20]
    result = hp._probe_param("https://example.com/go?next=/home", "next",
                              200, "<html>benign</html>", 5, budget)
    assert result is None


def test_probe_param_open_redirect_oracle_no_redirect_response_not_flagged(monkeypatch):
    monkeypatch.setattr(hp, "_fetch", lambda url, timeout: (200, "<html>benign</html>", 0.1))
    monkeypatch.setattr(hp, "_fetch_no_redirect", lambda url, timeout: (200, None))

    budget = [20]
    result = hp._probe_param("https://example.com/go?next=/home", "next",
                              200, "<html>benign</html>", 5, budget)
    assert result is None


# --- evidence redaction ---------------------------------------------------------

def test_redact_strips_aws_key_and_email_from_evidence(monkeypatch):
    text = "leaked AKIAABCDEFGHIJKLMNOP and contact admin@example.com for help"
    redacted = hp._redact(text)
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "admin@example.com" not in redacted
    assert "[REDACTED-SECRET]" in redacted
    assert "[REDACTED-EMAIL]" in redacted


def test_lfi_evidence_field_is_redacted(monkeypatch):
    # Plant a fake secret right next to the passwd signature to prove the
    # evidence field goes through _redact before being returned.
    leaky_body = "root:x:0:0:root:/root:/bin/sh # admin contact leak@example.com"

    def fake_fetch(url, timeout):
        if "etc%2Fpasswd" in url or "etc/passwd" in url:
            return 200, leaky_body, 0.1
        return 200, "<html>benign</html>", 0.1
    monkeypatch.setattr(hp, "_fetch", fake_fetch)
    monkeypatch.setattr(hp, "_fetch_no_redirect", lambda *a, **k: (None, None))

    budget = [20]
    result = hp._probe_param("https://example.com/view?file=report.txt", "file",
                              200, "<html>benign</html>", 5, budget)
    assert result is not None
    assert "leak@example.com" not in result["evidence"]


# --- fetch_no_redirect plumbing (no network) ------------------------------------

def test_fetch_no_redirect_returns_status_and_location(monkeypatch):
    class _FakeResp:
        status = 302
        headers = {"Location": "https://attacker.example/"}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    class _FakeOpener:
        def open(self, req, timeout=None):
            return _FakeResp()

    monkeypatch.setattr(hp, "_no_redirect_opener", lambda: _FakeOpener())
    status, location = hp._fetch_no_redirect("https://example.com/go?next=x", 5)
    assert status == 302
    assert location == "https://attacker.example/"


def test_fetch_no_redirect_network_failure_returns_none_none(monkeypatch):
    class _FakeOpener:
        def open(self, req, timeout=None):
            raise RuntimeError("network down")

    monkeypatch.setattr(hp, "_no_redirect_opener", lambda: _FakeOpener())
    status, location = hp._fetch_no_redirect("https://example.com/go?next=x", 5)
    assert status is None
    assert location is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
