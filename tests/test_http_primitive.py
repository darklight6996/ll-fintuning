"""
tests/test_http_primitive.py

Tests for the rich HTTP primitive: the engine's differential oracle
(core/tools/http_probe.py) and the shared action wrapper
(agents/common/http_primitive.py). No real network -- _fetch is monkeypatched
so the oracle logic is tested deterministically.
"""

import io
import json
from contextlib import redirect_stdout

import pytest

import core.tools.http_probe as hp
from agents.common.http_primitive import (
    HttpProbeParams, build_http_probe, extract_http_probe_facts, HTTP_PROBE,
)
from agents.bug_bounty.actions import build_bug_bounty_registry


# --- engine helpers --------------------------------------------------------

def test_has_sql_error_detects_signatures():
    assert hp._has_sql_error("You have an error in your SQL syntax near '")
    assert hp._has_sql_error("Warning: pg_query(): ...")
    assert not hp._has_sql_error("totally normal page content")


def test_set_param_replaces_only_that_param():
    out = hp._set_param("https://x/filter?category=Gifts&page=1", "category", "Gifts'")
    assert "category=Gifts%27" in out and "page=1" in out


# --- engine oracle (network monkeypatched) ---------------------------------

def _run(monkeypatch, fake_fetch, url, max_requests=12):
    monkeypatch.setattr(hp, "_fetch", fake_fetch)
    buf = io.StringIO()
    with redirect_stdout(buf):
        hp.mode_auto(url, timeout=5, max_requests=max_requests)
    lines = [l for l in buf.getvalue().splitlines() if l.strip().startswith("{")]
    assert lines, "engine emitted no JSON signal"
    return json.loads(lines[-1])["vuln_signal"]


def test_auto_detects_sql_error(monkeypatch):
    def fake(url, timeout):
        # The quote probe (%27) triggers a SQL error; baseline is clean.
        if "%27" in url and "AND" not in url:
            return 200, "You have an error in your SQL syntax; check the manual", 0.1
        return 200, "normal product listing page", 0.1
    sig = _run(monkeypatch, fake, "https://x/filter?category=Gifts")
    assert sig["confirmed"] is True and sig["vuln_class"] == "sqli"
    assert sig["parameter"] == "category"


def test_auto_detects_boolean_divergence(monkeypatch):
    base = "AAAA " * 50
    def fake(url, timeout):
        if "1%27%3D%271" in url:   # ' AND '1'='1  (TRUE) -> looks like baseline
            return 200, base, 0.1
        if "1%27%3D%272" in url:   # ' AND '1'='2  (FALSE) -> very different
            return 200, "ZZZZ " * 50, 0.1
        if "%27" in url:           # lone quote -> no SQL error here
            return 200, base, 0.1
        return 200, base, 0.1      # baseline
    sig = _run(monkeypatch, fake, "https://x/filter?category=Gifts")
    assert sig["confirmed"] is True and sig["vuln_class"] == "sqli"


def test_auto_reports_reflection_as_lead_not_finding(monkeypatch):
    def fake(url, timeout):
        # Echo back whatever canary value was set (reflection), no SQL error.
        from urllib.parse import urlsplit, parse_qsl
        q = dict(parse_qsl(urlsplit(url).query))
        return 200, f"search results for {q.get('category','')}", 0.1
    sig = _run(monkeypatch, fake, "https://x/filter?category=Gifts")
    # A reflected canary is a LEAD (XSS surface), never a confirmed finding.
    assert sig["confirmed"] is False
    assert sig.get("lead") is True and sig["vuln_class"] == "xss"


def test_auto_clean_target_is_nonfinding(monkeypatch):
    def fake(url, timeout):
        return 200, "static page, nothing reflects, no errors", 0.1
    sig = _run(monkeypatch, fake, "https://x/filter?category=Gifts")
    assert sig["confirmed"] is False and not sig.get("lead")


def test_auto_no_params_is_clean_nonfinding(monkeypatch):
    def fake(url, timeout):
        return 200, "home", 0.1
    sig = _run(monkeypatch, fake, "https://x/")
    assert sig["confirmed"] is False


# --- shared action wrapper -------------------------------------------------

def test_builder_auto_and_targeted():
    auto = build_http_probe(HttpProbeParams(target="https://x/f?category=Gifts"))[0]
    assert "--mode" in auto and auto[auto.index("--mode") + 1] == "auto"
    tgt = build_http_probe(HttpProbeParams(target="https://x/f?category=Gifts",
                                           focus_param="category", payload="Gifts'"))[0]
    assert tgt[tgt.index("--mode") + 1] == "targeted"
    assert "--param" in tgt and "--payload" in tgt


def test_extractor_parses_signal():
    out = '{"vuln_signal": {"confirmed": true, "severity": "high", "vuln_class": "sqli"}}'
    facts = extract_http_probe_facts(out, "")
    assert facts["vuln_signal"]["confirmed"] is True


def test_extractor_ignores_noise():
    assert extract_http_probe_facts("not json at all\nstill not", "") == {}


def test_http_probe_registered_with_builder():
    reg = build_bug_bounty_registry()
    defn = reg.get("http_probe")
    assert defn is not None
    assert callable(reg.get_builder(defn.command_builder))
    assert defn.state_change == "none"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
