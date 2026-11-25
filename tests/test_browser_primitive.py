"""
tests/test_browser_primitive.py

Tests for the headless-browser coverage primitive. The real Chromium launch is
validated live (heavy + needs Playwright installed); here we test the pure
harvesting helpers, graceful degradation when Playwright is absent, the shared
action wrapper, and registry wiring -- all deterministic, no browser needed.
"""

import io
import json
import sys
from contextlib import redirect_stdout

import pytest

import core.tools.browser_probe as bp
from agents.common.browser_primitive import (
    BrowserProbeParams, build_browser_probe, extract_browser_facts, BROWSER_PROBE,
)
from agents.bug_bounty.actions import build_bug_bounty_registry


# --- pure harvesting helpers ----------------------------------------------

def test_params_of():
    assert bp._params_of("https://x/f?category=Gifts&page=2") == ["category", "page"]
    assert bp._params_of("https://x/f") == []


def test_same_host():
    assert bp._same_host("https://x/a", "x")
    assert bp._same_host("/relative/path", "x")   # relative -> same host
    assert not bp._same_host("https://evil.com/a", "x")


def test_js_url_mining():
    js = 'fetch("/api/v1/users"); const u="https://api.x/callback"; img.src=`/img/1.png`'
    found = bp._URL_IN_JS.findall(js)
    assert "/api/v1/users" in found
    assert "https://api.x/callback" in found


# --- graceful degradation (no Playwright) ---------------------------------

def test_run_degrades_without_playwright(monkeypatch):
    # Setting the module to None in sys.modules makes `import playwright...` raise.
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        bp.run("https://x/", timeout_ms=1000)
    out = json.loads([l for l in buf.getvalue().splitlines() if l.strip().startswith("{")][-1])
    assert out["vuln_signal"]["confirmed"] is False
    assert "not installed" in out["vuln_signal"]["summary"].lower()


# --- shared action wrapper + wiring ---------------------------------------

def test_builder_argv():
    argv = build_browser_probe(BrowserProbeParams(target="https://x/app"))[0]
    assert argv[-2] == "--url" and argv[-1] == "https://x/app"


def test_extractor_parses_coverage():
    payload = json.dumps({
        "discovered_endpoints": ["https://x/a", "https://x/b"],
        "parameters": ["id", "q"],
        "vuln_signal": {"confirmed": False, "vuln_class": "coverage", "summary": "ok"},
    })
    facts = extract_browser_facts(payload, "")
    assert facts["parameters"] == ["id", "q"]
    assert facts["vuln_signal"]["confirmed"] is False


def test_extractor_ignores_noise():
    assert extract_browser_facts("launching chromium...\n[warn] foo", "") == {}


def test_browser_probe_registered_with_builder():
    reg = build_bug_bounty_registry()
    defn = reg.get("browser_probe")
    assert defn is not None
    assert callable(reg.get_builder(defn.command_builder))
    assert defn.state_change == "none"
    assert defn.phase == "endpoint_discovery"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
