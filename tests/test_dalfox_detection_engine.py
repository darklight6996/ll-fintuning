"""
tests/test_dalfox_detection_engine.py

Tests for the dalfox XSS detection-engine integration (the third engine, after
sqlmap and nuclei). Same contract: detection-only builder, verdict-based
extractor that always emits a vuln_signal, verified XSS -> Finding, reflected-
only -> lead, clean -> non-finding.
"""

import json

import pytest

from agents.bug_bounty.actions import DalfoxScanParams, build_bug_bounty_registry
from agents.bug_bounty.command_builders import build_dalfox_scan
from agents.bug_bounty.fact_extractors import extract_facts


DALFOX_VERIFIED = json.dumps([
    {"type": "V", "severity": "High", "param": "search", "data": "https://x/?search=<script>"},
    {"type": "R", "severity": "Medium", "param": "q", "data": "https://x/?q=test"},
])
DALFOX_REFLECTED_ONLY = json.dumps([
    {"type": "R", "severity": "Low", "param": "q", "data": "https://x/?q=canary"},
])
DALFOX_EMPTY = "[]"


def test_builder_is_detection_only():
    argv = build_dalfox_scan(DalfoxScanParams(target="https://x/?q=1"))[0]
    assert argv[0] == "dalfox" and argv[1] == "url"
    joined = " ".join(argv)
    assert "--format json" in joined
    for forbidden in ("--custom-payload", "--file-write", "-o ", "--output", "--blind"):
        assert forbidden not in joined


def test_params_rejects_unknown_knobs():
    with pytest.raises(Exception):
        DalfoxScanParams(target="https://x", blind="https://evil")


def test_extractor_verified_is_confirmed_high():
    sig = extract_facts("dalfox_scan", DALFOX_VERIFIED, "")["vuln_signal"]
    assert sig["confirmed"] is True and sig["vuln_class"] == "xss"
    assert sig["severity"] == "high"
    assert sig["parameter"] == "search"


def test_extractor_reflected_only_is_lead():
    sig = extract_facts("dalfox_scan", DALFOX_REFLECTED_ONLY, "")["vuln_signal"]
    assert sig["confirmed"] is False and sig.get("lead") is True


def test_extractor_empty_is_nonfinding():
    sig = extract_facts("dalfox_scan", DALFOX_EMPTY, "")["vuln_signal"]
    assert sig["confirmed"] is False and not sig.get("lead")


def test_dalfox_registered_with_builder():
    reg = build_bug_bounty_registry()
    defn = reg.get("dalfox_scan")
    assert defn is not None
    assert callable(reg.get_builder(defn.command_builder))
    assert defn.state_change == "none"
    assert defn.request_volume == "high"   # gated under stealth


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
