"""
tests/test_nuclei_detection_engine.py

Tests for the nuclei detection-engine integration (2026-11-14), the breadth
layer alongside sqlmap's SQLi depth. Same contract as sqlmap:
  - detection-only, no fuzzing/-dast flags reachable from the params model;
  - the fact extractor always emits a vuln_signal (verdict-based engine);
  - a match -> a Finding at the HIGHEST matched template severity; no match ->
    a recorded non-finding.
"""

import pytest

from agents.bug_bounty.actions import NucleiScanParams, build_bug_bounty_registry
from agents.bug_bounty.command_builders import build_nuclei_scan
from agents.bug_bounty.fact_extractors import extract_facts


NUCLEI_HIT = (
    '{"template-id":"CVE-2021-1234","info":{"name":"Example RCE","severity":"critical"},'
    '"matched-at":"https://x/vuln"}\n'
    '{"template-id":"tls-version","info":{"name":"TLS detect","severity":"low"},'
    '"matched-at":"https://x"}\n'
)
NUCLEI_EMPTY = ""   # nuclei -silent with no matches prints nothing


def test_builder_is_detection_only():
    argv = build_nuclei_scan(NucleiScanParams(target="https://x"))[0]
    assert argv[0] == "nuclei"
    joined = " ".join(argv)
    assert "-jsonl" in argv          # parseable verdict
    assert "-dast" not in joined     # no active fuzzing templates
    assert "-fuzz" not in joined


def test_params_model_rejects_unknown_knobs():
    with pytest.raises(Exception):
        NucleiScanParams(target="https://x", templates="/tmp/evil")


def test_extractor_reports_highest_severity_on_match():
    facts = extract_facts("nuclei_scan", NUCLEI_HIT, "")
    sig = facts.get("vuln_signal")
    assert sig and sig["confirmed"] is True
    assert sig["severity"] == "critical"          # highest of critical+low
    assert "2 issue" in sig["summary"] or "matched 2" in sig["summary"]


def test_extractor_no_match_is_nonfinding():
    facts = extract_facts("nuclei_scan", NUCLEI_EMPTY, "")
    sig = facts.get("vuln_signal")
    assert sig and sig["confirmed"] is False


def test_nuclei_scan_registered_with_builder():
    reg = build_bug_bounty_registry()
    defn = reg.get("nuclei_scan")
    assert defn is not None
    assert callable(reg.get_builder(defn.command_builder))
    assert defn.state_change == "none"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
