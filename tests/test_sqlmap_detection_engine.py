"""
tests/test_sqlmap_detection_engine.py

Tests for the first specialist detection-engine integration (sqlmap) in the
bug_bounty agent (2026-11-14). Covers the three things that make it a real
"find the bug, read the tool output, reason pass/fail" capability rather than a
blind payload cannon:

  1. The command builder is detection-only and non-destructive (hardcoded safe
     flags; no data-dumping/modifying options reachable from the params model).
  2. The fact extractor turns sqlmap's text into a structured confirmed/refuted
     verdict (and emits NOTHING on unrecognized output -- never a fabricated one).
  3. The verdict-aware evidence promotion records a real HIGH-severity Finding
     on a CONFIRMED verdict and NO finding on a CLEAN scan -- the clean scan is
     a recorded observation (progress), not a false positive.
"""

import pytest

from agents.bug_bounty.actions import SqliDeepScanParams, build_bug_bounty_registry
from agents.bug_bounty.command_builders import build_sqli_deep_scan
from agents.bug_bounty.fact_extractors import extract_facts


# --- Sample sqlmap outputs -------------------------------------------------

SQLMAP_CONFIRMED = """
sqlmap identified the following injection point(s) with a total of 72 HTTP(s) requests:
---
Parameter: category (GET)
    Type: boolean-based blind
    Title: AND boolean-based blind - WHERE or HAVING clause
    Payload: category=Gifts' AND 1=1-- -

    Type: time-based blind
    Title: MySQL >= 5.0.12 time-based blind
    Payload: category=Gifts' AND SLEEP(5)-- -
---
back-end DBMS: MySQL >= 5.0.12
"""

SQLMAP_CLEAN = """
[INFO] testing connection to the target URL
[INFO] testing if the target URL content is stable
[WARNING] GET parameter 'category' does not seem to be injectable
[CRITICAL] all tested parameters do not appear to be injectable.
"""

SQLMAP_GARBAGE = """
[CRITICAL] unable to connect to the target URL. sqlmap is going to retry the request(s)
[CRITICAL] connection timed out to the target URL
"""


# --- 1. Command builder safety --------------------------------------------

def test_builder_is_detection_only_and_non_destructive():
    argv = build_sqli_deep_scan(SqliDeepScanParams(target="https://x/filter?category=Gifts"))[0]
    assert argv[0] == "sqlmap"
    assert "--batch" in argv  # never hangs interactively
    joined = " ".join(argv)
    # No data-dumping / data-modifying / shell options anywhere.
    for forbidden in ("--dump", "--dump-all", "--os-shell", "--os-cmd",
                      "--sql-shell", "--file-read", "--file-write", "--passwords"):
        assert forbidden not in joined, f"unsafe flag present: {forbidden}"
    # Stacked-query technique 'S' (the data-modifying one) must be excluded.
    tech = [a for a in argv if a.startswith("--technique=")]
    assert tech and "S" not in tech[0].split("=", 1)[1], tech


def test_builder_focuses_single_param_when_given():
    argv = build_sqli_deep_scan(
        SqliDeepScanParams(target="https://x/f?category=Gifts", test_parameter="category")
    )[0]
    assert "-p" in argv and argv[argv.index("-p") + 1] == "category"


def test_params_model_rejects_unknown_knobs():
    # extra="forbid": the planner cannot sneak in --risk/--dump via params.
    with pytest.raises(Exception):
        SqliDeepScanParams(target="https://x", risk="3")


# --- 2. Fact extractor verdicts -------------------------------------------

def test_extractor_confirmed_verdict():
    facts = extract_facts("sqli_deep_scan", SQLMAP_CONFIRMED, "")
    sig = facts.get("vuln_signal")
    assert sig and sig["confirmed"] is True
    assert sig["severity"] == "high"
    assert sig["vuln_class"] == "sqli"
    assert sig["parameter"] == "category (GET)" or "category" in (sig["parameter"] or "")
    assert any("time-based" in t for t in (sig["techniques"] or []))


def test_extractor_clean_verdict():
    facts = extract_facts("sqli_deep_scan", SQLMAP_CLEAN, "")
    sig = facts.get("vuln_signal")
    assert sig and sig["confirmed"] is False


def test_extractor_unrecognized_output_is_inconclusive_nonfinding():
    # A verdict-based engine ALWAYS emits a signal so it never falls through to
    # the generic exit-0 milestone promotion. Unrecognized output -> confirmed
    # False (inconclusive), which produces NO finding -- never a fabricated one.
    facts = extract_facts("sqli_deep_scan", SQLMAP_GARBAGE, "")
    sig = facts.get("vuln_signal")
    assert sig is not None and sig["confirmed"] is False
    assert sig.get("inconclusive") is True


def test_extractor_no_parameters_found_is_nonfinding():
    # The exact bare-root case from the first live run: sqlmap found no param to
    # test. Must be a confirmed=False non-finding, not a generic info milestone.
    no_params = ("[CRITICAL] no parameter(s) found for testing in the provided "
                 "data. You are advised to rerun with '--crawl=2'")
    facts = extract_facts("sqli_deep_scan", no_params, "")
    sig = facts.get("vuln_signal")
    assert sig is not None and sig["confirmed"] is False


# --- 3. Verdict-aware evidence promotion ----------------------------------

def _promotion_harness(parsed_facts, stdout):
    """Run _promote_evidence_if_warranted with a single successful observation
    carrying the given parsed_facts, and return the resulting EvidenceStore."""
    from core.scope import Scope
    from core.session_manager import SessionManager
    from core.investigation import EvidenceStore, Observation
    from agents.bug_bounty.agent import BugBountyAgent
    from agents.bug_bounty.actions import SQLI_DEEP_SCAN

    scope = Scope(target="https://x.web-security-academy.net/", in_scope=["x.web-security-academy.net"],
                  active_testing_allowed=True, active_network_allowed=True)
    agent = BugBountyAgent(scope=scope, mode="automated", session_manager=SessionManager())

    store = EvidenceStore()
    obs = Observation(
        action_id="sqli_deep_scan",
        argv=["sqlmap", "-u", scope.target, "--batch"],
        raw_stdout=stdout,
        raw_stderr="",
        exit_code=0,
        parsed_facts=parsed_facts,
        phase="vuln_validation",
    )
    store.add_observation(obs)

    class _P:
        target = scope.target
    agent._promote_evidence_if_warranted(
        defn=SQLI_DEEP_SCAN, proposal=_P(), observations=[obs], evidence_store=store
    )
    return store


def test_confirmed_scan_records_high_severity_finding():
    facts = extract_facts("sqli_deep_scan", SQLMAP_CONFIRMED, "")
    store = _promotion_harness(facts, SQLMAP_CONFIRMED)
    assert len(store.findings) == 1
    finding = next(iter(store.findings.values()))
    assert finding.severity == "high"


def test_clean_scan_records_no_finding():
    facts = extract_facts("sqli_deep_scan", SQLMAP_CLEAN, "")
    store = _promotion_harness(facts, SQLMAP_CLEAN)
    assert len(store.findings) == 0  # clean scan is a non-finding, not a false positive


# --- registry wiring -------------------------------------------------------

def test_sqli_deep_scan_registered_with_builder():
    reg = build_bug_bounty_registry()
    defn = reg.get("sqli_deep_scan")
    assert defn is not None
    assert callable(reg.get_builder(defn.command_builder))
    assert defn.state_change == "none"  # detection-only


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
