"""
tests/test_blind_persistence_boundary.py

BUILD SPEC v1 (2026-11-28), workstream A part 1 -- regression tests for the
blind-mode persistence boundary (audit Defect #7/#23, highest severity).

Before the fix, raw execution results were stored into
autonomous_execution_history BEFORE the blind-scrub block (which only scrubbed
the local Observation stdout/stderr), so the real PortSwigger lab title leaked
into live history AND into both on-disk training-data files for every --blind
run. These tests drive a raw, answer-key-bearing result through the ACTUAL
production write paths -- the history-store boundary helper that run_autonomous
calls, and the real `_write_raw_trajectory` / `_build_chatml_record` writers --
and assert the leak is gone while useful non-sensitive evidence survives.

They are NOT standalone-scrubber tests: `_history_results_for_store` is the exact
method the autonomous loop uses at the store site, and the two writer functions
are imported and called directly from run_portswigger_lab.py.
"""
import os
os.environ.setdefault("PENTEST_ENABLE_SELF_SCRIPTING", "1")

import json

import run_portswigger_lab as rpl
from core.scope import Scope
from agents.bug_bounty.agent import BugBountyAgent

# The exact answer-key shape the audit found leaking (lab-head marker + real
# <title>), plus useful non-sensitive evidence that MUST survive sanitisation.
RAW_LEAKY_STDOUT = (
    "<!DOCTYPE html><html>"
    "<!--LAB_HEAD_START--><head><title>REAL NAME XXE SSRF Lab</title></head><!--LAB_HEAD_END-->"
    "<body>product grid: /product?productId=1 status=200 len=5012</body></html>"
)


def _blind_agent():
    """A real BugBountyAgent with the blind flag forced on. `_blind` is the exact
    attribute run_autonomous and _history_results_for_store read; setting it
    directly mirrors PENTEST_BLIND_LAB=1 without polluting the global env for
    other tests in the session."""
    a = BugBountyAgent(scope=Scope(
        target="h.example", in_scope=["h.example"],
        active_testing_allowed=True, single_host_only=True,
    ))
    a._blind = True
    return a


def _raw_results():
    return [{
        "argv": ["python", "probe.py", "https://h.example/"],
        "stdout": RAW_LEAKY_STDOUT,
        "stderr": "",
        "returncode": 0,
    }]


def test_history_store_boundary_scrubs_blind_results():
    """The autonomous-history storage path (the boundary run_autonomous uses)
    must strip the lab title and markers while keeping useful evidence."""
    a = _blind_agent()
    stored = a._history_results_for_store(_raw_results())
    blob = json.dumps(stored)
    assert "REAL NAME" not in blob
    assert "LAB_HEAD_START" not in blob
    assert "LAB_HEAD_END" not in blob
    assert "<title>REAL NAME" not in blob
    # Useful non-sensitive execution evidence survives sanitisation.
    assert "productId=1" in blob
    assert "status=200" in blob
    assert "len=5012" in blob


def test_non_blind_history_store_is_unchanged():
    """Non-blind runs must be byte-identical (no accidental scrubbing)."""
    a = _blind_agent()
    a._blind = False
    results = _raw_results()
    stored = a._history_results_for_store(results)
    assert stored == results
    assert "REAL NAME" in json.dumps(stored)  # untouched when not blind


def test_raw_trajectory_writer_has_no_leak(tmp_path, monkeypatch):
    """`_write_raw_trajectory` reads the stored history; once that history is
    scrubbed at the boundary, the on-disk raw JSONL carries no answer key."""
    a = _blind_agent()
    history = [{
        "step": 1, "action_id": "endpoint_discovery", "status": "success",
        "results": a._history_results_for_store(_raw_results()),
    }]
    monkeypatch.setattr(rpl, "RAW_LOG_PATH", tmp_path / "raw.jsonl")
    rpl._write_raw_trajectory("https://h.example/", "Lab", {"status": "x"}, history)
    text = (tmp_path / "raw.jsonl").read_text(encoding="utf-8")
    assert "REAL NAME" not in text
    assert "LAB_HEAD_START" not in text
    assert "productId=1" in text  # useful evidence preserved in the trajectory


def test_chatml_writer_has_no_leak():
    """`_build_chatml_record` builds the literal training text from the stored
    history; the scrubbed boundary keeps the answer key out of training data."""
    a = _blind_agent()
    history = [{
        "step": 1, "action_id": "endpoint_discovery", "status": "success",
        "results": a._history_results_for_store(_raw_results()),
    }]
    rec = rpl._build_chatml_record(
        "https://h.example/", "Lab",
        {"status": "terminated_incomplete", "steps_run": 1, "findings_count": 0},
        history, registry=None,
    )
    text = rec["text"]
    assert "REAL NAME" not in text
    assert "LAB_HEAD_START" not in text
    assert "<title>REAL NAME" not in text
    # The scrubbed-but-useful observed content still makes it into the record.
    assert "productId=1" in text


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
