"""
tests/test_blind_scrub_nested.py

BUG FIX (2026-11-28): _scrub_lab_facts() was shallow -- it only scrubbed
top-level string/list-of-string values in parsed_facts. A self_script's own
vuln_signal JSON commonly embeds a raw HTML snippet as evidence, nested
several levels deep (vuln_signal -> evidence -> hidden_products[i] ->
snippet). That nested string passed through untouched, leaking the literal
PortSwigger lab marker into what --blind mode is supposed to hide from the
model. Found live during a --sweep + OpenRouter run against a real lab.
"""
import os
os.environ["PENTEST_ENABLE_SELF_SCRIPTING"] = "1"

from core.scope import Scope
from agents.bug_bounty.agent import BugBountyAgent


def _agent():
    return BugBountyAgent(scope=Scope(target="h.example", in_scope=["h.example"],
                                      active_testing_allowed=True, single_host_only=True))


def test_scrub_lab_facts_reaches_nested_dict_list_dict_string():
    """Exact live shape: vuln_signal -> evidence -> hidden_products[list] -> {snippet}."""
    a = _agent()
    facts = {
        "vuln_signal": {
            "confirmed": True,
            "evidence": {
                "hidden_products": [
                    {"id": "16", "snippet": "<!--LAB_HEAD_START-->leaked<!--LAB_HEAD_END-->rest"},
                    {"id": "17", "snippet": "<title>Exploiting XXE to perform SSRF</title>"},
                ],
            },
        },
    }
    scrubbed = a._scrub_lab_facts(facts)
    snippets = [p["snippet"] for p in scrubbed["vuln_signal"]["evidence"]["hidden_products"]]
    assert "LAB_HEAD_START" not in snippets[0]
    assert "LAB_HEAD_END" not in snippets[0]
    assert "xxe" not in snippets[1].lower()
    assert "ssrf" not in snippets[1].lower()


def test_scrub_lab_facts_still_drops_portswigger_links_in_nested_lists():
    a = _agent()
    facts = {"recon": {"links": ["https://portswigger.net/web-security/ssrf", "https://h.example/ok"]}}
    scrubbed = a._scrub_lab_facts(facts)
    assert scrubbed["recon"]["links"] == ["https://h.example/ok"]


def test_scrub_lab_facts_preserves_non_string_leaf_values():
    a = _agent()
    facts = {"vuln_signal": {"confirmed": True, "count": 5, "ratio": 0.5, "tag": None}}
    scrubbed = a._scrub_lab_facts(facts)
    assert scrubbed["vuln_signal"]["confirmed"] is True
    assert scrubbed["vuln_signal"]["count"] == 5
    assert scrubbed["vuln_signal"]["ratio"] == 0.5
    assert scrubbed["vuln_signal"]["tag"] is None


def test_scrub_lab_facts_top_level_shallow_behavior_unchanged():
    """Regression guard: the original shallow cases must still work identically."""
    a = _agent()
    facts = {
        "urls": ["https://portswigger.net/x", "https://h.example/keep"],
        "title": "<title>Exploiting XXE to perform SSRF attacks</title>",
    }
    scrubbed = a._scrub_lab_facts(facts)
    assert scrubbed["urls"] == ["https://h.example/keep"]
    assert "xxe" not in scrubbed["title"].lower()


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
