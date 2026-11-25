"""
tests/test_class_sweep.py

--sweep mode: context gate, surface characterization, coverage ledger
bookkeeping, and BaseAgent._update_sweep_coverage wiring.
"""
import os
os.environ["PENTEST_ENABLE_SELF_SCRIPTING"] = "1"

from core.scope import Scope
from core.investigation import EvidenceStore, Observation
from core.tools.class_gate import characterize_tags, gate_classes_for_tags, gate_classes_for_entry
from agents.common.surface_characterize import characterize_observation, CLASS_TO_ENGINE
from agents.bug_bounty.agent import BugBountyAgent


def _agent():
    return BugBountyAgent(scope=Scope(target="h.example", in_scope=["h.example"],
                                      active_testing_allowed=True, single_host_only=True))


# --- class_gate ------------------------------------------------------------

def test_gate_fires_xxe_on_xml_content_type():
    assert "xxe" in gate_classes_for_tags(characterize_tags({"content_type": "application/xml"}))


def test_gate_fires_ssrf_on_url_shaped_value():
    assert "ssrf" in gate_classes_for_tags(characterize_tags({"value_url_shaped": True}))


def test_gate_fires_lfi_on_path_shaped_value():
    assert "lfi" in gate_classes_for_tags(characterize_tags({"value_path_shaped": True}))


def test_gate_fires_sqli_on_error_on_quote():
    assert "sqli" in gate_classes_for_tags(characterize_tags({"errors_on_quote": True}))


def test_gate_idor_requires_BOTH_numeric_id_and_auth_not_either_alone():
    assert "idor" not in gate_classes_for_tags(characterize_tags({"numeric_or_guid_id": True}))
    assert "idor" not in gate_classes_for_tags(characterize_tags({"behind_auth": True}))
    assert "idor" in gate_classes_for_tags(characterize_tags({"numeric_or_guid_id": True, "behind_auth": True}))


def test_gate_fires_nothing_for_an_unremarkable_entry():
    assert gate_classes_for_tags(characterize_tags({})) == []


def test_gate_classes_for_entry_convenience_matches_two_step_call():
    entry = {"value_url_shaped": True, "content_type": "text/html"}
    assert gate_classes_for_entry(entry) == gate_classes_for_tags(characterize_tags(entry))


# --- surface_characterize ---------------------------------------------------

def test_characterize_browser_probe_form_tags_url_shaped_param():
    facts = {"forms": [{"action": "https://h.example/product/stock", "method": "POST",
                         "inputs": ["productId", "stockApi"]}]}
    entries = characterize_observation("browser_probe", facts)
    by_param = {e["param"]: e for e in entries}
    assert by_param["stockApi"]["value_url_shaped"] is True
    assert by_param["productId"]["value_url_shaped"] is False


def test_characterize_xml_resource_hint_tags_accepts_xml_on_all_forms():
    facts = {"forms": [{"action": "/product/stock", "method": "POST", "inputs": ["storeId"]}],
             "resources": ["/resources/js/xmlStockCheckPayload.js"]}
    entries = characterize_observation("browser_probe", facts)
    assert entries[0]["accepts_xml"] is True


def test_characterize_recon_scan_forms():
    facts = {"recon": {"forms": [{"action": "https://h.example/search", "param": "fileName"}]}}
    entries = characterize_observation("recon_scan", facts)
    assert entries[0]["value_path_shaped"] is True
    assert entries[0]["param"] == "fileName"


def test_characterize_malformed_facts_never_raises():
    assert characterize_observation("browser_probe", {}) == []
    assert characterize_observation("browser_probe", {"forms": "not-a-list"}) == []
    assert characterize_observation("browser_probe", None) == []


def test_class_to_engine_has_no_entry_for_ssti_the_documented_fallback_class():
    assert "ssti" not in CLASS_TO_ENGINE
    assert CLASS_TO_ENGINE["xxe"] == "xxe_probe"
    assert CLASS_TO_ENGINE["ssrf"] == "ssrf_probe"


# --- EvidenceStore surface_map / coverage_ledger ----------------------------

def test_surface_entry_upsert_merges_not_overwrites():
    store = EvidenceStore()
    store.add_surface_entry("GET /x::p", value_url_shaped=True)
    store.add_surface_entry("GET /x::p", numeric_or_guid_id=True)
    entry = store.surface_map["GET /x::p"]
    assert entry["value_url_shaped"] is True
    assert entry["numeric_or_guid_id"] is True


def test_coverage_ledger_roundtrips_through_to_dict_from_dict():
    store = EvidenceStore()
    store.record_coverage("ssrf", "POST /x::p", "untested", engine="ssrf_probe")
    restored = EvidenceStore.from_dict(store.to_dict())
    assert restored.coverage_ledger == store.coverage_ledger
    assert restored.surface_map == store.surface_map


def test_coverage_summary_lines_are_stable_sorted():
    store = EvidenceStore()
    store.record_coverage("ssrf", "POST /b::p", "untested")
    store.record_coverage("lfi", "GET /a::p", "proven")
    lines = store.coverage_summary_lines()
    assert lines == sorted(lines, key=lambda l: l)  # sorted by ledger key, already alpha-stable here
    assert any("PROVEN" in l for l in lines)


# --- BaseAgent._update_sweep_coverage wiring --------------------------------

def test_update_sweep_coverage_populates_ledger_from_a_browser_probe_observation():
    a = _agent()
    store = EvidenceStore()
    obs = Observation(
        action_id="browser_probe", argv=[], exit_code=0, raw_stdout="{}",
        parsed_facts={"forms": [{"action": "https://h.example/product/stock", "method": "POST",
                                  "inputs": ["storeId"]}],
                      "resources": ["/resources/js/xmlStockCheckPayload.js"]},
    )
    store.add_observation(obs)
    a._update_sweep_coverage([obs], store)
    assert any(row["vuln_class"] == "xxe" and row["status"] == "untested"
               for row in store.coverage_ledger.values())
    assert any(row["engine"] == "xxe_probe" for row in store.coverage_ledger.values())


def test_update_sweep_coverage_no_engine_class_gets_no_engine_status_and_note():
    a = _agent()
    store = EvidenceStore()
    obs = Observation(
        action_id="browser_probe", argv=[], exit_code=0, raw_stdout="{}",
        parsed_facts={"forms": [{"action": "https://h.example/view", "method": "GET",
                                  "inputs": ["template"]}]},
    )
    store.add_observation(obs)
    a._update_sweep_coverage([obs], store)
    ssti_rows = [row for row in store.coverage_ledger.values() if row["vuln_class"] == "ssti"]
    assert ssti_rows and ssti_rows[0]["status"] == "no_engine"
    assert "no typed engine" in (ssti_rows[0]["note"] or "")


def test_update_sweep_coverage_advances_to_proven_on_a_confirmed_engine_signal():
    a = _agent()
    store = EvidenceStore()
    browse_obs = Observation(
        action_id="browser_probe", argv=[], exit_code=0, raw_stdout="{}",
        parsed_facts={"forms": [{"action": "https://h.example/fetch", "method": "GET",
                                  "inputs": ["targetUrl"]}]},
    )
    store.add_observation(browse_obs)
    a._update_sweep_coverage([browse_obs], store)
    key = next(k for k, row in store.coverage_ledger.items() if row["vuln_class"] == "ssrf")
    assert store.coverage_ledger[key]["status"] == "untested"

    engine_obs = Observation(
        action_id="ssrf_probe", argv=["python", "ssrf_probe_runner.py", "--target",
                                       "https://h.example/fetch"],
        exit_code=0, raw_stdout="{}",
        parsed_facts={"vuln_signal": {"confirmed": True, "vuln_class": "ssrf"}},
    )
    store.add_observation(engine_obs)
    a._update_sweep_coverage([engine_obs], store)
    assert store.coverage_ledger[key]["status"] == "proven"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
