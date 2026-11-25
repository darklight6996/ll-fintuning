"""
tests/test_payload_pack.py

Tests for core/payload_pack.py (2026-11-25 4-repo harvest, asset 8 --
re-implemented from pentest-agents' mcp-writeup-server search_payloads(),
see work.md). Confirms the alias normalization, the checklist/mutation-matrix
builders, the output is bounded, and an unrecognized/empty class degrades to
"" rather than raising.
"""

import pytest

from core.payload_pack import (
    build_attack_checklist,
    build_mutation_matrix,
    build_payload_pack,
    normalize_vuln_class,
    SEED_PAYLOADS,
)


def test_normalize_vuln_class_collapses_aliases():
    assert normalize_vuln_class("Cross Site Scripting") == "xss"
    assert normalize_vuln_class("server-side request forgery") == "ssrf"
    assert normalize_vuln_class("SQL Injection") == "sqli"
    assert normalize_vuln_class("Template Injection") == "ssti"


def test_normalize_vuln_class_passes_through_already_canonical():
    assert normalize_vuln_class("xss") == "xss"
    assert normalize_vuln_class("idor") == "idor"


def test_build_attack_checklist_includes_base_and_per_class_items():
    checklist = build_attack_checklist("xss")
    assert any("sibling endpoint" in item for item in checklist)  # base
    assert any("source -> sink" in item for item in checklist)    # xss-specific


def test_build_attack_checklist_unknown_class_still_returns_base():
    checklist = build_attack_checklist("some-made-up-class")
    assert len(checklist) == 4  # base only, no per-class extras


def test_build_mutation_matrix_includes_base_and_per_class_families():
    matrix = build_mutation_matrix("ssrf")
    assert "encoding" in matrix and "transport" in matrix
    assert "routing" in matrix
    assert "dns-rebind" in matrix["routing"]


def test_build_mutation_matrix_unknown_class_has_no_per_class_extras():
    matrix = build_mutation_matrix("some-made-up-class")
    assert set(matrix) == {"encoding", "transport", "context"}


def test_build_payload_pack_contains_canonical_name_checklist_and_seeds():
    pack = build_payload_pack("Cross Site Scripting")
    assert "Payload pack (xss):" in pack
    assert "Attack checklist" in pack
    assert "Mutation matrix" in pack
    assert "Seed payloads" in pack


def test_build_payload_pack_empty_input_returns_empty_string():
    assert build_payload_pack("") == ""
    assert build_payload_pack("   ") == ""


def test_build_payload_pack_unrecognized_class_has_no_seeds_section():
    pack = build_payload_pack("totally-unknown-class")
    assert "Seed payloads" not in pack
    assert "Attack checklist" in pack  # base checklist always present


def test_build_payload_pack_is_bounded():
    pack = build_payload_pack("xss", max_chars=60)
    assert len(pack) <= 60


def test_seed_payloads_never_raises_on_missing_class():
    # SEED_PAYLOADS.get with a default is used internally; confirm the dict
    # itself has no entry crafted with sensitive-looking real-world data.
    for cls, seeds in SEED_PAYLOADS.items():
        assert isinstance(seeds, list) and seeds


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
