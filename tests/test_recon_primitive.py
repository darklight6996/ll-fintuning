"""
tests/test_recon_primitive.py

Tests for the shared recon action wrapper (agents/common/recon_primitive.py):
ActionDefinition shape, the two command builders (full vs single-host-clamped
modes), the fact extractor, and registration in both the domain-wide and
single-host bug_bounty registries.
"""

import pytest

from agents.common.recon_primitive import (
    RECON_SCAN,
    ReconScanParams,
    build_recon_scan,
    build_recon_scan_single_host,
    extract_recon_scan_facts,
    RECON_MODES_DEFAULT,
    RECON_MODES_SINGLE_HOST,
)
from agents.bug_bounty.actions import build_bug_bounty_registry
from agents.bug_bounty.single_host_mode import build_single_host_bug_bounty_registry


def test_recon_scan_is_a_coverage_action_never_gated_behind_suspicion():
    # Same posture as header_analysis/endpoint_discovery: always offered.
    assert RECON_SCAN.vuln_class is None
    assert RECON_SCAN.requires_suspicion is False
    assert RECON_SCAN.network_effect == "read_only"
    assert RECON_SCAN.state_change == "none"
    assert RECON_SCAN.risk_level == "low"
    assert RECON_SCAN.satisfies_phase is True
    assert RECON_SCAN.phase == "recon"


def test_build_recon_scan_uses_default_modes_when_unset():
    argv = build_recon_scan(ReconScanParams(target="https://x.example/"))[0]
    assert "--modes" in argv
    assert argv[argv.index("--modes") + 1] == RECON_MODES_DEFAULT


def test_build_recon_scan_honors_explicit_modes_override():
    argv = build_recon_scan(ReconScanParams(target="https://x.example/", modes="dns,tls"))[0]
    assert argv[argv.index("--modes") + 1] == "dns,tls"


def test_recon_scan_params_accepts_a_list_of_modes():
    # Observed live 2026-11-23: the model proposed modes as a JSON array,
    # which the plain-str field used to reject at Gate 2. Must now be
    # accepted and joined into the same comma-separated form.
    params = ReconScanParams(target="https://x.example/", modes=["dns", "tls", "tech"])
    assert params.modes == "dns,tls,tech"
    argv = build_recon_scan(params)[0]
    assert argv[argv.index("--modes") + 1] == "dns,tls,tech"


def test_build_recon_scan_single_host_ignores_any_requested_modes():
    # The AI's requested override must be ignored -- the framework decides
    # single-host mode deterministically, never the model.
    argv = build_recon_scan_single_host(
        ReconScanParams(target="https://x.example/", modes="ports,subdomains")
    )[0]
    assert argv[argv.index("--modes") + 1] == RECON_MODES_SINGLE_HOST
    assert "ports" not in RECON_MODES_SINGLE_HOST.split(",")
    assert "subdomains" not in RECON_MODES_SINGLE_HOST.split(",")


def test_js_recon_and_auth_sweep_are_target_only_and_in_both_mode_sets():
    # Both modes only ever touch the one pinned host (homepage/JS mining,
    # OpenAPI spec served by that host) -- safe for single-host too, same
    # reasoning already applied to secrets/exposed_files/real_ip.
    for mode in ("js_recon", "auth_sweep"):
        assert mode in RECON_MODES_DEFAULT.split(",")
        assert mode in RECON_MODES_SINGLE_HOST.split(",")


def test_extractor_parses_recon_facts():
    out = '{"recon": {"target": "x.example", "ports": [{"port": 22, "risk_level": "medium"}]}}'
    facts = extract_recon_scan_facts(out, "")
    assert facts["recon"]["ports"][0]["port"] == 22


def test_extractor_ignores_noise():
    assert extract_recon_scan_facts("not json\nnope", "") == {}


def test_recon_scan_registered_in_domain_wide_registry_with_full_modes():
    reg = build_bug_bounty_registry()
    defn = reg.get("recon_scan")
    assert defn is not None
    builder = reg.get_builder(defn.command_builder)
    assert callable(builder)
    argv = builder(ReconScanParams(target="https://x.example/"))[0]
    assert argv[argv.index("--modes") + 1] == RECON_MODES_DEFAULT


def test_recon_scan_registered_in_single_host_registry_clamped():
    reg = build_single_host_bug_bounty_registry()
    defn = reg.get("recon_scan")
    assert defn is not None
    builder = reg.get_builder(defn.command_builder)
    assert callable(builder)
    # Even if a model somehow supplied ports/subdomains, the single-host
    # builder must still clamp to the narrower set.
    argv = builder(ReconScanParams(target="https://x.example/", modes="ports,subdomains"))[0]
    assert argv[argv.index("--modes") + 1] == RECON_MODES_SINGLE_HOST


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
