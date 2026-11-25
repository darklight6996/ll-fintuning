"""
tests/test_bounty_platform_primitive.py

Tests for agents/common/bounty_platform_primitive.py (2026-11-25 4-repo
harvest, asset 7): the ActionDefinition shape, the opt-in env-var gate
(mirrors self_scripting_enabled()'s pattern), the command builder, the fact
extractor, and registration gating in both bug_bounty registries.
"""

import pytest

from agents.common.bounty_platform_primitive import (
    BOUNTY_PLATFORM_RECON,
    BountyPlatformParams,
    build_bounty_platform_recon,
    bounty_platform_enabled,
    extract_bounty_platform_facts,
)


def test_bounty_platform_recon_is_local_target_and_read_only():
    # The "target" is a program handle, not an asset to test -- same posture
    # as agents/ctf/actions.py's filesystem-only actions.
    assert BOUNTY_PLATFORM_RECON.allowed_target_kinds == ["local"]
    assert BOUNTY_PLATFORM_RECON.network_effect == "read_only"
    assert BOUNTY_PLATFORM_RECON.state_change == "none"
    assert BOUNTY_PLATFORM_RECON.requires_active_testing is False


def test_bounty_platform_enabled_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("PENTEST_ENABLE_BOUNTY_PLATFORM", raising=False)
    assert bounty_platform_enabled() is False
    monkeypatch.setenv("PENTEST_ENABLE_BOUNTY_PLATFORM", "1")
    assert bounty_platform_enabled() is True
    monkeypatch.setenv("PENTEST_ENABLE_BOUNTY_PLATFORM", "true")  # only "1" counts
    assert bounty_platform_enabled() is False


def test_build_bounty_platform_recon_sync_mode_includes_target_dir():
    params = BountyPlatformParams(target="example", mode="sync", target_dir="/tmp/eng")
    argv = build_bounty_platform_recon(params)[0]
    assert "--program" in argv and argv[argv.index("--program") + 1] == "example"
    assert "--target-dir" in argv and argv[argv.index("--target-dir") + 1] == "/tmp/eng"
    assert argv[argv.index("--mode") + 1] == "sync"


def test_build_bounty_platform_recon_scope_mode_omits_target_dir():
    params = BountyPlatformParams(target="example", mode="scope")
    argv = build_bounty_platform_recon(params)[0]
    assert "--target-dir" not in argv


def test_extract_bounty_platform_facts_parses_json_line():
    out = '{"bounty": {"mode": "scope", "platform": "hackerone", "found": true}}'
    facts = extract_bounty_platform_facts(out, "")
    assert facts["bounty"]["found"] is True


def test_extract_bounty_platform_facts_ignores_noise():
    assert extract_bounty_platform_facts("not json\nalso not json", "") == {}


# --- registration gating --------------------------------------------------------

def test_not_registered_in_domain_wide_registry_by_default(monkeypatch):
    monkeypatch.delenv("PENTEST_ENABLE_BOUNTY_PLATFORM", raising=False)
    from agents.bug_bounty.actions import build_bug_bounty_registry
    registry = build_bug_bounty_registry()
    assert registry.get("bounty_platform_recon") is None


def test_registered_in_domain_wide_registry_when_opted_in(monkeypatch):
    monkeypatch.setenv("PENTEST_ENABLE_BOUNTY_PLATFORM", "1")
    from agents.bug_bounty.actions import build_bug_bounty_registry
    registry = build_bug_bounty_registry()
    assert registry.get("bounty_platform_recon") is not None


def test_registered_in_single_host_registry_when_opted_in(monkeypatch):
    monkeypatch.setenv("PENTEST_ENABLE_BOUNTY_PLATFORM", "1")
    from agents.bug_bounty.single_host_mode import build_single_host_bug_bounty_registry
    registry = build_single_host_bug_bounty_registry()
    assert registry.get("bounty_platform_recon") is not None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
