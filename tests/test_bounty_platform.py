"""
tests/test_bounty_platform.py

Tests for core/tools/bounty_platform.py (2026-11-25 4-repo harvest, asset 7 --
re-implemented from pentest-agents' mcp-bounty-server, see work.md). All
network calls (`urlopen`) are monkeypatched -- no real HTTP request ever
happens in this suite, and no real credential is ever used.
"""

import io
import json
from urllib.error import HTTPError

import pytest

import core.tools.bounty_platform as bp


class _FakeResp:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code):
    return HTTPError("https://api.hackerone.com/v1/x", code, "err", {}, io.BytesIO(b""))


# --- fetch_scope ---------------------------------------------------------------

def test_fetch_scope_builds_in_and_out_of_scope_lists(monkeypatch):
    def fake_urlopen(req, timeout=None):
        if "structured_scopes" in req.full_url:
            return _FakeResp({"data": [
                {"attributes": {"asset_identifier": "*.example.com", "asset_type": "WILDCARD",
                                 "eligible_for_bounty": True, "eligible_for_submission": True}},
                {"attributes": {"asset_identifier": "internal.example.com", "asset_type": "URL",
                                 "eligible_for_bounty": False, "eligible_for_submission": False}},
            ]})
        return _FakeResp({"data": {"attributes": {"name": "Example Program", "updated_at": "2026-01-01"}}})

    monkeypatch.setattr(bp, "urlopen", fake_urlopen)
    scope = bp.fetch_scope("example", "user", "token")
    assert scope["program_name"] == "Example Program"
    assert len(scope["in_scope"]) == 1
    assert scope["in_scope"][0]["asset"] == "*.example.com"
    assert len(scope["out_of_scope"]) == 1
    assert scope["out_of_scope"][0]["asset"] == "internal.example.com"


def test_fetch_scope_returns_none_for_unknown_program(monkeypatch):
    monkeypatch.setattr(bp, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(_http_error(404)))
    assert bp.fetch_scope("nonexistent", "user", "token") is None


def test_fetch_scope_raises_permission_error_on_401(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise _http_error(401)
    monkeypatch.setattr(bp, "urlopen", fake_urlopen)
    with pytest.raises(PermissionError):
        bp.fetch_scope("example", "baduser", "badtoken")


def test_fetch_scope_permission_error_never_echoes_credentials(monkeypatch):
    monkeypatch.setattr(bp, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(_http_error(403)))
    with pytest.raises(PermissionError) as exc:
        bp.fetch_scope("example", "secret-user", "secret-token-value")
    assert "secret-user" not in str(exc.value)
    assert "secret-token-value" not in str(exc.value)


# --- fetch_policy ---------------------------------------------------------------

def test_fetch_policy_reports_text_and_restrictions(monkeypatch):
    monkeypatch.setattr(bp, "urlopen", lambda req, timeout=None: _FakeResp(
        {"data": {"attributes": {"policy": "Be nice.", "submission_state": "paused",
                                  "gold_standard_safe_harbor": True}}}))
    policy = bp.fetch_policy("example", "user", "token")
    assert policy["policy_text"] == "Be nice."
    assert policy["policy_text_available"] is True
    assert policy["safe_harbor"] is True
    assert policy["testing_restrictions"] == ["Submission state: paused"]


def test_fetch_policy_empty_text_is_reported_honestly_not_scraped(monkeypatch):
    # Some programs' REST payload has no policy text -- this must be reported
    # as unavailable, never silently filled in via website scraping.
    monkeypatch.setattr(bp, "urlopen", lambda req, timeout=None: _FakeResp(
        {"data": {"attributes": {"policy": "", "submission_state": "open",
                                  "gold_standard_safe_harbor": False}}}))
    policy = bp.fetch_policy("example", "user", "token")
    assert policy["policy_text_available"] is False
    assert policy["testing_restrictions"] == []


def test_fetch_policy_returns_none_for_unknown_program(monkeypatch):
    monkeypatch.setattr(bp, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(_http_error(404)))
    assert bp.fetch_policy("nonexistent", "user", "token") is None


# --- do_sync / placeholder sentinel ---------------------------------------------

def test_do_sync_writes_real_scope_and_policy_on_success(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        if "structured_scopes" in req.full_url:
            return _FakeResp({"data": []})
        return _FakeResp({"data": {"attributes": {
            "name": "Example Program", "updated_at": "2026-01-01",
            "policy": "Be nice.", "submission_state": "open", "gold_standard_safe_harbor": True,
        }}})
    monkeypatch.setattr(bp, "urlopen", fake_urlopen)

    result = bp.do_sync("hackerone", "example", "user", "token", str(tmp_path))
    scope_text = (tmp_path / "bounty_scope.yaml").read_text()
    policy_text = (tmp_path / "bounty_policy.md").read_text()
    assert "scope_mode: placeholder" not in scope_text
    assert "Example Program" in scope_text
    assert "Be nice." in policy_text
    assert any("scope written" in r for r in result["results"])
    assert any("policy written" in r for r in result["results"])


def test_do_sync_writes_placeholder_scope_on_api_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(bp, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(_http_error(404)))

    result = bp.do_sync("hackerone", "ghost-program", "user", "token", str(tmp_path))
    scope_text = (tmp_path / "bounty_scope.yaml").read_text()
    policy_text = (tmp_path / "bounty_policy.md").read_text()
    assert "scope_mode: placeholder" in scope_text
    assert "DO NOT test any target against this file" in scope_text
    assert "unavailable from API" in policy_text
    assert any("scope unavailable" in r for r in result["results"])


def test_do_sync_placeholder_on_bad_credentials_never_contains_token(monkeypatch, tmp_path):
    monkeypatch.setattr(bp, "urlopen", lambda req, timeout=None: (_ for _ in ()).throw(_http_error(401)))
    bp.do_sync("hackerone", "example", "user", "super-secret-token", str(tmp_path))
    scope_text = (tmp_path / "bounty_scope.yaml").read_text()
    assert "super-secret-token" not in scope_text


# --- main() CLI plumbing --------------------------------------------------------

def test_main_unsupported_platform_emits_not_configured(monkeypatch, capsys):
    monkeypatch.setattr(bp.sys, "argv", ["bounty_platform.py", "--program", "x", "--platform", "bugcrowd"])
    bp.main(["--program", "x", "--platform", "bugcrowd"])
    out = json.loads(capsys.readouterr().out.strip())
    assert out["bounty"]["configured"] is False
    assert "bugcrowd" in out["bounty"]["error"]


def test_main_missing_credentials_emits_not_configured(monkeypatch, capsys):
    monkeypatch.delenv("HACKERONE_USERNAME", raising=False)
    monkeypatch.delenv("HACKERONE_TOKEN", raising=False)
    bp.main(["--program", "x", "--mode", "scope"])
    out = json.loads(capsys.readouterr().out.strip())
    assert out["bounty"]["configured"] is False


def test_main_never_crashes_without_emitting_json(monkeypatch, capsys):
    monkeypatch.setenv("HACKERONE_USERNAME", "u")
    monkeypatch.setenv("HACKERONE_TOKEN", "t")

    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(bp, "fetch_scope", boom)
    bp.main(["--program", "x", "--mode", "scope"])
    out = json.loads(capsys.readouterr().out.strip())
    assert out["bounty"]["found"] is False
    assert "network down" in out["bounty"]["error"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
