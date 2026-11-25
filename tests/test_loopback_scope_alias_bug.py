"""
tests/test_loopback_scope_alias_bug.py

Regression test for the loopback-alias fix in
core/scope.py::Scope.is_asset_in_scope().

Found live (pentest agent, Qwen2.5-7B-Instruct + qwen-qlora smoke test,
2026-11-14): after host_discovery's own nmap output printed "Nmap scan report
for localhost (127.0.0.1)", the model proposed its next action with
target="localhost" -- rejected 6 steps in a row with "Asset 'localhost' is
not listed in authorized in-scope boundaries", even though
Scope.in_scope=["127.0.0.1"] already authorizes this exact machine. Per
RFC 6761, "localhost" MUST always resolve to the loopback interface and can
never legitimately refer to a different host, so these are the same single
authorized target under different spellings, not scope creep.
"""

from core.scope import Scope


def test_localhost_is_in_scope_when_127_0_0_1_is_authorized():
    scope = Scope(target="127.0.0.1", in_scope=["127.0.0.1"])
    ok, reason = scope.is_asset_in_scope("localhost")
    assert ok, reason


def test_127_0_0_1_is_in_scope_when_localhost_is_authorized():
    """Symmetric case: the operator authorized "localhost" by name; a
    proposal using the literal IP must also be recognized as the same host."""
    scope = Scope(target="localhost", in_scope=["localhost"])
    ok, reason = scope.is_asset_in_scope("127.0.0.1")
    assert ok, reason


def test_ipv6_loopback_is_in_scope_when_127_0_0_1_is_authorized():
    scope = Scope(target="127.0.0.1", in_scope=["127.0.0.1"])
    ok, reason = scope.is_asset_in_scope("::1")
    assert ok, reason


def test_localhost_still_rejected_when_no_loopback_spelling_is_in_scope():
    """The fix must never widen scope on its own -- if the operator
    authorized a completely different host, "localhost" must still be
    rejected exactly as before."""
    scope = Scope(target="example.com", in_scope=["example.com"])
    ok, reason = scope.is_asset_in_scope("localhost")
    assert not ok
    assert "not listed in authorized in-scope boundaries" in reason


def test_localhost_still_rejected_when_in_scope_is_a_different_ip():
    scope = Scope(target="10.0.0.5", in_scope=["10.0.0.5"])
    ok, reason = scope.is_asset_in_scope("localhost")
    assert not ok


def test_loopback_alias_normalization_respects_out_of_scope():
    """If localhost/127.0.0.1 is explicitly excluded, the alias fix must not
    let the other spelling sneak past that exclusion."""
    scope = Scope(
        target="127.0.0.1",
        in_scope=["127.0.0.1"],
        out_of_scope=["localhost"],
    )
    ok, reason = scope.is_asset_in_scope("127.0.0.1")
    assert not ok, "out_of_scope must still override in_scope after alias normalization"
