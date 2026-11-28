"""
tests/test_render_fidelity.py

BUILD SPEC v1 (2026-11-28), workstream A part 1 -- regression tests for
full-fidelity discovered-facts rendering (audit Defect #1 / Section 1).

The false-IDOR incident was caused by `_format_discovered_facts` rendering a
hard `value[:15]` slice + a bare `(+N more)` suffix: the model saw productId
1-15 and a count, formed the hypothesis "IDs 1-15 are public, higher IDs are
hidden", and never saw that IDs 16-20 were in the SAME list one layer below the
cut. These tests exercise the ACTUAL rendering code (not a duplicated test-only
implementation) and assert the 15-item silent truncation can no longer happen.
"""
from core.ai_planner import AIPlanner
from core.investigation import EvidenceStore, Observation


def _planner():
    # _format_discovered_facts / _render_list_faithfully use only self + env;
    # they never touch registry/chat_assistant/scope, so None is fine here.
    return AIPlanner(registry=None, chat_assistant=None, scope=None)


def _store_with_endpoints(n=20):
    store = EvidenceStore()
    store.add_observation(Observation(
        action_id="endpoint_discovery",
        argv=[],
        raw_stdout="",
        parsed_facts={"endpoints": [f"/product?productId={i}" for i in range(1, n + 1)]},
    ))
    return store


def test_discovered_facts_communicates_all_20_ids():
    """The core regression: all 20 productId values must be communicated, and the
    old bare `(+N more)` silent-truncation marker must be gone. Fails against the
    previous value[:15] implementation (which hid 16-20 behind '(+32 more)')."""
    out = _planner()._format_discovered_facts(_store_with_endpoints(20))
    # Under the default context budget, 20 short endpoints fit -> all rendered.
    for i in range(1, 21):
        assert f"productId={i}" in out, f"productId={i} missing from rendered facts:\n{out}"
    # The specific concealing marker that caused the incident must not appear.
    assert "(+" not in out


def test_discovered_facts_never_renders_bare_plus_n_more():
    """Even with a huge list, the output must never be a bare '(+N more)' that
    conceals which items were dropped."""
    out = _planner()._format_discovered_facts(_store_with_endpoints(500))
    assert "(+" not in out
    # It must still truthfully communicate the real total.
    assert "500" in out


def test_faithful_list_small_budget_gives_truthful_contiguous_range():
    """When the full list genuinely cannot fit, a sequential-ID list is rendered
    as a truthful CONTIGUOUS range (1-20) with an EvidenceStore reference -- an
    acceptable outcome per the spec -- never as a 1-15 partial that looks whole."""
    vals = [f"/product?productId={i}" for i in range(1, 21)]
    out = _planner()._render_list_faithfully(vals, budget_chars=120)
    assert "1-20" in out
    assert "CONTIGUOUS" in out.upper()
    assert "EvidenceStore" in out
    # Must communicate the real total and must not pretend to be the full literal list.
    assert "20 values" in out


def test_faithful_list_small_budget_discloses_gaps_not_silently_drops():
    """A non-contiguous sequential-ID list must disclose the gap, never silently
    represent itself as a continuous range."""
    vals = [f"/product?productId={i}" for i in range(1, 11)] + ["/product?productId=12"]  # 11 missing
    out = _planner()._render_list_faithfully(vals, budget_chars=80)
    assert "11" in out          # the missing value is explicitly disclosed
    assert "1-12" in out        # the true span is stated
    assert "EvidenceStore" in out


def test_faithful_list_generic_partial_is_marked_not_complete():
    """A non-numeric list that overflows budget must be explicitly marked PARTIAL
    with the true total, not silently cut."""
    vals = [f"/path/{name}" for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")]
    out = _planner()._render_list_faithfully(vals, budget_chars=30)
    assert "PARTIAL LIST" in out
    assert f"of {len(vals)}" in out
    assert "EvidenceStore" in out


def test_faithful_list_full_render_when_it_fits():
    """When the list fits the budget it is rendered verbatim (no spurious
    truncation chatter)."""
    vals = ["/a", "/b", "/c"]
    out = _planner()._render_list_faithfully(vals, budget_chars=10_000)
    assert out == repr(vals)
    assert "PARTIAL" not in out
    assert "EvidenceStore" not in out


def test_render_budgets_track_configured_context(monkeypatch):
    """The render budget is derived from the ACTUAL configured context window
    (OLLAMA_NUM_CTX), not a hardcoded constant."""
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    small = _planner()._render_budgets()
    monkeypatch.setenv("OLLAMA_NUM_CTX", "65536")
    big = _planner()._render_budgets()
    assert big["discovered_facts"] > small["discovered_facts"]
    # Sanity: all three evidence sections get a positive share.
    for section in ("discovered_facts", "observations", "history"):
        assert small[section] > 0 and big[section] > 0


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
