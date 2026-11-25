"""
tests/test_campaign_judgment.py

Tests for core/campaign_judgment.py (2026-11-25 harvest: re-authored
pentest-agents judgment calibration -- never-submit/chain/mistakes
methodology, ported as our own prompt text since that source repo has no
clear LICENSE). Confirms the rendered section is bounded, non-empty, and
actually reaches the planner's prompt template.
"""

import pytest

from core.campaign_judgment import (
    render_campaign_judgment, WEAK_ALONE_PATTERNS, CHAIN_HINTS, OPERATIONAL_LESSONS,
    PER_CLASS_IMPACT_CHECKS, DEVILS_ADVOCATE_FRAMING,
)


def test_render_includes_all_sections():
    rendered = render_campaign_judgment()
    assert "NOT worth a finding on their own" in rendered
    assert "chains further" in rendered
    assert "Operational discipline" in rendered
    assert "actually match the claimed impact" in rendered
    assert "Devil's advocate" in rendered


def test_per_class_impact_checks_cover_key_vuln_classes():
    rendered = render_campaign_judgment(max_impact_checks=len(PER_CLASS_IMPACT_CHECKS))
    for cls in ("XSS", "IDOR", "SSRF", "Race condition", "Auth bypass"):
        assert cls in rendered


def test_devils_advocate_framing_is_nonempty():
    assert len(DEVILS_ADVOCATE_FRAMING) > 50


def test_render_is_capped_not_unbounded():
    rendered = render_campaign_judgment(max_chain_hints=2, max_lessons=2)
    # 6 weak-alone patterns are always shown (first 6), plus the capped
    # chain hints/lessons -- total line count should reflect the caps.
    chain_lines = [l for l in rendered.splitlines() if l.strip().startswith("- ") and
                   any(h[:30] in l for h in CHAIN_HINTS)]
    assert len(chain_lines) <= 2


def test_static_lists_are_nonempty_and_reasonably_sized():
    assert 5 <= len(WEAK_ALONE_PATTERNS) <= 20
    assert 3 <= len(CHAIN_HINTS) <= 15
    assert 3 <= len(OPERATIONAL_LESSONS) <= 15


def test_render_reaches_the_planner_prompt_template():
    from core.ai_planner import _PLANNER_USER_TEMPLATE
    assert "{campaign_judgment}" in _PLANNER_USER_TEMPLATE
    assert "CAMPAIGN JUDGMENT" in _PLANNER_USER_TEMPLATE


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
