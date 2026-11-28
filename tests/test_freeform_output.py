"""
tests/test_freeform_output.py

BUILD SPEC v1 workstream C (2026-11-28): free-form-reason-then-convert output
(#13) + widened/relevance-ordered candidate presentation (#4/#5).

The planner no longer forces strict inline JSON. The model reasons in free prose
and ends with ONE fenced ```action block; a deterministic extractor converts it
at the boundary, falling back to a CHEAP model only when ambiguous -- never the
primary reasoning model for formatting. Unknown action_ids are rejected
(fail-safe); nothing is ever fabricated.
"""
import os
os.environ.setdefault("PENTEST_ENABLE_SELF_SCRIPTING", "1")

import json
import pytest

from core.ai_planner import AIPlanner
from core.scope import Scope
from core.investigation import EvidenceStore, Observation
from agents.bug_bounty.actions import build_bug_bounty_registry
from agents.bug_bounty.completion import BUG_BOUNTY_COMPLETION


def _planner():
    return AIPlanner(
        registry=build_bug_bounty_registry(),
        chat_assistant=None,
        scope=Scope(target="https://x.web-security-academy.net/", in_scope=["x.web-security-academy.net"],
                    active_testing_allowed=True, active_network_allowed=True),
    )


class _Chat:
    """Scripted chat_assistant; records calls."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_raw(self, prompt, max_new_tokens=512, system_prompt=None, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return self._responses.pop(0)


class _Normalizer:
    """Fake CHEAP normalizer; records calls and returns scripted JSON."""
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_raw(self, prompt, max_new_tokens=512, system_prompt=None, **kwargs):
        self.calls += 1
        return self.response


class _NormalizerMustNotRun:
    def generate_raw(self, *a, **k):
        raise AssertionError("cheap-model normalizer was called on a deterministically-parseable block")


def _decide(planner):
    return planner.decide_next(
        session_state={}, evidence_store=EvidenceStore(), completion_criteria=BUG_BOUNTY_COMPLETION,
        execution_history=[], retry_budget_status={},
    )


# --- Part 1: deterministic free-form extraction -----------------------------

FREEFORM_PROSE_PLUS_BLOCK = """\
Let me think. The homepage lists several product endpoints and a stock checker.
Recon first is the right move before I form any hypothesis.

```action
{"decision": "propose_action", "action_id": "recon_scan", "target": null, "parameters": {}, "rationale": "recon first", "confidence": 0.9}
```
"""


def test_freeform_prose_plus_action_block_parses_deterministically_no_model_call():
    planner = _planner()
    planner.chat_assistant = _Chat([FREEFORM_PROSE_PLUS_BLOCK])
    planner._normalizer = _NormalizerMustNotRun()   # would raise if the normalizer ran
    proposal = _decide(planner)
    assert proposal.decision == "propose_action"
    assert proposal.action_id == "recon_scan"
    assert len(planner.chat_assistant.calls) == 1   # one primary call; no extra formatting call


def test_action_block_key_value_form_parses():
    """The block may be simple key: value lines, not only JSON."""
    planner = _planner()
    block = (
        "Reasoning: recon is the obvious first step.\n\n"
        "```action\n"
        "decision: propose_action\n"
        "action_id: recon_scan\n"
        "rationale: start with recon\n"
        "confidence: 0.8\n"
        "```\n"
    )
    planner.chat_assistant = _Chat([block])
    planner._normalizer = _NormalizerMustNotRun()
    proposal = _decide(planner)
    assert proposal.decision == "propose_action"
    assert proposal.action_id == "recon_scan"


def test_last_action_block_wins_when_multiple():
    planner = _planner()
    raw = (
        "First I considered this:\n```action\n{\"decision\": \"propose_completion\"}\n```\n"
        "But on reflection recon is better:\n"
        "```action\n{\"decision\": \"propose_action\", \"action_id\": \"recon_scan\", \"parameters\": {}}\n```\n"
    )
    proposal = planner._parse_proposal(raw)
    assert proposal.decision == "propose_action"
    assert proposal.action_id == "recon_scan"


def test_prose_outside_block_is_ignored():
    """A decision-shaped sentence in the PROSE must not be parsed -- only the block."""
    planner = _planner()
    raw = (
        'I will propose_action recon_scan... actually no, let me reconsider.\n'
        "```action\n{\"decision\": \"propose_completion\", \"action_id\": null}\n```\n"
    )
    proposal = planner._parse_proposal(raw)
    assert proposal.decision == "propose_completion"   # from the block, not the prose


# --- Part 1: cheap-model normalizer fallback --------------------------------

def test_ambiguous_block_triggers_cheap_normalizer():
    planner = _planner()
    # An action block that is neither valid JSON nor key:value (deterministic parse fails).
    malformed = "Here is my choice:\n```action\n<recon_scan, please>\n```\n"
    planner.chat_assistant = _Chat([malformed])
    norm = _Normalizer(json.dumps({"decision": "propose_action", "action_id": "recon_scan",
                                   "parameters": {}, "rationale": "normalized", "confidence": 0.7}))
    planner._normalizer = norm
    proposal = _decide(planner)
    assert proposal.action_id == "recon_scan"
    assert norm.calls == 1                       # the cheap model did the conversion
    assert len(planner.chat_assistant.calls) == 1  # no corrective retry needed


# --- Part 1: unknown action_id rejected (fail-safe) -------------------------

def test_unknown_action_id_is_rejected_not_executed():
    planner = _planner()
    bad = "```action\n{\"decision\": \"propose_action\", \"action_id\": \"definitely_not_registered\"}\n```\n"
    # Same bad response on every retry; no normalizer -> must end in rejection.
    planner.chat_assistant = _Chat([bad, bad, bad])
    planner._normalizer = False   # explicitly unavailable
    with pytest.raises(ValueError) as exc:
        _decide(planner)
    assert "unregistered action_id" in str(exc.value) or "failed to extract" in str(exc.value)


def test_reject_if_unregistered_helper():
    from core.action_contract import AIActionProposal
    planner = _planner()
    good = AIActionProposal(decision="propose_action", action_id="recon_scan", parameters={})
    planner._reject_if_unregistered(good)   # no raise
    bad = AIActionProposal(decision="propose_action", action_id="nope_nope", parameters={})
    with pytest.raises(ValueError):
        planner._reject_if_unregistered(bad)
    # Non-action decisions are never rejected on action_id grounds.
    planner._reject_if_unregistered(AIActionProposal(decision="propose_completion", action_id=None))


# --- Part 1: all-parse-failure surfaces error, fabricates nothing -----------

def test_all_parse_failure_raises_and_fabricates_nothing():
    planner = _planner()
    garbage = "I am thinking out loud but never emit any action block at all."
    planner.chat_assistant = _Chat([garbage, garbage, garbage])
    planner._normalizer = False
    with pytest.raises(ValueError) as exc:
        _decide(planner)
    assert "failed to extract a valid action" in str(exc.value)


def test_strict_json_backward_compat_still_parses():
    """A bare strict-JSON response (no action block) still parses via the fallback."""
    planner = _planner()
    raw = json.dumps({"decision": "propose_action", "action_id": "recon_scan", "parameters": {}})
    proposal = planner._parse_proposal(raw)
    assert proposal.action_id == "recon_scan"


# --- Part 1: reason-then-format still works as a fallback -------------------

def test_reason_then_format_still_works_with_action_block(monkeypatch):
    monkeypatch.setenv("PLANNER_REASON_THEN_FORMAT", "1")
    planner = _planner()
    # call 1 = free reasoning prose; call 2 = prose + action block.
    planner.chat_assistant = _Chat(["free reasoning about recon", FREEFORM_PROSE_PLUS_BLOCK])
    planner._normalizer = _NormalizerMustNotRun()
    proposal = _decide(planner)
    assert proposal.action_id == "recon_scan"
    assert len(planner.chat_assistant.calls) == 2   # reasoning pass + strict pass


# --- Part 2: candidate presentation (relevance order, advisory, not a filter) ---

def _store_ssrf_and_idor():
    """stockApi carries a real URL (SSRF, evidence weight 3); productId is a
    numeric id (IDOR, lower weight). Alphabetically 'idor' < 'ssrf', so an
    alphabetical ordering would wrongly put idor first."""
    store = EvidenceStore()
    store.add_observation(Observation(
        action_id="browser_probe", argv=[], raw_stdout="{}", exit_code=0,
        parsed_facts={
            "discovered_endpoints": [
                "https://x.web-security-academy.net/product/stock?stockApi=http://169.254.169.254/",
                "https://x.web-security-academy.net/product?productId=3",
            ],
        },
    ))
    return store


def test_candidates_ordered_by_relevance_not_alphabetically():
    planner = _planner()
    text, classes = planner._suspicion_candidates(_store_ssrf_and_idor(), set())
    assert "ssrf" in classes and "idor" in classes
    # Relevance order: ssrf (URL value, weight 3) before idor (numeric id). This is
    # the OPPOSITE of alphabetical (idor < ssrf), proving ordering is by evidence.
    assert classes.index("ssrf") < classes.index("idor"), classes
    # And in the rendered text too.
    assert text.lower().index("ssrf") < text.lower().index("idor")


def test_candidates_are_advisory_not_a_filter():
    planner = _planner()
    text, _classes = planner._suspicion_candidates(_store_ssrf_and_idor(), set())
    low = text.lower()
    assert "advisory" in low
    assert "not a menu" in low or "not a filter" in low
    # Explicitly tells the model it may pursue an unlisted class or vuln_class=None.
    assert "vuln_class=none" in low.replace(" ", "")


def test_candidate_classes_is_ordered_list_not_set():
    planner = _planner()
    _text, classes = planner._suspicion_candidates(_store_ssrf_and_idor(), set())
    assert isinstance(classes, list)   # ordered; a set would lose relevance order


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
