"""
tests/test_suspicion_gate.py

Tests for the SUSPICION GATE + feature->vuln-class bridge (2026-11-16).

The behaviour under test (the user's requirement: "it shouldn't fire tools
unnecessarily, it should suspect the vuln at least before firing the tool"):

  1. action_is_unlocked() -- the pure gate: ungated actions are always offered;
     a class-gated action needs a matching open hypothesis; a generic gated
     action (vuln_class=None, e.g. self_script) needs ANY open hypothesis.
  2. AIPlanner._suspicion_candidates() -- turns observed parameters into concrete,
     class-tagged leads (a URL-ish param -> SSRF; a numeric id param -> IDOR)
     WITHOUT any per-bug hardcoding in the catalog.
  3. The planner withholds offensive tools (self_script, sqli_deep_scan) from the
     REGISTERED ACTIONS catalog until a matching suspicion exists, listing them
     under LOCKED TOOLS instead -- and releases each one as soon as the matching
     hypothesis is formed.
"""

import os
os.environ["PENTEST_ENABLE_SELF_SCRIPTING"] = "1"  # so self_script/nuclei_template register

from pydantic import BaseModel

from core.action_contract import ActionDefinition, action_is_unlocked
from core.ai_planner import AIPlanner
from core.investigation import EvidenceStore, Observation, Hypothesis
from core.scope import Scope
from agents.bug_bounty.single_host_mode import build_single_host_bug_bounty_registry


class _NoParams(BaseModel):
    model_config = {"extra": "forbid"}


def _defn(action_id, vuln_class=None, requires_suspicion=False):
    return ActionDefinition(
        action_id=action_id, phase="vuln_validation", description="x",
        params_model=_NoParams, command_builder="noop",
        vuln_class=vuln_class, requires_suspicion=requires_suspicion,
    )


# --- 1. pure gate ----------------------------------------------------------

def test_ungated_action_always_unlocked():
    d = _defn("recon")
    assert action_is_unlocked(d, set(), False) is True


def test_class_gated_unlocks_on_any_suspicion_not_class_match():
    """Part 3 (BUILD SPEC v1 workstream B): the vuln_class string is a ranking
    HINT, not a hard lock. A class-gated specialist stays locked with NO
    suspicion, but unlocks once ANY open hypothesis exists -- even one whose
    class does not match (the SVG-lab mis-inferred-class lockout is gone).
    Execution is still governed by the approval gate, not the class label."""
    d = _defn("sqli_deep_scan", vuln_class="sqli", requires_suspicion=True)
    assert action_is_unlocked(d, set(), False) is False        # no suspicion -> locked
    assert action_is_unlocked(d, {"xss"}, True) is True         # different class STILL unlocks
    assert action_is_unlocked(d, {"sqli"}, True) is True        # matching class unlocks


def test_generic_gated_needs_any_suspicion():
    d = _defn("self_script", vuln_class=None, requires_suspicion=True)
    assert action_is_unlocked(d, set(), False) is False
    assert action_is_unlocked(d, set(), True) is True          # any suspicion unlocks it


# --- 2. feature -> suspicion bridge ----------------------------------------

def _planner():
    scope = Scope(target="h.example", active_testing_allowed=True, single_host_only=True)
    registry = build_single_host_bug_bounty_registry()
    return AIPlanner(registry=registry, chat_assistant=None, scope=scope)


def _store_with_coverage():
    store = EvidenceStore()
    store.add_observation(Observation(
        action_id="browser_probe", argv=[], raw_stdout="{}", exit_code=0,
        parsed_facts={
            "parameters": ["stockApi", "productId"],
            "discovered_endpoints": ["https://h.example/product?productId=1"],
            "forms": [{"action": "https://h.example/product/stock", "method": "post",
                       "inputs": ["stockApi"]}],
        },
    ))
    return store


def test_suspicion_candidates_flag_ssrf_and_idor():
    planner = _planner()
    text, classes = planner._suspicion_candidates(_store_with_coverage(), set())
    # Both the URL-ish param and the numeric-id param surface for reasoning; their
    # class labels live in `classes` as retrieval hints, not as a privileged verdict
    # spoon-fed in the text (the old '"SSRF" in text' assertion is gone -- candidates
    # are now presented neutrally; reasoning, not the table, decides).
    assert "ssrf" in classes, text
    assert "stockApi" in text, text
    assert "idor" in classes, text


# --- 3. end-to-end catalog gating ------------------------------------------

def _sections(prompt):
    reg = prompt[prompt.find("=== REGISTERED ACTIONS"):prompt.find("=== LOCKED TOOLS")]
    locked = prompt[prompt.find("=== LOCKED TOOLS"):prompt.find("=== COMPLETION CRITERIA")]
    return reg, locked


def test_offensive_tools_locked_until_suspicion_then_released():
    planner = _planner()
    store = _store_with_coverage()

    prompt = planner._build_prompt({}, store, None, [], {})
    reg, locked = _sections(prompt)
    # No hypothesis yet -> offensive tools are LOCKED, not offered.
    assert "self_script" in locked and "self_script" not in reg
    assert "sqli_deep_scan" in locked and "sqli_deep_scan" not in reg
    # Recon/coverage stays available.
    assert "browser_probe" in reg
    # The example should steer toward forming a hypothesis (candidates exist, none open).
    assert planner._example_proposal is not None
    assert planner._example_proposal["decision"] == "form_hypothesis"

    # Form a suspicion (any class) -> ALL requires_suspicion engines unlock,
    # not just a class-matching one. Part 3 (workstream B): the class string is a
    # ranking hint, not an access lock, so the SSRF hypothesis releases both the
    # generic self_script AND the class-gated sqli_deep_scan. Suspicion-before-
    # firing is preserved (nothing unlocked before this hypothesis existed); the
    # approval gate, not the class label, governs which engine actually runs.
    obs_id = next(iter(store.observations))
    store.add_hypothesis(Hypothesis(
        observation_ids=[obs_id],
        description="stockApi holds a URL -> possible SSRF",
        validation_test="self_script", vuln_class="ssrf",
    ))
    reg2, locked2 = _sections(planner._build_prompt({}, store, None, [], {}))
    assert "self_script" in reg2, reg2          # released by the suspicion
    assert "sqli_deep_scan" in reg2, reg2       # ALSO released -- no class-match lock anymore


# --- 4. ranking + progress (regressions from the live SSRF run, 2026-11-16) ------

def test_evidence_confirmed_param_outranks_name_only_guess():
    """Ranking is by EVIDENCE, not vuln-class identity: a param whose VALUE confirms
    its shape (a real URL reaching the server) must rank above a bare name-only
    guess. No injection class is privileged by name -- the old SSRF>IDOR hardcode is
    gone; reasoning, not the table, prioritizes."""
    planner = _planner()
    store = EvidenceStore()
    store.add_observation(Observation(
        action_id="browser_probe", argv=[], raw_stdout="{}", exit_code=0,
        parsed_facts={
            # stockApi carries an actual URL value -> confirmed evidence (weight 3)
            "discovered_endpoints": ["https://h.example/product/stock?stockApi=http://169.254.169.254/"],
            # productId is only a name, no value seen -> speculative (weight 2)
            "parameters": ["productId"],
        },
    ))
    text, _ = planner._suspicion_candidates(store, set())
    i_conf = text.lower().find("stockapi")
    i_name = text.lower().find("productid")
    assert i_conf != -1, text
    assert i_name == -1 or i_conf < i_name, f"value-confirmed param should rank first:\n{text}"


def test_example_steers_to_action_not_more_hypotheses_when_one_is_open():
    """With an open suspicion, the example must push the model to TEST it (an action),
    not form another hypothesis -- the fix for the 8x form_hypothesis spam loop."""
    planner = _planner()
    store = _store_with_coverage()
    obs_id = next(iter(store.observations))
    store.add_hypothesis(Hypothesis(
        observation_ids=[obs_id],
        description="stockApi on /product/stock takes a URL -> SSRF",
        validation_test="self_script", vuln_class="ssrf",
    ))
    planner._build_prompt({}, store, None, [], {})
    assert planner._example_proposal is not None
    assert planner._example_proposal["decision"] == "propose_action", planner._example_proposal
    assert planner._hyp_steer_note, "expected a steer note telling the model to TEST the open hypothesis"


def test_candidate_tool_swaps_to_self_script_when_specialist_exhausted():
    """If the specialist a candidate would use is retry-exhausted, the candidate must
    point at self_script (always-available) instead of the dead tool."""
    planner = _planner()
    store = _store_with_coverage()
    text, _ = planner._suspicion_candidates(store, set(), exhausted_action_ids={"idor_probe"})
    # the productId IDOR candidate should no longer tell the model to use idor_probe
    assert "idor_probe" not in text, text
