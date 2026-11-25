"""
tests/test_investigator_cognition.py

Tests for the Investigator Cognition overhaul:
  - AppModel working memory in EvidenceStore (components, assumptions, invariants)
  - Free-form hypothesis formation & update_app_model decision verb
  - Generalized confirmation oracle for observable invariant violations
  - Concurrency & threading capability in self-scripting for race condition testing
"""

import os
import json
os.environ["PENTEST_ENABLE_SELF_SCRIPTING"] = "1"

from core.scope import Scope
from core.action_contract import ActionDefinition, AIActionProposal
from core.investigation import EvidenceStore, Observation, Hypothesis, Validation, Component, Assumption, AppModel
from core.ai_planner import AIPlanner
from agents.bug_bounty.single_host_mode import build_single_host_bug_bounty_registry
from agents.bug_bounty.agent import BugBountyAgent
from core.tools.script_runner import run as run_script, screen as screen_script


def test_app_model_working_memory_lifecycle():
    store = EvidenceStore()
    assert store.get_app_model_summary() == "  (no application model built yet)"

    store.update_app_model(
        purpose="E-commerce shopping application with product catalog and discount system",
        components=[
            {"name": "CartService", "endpoints": ["/cart", "/cart/coupon", "/cart/checkout"]},
            Component(name="CatalogService", endpoints=["/product", "/filter"]),
        ],
        roles=["guest", "authenticated_customer", "admin"],
        state_and_resources=["discount_coupon_single_use", "account_balance", "product_stock"],
        assumptions=[
            {"target_or_component": "/cart/coupon", "description": "Coupon redemption check is atomic and prevents multi-application", "fragile": True},
            Assumption(target_or_component="/cart/checkout", description="Balance check prevents overdraft", fragile=False),
        ]
    )

    summary = store.get_app_model_summary()
    assert "CartService" in summary
    assert "discount_coupon_single_use" in summary
    assert "FRAGILE" in summary
    assert "Coupon redemption check is atomic" in summary

    # Test serialization round-trip
    data = store.to_dict()
    restored = EvidenceStore.from_dict(data)
    assert restored.app_model.purpose == store.app_model.purpose
    assert len(restored.app_model.components) == 2
    assert len(restored.app_model.assumptions) == 2
    assert restored.get_app_model_summary() == summary


def test_update_app_model_decision_verb_handling():
    scope = Scope(target="https://target.example", active_testing_allowed=True, single_host_only=True)
    agent = BugBountyAgent(scope=scope)
    store = EvidenceStore()

    proposal = AIActionProposal(
        decision="update_app_model",
        parameters={
            "purpose": "Online bookstore with promotional vouchers",
            "state_and_resources": ["promo_voucher_20pct"],
            "assumptions": [
                {"target_or_component": "/apply-coupon", "description": "Single-use voucher cannot be redeemed simultaneously", "fragile": True}
            ]
        },
        rationale="Modeling checkout and promotion flow",
    )

    result = agent._handle_update_app_model(proposal, store)
    assert result["status"] == "success"
    assert "Online bookstore" in store.app_model.purpose
    assert len(store.app_model.assumptions) == 1


def test_race_condition_and_logic_in_suspicion_candidates():
    planner = AIPlanner(registry=build_single_host_bug_bounty_registry(), chat_assistant=None, scope=Scope(target="target.example"))
    store = EvidenceStore()
    store.add_observation(Observation(
        action_id="browser_probe",
        argv=[],
        raw_stdout="{}",
        exit_code=0,
        parsed_facts={
            "forms": [{"action": "https://target.example/cart/coupon", "method": "post", "inputs": ["coupon", "csrf"]}],
            "parameters": ["coupon"],
            "discovered_endpoints": ["https://target.example/cart/checkout"],
        }
    ))

    text, classes = planner._suspicion_candidates(store, set())
    assert "race-condition" in classes, f"Expected race-condition in classes, got {classes}"
    assert "coupon" in text.lower()
    assert "self_script" in text


def test_confirmation_rigor_recognizes_invariant_violation():
    scope = Scope(target="https://target.example", active_testing_allowed=True, single_host_only=True)
    agent = BugBountyAgent(scope=scope)

    # 1. Invariant violation in race condition (coupon applied multiple times exceeding 1-time limit)
    strength, reason = agent._confirmation_strength("self_script", {
        "confirmed": True,
        "vuln_class": "race-condition",
        "summary": "Coupon SIGNUP20 applied multiple times in parallel requests",
        "evidence": "Applied 6 times concurrently; total discount $120. Order total reduced from $100 to -$20. Control sequential attempt was rejected.",
    })
    assert strength == "proven", f"Expected proven, got {strength} ({reason})"

    # 2. Invariant violation flag in signal
    strength, reason = agent._confirmation_strength("self_script", {
        "confirmed": True,
        "invariant_violated": True,
        "summary": "Balance limit exceeded via concurrent transfer",
        "evidence": "Account balance overdrafted to -500; transfer succeeded twice concurrently.",
    })
    assert strength == "proven", f"Expected proven, got {strength} ({reason})"

    # 3. Weak response divergence should still be rejected as a lead
    strength, reason = agent._confirmation_strength("self_script", {
        "confirmed": True,
        "summary": "Response divergence detected on concurrent requests",
        "evidence": "Different response received from server when tested",
    })
    assert strength == "lead", f"Expected lead, got {strength} ({reason})"


def test_concurrency_allowed_in_script_runner_screen():
    # Threading, concurrent.futures, asyncio, and requests must pass the static screen
    code = '''
import threading
import requests
import json
import time

def worker(url, results):
    try:
        r = requests.get(url, timeout=5)
        results.append(r.status_code)
    except Exception as e:
        results.append(str(e))

threads = []
results = []
for _ in range(5):
    t = threading.Thread(target=worker, args=("http://example.com", results))
    threads.append(t)
    t.start()
for t in threads:
    t.join()

print(json.dumps({"vuln_signal": {"confirmed": False, "summary": f"Completed {len(results)} requests"}}))
'''
    ok, reason = screen_script(code)
    assert ok is True, f"Screen rejected concurrency code: {reason}"
