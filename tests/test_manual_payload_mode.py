"""
tests/test_manual_payload_mode.py

Tests for --manual-payload mode: agents/common/manual_payload.py's curl
parser/wrapper/solving-curl helpers, and BaseAgent._handle_manual_payload() /
_emit_manual_payload_solving_curl() in agents/base_agent.py. The human's
input is converted into a REPLACEMENT AIActionProposal that would then flow
through the SAME 10-gate execution path as any model-authored proposal --
these tests cover the conversion itself, not the gated execution (already
covered by test_http_probe.py / test_self_scripting.py / test_script_runner_screen.py).
"""

import ast

import pytest

from core.scope import Scope
from core.investigation import EvidenceStore, Hypothesis, Observation
from core.tools.script_runner import screen
from agents.common.manual_payload import (
    build_curl_replay_script,
    build_payload_brief,
    build_solving_curl,
    describe_technique,
    infer_param_name,
    parse_curl,
    render_request_template,
)


class _DummyChatAssistant:
    def __init__(self, *args, **kwargs):
        pass

    def synthesize_report(self, state, context=None):
        return "stub report"


def _make_agent(monkeypatch):
    import agents.base_agent as base_agent_mod
    from agents.pentest.agent import PentestAgent
    from core.session_manager import SessionManager

    monkeypatch.setattr(base_agent_mod, "ChatAssistant", _DummyChatAssistant)
    scope = Scope(target="https://x.example/product?shopId=1", in_scope=["x.example"],
                  active_testing_allowed=True, active_network_allowed=True)
    agent = PentestAgent(scope=scope, mode="automated", session_manager=SessionManager())
    agent.chat_assistant = _DummyChatAssistant()
    return agent


class _FakeProposal:
    def __init__(self, target, parameters):
        self.target = target
        self.parameters = parameters


def _feed_lines(monkeypatch, lines):
    """Make input() return each of `lines` in turn, then raise EOFError."""
    it = iter(lines)

    def fake_input():
        try:
            return next(it)
        except StopIteration:
            raise EOFError()
    monkeypatch.setattr("builtins.input", fake_input)


# --- infer_param_name -----------------------------------------------------------

def test_infer_param_name_from_url_query_string():
    assert infer_param_name("https://x.example/product?shopId=1", "") == "shopId"


def test_infer_param_name_from_hypothesis_text():
    assert infer_param_name("https://x.example/product",
                             "the parameter 'stockApi' on /product/stock") == "stockApi"


def test_infer_param_name_none_when_neither_matches():
    assert infer_param_name("https://x.example/product", "no parameter named here") is None


# --- parse_curl -------------------------------------------------------------------

def test_parse_curl_simple_get():
    method, url, headers, body = parse_curl("curl 'https://x.example/product?shopId=127.0.0.1:8080'")
    assert method == "GET"
    assert url == "https://x.example/product?shopId=127.0.0.1:8080"
    assert headers == {}
    assert body is None


def test_parse_curl_post_with_headers_and_body():
    method, url, headers, body = parse_curl(
        'curl -X POST -H "Content-Type: application/json" -d \'{"a": 1}\' https://x.example/api'
    )
    assert method == "POST"
    assert url == "https://x.example/api"
    assert headers == {"Content-Type": "application/json"}
    assert body == '{"a": 1}'


def test_parse_curl_defaults_to_post_when_data_present_without_explicit_method():
    method, _, _, _ = parse_curl("curl -d 'x=1' https://x.example/api")
    assert method == "POST"


def test_parse_curl_raises_when_no_url_found():
    with pytest.raises(ValueError):
        parse_curl("curl -X GET -H 'A: B'")


# --- build_curl_replay_script -----------------------------------------------------

def test_curl_replay_script_is_valid_python_and_passes_the_safety_screen():
    code = build_curl_replay_script("GET", "https://x.example/", {}, None)
    ast.parse(code)  # raises SyntaxError if malformed
    ok, reason = screen(code)
    assert ok, reason


def test_curl_replay_script_always_tags_lead_never_proven():
    code = build_curl_replay_script("POST", "https://x.example/api", {"X": "Y"}, "body")
    assert "'strength': 'lead'" in code
    assert "'confirmed': False" in code


# --- build_solving_curl ------------------------------------------------------------

def test_build_solving_curl_get():
    assert build_solving_curl("GET", "https://x.example/go?next=x") == \
        "curl -s 'https://x.example/go?next=x'"


def test_build_solving_curl_post_with_headers_and_body():
    out = build_solving_curl("POST", "https://x.example/api", {"X-Test": "1"}, "a=1")
    assert out.startswith("curl -s -X POST")
    assert "-H 'X-Test: 1'" in out
    assert "-d 'a=1'" in out
    assert out.endswith("'https://x.example/api'")


# --- describe_technique / render_request_template / build_payload_brief ----------
# (2026-11-25 rebuild after operator feedback: the brief must confirm the
# exploitation TECHNIQUE and show the request/tamper point BEFORE asking the
# operator for anything -- not just hand them a bare parameter name.)

def test_describe_technique_known_class():
    text = describe_technique("ssrf")
    assert "SERVER itself makes an outbound request" in text
    assert "XXE" in text  # SSRF's description explicitly calls out the XXE variant


def test_describe_technique_unknown_class_falls_back_to_generic_reasoning_prompt():
    text = describe_technique("some-made-up-class")
    assert "No specific technique template" in text


def test_describe_technique_none_falls_back_to_generic_reasoning_prompt():
    assert "No specific technique template" in describe_technique(None)


def test_render_request_template_marks_tamper_point():
    out = render_request_template("https://x.example/product?shopId=1", "shopId")
    assert "GET /product?shopId=1 HTTP/1.1" in out
    assert "Host: x.example" in out
    assert "Tamper point: the 'shopId' value" in out


def test_render_request_template_no_param_says_so():
    out = render_request_template("https://x.example/product", None)
    assert "parameter not yet identified" in out


def test_build_payload_brief_leads_with_technique_before_asking_for_anything():
    brief = build_payload_brief(
        "https://x.example/product?shopId=1", "shopId", "ssrf",
        "shopId triggers a server-side lookup against an internal admin panel.",
        "(no prior observation)",
    )
    # Technique must appear BEFORE any payload-crafting-guidance section --
    # confirm-then-guide, not guide-then-confirm.
    assert brief.index("Exploitation technique:") < brief.index("Payload-crafting guidance:")
    assert "SERVER itself makes an outbound request" in brief
    assert "Tamper point: the 'shopId' value" in brief


def test_build_payload_brief_includes_seed_payloads_for_known_class():
    brief = build_payload_brief("https://x.example/product?shopId=1", "shopId", "ssrf",
                                 "ssrf hypothesis", "")
    assert "169.254.169.254" in brief  # core.payload_pack's SSRF seed payload


def test_build_payload_brief_unknown_class_has_no_fabricated_guidance():
    brief = build_payload_brief("https://x.example/product?x=1", "x", None, "unclassified", "")
    assert "no vuln_class yet" in brief


# --- BaseAgent._find_hypothesis_for_manual_payload ---------------------------------

def test_find_hypothesis_for_manual_payload_picks_strongest_open_hypothesis(monkeypatch):
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    weak = Hypothesis(observation_ids=["obs_1"], description="weak", validation_test="self_script",
                       vuln_class="idor", confidence="speculative")
    strong = Hypothesis(observation_ids=["obs_1"], description="strong", validation_test="self_script",
                         vuln_class="ssrf", confidence="likely")
    store.hypotheses[weak.hyp_id] = weak
    store.hypotheses[strong.hyp_id] = strong
    found = agent._find_hypothesis_for_manual_payload(store)
    assert found is strong


def test_find_hypothesis_for_manual_payload_ignores_milestones_and_confirmed(monkeypatch):
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    milestone = Hypothesis(observation_ids=["obs_1"], description="m", validation_test="recon_scan",
                            confidence="likely", milestone=True)
    confirmed = Hypothesis(observation_ids=["obs_1"], description="c", validation_test="self_script",
                            vuln_class="sqli", confidence="likely", status="confirmed")
    store.hypotheses[milestone.hyp_id] = milestone
    store.hypotheses[confirmed.hyp_id] = confirmed
    assert agent._find_hypothesis_for_manual_payload(store) is None


def test_find_hypothesis_for_manual_payload_empty_store_returns_none(monkeypatch):
    agent = _make_agent(monkeypatch)
    assert agent._find_hypothesis_for_manual_payload(EvidenceStore()) is None


def test_handle_manual_payload_brief_surfaces_vuln_class_from_matching_hypothesis(monkeypatch, capsys):
    agent = _make_agent(monkeypatch)
    store = EvidenceStore()
    hyp = Hypothesis(observation_ids=["obs_1"], description="shopId is an SSRF vector",
                      validation_test="self_script", vuln_class="ssrf", confidence="likely")
    store.hypotheses[hyp.hyp_id] = hyp
    _feed_lines(monkeypatch, [""])  # skip -- just inspect the printed brief
    proposal = _FakeProposal(
        target="https://x.example/product?shopId=1",
        parameters={"target": "https://x.example/product?shopId=1", "hypothesis": "thin hint"},
    )
    agent._handle_manual_payload(proposal, store)
    out = capsys.readouterr().out
    assert "Vulnerability class:   ssrf" in out
    assert "shopId is an SSRF vector" in out  # the Hypothesis's own (richer) description won out


# --- BaseAgent._handle_manual_payload ---------------------------------------------

def test_handle_manual_payload_bare_payload_infers_param_from_target(monkeypatch):
    agent = _make_agent(monkeypatch)
    _feed_lines(monkeypatch, ["PAYLOAD:127.0.0.1:8080", ""])
    proposal = _FakeProposal(
        target="https://x.example/product?shopId=1",
        parameters={"target": "https://x.example/product?shopId=1",
                    "hypothesis": "shopId is an SSRF vector"},
    )
    result = agent._handle_manual_payload(proposal, EvidenceStore())
    assert result is not None
    assert result.action_id == "http_probe"
    assert result.parameters["focus_param"] == "shopId"
    assert result.parameters["payload"] == "127.0.0.1:8080"
    assert agent._manual_payload_context["kind"] == "payload"


def test_handle_manual_payload_explicit_param_equals_value(monkeypatch):
    agent = _make_agent(monkeypatch)
    _feed_lines(monkeypatch, ["PAYLOAD:storeId=127.0.0.1:8080", ""])
    proposal = _FakeProposal(
        target="https://x.example/product",  # no query string to infer from
        parameters={"target": "https://x.example/product", "hypothesis": "no named parameter here"},
    )
    result = agent._handle_manual_payload(proposal, EvidenceStore())
    assert result is not None
    assert result.parameters["focus_param"] == "storeId"
    assert result.parameters["payload"] == "127.0.0.1:8080"


def test_handle_manual_payload_payload_without_param_and_no_inference_is_rejected(monkeypatch):
    agent = _make_agent(monkeypatch)
    _feed_lines(monkeypatch, ["PAYLOAD:just-a-value", ""])
    proposal = _FakeProposal(target="https://x.example/product",
                              parameters={"target": "https://x.example/product", "hypothesis": ""})
    result = agent._handle_manual_payload(proposal, EvidenceStore())
    assert result is None


def test_handle_manual_payload_curl_builds_self_script_proposal(monkeypatch):
    agent = _make_agent(monkeypatch)
    _feed_lines(monkeypatch, ["CURL:curl 'https://x.example/product?shopId=127.0.0.1:8080'", ""])
    proposal = _FakeProposal(target="https://x.example/product",
                              parameters={"target": "https://x.example/product", "hypothesis": "ssrf"})
    result = agent._handle_manual_payload(proposal, EvidenceStore())
    assert result is not None
    assert result.action_id == "self_script"
    ast.parse(result.parameters["code"])
    ok, reason = screen(result.parameters["code"])
    assert ok, reason
    assert agent._manual_payload_context["kind"] == "curl"


def test_handle_manual_payload_malformed_curl_is_rejected(monkeypatch):
    agent = _make_agent(monkeypatch)
    _feed_lines(monkeypatch, ["CURL:curl -X GET -H 'A: B'", ""])  # no URL
    proposal = _FakeProposal(target="https://x.example/product",
                              parameters={"target": "https://x.example/product", "hypothesis": ""})
    result = agent._handle_manual_payload(proposal, EvidenceStore())
    assert result is None


def test_handle_manual_payload_script_passes_human_code_through_verbatim(monkeypatch):
    agent = _make_agent(monkeypatch)
    human_code = (
        "import requests, json\n"
        "r = requests.get('https://x.example/')\n"
        "print(json.dumps({'vuln_signal': {'confirmed': False}}))\n"
    )
    _feed_lines(monkeypatch, ["SCRIPT:"] + human_code.splitlines() + [""])
    proposal = _FakeProposal(target="https://x.example/product",
                              parameters={"target": "https://x.example/product", "hypothesis": "x"})
    result = agent._handle_manual_payload(proposal, EvidenceStore())
    assert result is not None
    assert result.action_id == "self_script"
    assert "requests.get" in result.parameters["code"]
    assert agent._manual_payload_context["kind"] == "script"


def test_handle_manual_payload_empty_input_returns_none(monkeypatch):
    agent = _make_agent(monkeypatch)
    _feed_lines(monkeypatch, [""])
    proposal = _FakeProposal(target="https://x.example/product",
                              parameters={"target": "https://x.example/product", "hypothesis": ""})
    result = agent._handle_manual_payload(proposal, EvidenceStore())
    assert result is None


def test_handle_manual_payload_unrecognized_prefix_returns_none(monkeypatch):
    agent = _make_agent(monkeypatch)
    _feed_lines(monkeypatch, ["just some random text", ""])
    proposal = _FakeProposal(target="https://x.example/product",
                              parameters={"target": "https://x.example/product", "hypothesis": ""})
    result = agent._handle_manual_payload(proposal, EvidenceStore())
    assert result is None


# --- BaseAgent._emit_manual_payload_solving_curl -----------------------------------

def test_emit_solving_curl_for_payload_context(monkeypatch, capsys):
    agent = _make_agent(monkeypatch)
    ctx = {"kind": "payload", "target": "https://x.example/product?shopId=1",
           "param": "shopId", "payload": "127.0.0.1:8080"}
    agent._emit_manual_payload_solving_curl(ctx)
    out = capsys.readouterr().out
    assert "CONFIRMED" in out
    assert "127.0.0.1%3A8080" in out or "127.0.0.1:8080" in out


def test_emit_solving_curl_for_curl_context_echoes_original(monkeypatch, capsys):
    agent = _make_agent(monkeypatch)
    ctx = {"kind": "curl", "method": "GET", "url": "https://x.example/go?next=evil",
           "headers": {}, "body": None}
    agent._emit_manual_payload_solving_curl(ctx)
    out = capsys.readouterr().out
    assert "https://x.example/go?next=evil" in out


def test_emit_solving_curl_for_script_context_has_no_fabricated_request(monkeypatch, capsys):
    agent = _make_agent(monkeypatch)
    agent._emit_manual_payload_solving_curl({"kind": "script"})
    out = capsys.readouterr().out
    assert "curl" not in out.lower() or "no single request" in out.lower()


# --- end-to-end loop wiring (run_autonomous(manual_payload=True)) -----------------

def test_run_autonomous_manual_payload_mode_intercepts_self_script_and_confirms(monkeypatch, tmp_path):
    """
    Full-loop smoke test: the planner proposes self_script, manual-payload
    mode intercepts it, the human pastes a PAYLOAD:, that gets converted to
    an http_probe proposal, a (mocked) confirmed execution promotes a proven
    finding, and the solving curl gets printed -- all through the REAL
    run_autonomous() loop, not a reimplementation of it.
    """
    from core.action_contract import AIActionProposal
    from core.ai_planner import AIPlanner
    from agents.pentest.agent import PentestAgent
    from core.session_manager import SessionManager
    import agents.base_agent as base_agent_mod

    monkeypatch.setattr(base_agent_mod, "ChatAssistant", _DummyChatAssistant)
    scope = Scope(target="https://x.example/product?shopId=1", in_scope=["x.example"],
                  active_testing_allowed=True, active_network_allowed=True)
    session_mgr = SessionManager(base_dir=str(tmp_path))
    agent = PentestAgent(scope=scope, mode="automated", session_manager=session_mgr)
    agent.chat_assistant = _DummyChatAssistant()
    registry = agent.get_action_registry()
    criteria = agent.get_completion_criteria()

    # Planner always proposes the same self_script hypothesis-test.
    def fake_decide_next(self, **kwargs):
        return AIActionProposal(
            decision="propose_action", action_id="self_script",
            target="https://x.example/product?shopId=1",
            parameters={
                "target": "https://x.example/product?shopId=1",
                "hypothesis": "shopId is an SSRF vector into the internal admin panel",
                "code": "print('should never run -- manual-payload intercepts this')",
            },
            rationale="test", confidence=0.8,
        )
    monkeypatch.setattr(AIPlanner, "decide_next", fake_decide_next)

    # Human supplies a bare payload for the inferred 'shopId' parameter.
    _feed_lines(monkeypatch, ["PAYLOAD:127.0.0.1:8080", ""])

    # Mock the gated executor so this test never makes a real network call --
    # it reports a CONFIRMED http_probe hit, exercising the real promotion +
    # solving-curl path on a realistic vuln_signal shape.
    def fake_execute_action(proposal, registry, scope, execution_history=None, **kwargs):
        assert proposal.action_id == "http_probe"
        assert proposal.parameters["focus_param"] == "shopId"
        assert proposal.parameters["payload"] == "127.0.0.1:8080"
        return {
            "status": "success",
            "action_identity": "id1",
            "results": [{
                "argv": ["http_probe"], "returncode": 0,
                "stdout": '{"vuln_signal": {"confirmed": true, "severity": "high", '
                          '"vuln_class": "open-redirect", "parameter": "shopId", '
                          '"summary": "confirmed", "evidence": "Location host matches"}}',
                "stderr": "",
            }],
        }
    monkeypatch.setattr(agent.executor, "execute_action", fake_execute_action)

    result = agent.run_autonomous(registry=registry, completion_criteria=criteria, max_steps=1,
                                   manual_payload=True)

    assert result["steps_run"] == 1
    assert result["proven_findings"] == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
