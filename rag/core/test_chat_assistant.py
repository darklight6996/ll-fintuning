"""
test_chat_assistant.py

Unit test suite verifying Phase 9: ChatAssistant integration with CyberAgent.
Tests:
  1. Default initialization and subcomponent wiring
  2. Custom dependency injection (router, prompt_builder, intelligence_extractor)
  3. Safe execution with None session_state (no crash, returns text)
  4. State synchronization with active session (chat_history populated)
  5. Reset confirmation flow via CyberAgent
  6. Operator steering intent flow via CyberAgent
  7. Reconnaissance query methodology flow via CyberAgent
  8. CTF binary privilege escalation flow via CyberAgent
  9. Backward-compatibility helper methods
 10. Agent injection and direct delegation
"""

import pytest
from unittest.mock import MagicMock
from pathlib import Path
import sys

CORE_DIR = Path(__file__).resolve().parent
RAG_DIR = CORE_DIR.parent
ROOT_DIR = RAG_DIR.parent.parent

for p in (ROOT_DIR, RAG_DIR, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.chat_assistant import ChatAssistant
from core.routers.intent_router import IntentRouter, Intent
from core.agent import CyberAgent, AgentRequest, AgentResponse


def test_chat_assistant_init_defaults():
    assistant = ChatAssistant()
    assert assistant.intent_router is not None
    assert assistant.intelligence_extractor is not None
    assert assistant.prompt_builder is not None
    assert assistant.context_builder is not None


def test_chat_assistant_custom_injection():
    mock_router = MagicMock(spec=IntentRouter)
    mock_pb = MagicMock()
    assistant = ChatAssistant(intent_router=mock_router, prompt_builder=mock_pb)
    assert assistant.intent_router is mock_router
    assert assistant.prompt_builder is mock_pb


def test_chat_assistant_respond_no_session():
    assistant = ChatAssistant()
    # Safe to call without active session
    res = assistant.respond("Hello! What are you?")
    assert isinstance(res, str)
    assert len(res) > 0


def test_chat_assistant_respond_with_session():
    assistant = ChatAssistant()
    state = {
        "name": "audit_box",
        "notes": [],
        "evidence": [],
        "completed_steps": []
    }
    res = assistant.respond("What ports should I check?", session_state=state)
    assert isinstance(res, str)
    assert len(res) > 0
    assert "chat_history" in state
    assert len(state["chat_history"]) == 2
    assert state["chat_history"][0]["role"] == "operator"
    assert state["chat_history"][0]["message"] == "What ports should I check?"
    assert state["chat_history"][1]["role"] == "assistant"
    assert state["chat_history"][1]["message"] == res


def test_chat_assistant_respond_reset_gate():
    assistant = ChatAssistant()
    state = {
        "name": "ctf_box",
        "notes": [{"content": "old intel"}],
        "evidence": [{"tool": "nmap", "output": "port 80"}],
        "completed_steps": ["step1"]
    }
    # Unconfirmed reset should warn
    res1 = assistant.respond("reset", session_state=state)
    assert "confirm" in res1.lower() or "reset" in res1.lower()
    assert len(state["notes"]) == 1

    # Confirmed reset should wipe
    res2 = assistant.respond("reset confirm", session_state=state)
    assert "reset" in res2.lower() or "wiped" in res2.lower()
    assert len(state["notes"]) == 0
    assert len(state["evidence"]) == 0


def test_chat_assistant_respond_steering():
    assistant = ChatAssistant()
    state = {
        "name": "ctf_box",
        "notes": [],
        "evidence": [],
        "completed_steps": []
    }
    res = assistant.respond("the AI went off on a tangent, stop enumerating", session_state=state)
    assert isinstance(res, str)
    assert any(w in res.lower() for w in ["tangent", "steer", "stop", "align", "exploit", "focus"])


def test_chat_assistant_respond_recon():
    assistant = ChatAssistant()
    state = {"name": "recon_box"}
    res = assistant.respond("where do I start? Give me the reconnaissance commands.", session_state=state)
    assert isinstance(res, str)
    assert any(cmd in res for cmd in ["whoami", "uname", "id", "sudo -l", "find", "recon"])


def test_chat_assistant_respond_ctf_with_binary():
    assistant = ChatAssistant()
    state = {
        "name": "suid_challenge",
        "notes": [{"content": "Found SUID binary /challenge/run that calls system('date')"}],
        "evidence": [],
        "completed_steps": []
    }
    res = assistant.respond("how do I exploit /challenge/run?", session_state=state)
    assert isinstance(res, str)
    assert len(res) > 20


def test_chat_assistant_backward_compatibility_methods():
    assistant = ChatAssistant()
    state = {"name": "compat_box", "notes": [{"content": "uid=1000(hacker)"}]}

    # 1. extract_session_intelligence
    intel = assistant.extract_session_intelligence(state)
    assert isinstance(intel, dict)
    assert intel.get("user") == "hacker"

    # 2. _detect_operator_steering
    assert assistant._detect_operator_steering("AI went off on a tangent") is True
    assert assistant._detect_operator_steering("what is port 80") is False

    # 3. _detect_reset_or_different_challenge
    assert assistant._detect_reset_or_different_challenge("reset confirm") is True
    assert assistant._detect_reset_or_different_challenge("nmap scan") is False

    # 4. _detect_recon_request
    assert assistant._detect_recon_request("where do I start?") is True
    assert assistant._detect_recon_request("exploit the heap") is False

    # 5. _execute_reset
    wipe_state = {"notes": [1, 2], "evidence": [3]}
    res = assistant._execute_reset(wipe_state)
    assert len(wipe_state["notes"]) == 0

    # 6. _build_llm_prompt
    prompt = assistant._build_llm_prompt(
        query="test query",
        state={"name": "test"},
        intel={"user": "root"},
        rag_context="test rag",
        yaml_tactics=["step 1"]
    )
    assert isinstance(prompt, str)
    assert "test query" in prompt


def test_chat_assistant_agent_delegation():
    mock_agent = MagicMock(spec=CyberAgent)
    mock_agent.run.return_value = AgentResponse(
        text="Mocked agent response for offensive operations",
        intent="general",
        source="llm"
    )
    assistant = ChatAssistant()
    assistant._agent = mock_agent

    state = {"name": "delegated_session"}
    reply = assistant.respond("Find the vulnerability", session_state=state)

    assert reply == "Mocked agent response for offensive operations"
    assert mock_agent.run.called
    call_args = mock_agent.run.call_args[0][0]
    assert isinstance(call_args, AgentRequest)
    assert call_args.query == "Find the vulnerability"
    assert call_args.session_id == "delegated_session"
