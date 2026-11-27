"""
tests/test_llm_service_openrouter.py

OpenRouter pluggable backend (ll-finetuning/rag/core/services/llm_service.py,
2026-11-27): a frontier-model alternative to local Ollama for the planner
brain, requested so the TOOL (gates/engines/confirmation-rigor) can be tested
independently of the weak local model. Key read from env ONLY, never logged.
"""
import io
import json
import sys

import pytest

from core.services.llm_service import LLMService


def test_openrouter_requires_both_key_and_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    svc = LLMService(verbose=False)
    # Missing OPENROUTER_MODEL -- must NOT silently pick a default model.
    assert getattr(svc, "_backend", None) != "openrouter"
    assert svc.is_available is False


def test_openrouter_backend_selected_when_key_and_model_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    svc = LLMService(verbose=False)
    assert svc._backend == "openrouter"
    assert svc.mode == "openrouter"
    assert svc.model_name == "anthropic/claude-3.5-sonnet"
    assert svc.is_available is True


def test_openrouter_takes_priority_over_ollama_when_both_configured(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    svc = LLMService(verbose=False)
    assert svc._backend == "openrouter"


def test_openrouter_generate_sends_bearer_header_and_never_logs_key(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-super-secret-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    svc = LLMService(verbose=False)

    captured_req = {}

    class _FakeResp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "hello from the model"}}]}).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured_req["headers"] = dict(req.headers)
        captured_req["url"] = req.full_url
        return _FakeResp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    out = svc.generate("test prompt", system_prompt="be helpful")
    assert out == "hello from the model"
    # urllib lower-cases header names it stores internally as Capitalized-Case
    assert captured_req["headers"].get("Authorization") == "Bearer sk-super-secret-key"
    assert "chat/completions" in captured_req["url"]
    # The secret must never appear in anything printed to stdout.
    assert "sk-super-secret-key" not in capsys.readouterr().out


def test_openrouter_generate_reraises_on_transport_failure_no_silent_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    svc = LLMService(verbose=False)

    import urllib.request

    def fake_urlopen(req, timeout=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ConnectionError):
        svc.generate("test prompt")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
