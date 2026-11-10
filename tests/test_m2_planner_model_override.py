"""
tests/test_m2_planner_model_override.py

Regression tests for M2: the AI planner's model is NOT changed by default, but can
be overridden independently of the rest of the framework via
core.config.PLANNER_MODEL_NAME. These tests stub out the actual model loading (no
torch/transformers download needed) and only verify the role-resolution logic in
core.services.llm_service.get_llm_service().
"""

import importlib
import sys

import pytest


@pytest.fixture
def fresh_llm_service_module(monkeypatch):
    """
    Reload core.services.llm_service with its module-level singleton caches reset,
    and with LLMService.__init__ stubbed to avoid any real model loading.
    """
    for mod_name in ("core.services.llm_service",):
        sys.modules.pop(mod_name, None)
    import core.services.llm_service as llm_service_module

    class _FakeLLMService:
        def __init__(self, verbose=True, model_name_override=None,
                     max_input_tokens_override=None, max_new_tokens_override=None):
            self.model_name = model_name_override or "DEFAULT_MODEL"
            self.is_available = True

    monkeypatch.setattr(llm_service_module, "LLMService", _FakeLLMService)
    llm_service_module._INSTANCES = {}
    llm_service_module._LOAD_ATTEMPTED = {}
    return llm_service_module


def test_planner_role_falls_back_to_default_when_unset(fresh_llm_service_module, monkeypatch):
    import core.config as config_module
    monkeypatch.setattr(config_module, "PLANNER_MODEL_NAME", None, raising=False)

    default_instance = fresh_llm_service_module.get_llm_service(role="default")
    planner_instance = fresh_llm_service_module.get_llm_service(role="planner")

    # No override configured -- must be the SAME cached instance, not a second load.
    assert planner_instance is default_instance


def test_planner_role_loads_separate_model_when_configured(fresh_llm_service_module, monkeypatch):
    import core.config as config_module
    monkeypatch.setattr(config_module, "PLANNER_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct", raising=False)
    monkeypatch.setattr(config_module, "PLANNER_MAX_INPUT_TOKENS", None, raising=False)
    monkeypatch.setattr(config_module, "PLANNER_MAX_NEW_TOKENS", None, raising=False)

    default_instance = fresh_llm_service_module.get_llm_service(role="default")
    planner_instance = fresh_llm_service_module.get_llm_service(role="planner")

    assert planner_instance is not default_instance
    assert planner_instance.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert default_instance.model_name == "DEFAULT_MODEL"


def test_planner_role_is_cached_on_second_call(fresh_llm_service_module, monkeypatch):
    import core.config as config_module
    monkeypatch.setattr(config_module, "PLANNER_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct", raising=False)
    monkeypatch.setattr(config_module, "PLANNER_MAX_INPUT_TOKENS", None, raising=False)
    monkeypatch.setattr(config_module, "PLANNER_MAX_NEW_TOKENS", None, raising=False)

    first = fresh_llm_service_module.get_llm_service(role="planner")
    second = fresh_llm_service_module.get_llm_service(role="planner")
    assert first is second


def test_chat_assistant_role_is_stored_and_passed_through(monkeypatch):
    """ChatAssistant(role=...) must thread through to get_llm_service(role=...)."""
    import core.chat_assistant as chat_assistant_module

    captured = {}

    def _fake_get_llm_service(verbose=True, role="default"):
        captured["role"] = role
        return None

    monkeypatch.setattr(chat_assistant_module, "get_llm_service", _fake_get_llm_service)

    assistant = chat_assistant_module.ChatAssistant(role="planner")
    assert assistant.role == "planner"
    assistant._get_llm_router()
    assert captured["role"] == "planner"
