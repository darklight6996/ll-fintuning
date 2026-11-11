"""
tests/test_model_router_system_prompt.py

Regression test for a bug found via a live autonomous-loop smoke test on
2026-11-11: ModelRouter._format_prompt() silently dropped system_prompt entirely
for any model that is neither a chat-template instruct model nor TinyLlama/Alpaca
-- in particular google/flan-t5-large, the actual default CPU model. That meant the
AIPlanner's _PLANNER_SYSTEM_PROMPT (the only copy of "return exactly one JSON
object" plus the full output format) never reached the model at all on CPU,
producing degenerate, unparseable output and tripping MED-1's planner-failure abort
-- not because MED-1 was wrong to abort, but because the model was never actually
given its task.

Constructs a ModelRouter instance WITHOUT calling __init__ (no real model/tokenizer
loading needed -- _format_prompt only reads self.model_name and self.tokenizer).
"""

from core.model_router import ModelRouter


def _make_router(model_name, tokenizer=None):
    router = object.__new__(ModelRouter)
    router.model_name = model_name
    router.tokenizer = tokenizer
    return router


class _TokenizerNoChatTemplate:
    chat_template = None


class _TokenizerWithChatTemplate:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        sys_msg = next(m["content"] for m in messages if m["role"] == "system")
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        return f"<|system|>{sys_msg}<|user|>{user_msg}"


def test_flan_t5_includes_system_prompt_as_plain_prefix():
    router = _make_router("google/flan-t5-large", _TokenizerNoChatTemplate())
    result = router._format_prompt("What is the next action?", system_prompt="You must return JSON.")
    assert "You must return JSON." in result
    assert "What is the next action?" in result
    assert result.index("You must return JSON.") < result.index("What is the next action?")


def test_flan_t5_with_no_system_prompt_returns_prompt_unchanged():
    router = _make_router("google/flan-t5-large", _TokenizerNoChatTemplate())
    result = router._format_prompt("just the prompt", system_prompt=None)
    assert result == "just the prompt"


def test_unknown_model_family_also_gets_system_prompt_fallback():
    """Any future model that isn't chat-template-having and isn't tinyllama/alpaca
    must still receive system_prompt -- this is the general fallback, not a
    flan-t5-specific special case."""
    router = _make_router("some-org/some-future-base-model", _TokenizerNoChatTemplate())
    result = router._format_prompt("do the task", system_prompt="system instructions here")
    assert "system instructions here" in result
    assert "do the task" in result


def test_chat_template_model_still_uses_chat_template_not_plain_prefix():
    router = _make_router("Qwen/Qwen2.5-7B-Instruct", _TokenizerWithChatTemplate())
    result = router._format_prompt("hello", system_prompt="be helpful")
    assert result == "<|system|>be helpful<|user|>hello"


def test_tinyllama_still_uses_instruction_response_framing():
    router = _make_router("TinyLlama/TinyLlama-1.1B-Chat-v1.0", _TokenizerNoChatTemplate())
    result = router._format_prompt("hello", system_prompt="be helpful")
    assert result.startswith("### Instruction:\nbe helpful\n\nhello")
    assert result.rstrip().endswith("### Response:")
