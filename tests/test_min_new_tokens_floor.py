"""
tests/test_min_new_tokens_floor.py

Regression coverage for the min_new_tokens floor added to
ModelRouter.generate() / ChatAssistant.generate_raw() after a live GPU smoke
test (Qwen2.5-7B-Instruct + qwen-qlora, 2026-11-11) showed the planner's JSON
output getting cut short: M1's constrained decoding correctly forced a valid
"action_id" value including its closing quote, but the very next (now
unconstrained) generation step chose <|im_end|> instead of continuing into
"parameters"/"rationale"/"confidence" -- a premature-stop model behavior, not
a masking bug. min_new_tokens makes eos/<|im_end|> ineligible until a floor is
reached, which is the standard mitigation for exactly this failure mode.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_model_router_generate_passes_min_new_tokens_to_model_generate(monkeypatch):
    import torch
    from core import model_router as model_router_module

    captured = {}

    class _FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            # Return a minimal valid-shaped output tensor: prompt + 1 new token.
            return torch.zeros((1, kwargs_input_length[0] + 1), dtype=torch.long)

    class _FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, *a, **k):
            input_ids = torch.zeros((1, 5), dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        def convert_tokens_to_ids(self, tok):
            return None

        def decode(self, ids, skip_special_tokens=True):
            return ""

    kwargs_input_length = [5]

    router = object.__new__(model_router_module.ModelRouter)
    router.model_name = "Qwen/Qwen2.5-7B-Instruct"
    router.tokenizer = _FakeTokenizer()
    router.model = _FakeModel()
    router.device = "cpu"
    router.max_new_tokens = 512
    router.max_input_tokens = 4096

    router.generate("some prompt", do_sample=False, min_new_tokens=40)

    assert captured.get("min_new_tokens") == 40


def test_model_router_generate_omits_min_new_tokens_when_not_passed(monkeypatch):
    import torch
    from core import model_router as model_router_module

    captured = {}

    class _FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 6), dtype=torch.long)

    class _FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, *a, **k):
            input_ids = torch.zeros((1, 5), dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        def convert_tokens_to_ids(self, tok):
            return None

        def decode(self, ids, skip_special_tokens=True):
            return ""

    router = object.__new__(model_router_module.ModelRouter)
    router.model_name = "Qwen/Qwen2.5-7B-Instruct"
    router.tokenizer = _FakeTokenizer()
    router.model = _FakeModel()
    router.device = "cpu"
    router.max_new_tokens = 512
    router.max_input_tokens = 4096

    router.generate("some prompt", do_sample=False)

    assert "min_new_tokens" not in captured


def test_generate_raw_default_min_new_tokens_is_40():
    sys.path.insert(0, os.path.join(ROOT, "ll-finetuning", "rag"))
    import inspect
    from core.chat_assistant import ChatAssistant
    sig = inspect.signature(ChatAssistant.generate_raw)
    assert sig.parameters["min_new_tokens"].default == 40


def test_generate_raw_forwards_min_new_tokens_to_llm_generate(monkeypatch):
    from core.chat_assistant import ChatAssistant

    captured = {}

    class _FakeLLM:
        def generate(self, prompt, **kwargs):
            captured.update(kwargs)
            return '{"decision": "propose_completion", "summary": "x"}'

    ca = object.__new__(ChatAssistant)
    monkeypatch.setattr(ca, "_get_llm_router", lambda: _FakeLLM(), raising=False)

    ca.generate_raw("prompt", max_new_tokens=512, min_new_tokens=40)

    assert captured.get("min_new_tokens") == 40
